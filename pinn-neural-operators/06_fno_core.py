"""Native-complex, dimension-independent Fourier Neural Operator core.

This tutorial is executable documentation.  Its demo assertions verify the
spectral truncation, parallel bypass path, gradients, and resolution-transfer
contract implemented below.
"""

from __future__ import annotations

import argparse
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


def _normalize_k_max(
    k_max: int | Sequence[int], spatial_dim: int
) -> tuple[int, ...]:
    if spatial_dim not in (1, 2, 3):
        raise ValueError("spatial_dim must be 1, 2, or 3")
    if isinstance(k_max, bool):
        raise ValueError("k_max entries must be nonnegative integers")
    if isinstance(k_max, int):
        values = (k_max,) * spatial_dim
    else:
        values = tuple(k_max)
    if len(values) != spatial_dim:
        raise ValueError("k_max must have one entry per spatial dimension")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise ValueError("k_max entries must be nonnegative integers")
    return values


def _validate_complex_channels(
    value: Tensor, *, channels: int, spatial_dim: int, name: str = "input"
) -> None:
    expected_rank = spatial_dim + 2
    if value.ndim != expected_rank:
        raise ValueError(
            f"{name} must have shape (batch, channels, *{spatial_dim}D spatial); "
            f"got rank {value.ndim}"
        )
    if value.shape[1] != channels:
        raise ValueError(f"{name} must have {channels} channels; got {value.shape[1]}")
    if not value.is_complex():
        raise ValueError(f"{name} must be a native complex tensor")


def complex_gelu(value: Tensor) -> Tensor:
    """Componentwise GELU for a native complex tensor.

    PyTorch's GELU is defined only on real tensors.  Here ``GELU(Re z)`` and
    ``GELU(Im z)`` are evaluated independently, which is the literal complex
    extension used by every block in this tutorial.  This choice is smooth and
    trainable, but it is not equivariant to arbitrary global phase rotations.
    """

    if not value.is_complex():
        raise ValueError("complex_gelu expects a native complex tensor")
    return torch.complex(F.gelu(value.real), F.gelu(value.imag))


class ComplexPointwiseLinear(nn.Module):
    """A learned complex 1x1 convolution on channel-first ND fields.

    The same channel-mixing matrix is applied independently at every spatial
    point, so the operation accepts any spatial rank and grid resolution.  A
    module and its input must share a complex dtype; use
    ``module.to(dtype=torch.complex128)`` for double-complex calculations.
    """

    def __init__(
        self, in_channels: int, out_channels: int, *, bias: bool = True
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        scale = 1.0 / math.sqrt(2.0 * in_channels)
        weight = scale * torch.complex(
            torch.randn(out_channels, in_channels),
            torch.randn(out_channels, in_channels),
        )
        self.weight = nn.Parameter(weight)
        self.bias = (
            nn.Parameter(torch.zeros(out_channels, dtype=torch.cfloat))
            if bias
            else None
        )

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim < 3:
            raise ValueError("input must have batch, channel, and spatial axes")
        if value.shape[1] != self.in_channels:
            raise ValueError(
                f"input must have {self.in_channels} channels; got {value.shape[1]}"
            )
        if not value.is_complex():
            raise ValueError("input must be a native complex tensor")
        if value.dtype != self.weight.dtype:
            raise ValueError(
                f"input dtype {value.dtype} must match parameter dtype {self.weight.dtype}"
            )
        result = torch.einsum("oi,bi...->bo...", self.weight, value)
        if self.bias is not None:
            result = result + self.bias.reshape(1, -1, *((1,) * (value.ndim - 2)))
        return result


class SpectralConvNd(nn.Module):
    """Global complex convolution restricted to an ND low-frequency box.

    Inputs have shape ``(batch, in_channels, *spatial_shape)``.  A full FFT is
    used because the fields are natively complex.  The parameter tensor stores
    one complex channel-mixing matrix for every signed integer mode in
    ``[-Kmax_i, Kmax_i]`` along each spatial axis.  Those parameters do not
    depend on the grid size: Fourier index ``k`` refers to the same mode on any
    sufficiently large grid.

    All output Fourier coefficients outside ``|k_i| <= Kmax_i`` are left at
    exactly zero before the inverse FFT.  Each grid axis must therefore contain
    at least ``2*Kmax_i + 1`` points, preventing positive and negative retained
    modes from aliasing onto the same discrete coefficient.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        spatial_dim: int,
        k_max: int | Sequence[int],
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dim = spatial_dim
        self.k_max = _normalize_k_max(k_max, spatial_dim)

        mode_shape = tuple(2 * value + 1 for value in self.k_max)

        scale = 1.0 / math.sqrt(2.0 * in_channels)

        weight = scale * torch.complex(
            torch.randn(out_channels, in_channels, *mode_shape),
            torch.randn(out_channels, in_channels, *mode_shape),
        )
        self.weight = nn.Parameter(weight)

    @property
    def spatial_axes(self) -> tuple[int, ...]:
        return tuple(range(-self.spatial_dim, 0))

    def _retained_indices(
        self, spatial_shape: Sequence[int], device: torch.device
    ) -> tuple[Tensor, ...]:
        indices = []
        for axis, (size, limit) in enumerate(zip(spatial_shape, self.k_max)):
            required = 2 * limit + 1
            if size < required:
                raise ValueError(
                    f"spatial axis {axis} has size {size}, but k_max={limit} "
                    f"requires at least {required} points"
                )
            signed = torch.arange(-limit, limit + 1, device=device)
            indices.append(torch.remainder(signed, size))
        return tuple(indices)

    def forward(self, value: Tensor) -> Tensor:
        _validate_complex_channels(
            value,
            channels=self.in_channels,
            spatial_dim=self.spatial_dim,
        )
        if value.dtype != self.weight.dtype:
            raise ValueError(
                f"input dtype {value.dtype} must match parameter dtype {self.weight.dtype}"
            )
        spatial_shape = value.shape[-self.spatial_dim :]
        retained_indices = self._retained_indices(spatial_shape, value.device)
        mode_grid = torch.meshgrid(*retained_indices, indexing="ij")

        transformed = torch.fft.fftn(value, dim=self.spatial_axes)
        index = (slice(None), slice(None), *mode_grid)
        retained_input = transformed[index]
        retained_output = torch.einsum(
            "bi...,oi...->bo...", retained_input, self.weight
        )

        output_spectrum = torch.zeros(
            value.shape[0],
            self.out_channels,
            *spatial_shape,
            dtype=value.dtype,
            device=value.device,
        )
        output_spectrum[(slice(None), slice(None), *mode_grid)] = retained_output
        return torch.fft.ifftn(output_spectrum, dim=self.spatial_axes)


class FourierBlockNd(nn.Module):
    """Parallel global-spectral and local-pointwise FNO paths.

    For latent features ``z`` this module evaluates exactly

    ``complex_gelu(IFFT(R_theta * FFT(z)) + W z)``.

    The spectral path supplies a global receptive field through retained modes;
    the 1x1 bypass keeps a local path for information removed by truncation.
    """

    def __init__(
        self,
        channels: int,
        *,
        spatial_dim: int,
        k_max: int | Sequence[int],
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.spectral = SpectralConvNd(
            channels,
            channels,
            spatial_dim=spatial_dim,
            k_max=k_max,
        )
        self.bypass = ComplexPointwiseLinear(channels, channels)

    def forward(self, value: Tensor) -> Tensor:
        return complex_gelu(self.spectral(value) + self.bypass(value))


class FNOCoreNd(nn.Module):
    """Native-complex lift-block-project FNO core for 1D, 2D, or 3D fields.

    Callers provide physical fields, coordinates, and PDE parameters as explicit
    input channels.  The core lifts those channels to ``width`` latent channels,
    applies ``n_layers`` Fourier blocks (including GELU in every block), and
    projects pointwise to the requested output channels without a final
    activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        width: int = 32,
        n_layers: int = 4,
        spatial_dim: int = 2,
        k_max: int | Sequence[int] = 8,
    ) -> None:
        super().__init__()
        if width <= 0 or n_layers <= 0:
            raise ValueError("width and n_layers must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.n_layers = n_layers
        self.spatial_dim = spatial_dim
        self.k_max = _normalize_k_max(k_max, spatial_dim)

        self.lift = ComplexPointwiseLinear(in_channels, width)

        self.blocks = nn.ModuleList(
            FourierBlockNd(width, spatial_dim=spatial_dim, k_max=self.k_max)
            for _ in range(n_layers)
        )
        self.project = ComplexPointwiseLinear(width, out_channels)

    def forward(self, value: Tensor) -> Tensor:
        _validate_complex_channels(
            value,
            channels=self.in_channels,
            spatial_dim=self.spatial_dim,
        )
        hidden = self.lift(value)
        for block in self.blocks:
            hidden = block(hidden)
        return self.project(hidden)


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return value


def _low_mode_mask(
    shape: Sequence[int], k_max: Sequence[int], device: torch.device
) -> Tensor:
    axes = [
        torch.round(torch.fft.fftfreq(n, device=device) * n).abs() <= k
        for n, k in zip(shape, k_max)
    ]
    grids = torch.meshgrid(*axes, indexing="ij")
    mask = torch.ones(tuple(shape), dtype=torch.bool, device=device)
    for grid in grids:
        mask &= grid
    return mask


def _expect_value_error(callable_) -> None:
    try:
        callable_()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def run_demo(
    *, dim: int, grid_size: int, k_max: int, width: int, layers: int, seed: int
) -> dict[str, float]:
    """Exercise the complete public contract without training a PDE model."""

    torch.manual_seed(seed)
    shape = (grid_size,) * dim
    input_field = torch.randn(2, 2, *shape, dtype=torch.cfloat, requires_grad=True)
    model = FNOCoreNd(
        2,
        3,
        width=width,
        n_layers=layers,
        spatial_dim=dim,
        k_max=k_max,
    )

    output = model(input_field)
    assert output.shape == (2, 3, *shape)
    assert output.is_complex()

    activation_probe = torch.tensor([-1.0 + 2.0j], dtype=torch.cfloat)
    assert torch.allclose(
        complex_gelu(activation_probe),
        torch.complex(
            F.gelu(activation_probe.real), F.gelu(activation_probe.imag)
        ),
    )

    latent = torch.randn(2, width, *shape, dtype=torch.cfloat)
    block = FourierBlockNd(width, spatial_dim=dim, k_max=k_max)
    spectral = block.spectral(latent)
    spectral_hat = torch.fft.fftn(spectral, dim=tuple(range(-dim, 0)))
    mode_box = _low_mode_mask(shape, block.spectral.k_max, spectral.device)
    discarded = spectral_hat[..., ~mode_box]
    high_mode_max = discarded.abs().max().item() if discarded.numel() else 0.0
    low_mode_max = spectral_hat[..., mode_box].abs().max().item()
    high_mode_ratio = high_mode_max / max(low_mode_max, torch.finfo(torch.float32).eps)
    assert high_mode_ratio < 1e-5

    explicit = complex_gelu(block.spectral(latent) + block.bypass(latent))
    assert torch.allclose(block(latent), explicit)

    per_axis_modes = SpectralConvNd(
        1, 1, spatial_dim=dim, k_max=(k_max,) * dim
    )
    per_axis_output = per_axis_modes(
        torch.randn(1, 1, *shape, dtype=torch.cfloat)
    )
    assert per_axis_output.shape == (1, 1, *shape)

    output.abs().square().mean().backward()
    assert input_field.grad is not None and torch.isfinite(input_field.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    larger_shape = tuple(2 * n for n in shape)
    larger_input = torch.randn(1, 2, *larger_shape, dtype=torch.cfloat)
    with torch.no_grad():
        larger_output = model(larger_input)
    assert larger_output.shape == (1, 3, *larger_shape)

    _expect_value_error(lambda: model(torch.randn(1, 2, *shape)))
    _expect_value_error(
        lambda: SpectralConvNd(2, 3, spatial_dim=dim, k_max=(-1,) * dim)
    )
    too_small = (2 * k_max,) * dim
    _expect_value_error(
        lambda: model(torch.randn(1, 2, *too_small, dtype=torch.cfloat))
    )

    return {
        "parameters": float(sum(parameter.numel() for parameter in model.parameters())),
        "high_mode_ratio": high_mode_ratio,
        "gradient_norm": input_field.grad.norm().item(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--grid-size", type=_positive_int)
    parser.add_argument("--k-max", type=_nonnegative_int, default=4)
    parser.add_argument("--width", type=_positive_int, default=8)
    parser.add_argument("--layers", type=_positive_int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, float]:
    args = parse_args(argv)
    defaults = {1: 64, 2: 24, 3: 10}
    grid_size = args.grid_size or defaults[args.dim]
    metrics = run_demo(
        dim=args.dim,
        grid_size=grid_size,
        k_max=args.k_max,
        width=args.width,
        layers=args.layers,
        seed=args.seed,
    )
    print(
        f"native-complex FNO | dim={args.dim} grid={(grid_size,) * args.dim} "
        f"Kmax={args.k_max} width={args.width} layers={args.layers}"
    )
    print(
        f"parameters={int(metrics['parameters']):,}  "
        f"discarded/retained max={metrics['high_mode_ratio']:.3e}  "
        f"input-gradient norm={metrics['gradient_norm']:.3e}"
    )
    print(
        f"resolution transfer: N={grid_size} -> N={2 * grid_size} "
        "with shared weights"
    )
    return metrics


if __name__ == "__main__":
    main()
