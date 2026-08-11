"""Periodic spectral geometry, invariants, and projections.

Ported from the validated tutorial toolkit ``pinn-neural-operators/05_3d_equations.py``
(``PeriodicDomain``, ``spectral_gradient``, ``spectral_laplacian``, ``l2_mass``,
``project_to_mass``).  Only the pieces the NLS study needs are carried over; the
incompressible-flow and density helpers stay behind in the tutorial.

Everything here is grid-resolution agnostic in the sense that it derives its wave
vectors from the domain, so the same code serves N=64 and N=128 unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class PeriodicDomain:
    """Uniform endpoint-free rectangular periodic grid in any dimension."""

    shape: tuple[int, ...]
    lengths: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "lengths", tuple(float(v) for v in self.lengths))
        if not self.shape:
            raise ValueError("shape must contain at least one spatial dimension")
        if len(self.shape) != len(self.lengths):
            raise ValueError("shape and lengths must have the same dimension")
        if any(not isinstance(n, int) or n <= 0 for n in self.shape):
            raise ValueError("every grid size must be a positive integer")
        if any(not math.isfinite(length) or length <= 0 for length in self.lengths):
            raise ValueError("every domain length must be positive and finite")

    @classmethod
    def periodic_1d(cls, n: int) -> "PeriodicDomain":
        """The thesis default: ``n`` points on ``[0, 2*pi)``."""

        return cls((n,), (2 * math.pi,))

    @property
    def dim(self) -> int:
        return len(self.shape)

    @property
    def spacing(self) -> tuple[float, ...]:
        return tuple(length / n for n, length in zip(self.shape, self.lengths))

    @property
    def cell_volume(self) -> float:
        return math.prod(self.spacing)

    @property
    def spatial_axes(self) -> tuple[int, ...]:
        return tuple(range(-self.dim, 0))

    def mesh(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> tuple[Tensor, ...]:
        axes = [
            torch.arange(n, device=device, dtype=dtype) * dx
            for n, dx in zip(self.shape, self.spacing)
        ]
        return tuple(torch.meshgrid(*axes, indexing="ij"))

    def wave_vectors(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> tuple[Tensor, ...]:
        axes = [
            2 * math.pi * torch.fft.fftfreq(n, d=dx, device=device, dtype=dtype)
            for n, dx in zip(self.shape, self.spacing)
        ]
        return tuple(torch.meshgrid(*axes, indexing="ij"))

    def wave_number_squared(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        """``|k|^2`` on the grid: the only spatial feature the kinetic step needs."""

        return sum(k**2 for k in self.wave_vectors(device=device, dtype=dtype))

    def validate_field(self, field: Tensor) -> None:
        if field.ndim < self.dim or tuple(field.shape[-self.dim :]) != self.shape:
            raise ValueError(
                f"field must end in spatial shape {self.shape}; got {tuple(field.shape)}"
            )

    def validate_batched_field(self, field: Tensor) -> None:
        """Enforce the ``(batch, *shape)`` layout every operator in this project uses."""

        self.validate_field(field)
        if field.ndim != self.dim + 1:
            raise ValueError(
                f"field must have exactly one leading batch axis; got {tuple(field.shape)}"
            )


def _first_derivative_wave_vectors(
    field: Tensor, domain: PeriodicDomain
) -> tuple[Tensor, ...]:
    """Wave vectors compatible with real fields on even-sized grids.

    A real grid's Nyquist mode is self-conjugate and therefore has no signed first
    derivative.  Zeroing that component preserves Hermitian symmetry; complex fields
    retain the full signed Fourier convention.
    """

    wave_vectors = list(domain.wave_vectors(device=field.device, dtype=field.real.dtype))
    if field.is_complex():
        return tuple(wave_vectors)
    for axis, n in enumerate(domain.shape):
        if n % 2 == 0:
            index = [slice(None)] * domain.dim
            index[axis] = n // 2
            wave_vectors[axis] = wave_vectors[axis].clone()
            wave_vectors[axis][tuple(index)] = 0
    return tuple(wave_vectors)


def spectral_gradient(field: Tensor, domain: PeriodicDomain) -> Tensor:
    """FFT gradient, exact for modes represented on the periodic grid."""

    domain.validate_field(field)
    transformed = torch.fft.fftn(field, dim=domain.spatial_axes)
    wave_vectors = _first_derivative_wave_vectors(field, domain)
    derivatives = [
        torch.fft.ifftn(1j * k * transformed, dim=domain.spatial_axes)
        for k in wave_vectors
    ]
    if not field.is_complex():
        derivatives = [value.real for value in derivatives]
    component_axis = field.ndim - domain.dim
    return torch.stack(derivatives, dim=component_axis)


def spectral_laplacian(field: Tensor, domain: PeriodicDomain) -> Tensor:
    """FFT Laplacian using the multiplier ``-|k|^2``."""

    domain.validate_field(field)
    transformed = torch.fft.fftn(field, dim=domain.spatial_axes)
    k_squared = domain.wave_number_squared(
        device=field.device, dtype=field.real.dtype
    )
    result = torch.fft.ifftn(-k_squared * transformed, dim=domain.spatial_axes)
    return result if field.is_complex() else result.real


def l2_mass(field: Tensor, domain: PeriodicDomain) -> Tensor:
    """Spatial integral of ``|field|^2``, retaining any leading batch axes."""

    domain.validate_field(field)
    return (
        torch.sum(torch.abs(field) ** 2, dim=domain.spatial_axes) * domain.cell_volume
    )


def spatial_broadcast(values: Tensor, domain: PeriodicDomain) -> Tensor:
    """Append singleton spatial axes so per-sample scalars broadcast over the grid."""

    return values.reshape(*values.shape, *((1,) * domain.dim))


def batch_parameter(
    value: Tensor | float,
    batch: int,
    domain: PeriodicDomain,
    reference: Tensor,
    name: str,
) -> Tensor:
    """Normalize a scalar-or-per-sample PDE parameter to a grid-broadcastable tensor.

    Rejects a silent precision downcast.  ``torch.tensor([0.9])`` is float32, and
    feeding it to a float64 field costs ~2.6e-8 in alpha -- enough to move the
    one-step phase at k=30 by 5e-7 and to invalidate every 1e-10 assertion in the
    Phase 0 suite.  Pass a python float or an explicitly-typed tensor instead.
    """

    if isinstance(value, Tensor) and value.is_floating_point():
        target_dtype = reference.real.dtype
        if torch.finfo(value.dtype).bits < torch.finfo(target_dtype).bits:
            raise ValueError(
                f"{name} is {value.dtype} but the field is {target_dtype}; this would "
                f"silently lose precision. Pass a python float or a {target_dtype} tensor."
            )
    tensor = torch.as_tensor(
        value, device=reference.device, dtype=reference.real.dtype
    )
    if tensor.ndim == 0:
        tensor = tensor.expand(batch)
    if tensor.shape != (batch,):
        raise ValueError(f"{name} must be scalar or have shape (batch,)")
    return tensor.reshape(batch, *((1,) * domain.dim))


def project_to_mass(
    prediction: Tensor,
    reference: Tensor,
    domain: PeriodicDomain,
    *,
    eps: float = 1e-14,
) -> Tensor:
    """Rescale each predicted field to the corresponding reference mass.

    This is Model B's entire physics content: it enforces one scalar invariant and
    says nothing about phase, energy, or reversibility.
    """

    domain.validate_field(prediction)
    domain.validate_field(reference)
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference must have identical shapes")
    predicted_mass = l2_mass(prediction, domain)
    reference_mass = l2_mass(reference, domain)
    if torch.any((predicted_mass <= eps) & (reference_mass > eps)):
        raise ValueError("cannot project a zero field to positive mass")
    scale = torch.where(
        reference_mass <= eps,
        torch.zeros_like(reference_mass),
        torch.sqrt(reference_mass / torch.clamp(predicted_mass, min=eps)),
    )
    return prediction * spatial_broadcast(scale, domain)
