"""Model A: the unrestricted Fourier Neural Operator baseline.

Adapted from ``pinn-neural-operators/03_fno1d.py`` (``SpectralConv1d``, ``FNO1d``).
Three deliberate departures from that tutorial version, each tied to a confound:

* **Five input channels** ``(Re psi, Im psi, V, alpha, beta)`` rather than one, since
  this is a *parametric* operator-learning problem.  ``alpha`` and ``beta`` are
  broadcast over the grid and rescaled to ``[-1, 1]`` from their known sampling
  ranges, so conditioning enters at O(1) magnitude.

* **No absolute-coordinate channel by default.**  The tutorial feeds ``x`` into the
  lift.  Here the joint operator ``(psi, V) -> psi'`` is translation-equivariant --
  the potential, not the coordinate, is what breaks homogeneity -- so feeding ``x``
  lets the model memorize position instead of learning the operator.  Kept behind
  ``use_coordinate_channel`` as an ablation.

* **A single global amplitude scale**, not per-sample normalization.  The NLS
  nonlinearity ``beta|psi|^2 psi`` is not scale-invariant, and mass varies across
  samples by design; normalizing each sample by its own norm would destroy exactly the
  information the nonlinear term depends on.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..domain import PeriodicDomain
from .base import StepOperator

Tensor = torch.Tensor


class SpectralConv1d(nn.Module):
    """Global convolution as a pointwise multiply on the lowest ``modes`` frequencies."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        if modes < 1:
            raise ValueError("modes must be positive")
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale
            * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, _, n = x.shape
        transformed = torch.fft.rfft(x)
        # Guard borrowed from ``pinn-neural-operators/06_fno_core.py`` (:110, :187).
        # Global Constraint 1 documents two silent failure modes this converts into
        # exceptions: ``.to(torch.float64)`` casts the complex spectral weights to
        # *real*, destroying the operator with only a warning, and feeding a float32
        # field to a widened model would otherwise promote instead of complaining.
        if not self.weight.is_complex():
            raise ValueError(
                f"{type(self).__name__} weights are {self.weight.dtype}, not complex. "
                "They were cast to real -- almost certainly by .to(torch.float64), "
                "which destroys the operator. Use precision.widen_to_double."
            )
        if transformed.dtype != self.weight.dtype:
            raise ValueError(
                f"input transforms to {transformed.dtype} but the spectral weights are "
                f"{self.weight.dtype}; this would silently promote. Widen the model "
                "with precision.widen_to_double, or pass a matching field dtype."
            )
        retained = min(self.modes, n // 2 + 1)
        # Derive the buffer dtype from the weight rather than hardcoding cfloat, so
        # ``model.double()`` works.  Invariant measurements are run in float64 to
        # separate an architecture's mass behaviour from float32 arithmetic drift,
        # which accumulates linearly at ~1.2e-7 per split step.
        output = torch.zeros(
            batch,
            self.weight.shape[1],
            n // 2 + 1,
            dtype=self.weight.dtype,
            device=x.device,
        )
        output[:, :, :retained] = torch.einsum(
            "bik,iok->bok", transformed[:, :, :retained], self.weight[:, :, :retained]
        )
        # Passing n keeps the same weights valid on any grid size: modes are indexed by
        # frequency, not by array position.  This is the only source of the FNO's
        # resolution transfer, and it holds only for k <= modes.
        return torch.fft.irfft(output, n=n)


class FNOStepOperator(StepOperator):
    """Unrestricted one-step map ``(psi, V, alpha, beta) -> psi``."""

    def __init__(
        self,
        domain: PeriodicDomain,
        *,
        modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
        alpha_range: tuple[float, float] = (0.7, 1.1),
        beta_range: tuple[float, float] = (-0.4, 0.6),
        field_scale: float = 1.0,
        use_coordinate_channel: bool = False,
        trained_dt: float | None = None,
    ) -> None:
        super().__init__(domain, trained_dt)
        if domain.dim != 1:
            raise NotImplementedError("FNOStepOperator is implemented for 1D domains")
        self.modes = modes
        self.width = width
        self.use_coordinate_channel = use_coordinate_channel
        self.register_buffer("alpha_range", torch.tensor(alpha_range))
        self.register_buffer("beta_range", torch.tensor(beta_range))
        self.register_buffer("field_scale", torch.tensor(float(field_scale)))

        in_channels = 5 + (1 if use_coordinate_channel else 0)
        self.lift = nn.Linear(in_channels, width)
        self.spectral = nn.ModuleList(
            SpectralConv1d(width, width, modes) for _ in range(n_layers)
        )
        self.local = nn.ModuleList(
            nn.Conv1d(width, width, 1) for _ in range(n_layers)
        )
        self.project = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 2)
        )

    def _rescale(self, values: Tensor, bounds: Tensor) -> Tensor:
        low, high = bounds[0], bounds[1]
        return 2 * (values - low) / (high - low) - 1

    def _features(
        self, field: Tensor, potential: Tensor, alpha: Tensor, beta: Tensor
    ) -> Tensor:
        batch, n = field.shape
        scale = self.field_scale
        channels = [
            field.real / scale,
            field.imag / scale,
            potential,
            self._rescale(alpha, self.alpha_range).reshape(batch, 1).expand(batch, n),
            self._rescale(beta, self.beta_range).reshape(batch, 1).expand(batch, n),
        ]
        if self.use_coordinate_channel:
            (x,) = self.domain.mesh(device=field.device, dtype=field.real.dtype)
            channels.append(x.reshape(1, n).expand(batch, n))
        return torch.stack(channels, dim=-1)

    def step(
        self,
        field: Tensor,
        potential: Tensor,
        alpha: Tensor,
        beta: Tensor,
        dt: float,
    ) -> Tensor:
        hidden = self.lift(self._features(field, potential, alpha, beta))
        hidden = hidden.permute(0, 2, 1)
        for index, (spectral, local) in enumerate(zip(self.spectral, self.local)):
            updated = spectral(hidden) + local(hidden)
            hidden = F.gelu(updated) if index < len(self.spectral) - 1 else updated
        output = self.project(hidden.permute(0, 2, 1))
        return torch.complex(output[..., 0], output[..., 1]) * self.field_scale
