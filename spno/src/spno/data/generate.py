"""Samplers for the parametric NLS training distribution.

Two design choices here exist specifically to keep later experiments honest:

* **Sharp spectral cutoff.**  Initial conditions are exactly band-limited to
  ``|k| <= bandwidth``, not merely rolled off by a Gaussian.  The spectral-shift
  experiments (G4) compare a model trained on one band against fields carrying
  genuine energy in another; a soft filter that leaks high-k energy into the
  training set would void that design without any visible symptom.
* **Mass varies across samples.**  Normalizing every initial condition to unit mass
  makes mass conservation memorizable, which flatters the unconstrained baselines and
  turns Model B's projection into a no-op.  Mass is drawn per sample instead.

Generation runs in float64 on CPU.  Training may later cast to complex64.
"""

from __future__ import annotations

import math
from typing import Literal

import torch

from ..domain import PeriodicDomain, l2_mass, spatial_broadcast

Tensor = torch.Tensor

PotentialFamily = Literal["random", "zero", "cosine", "gaussian_well", "harmonic"]


def _wave_number_magnitude(domain: PeriodicDomain) -> Tensor:
    """``|k|`` on the grid, in integer units when lengths are ``2*pi``."""

    return torch.sqrt(domain.wave_number_squared())


def band_limited_field(
    domain: PeriodicDomain,
    batch: int,
    bandwidth: int,
    generator: torch.Generator,
    *,
    complex_valued: bool = True,
) -> Tensor:
    """Smooth random field with energy exactly zero above ``bandwidth``.

    Real and imaginary parts are drawn independently, as the NLS state is a genuine
    complex field rather than a pair of coupled real ones.
    """

    if bandwidth < 1:
        raise ValueError("bandwidth must be at least 1")
    magnitude = _wave_number_magnitude(domain)
    # Gaussian envelope inside the band, hard zero outside it.
    envelope = torch.exp(-0.5 * (magnitude / bandwidth) ** 2)
    envelope = torch.where(magnitude <= bandwidth, envelope, torch.zeros_like(envelope))

    def _draw() -> Tensor:
        noise = torch.randn(
            batch, *domain.shape, generator=generator, dtype=torch.float64
        )
        return torch.fft.ifftn(
            torch.fft.fftn(noise, dim=domain.spatial_axes) * envelope,
            dim=domain.spatial_axes,
        )

    if not complex_valued:
        return _draw().real
    return _draw().real + 1j * _draw().real


def sample_initial_conditions(
    domain: PeriodicDomain,
    batch: int,
    bandwidth: int,
    mass_range: tuple[float, float],
    generator: torch.Generator,
) -> Tensor:
    """Band-limited complex initial conditions with per-sample mass in ``mass_range``."""

    low, high = mass_range
    if not 0 < low <= high:
        raise ValueError("mass_range must satisfy 0 < low <= high")
    field = band_limited_field(domain, batch, bandwidth, generator)
    target_mass = low + (high - low) * torch.rand(
        batch, generator=generator, dtype=torch.float64
    )
    scale = torch.sqrt(target_mass / torch.clamp(l2_mass(field, domain), min=1e-30))
    return field * spatial_broadcast(scale, domain)


def sample_potentials(
    domain: PeriodicDomain,
    batch: int,
    amplitude_range: tuple[float, float],
    correlation_length: float,
    generator: torch.Generator,
    *,
    family: PotentialFamily = "random",
) -> Tensor:
    """Real smooth potentials, normalized so ``max|V|`` equals the sampled amplitude.

    Amplitude, correlation length, and functional family are controlled independently
    so the potential-generalization experiment (G3) can vary one at a time.
    """

    low, high = amplitude_range
    if low < 0 or high < low:
        raise ValueError("amplitude_range must satisfy 0 <= low <= high")
    amplitude = low + (high - low) * torch.rand(
        batch, generator=generator, dtype=torch.float64
    )

    if family == "zero":
        return torch.zeros(batch, *domain.shape, dtype=torch.float64)
    if family != "random" and domain.dim != 1:
        # These are explicitly 1D constructions.  Using only the first coordinate in
        # higher dimensions would silently drop the other axes' dependence, which is
        # worse than refusing.
        raise NotImplementedError(
            f"potential family {family!r} is only defined for 1D domains"
        )
    (x,) = domain.mesh() if domain.dim == 1 else (None,)

    if family == "cosine":
        shape = torch.cos(2 * x).expand(batch, *domain.shape)
    elif family == "gaussian_well":
        center = math.pi
        width = max(correlation_length, 1e-3)
        shape = -torch.exp(-0.5 * ((x - center) / width) ** 2).expand(
            batch, *domain.shape
        )
    elif family == "harmonic":
        # Periodic stand-in for a harmonic well: smooth, single-minimum, C-infinity.
        shape = (1 - torch.cos(x - math.pi)).expand(batch, *domain.shape)
    elif family == "random":
        magnitude = _wave_number_magnitude(domain)
        envelope = torch.exp(-0.5 * (magnitude * correlation_length) ** 2)
        noise = torch.randn(
            batch, *domain.shape, generator=generator, dtype=torch.float64
        )
        shape = torch.fft.ifftn(
            torch.fft.fftn(noise, dim=domain.spatial_axes) * envelope,
            dim=domain.spatial_axes,
        ).real
    else:
        raise ValueError(f"unknown potential family {family!r}")

    peak = torch.amax(torch.abs(shape), dim=domain.spatial_axes)
    normalized = shape / spatial_broadcast(torch.clamp(peak, min=1e-30), domain)
    return normalized * spatial_broadcast(amplitude, domain)


def sample_parameters(
    batch: int,
    alpha_range: tuple[float, float],
    beta_range: tuple[float, float],
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Draw ``(alpha, beta)`` uniformly, in float64 to avoid a silent downcast."""

    def _uniform(bounds: tuple[float, float]) -> Tensor:
        low, high = bounds
        if high < low:
            raise ValueError("range bounds must satisfy low <= high")
        return low + (high - low) * torch.rand(
            batch, generator=generator, dtype=torch.float64
        )

    return _uniform(alpha_range), _uniform(beta_range)


def energy_fraction_above(field: Tensor, domain: PeriodicDomain, cutoff: float) -> Tensor:
    """Fraction of spectral energy strictly above wave number ``cutoff``.

    Used to check whether in-distribution rollouts stay inside the training band.  The
    NLS nonlinearity moves energy up in ``k``, so a "parameter interpolation" run can
    silently become a spectral-extrapolation run after enough steps -- which would
    stop G1 from being a clean control.  Measured, not assumed.
    """

    domain.validate_field(field)
    magnitude = _wave_number_magnitude(domain)
    spectrum = torch.abs(torch.fft.fftn(field, dim=domain.spatial_axes)) ** 2
    total = torch.sum(spectrum, dim=domain.spatial_axes)
    above = torch.sum(
        torch.where(magnitude > cutoff, spectrum, torch.zeros_like(spectrum)),
        dim=domain.spatial_axes,
    )
    return above / torch.clamp(total, min=1e-30)


def spectral_tail_fraction(field: Tensor, domain: PeriodicDomain, band_start: float) -> Tensor:
    """Fraction of spectral energy above ``band_start * k_nyquist``.

    The resolution-adequacy screen: focusing NLS (beta > 0) can steepen a field until
    the grid no longer resolves it, and the resulting error looks exactly like an
    out-of-distribution failure.  Trajectories whose tail exceeds a threshold are
    under-resolved and must be excluded or the (alpha, beta) box narrowed.
    """

    domain.validate_field(field)
    magnitude = _wave_number_magnitude(domain)
    nyquist = max(domain.shape) // 2
    spectrum = torch.abs(torch.fft.fftn(field, dim=domain.spatial_axes)) ** 2
    total = torch.sum(spectrum, dim=domain.spatial_axes)
    tail = torch.sum(
        torch.where(magnitude >= band_start * nyquist, spectrum, torch.zeros_like(spectrum)),
        dim=domain.spatial_axes,
    )
    return tail / torch.clamp(total, min=1e-30)
