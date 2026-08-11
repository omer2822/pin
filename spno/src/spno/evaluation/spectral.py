"""Per-mode error, phase error, and the nonlinear cascade.

Field-space L2 hides the two failures this study is about.  A model can have small
relative L2 while getting the *dispersion relation* wrong -- error concentrated in a
few high-k modes barely moves the norm -- and phase error is invisible to any metric
that looks at ``|psi|``.  These functions decompose the error by wavenumber and
separate amplitude from phase.

``k_wrap = sqrt(pi / (alpha dt))`` is marked on every spectral plot: above it the
one-step map determines omega only modulo ``2 pi / dt`` at fixed alpha, so error growth
there is expected rather than anomalous.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..domain import PeriodicDomain

Tensor = torch.Tensor


def wave_numbers(domain: PeriodicDomain, *, dtype: torch.dtype = torch.float64) -> Tensor:
    """``|k|`` on the grid, in integer units when the domain length is ``2*pi``."""

    return torch.sqrt(domain.wave_number_squared(dtype=dtype))


def mode_error_spectrum(
    prediction: Tensor, target: Tensor, domain: PeriodicDomain, *, relative: bool = True
) -> Tensor:
    """``E(k) = |psi_pred(k) - psi_true(k)|`` averaged over the batch.

    With ``relative=True`` each mode is normalized by the target's amplitude at that
    mode, which is what makes low-energy high-k modes visible at all; the absolute
    version is dominated by whichever modes carry the energy.
    """

    domain.validate_field(prediction)
    domain.validate_field(target)
    predicted_hat = torch.fft.fftn(prediction, dim=domain.spatial_axes)
    target_hat = torch.fft.fftn(target, dim=domain.spatial_axes)
    difference = torch.abs(predicted_hat - target_hat)
    if relative:
        difference = difference / torch.abs(target_hat).clamp_min(1e-30)
    return difference.mean(dim=0)


def banded_error(
    prediction: Tensor,
    target: Tensor,
    domain: PeriodicDomain,
    edges: tuple[float, ...],
) -> dict[str, float]:
    """Relative L2 restricted to wavenumber bands, e.g. below/above ``k_wrap``."""

    magnitude = wave_numbers(domain, dtype=prediction.real.dtype)
    predicted_hat = torch.fft.fftn(prediction, dim=domain.spatial_axes)
    target_hat = torch.fft.fftn(target, dim=domain.spatial_axes)
    results = {}
    for low, high in zip((0.0, *edges), (*edges, float("inf"))):
        mask = (magnitude >= low) & (magnitude < high)
        if not bool(mask.any()):
            continue
        numerator = torch.sqrt(
            (torch.abs(predicted_hat - target_hat) ** 2 * mask).sum(
                dim=domain.spatial_axes
            )
        )
        denominator = torch.sqrt(
            (torch.abs(target_hat) ** 2 * mask).sum(dim=domain.spatial_axes)
        )
        label = f"{low:g}-{high:g}" if high != float("inf") else f"{low:g}+"
        results[label] = float((numerator / denominator.clamp_min(1e-30)).mean())
    return results


def phase_error(prediction: Tensor, target: Tensor, domain: PeriodicDomain) -> Tensor:
    """Per-mode phase discrepancy in radians, wrapped to ``(-pi, pi]``.

    Averaged over the batch with the target's amplitude as the weight.  This is a
    *curve* over wavenumber and is meaningful only where the target carries energy: in
    an empty mode the phase is roundoff noise, uniformly distributed on ``(-pi, pi]``.
    Reduce it with :func:`mean_phase_error`, never with a bare ``max`` -- that reports
    the noisiest empty mode, not the model's phase behaviour.
    """

    predicted_hat = torch.fft.fftn(prediction, dim=domain.spatial_axes)
    target_hat = torch.fft.fftn(target, dim=domain.spatial_axes)
    difference = torch.angle(predicted_hat * torch.conj(target_hat))
    weight = torch.abs(target_hat)
    return (difference.abs() * weight).sum(dim=0) / weight.sum(dim=0).clamp_min(1e-30)


def mean_phase_error(
    prediction: Tensor, target: Tensor, domain: PeriodicDomain
) -> float:
    """Amplitude-weighted mean phase error **across modes**: the scalar summary.

    Weighting across modes is what makes this a usable number.  Without it, a field
    whose energy sits in a handful of modes reports the roundoff phase of the empty
    ones, which is O(1) regardless of how good the model is.
    """

    target_hat = torch.fft.fftn(target, dim=domain.spatial_axes)
    predicted_hat = torch.fft.fftn(prediction, dim=domain.spatial_axes)
    difference = torch.angle(predicted_hat * torch.conj(target_hat)).abs()
    weight = torch.abs(target_hat)
    return float(
        (difference * weight).sum() / weight.sum().clamp_min(1e-30)
    )


def amplitude_and_phase_split(
    prediction: Tensor, target: Tensor, domain: PeriodicDomain
) -> dict[str, float]:
    """Split the field error into an amplitude part and a phase part.

    ``|psi|`` can be right while the phase drifts, which is the failure mode a plain L2
    on the complex field reports as a single opaque number.
    """

    amplitude_error = torch.sqrt(
        torch.sum(
            (torch.abs(prediction) - torch.abs(target)) ** 2, dim=domain.spatial_axes
        )
    )
    total_error = torch.sqrt(
        torch.sum(torch.abs(prediction - target) ** 2, dim=domain.spatial_axes)
    )
    scale = torch.sqrt(
        torch.sum(torch.abs(target) ** 2, dim=domain.spatial_axes)
    ).clamp_min(1e-30)
    # Everything the amplitude error does not explain is phase.
    phase_component = torch.sqrt((total_error**2 - amplitude_error**2).clamp_min(0))
    return {
        "total": float((total_error / scale).mean()),
        "amplitude": float((amplitude_error / scale).mean()),
        "phase": float((phase_component / scale).mean()),
    }


def energy_spectrum(field: Tensor, domain: PeriodicDomain) -> Tensor:
    """``|psi_hat(k)|^2`` averaged over the batch: the cascade diagnostic."""

    return (
        torch.abs(torch.fft.fftn(field, dim=domain.spatial_axes)) ** 2
    ).mean(dim=0)


@dataclass
class SpectralMetrics:
    wave_numbers: list[float]
    relative_mode_error: list[float]
    phase_error: list[float]
    banded: dict[str, float]
    amplitude_phase: dict[str, float]
    mean_phase_error: float

    def as_dict(self) -> dict:
        return {
            "wave_numbers": self.wave_numbers,
            "relative_mode_error": self.relative_mode_error,
            "phase_error": self.phase_error,
            "banded": self.banded,
            "amplitude_phase": self.amplitude_phase,
            "mean_phase_error": self.mean_phase_error,
        }


@torch.no_grad()
def evaluate_spectral(
    prediction: Tensor,
    target: Tensor,
    domain: PeriodicDomain,
    *,
    band_edges: tuple[float, ...] = (8.0, 16.9),
) -> SpectralMetrics:
    """Default band edges are ``k_train`` and ``k_wrap`` for the production config."""

    magnitude = wave_numbers(domain, dtype=prediction.real.dtype)
    order = torch.argsort(magnitude)
    return SpectralMetrics(
        wave_numbers=magnitude[order].tolist(),
        relative_mode_error=mode_error_spectrum(prediction, target, domain)[order].tolist(),
        phase_error=phase_error(prediction, target, domain)[order].tolist(),
        banded=banded_error(prediction, target, domain, band_edges),
        amplitude_phase=amplitude_and_phase_split(prediction, target, domain),
        mean_phase_error=mean_phase_error(prediction, target, domain),
    )
