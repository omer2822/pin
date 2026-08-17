"""Phase 8: spectral resampling, domain rebinding, and an honest grid-dependence report.

**The distinction this module exists to keep straight.**  Upsampling a *band-limited*
field and re-running an FNO barely changes its output, because the FNO only touches
``k <= n_modes`` and those modes are unchanged by the resample.  That is not evidence of
resolution transfer and must never be presented as such.  The genuine test is new
energy above the training grid's Nyquist -- which is G4 (spectral extrapolation) wearing
a different hat.  :func:`grid_dependence` reports which parameters actually carry the
transfer and which merely survive it.
"""

from __future__ import annotations

import torch

from ..domain import PeriodicDomain

Tensor = torch.Tensor

#: Relative energy at the source Nyquist above which upsampling is refused.
NYQUIST_TOLERANCE = 1e-12


def spectral_resample(
    field: Tensor, source: PeriodicDomain, target: PeriodicDomain
) -> Tensor:
    """Move a periodic field between grids by truncating or zero-padding its spectrum.

    Exact for any field band-limited below both Nyquists, which is the case the Phase 8
    band-limited arm needs.  With ``norm="backward"`` the inverse transform carries the
    ``1/N``, so coefficients are rescaled by ``N_target / N_source`` to keep the
    physical amplitude fixed rather than the discrete one.

    Upsampling a field with energy at the **source Nyquist** is refused.  That mode is
    self-conjugate -- it represents ``cos(N x / 2)`` with no way to distinguish ``+k``
    from ``-k`` -- so splitting it across the two target modes is a *choice*, not a
    fact, and silently making one would put an arbitrary phase into every downstream
    number.
    """

    if source.dim != 1 or target.dim != 1:
        raise NotImplementedError("spectral_resample is implemented for 1D domains")
    source.validate_field(field)
    n_source = source.shape[0]
    n_target = target.shape[0]
    if n_source == n_target:
        return field.clone()

    hat = torch.fft.fftn(field, dim=source.spatial_axes)
    nyquist_index = n_source // 2

    if n_target > n_source:
        total = torch.abs(hat).pow(2).sum(dim=-1).clamp_min(1e-300)
        at_nyquist = torch.abs(hat[..., nyquist_index]).pow(2)
        if float((at_nyquist / total).max()) > NYQUIST_TOLERANCE:
            raise ValueError(
                "cannot upsample a field carrying energy at the source Nyquist mode "
                f"(k={nyquist_index}): it is self-conjugate, so splitting it between "
                "+k and -k on the finer grid is a choice rather than a fact. "
                "Band-limit the field below Nyquist first."
            )

    keep = min(n_source, n_target) // 2
    shape = list(field.shape)
    shape[-1] = n_target
    resampled = torch.zeros(shape, dtype=hat.dtype, device=hat.device)
    for wave_number in range(-keep + 1, keep):
        resampled[..., wave_number % n_target] = hat[..., wave_number % n_source]

    resampled = resampled * (n_target / n_source)
    return torch.fft.ifftn(resampled, dim=target.spatial_axes)


def rebind_domain(model, domain: PeriodicDomain) -> None:
    """Point a trained model at a new grid, in place, preserving what it learned.

    Dispatches by type rather than by a shared method, because *what* has to move
    differs: the FNO's spectral weights are indexed by frequency and need nothing, while
    ``KineticPhase`` holds a grid-shaped buffer whose normalizer must be frozen.
    """

    # Imported here rather than at module scope: evaluation importing models at import
    # time would make the dependency circular via models -> domain -> evaluation.
    from ..models.fno import FNOStepOperator
    from ..models.projected import MassProjectedOperator
    from ..models.split_learned import LearnedSplitStep

    if isinstance(model, MassProjectedOperator):
        rebind_domain(model.core, domain)
        model.domain = domain
        return

    if isinstance(model, FNOStepOperator):
        limit = domain.shape[0] // 2 + 1
        if model.modes > limit:
            raise ValueError(
                f"model keeps {model.modes} modes but a grid of {domain.shape[0]} "
                f"points resolves only {limit}. Retaining more modes than the grid "
                "supports would alias distinct wave numbers onto one coefficient."
            )
        model.domain = domain
        return

    if isinstance(model, LearnedSplitStep):
        model.domain = domain
        if model.kinetic is not None:
            model.kinetic.rebind_domain(domain)
        # LocalPhaseLadder is pointwise in (rho, V, alpha, beta) and holds no
        # grid-shaped buffer, so it is grid-independent with nothing to rebind.
        # FieldPhaseFNO (C2's phase net) has no domain attribute either: it is built
        # from SpectralConv1d, which indexes by frequency and passes n through at call
        # time.
        return

    raise NotImplementedError(
        f"rebind_domain does not know how to move a {type(model).__name__}; "
        "add an explicit branch rather than letting it silently keep the old grid."
    )


def grid_dependence(model) -> dict[str, str]:
    """One line per parameter group: is it grid-independent, and why or why not.

    Phase 8 asks the thesis to "write out which parameters are grid-independent and
    which are not".  This is that statement, produced from the model rather than from
    prose, so it cannot drift away from the code.
    """

    from ..models.fno import FNOStepOperator
    from ..models.projected import MassProjectedOperator
    from ..models.split_learned import LearnedSplitStep

    if isinstance(model, MassProjectedOperator):
        report = grid_dependence(model.core)
        report["projection"] = (
            "grid-independent: rescaling by a mass ratio involves no grid-shaped "
            "parameter"
        )
        return report

    if isinstance(model, FNOStepOperator):
        return {
            "spectral.weight": (
                f"grid-independent for k <= modes ({model.modes}): weights are indexed "
                "by frequency, and irfft receives n at call time. This is the ONLY "
                "source of the FNO's resolution transfer, and it says nothing about "
                "modes above the budget"
            ),
            "lift/local/project": (
                "grid-independent: pointwise in the channel axis, applied identically "
                "at every point"
            ),
        }

    if isinstance(model, LearnedSplitStep):
        report: dict[str, str] = {}
        if model.kinetic is not None:
            scale = float(model.kinetic.k_squared_scale)
            highest = float(model.kinetic.k_squared.max())
            if highest > scale:
                report["kinetic.k_squared"] = (
                    f"grid-dependent input range: the normalizer is frozen at "
                    f"{scale:g} (the training grid) while this grid reaches {highest:g}, "
                    f"so modes above k^2={scale:g} enter the MLP at k^2/k^2_max > 1 -- "
                    "outside the range it was ever trained on. Report this arm as "
                    "spectral extrapolation (G4), not as resolution transfer"
                )
            else:
                report["kinetic.k_squared"] = (
                    f"grid-independent: normalizer frozen at {scale:g} and this grid "
                    f"reaches only {highest:g}, so every mode enters the MLP within "
                    "the trained input range"
                )
        report["local phase"] = (
            "grid-independent: pointwise in (rho, V, alpha, beta), no grid-shaped "
            "parameter"
        )
        return report

    raise NotImplementedError(
        f"grid_dependence has no entry for {type(model).__name__}"
    )
