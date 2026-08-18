"""The plane-wave dispersion probe: ``omega_model(k; alpha, beta, A)`` for any operator.

*Theorem (single-alpha ambiguity).*  A one-step map observed on a plane wave yields the
Fourier multiplier ``m(k) = exp(-i omega dt)``.  ``arg m`` is defined modulo ``2 pi``, so
``omega`` is determined only modulo ``2 pi / dt``.  Above
``k_wrap = sqrt(pi / (alpha dt))`` the kinetic phase ``alpha k^2 dt`` exceeds ``pi`` and
the branch is genuinely unrecoverable from fixed-alpha data.  Multi-step data at the
same ``dt`` adds nothing: the n-step multiplier is a function of the one-step multiplier.

*What this instrument is not.*  Three limits, each load-bearing:

1.  Plane waves are **out of distribution** for every model here -- they are trained on
    random band-limited fields.  This probes the learned operator, not generalization.
2.  For a plane wave in a **constant** potential the Strang step is *exact*: the local
    phase reads ``|midpoint|^2``, which is constant in x.  So ``eps_split`` (6.239e-5,
    measured on random fields) is **not** the floor for probe residuals; the floor here
    is float64 roundoff, ~1e-13 in omega at dt=0.01.
3.  :func:`unwrap_to_reference` selects the branch **using the truth**.  Below
    ``k_wrap`` the branch is zero and the lift is a consistency check; above it the lift
    is an assist and the "recovered" omega is not an independent measurement.  That
    asymmetry is the G5a result, not a defect in the probe.

The probe is **constant-``V0`` only** -- a plane wave is an exact solution only then --
and runs in float64 on CPU, because a phase measurement at 1e-13 has no meaning in
float32 and MPS has no float64 at all.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch

from ..domain import PeriodicDomain
from ..equations.nls import (
    alpha_sampling_is_dense_enough,
    exact_dispersion,
    plane_wave,
    wrap_wavenumber,
)
from ..precision import widen_to_double

Tensor = torch.Tensor


def probe_amplitude(domain: PeriodicDomain, mass_range: tuple[float, float]) -> float:
    """Amplitude whose plane-wave mass sits at the centre of the training mass range.

    ``plane_wave_mass = A^2 * prod(lengths)``, so ``A = sqrt(centre / prod(lengths))``.
    Choosing it any other way confounds spectral extrapolation with mass
    extrapolation: a probe amplitude outside the training mass distribution makes every
    high-k reading a measurement of two shifts at once.
    """

    centre = 0.5 * (float(mass_range[0]) + float(mass_range[1]))
    return math.sqrt(centre / math.prod(domain.lengths))


def mode_index(wave_number: int, domain: PeriodicDomain) -> int:
    """Index of mode ``k`` under unshifted FFT ordering: ``k mod N``.

    ``k = N/2`` is the Nyquist mode and is self-conjugate -- it is stored as ``-N/2``.
    Harmless for this probe, since ``omega`` is even in ``k``.
    """

    (n,) = domain.shape
    if abs(int(wave_number)) > n // 2:
        raise ValueError(
            f"wave_number {wave_number} is beyond Nyquist ({n // 2}) for N={n}"
        )
    return int(wave_number) % n


@torch.no_grad()
def one_step_multiplier(
    model,
    domain: PeriodicDomain,
    wave_number: int,
    *,
    alpha: float,
    beta: float,
    amplitude: float,
    potential_constant: float,
    dt: float,
) -> complex:
    """``m(k) = psi_hat_out(k) / psi_hat_in(k)`` for a plane-wave input.

    ``model`` may be any object with the project's call signature
    ``(field, potential, alpha, beta, dt)`` -- a :class:`StepOperator` or the reference
    solver.  Parameters are built as explicit float64 tensors: a bare
    ``torch.tensor([0.9])`` is float32 and would move the phase at k=30 by ~5e-7.
    """

    field = plane_wave(
        domain, (int(wave_number),), amplitude=float(amplitude),
        dtype=torch.complex128,
    ).unsqueeze(0)  # time=0, so the dispersion arguments of plane_wave are unused
    potential = torch.full_like(field.real, float(potential_constant))
    alpha_t = torch.tensor([float(alpha)], dtype=torch.float64)
    beta_t = torch.tensor([float(beta)], dtype=torch.float64)

    evolved = model(field, potential, alpha_t, beta_t, float(dt))

    index = mode_index(wave_number, domain)
    before = torch.fft.fftn(field, dim=domain.spatial_axes)[0, index]
    after = torch.fft.fftn(evolved, dim=domain.spatial_axes)[0, index]
    return complex(after / before)


def principal_frequency(multiplier: complex, dt: float) -> float:
    """``omega`` on the principal branch, from ``arg m = -omega dt`` (mod 2 pi).

    Lands in ``[-pi/dt, pi/dt)``.  Equal to the true omega only when
    ``|omega| dt <= pi``, i.e. below ``k_wrap``.
    """

    return -cmath.phase(multiplier) / float(dt)


def unwrap_to_reference(
    principal: float, reference: float, dt: float
) -> tuple[float, int]:
    """Lift ``principal`` onto the branch of ``2 pi / dt`` nearest ``reference``.

    Returns ``(omega, branch)``.  ``branch != 0`` is the signature of wrapping and is
    reported rather than hidden: above ``k_wrap`` a nonzero branch means the truth was
    used to choose it, so the lifted value is an assist, not a measurement.
    """

    period = 2 * math.pi / float(dt)
    branch = int(round((float(reference) - float(principal)) / period))
    return float(principal) + branch * period, branch


@dataclass
class DispersionCurve:
    wave_numbers: list[int]
    principal: list[float]
    unwrapped: list[float]
    branch: list[int]
    truth: list[float]
    residual: list[float]
    k_wrap: float
    alpha: float
    beta: float
    amplitude: float
    potential_constant: float
    dt: float

    def as_dict(self) -> dict:
        return {
            "wave_numbers": self.wave_numbers,
            "principal": self.principal,
            "unwrapped": self.unwrapped,
            "branch": self.branch,
            "truth": self.truth,
            "residual": self.residual,
            "k_wrap": self.k_wrap,
            "alpha": self.alpha,
            "beta": self.beta,
            "amplitude": self.amplitude,
            "potential_constant": self.potential_constant,
            "dt": self.dt,
        }


@torch.no_grad()
def dispersion_curve(
    model,
    domain: PeriodicDomain,
    wave_numbers: Iterable[int],
    *,
    alpha: float,
    beta: float,
    amplitude: float,
    potential_constant: float,
    dt: float,
    widen: bool = False,
) -> DispersionCurve:
    """``omega_model(k)`` across a range of ``k``, with the branch reported per mode.

    Set ``widen=True`` for a trained model: it deep-copies onto CPU/float64 via
    :func:`widen_to_double`, leaving the original on its training device.  The
    reference solver needs no widening and the default avoids the copy.
    """

    probe = widen_to_double(model, device="cpu").eval() if widen else model
    ks = [int(k) for k in wave_numbers]
    principal, unwrapped, branches, truths, residuals = [], [], [], [], []
    for k in ks:
        multiplier = one_step_multiplier(
            probe, domain, k, alpha=alpha, beta=beta, amplitude=amplitude,
            potential_constant=potential_constant, dt=dt,
        )
        raw = principal_frequency(multiplier, dt)
        truth = float(
            exact_dispersion(
                (k,), alpha=alpha, beta=beta, amplitude=amplitude,
                potential_constant=potential_constant,
            )
        )
        lifted, branch = unwrap_to_reference(raw, truth, dt)
        principal.append(raw)
        unwrapped.append(lifted)
        branches.append(branch)
        truths.append(truth)
        residuals.append(lifted - truth)

    return DispersionCurve(
        wave_numbers=ks, principal=principal, unwrapped=unwrapped, branch=branches,
        truth=truths, residual=residuals,
        k_wrap=wrap_wavenumber(abs(alpha), dt) if alpha != 0 else float("inf"),
        alpha=float(alpha), beta=float(beta), amplitude=float(amplitude),
        potential_constant=float(potential_constant), dt=float(dt),
    )


@dataclass
class AlphaDerivative:
    wave_number: int
    alphas: list[float]
    slopes: list[float]
    estimate: float
    truth: float
    max_relative_error: float

    def as_dict(self) -> dict:
        return {
            "wave_number": self.wave_number, "alphas": self.alphas,
            "slopes": self.slopes, "estimate": self.estimate, "truth": self.truth,
            "max_relative_error": self.max_relative_error,
        }


def _strict_alpha_grid(alphas: Iterable[float], *, name: str) -> list[float]:
    """Materialize and validate the ordered alpha samples used by phase estimators."""

    values = [float(value) for value in alphas]
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} needs at least two finite, strictly increasing alphas")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(f"{name} alpha grid must be finite and strictly increasing")
    return values


@torch.no_grad()
def alpha_phase_derivative(
    model,
    domain: PeriodicDomain,
    wave_number: int,
    *,
    alphas: Iterable[float],
    beta: float,
    amplitude: float,
    potential_constant: float,
    dt: float,
    widen: bool = False,
) -> AlphaDerivative:
    """``d arg m / d alpha`` from wrapped increments.  Truth: ``-k^2 dt``, wrap-free.

    *Theorem.*  ``arg m(k, alpha) = -(alpha k^2 - beta A^2 + V0) dt``, so
    ``d arg m / d alpha = -k^2 dt`` with ``beta`` and ``V0`` dropping out entirely.
    Consecutive increments are read as ``arg(m_{j+1} conj(m_j))``, which is
    automatically wrapped to ``(-pi, pi]`` -- and therefore *equals* the true increment
    whenever ``|d alpha| k^2 dt < pi``.  This is why identifiability survives above
    ``k_wrap`` under alpha-varying supervision even though the absolute phase does not.

    The precondition is **checked here on the caller's alpha grid**, not assumed.  The
    dataset's own alpha gap (3.47e-3 at k=32) passes with huge margin, but the probe
    chooses its own spacing and a coarse grid silently returns garbage.

    Pass **in-range alphas only**.  Extending the grid to alpha=0 turns this into a
    measurement of alpha-extrapolation; use :func:`omega_by_alpha_continuation` for
    that, and report it as a different claim.
    """

    values = _strict_alpha_grid(alphas, name="alpha_phase_derivative")
    gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    max_gap = max(abs(gap) for gap in gaps)
    if not alpha_sampling_is_dense_enough(max_gap, abs(int(wave_number)), float(dt)):
        raise ValueError(
            f"alpha grid is not dense enough to be wrap-free at k={wave_number}: "
            f"max gap {max_gap:g} * k^2 * dt = "
            f"{max_gap * wave_number**2 * float(dt):g} >= pi"
        )

    probe = widen_to_double(model, device="cpu").eval() if widen else model
    multipliers = [
        one_step_multiplier(probe, domain, wave_number, alpha=a, beta=beta,
                            amplitude=amplitude, potential_constant=potential_constant,
                            dt=dt)
        for a in values
    ]
    slopes = [
        cmath.phase(multipliers[i + 1] * multipliers[i].conjugate()) / gaps[i]
        for i in range(len(gaps))
    ]
    truth = -float(wave_number) ** 2 * float(dt)
    estimate = sum(slopes) / len(slopes)
    scale = max(abs(truth), 1e-30)
    return AlphaDerivative(
        wave_number=int(wave_number), alphas=values, slopes=slopes,
        estimate=estimate, truth=truth,
        max_relative_error=max(abs(s - truth) / scale for s in slopes),
    )


@torch.no_grad()
def omega_by_alpha_continuation(
    model,
    domain: PeriodicDomain,
    wave_number: int,
    *,
    alphas: Iterable[float],
    beta: float,
    amplitude: float,
    potential_constant: float,
    dt: float,
    widen: bool = False,
) -> float:
    """Absolute ``omega(alphas[-1])`` by accumulating wrapped increments from an anchor.

    A **different claim** from :func:`alpha_phase_derivative`, and it must be reported
    separately.  It requires an anchor alpha whose own one-step phase is unwrapped
    (``|omega(anchor)| dt <= pi``) -- in practice alpha near 0, which is far outside the
    training range.  For an FNO, ``_rescale`` maps [0.7, 1.1] to [-1, 1], so alpha=0
    enters the lift channel at about -4.5.  A failure here is therefore evidence about
    alpha-extrapolation, not about whether the alpha-derivative information was
    extracted.
    """

    values = _strict_alpha_grid(alphas, name="omega_by_alpha_continuation")
    anchor_omega = float(
        exact_dispersion((int(wave_number),), alpha=values[0], beta=beta,
                         amplitude=amplitude, potential_constant=potential_constant)
    )
    if abs(anchor_omega) * float(dt) > math.pi:
        raise ValueError(
            f"anchor alpha={values[0]:g} is itself wrapped at k={wave_number} "
            f"(|omega| dt = {abs(anchor_omega) * float(dt):g} > pi); anchor nearer 0"
        )
    gaps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    max_gap = max(abs(gap) for gap in gaps)
    if not alpha_sampling_is_dense_enough(max_gap, abs(int(wave_number)), float(dt)):
        raise ValueError(
            f"alpha grid is not dense enough to be wrap-free at k={wave_number}"
        )

    probe = widen_to_double(model, device="cpu").eval() if widen else model
    multipliers = [
        one_step_multiplier(probe, domain, wave_number, alpha=a, beta=beta,
                            amplitude=amplitude, potential_constant=potential_constant,
                            dt=dt)
        for a in values
    ]
    phase = cmath.phase(multipliers[0])
    for index in range(len(gaps)):
        phase += cmath.phase(multipliers[index + 1] * multipliers[index].conjugate())
    return -phase / float(dt)


def validate_probe(
    domain: PeriodicDomain,
    *,
    dt: float,
    alpha: float,
    beta: float,
    amplitude: float,
    potential_constant: float,
    wave_numbers: Iterable[int],
    substeps: int = 32,
    alpha_grid: Iterable[float] | None = None,
    tolerance: float = 1e-10,
) -> dict:
    """Run the probe on the reference solver; raise if it does not recover omega.

    ``raise RuntimeError``, not ``assert``: ``python -O`` strips asserts, and this gate
    is the only thing between an instrument bug and a full run of plausible-but-
    fictional omega curves.  Call it at the top of every script that probes a model.

    Checks three things, all against the solver that *defines* omega: (1) the principal
    branch matches truth below ``k_wrap``; (2) the branch index is zero below
    ``k_wrap``; (3) the alpha-derivative estimator returns ``-k^2 dt`` at the largest
    requested ``k``, i.e. above the horizon where (1) no longer holds.
    """

    from ..solvers.split_step import SubsteppedReference

    solver = SubsteppedReference(domain, substeps)
    curve = dispersion_curve(
        solver, domain, wave_numbers, alpha=alpha, beta=beta, amplitude=amplitude,
        potential_constant=potential_constant, dt=dt,
    )
    below = [
        (k, r, b)
        for k, r, b in zip(curve.wave_numbers, curve.residual, curve.branch)
        if k < curve.k_wrap
    ]
    max_below = max((abs(r) for _, r, _ in below), default=0.0)
    wrapped_below = [k for k, _, b in below if b != 0]

    highest = max(curve.wave_numbers)
    grid = list(alpha_grid) if alpha_grid is not None else [
        alpha - 0.2 + 0.05 * j for j in range(9)
    ]
    derivative = alpha_phase_derivative(
        solver, domain, highest, alphas=grid, beta=beta, amplitude=amplitude,
        potential_constant=potential_constant, dt=dt,
    )

    failures = []
    if max_below > tolerance:
        failures.append(
            f"residual below k_wrap is {max_below:.3e} (tolerance {tolerance:.1e})"
        )
    if wrapped_below:
        failures.append(f"branch is nonzero below k_wrap at k={wrapped_below}")
    if derivative.max_relative_error > tolerance:
        failures.append(
            f"alpha-derivative relative error is {derivative.max_relative_error:.3e}"
        )
    if failures:
        raise RuntimeError(
            "dispersion probe failed validation on the reference solver: "
            + "; ".join(failures)
        )

    return {
        "max_residual_below_k_wrap": max_below,
        "alpha_derivative_max_relative_error": derivative.max_relative_error,
        "alpha_derivative_estimate": derivative.estimate,
        "alpha_derivative_truth": derivative.truth,
        "k_wrap": curve.k_wrap,
        "tolerance": float(tolerance),
        "wave_numbers": curve.wave_numbers,
    }
