"""Phase 6: the plane-wave dispersion probe, validated on the solver that defines omega.

Every assertion here runs against the reference solver or against analytic truth.  A
probe validated only on models proves nothing: an instrument that agrees with a model
it cannot check is indistinguishable from one that reports the model back to itself.
"""

import math

import pytest
import torch

from spno.domain import PeriodicDomain
from spno.equations.nls import exact_dispersion, plane_wave_mass, wrap_wavenumber
from spno.evaluation.dispersion import (
    alpha_phase_derivative,
    dispersion_curve,
    mode_index,
    omega_by_alpha_continuation,
    one_step_multiplier,
    principal_frequency,
    probe_amplitude,
    unwrap_to_reference,
    validate_probe,
)
from spno.solvers.split_step import SplitStepNLSOperator, SubsteppedReference

N = 64
DT = 0.01
ALPHA = 0.9
BETA = 0.3
V0 = 0.2
MASS_RANGE = (1.0, 3.0)


def _domain():
    return PeriodicDomain.periodic_1d(N)


def test_probe_amplitude_lands_at_the_centre_of_the_training_mass_range():
    """Otherwise the probe confounds spectral extrapolation with mass extrapolation."""

    domain = _domain()
    amplitude = probe_amplitude(domain, MASS_RANGE)
    centre = 0.5 * (MASS_RANGE[0] + MASS_RANGE[1])
    assert plane_wave_mass(domain, amplitude) == pytest.approx(centre, rel=1e-12)


def test_mode_index_is_k_mod_n_and_rejects_beyond_nyquist():
    domain = _domain()
    assert mode_index(5, domain) == 5
    assert mode_index(-5, domain) == N - 5
    assert mode_index(N // 2, domain) == N // 2
    with pytest.raises(ValueError, match="beyond Nyquist"):
        mode_index(N // 2 + 1, domain)


@pytest.mark.parametrize("wave_number", [0, 1, 5, 8, 12, 16])
def test_the_probe_recovers_omega_on_the_reference_solver_below_k_wrap(wave_number):
    """The load-bearing validation: below k_wrap the principal branch *is* omega."""

    domain = _domain()
    assert wave_number < wrap_wavenumber(ALPHA, DT)
    amplitude = probe_amplitude(domain, MASS_RANGE)
    solver = SubsteppedReference(domain, 32)

    multiplier = one_step_multiplier(
        solver, domain, wave_number, alpha=ALPHA, beta=BETA,
        amplitude=amplitude, potential_constant=V0, dt=DT,
    )
    measured = principal_frequency(multiplier, DT)
    truth = exact_dispersion(
        (wave_number,), alpha=ALPHA, beta=BETA, amplitude=amplitude,
        potential_constant=V0,
    )
    assert measured == pytest.approx(truth, abs=1e-10)


def test_substepping_does_not_change_the_probe_because_the_split_step_is_exact_here():
    """A plane wave in a constant potential has eps_split = 0.

    |midpoint| is constant in x, so the nonlinear phase is exact and the Strang step
    reproduces exp(-i omega dt) exactly.  Consequence for the thesis: eps_split
    (6.239e-5, measured on random fields) is NOT the error floor for probe results.
    """

    domain = _domain()
    amplitude = probe_amplitude(domain, MASS_RANGE)
    kwargs = dict(alpha=ALPHA, beta=BETA, amplitude=amplitude,
                  potential_constant=V0, dt=DT)
    coarse = one_step_multiplier(SplitStepNLSOperator(domain), domain, 7, **kwargs)
    fine = one_step_multiplier(SubsteppedReference(domain, 32), domain, 7, **kwargs)
    assert abs(coarse - fine) < 1e-13


def test_above_k_wrap_the_principal_branch_is_wrong_and_the_lift_repairs_it():
    """G5a in miniature: the observable is omega mod 2 pi / dt, and nothing more."""

    domain = _domain()
    amplitude = probe_amplitude(domain, MASS_RANGE)
    wave_number = 30
    assert wave_number > wrap_wavenumber(ALPHA, DT)

    multiplier = one_step_multiplier(
        SubsteppedReference(domain, 32), domain, wave_number, alpha=ALPHA, beta=BETA,
        amplitude=amplitude, potential_constant=V0, dt=DT,
    )
    principal = principal_frequency(multiplier, DT)
    truth = exact_dispersion(
        (wave_number,), alpha=ALPHA, beta=BETA, amplitude=amplitude,
        potential_constant=V0,
    )

    # The principal branch is wrong by a whole number of 2 pi / dt.
    assert abs(principal - truth) > 0.5 * (2 * math.pi / DT)
    lifted, branch = unwrap_to_reference(principal, truth, DT)
    assert branch != 0
    assert lifted == pytest.approx(truth, abs=1e-10)


def test_dispersion_curve_matches_truth_below_k_wrap_and_reports_the_branch_above():
    domain = _domain()
    amplitude = probe_amplitude(domain, MASS_RANGE)
    curve = dispersion_curve(
        SubsteppedReference(domain, 32), domain, range(0, N // 2 + 1),
        alpha=ALPHA, beta=BETA, amplitude=amplitude, potential_constant=V0, dt=DT,
    )
    k_wrap = wrap_wavenumber(ALPHA, DT)
    for k, residual, branch in zip(curve.wave_numbers, curve.residual, curve.branch):
        if k < k_wrap:
            assert branch == 0, f"unexpected wrap below k_wrap at k={k}"
        assert abs(residual) < 1e-10, f"k={k}"


# --------------------------------------------------------------------------------
# (2) Identifiability: the ambiguity and its escape hatch, at the same wavenumber
# --------------------------------------------------------------------------------


def test_alias_family_gives_an_identical_map_at_fixed_alpha():
    """The theorem, in code: alphas separated by 2 pi / (k^2 dt) are indistinguishable.

    The offsets are *computed*, never hardcoded.  The spec's rounded literals
    (0.2019, -0.4963) carry ~7e-5 of residual alpha, which moves the phase at k=30 by
    ~6e-4 rad -- a failure with nothing to do with the physics.  Measured agreement
    across the family is 3.1e-15, i.e. float64 roundoff.
    """

    domain = _domain()
    amplitude = probe_amplitude(domain, MASS_RANGE)
    solver = SubsteppedReference(domain, 32)
    k = 30
    assert k > wrap_wavenumber(ALPHA, DT)

    period = 2 * math.pi / (k**2 * DT)
    alphas = [ALPHA - j * period for j in range(3)]
    # Sanity: the computed grid agrees with the spec's rounded literals.
    assert alphas[1] == pytest.approx(0.2019, abs=1e-3)
    assert alphas[2] == pytest.approx(-0.4963, abs=1e-3)

    multipliers = [
        one_step_multiplier(solver, domain, k, alpha=a, beta=BETA,
                            amplitude=amplitude, potential_constant=V0, dt=DT)
        for a in alphas
    ]
    for other in multipliers[1:]:
        assert abs(other - multipliers[0]) < 1e-12


def test_below_k_wrap_the_nearest_alias_lies_outside_any_admissible_alpha():
    """The paired negative -- and the precise version of the claim.

    Aliasing is not absent below ``k_wrap``; it is *unreachable*.  The construction
    ``alpha - 2 pi / (k^2 dt)`` produces an exactly-indistinguishable map at every
    wavenumber, k=5 included.  What ``k_wrap`` separates is whether the nearest alias
    falls inside the admissible alpha range:

        k = 30   period 0.698   nearest alias 0.202   inside [0.7, 1.1]-ish physics
        k =  5   period 25.13   nearest alias -24.23  negative, so not a diffusion
                                               coefficient at all

    So the honest statement is "above k_wrap a single alpha does not determine omega
    *within the physical range*", not "below k_wrap the map is injective in alpha".
    Getting this backwards was a real error in an earlier draft of this test, which
    asserted the k=5 family was distinguishable and measured 2.6e-15.
    """

    domain = _domain()
    amplitude = probe_amplitude(domain, MASS_RANGE)
    solver = SubsteppedReference(domain, 32)

    def nearest_alias(k):
        return ALPHA - 2 * math.pi / (k**2 * DT)

    assert nearest_alias(30) == pytest.approx(0.2019, abs=1e-3)
    assert nearest_alias(5) == pytest.approx(-24.23, abs=1e-2)

    # Above k_wrap the alias is a physically plausible alpha; below it, it is not.
    assert 0.0 < nearest_alias(30) < 1.5
    assert nearest_alias(5) < -1.0

    # And the alias really is exact at k=5 too -- the mechanism is k-independent.
    far = one_step_multiplier(
        solver, domain, 5, alpha=nearest_alias(5), beta=BETA,
        amplitude=amplitude, potential_constant=V0, dt=DT,
    )
    near = one_step_multiplier(
        solver, domain, 5, alpha=ALPHA, beta=BETA,
        amplitude=amplitude, potential_constant=V0, dt=DT,
    )
    assert abs(far - near) < 1e-12


def test_the_alpha_derivative_is_exact_at_the_same_k_where_the_map_is_ambiguous():
    """The other half of the identifiability result, at the same wavenumber.

    ``d arg m / d alpha = -k^2 dt`` is wrap-free, so the information the fixed-alpha map
    destroys is still present across an alpha family.  Measured relative error at k=30
    is 3.8e-15.  Together with the alias test above, these two *are* the G5a/G5b result.
    """

    domain = _domain()
    amplitude = probe_amplitude(domain, MASS_RANGE)
    k = 30
    alphas = [0.7 + 0.05 * j for j in range(9)]  # in-range only, gap 0.05
    result = alpha_phase_derivative(
        SubsteppedReference(domain, 32), domain, k, alphas=alphas, beta=BETA,
        amplitude=amplitude, potential_constant=V0, dt=DT,
    )
    assert result.truth == pytest.approx(-(k**2) * DT)
    assert result.estimate == pytest.approx(result.truth, rel=1e-10)
    assert result.max_relative_error < 1e-10


def test_the_alpha_derivative_is_independent_of_beta_and_the_potential():
    """The paired check: beta and V0 drop out of the derivative analytically.

    If they did not, the estimator would be measuring the nonlinearity rather than the
    dispersion relation, and G5b would not isolate what it claims to.
    """

    domain = _domain()
    amplitude = probe_amplitude(domain, MASS_RANGE)
    alphas = [0.7 + 0.05 * j for j in range(9)]
    estimates = [
        alpha_phase_derivative(
            SubsteppedReference(domain, 32), domain, 20, alphas=alphas, beta=beta,
            amplitude=amplitude, potential_constant=v0, dt=DT,
        ).estimate
        for beta, v0 in ((0.3, 0.2), (-0.4, 0.0), (0.6, 1.5))
    ]
    for estimate in estimates[1:]:
        assert estimate == pytest.approx(estimates[0], rel=1e-12)


def test_the_alpha_derivative_refuses_a_grid_too_coarse_to_be_wrap_free():
    """The precondition is checked on the *probe's* alpha grid, not the dataset's.

    The dataset's own gap (3.47e-3 at k=32) passes with enormous margin; a probe that
    picks its own spacing can silently violate it and return garbage.
    """

    domain = _domain()
    amplitude = probe_amplitude(domain, MASS_RANGE)
    with pytest.raises(ValueError, match="dense enough"):
        alpha_phase_derivative(
            SubsteppedReference(domain, 32), domain, 30, alphas=[0.7, 1.1],
            beta=BETA, amplitude=amplitude, potential_constant=V0, dt=DT,
        )


@pytest.mark.parametrize(
    "alphas",
    [[0.7, 0.7], [0.8, 0.7], [0.7, float("nan")], [0.7, float("inf")]],
)
def test_alpha_estimators_require_a_finite_strictly_increasing_grid(alphas):
    """Invalid grids must fail before they can create undefined phase slopes."""

    domain = _domain()
    kwargs = dict(
        beta=BETA, amplitude=probe_amplitude(domain, MASS_RANGE),
        potential_constant=V0, dt=DT,
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        alpha_phase_derivative(
            SubsteppedReference(domain, 32), domain, 8, alphas=alphas, **kwargs
        )


def test_alpha_continuation_reconstructs_absolute_omega_above_k_wrap():
    """A *different* claim from the derivative: absolute omega, not its slope.

    Requires an anchor near alpha=0, far outside the training range, so a failure here
    is evidence about alpha-extrapolation rather than about identifiability.
    """

    domain = _domain()
    amplitude = probe_amplitude(domain, MASS_RANGE)
    k = 30
    alphas = [0.05 * j for j in range(19)]  # anchor at 0, walk to 0.9
    omega = omega_by_alpha_continuation(
        SubsteppedReference(domain, 32), domain, k, alphas=alphas, beta=BETA,
        amplitude=amplitude, potential_constant=V0, dt=DT,
    )
    truth = exact_dispersion((k,), alpha=alphas[-1], beta=BETA,
                             amplitude=amplitude, potential_constant=V0)
    assert omega == pytest.approx(float(truth), abs=1e-9)


def test_alpha_continuation_refuses_an_anchor_whose_own_phase_is_already_wrapped():
    domain = _domain()
    amplitude = probe_amplitude(domain, MASS_RANGE)
    with pytest.raises(ValueError, match="anchor"):
        omega_by_alpha_continuation(
            SubsteppedReference(domain, 32), domain, 30, alphas=[0.9, 0.95],
            beta=BETA, amplitude=amplitude, potential_constant=V0, dt=DT,
        )


# --------------------------------------------------------------------------------
# (3) The run gate
# --------------------------------------------------------------------------------


def test_validate_probe_passes_on_the_reference_solver_and_reports_its_margin():
    domain = _domain()
    report = validate_probe(
        domain, dt=DT, alpha=ALPHA, beta=BETA,
        amplitude=probe_amplitude(domain, MASS_RANGE), potential_constant=V0,
        wave_numbers=range(0, N // 2 + 1),
    )
    assert report["max_residual_below_k_wrap"] < 1e-10
    assert report["alpha_derivative_max_relative_error"] < 1e-10
    assert report["k_wrap"] == pytest.approx(wrap_wavenumber(ALPHA, DT))


def test_validate_probe_raises_rather_than_returning_a_bad_number():
    """The paired negative: an impossible tolerance must fail loudly, not silently.

    ``raise``, not ``assert`` -- ``python -O`` strips asserts, and this gate is the only
    thing between an instrument bug and a run of plausible-but-fictional omega curves.
    """

    domain = _domain()
    with pytest.raises(RuntimeError, match="dispersion probe failed"):
        validate_probe(
            domain, dt=DT, alpha=ALPHA, beta=BETA,
            amplitude=probe_amplitude(domain, MASS_RANGE), potential_constant=V0,
            wave_numbers=range(0, 9), tolerance=1e-18,
        )
