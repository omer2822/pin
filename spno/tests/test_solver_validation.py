"""Phase 0: the reference solver must earn its status as ground truth.

These tests establish the properties the whole thesis leans on.  Each one states
which category of claim it verifies -- exact structural guarantee, convergence order,
or information-theoretic identifiability -- because the thesis must never present one
as another.

All of Phase 0 runs in float64: MPS has no float64, and the 1e-12 assertions here are
meaningless in float32.
"""

from __future__ import annotations

import math

import pytest
import torch

from spno.domain import PeriodicDomain, l2_mass, project_to_mass
from spno.equations.nls import (
    alpha_sampling_is_dense_enough,
    exact_dispersion,
    nls_hamiltonian,
    plane_wave,
    wrap_wavenumber,
)
from spno.solvers.split_step import (
    SplitStepNLSOperator,
    SubsteppedReference,
    fitted_splitting_floor,
    integrate,
    relative_l2,
    splitting_floor,
)

ALPHA = 0.9
BETA = 0.4
DT = 0.01


def _problem(n: int = 64, batch: int = 4, seed: int = 0):
    """A batched, non-constant-potential, varying-parameter test problem."""

    domain = PeriodicDomain.periodic_1d(n)
    generator = torch.Generator().manual_seed(seed)
    (x,) = domain.mesh()
    field = torch.complex(
        torch.randn(batch, n, generator=generator, dtype=torch.float64),
        torch.randn(batch, n, generator=generator, dtype=torch.float64),
    )
    # Smooth the field so it is resolved on the grid rather than white noise.
    spectrum = torch.exp(-0.5 * domain.wave_number_squared() ** 2 / 6.0**2)
    field = torch.fft.ifft(torch.fft.fft(field) * spectrum)
    potential = (0.3 * torch.cos(x) - 0.2 * torch.sin(2 * x)).expand(batch, n).clone()
    alpha = torch.linspace(0.7, 1.1, batch, dtype=torch.float64)
    beta = torch.linspace(-0.4, 0.6, batch, dtype=torch.float64)
    return domain, field, potential, alpha, beta


# --------------------------------------------------------------------------------
# Exact structural guarantees of the reference scheme
# --------------------------------------------------------------------------------


def test_mass_is_conserved_to_machine_precision_over_many_steps():
    """Structural guarantee: modulus-one multipliers + Parseval."""

    domain, field, potential, alpha, beta = _problem()
    operator = SplitStepNLSOperator(domain)
    initial_mass = l2_mass(field, domain)

    evolved = field
    for _ in range(1000):
        evolved = operator(evolved, potential, alpha, beta, DT)

    drift = torch.abs(l2_mass(evolved, domain) / initial_mass - 1)
    assert float(drift.max()) < 1e-12


def test_scheme_is_exactly_reversible():
    """Structural guarantee: the local phase reads |psi|^2, which it cannot change."""

    domain, field, potential, alpha, beta = _problem()
    operator = SplitStepNLSOperator(domain)

    evolved = field
    for _ in range(50):
        evolved = operator(evolved, potential, alpha, beta, DT)
    for _ in range(50):
        evolved = operator(evolved, potential, alpha, beta, -DT)

    assert float(relative_l2(evolved, field, domain).max()) < 1e-12


def test_substepped_reference_is_also_exactly_mass_preserving():
    domain, field, potential, alpha, beta = _problem()
    reference = SubsteppedReference(domain, substeps=32)

    evolved = reference(field, potential, alpha, beta, DT)

    drift = torch.abs(l2_mass(evolved, domain) / l2_mass(field, domain) - 1)
    assert float(drift.max()) < 1e-13


def test_substeps_must_be_a_positive_integer():
    domain = PeriodicDomain.periodic_1d(16)
    with pytest.raises(ValueError):
        SubsteppedReference(domain, substeps=0)


# --------------------------------------------------------------------------------
# Convergence order and the splitting floor
# --------------------------------------------------------------------------------


def test_strang_splitting_is_second_order_in_time():
    """Numerical observation: halving dt must quarter the error.

    Measured only in the asymptotic regime.  Phase 0 found that ``dt >= 0.02`` on this
    problem is pre-asymptotic -- every sample shows a non-monotone order sequence
    (~1.3, ~4.3, 2.22, 2.05) as higher-order terms and a sign change in the leading
    error contaminate the coarse end.  The production ``dt = 0.01`` sits at the edge
    of that regime, which is worth knowing but does not affect the reference: targets
    are generated with 32 substeps at ``dt/32``, deep inside asymptotic territory.
    """

    domain, field, potential, alpha, beta = _problem(batch=2)
    horizon = 0.16
    truth = SubsteppedReference(domain, substeps=1024)(
        field, potential, alpha, beta, horizon
    )

    errors = []
    for steps in (16, 32, 64):  # dt = 0.01, 0.005, 0.0025
        evolved = SubsteppedReference(domain, substeps=steps)(
            field, potential, alpha, beta, horizon
        )
        errors.append(float(relative_l2(evolved, truth, domain).mean()))

    orders = [math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]

    assert all(order == pytest.approx(2.0, abs=0.08) for order in orders), f"{orders=}"


def test_reference_is_converged_at_32_substeps():
    """M=32 vs M=64 must be far below the splitting floor it is used to measure."""

    domain, field, potential, alpha, beta = _problem()
    coarse = SubsteppedReference(domain, 32)(field, potential, alpha, beta, DT)
    fine = SubsteppedReference(domain, 64)(field, potential, alpha, beta, DT)

    convergence_gap = float(relative_l2(coarse, fine, domain).max())
    floor = float(
        splitting_floor(domain, field, potential, alpha, beta, DT, substeps=32).max()
    )
    assert convergence_gap < 0.02 * floor, f"gap={convergence_gap:.3e} floor={floor:.3e}"


@pytest.mark.parametrize("per_mode", [False, True])
def test_learned_rates_cannot_beat_the_exact_rates(per_mode):
    """Measured over two nested families: eps_split really is the bound for Model C.

    A learned split step is free to pick *effective* generators, so in principle it
    could absorb part of the O(dt^2) Strang commutator and land below the exact-rate
    error.  Fitting (a) one scalar per generator and (b) a free per-wavenumber kinetic
    correction -- the K2 family Phase 4 gives C1 -- buys almost nothing: the optimum
    sits within 1e-4 of the exact rates, and the error falls by at most a few percent.
    The gain is distribution-dependent (~5% on this smooth field, ~0% on the wider
    production band), which is exactly what one expects if the commutator is very
    nearly orthogonal to the span of the generators.

    This is evidence over these families, not a proof over all of them.  What it
    licenses is the Phase 4 criterion: C1 should land *at* eps_split, give or take a
    few percent -- not an order of magnitude below it.  A C1 well under eps_split is a
    signal to look for leakage rather than a breakthrough.
    """

    domain, field, potential, alpha, beta = _problem(batch=32)
    exact = float(
        splitting_floor(domain, field, potential, alpha, beta, DT, substeps=32).mean()
    )

    achieved, scales = fitted_splitting_floor(
        domain, field, potential, alpha, beta, DT, substeps=32, per_mode=per_mode
    )

    # Never worse than the exact rates (they are inside the family), and never better
    # by more than a small constant factor.
    assert 0.85 * exact <= float(achieved.mean()) <= 1.001 * exact
    assert torch.allclose(scales, torch.ones_like(scales), atol=1e-3)


def test_splitting_floor_is_strictly_positive():
    """If this were zero, the learned split-step class would contain the truth."""

    domain, field, potential, alpha, beta = _problem()

    floor = splitting_floor(domain, field, potential, alpha, beta, DT, substeps=32)

    assert float(floor.min()) > 0.0
    assert torch.isfinite(floor).all()


# --------------------------------------------------------------------------------
# Exact solutions: the dispersion relation
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("wave_number", [1, 3, 7])
def test_plane_wave_advances_at_the_exact_dispersion_frequency(wave_number):
    """Verifies the three sign conventions agree: kinetic, local, and omega."""

    domain = PeriodicDomain.periodic_1d(64)
    amplitude, potential_constant = 1.1, 0.2
    initial = plane_wave(domain, (wave_number,), amplitude=amplitude).unsqueeze(0)
    potential = torch.full((1, *domain.shape), potential_constant, dtype=torch.float64)

    evolved = SplitStepNLSOperator(domain)(
        initial, potential, ALPHA, BETA, DT
    )
    expected = plane_wave(
        domain,
        (wave_number,),
        amplitude=amplitude,
        time=DT,
        alpha=ALPHA,
        beta=BETA,
        potential_constant=potential_constant,
    )

    assert torch.allclose(evolved[0], expected, atol=1e-10)


def test_hamiltonian_matches_the_analytic_plane_wave_energy():
    domain = PeriodicDomain.periodic_1d(32)
    amplitude, potential_constant, wave_number = 1.1, 0.2, 3
    wave = plane_wave(domain, (wave_number,), amplitude=amplitude).unsqueeze(0)
    potential = torch.full((1, *domain.shape), potential_constant, dtype=torch.float64)

    actual = nls_hamiltonian(
        wave, potential, domain, ALPHA, BETA
    )
    expected_density = (
        ALPHA * wave_number**2 * amplitude**2
        + potential_constant * amplitude**2
        - 0.5 * BETA * amplitude**4
    )

    assert actual.item() == pytest.approx(2 * math.pi * expected_density, abs=1e-9)


def test_energy_drift_is_bounded_not_secular():
    """Numerical observation predicted by symplecticity: O(dt^2), no linear growth."""

    domain, field, potential, alpha, beta = _problem()
    operator = SplitStepNLSOperator(domain)
    initial_energy = nls_hamiltonian(field, potential, domain, alpha, beta)

    drifts = []
    evolved = field
    for _ in range(20):
        for _ in range(50):
            evolved = operator(evolved, potential, alpha, beta, DT)
        energy = nls_hamiltonian(evolved, potential, domain, alpha, beta)
        drifts.append(float(torch.abs(energy / initial_energy - 1).max()))

    # Bounded: the late-window drift must not exceed the early-window drift by much.
    assert max(drifts[10:]) < 5 * max(drifts[:10]) + 1e-12
    assert max(drifts) < 1e-3


# --------------------------------------------------------------------------------
# Identifiability: the theorem the spectral-generalization chapter rests on
# --------------------------------------------------------------------------------


def _one_step_phase(domain, alpha, wave_number, dt=DT, amplitude=1.0):
    """arg of the one-step Fourier multiplier at ``wave_number``, wrapped to (-pi, pi]."""

    initial = plane_wave(domain, (wave_number,), amplitude=amplitude).unsqueeze(0)
    potential = torch.zeros(1, *domain.shape, dtype=torch.float64)
    evolved = SplitStepNLSOperator(domain)(
        initial, potential, alpha, 0.0, dt
    )
    before = torch.fft.fft(initial[0])[wave_number]
    after = torch.fft.fft(evolved[0])[wave_number]
    return float(torch.angle(after / before))


def test_below_the_wrap_horizon_omega_is_directly_recoverable():
    domain = PeriodicDomain.periodic_1d(64)
    wave_number = 8
    assert wave_number < wrap_wavenumber(ALPHA, DT)

    omega = -_one_step_phase(domain, ALPHA, wave_number) / DT

    assert omega == pytest.approx(exact_dispersion((wave_number,), alpha=ALPHA), abs=1e-8)


def test_above_the_wrap_horizon_fixed_alpha_data_is_genuinely_ambiguous():
    """Proof, verified numerically: distinct alphas produce an identical one-step map.

    This is the control arm G5a -- the regime where *no* architecture can recover
    omega, because the supervision does not contain it.
    """

    domain = PeriodicDomain.periodic_1d(64)
    wave_number = 30
    assert wave_number > wrap_wavenumber(ALPHA, DT)

    aliases = [
        (ALPHA * wave_number**2 * DT - 2 * math.pi * n) / (wave_number**2 * DT)
        for n in range(3)
    ]
    phases = [_one_step_phase(domain, a, wave_number) for a in aliases]

    assert len({round(a, 6) for a in aliases}) == 3, "aliases must be distinct"
    for phase in phases[1:]:
        assert phase == pytest.approx(phases[0], abs=1e-12)


def test_the_alpha_derivative_is_wrap_free_above_the_horizon():
    """Proof, verified numerically: d(arg m)/d(alpha) = -k^2 dt, no unwrapping needed.

    This is why G5b is an *architecture* question and not an information-theoretic
    one: across an alpha-family the dispersion relation is identifiable at every k.
    """

    domain = PeriodicDomain.periodic_1d(64)
    wave_number, alpha_gap = 30, 0.01
    assert wave_number > wrap_wavenumber(ALPHA, DT)
    assert alpha_sampling_is_dense_enough(alpha_gap, wave_number, DT)

    difference = _one_step_phase(domain, ALPHA + alpha_gap, wave_number) - _one_step_phase(
        domain, ALPHA, wave_number
    )

    assert difference == pytest.approx(-(wave_number**2) * DT * alpha_gap, abs=1e-9)


def test_alpha_sampling_density_precondition_rejects_sparse_grids():
    # A coarse alpha grid wraps at high k and would corrupt the derivative estimate.
    assert not alpha_sampling_is_dense_enough(0.4, 32, DT)
    assert alpha_sampling_is_dense_enough(0.0005, 32, DT)


# --------------------------------------------------------------------------------
# Supporting utilities
# --------------------------------------------------------------------------------


def test_project_to_mass_enforces_the_reference_mass_exactly():
    domain, field, potential, alpha, beta = _problem()
    perturbed = field * 1.7

    projected = project_to_mass(perturbed, field, domain)

    assert torch.allclose(l2_mass(projected, domain), l2_mass(field, domain))


def test_integrate_stores_the_initial_frame_and_requested_stride():
    domain, field, potential, alpha, beta = _problem(batch=2)
    operator = SplitStepNLSOperator(domain)

    trajectory = integrate(
        operator, field, potential, alpha, beta, DT, steps=10, store_every=5
    )

    assert trajectory.shape == (2, 3, *domain.shape)
    assert torch.allclose(trajectory[:, 0], field)


def test_batched_field_layout_is_enforced():
    domain = PeriodicDomain.periodic_1d(16)
    operator = SplitStepNLSOperator(domain)
    potential = torch.zeros(16, dtype=torch.float64)
    unbatched = torch.zeros(16, dtype=torch.complex128)

    with pytest.raises(ValueError):
        operator(unbatched, potential, 1.0, 0.0, DT)


def test_float32_parameters_are_rejected_against_a_float64_field():
    """A silent downcast of alpha costs 2.6e-8, which is 5e-7 of phase at k=30.

    Caught while writing this suite: ``torch.tensor([0.9])`` is float32, and it
    quietly broke four 1e-9 assertions.  Data generation must never absorb it.
    """

    domain = PeriodicDomain.periodic_1d(16)
    operator = SplitStepNLSOperator(domain)
    field = torch.zeros(1, 16, dtype=torch.complex128)
    potential = torch.zeros(1, 16, dtype=torch.float64)

    with pytest.raises(ValueError, match="silently lose precision"):
        operator(field, potential, torch.tensor([0.9]), 0.0, DT)

    # Python floats and correctly-typed tensors are both fine.
    operator(field, potential, 0.9, 0.0, DT)
    operator(field, potential, torch.tensor([0.9], dtype=torch.float64), 0.0, DT)


def test_real_field_and_complex_potential_are_rejected():
    domain = PeriodicDomain.periodic_1d(16)
    operator = SplitStepNLSOperator(domain)
    field = torch.zeros(1, 16, dtype=torch.complex128)

    with pytest.raises(ValueError):
        operator(field.real, torch.zeros(1, 16, dtype=torch.float64), 1.0, 0.0, DT)
    with pytest.raises(ValueError):
        operator(field, torch.zeros(1, 16, dtype=torch.complex128), 1.0, 0.0, DT)
