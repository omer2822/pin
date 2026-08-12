"""Phase 4: the structural guarantees of the learned split step.

These are the mathematical spine of the thesis, so each one is asserted on **random
untrained weights** -- that is what "architectural guarantee" means -- and each is
paired with a negative case proving the test can fail.  A conservation test that would
pass on a model with no such structure tells you nothing.

Guarantee matrix (rho = |psi|^2):

    model  local phase reads   mass  reversible   symplectic  U(1)
    C1     rho, pointwise       yes   exact        yes         yes
    C2     rho, nonlocal FNO    yes   exact        no          yes
    C3     Re psi, Im psi       yes   O(dt^2)      no          no

"exact" means machine precision *independently of dt*; C3's violation is a clean
second-order power law, which at small dt is small enough that an absolute threshold
would mistake it for exactness.
"""

from __future__ import annotations

import math

import pytest
import torch

from spno.domain import PeriodicDomain, l2_mass
from spno.models.fno import FNOStepOperator
from spno.models.projected import mass_drift
from spno.models.split_learned import (
    DensityPhaseSplitStep,
    ExactSplitStep,
    FieldDensityPhaseSplitStep,
    FullFieldPhaseSplitStep,
    KineticPhase,
    LocalPhaseLadder,
)
from spno.solvers.split_step import SplitStepNLSOperator, relative_l2

DT = 0.01
N = 64


def _domain() -> PeriodicDomain:
    return PeriodicDomain.periodic_1d(N)


def _inputs(domain, batch=4, seed=0):
    """float64 inputs: these tests measure architecture, not float32 arithmetic."""

    generator = torch.Generator().manual_seed(seed)
    field = torch.complex(
        torch.randn(batch, *domain.shape, generator=generator, dtype=torch.float64),
        torch.randn(batch, *domain.shape, generator=generator, dtype=torch.float64),
    )
    potential = torch.randn(
        batch, *domain.shape, generator=generator, dtype=torch.float64
    )
    alpha = torch.rand(batch, generator=generator, dtype=torch.float64) * 0.4 + 0.7
    beta = torch.rand(batch, generator=generator, dtype=torch.float64) - 0.4
    return field, potential, alpha, beta


def _build(kind, domain, seed=0, **kwargs):
    torch.manual_seed(seed)
    return _widen(kind(domain, trained_dt=DT, **kwargs))


def _widen(model):
    from spno.precision import widen_to_double

    return widen_to_double(model)


STRUCTURED = (DensityPhaseSplitStep, FieldDensityPhaseSplitStep)
ALL_SPLIT = STRUCTURED + (FullFieldPhaseSplitStep,)


# --------------------------------------------------------------------------------
# Correctness fixture: the architecture is wired right
# --------------------------------------------------------------------------------


def test_exact_rates_reproduce_the_reference_solver():
    """Guards the sign convention, half-step placement, and normalization at once."""

    domain = _domain()
    field, potential, alpha, beta = _inputs(domain)

    learned = ExactSplitStep(domain, trained_dt=DT)(field, potential, alpha, beta, DT)
    reference = SplitStepNLSOperator(domain)(field, potential, alpha, beta, DT)

    assert float(relative_l2(learned, reference, domain).max()) < 1e-13


def test_k2_kinetic_rung_starts_at_the_exact_rate():
    """K2 is kappa = -alpha k^2 (1 + MLP); the correction is initialized to zero."""

    domain = _domain()
    torch.manual_seed(0)
    kinetic = KineticPhase(domain, mode="K2").to(torch.float64)
    parameters = torch.tensor([[0.9, 0.3]], dtype=torch.float64)

    rate = kinetic(parameters)
    expected = -0.9 * domain.wave_number_squared().reshape(1, -1)

    assert torch.allclose(rate, expected, atol=1e-12)


@pytest.mark.parametrize("mode", ["K0", "K1", "K2"])
def test_kinetic_rate_is_real_and_grid_shaped(mode):
    domain = _domain()
    torch.manual_seed(0)
    kinetic = KineticPhase(domain, mode=mode).to(torch.float64)

    rate = kinetic(torch.tensor([[0.9, 0.3], [1.1, -0.2]], dtype=torch.float64))

    assert rate.shape == (2, N)
    assert not rate.is_complex()


# --------------------------------------------------------------------------------
# (1) Mass -- exact for every model in the family, including the C3 control
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ALL_SPLIT)
def test_mass_is_exact_at_random_untrained_weights(kind):
    domain = _domain()
    model = _build(kind, domain)
    field, potential, alpha, beta = _inputs(domain)

    with torch.no_grad():
        output = model(field, potential, alpha, beta, DT)

    assert float(mass_drift(output, field, domain).max()) < 1e-13


@pytest.mark.parametrize("kind", ALL_SPLIT)
def test_mass_is_exact_over_a_long_rollout(kind):
    domain = _domain()
    model = _build(kind, domain)
    field, potential, alpha, beta = _inputs(domain)

    state = field
    with torch.no_grad():
        for _ in range(200):
            state = model(state, potential, alpha, beta, DT)

    assert torch.isfinite(state).all()
    assert float(mass_drift(state, field, domain).max()) < 1e-12


def test_an_fno_does_not_preserve_mass():
    """The paired negative: the mass tests above are not vacuous."""

    domain = _domain()
    torch.manual_seed(0)
    model = _widen(FNOStepOperator(domain, modes=8, width=16, n_layers=2, trained_dt=DT))
    field, potential, alpha, beta = _inputs(domain)

    with torch.no_grad():
        output = model(field, potential, alpha, beta, DT)

    assert float(mass_drift(output, field, domain).max()) > 1e-3


# --------------------------------------------------------------------------------
# (2) Reversibility -- C1 and C2 only
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", STRUCTURED)
def test_structured_models_are_exactly_reversible(kind):
    """rho is unchanged by a modulus-one multiplier, so the backward step cancels it."""

    domain = _domain()
    model = _build(kind, domain)
    field, potential, alpha, beta = _inputs(domain)

    with torch.no_grad():
        state = field
        for _ in range(20):
            state = model(state, potential, alpha, beta, DT)
        for _ in range(20):
            state = model(state, potential, alpha, beta, -DT)

    assert float(relative_l2(state, field, domain).max()) < 1e-12


def _reversibility_error(model, field, potential, alpha, beta, dt, domain) -> float:
    with torch.no_grad():
        forward = model(field, potential, alpha, beta, dt)
        recovered = model(forward, potential, alpha, beta, -dt)
    return float(relative_l2(recovered, field, domain).max())


def test_full_field_phase_degrades_reversibility_from_exact_to_second_order():
    """The paired negative -- and the precise version of the claim.

    C3 differs from C2 *only* in reading Re/Im psi instead of rho.  It is not simply
    "not reversible": measured here, its violation scales as **dt^2** (order 2.00 at
    every refinement), because the local step rotates psi by ``dt*nu`` and the backward
    step therefore reads a slightly different input.  C1 and C2, whose phase depends on
    rho alone, sit at machine precision *independently of dt* -- that is what exact
    means.

    So the honest statement for the thesis is "rho-only parameter sharing makes
    reversibility exact rather than second-order", not "C3 is irreversible".  At small
    dt and untrained weights C3's violation is ~1e-9, which a naive absolute threshold
    would have mistaken for exactness.
    """

    domain = _domain()
    field, potential, alpha, beta = _inputs(domain)
    control = _build(FullFieldPhaseSplitStep, domain)
    structured = _build(DensityPhaseSplitStep, domain)

    dts = (0.01, 0.02, 0.04)
    control_errors, structured_errors = [], []
    for dt in dts:
        control.trained_dt = structured.trained_dt = dt
        control_errors.append(
            _reversibility_error(control, field, potential, alpha, beta, dt, domain)
        )
        structured_errors.append(
            _reversibility_error(structured, field, potential, alpha, beta, dt, domain)
        )

    orders = [
        math.log2(control_errors[i + 1] / control_errors[i])
        for i in range(len(dts) - 1)
    ]
    assert all(order == pytest.approx(2.0, abs=0.1) for order in orders), f"{orders=}"

    # Exact: flat in dt, at machine precision.
    assert max(structured_errors) < 1e-13
    assert max(structured_errors) / min(structured_errors) < 2

    # And the gap is enormous, so the two regimes are never confusable.
    assert min(control_errors) > 1e4 * max(structured_errors)

    # Mass survives regardless -- it is the one guarantee C3 keeps.
    with torch.no_grad():
        output = control(field, potential, alpha, beta, dts[-1])
    assert float(mass_drift(output, field, domain).max()) < 1e-13


def test_an_fno_may_not_be_probed_at_negative_dt():
    """An FNO ignores dt, so a "reversibility" number measured on it is an artefact."""

    domain = _domain()
    torch.manual_seed(0)
    model = FNOStepOperator(domain, modes=8, width=16, n_layers=2, trained_dt=DT)
    field, potential, alpha, beta = _inputs(domain)

    assert model.supports_time_reversal is False
    with pytest.raises(ValueError, match="does not support dt transfer"):
        model(field.to(torch.complex64), potential.float(), alpha.float(), beta.float(), -DT)


# --------------------------------------------------------------------------------
# (4) U(1) equivariance -- C1 and C2 only
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", STRUCTURED)
def test_structured_models_are_u1_equivariant(kind):
    """Phi(e^{ic} psi) = e^{ic} Phi(psi): rho and |psi_hat| are phase-blind."""

    domain = _domain()
    model = _build(kind, domain)
    field, potential, alpha, beta = _inputs(domain)
    rotation = torch.exp(torch.tensor(1j * 0.7, dtype=torch.complex128))

    with torch.no_grad():
        rotated_then_stepped = model(rotation * field, potential, alpha, beta, DT)
        stepped_then_rotated = rotation * model(field, potential, alpha, beta, DT)

    assert float(relative_l2(rotated_then_stepped, stepped_then_rotated, domain).max()) < 1e-13


def test_full_field_phase_breaks_u1_equivariance():
    domain = _domain()
    model = _build(FullFieldPhaseSplitStep, domain)
    field, potential, alpha, beta = _inputs(domain)
    rotation = torch.exp(torch.tensor(1j * 0.7, dtype=torch.complex128))

    with torch.no_grad():
        rotated_then_stepped = model(rotation * field, potential, alpha, beta, DT)
        stepped_then_rotated = rotation * model(field, potential, alpha, beta, DT)

    assert float(relative_l2(rotated_then_stepped, stepped_then_rotated, domain).min()) > 1e-6


def test_the_reference_solver_is_also_u1_equivariant():
    """Sanity check on the probe itself, against a map known to have the property."""

    domain = _domain()
    field, potential, alpha, beta = _inputs(domain)
    solver = SplitStepNLSOperator(domain)
    rotation = torch.exp(torch.tensor(1j * 0.7, dtype=torch.complex128))

    a = solver(rotation * field, potential, alpha, beta, DT)
    b = rotation * solver(field, potential, alpha, beta, DT)

    assert float(relative_l2(a, b, domain).max()) < 1e-13


# --------------------------------------------------------------------------------
# Contracts and capacity
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ALL_SPLIT)
def test_gradients_reach_every_parameter(kind):
    domain = _domain()
    model = _build(kind, domain)
    field, potential, alpha, beta = _inputs(domain)

    model(field, potential, alpha, beta, DT).abs().sum().backward()

    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_c1_is_tiny_and_c2_matches_the_fno_capacity():
    """Parameter counts must accompany any claim about which model is better."""

    domain = _domain()
    c1 = DensityPhaseSplitStep(domain, trained_dt=DT)
    c2 = FieldDensityPhaseSplitStep(domain, trained_dt=DT)
    fno = FNOStepOperator(domain, trained_dt=DT)

    assert c1.parameter_count() < 5_000
    assert 0.8 < c2.parameter_count() / fno.parameter_count() < 1.2


def test_a_complex_local_phase_is_rejected():
    """The real-valued phase is what makes the multiplier modulus one."""

    domain = _domain()

    class ComplexPhase(DensityPhaseSplitStep):
        def local_phase(self, field, potential, alpha, beta):
            return super().local_phase(field, potential, alpha, beta) + 0j

    model = _widen(ComplexPhase(domain, trained_dt=DT))
    field, potential, alpha, beta = _inputs(domain)

    with pytest.raises(ValueError, match="must be real"):
        model(field, potential, alpha, beta, DT)


@pytest.mark.parametrize("mode", ["K0", "K1", "K2"])
def test_every_kinetic_rung_preserves_the_structural_guarantees(mode):
    """The ladder changes expressivity, never the invariants."""

    domain = _domain()
    model = _build(DensityPhaseSplitStep, domain, kinetic_mode=mode)
    field, potential, alpha, beta = _inputs(domain)

    with torch.no_grad():
        output = model(field, potential, alpha, beta, DT)
        recovered = model(output, potential, alpha, beta, -DT)

    assert float(mass_drift(output, field, domain).max()) < 1e-13
    assert float(relative_l2(recovered, field, domain).max()) < 1e-13


# --------------------------------------------------------------------------------
# The local-phase ladder (mirror of the kinetic ladder)
# --------------------------------------------------------------------------------


def test_l2_local_rung_starts_at_the_exact_rate():
    """L2 is nu = beta*rho - V + MLP with the correction zero-initialized."""

    torch.manual_seed(0)
    ladder = LocalPhaseLadder(mode="L2").to(torch.float64)
    density = torch.rand(3, 16, dtype=torch.float64)
    potential = torch.randn(3, 16, dtype=torch.float64)
    alpha = torch.rand(3, dtype=torch.float64) + 0.7
    beta = torch.rand(3, dtype=torch.float64) - 0.4

    rate = ladder(density, potential, alpha, beta)
    expected = beta.reshape(-1, 1) * density - potential

    assert torch.allclose(rate, expected, atol=1e-14)


@pytest.mark.parametrize("mode", ["L0", "L1", "L2"])
def test_every_local_rung_is_real_and_keeps_the_guarantees(mode):
    domain = _domain()
    model = _build(DensityPhaseSplitStep, domain, local_mode=mode)
    field, potential, alpha, beta = _inputs(domain)

    with torch.no_grad():
        output = model(field, potential, alpha, beta, DT)
        recovered = model(output, potential, alpha, beta, -DT)

    assert output.is_complex()
    assert float(mass_drift(output, field, domain).max()) < 1e-13
    assert float(relative_l2(recovered, field, domain).max()) < 1e-13


def test_c2_rejects_the_pointwise_only_rung():
    """L1 supplies a pointwise product feature; C2's phase is an FNO over channels."""

    domain = _domain()
    with pytest.raises(ValueError, match="pointwise-feature rung"):
        FieldDensityPhaseSplitStep(domain, local_mode="L1", trained_dt=DT)


def test_exact_split_step_keeps_a_sibling_compatible_state_dict():
    """It must skip constructing the kinetic net, not construct and delete it.

    Deleting would drop the submodule from ``_modules``, leaving a state_dict that no
    longer round-trips against the rest of the family.
    """

    domain = _domain()
    exact = ExactSplitStep(domain, trained_dt=DT)

    assert exact.kinetic is None
    assert not any(k.startswith("kinetic.") for k in exact.state_dict())
    # And it still widens cleanly, which is what invariant evaluation needs.
    field, potential, alpha, beta = _inputs(domain)
    with torch.no_grad():
        _widen(exact)(field, potential, alpha, beta, DT)


# --------------------------------------------------------------------------------
# (8) Device transferability -- the training device is part of the architecture
# --------------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an MPS device"
)
@pytest.mark.parametrize("kind", ALL_SPLIT)
def test_the_split_family_moves_to_mps(kind):
    """MPS has no float64, so any float64 buffer bars the model from the GPU entirely.

    This is not a performance nicety.  ``KineticPhase`` registered ``k_squared`` at
    float64 (``wave_number_squared`` defaults to it) on an otherwise-float32 module, so
    ``model.to("mps")`` raised ``TypeError`` and the whole C family could only be
    trained on CPU -- roughly 6x slower.  The buffer now follows the model's dtype, and
    ``widen_to_double`` still widens it for float64 evaluation.
    """

    domain = _domain()
    torch.manual_seed(0)
    model = kind(domain, trained_dt=DT)  # not widened: this is the training path
    model.to("mps")
    assert {str(b.dtype) for b in model.buffers()} <= {"torch.float32"}


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an MPS device"
)
def test_the_fno_also_moves_to_mps():
    """The paired positive: the test above is not passing because MPS accepts anything.

    Model A always moved to MPS -- it has no float64 buffers -- so it establishes that
    the assertion has teeth and that the C family was the outlier, not the rule.
    """

    domain = _domain()
    torch.manual_seed(0)
    FNOStepOperator(domain, modes=8, width=16, n_layers=2, trained_dt=DT).to("mps")


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an MPS device"
)
def test_a_float64_buffer_is_what_breaks_the_move():
    """The paired negative: re-registering the buffer at float64 restores the failure.

    Without this, a later change that reintroduces a float64 buffer would make the
    tests above pass vacuously only until someone looked.
    """

    domain = _domain()
    torch.manual_seed(0)
    model = DensityPhaseSplitStep(domain, trained_dt=DT)
    model.kinetic.register_buffer("k_squared", model.kinetic.k_squared.double())
    with pytest.raises(TypeError, match="float64"):
        model.to("mps")


def test_widening_recovers_float64_and_lands_on_exact_integers():
    """Storing k^2 at float32 costs nothing in float64 evaluation -- it gains.

    ``2*pi*fftfreq(n, d=2*pi/n)`` does not produce exact integers: measured at N=64 the
    float64 construction is off by up to 2.3e-13 absolute (3.4e-16 relative), and
    narrowing that to float32 lands on exactly the integers.  So the widened buffer is
    if anything closer to the truth than the original.

    The order matters, and getting it wrong is silent.  Evaluating the *same expression*
    natively in float32 -- rather than in float64 and narrowing -- misses the integers
    by 1.2e-4 in k^2, which survives widening and would sit nine orders of magnitude
    above the 1e-13 bounds the tests above assert.  The exact-integer check below is
    what separates the two constructions; a dtype check alone passes for both.
    """

    domain = _domain()
    torch.manual_seed(0)
    widened = _widen(DensityPhaseSplitStep(domain, trained_dt=DT))

    k_squared = widened.kinetic.k_squared
    assert k_squared.dtype == torch.float64
    assert torch.equal(k_squared, torch.round(k_squared))
    assert float(k_squared.max()) == (N // 2) ** 2

    # The paired negative: the native-float32 construction fails this check, so the
    # assertion above is discriminating rather than decorative.
    native = domain.wave_number_squared(dtype=torch.float32).double()
    assert not torch.equal(native, torch.round(native))
    assert float((native - torch.round(native)).abs().max()) > 1e-5
