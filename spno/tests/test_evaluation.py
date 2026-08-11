"""Phase 5: the evaluation suite must be correct before it judges anything.

Each evaluator is checked against a case whose answer is known independently -- the
reference solver, an analytic plane wave, or a synthetic drift series -- because an
evaluator that is quietly wrong produces a plausible table that is entirely fictional.
"""

from __future__ import annotations

import math

import pytest
import torch

from spno.config import DataConfig
from spno.data.datasets import RolloutBatches, generate_shard
from spno.domain import PeriodicDomain
from spno.equations.nls import plane_wave
from spno.evaluation.conservation import classify_drift, evaluate_conservation
from spno.evaluation.reversibility import (
    evaluate_reversibility,
    reversibility_error,
    reversibility_order,
)
from spno.evaluation.rollout import evaluate_rollout
from spno.evaluation.spectral import (
    amplitude_and_phase_split,
    banded_error,
    evaluate_spectral,
    mean_phase_error,
    mode_error_spectrum,
    phase_error,
)
from spno.models.fno import FNOStepOperator
from spno.models.split_learned import DensityPhaseSplitStep, FullFieldPhaseSplitStep
from spno.precision import widen_to_double
from spno.solvers.split_step import SplitStepNLSOperator, SubsteppedReference

SMALL = DataConfig(n_train=6, n_val=4, n_test=4, steps=40)
DT = SMALL.dt


@pytest.fixture(scope="module")
def shard():
    return generate_shard(SMALL, "test")


def _pieces(shard):
    return (
        shard.trajectories[:, 0],
        shard.trajectories,
        shard.potential,
        shard.alpha,
        shard.beta,
    )


# --------------------------------------------------------------------------------
# Rollout: the evaluator must score the ground truth as ~zero
# --------------------------------------------------------------------------------


def test_rollout_of_the_generating_solver_is_near_zero(shard):
    """The strongest available check: the data's own generator must score ~0."""

    domain = SMALL.domain
    initial, trajectories, potential, alpha, beta = _pieces(shard)
    solver = SubsteppedReference(domain, SMALL.substeps)

    metrics = evaluate_rollout(
        solver, domain, initial, trajectories, potential, alpha, beta, DT,
        checkpoints=(1, 10, 20, 40),
    )

    assert max(metrics.relative_error) < 1e-12
    # 40 model steps x 32 substeps = 1280 float64 split steps; roundoff accumulates
    # roughly linearly, so this is arithmetic rather than a structural violation.
    assert max(metrics.mass_drift) < 1e-12
    assert metrics.diverged_at is None


def test_rollout_of_a_coarser_solver_scores_near_the_splitting_floor(shard):
    """A single Strang step per model step should land at ~eps_split after one step."""

    domain = SMALL.domain
    initial, trajectories, potential, alpha, beta = _pieces(shard)

    metrics = evaluate_rollout(
        SplitStepNLSOperator(domain), domain, initial, trajectories,
        potential, alpha, beta, DT, checkpoints=(1,),
    )

    assert 1e-6 < metrics.error_at(1) < 1e-3


def test_rollout_records_divergence_instead_of_averaging_it_in(shard):
    domain = SMALL.domain
    initial, trajectories, potential, alpha, beta = _pieces(shard)

    class Diverging(SplitStepNLSOperator):
        def forward(self, field, potential, alpha, beta, dt):
            return field * 1e30

    metrics = evaluate_rollout(
        Diverging(domain), domain, initial, trajectories, potential, alpha, beta, DT,
        checkpoints=(1, 10, 20),
    )

    assert metrics.diverged_at is not None


# --------------------------------------------------------------------------------
# Conservation and the drift classifier
# --------------------------------------------------------------------------------


def test_classifier_separates_flat_from_linear_growth():
    steps = [10, 20, 50, 100, 200]

    flat = classify_drift(steps, [1e-6, 1.1e-6, 0.9e-6, 1.05e-6, 1e-6])
    linear = classify_drift(steps, [1e-6 * s for s in steps])

    assert flat.classification == "bounded"
    assert abs(flat.slope) < 0.1
    assert linear.classification == "secular"
    assert linear.slope == pytest.approx(1.0, abs=0.05)


def test_classifier_behaviour_at_the_bounded_secular_boundary():
    """The cut is at slope 0.25, so pin down what happens near it.

    C1's energy drift is a model-error term that need not be perfectly flat.  If it
    lands at slope ~0.3 the classifier says "secular" for something the theory calls
    bounded, and the headline prediction would fail on a threshold choice rather than
    on physics.  Report the slope with its stderr as the primary number and treat the
    label as secondary.
    """

    steps = [10, 20, 50, 100, 200]

    just_under = classify_drift(steps, [1e-6 * s**0.20 for s in steps])
    just_over = classify_drift(steps, [1e-6 * s**0.30 for s in steps])

    assert just_under.classification == "bounded"
    assert just_under.slope == pytest.approx(0.20, abs=0.02)
    assert just_over.classification == "secular"
    assert just_over.slope == pytest.approx(0.30, abs=0.02)
    # A clean power law must have a small stderr, so the number is trustworthy even
    # when the label is borderline.
    assert just_over.slope_stderr < 0.01


def test_classifier_reports_insufficient_data_rather_than_guessing():
    assert classify_drift([10, 20], [1e-6, 2e-6]).classification == "insufficient-data"
    assert classify_drift([10, 20, 50], [0.0, 0.0, 0.0]).classification == "insufficient-data"


def test_reference_solver_conserves_mass_and_shows_bounded_energy(shard):
    """The theory says exact mass and bounded (non-secular) energy; this checks both."""

    domain = SMALL.domain
    initial, _, potential, alpha, beta = _pieces(shard)

    metrics = evaluate_conservation(
        SplitStepNLSOperator(domain), domain, initial, potential, alpha, beta, DT,
        steps=200, stride=10,
    )

    assert max(metrics.mass_drift) < 1e-13
    assert metrics.energy_trend.classification == "bounded"


# --------------------------------------------------------------------------------
# Reversibility
# --------------------------------------------------------------------------------


def test_reference_solver_is_classified_as_exactly_reversible(shard):
    domain = SMALL.domain
    initial, _, potential, alpha, beta = _pieces(shard)
    solver = SplitStepNLSOperator(domain)
    solver.supports_time_reversal = True
    solver.trained_dt = DT

    regime, order, errors = reversibility_order(
        solver, domain, initial, potential, alpha, beta
    )

    assert regime == "exact"
    assert max(errors) < 1e-13


def test_structured_model_is_exact_and_the_control_is_second_order(shard):
    """The classifier must reproduce the Phase 4 finding on untrained weights."""

    domain = SMALL.domain
    initial, _, potential, alpha, beta = _pieces(shard)
    torch.manual_seed(0)
    structured = widen_to_double(DensityPhaseSplitStep(domain, trained_dt=DT))
    torch.manual_seed(0)
    control = widen_to_double(FullFieldPhaseSplitStep(domain, trained_dt=DT))

    exact_regime, _, _ = reversibility_order(
        structured, domain, initial, potential, alpha, beta
    )
    control_regime, control_order, _ = reversibility_order(
        control, domain, initial, potential, alpha, beta
    )

    assert exact_regime == "exact"
    assert control_regime == "approximate"
    assert control_order == pytest.approx(2.0, abs=0.15)


def test_reversibility_refuses_a_model_that_ignores_dt(shard):
    """An FNO stepped at -dt re-applies the forward map; the number would be fiction."""

    domain = SMALL.domain
    initial, _, potential, alpha, beta = _pieces(shard)
    torch.manual_seed(0)
    model = widen_to_double(
        FNOStepOperator(domain, modes=8, width=16, n_layers=2, trained_dt=DT)
    )

    with pytest.raises(ValueError, match="does not support time reversal"):
        reversibility_error(model, domain, initial, potential, alpha, beta, DT, 5)


def test_evaluate_reversibility_reports_mass_alongside_the_error(shard):
    domain = SMALL.domain
    initial, _, potential, alpha, beta = _pieces(shard)
    torch.manual_seed(0)
    model = widen_to_double(DensityPhaseSplitStep(domain, trained_dt=DT))

    metrics = evaluate_reversibility(
        model, domain, initial, potential, alpha, beta, DT, steps=10
    )

    assert metrics.relative_error < 1e-12
    assert metrics.mass_drift < 1e-13
    assert metrics.regime == "exact"
    # The probe must leave trained_dt untouched for later use.
    assert model.trained_dt == DT


# --------------------------------------------------------------------------------
# Spectral diagnostics
# --------------------------------------------------------------------------------


def test_mode_error_is_zero_for_an_identical_field(shard):
    domain = SMALL.domain
    field = shard.trajectories[:, 0]

    spectrum = mode_error_spectrum(field, field, domain)

    assert float(spectrum.max()) < 1e-12


def test_mode_error_localizes_a_single_corrupted_mode():
    """The whole point of E(k): a norm-invisible error must show up at its own k."""

    domain = PeriodicDomain.periodic_1d(64)
    target = plane_wave(domain, (3,)).unsqueeze(0)
    corrupted_hat = torch.fft.fft(target)
    corrupted_hat[:, 20] += 0.05 * corrupted_hat[:, 3]
    corrupted = torch.fft.ifft(corrupted_hat)

    spectrum = mode_error_spectrum(corrupted, target, domain, relative=False)

    assert int(torch.argmax(spectrum)) == 20


def test_phase_error_detects_a_pure_phase_rotation():
    """A global phase changes no amplitude anywhere; |psi|-based metrics see nothing."""

    domain = PeriodicDomain.periodic_1d(64)
    target = plane_wave(domain, (3,)).unsqueeze(0)
    rotated = target * math.e ** (1j * 0.3)

    split = amplitude_and_phase_split(rotated, target, domain)
    curve = phase_error(rotated, target, domain)
    summary = mean_phase_error(rotated, target, domain)

    assert split["amplitude"] < 1e-12
    assert split["phase"] > 0.1
    # The scalar summary weights across modes, so it recovers the rotation exactly.
    assert summary == pytest.approx(0.3, abs=1e-6)
    # The curve is right where there is energy (k=3) and noise where there is none,
    # which is why a bare max over the curve is the wrong reduction.
    assert float(curve[3]) == pytest.approx(0.3, abs=1e-6)
    assert float(curve.max()) > 1.0


def test_amplitude_and_phase_components_account_for_the_total():
    domain = PeriodicDomain.periodic_1d(32)
    generator = torch.Generator().manual_seed(0)
    target = torch.complex(
        torch.randn(3, 32, generator=generator, dtype=torch.float64),
        torch.randn(3, 32, generator=generator, dtype=torch.float64),
    )
    prediction = target + 0.1 * torch.complex(
        torch.randn(3, 32, generator=generator, dtype=torch.float64),
        torch.randn(3, 32, generator=generator, dtype=torch.float64),
    )

    split = amplitude_and_phase_split(prediction, target, domain)

    assert split["amplitude"] ** 2 + split["phase"] ** 2 == pytest.approx(
        split["total"] ** 2, rel=0.15
    )


def test_banded_error_splits_at_the_requested_edges():
    domain = PeriodicDomain.periodic_1d(64)
    target = plane_wave(domain, (3,)).unsqueeze(0)
    corrupted_hat = torch.fft.fft(target)
    corrupted_hat[:, 25] += 0.1 * corrupted_hat[:, 3]
    corrupted = torch.fft.ifft(corrupted_hat)

    bands = banded_error(corrupted, target, domain, (8.0, 16.9))

    assert bands["0-8"] < 1e-12          # the energy-carrying band is untouched
    assert bands["16.9+"] > 0.5          # all the error is above k_wrap


def test_evaluate_spectral_returns_sorted_wavenumbers(shard):
    domain = SMALL.domain
    prediction = shard.trajectories[:, 1]
    target = shard.trajectories[:, 2]

    metrics = evaluate_spectral(prediction, target, domain)

    assert metrics.wave_numbers == sorted(metrics.wave_numbers)
    assert len(metrics.relative_mode_error) == len(metrics.wave_numbers)


# --------------------------------------------------------------------------------
# Rollout training data path
# --------------------------------------------------------------------------------


def test_rollout_batches_windows_are_consecutive_and_in_trajectory(shard):
    batches = RolloutBatches(shard, horizon=3, device="cpu", dtype=torch.complex128)
    generator = torch.Generator().manual_seed(0)

    batch = next(batches.batches(len(batches), generator, shuffle=False))

    assert batch["targets"].shape[1] == 3
    assert torch.allclose(batch["psi"][0], shard.trajectories[0, 0])
    for step in range(3):
        assert torch.allclose(batch["targets"][0, step], shard.trajectories[0, step + 1])


def test_rollout_batches_reject_an_oversized_horizon(shard):
    with pytest.raises(ValueError):
        RolloutBatches(shard, horizon=shard.n_frames, device="cpu")
