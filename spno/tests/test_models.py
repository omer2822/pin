"""Phase 2-3: model contracts, and the guarantees each model does and does not have.

The negative assertions matter as much as the positive ones.  Model B preserves mass
and *nothing else*; if a test ever showed it preserving reversibility, that would mean
the test was vacuous, not that the model was better than advertised.
"""

from __future__ import annotations

import pytest
import torch

from spno.config import DataConfig
from spno.data.datasets import OneStepBatches, generate_shard
from spno.domain import PeriodicDomain, l2_mass
from spno.losses.relative_l2 import relative_l2_loss
from spno.models.base import allow_dt_transfer
from spno.models.fno import FNOStepOperator
from spno.models.projected import MassProjectedOperator, mass_drift
from spno.models.split_learned import DensityPhaseSplitStep
from spno.precision import widen_to_double
from spno.solvers.split_step import SplitStepNLSOperator
from spno.train import TrainConfig, train_one_step

SMALL = DataConfig(n_train=8, n_val=4, n_test=4, steps=6)


@pytest.fixture(scope="module")
def shards():
    return {split: generate_shard(SMALL, split) for split in ("train", "val", "test")}


def _model(domain, **kwargs) -> FNOStepOperator:
    torch.manual_seed(0)
    return FNOStepOperator(
        domain, modes=8, width=16, n_layers=2, trained_dt=SMALL.dt, **kwargs
    )


def _inputs(domain, batch=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    field = torch.complex(
        torch.randn(batch, *domain.shape, generator=generator),
        torch.randn(batch, *domain.shape, generator=generator),
    )
    potential = torch.randn(batch, *domain.shape, generator=generator)
    alpha = torch.rand(batch, generator=generator) * 0.4 + 0.7
    beta = torch.rand(batch, generator=generator) - 0.4
    return field, potential, alpha, beta


# --------------------------------------------------------------------------------
# Interface contract
# --------------------------------------------------------------------------------


def test_fno_returns_a_complex_field_of_the_input_shape():
    domain = SMALL.domain
    model = _model(domain)
    field, potential, alpha, beta = _inputs(domain)

    with torch.no_grad():
        output = model(field, potential, alpha, beta, SMALL.dt)

    assert output.shape == field.shape
    assert output.is_complex()


def test_reference_solver_satisfies_the_same_interface():
    """The ground truth must be a valid StepOperator so evaluators can be checked on it."""

    domain = SMALL.domain
    field, potential, alpha, beta = _inputs(domain)
    # Note: ``.double()`` on a complex tensor silently discards the imaginary part.
    # Always widen complex tensors with an explicit complex dtype.
    field = field.to(torch.complex128)
    potential, alpha, beta = potential.double(), alpha.double(), beta.double()

    output = SplitStepNLSOperator(domain)(field, potential, alpha, beta, SMALL.dt)

    assert output.shape == field.shape and output.is_complex()


def test_models_refuse_a_dt_they_were_not_trained_at():
    """Silently answering at the wrong dt would corrupt the Phase 6 transfer test."""

    domain = SMALL.domain
    model = _model(domain)
    field, potential, alpha, beta = _inputs(domain)

    model(field, potential, alpha, beta, SMALL.dt)
    with pytest.raises(ValueError, match="does not support dt transfer"):
        model(field, potential, alpha, beta, 2 * SMALL.dt)


def test_real_input_is_rejected():
    domain = SMALL.domain
    model = _model(domain)
    field, potential, alpha, beta = _inputs(domain)

    with pytest.raises(ValueError, match="complex"):
        model(field.real, potential, alpha, beta, SMALL.dt)


def test_forward_is_deterministic_under_a_fixed_seed():
    domain = SMALL.domain
    field, potential, alpha, beta = _inputs(domain)

    with torch.no_grad():
        first = _model(domain)(field, potential, alpha, beta, SMALL.dt)
        second = _model(domain)(field, potential, alpha, beta, SMALL.dt)

    assert torch.equal(first, second)


def test_coordinate_channel_is_off_by_default():
    """Feeding absolute x breaks translation equivariance of the joint operator."""

    domain = SMALL.domain
    assert _model(domain).use_coordinate_channel is False
    assert _model(domain, use_coordinate_channel=True).parameter_count() > _model(
        domain
    ).parameter_count()


# --------------------------------------------------------------------------------
# Model B: mass, and only mass
# --------------------------------------------------------------------------------


def test_projection_preserves_mass_for_untrained_weights():
    """Structural, so it must hold at random initialization, not just after training."""

    domain = PeriodicDomain.periodic_1d(64)
    model = MassProjectedOperator(_model(domain))
    field, potential, alpha, beta = _inputs(domain)

    with torch.no_grad():
        output = model(field, potential, alpha, beta, SMALL.dt)

    assert float(mass_drift(output, field, domain).max()) < 1e-6


def test_projection_preserves_mass_over_a_long_rollout():
    domain = PeriodicDomain.periodic_1d(64)
    model = MassProjectedOperator(_model(domain))
    field, potential, alpha, beta = _inputs(domain)

    state = field
    with torch.no_grad():
        for _ in range(100):
            state = model(state, potential, alpha, beta, SMALL.dt)

    # float32 floor, quoted rather than rounded away.
    assert float(mass_drift(state, field, domain).max()) < 1e-4


def test_unprojected_core_does_not_preserve_mass():
    """Guards against a vacuous projection test: the core must genuinely drift."""

    domain = PeriodicDomain.periodic_1d(64)
    core = _model(domain)
    field, potential, alpha, beta = _inputs(domain)

    with torch.no_grad():
        output = core(field, potential, alpha, beta, SMALL.dt)

    assert float(mass_drift(output, field, domain).max()) > 1e-3


def test_projection_does_not_confer_reversibility():
    """Model B enforces one scalar invariant; it has no time-reversal structure.

    Stated as a test so the thesis cannot accidentally claim otherwise.
    """

    domain = PeriodicDomain.periodic_1d(64)
    model = MassProjectedOperator(_model(domain))
    field, potential, alpha, beta = _inputs(domain)

    with torch.no_grad():
        forward = model(field, potential, alpha, beta, SMALL.dt)
        model.supports_dt_transfer = True  # allow the backward probe
        recovered = model(forward, potential, alpha, beta, -SMALL.dt)
        relative = torch.linalg.vector_norm(
            (recovered - field).flatten(1), dim=1
        ) / torch.linalg.vector_norm(field.flatten(1), dim=1)

    assert float(relative.min()) > 1e-2


def test_disabling_projection_recovers_the_core_exactly():
    domain = PeriodicDomain.periodic_1d(64)
    core = _model(domain)
    wrapped = MassProjectedOperator(core, enabled=False)
    field, potential, alpha, beta = _inputs(domain)

    with torch.no_grad():
        assert torch.equal(
            wrapped(field, potential, alpha, beta, SMALL.dt),
            core(field, potential, alpha, beta, SMALL.dt),
        )


def test_projection_reports_the_core_parameter_count():
    domain = SMALL.domain
    core = _model(domain)

    assert MassProjectedOperator(core).parameter_count() == core.parameter_count()


def test_gradients_flow_through_the_projection():
    domain = PeriodicDomain.periodic_1d(64)
    model = MassProjectedOperator(_model(domain))
    field, potential, alpha, beta = _inputs(domain)

    output = model(field, potential, alpha, beta, SMALL.dt)
    output.abs().sum().backward()

    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert max(float(g.abs().max()) for g in gradients) > 0


# --------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------


def test_model_can_overfit_a_tiny_training_set(shards):
    """If it cannot fit 8 trajectories, nothing downstream is interpretable."""

    domain = SMALL.domain
    model = _model(domain)
    config = TrainConfig(epochs=60, batch_size=16, patience=60, learning_rate=3e-3)

    history = train_one_step(model, shards["train"], shards["val"], SMALL, config, verbose=False)

    assert history.train_loss[-1] < 0.1 * history.train_loss[0]
    assert history.best_val < float("inf")


def test_training_is_reproducible(shards):
    domain = SMALL.domain
    config = TrainConfig(epochs=3, batch_size=16, patience=10)

    losses = []
    for _ in range(2):
        model = _model(domain)
        history = train_one_step(
            model, shards["train"], shards["val"], SMALL, config, verbose=False
        )
        losses.append(history.train_loss)

    assert losses[0] == pytest.approx(losses[1], rel=1e-9)


def test_relative_l2_is_scale_invariant_in_the_target():
    domain = PeriodicDomain.periodic_1d(32)
    prediction = torch.randn(4, 32, dtype=torch.complex64)
    target = torch.randn(4, 32, dtype=torch.complex64)

    with torch.no_grad():
        base = relative_l2_loss(prediction, target, domain)
        scaled = relative_l2_loss(3 * prediction, 3 * target, domain)

    assert float(base) == pytest.approx(float(scaled), rel=1e-6)


def test_one_step_batches_match_the_reference_dataset(shards):
    """The fast path must agree with the readable per-sample path."""

    from spno.data.datasets import OneStepDataset, as_complex

    shard = shards["val"]
    slow = OneStepDataset(shard, dtype=torch.float32)
    fast = OneStepBatches(shard, device="cpu")
    generator = torch.Generator().manual_seed(0)

    batch = next(fast.batches(len(fast), generator, shuffle=False))
    for index in range(min(len(slow), 10)):
        expected = slow[index]
        assert torch.allclose(batch["psi"][index], as_complex(expected["psi"]))
        assert torch.allclose(batch["target"][index], as_complex(expected["target"]))
        assert torch.allclose(batch["potential"][index], expected["potential"])


# --------------------------------------------------------------------------------
# Precision widening (invariant evaluation depends on it)
# --------------------------------------------------------------------------------


def test_widening_preserves_complex_weights_and_predictions():
    """``.double()`` skips complex params and ``.to(float64)`` destroys them.

    Both were observed on this model, and the second fails silently, so the explicit
    helper is tested against the actual predictions rather than just dtypes.
    """

    from spno.precision import parameter_dtypes, widen_to_double

    domain = PeriodicDomain.periodic_1d(64)
    model = _model(domain)
    field, potential, alpha, beta = _inputs(domain)

    widened = widen_to_double(model)

    assert parameter_dtypes(widened) == {"torch.float64", "torch.complex128"}
    with torch.no_grad():
        single = model(field, potential, alpha, beta, SMALL.dt)
        double = widened(
            field.to(torch.complex128),
            potential.double(),
            alpha.double(),
            beta.double(),
            SMALL.dt,
        )
    assert torch.allclose(single, double.to(torch.complex64), atol=1e-5)


def test_stock_to_float64_would_destroy_the_spectral_weights():
    """Documents why the helper exists; if torch ever fixes this, revisit."""

    domain = PeriodicDomain.periodic_1d(64)
    model = _model(domain)

    with pytest.warns(UserWarning, match="discards the imaginary part"):
        broken = model.to(torch.float64)

    assert "torch.complex128" not in {str(p.dtype) for p in broken.parameters()}


# ---------------------------------------------------------------------------
# Task 8: dt transfer is an explicit, scoped opt-in (G6b)
# ---------------------------------------------------------------------------


def _double_inputs(batch=2, n=32):
    field = torch.randn(batch, n, dtype=torch.complex128)
    potential = torch.zeros(batch, n, dtype=torch.float64)
    alpha = torch.full((batch,), 0.9, dtype=torch.float64)
    beta = torch.full((batch,), 0.3, dtype=torch.float64)
    return field, potential, alpha, beta


def test_dt_transfer_is_refused_by_default_for_every_model():
    domain = PeriodicDomain.periodic_1d(32)
    for model in (
        FNOStepOperator(domain, modes=4, width=8, n_layers=1, trained_dt=0.01),
        DensityPhaseSplitStep(domain, trained_dt=0.01),
    ):
        assert model.supports_dt_transfer is False


def test_allow_dt_transfer_is_scoped_and_restores_the_refusal():
    domain = PeriodicDomain.periodic_1d(32)
    torch.manual_seed(0)
    model = widen_to_double(DensityPhaseSplitStep(domain, trained_dt=0.01))
    field, potential, alpha, beta = _double_inputs()

    with pytest.raises(ValueError, match="does not support dt transfer"):
        model(field, potential, alpha, beta, 0.02)
    with allow_dt_transfer(model):
        assert model(field, potential, alpha, beta, 0.02).shape == field.shape
    with pytest.raises(ValueError, match="does not support dt transfer"):
        model(field, potential, alpha, beta, 0.02)


def test_allow_dt_transfer_restores_the_flag_after_an_exception():
    """The paired negative: a leak would license every later dt-transfer measurement."""

    domain = PeriodicDomain.periodic_1d(32)
    model = DensityPhaseSplitStep(domain, trained_dt=0.01)
    with pytest.raises(RuntimeError):
        with allow_dt_transfer(model):
            raise RuntimeError("boom")
    assert model.supports_dt_transfer is False


def test_allow_dt_transfer_leaves_no_instance_attribute_behind():
    """``supports_dt_transfer`` is a CLASS attribute.  Assigning on the instance shadows
    it, so a naive restore would leave a lingering instance attribute that reads
    correctly but is no longer bound to the class default."""

    domain = PeriodicDomain.periodic_1d(32)
    model = DensityPhaseSplitStep(domain, trained_dt=0.01)
    assert "supports_dt_transfer" not in vars(model)
    with allow_dt_transfer(model):
        pass
    assert "supports_dt_transfer" not in vars(model)


def test_the_fno_returns_the_same_map_at_every_dt_so_transfer_is_meaningless():
    """Not a bug -- a property of the class, and the reason G6b is C-family only."""

    domain = PeriodicDomain.periodic_1d(32)
    torch.manual_seed(0)
    model = widen_to_double(
        FNOStepOperator(domain, modes=4, width=8, n_layers=1, trained_dt=0.01)
    )
    field, potential, alpha, beta = _double_inputs()
    with allow_dt_transfer(model), torch.no_grad():
        assert torch.equal(
            model(field, potential, alpha, beta, 0.01),
            model(field, potential, alpha, beta, 0.02),
        )


def test_the_split_family_does_depend_on_dt():
    """The paired positive: C1's map genuinely moves with dt, which is what makes a
    transfer measurement meaningful for it and vacuous for the FNO."""

    domain = PeriodicDomain.periodic_1d(32)
    torch.manual_seed(0)
    model = widen_to_double(DensityPhaseSplitStep(domain, trained_dt=0.01))
    field, potential, alpha, beta = _double_inputs()
    with allow_dt_transfer(model), torch.no_grad():
        near = model(field, potential, alpha, beta, 0.01)
        far = model(field, potential, alpha, beta, 0.02)
    assert float(torch.abs(near - far).max()) > 1e-6


def test_train_multi_dt_runs_and_reports_one_step_validation():
    """train_multi_dt is otherwise untested code on the G6a path.  Tiny budget: this
    checks the plumbing (batches carry dt, history is populated, the model survives),
    not the physics."""

    import dataclasses

    from spno.data.datasets import MultiDtBatches
    from spno.train import train_multi_dt

    domain = PeriodicDomain.periodic_1d(32)
    small = dataclasses.replace(SMALL, grid_size=32, n_train=8, n_val=4, steps=3)
    shards = {
        dt: generate_shard(dataclasses.replace(small, dt=dt), "train")
        for dt in (0.005, 0.01)
    }
    train_batches = MultiDtBatches(shards, device="cpu")
    val_batches = OneStepBatches(generate_shard(small, "val"), device="cpu")
    config = dataclasses.replace(
        TrainConfig(), epochs=2, batch_size=4, device="cpu", log_every=100
    )

    torch.manual_seed(0)
    model = FNOStepOperator(domain, modes=4, width=8, n_layers=1, trained_dt=None)
    history = train_multi_dt(
        model, train_batches, val_batches, small, config, verbose=False
    )

    assert len(history.train_loss) == 2
    assert len(history.val_loss) == 2
    assert history.best_epoch >= 0
    assert all(loss == loss for loss in history.val_loss), "no NaN"


def test_multi_dt_training_refuses_a_model_that_claims_a_single_trained_dt():
    """The gate, and the reason it exists.

    ``_check_dt`` refuses any dt other than ``trained_dt`` -- the same guard that makes
    the G6b transfer measurement meaningful.  It therefore also blocks multi-dt
    *training* on a model claiming one dt, and it should: such a model would either
    crash mid-epoch or, if silenced with allow_dt_transfer, quietly conflate G6a with
    G6b.  Failing loudly up front is the correct behaviour.
    """

    import dataclasses

    from spno.data.datasets import MultiDtBatches
    from spno.train import train_multi_dt

    domain = PeriodicDomain.periodic_1d(32)
    small = dataclasses.replace(SMALL, grid_size=32, n_train=8, n_val=4, steps=3)
    shards = {
        dt: generate_shard(dataclasses.replace(small, dt=dt), "train")
        for dt in (0.005, 0.01)
    }
    config = dataclasses.replace(TrainConfig(), epochs=1, batch_size=4, device="cpu")
    model = FNOStepOperator(domain, modes=4, width=8, n_layers=1, trained_dt=small.dt)

    with pytest.raises(RuntimeError, match="trained_dt=None"):
        train_multi_dt(
            model,
            MultiDtBatches(shards, device="cpu"),
            OneStepBatches(generate_shard(small, "val"), device="cpu"),
            small,
            config,
            verbose=False,
        )
