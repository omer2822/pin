"""The convergence gate: a run whose best epoch is its last is budget-bound.

Phases 2-3 shipped six runs with ``best_epoch == 24`` at a 25-epoch cap -- early
stopping never fired and validation loss was still falling 26-50% over the final five
epochs.  Those numbers describe the training budget, not the models.  The gate below is
what makes that condition impossible to ship silently, so it lives in the library rather
than being re-derived in each phase script.
"""

from __future__ import annotations

from spno.experiments import budget_warning, converged, run_identifier
from spno.train import TrainHistory


def _history(val_loss: list[float], best_epoch: int) -> TrainHistory:
    history = TrainHistory()
    history.val_loss = list(val_loss)
    history.best_epoch = best_epoch
    return history


def test_best_epoch_on_the_final_epoch_is_not_converged():
    assert converged(_history([3.0, 2.0, 1.0], best_epoch=2)) is False


def test_early_stopping_having_fired_is_converged():
    assert converged(_history([3.0, 1.0, 1.5, 1.6], best_epoch=1)) is True


def test_a_run_with_no_epochs_is_not_converged():
    """The degenerate case must not read as success: best_epoch is -1, len is 0."""

    assert converged(_history([], best_epoch=-1)) is False


def test_budget_warning_names_every_unconverged_model_and_is_none_when_clean():
    summary = {"A": {"converged": [True, False, True]}, "C1": {"converged": [True]}}
    message = budget_warning(summary)
    assert message is not None and "A" in message and "C1" not in message
    assert budget_warning({"A": {"converged": [True, True]}}) is None


def test_budget_warning_treats_a_single_unconverged_seed_as_unconverged():
    """Reporting a mean over seeds where one was budget-bound would launder it."""

    assert budget_warning({"A": {"converged": [True, True, False]}}) is not None


def test_a_quick_run_never_shares_a_directory_with_the_real_one():
    """The paired negative for the identifier: a smoke run must not be able to clobber.

    Observed, not hypothetical -- running the documented ``--quick`` smoke test
    overwrote ``results/phase23-bd4e108527/metrics.json`` with three-epoch numbers.
    """

    assert run_identifier("bd4e108527") == "bd4e108527"
    assert run_identifier("bd4e108527", quick=True) == "bd4e108527-quick"
    assert run_identifier("bd4e108527") != run_identifier("bd4e108527", quick=True)


def test_run_identifier_folds_in_every_swept_knob():
    assert (
        run_identifier("bd4e108527", "one-step", "K0L0")
        == "bd4e108527-one-step-K0L0"
    )
    assert run_identifier("bd4e108527", "", "K0L0") == "bd4e108527-K0L0"
