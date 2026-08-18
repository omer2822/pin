from __future__ import annotations

import argparse
import math
from dataclasses import replace

import pytest
import torch

from spno.checkpoints import checkpoint_path, load_checkpoint_payload
from spno.config import DataConfig
from spno.data.datasets import TrajectoryShard
from spno.models.fno import FNOStepOperator
from spno.train import (
    TrainConfig,
    TrainHistory,
    field_scale,
    train_one_step,
    train_pino,
)


def _two_trajectory_shards(data: DataConfig, *, steps: int) -> dict[str, TrajectoryShard]:
    generator = torch.Generator().manual_seed(9182)
    initial = torch.complex(
        torch.randn(2, data.grid_size, generator=generator, dtype=torch.float64),
        torch.randn(2, data.grid_size, generator=generator, dtype=torch.float64),
    )
    trajectories = initial[:, None].repeat(1, steps + 1, 1)
    potential = torch.zeros(2, data.grid_size, dtype=torch.float64)
    alpha = torch.tensor([0.8, 1.0], dtype=torch.float64)
    beta = torch.tensor([-0.1, 0.2], dtype=torch.float64)
    return {
        split: TrajectoryShard(
            trajectories=trajectories.clone(),
            potential=potential.clone(),
            alpha=alpha.clone(),
            beta=beta.clone(),
            trajectory_ids=torch.arange(2) + offset,
            dt=data.dt,
            split=split,
        )
        for split, offset in (("train", 0), ("val", 10_000), ("test", 20_000))
    }


def _tiny_fno(domain, **kwargs) -> FNOStepOperator:
    return FNOStepOperator(
        domain,
        modes=2,
        width=4,
        n_layers=1,
        alpha_range=kwargs.get("alpha_range", (0.7, 1.1)),
        beta_range=kwargs.get("beta_range", (-0.4, 0.6)),
        field_scale=kwargs.get("field_scale", 1.0),
        trained_dt=kwargs.get("trained_dt"),
    )


def test_phase7_aggregates_test_and_horizon_100_metrics():
    from scripts.run_phase7 import aggregate_pino_seed_metrics

    per_seed = [
        {
            "converged": True,
            "metrics": {
                "one_step_test": 0.2,
                # Step 100 is deliberately not at a fixed offset.
                "rollout": {
                    "steps": [100, 1],
                    "mass_drift": [0.03, 0.01],
                    "energy_drift": [0.04, 0.02],
                },
            },
        },
        {
            "converged": False,
            "metrics": {
                "one_step_test": 0.4,
                "rollout": {
                    "steps": [1, 100],
                    "mass_drift": [0.02, 0.05],
                    "energy_drift": [0.03, 0.06],
                },
            },
        },
    ]

    result = aggregate_pino_seed_metrics(per_seed)

    assert result["one_step_mean"] == pytest.approx(0.3)
    assert result["one_step_min"] == 0.2
    assert result["one_step_max"] == 0.4
    assert result["mass_drift_100_mean"] == pytest.approx(0.04)
    assert result["energy_drift_100_mean"] == pytest.approx(0.05)
    assert result["converged"] == [True, False]


def test_phase7_model_factory_is_deterministic_for_a_seed():
    from scripts.run_phase7 import build_pino_model

    data = replace(DataConfig(), grid_size=8)
    first = build_pino_model(data, scale=0.75, seed=17)
    second = build_pino_model(data, scale=0.75, seed=17)

    assert first.state_dict().keys() == second.state_dict().keys()
    assert all(
        torch.equal(value, second.state_dict()[key])
        for key, value in first.state_dict().items()
    )
    assert float(first.field_scale) == pytest.approx(0.75)


def test_phase7_run_sweep_reports_real_test_and_horizon_metrics(
    tmp_path, monkeypatch
):
    import scripts.run_phase7 as phase7

    data = replace(DataConfig(), grid_size=8, n_train=2, n_val=2, n_test=2, steps=100)
    shards = _two_trajectory_shards(data, steps=100)
    args = argparse.Namespace(
        quick=False,
        seeds=[0],
        epochs=1,
        device="cpu",
        lambdas=[0.0, 0.1],
    )
    saved = {}

    def save(phase, identifier, payload):
        output = tmp_path / f"{phase}-{identifier}"
        (output / "plots").mkdir(parents=True)
        saved.update(payload)
        return output

    monkeypatch.setattr(phase7, "FNOStepOperator", _tiny_fno)
    payload = phase7.run_sweep(shards, data, args, save=save)

    assert payload == saved
    assert payload["field_scale"] == pytest.approx(
        field_scale(shards["train"], data.domain)
    )
    assert isinstance(payload["budget_warning"], str)
    for weight in (0.0, 0.1):
        entry = payload["sweep"][str(weight)]
        assert math.isfinite(entry["one_step_mean"])
        assert math.isfinite(entry["mass_drift_100_mean"])
        assert math.isfinite(entry["energy_drift_100_mean"])
        assert entry["per_seed"][0]["config"]["train"]["epochs"] == 1
        assert entry["per_seed"][0]["metrics"]["rollout"]["steps"] == [1, 10, 20, 50, 100]


def test_pino_zero_weight_matches_one_step_history_and_weights_bitwise():
    data = replace(DataConfig(), grid_size=8, n_train=2, n_val=2, steps=1)
    shards = _two_trajectory_shards(data, steps=1)
    config = TrainConfig(
        epochs=2,
        batch_size=4,
        patience=3,
        seed=3,
        device="cpu",
    )
    torch.manual_seed(44)
    initial = _tiny_fno(data.domain, trained_dt=data.dt).state_dict()
    plain = _tiny_fno(data.domain, trained_dt=data.dt)
    pino = _tiny_fno(data.domain, trained_dt=data.dt)
    plain.load_state_dict(initial)
    pino.load_state_dict(initial)

    plain_history = train_one_step(
        plain, shards["train"], shards["val"], data, config, verbose=False
    )
    pino_history = train_pino(
        pino,
        shards["train"],
        shards["val"],
        data,
        config,
        physics_weight=0.0,
        verbose=False,
    )

    assert pino_history.train_loss == plain_history.train_loss
    assert pino_history.val_loss == plain_history.val_loss
    assert pino_history.best_val == plain_history.best_val
    assert pino_history.best_epoch == plain_history.best_epoch
    assert all(
        torch.equal(value, pino.state_dict()[key])
        for key, value in plain.state_dict().items()
    )


def _converged_history() -> TrainHistory:
    return TrainHistory(
        train_loss=[1.0, 1.0],
        val_loss=[1.0, 1.1],
        best_val=1.0,
        best_epoch=0,
    )


def _evaluation_metrics(model, *args, **kwargs):
    return {
        "one_step_test": 0.1,
        "parameters": model.parameter_count(),
        "rollout": {
            "steps": [1, 100],
            "relative_error": [0.1, 0.2],
            "mass_drift": [0.01, 0.02],
            "energy_drift": [0.03, 0.04],
            "energy_classification": "bounded",
            "diverged_at": None,
        },
    }


class _StaticResult:
    def __init__(self, payload):
        self.payload = payload

    def as_dict(self):
        return self.payload


def test_phase23_saves_a_and_b_loop_but_not_b_post_checkpoints(tmp_path, monkeypatch):
    import scripts.run_phase23 as phase23

    data = replace(DataConfig(), grid_size=8)
    shard = _two_trajectory_shards(data, steps=1)
    monkeypatch.setattr(
        phase23,
        "build_fno",
        lambda config, scale, seed: _tiny_fno(
            config.domain, field_scale=scale, trained_dt=config.dt
        ),
    )
    monkeypatch.setattr(
        phase23, "train_one_step", lambda *args, **kwargs: _converged_history()
    )
    monkeypatch.setattr(phase23, "evaluate_model", _evaluation_metrics)

    phase23.run_seed(
        3,
        shard,
        data,
        TrainConfig(),
        0.75,
        identifier="dataset-quick",
        quick=True,
        checkpoint_root=tmp_path,
    )

    for name in ("A", "B-loop"):
        path = checkpoint_path(tmp_path, "phase23", "dataset-quick", name, 3)
        payload = load_checkpoint_payload(path)
        assert payload.metadata.model_name == name
        assert payload.metadata.field_scale == pytest.approx(0.75)
        assert payload.metadata.train_mode == "one-step"
        assert payload.metadata.converged is False
    assert not checkpoint_path(
        tmp_path, "phase23", "dataset-quick", "B-post", 3
    ).exists()


def test_phase45_saves_every_live_model_checkpoint(tmp_path, monkeypatch):
    import scripts.run_phase45 as phase45

    data = replace(DataConfig(), grid_size=8)
    shard = _two_trajectory_shards(data, steps=1)
    models = {
        name: _tiny_fno(data.domain, trained_dt=data.dt)
        for name in ("A", "B-loop", "C1", "C2", "C3")
    }
    monkeypatch.setattr(phase45, "build_models", lambda *args, **kwargs: models)
    monkeypatch.setattr(phase45, "train_one_step", lambda *args, **kwargs: _converged_history())
    monkeypatch.setattr(phase45, "train_rollout", lambda *args, **kwargs: _converged_history())
    monkeypatch.setattr(phase45, "evaluate_model", _evaluation_metrics)
    monkeypatch.setattr(
        phase45,
        "rollout_inputs",
        lambda *args, **kwargs: {
            "initial": torch.ones(2, 8, dtype=torch.complex128),
            "trajectories": torch.ones(2, 101, 8, dtype=torch.complex128),
            "potential": torch.zeros(2, 8, dtype=torch.float64),
            "alpha": torch.ones(2, dtype=torch.float64),
            "beta": torch.zeros(2, dtype=torch.float64),
        },
    )
    monkeypatch.setattr(
        phase45,
        "evaluate_conservation",
        lambda *args, **kwargs: _StaticResult(
            {"energy_trend": {"slope": 0.0, "classification": "bounded"}}
        ),
    )
    monkeypatch.setattr(
        phase45,
        "evaluate_spectral",
        lambda *args, **kwargs: _StaticResult(
            {"mean_phase_error": 0.0, "banded": {}}
        ),
    )

    phase45.run_seed(
        5,
        shard,
        data,
        TrainConfig(),
        0.5,
        "K1",
        "one-step",
        "L2",
        identifier="dataset-one-step-K1L2-quick",
        quick=True,
        checkpoint_root=tmp_path,
    )

    for name in models:
        path = checkpoint_path(
            tmp_path, "phase45", "dataset-one-step-K1L2-quick", name, 5
        )
        payload = load_checkpoint_payload(path)
        assert payload.metadata.model_name == name
        assert payload.metadata.train_mode == "one-step"
        assert payload.metadata.architecture["kinetic_mode"] == "K1"
        assert payload.metadata.converged is False
