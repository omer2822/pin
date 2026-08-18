from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from spno.checkpoints import (
    CheckpointMetadata,
    checkpoint_path,
    load_checkpoint_payload,
    save_checkpoint,
)
from spno.config import DataConfig, config_hash
from spno.data.datasets import SPLIT_SEED_OFFSET, TrajectoryShard, generate_shard
from spno.data.shift import ShiftSpec, shift_identifier
from spno.experiments import run_identifier
from spno.models.fno import FNOStepOperator
from spno.models.projected import MassProjectedOperator
from spno.models.split_learned import (
    DensityPhaseSplitStep,
    FieldDensityPhaseSplitStep,
    FullFieldPhaseSplitStep,
)
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


def test_phase6_continuation_grid_is_wrap_free_to_nyquist():
    from scripts.run_phase6 import alpha_continuation_grid

    values = alpha_continuation_grid(0.9, max_k=32, dt=0.01)

    assert values[0] == 0.0 and values[-1] == 0.9
    assert max(b - a for a, b in zip(values, values[1:])) * 32**2 * 0.01 < math.pi


def _checkpoint_metadata(
    name: str,
    data_hash: str,
    architecture: dict,
    *,
    trained_dt: float | None,
) -> CheckpointMetadata:
    return CheckpointMetadata(
        schema_version=1,
        model_name=name,
        data_hash=data_hash,
        seed=0,
        train_mode="one-step",
        field_scale=1.0,
        trained_dt=trained_dt,
        architecture=architecture,
        converged=False,
        best_epoch=0,
    )


def _save_tiny_phase6_checkpoints(root, data: DataConfig) -> None:
    from scripts.run_phase6 import multi_dt_data_hash

    phase23_identifier = run_identifier(config_hash(data), quick=True)
    phase45_identifier = run_identifier(
        config_hash(data), "one-step", "K0L0", quick=True
    )
    phase6_identifier = run_identifier(config_hash(data), "K0", quick=True)
    fno_architecture = {
        "modes": 2,
        "width": 4,
        "n_layers": 1,
        "use_coordinate_channel": False,
        "alpha_range": list(data.alpha_range),
        "beta_range": list(data.beta_range),
    }

    core = _tiny_fno(data.domain, trained_dt=data.dt)
    base_models = {
        "A": (core, fno_architecture),
        "B-loop": (
            MassProjectedOperator(_tiny_fno(data.domain, trained_dt=data.dt)),
            {**fno_architecture, "projection": "mass"},
        ),
    }
    for name, (model, architecture) in base_models.items():
        save_checkpoint(
            checkpoint_path(root, "phase23", phase23_identifier, name, 0),
            model,
            _checkpoint_metadata(
                name, config_hash(data), architecture, trained_dt=data.dt
            ),
        )

    structured = {
        "C1": (
            DensityPhaseSplitStep(
                data.domain, kinetic_mode="K0", width=4, trained_dt=data.dt
            ),
            {"kinetic_mode": "K0", "local_mode": "L0", "width": 4},
        ),
        "C2": (
            FieldDensityPhaseSplitStep(
                data.domain,
                kinetic_mode="K0",
                modes=2,
                width=4,
                n_layers=1,
                trained_dt=data.dt,
            ),
            {
                "kinetic_mode": "K0",
                "local_mode": "L0",
                "modes": 2,
                "width": 4,
                "n_layers": 1,
            },
        ),
        "C3": (
            FullFieldPhaseSplitStep(
                data.domain,
                kinetic_mode="K0",
                modes=2,
                width=4,
                n_layers=1,
                trained_dt=data.dt,
            ),
            {
                "kinetic_mode": "K0",
                "local_mode": None,
                "modes": 2,
                "width": 4,
                "n_layers": 1,
            },
        ),
    }
    for name, (model, architecture) in structured.items():
        save_checkpoint(
            checkpoint_path(root, "phase45", phase45_identifier, name, 0),
            model,
            _checkpoint_metadata(
                name, config_hash(data), architecture, trained_dt=data.dt
            ),
        )

    save_checkpoint(
        checkpoint_path(root, "phase6", phase6_identifier, "A-wide", 0),
        _tiny_fno(data.domain, trained_dt=data.dt),
        _checkpoint_metadata(
            "A-wide", config_hash(data), fno_architecture, trained_dt=data.dt
        ),
    )

    fixed = replace(data, alpha_range=(0.9, 0.9), seed=120)
    fixed_models = {
        "A": (_tiny_fno(data.domain, trained_dt=data.dt), fno_architecture),
        "C1": structured["C1"],
        "C2": structured["C2"],
    }
    for name, (model, architecture) in fixed_models.items():
        save_checkpoint(
            checkpoint_path(
                root, "phase6", phase6_identifier, f"G7-alpha-fixed/{name}", 0
            ),
            model,
            _checkpoint_metadata(
                f"G7-alpha-fixed/{name}",
                config_hash(fixed),
                architecture,
                trained_dt=data.dt,
            ),
        )

    for name, (_, architecture) in structured.items():
        if name == "C1":
            model = DensityPhaseSplitStep(
                data.domain, kinetic_mode="K0", width=4, trained_dt=None
            )
        elif name == "C2":
            model = FieldDensityPhaseSplitStep(
                data.domain,
                kinetic_mode="K0",
                modes=2,
                width=4,
                n_layers=1,
                trained_dt=None,
            )
        else:
            model = FullFieldPhaseSplitStep(
                data.domain,
                kinetic_mode="K0",
                modes=2,
                width=4,
                n_layers=1,
                trained_dt=None,
            )
        save_checkpoint(
            checkpoint_path(root, "phase6", phase6_identifier, f"G6a/{name}", 0),
            model,
            _checkpoint_metadata(
                f"G6a/{name}",
                multi_dt_data_hash(data, quick=True),
                {**architecture, "dt_values": [0.005, 0.01, 0.02]},
                trained_dt=None,
            ),
        )


def test_phase6_selected_arms_produce_complete_checkpoint_backed_measurements(tmp_path):
    from scripts.run_phase6 import multi_dt_identifier, run_selected_arms

    data = replace(
        DataConfig(),
        grid_size=16,
        steps=2,
        n_train=2,
        n_val=2,
        n_test=2,
        substeps=2,
        initial_bandwidth=4,
    )
    _save_tiny_phase6_checkpoints(tmp_path, data)

    shift_specs = {
        "G1-interpolation": ShiftSpec(
            "G1-interpolation", replace(data, n_train=0, n_val=0, seed=101)
        ),
        "G4-bandwidth-12": ShiftSpec(
            "G4-bandwidth-12",
            replace(
                data,
                n_train=0,
                n_val=0,
                initial_bandwidth=12,
                seed=122,
            ),
        ),
        "G7-alpha-fixed": ShiftSpec(
            "G7-alpha-fixed", replace(data, alpha_range=(0.9, 0.9), seed=120)
        ),
    }
    for spec in shift_specs.values():
        for split in (
            ("train", "val", "test")
            if spec.name == "G7-alpha-fixed"
            else ("test",)
        ):
            shard = generate_shard(
                spec.config, split, potential_family=spec.potential_family
            )
            shard.metadata.update(
                shift_name=spec.name,
                config_hash=config_hash(spec.config),
                seed=spec.config.seed,
            )
            shard.save(
                tmp_path / "shifts" / f"nls1d-{shift_identifier(spec)}" / f"{split}.pt"
            )

    for dt in (0.005, 0.01, 0.02):
        dt_config = replace(data, dt=dt)
        generate_shard(dt_config, "test").save(
            tmp_path
            / "shifts"
            / f"nls1d-{multi_dt_identifier(dt_config)}"
            / "test.pt"
        )

    result = run_selected_arms(
        ("G1", "G4", "G5a", "G5b", "G6a", "G6b", "G7", "G9"),
        data,
        seeds=(0,),
        kinetic="K0",
        device="cpu",
        quick=True,
        checkpoint_root=tmp_path,
        shift_root=tmp_path / "shifts",
        shift_specs=shift_specs,
    )

    assert result["G1"]["measurements"]["G1-interpolation"]["by_model"]["A"][
        "one_step_test"
    ] >= 0
    c1_g4 = result["G4"]["measurements"]["G4-bandwidth-12"]["by_model"]["C1"]
    assert c1_g4["spectral"]
    assert c1_g4["spectral_min"]
    assert c1_g4["spectral_max"]
    assert c1_g4["rollout_min"]
    assert c1_g4["rollout_max"]
    assert result["G5b"]["alpha_derivative"]["by_model"]
    assert result["G6a"]["by_model"]
    assert "A" not in result["G6a"]["by_model"]
    assert result["G6a"]["unsupported"]["A"]["supported"] is False
    assert result["G7"]["varying_alpha"]["band_ratio"]
    assert result["G7"]["fixed_alpha"]["band_ratio"]
    assert result["G9"]["A"]["fraction_above_cutoff"]


def test_phase6_arm_trainer_persists_all_required_quick_artifacts(tmp_path, monkeypatch):
    import scripts.train_phase6_arms as trainer

    data = replace(
        DataConfig(),
        grid_size=16,
        steps=2,
        n_train=2,
        n_val=2,
        n_test=2,
        substeps=2,
        initial_bandwidth=4,
    )
    production_root = tmp_path / "production"
    for split in ("train", "val", "test"):
        generate_shard(data, split).save(
            production_root / f"nls1d-{config_hash(data)}" / f"{split}.pt"
        )
    shift_specs = {
        "G1-interpolation": ShiftSpec(
            "G1-interpolation", replace(data, n_train=0, n_val=0, seed=101)
        ),
        "G7-alpha-fixed": ShiftSpec(
            "G7-alpha-fixed", replace(data, alpha_range=(0.9, 0.9), seed=120)
        ),
    }
    monkeypatch.setattr(
        trainer, "train_one_step", lambda *args, **kwargs: _converged_history()
    )
    monkeypatch.setattr(
        trainer, "train_multi_dt", lambda *args, **kwargs: _converged_history()
    )

    result = trainer.train_phase6_arms(
        data,
        seeds=(0,),
        epochs=1,
        device="cpu",
        kinetic="K0",
        quick=True,
        data_root=production_root,
        checkpoint_root=tmp_path,
        artifact_root=tmp_path / "quick-artifacts",
        shift_specs=shift_specs,
    )

    identifier = run_identifier(config_hash(data), "K0", quick=True)
    expected = [
        "A-wide",
        "G7-alpha-fixed/A",
        "G7-alpha-fixed/C1",
        "G7-alpha-fixed/C2",
        "G6a/C1",
        "G6a/C2",
        "G6a/C3",
    ]
    for name in expected:
        payload = load_checkpoint_payload(
            checkpoint_path(tmp_path, "phase6", identifier, name, 0)
        )
        assert payload.metadata.model_name == name
        assert payload.metadata.converged is False
        assert payload.metadata.architecture["kinetic_mode"] == "K0"

    fixed_spec = shift_specs["G7-alpha-fixed"]
    fixed_train = TrajectoryShard.load(
        tmp_path
        / "quick-artifacts"
        / f"nls1d-{shift_identifier(fixed_spec)}"
        / "train.pt"
    )
    torch.manual_seed(0)
    expected_fixed_a = FNOStepOperator(
        data.domain,
        modes=16,
        width=64,
        n_layers=4,
        alpha_range=data.alpha_range,
        beta_range=data.beta_range,
        field_scale=field_scale(fixed_train, data.domain),
        trained_dt=data.dt,
    )
    fixed_a = load_checkpoint_payload(
        checkpoint_path(
            tmp_path, "phase6", identifier, "G7-alpha-fixed/A", 0
        )
    )
    assert all(
        torch.equal(value, fixed_a.state_dict[key])
        for key, value in expected_fixed_a.state_dict().items()
    )
    assert "A" not in result["G6a"]["by_model"]
    assert result["G6a"]["unsupported"]["A"]["supported"] is False

    spec = shift_specs["G1-interpolation"]
    shard = TrajectoryShard.load(
        tmp_path
        / "quick-artifacts"
        / f"nls1d-{shift_identifier(spec)}"
        / "test.pt"
    )
    assert shard.trajectories.shape == (2, 3, 16)
    assert shard.metadata["shift_name"] == spec.name
    assert shard.metadata["potential_family"] == spec.potential_family
    assert shard.metadata["config_hash"] == config_hash(quick_data_config := replace(
        spec.config, n_train=2, n_val=2, n_test=2, steps=2
    ))
    assert shard.metadata["seed"] == (
        quick_data_config.seed + SPLIT_SEED_OFFSET["test"]
    )
    assert shard.metadata["config_seed"] == quick_data_config.seed


def test_phase6_arm_trainer_repairs_reused_shift_provenance(tmp_path):
    import scripts.train_phase6_arms as trainer

    config = replace(
        DataConfig(), grid_size=16, n_test=2, steps=2, substeps=2, seed=101
    )
    identifier = "shift"
    path = tmp_path / f"nls1d-{identifier}" / "test.pt"
    stale = generate_shard(config, "test")
    stale.metadata["seed"] = config.seed
    stale.save(path)

    trainer._ensure_shard(
        tmp_path,
        identifier,
        config,
        "test",
        shift_name="G1-interpolation",
    )

    repaired = TrajectoryShard.load(path)
    assert repaired.metadata["seed"] == config.seed + SPLIT_SEED_OFFSET["test"]
    assert repaired.metadata["config_seed"] == config.seed
    assert repaired.metadata["shift_name"] == "G1-interpolation"


def test_phase6_arm_trainer_is_directly_executable():
    root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, "scripts/train_phase6_arms.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--kinetic" in completed.stdout


def test_phase6_plots_derivative_slopes_at_interval_midpoints(tmp_path):
    from scripts.run_phase6 import make_plots

    output = tmp_path / "phase6"
    (output / "plots").mkdir(parents=True)
    payload = {
        "data_hash": "tiny",
        "alpha_train_range": [0.7, 1.1],
        "cascade_cutoff": 4,
        "experiments": {
            "G5b": {
                "alpha_derivative": {
                    "by_model": {
                        "A": {
                            "8": {
                                "alphas": [0.7, 0.9, 1.1],
                                "slopes": [-1.0, -1.1],
                                "truth": -1.0,
                                "max_relative_error": 0.1,
                            }
                        }
                    }
                }
            }
        },
    }

    make_plots(payload, output)

    assert (output / "plots" / "phase6_parameters.png").exists()


def test_phase6_checkpoint_dial_mismatch_reports_path_and_creation_command(tmp_path):
    from scripts.run_phase6 import load_models

    data = replace(
        DataConfig(),
        grid_size=16,
        steps=2,
        n_train=2,
        n_val=2,
        n_test=2,
        substeps=2,
        initial_bandwidth=4,
    )
    _save_tiny_phase6_checkpoints(tmp_path, data)
    identifier = run_identifier(
        config_hash(data), "one-step", "K0L0", quick=True
    )
    path = checkpoint_path(tmp_path, "phase45", identifier, "C1", 0)
    model = DensityPhaseSplitStep(
        data.domain, kinetic_mode="K0", width=4, trained_dt=data.dt
    )
    save_checkpoint(
        path,
        model,
        _checkpoint_metadata(
            "C1",
            config_hash(data),
            {"kinetic_mode": "K1", "local_mode": "L0", "width": 4},
            trained_dt=data.dt,
        ),
    )

    with pytest.raises(RuntimeError) as caught:
        load_models(
            tmp_path,
            data,
            (0,),
            "K0",
            allow_budget_bound=True,
        )

    message = str(caught.value)
    assert str(path) in message
    assert "scripts/run_phase45.py" in message


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
