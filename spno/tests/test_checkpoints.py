from dataclasses import replace

import pytest
import torch

from spno.checkpoints import (
    CheckpointMetadata,
    load_checkpoint_payload,
    restore_checkpoint,
    save_checkpoint,
)
from spno.config import DataConfig, config_hash
from spno.models.fno import FNOStepOperator


def _model(data, scale=0.5):
    return FNOStepOperator(
        data.domain,
        modes=4,
        width=8,
        n_layers=1,
        alpha_range=data.alpha_range,
        beta_range=data.beta_range,
        field_scale=scale,
        trained_dt=data.dt,
    )


def test_checkpoint_roundtrip_carries_preprocessing_and_convergence(tmp_path):
    data = replace(DataConfig(), grid_size=16)
    model = _model(data)
    meta = CheckpointMetadata(
        schema_version=1,
        model_name="A",
        data_hash=config_hash(data),
        seed=3,
        train_mode="one-step",
        field_scale=0.5,
        trained_dt=data.dt,
        architecture={"modes": 4, "width": 8, "n_layers": 1},
        converged=True,
        best_epoch=7,
    )
    path = tmp_path / "A-seed3.pt"
    save_checkpoint(path, model, meta)
    payload = load_checkpoint_payload(path)
    restored = _model(data)
    restore_checkpoint(restored, payload, expected_data_hash=config_hash(data))
    assert payload.metadata == meta
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def test_nonconverged_or_wrong_dataset_checkpoint_is_refused(tmp_path):
    data = replace(DataConfig(), grid_size=16)
    path = tmp_path / "bad.pt"
    meta = CheckpointMetadata(
        schema_version=1,
        model_name="A",
        data_hash="wrong",
        seed=0,
        train_mode="one-step",
        field_scale=0.5,
        trained_dt=data.dt,
        architecture={"modes": 4, "width": 8, "n_layers": 1},
        converged=False,
        best_epoch=1,
    )
    save_checkpoint(path, _model(data), meta)
    payload = load_checkpoint_payload(path)
    with pytest.raises(RuntimeError, match="dataset hash"):
        restore_checkpoint(_model(data), payload, expected_data_hash=config_hash(data))


def test_budget_bound_checkpoint_requires_an_explicit_quick_run_override(tmp_path):
    data = replace(DataConfig(), grid_size=16)
    path = tmp_path / "quick.pt"
    meta = CheckpointMetadata(
        schema_version=1,
        model_name="A",
        data_hash=config_hash(data),
        seed=0,
        train_mode="one-step",
        field_scale=0.5,
        trained_dt=data.dt,
        architecture={"modes": 4, "width": 8, "n_layers": 1},
        converged=False,
        best_epoch=0,
    )
    save_checkpoint(path, _model(data), meta)
    payload = load_checkpoint_payload(path)
    with pytest.raises(RuntimeError, match="budget-bound"):
        restore_checkpoint(_model(data), payload, expected_data_hash=config_hash(data))
    restore_checkpoint(
        _model(data),
        payload,
        expected_data_hash=config_hash(data),
        allow_budget_bound=True,
    )
