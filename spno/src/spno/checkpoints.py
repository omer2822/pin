from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class CheckpointMetadata:
    schema_version: int
    model_name: str
    data_hash: str
    seed: int
    train_mode: str
    field_scale: float | None
    trained_dt: float | None
    architecture: dict[str, Any]
    converged: bool
    best_epoch: int


@dataclass(frozen=True)
class CheckpointPayload:
    metadata: CheckpointMetadata
    state_dict: dict[str, torch.Tensor]


def checkpoint_path(root: Path, family: str, identifier: str, model: str, seed: int) -> Path:
    return root / "checkpoints" / family / identifier / f"{model}-seed{seed}.pt"


def save_checkpoint(path: Path, model, metadata: CheckpointMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"metadata": asdict(metadata), "state_dict": model.state_dict()}, temporary)
    os.replace(temporary, path)


def load_checkpoint_payload(path: Path) -> CheckpointPayload:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    metadata = CheckpointMetadata(**raw["metadata"])
    if metadata.schema_version != 1:
        raise RuntimeError(f"unsupported checkpoint schema {metadata.schema_version}")
    return CheckpointPayload(metadata, raw["state_dict"])


def restore_checkpoint(
    model,
    payload: CheckpointPayload,
    *,
    expected_data_hash: str,
    allow_budget_bound: bool = False,
):
    if payload.metadata.data_hash != expected_data_hash:
        raise RuntimeError("checkpoint dataset hash does not match requested data")
    if not payload.metadata.converged and not allow_budget_bound:
        raise RuntimeError("checkpoint is budget-bound; raise the training epoch budget")
    model.load_state_dict(payload.state_dict, strict=True)
    return model
