"""Train the Phase 6-only ablations without mixing training into evaluation.

This produces the matched-bandwidth ``A-wide`` checkpoint, the fixed-alpha G7
checkpoints, and the multi-dt G6a checkpoints.  Phase 6 evaluation remains a pure
checkpoint consumer in :mod:`scripts.run_phase6`.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from spno.checkpoints import CheckpointMetadata, checkpoint_path, save_checkpoint
from spno.config import DataConfig, config_hash
from spno.data.datasets import (
    MultiDtBatches,
    OneStepBatches,
    SPLIT_SEED_OFFSET,
    TrajectoryShard,
    generate_shard,
    shard_paths,
)
from spno.data.shift import SHIFT_SPECS, ShiftSpec, shift_identifier
from spno.experiments import (
    DATA_ROOT,
    RESULTS_ROOT,
    converged,
    pick_device,
    run_identifier,
    save_run,
)
from spno.models.fno import FNOStepOperator
from spno.models.split_learned import (
    DensityPhaseSplitStep,
    FieldDensityPhaseSplitStep,
    FullFieldPhaseSplitStep,
)
from spno.train import (
    TrainConfig,
    field_scale,
    train_multi_dt,
    train_one_step,
)

if __package__:
    from scripts.run_phase6 import (
        DT_VALUES,
        multi_dt_data_hash,
        multi_dt_identifier,
        phase6_artifact_root,
        quick_data_config,
    )
else:
    from run_phase6 import (
        DT_VALUES,
        multi_dt_data_hash,
        multi_dt_identifier,
        phase6_artifact_root,
        quick_data_config,
    )


def _load_shards(root: Path, config: DataConfig) -> dict[str, TrajectoryShard]:
    paths = shard_paths(root, config_hash(config))
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "production shards are required before Phase 6 arm training; run "
            "scripts/run_phase1.py first.\n  missing: " + "\n  ".join(missing)
        )
    return {split: TrajectoryShard.load(path) for split, path in paths.items()}


def _ensure_shard(
    root: Path,
    identifier: str,
    config: DataConfig,
    split: str,
    *,
    potential_family: str = "random",
    shift_name: str | None = None,
) -> TrajectoryShard:
    path = shard_paths(root, identifier)[split]
    if path.exists():
        shard = TrajectoryShard.load(path)
    else:
        shard = generate_shard(
            config, split, potential_family=potential_family
        )
    provenance = {
        "shift_name": shift_name or identifier,
        "potential_family": potential_family,
        "config_hash": config_hash(config),
        "config_seed": config.seed,
        "seed": config.seed + SPLIT_SEED_OFFSET[split],
    }
    if any(shard.metadata.get(key) != value for key, value in provenance.items()):
        shard.metadata.update(provenance)
        shard.save(path)
    return shard


def _ensure_shift_shards(
    root: Path,
    specs: dict[str, ShiftSpec],
    *,
    quick: bool,
) -> dict[str, dict[str, TrajectoryShard]]:
    generated = {}
    for name, spec in specs.items():
        if not (name.startswith(("G1", "G2", "G3", "G4")) or name == "G7-alpha-fixed"):
            continue
        config = quick_data_config(spec.config) if quick else spec.config
        splits = ("train", "val", "test") if name == "G7-alpha-fixed" else ("test",)
        generated[name] = {
            split: _ensure_shard(
                root,
                shift_identifier(spec),
                config,
                split,
                potential_family=spec.potential_family,
                shift_name=spec.name,
            )
            for split in splits
        }
    return generated


def _fno_architecture(
    data_config: DataConfig,
    *,
    modes: int,
    width: int,
    n_layers: int,
    kinetic: str,
    dial: str,
) -> dict:
    return {
        "modes": modes,
        "width": width,
        "n_layers": n_layers,
        "use_coordinate_channel": False,
        "alpha_range": list(data_config.alpha_range),
        "beta_range": list(data_config.beta_range),
        "grid_size": data_config.grid_size,
        "kinetic_mode": kinetic,
        "dial": dial,
    }


def _structured_architecture(
    name: str, data_config: DataConfig, kinetic: str, dial: str
) -> dict:
    architecture = {
        "kinetic_mode": kinetic,
        "local_mode": None if name == "C3" else "L0",
        "grid_size": data_config.grid_size,
        "dial": dial,
    }
    if name == "C1":
        architecture["width"] = 32
    else:
        architecture.update(modes=16, width=64, n_layers=4)
    return architecture


def _structured_model(
    name: str,
    data_config: DataConfig,
    *,
    kinetic: str,
    trained_dt: float | None,
):
    if name == "C1":
        return DensityPhaseSplitStep(
            data_config.domain,
            kinetic_mode=kinetic,
            local_mode="L0",
            width=32,
            trained_dt=trained_dt,
        )
    if name == "C2":
        return FieldDensityPhaseSplitStep(
            data_config.domain,
            kinetic_mode=kinetic,
            local_mode="L0",
            modes=16,
            width=64,
            n_layers=4,
            trained_dt=trained_dt,
        )
    if name == "C3":
        return FullFieldPhaseSplitStep(
            data_config.domain,
            kinetic_mode=kinetic,
            modes=16,
            width=64,
            n_layers=4,
            trained_dt=trained_dt,
        )
    raise ValueError(f"unknown structured model {name}")


def _metadata(
    *,
    name: str,
    data_hash: str,
    seed: int,
    train_mode: str,
    scale: float,
    trained_dt: float | None,
    architecture: dict,
    history,
    quick: bool,
) -> CheckpointMetadata:
    return CheckpointMetadata(
        schema_version=1,
        model_name=name,
        data_hash=data_hash,
        seed=seed,
        train_mode=train_mode,
        field_scale=scale,
        trained_dt=trained_dt,
        architecture=architecture,
        converged=False if quick else converged(history),
        best_epoch=history.best_epoch,
    )


def train_phase6_arms(
    data_config: DataConfig,
    *,
    seeds,
    epochs: int,
    device: str,
    kinetic: str,
    quick: bool,
    data_root=DATA_ROOT,
    checkpoint_root=RESULTS_ROOT,
    artifact_root: Path | None = None,
    shift_specs: dict[str, ShiftSpec] = SHIFT_SPECS,
) -> dict:
    """Train and persist every Phase 6-only arm."""

    seeds = tuple(seeds[:1] if quick else seeds)
    data_root = Path(data_root)
    checkpoint_root = Path(checkpoint_root)
    artifact_root = (
        phase6_artifact_root(quick=quick)
        if artifact_root is None
        else Path(artifact_root)
    )
    production = _load_shards(data_root, data_config)
    production_scale = field_scale(production["train"], data_config.domain)
    shift_shards = _ensure_shift_shards(
        artifact_root, shift_specs, quick=quick
    )
    if "G7-alpha-fixed" not in shift_specs:
        raise RuntimeError("G7-alpha-fixed is missing from the shift registry")
    fixed_spec = shift_specs["G7-alpha-fixed"]
    fixed_config = quick_data_config(fixed_spec.config) if quick else fixed_spec.config
    fixed_shards = shift_shards["G7-alpha-fixed"]
    fixed_scale = field_scale(fixed_shards["train"], fixed_config.domain)

    dt_configs = {
        dt: quick_data_config(replace(data_config, dt=dt))
        if quick
        else replace(data_config, dt=dt)
        for dt in DT_VALUES
    }
    dt_shards = {
        dt: {
            split: _ensure_shard(
                artifact_root,
                multi_dt_identifier(config),
                config,
                split,
                shift_name=f"G6a-dt-{dt:g}",
            )
            for split in ("train", "val", "test")
        }
        for dt, config in dt_configs.items()
    }

    train_config = TrainConfig(
        epochs=epochs,
        batch_size=256,
        learning_rate=1e-3,
        patience=8,
        device=device,
        max_train_pairs=2 if quick else None,
    )
    identifier = run_identifier(
        config_hash(data_config), kinetic, quick=quick
    )
    result = {
        "identifier": identifier,
        "data_hash": config_hash(data_config),
        "field_scale": production_scale,
        "seeds": list(seeds),
        "quick": quick,
        "kinetic_mode": kinetic,
        "artifact_root": str(artifact_root),
        "A-wide": {"by_seed": {}},
        "G7-alpha-fixed": {"by_model": {}},
        "G6a": {
            "data_hash": multi_dt_data_hash(data_config, quick=quick),
            "dt_values": list(DT_VALUES),
            "by_model": {},
            "unsupported": {
                "A": {
                    "supported": False,
                    "reason": "the FNO ignores dt, so multi-dt supervision cannot be represented",
                }
            },
        },
    }

    for seed in seeds:
        seed_config = replace(train_config, seed=seed)
        torch.manual_seed(seed)
        wide_architecture = _fno_architecture(
            data_config,
            modes=32,
            width=64,
            n_layers=4,
            kinetic=kinetic,
            dial="matched-bandwidth",
        )
        wide = FNOStepOperator(
            data_config.domain,
            modes=32,
            width=64,
            n_layers=4,
            alpha_range=data_config.alpha_range,
            beta_range=data_config.beta_range,
            field_scale=production_scale,
            trained_dt=data_config.dt,
        )
        wide_history = train_one_step(
            wide,
            production["train"],
            production["val"],
            data_config,
            seed_config,
        )
        wide_path = checkpoint_path(
            checkpoint_root, "phase6", identifier, "A-wide", seed
        )
        save_checkpoint(
            wide_path,
            wide,
            _metadata(
                name="A-wide",
                data_hash=config_hash(data_config),
                seed=seed,
                train_mode="one-step",
                scale=production_scale,
                trained_dt=data_config.dt,
                architecture=wide_architecture,
                history=wide_history,
                quick=quick,
            ),
        )
        result["A-wide"]["by_seed"][str(seed)] = {
            "checkpoint": str(wide_path),
            "history": wide_history.as_dict(),
        }

        for name in ("A", "C1", "C2"):
            torch.manual_seed(seed)
            if name == "A":
                model = FNOStepOperator(
                    fixed_config.domain,
                    modes=16,
                    width=64,
                    n_layers=4,
                    # A zero-width fixed-alpha normalization interval would divide by
                    # zero. Keep production preprocessing while fixing observed alpha.
                    alpha_range=data_config.alpha_range,
                    beta_range=data_config.beta_range,
                    field_scale=fixed_scale,
                    trained_dt=fixed_config.dt,
                )
            else:
                model = _structured_model(
                    name,
                    fixed_config,
                    kinetic=kinetic,
                    trained_dt=fixed_config.dt,
                )
            history = train_one_step(
                model,
                fixed_shards["train"],
                fixed_shards["val"],
                fixed_config,
                seed_config,
            )
            tagged_name = f"G7-alpha-fixed/{name}"
            architecture = (
                _fno_architecture(
                    data_config,
                    modes=16,
                    width=64,
                    n_layers=4,
                    kinetic=kinetic,
                    dial="alpha-fixed",
                )
                if name == "A"
                else _structured_architecture(
                    name, fixed_config, kinetic, "alpha-fixed"
                )
            )
            path = checkpoint_path(
                checkpoint_root, "phase6", identifier, tagged_name, seed
            )
            save_checkpoint(
                path,
                model,
                _metadata(
                    name=tagged_name,
                    data_hash=config_hash(fixed_config),
                    seed=seed,
                    train_mode="one-step",
                    scale=fixed_scale,
                    trained_dt=fixed_config.dt,
                    architecture=architecture,
                    history=history,
                    quick=quick,
                ),
            )
            result["G7-alpha-fixed"]["by_model"].setdefault(name, {})[
                str(seed)
            ] = {"checkpoint": str(path), "history": history.as_dict()}

        multi_train = MultiDtBatches(
            {dt: shards["train"] for dt, shards in dt_shards.items()},
            device=device,
        )
        validation = OneStepBatches(dt_shards[data_config.dt]["val"], device=device)
        for name in ("C1", "C2", "C3"):
            torch.manual_seed(seed)
            model = _structured_model(
                name, data_config, kinetic=kinetic, trained_dt=None
            )
            history = train_multi_dt(
                model,
                multi_train,
                validation,
                data_config,
                seed_config,
            )
            tagged_name = f"G6a/{name}"
            architecture = _structured_architecture(
                name, data_config, kinetic, "multi-dt"
            )
            architecture["dt_values"] = list(DT_VALUES)
            path = checkpoint_path(
                checkpoint_root, "phase6", identifier, tagged_name, seed
            )
            save_checkpoint(
                path,
                model,
                _metadata(
                    name=tagged_name,
                    data_hash=multi_dt_data_hash(data_config, quick=quick),
                    seed=seed,
                    train_mode="multi-dt",
                    scale=production_scale,
                    trained_dt=None,
                    architecture=architecture,
                    history=history,
                    quick=quick,
                ),
            )
            result["G6a"]["by_model"].setdefault(name, {})[str(seed)] = {
                "checkpoint": str(path),
                "history": history.as_dict(),
            }
    return result


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--kinetic", default="K0", choices=("K0", "K1", "K2"))
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    device = pick_device(args.device)
    data_config = DataConfig()
    result = train_phase6_arms(
        data_config,
        seeds=args.seeds,
        epochs=args.epochs,
        device=device,
        kinetic=args.kinetic,
        quick=args.quick,
    )
    output = save_run("phase6-arms", result["identifier"], result)
    print(f"{'phase':<22}6 arm training")
    print(f"{'identifier':<22}{result['identifier']}")
    print(f"{'device':<22}{device}")
    print(f"{'artifact root':<22}{result['artifact_root']}")
    print(f"{'written to':<22}{output}")
    return result


if __name__ == "__main__":
    main()
