"""Phase 9: the misspecification sweep -- "when does a hard invariant stop helping?"

This is the only phase that makes "structure beats FNO" a *question* rather than a
tautology, because it moves the truth **outside** the constrained model class.

Two dials on the data-generating equation, each breaking a different assumption and each
recovering NLS at zero:

``sigma``  nonlocal nonlinearity.  Breaks *locality*.  Still Hamiltonian, still U(1),
           still exactly mass-conserving -- so B's projection stays **correct** and only
           C1's pointwise ``nu`` becomes unable to represent the truth.  This is what
           separates C1 from C2.
``gamma``  weak gain/loss.  Breaks *conservation itself*, so B's hard mass constraint
           becomes actively **wrong**.

The deliverable is ``payload["crossover"]``: the dial value at which each of B and C
stops beating A, **or** an explicit bounded statement when none is found.  A located
crossover is the result; a bounded one is also publishable, so the bounded form is
emitted rather than the key being left absent.

Usage:
    python scripts/run_phase9.py [--quick] [--seeds 0 1 2] [--epochs 60]
                                 [--device auto]
                                 [--sigmas 0.0 0.1 0.25 0.5 1.0]
                                 [--gammas 0.0 1e-4 1e-3 1e-2]
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from spno.checkpoints import CheckpointMetadata, checkpoint_path, save_checkpoint
from spno.config import DataConfig, config_hash
from spno.data.datasets import (
    SPLIT_SEED_OFFSET,
    TrajectoryShard,
    assert_no_leakage,
    generate_shard,
    shard_paths,
)
from spno.evaluation.payloads import require_phase_arms
from spno.experiments import (
    DATA_ROOT,
    RESULTS_ROOT,
    converged,
    describe_config,
    evaluate_model,
    pick_device,
    run_identifier,
    save_run,
)
from spno.misspecification import MisspecificationConfig
from spno.models.fno import FNOStepOperator
from spno.models.projected import MassProjectedOperator
from spno.models.split_learned import (
    DensityPhaseSplitStep,
    FieldDensityPhaseSplitStep,
    FullFieldPhaseSplitStep,
)
from spno.solvers.perturbed import (
    GainLossSplitStepNLSOperator,
    NonlocalSplitStepNLSOperator,
)
from spno.solvers.split_step import SplitStepNLSOperator, SubsteppedReference
from spno.seeding import seed_everything
from spno.train import TrainConfig, field_scale, train_one_step


MODEL_NAMES = ("A", "B-loop", "C1", "C2", "C3")
COMPARISON_MODELS = MODEL_NAMES[1:]


def bitwise_gate(data_config: DataConfig) -> dict:
    """The dial-zero generator must reproduce the unperturbed solver **bitwise**.

    ``raise RuntimeError``, not ``assert``: ``python -O`` strips asserts, and without
    this every crossover in the sweep would be uninterpretable -- a shift of 1e-16 at
    dial zero is indistinguishable from a small real effect once it compounds over a
    200-step rollout.
    """

    domain = data_config.domain
    torch.manual_seed(0)
    field = torch.randn(4, domain.shape[0], dtype=torch.complex128)
    potential = torch.randn(4, domain.shape[0], dtype=torch.float64) * 0.3
    alpha = torch.full((4,), 0.9, dtype=torch.float64)
    beta = torch.full((4,), 0.4, dtype=torch.float64)

    # Compare the DIAL-ZERO GENERATOR against the production generator, both full
    # substepped references. No fallback branch: a gate that can silently succeed is
    # worse than no gate, so this must either compare the real objects or raise.
    production = SubsteppedReference(domain, data_config.substeps)
    zero_dial = MisspecificationConfig().reference(data_config)
    if not torch.equal(
        zero_dial(field, potential, alpha, beta, data_config.dt),
        production(field, potential, alpha, beta, data_config.dt),
    ):
        raise RuntimeError(
            "the dial-zero generator does not reproduce the production generator "
            "bitwise; every crossover in this sweep would be uninterpretable"
        )

    # And the perturbed operators themselves must short-circuit to the plain Strang
    # step at zero -- the FFT round trip in the nonlocal kernel is accurate, not exact.
    exact_step = SplitStepNLSOperator(domain)(
        field, potential, alpha, beta, data_config.dt
    )
    for build in (
        lambda: NonlocalSplitStepNLSOperator(domain, sigma=0.0),
        lambda: GainLossSplitStepNLSOperator(domain, gamma=0.0),
    ):
        operator = build()
        if not torch.equal(
            operator(field, potential, alpha, beta, data_config.dt), exact_step
        ):
            raise RuntimeError(
                f"{type(operator).__name__} at dial zero is not bitwise the "
                "unperturbed Strang step; its zero short-circuit is broken"
            )
    if MisspecificationConfig().identifier(data_config) != config_hash(data_config):
        raise RuntimeError(
            "the dial-zero identifier does not collapse to the production data hash; "
            "the exact arm would regenerate 206 MB of shards instead of reusing them"
        )
    return {"bitwise_at_zero": True, "identifier_collapses": True}


def build_models(data_config: DataConfig, scale: float, seed: int) -> dict:
    """Construct all five arms from the requested paired seed and normalization."""

    domain = data_config.domain
    common = {
        "modes": 16,
        "width": 64,
        "n_layers": 4,
        "alpha_range": data_config.alpha_range,
        "beta_range": data_config.beta_range,
        "field_scale": scale,
        "trained_dt": data_config.dt,
    }

    def seeded(build):
        seed_everything(seed)
        return build()

    return {
        "A": seeded(lambda: FNOStepOperator(domain, **common)),
        "B-loop": seeded(
            lambda: MassProjectedOperator(FNOStepOperator(domain, **common))
        ),
        "C1": seeded(
            lambda: DensityPhaseSplitStep(
                domain, kinetic_mode="K0", trained_dt=data_config.dt
            )
        ),
        "C2": seeded(
            lambda: FieldDensityPhaseSplitStep(
                domain, kinetic_mode="K0", trained_dt=data_config.dt
            )
        ),
        "C3": seeded(
            lambda: FullFieldPhaseSplitStep(domain, trained_dt=data_config.dt)
        ),
    }


def model_architecture(
    name: str,
    *,
    misspecification_identifier: str,
    dial: str,
    value: float,
) -> dict:
    """Checkpoint architecture and Phase 9 provenance for one trained arm."""

    shared = {
        "misspecification_identifier": misspecification_identifier,
        "dial": dial,
        "dial_value": value,
    }
    if name in ("A", "B-loop"):
        architecture = {
            "modes": 16,
            "width": 64,
            "n_layers": 4,
            "use_coordinate_channel": False,
        }
        if name == "B-loop":
            architecture["projection"] = "mass"
    elif name == "C1":
        architecture = {"kinetic_mode": "K0", "local_mode": "L0", "width": 32}
    else:
        architecture = {
            "kinetic_mode": "K0",
            "local_mode": None if name == "C3" else "L0",
            "modes": 16,
            "width": 64,
            "n_layers": 4,
        }
    return {**architecture, **shared}


def _positive_finite(value: object) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def ensure_complete_ratios(by_model: Mapping[str, Mapping[str, object]]) -> dict[str, float]:
    """Compute A-relative ratios only from complete positive finite measurements."""

    missing = [name for name in MODEL_NAMES if name not in by_model]
    if missing:
        raise RuntimeError(
            f"Phase 9 needs complete positive finite model errors; missing {missing}"
        )
    means = {
        name: by_model[name].get("rollout_error_mean") for name in MODEL_NAMES
    }
    invalid = [name for name, value in means.items() if not _positive_finite(value)]
    if invalid:
        raise RuntimeError(
            "Phase 9 needs complete positive finite model errors; invalid "
            f"measurements for {invalid}"
        )
    denominator = float(means["A"])
    return {
        name: float(means[name]) / denominator for name in COMPARISON_MODELS
    }


def aggregate_seed_measurements(
    per_seed: Mapping[int, Mapping[str, Mapping[str, object]]],
    *,
    selected_horizon: int,
) -> dict:
    """Aggregate one dial arm while preserving every observation behind its ratios."""

    if not per_seed:
        raise RuntimeError("Phase 9 needs at least one seed measurement")

    by_model: dict[str, dict] = {}
    for name in MODEL_NAMES:
        observations = []
        for seed in sorted(per_seed):
            try:
                result = per_seed[seed][name]
                metrics = result["metrics"]
                rollout = metrics["rollout"]
                horizon_index = rollout["steps"].index(selected_horizon)
                rollout_error = rollout["relative_error"][horizon_index]
                one_step_error = metrics["one_step_test"]
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                raise RuntimeError(
                    f"Phase 9 seed {seed} model {name} lacks a measurement at "
                    f"horizon {selected_horizon}"
                ) from exc
            if not _positive_finite(rollout_error):
                raise RuntimeError(
                    f"Phase 9 seed {seed} model {name} needs a positive finite "
                    "rollout error"
                )
            if (
                not isinstance(one_step_error, Real)
                or isinstance(one_step_error, bool)
                or not math.isfinite(float(one_step_error))
                or float(one_step_error) < 0.0
            ):
                raise RuntimeError(
                    f"Phase 9 seed {seed} model {name} needs a finite nonnegative "
                    "one-step error"
                )
            observations.append(
                {
                    "seed": seed,
                    "one_step_test": float(one_step_error),
                    "rollout_error": float(rollout_error),
                    "converged": bool(result["converged"]),
                    **(
                        {"history": result["history"]}
                        if "history" in result
                        else {}
                    ),
                    **(
                        {"checkpoint": str(result["checkpoint"])}
                        if "checkpoint" in result
                        else {}
                    ),
                }
            )

        rollout_errors = [entry["rollout_error"] for entry in observations]
        one_step_errors = [entry["one_step_test"] for entry in observations]
        parameters = int(per_seed[next(iter(sorted(per_seed)))][name]["metrics"]["parameters"])
        by_model[name] = {
            "parameters": parameters,
            "one_step_mean": sum(one_step_errors) / len(one_step_errors),
            "one_step_min": min(one_step_errors),
            "one_step_max": max(one_step_errors),
            "rollout_error_mean": sum(rollout_errors) / len(rollout_errors),
            "rollout_error_min": min(rollout_errors),
            "rollout_error_max": max(rollout_errors),
            "converged": [entry["converged"] for entry in observations],
            "per_seed": observations,
        }

    return {
        "selected_horizon": selected_horizon,
        "by_model": by_model,
        "relative_to_A": ensure_complete_ratios(by_model),
    }


def quick_data_config() -> DataConfig:
    """The documented smoke-test distribution; generated in memory only."""

    return DataConfig(
        grid_size=16,
        n_train=2,
        n_val=1,
        n_test=2,
        steps=2,
    )


def validate_shards(
    shards: Mapping[str, TrajectoryShard],
    data_config: DataConfig,
    spec: MisspecificationConfig,
) -> None:
    """Fail closed when a loaded shard set does not match its requested arm."""

    if set(shards) != {"train", "val", "test"}:
        raise RuntimeError("Phase 9 needs complete train/val/test shard sets")
    expected_counts = {
        "train": data_config.n_train,
        "val": data_config.n_val,
        "test": data_config.n_test,
    }
    expected_reference = spec.provenance(data_config)
    for split, shard in shards.items():
        failures = []
        if shard.split != split:
            failures.append(f"split={shard.split!r}")
        if shard.n_trajectories != expected_counts[split]:
            failures.append(f"n_trajectories={shard.n_trajectories}")
        if shard.n_frames != data_config.steps + 1:
            failures.append(f"n_frames={shard.n_frames}")
        if tuple(shard.trajectories.shape[2:]) != data_config.domain.shape:
            failures.append(f"grid={tuple(shard.trajectories.shape[2:])}")
        if shard.dt != data_config.dt:
            failures.append(f"dt={shard.dt}")
        metadata_expectations = {
            "grid_size": data_config.grid_size,
            "substeps": data_config.substeps,
            "steps": data_config.steps,
            "seed": data_config.seed + SPLIT_SEED_OFFSET[split],
        }
        for key, expected in metadata_expectations.items():
            if shard.metadata.get(key) != expected:
                failures.append(f"metadata.{key}={shard.metadata.get(key)!r}")
        if not spec.is_exact and shard.metadata.get("reference") != expected_reference:
            failures.append("metadata.reference does not match requested provenance")
        if failures:
            raise RuntimeError(
                f"Phase 9 shard {split} does not match {spec.identifier(data_config)}: "
                + ", ".join(failures)
            )
    assert_no_leakage(dict(shards))


def prepare_shards(
    data_config: DataConfig,
    spec: MisspecificationConfig,
    *,
    quick: bool,
    data_root: Path = DATA_ROOT,
) -> dict[str, TrajectoryShard]:
    """Generate quick arms in memory or safely load/create production arms."""

    dataset_id = spec.identifier(data_config)
    reference = spec.reference(data_config)
    reference_metadata = None if spec.is_exact else spec.provenance(data_config)
    if quick:
        shards = {
            split: generate_shard(
                data_config,
                split,
                reference=reference,
                reference_metadata=reference_metadata,
            )
            for split in ("train", "val", "test")
        }
        validate_shards(shards, data_config, spec)
        return shards

    paths = shard_paths(data_root, dataset_id)
    existing = {split: path.exists() for split, path in paths.items()}
    if spec.is_exact:
        missing = [str(paths[split]) for split, present in existing.items() if not present]
        if missing:
            raise FileNotFoundError(
                "production shards are required for the exact Phase 9 arm; run "
                "scripts/run_phase1.py first. Missing: " + ", ".join(missing)
            )
    elif any(existing.values()) and not all(existing.values()):
        raise RuntimeError(
            f"refusing to overwrite incomplete Phase 9 shard set {dataset_id}: "
            f"present={[split for split, present in existing.items() if present]}"
        )
    elif not any(existing.values()):
        generated = {}
        for split, path in paths.items():
            print(f"generating {split} for {dataset_id} ...")
            shard = generate_shard(
                data_config,
                split,
                reference=reference,
                reference_metadata=reference_metadata,
            )
            shard.save(path)
            generated[split] = shard
        validate_shards(generated, data_config, spec)
        return generated

    loaded = {split: TrajectoryShard.load(path) for split, path in paths.items()}
    validate_shards(loaded, data_config, spec)
    return loaded


def locate_crossover(by_dial: Mapping[str, Mapping[str, object]], model: str) -> dict:
    """Where a model's A-relative rollout error first reaches 1.0, or a bounded note.

    A located crossover is the result; a bounded one ("no crossover within the swept
    range") is also publishable.  The bounded form is emitted explicitly rather than
    leaving the key absent, so a reader cannot mistake "not found" for "not looked for".
    """

    if not by_dial:
        raise RuntimeError(
            f"Phase 9 {model} needs complete finite A-relative ratios"
        )
    try:
        ordered = sorted(
            (float(dial), measurement["relative_to_A"][model])
            for dial, measurement in by_dial.items()
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Phase 9 {model} needs complete finite A-relative ratios"
        ) from exc
    if any(not _positive_finite(ratio) for _, ratio in ordered):
        raise RuntimeError(
            f"Phase 9 {model} needs complete finite A-relative ratios"
        )

    for dial, ratio in ordered:
        if ratio >= 1.0:
            return {"crossover": dial, "bounded": None}
    dials = [dial for dial, _ in ordered]
    return {
        "crossover": None,
        "bounded": f"no crossover within [{min(dials, default=0)}, "
        f"{max(dials, default=0)}]: {model} still beats A at every swept value",
    }


def make_plots(payload: dict, output) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, dial in zip(axes, ("sigma", "gamma")):
        by_dial = payload["sweeps"].get(dial, {}).get("measurements", {})
        if by_dial:
            values = sorted(float(key) for key in by_dial)
            models = sorted(
                {
                    m
                    for v in values
                    for m in by_dial[str(v)].get("relative_to_A", {})
                }
            )
            for model in models:
                axis.plot(
                    values,
                    [
                        by_dial[str(v)]["relative_to_A"][model]
                        for v in values
                    ],
                    marker="o",
                    label=f"{model} ({payload['parameter_counts'].get(model, 0):,} par)",
                )
        axis.axhline(1.0, color="k", ls="--", label="crossover (= Model A)")
        axis.set_xlabel(dial)
        axis.set_ylabel("rollout error relative to A")
        axis.set_title(f"{dial} sweep")
        axis.grid(True, alpha=0.3)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=7)
    figure.suptitle(f"Phase 9 misspecification  [{payload['data_hash']}]")
    figure.tight_layout()
    figure.savefig(output / "plots" / "phase9_misspecification.png", dpi=150)
    plt.close(figure)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--sigmas", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 1.0]
    )
    parser.add_argument(
        "--gammas", type=float, nargs="+", default=[0.0, 1e-4, 1e-3, 1e-2]
    )
    from spno.phase_workflow import add_workflow_arguments, parse_workflow_args
    add_workflow_arguments(parser)
    return parse_workflow_args(parser, argv)


def run_dial_measurement(
    data_config: DataConfig,
    spec: MisspecificationConfig,
    *,
    dial: str,
    value: float,
    seeds: tuple[int, ...],
    train_config: TrainConfig,
    quick: bool,
    run_id: str,
    data_root: Path = DATA_ROOT,
    checkpoint_root: Path = RESULTS_ROOT,
) -> dict:
    """Train and evaluate one complete Phase 9 dial value."""

    dataset_id = spec.identifier(data_config)
    shards = prepare_shards(
        data_config, spec, quick=quick, data_root=data_root
    )
    scale = field_scale(shards["train"], data_config.domain)
    selected_horizon = min(
        100, *(shard.n_frames - 1 for shard in shards.values())
    )
    checkpoint_identifier = run_identifier(run_id, dial, f"{value:g}")
    per_seed: dict[int, dict[str, dict]] = {}

    print(f"\n--- {dial}={value} ({dataset_id}) ---")
    for seed in seeds:
        config_for_seed = dataclasses.replace(train_config, seed=seed)
        models = build_models(data_config, scale, seed)
        per_seed[seed] = {}
        for name, model in models.items():
            print(f"  training {name} (seed {seed})...")
            history = train_one_step(
                model,
                shards["train"],
                shards["val"],
                data_config,
                config_for_seed,
                verbose=not quick,
            )
            metrics = evaluate_model(
                model,
                shards,
                data_config,
                config_for_seed,
                checkpoints=(selected_horizon,),
                n_rollout=min(100, shards["test"].n_trajectories),
            )
            did_converge = converged(history)
            path = checkpoint_path(
                checkpoint_root,
                "phase9",
                checkpoint_identifier,
                name,
                seed,
            )
            save_checkpoint(
                path,
                model,
                CheckpointMetadata(
                    schema_version=1,
                    model_name=name,
                    data_hash=dataset_id,
                    seed=seed,
                    train_mode="one-step",
                    field_scale=scale,
                    trained_dt=data_config.dt,
                    architecture=model_architecture(
                        name,
                        misspecification_identifier=dataset_id,
                        dial=dial,
                        value=value,
                    ),
                    converged=False if quick else did_converge,
                    best_epoch=history.best_epoch,
                ),
            )
            per_seed[seed][name] = {
                "metrics": metrics,
                "history": history.as_dict(),
                "converged": did_converge,
                "checkpoint": path,
            }

    aggregated = aggregate_seed_measurements(
        per_seed, selected_horizon=selected_horizon
    )
    return {
        "value": value,
        "identifier": dataset_id,
        "reuses_production_shards": spec.is_exact and not quick,
        "config": spec.as_dict(),
        "reference": spec.provenance(data_config),
        "field_scale": scale,
        "seeds": list(seeds),
        "note": "all five models retrained on this shard set; ratios use the "
        "selected-horizon rollout error relative to Model A",
        **aggregated,
    }


def run_sweep(
    data_config: DataConfig,
    args: argparse.Namespace,
    *,
    data_root: Path = DATA_ROOT,
    checkpoint_root: Path = RESULTS_ROOT,
    save=save_run,
) -> dict:
    """Execute both measured dial sweeps with injectable artifact boundaries."""

    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if not args.seeds:
        raise ValueError("at least one seed is required")

    device = pick_device(args.device)
    epochs = 1 if args.quick else args.epochs
    seeds = tuple(args.seeds[:1] if args.quick else args.seeds)
    run_id = run_identifier(
        config_hash(data_config), "misspec", quick=args.quick
    )
    train_config = TrainConfig(
        epochs=epochs,
        batch_size=256,
        learning_rate=1e-3,
        patience=6,
        device=device,
        max_train_pairs=256 if args.quick else None,
    )
    parameter_models = build_models(data_config, scale=1.0, seed=0)
    payload: dict = {
        "phase": 9,
        "data_hash": config_hash(data_config),
        "identifier": run_id,
        "device": device,
        "quick": args.quick,
        "gates": bitwise_gate(data_config),
        "parameter_counts": {
            name: model.parameter_count() for name, model in parameter_models.items()
        },
        "dials": {
            "sigma": "nonlocal nonlinearity: breaks locality but preserves mass",
            "gamma": "weak gain/loss: breaks conservation itself",
            "excluded": "saturable and quintic terms remain representable by C1",
        },
        "config": describe_config(data_config, train_config),
        "sweeps": {
            "sigma": {"measurements": {}},
            "gamma": {"measurements": {}},
        },
        "crossover": {},
    }

    for dial, values in (("sigma", args.sigmas), ("gamma", args.gammas)):
        measurements = payload["sweeps"][dial]["measurements"]
        for raw_value in values:
            value = float(raw_value)
            spec = (
                MisspecificationConfig(nonlocal_sigma=value)
                if dial == "sigma"
                else MisspecificationConfig(gain_loss_gamma=value)
            )
            measurements[str(value)] = run_dial_measurement(
                data_config,
                spec,
                dial=dial,
                value=value,
                seeds=seeds,
                train_config=train_config,
                quick=args.quick,
                run_id=run_id,
                data_root=data_root,
                checkpoint_root=checkpoint_root,
            )

    require_phase_arms(9, ("sigma", "gamma"), payload["sweeps"])
    for dial in ("sigma", "gamma"):
        measurements = payload["sweeps"][dial]["measurements"]
        payload["crossover"][dial] = {
            model: locate_crossover(measurements, model)
            for model in COMPARISON_MODELS
        }
    require_phase_arms(9, ("sigma", "gamma"), payload["sweeps"])

    output = save("phase9", run_id, payload)
    make_plots(payload, output)

    print(f"{'phase':<22}9 -- misspecification sweep")
    print(f"{'data hash':<22}{payload['data_hash']}")
    print(f"{'identifier':<22}{run_id}")
    print(f"{'device':<22}{device}")
    print(f"{'bitwise gate':<22}passed at dial zero")
    print(f"{'sigmas':<22}{' '.join(str(v) for v in args.sigmas)}")
    print(f"{'gammas':<22}{' '.join(str(v) for v in args.gammas)}")
    print(f"{'seeds':<22}{' '.join(str(seed) for seed in seeds)}  epochs {epochs}")
    for name, count in payload["parameter_counts"].items():
        print(f"{'  params ' + name:<22}{count:,}")
    for dial in ("sigma", "gamma"):
        for model, entry in payload["crossover"][dial].items():
            state = (
                entry["crossover"]
                if entry["crossover"] is not None
                else "bounded"
            )
            print(f"{'  crossover ' + dial + ' ' + model:<22}{state}")
    print(f"{'written to':<22}{output}")
    return payload


def main(argv=None) -> dict:
    args = parse_args(argv)
    if args.stage is not None:
        from spno.phase_workflow import run_stage
        return run_stage(9, args)

    data_config = quick_data_config() if args.quick else DataConfig()
    return run_sweep(data_config, args)


if __name__ == "__main__":
    main()
