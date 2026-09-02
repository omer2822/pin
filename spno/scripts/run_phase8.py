"""Phase 8: checkpoint-backed resolution transfer and robustness measurements.

The two resolution arms intentionally answer different questions. A band-limited
coarse test set is resampled without adding information; it checks whether a trained
operator can be evaluated on a finer mesh. The high-k arm generates a distinct fine
test set with energy above the coarse-grid Nyquist; that is spectral extrapolation and
is reported separately rather than being presented as ordinary resolution transfer.

Usage:
    python scripts/run_phase8.py [--quick] [--seeds 0 1 2] [--epochs 60]
                                 [--device auto] [--grid 128]
                                 [--noise 0.0 0.001 0.01 0.05]
                                 [--train-fractions 0.05 0.1 0.25 0.5 1.0]
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from spno.checkpoints import checkpoint_path
from spno.config import DataConfig, config_hash
from spno.data.corruption import add_field_noise, add_potential_noise
from spno.data.datasets import TrajectoryShard, generate_shard
from spno.domain import PeriodicDomain
from spno.evaluation.payloads import require_phase_arms
from spno.evaluation.resolution import grid_dependence, rebind_domain, spectral_resample
from spno.evaluation.rollout import evaluate_rollout
from spno.experiments import (
    ROOT,
    converged,
    describe_config,
    evaluate_model,
    load_shards,
    pick_device,
    run_identifier,
    save_run,
)
from spno.losses.relative_l2 import relative_l2_per_sample
from spno.models.fno import FNOStepOperator
from spno.models.split_learned import DensityPhaseSplitStep
from spno.precision import widen_to_double
from spno.seeding import seed_everything
from spno.train import TrainConfig, field_scale, train_one_step


BAND_LIMITED_CAVEAT = (
    "NOT evidence of resolution transfer: an FNO only touches k <= n_modes, and "
    "upsampling a band-limited field leaves those modes unchanged, so its output barely "
    "moves by construction. Quote this sentence beside the numbers."
)
NEW_HIGH_K_CAVEAT = (
    "This is spectral extrapolation (G4 in disguise): the fine test contains energy "
    "above the training-grid Nyquist, so it is not an ordinary resolution-transfer claim."
)
MODEL_NAMES = ("A", "C1")


def quick_checkpoint_config() -> DataConfig:
    """The production spatial distribution named by Phase 2--5 checkpoints."""

    return DataConfig()


def quick_data_config() -> DataConfig:
    """In-memory smoke distribution preserving the checkpoint's coarse grid."""

    return replace(quick_checkpoint_config(), n_train=10, n_val=1, n_test=2, steps=2)


def load_phase8_models(
    checkpoint_root: Path,
    data_config: DataConfig,
    *,
    seeds: list[int],
    quick: bool = False,
    allow_budget_bound: bool = False,
) -> dict[int, dict[str, object]]:
    """Restore the production A/C1 weights used for every Phase 8 measurement."""

    # Phase 6 already owns the checkpoint-schema-aware constructors. Reusing it keeps
    # Phase 8 from guessing architecture defaults when restoring historical runs.
    try:
        from scripts.run_phase6 import _load_model
    except ModuleNotFoundError:  # ``python scripts/run_phase8.py`` exposes scripts itself.
        from run_phase6 import _load_model

    phase23_identifier = run_identifier(config_hash(data_config), quick=quick)
    phase45_identifier = run_identifier(
        config_hash(data_config), "one-step", "K0L0", quick=quick
    )
    commands = {
        "A": "python scripts/run_phase23.py "
        f"{'--quick ' if quick else ''}--seeds {' '.join(map(str, seeds))}",
        "C1": "python scripts/run_phase45.py "
        f"{'--quick ' if quick else ''}--mode one-step --kinetic K0 --local L0 "
        f"--seeds {' '.join(map(str, seeds))}",
    }
    locations = {
        "A": ("phase23", phase23_identifier),
        "C1": ("phase45", phase45_identifier),
    }

    restored: dict[int, dict[str, object]] = {}
    for seed in seeds:
        restored[seed] = {}
        for name, (family, identifier) in locations.items():
            path = checkpoint_path(checkpoint_root, family, identifier, name, seed)
            restored[seed][name] = _load_model(
                path,
                data_config,
                expected_name=name,
                expected_data_hash=config_hash(data_config),
                allow_budget_bound=allow_budget_bound,
                creation_command=commands[name],
            )
    return restored


def resample_shard(
    shard: TrajectoryShard, source: PeriodicDomain, target: PeriodicDomain
) -> TrajectoryShard:
    """Move every frame and potential to a fine grid as a strictly band-limited arm."""

    def discard_source_nyquist(field: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Remove the aliased source-Nyquist coefficient instead of inventing its phase."""

        if source.shape[0] % 2:
            return field.clone(), 0.0
        spectrum = torch.fft.fftn(field, dim=source.spatial_axes)
        total = torch.abs(spectrum).pow(2).sum().clamp_min(1e-300)
        removed = torch.abs(spectrum[..., source.shape[0] // 2]).pow(2).sum()
        spectrum[..., source.shape[0] // 2] = 0
        filtered = torch.fft.ifftn(spectrum, dim=source.spatial_axes)
        if not field.is_complex():
            filtered = filtered.real
        return filtered, float(removed / total)

    trajectories, trajectory_nyquist_energy = discard_source_nyquist(shard.trajectories)
    potential, potential_nyquist_energy = discard_source_nyquist(shard.potential)

    return TrajectoryShard(
        trajectories=spectral_resample(trajectories, source, target),
        potential=spectral_resample(potential, source, target).real,
        alpha=shard.alpha.clone(),
        beta=shard.beta.clone(),
        trajectory_ids=shard.trajectory_ids.clone(),
        dt=shard.dt,
        split=shard.split,
        metadata={
            **shard.metadata,
            "phase8_resampled_from_grid": source.shape[0],
            "phase8_resampled_to_grid": target.shape[0],
            "phase8_source_nyquist_energy_removed": {
                "trajectories": trajectory_nyquist_energy,
                "potential": potential_nyquist_energy,
            },
        },
    )


def evaluate_on_fine_grid(
    model,
    coarse_domain: PeriodicDomain,
    fine_domain: PeriodicDomain,
    fine_shards: Mapping[str, TrajectoryShard],
    fine_config: DataConfig,
    train_config: TrainConfig,
    *,
    selected_horizon: int | None = None,
    evaluator: Callable = evaluate_model,
) -> dict:
    """Evaluate after rebinding, always restoring the trained model's coarse domain."""

    try:
        rebind_domain(model, fine_domain)
        model.to(train_config.device)
        kwargs = {}
        if selected_horizon is not None:
            kwargs["checkpoints"] = (selected_horizon,)
        return evaluator(model, dict(fine_shards), fine_config, train_config, **kwargs)
    finally:
        rebind_domain(model, coarse_domain)


def selected_horizon(data_config: DataConfig) -> int:
    return min(100, data_config.steps)


def _measurement(metrics: Mapping[str, object], horizon: int) -> dict[str, float]:
    try:
        rollout = metrics["rollout"]
        index = rollout["steps"].index(horizon)
        one_step = float(metrics["one_step_test"])
        rollout_error = float(rollout["relative_error"][index])
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise RuntimeError(
            f"missing Phase 8 measurement at horizon {horizon}"
        ) from error
    if not all(torch.isfinite(torch.tensor(value)) for value in (one_step, rollout_error)):
        raise RuntimeError("Phase 8 produced a non-finite error")
    return {"one_step_error": one_step, "rollout_error": rollout_error}


def aggregate_measurements(
    per_seed: Mapping[int, Mapping[str, Mapping[str, object]]], horizon: int
) -> dict[str, dict]:
    """Average paired checkpoint measurements while retaining the seed observations."""

    if not per_seed:
        raise RuntimeError("Phase 8 needs at least one seed")
    by_model: dict[str, dict] = {}
    for name in MODEL_NAMES:
        observations = []
        for seed in sorted(per_seed):
            measurement = _measurement(per_seed[seed][name], horizon)
            observations.append({"seed": seed, **measurement})
        one_step = [entry["one_step_error"] for entry in observations]
        rollout = [entry["rollout_error"] for entry in observations]
        by_model[name] = {
            "one_step_error": sum(one_step) / len(one_step),
            "rollout_error": sum(rollout) / len(rollout),
            "one_step_error_min": min(one_step),
            "one_step_error_max": max(one_step),
            "rollout_error_min": min(rollout),
            "rollout_error_max": max(rollout),
            "per_seed": observations,
        }
    return by_model


def resolution_sections(
    models_by_seed: Mapping[int, Mapping[str, object]],
    shards: Mapping[str, TrajectoryShard],
    data_config: DataConfig,
    grid: int,
    train_config: TrainConfig,
) -> dict:
    """Measure full-trajectory band transfer and genuinely new fine-grid high-k data."""

    coarse = data_config.domain
    if grid <= coarse.shape[0]:
        raise ValueError("--grid must be finer than the checkpoint's training grid")
    fine_config = replace(data_config, grid_size=grid)
    fine = fine_config.domain
    horizon = selected_horizon(fine_config)
    band_shards = {
        split: resample_shard(shard, coarse, fine) for split, shard in shards.items()
    }

    def evaluate_all(fine_test_shards: Mapping[str, TrajectoryShard]) -> dict[int, dict]:
        return {
            seed: {
                name: evaluate_on_fine_grid(
                    model,
                    coarse,
                    fine,
                    fine_test_shards,
                    fine_config,
                    train_config,
                    selected_horizon=horizon,
                )
                for name, model in models.items()
            }
            for seed, models in models_by_seed.items()
        }

    band = evaluate_all(band_shards)
    coarse_nyquist = data_config.max_wave_number
    occupied_bandwidth = min(fine_config.max_wave_number - 1, coarse_nyquist + 8)
    if occupied_bandwidth <= coarse_nyquist:
        raise RuntimeError("fine grid cannot represent a band above the coarse Nyquist")
    high_config = replace(
        fine_config,
        initial_bandwidth=occupied_bandwidth,
        seed=data_config.seed + 8_008,
    )
    high = evaluate_all({"test": generate_shard(high_config, "test")})

    first_models = models_by_seed[min(models_by_seed)]
    return {
        "resolution_band_limited": {
            "caveat": BAND_LIMITED_CAVEAT,
            "selected_horizon": horizon,
            "full_trajectory_resample": True,
            "by_model": aggregate_measurements(band, horizon),
            "grid_dependence": {
                name: grid_dependence(model) for name, model in first_models.items()
            },
        },
        "resolution_new_high_k": {
            "caveat": NEW_HIGH_K_CAVEAT,
            "cross_reference": "Phase 6 G4 bandwidth arms",
            "selected_horizon": horizon,
            "coarse_nyquist": coarse_nyquist,
            "fine_initial_bandwidth": occupied_bandwidth,
            "by_model": aggregate_measurements(high, horizon),
        },
    }


@torch.no_grad()
def evaluate_corrupted_inputs(
    model,
    shard: TrajectoryShard,
    data_config: DataConfig,
    *,
    field: torch.Tensor,
    potential: torch.Tensor,
    horizon: int,
) -> dict[str, object]:
    """Score one corrupted input channel against clean one-step and rollout targets."""

    domain = data_config.domain
    evaluation_model = widen_to_double(model, device="cpu").eval()
    clean = shard.trajectories.to(dtype=torch.complex128)
    potential = potential.to(dtype=torch.float64)
    alpha = shard.alpha.to(dtype=torch.float64)
    beta = shard.beta.to(dtype=torch.float64)
    field = field.to(dtype=torch.complex128)
    prediction = evaluation_model(field, potential, alpha, beta, data_config.dt)
    one_step = float(relative_l2_per_sample(prediction, clean[:, 1], domain).mean())
    rollout = evaluate_rollout(
        evaluation_model,
        domain,
        field,
        clean,
        potential,
        alpha,
        beta,
        data_config.dt,
        checkpoints=(horizon,),
    )
    return {"one_step_test": one_step, "rollout": rollout.as_dict()}


def build_sample_efficiency_models(
    data_config: DataConfig, scale: float, seed: int
) -> dict[str, object]:
    """Construct paired A/C1 training runs with the train-split normalization."""

    def seeded(build):
        seed_everything(seed)
        return build()

    return {
        "A": seeded(
            lambda: FNOStepOperator(
                data_config.domain,
                modes=16,
                width=64,
                n_layers=4,
                alpha_range=data_config.alpha_range,
                beta_range=data_config.beta_range,
                field_scale=scale,
                trained_dt=data_config.dt,
            )
        ),
        "C1": seeded(
            lambda: DensityPhaseSplitStep(
                data_config.domain, kinetic_mode="K0", trained_dt=data_config.dt
            )
        ),
    }


def _numeric_model_errors(by_model: Mapping[str, Mapping[str, object]]) -> dict:
    """Keep the gate's robustness subtree entirely numeric and finite."""

    return {
        name: {
            "one_step_error": float(metrics["one_step_error"]),
            "rollout_error": float(metrics["rollout_error"]),
        }
        for name, metrics in by_model.items()
    }


def robustness_section(
    models_by_seed: Mapping[int, Mapping[str, object]],
    shards: Mapping[str, TrajectoryShard],
    data_config: DataConfig,
    train_config: TrainConfig,
    args: argparse.Namespace,
) -> dict:
    """Measure separate noisy channels and retrained sample-efficiency curves."""

    horizon = selected_horizon(data_config)
    domain = data_config.domain
    test = shards["test"]
    field_noise, potential_noise, model_errors = {}, {}, {
        "field_noise": {}, "potential_noise": {}, "sample_efficiency": {}
    }
    for level in args.noise:
        field_per_seed, potential_per_seed = {}, {}
        field_calibration, potential_calibration = [], []
        for seed, models in models_by_seed.items():
            field_generator = torch.Generator().manual_seed(10_000 + seed)
            noisy_field = add_field_noise(
                test.trajectories[:, 0], domain, level=level, generator=field_generator
            )
            potential_generator = torch.Generator().manual_seed(20_000 + seed)
            noisy_potential = add_potential_noise(
                test.potential, level=level, generator=potential_generator
            )
            field_calibration.append(
                float(
                    (
                        torch.abs(noisy_field - test.trajectories[:, 0]).pow(2).sum(-1).sqrt()
                        / torch.abs(test.trajectories[:, 0]).pow(2).sum(-1).sqrt()
                    ).mean()
                )
            )
            potential_calibration.append(float((noisy_potential - test.potential).abs().max()))
            field_per_seed[seed] = {
                name: evaluate_corrupted_inputs(
                    model,
                    test,
                    data_config,
                    field=noisy_field,
                    potential=test.potential,
                    horizon=horizon,
                )
                for name, model in models.items()
            }
            potential_per_seed[seed] = {
                name: evaluate_corrupted_inputs(
                    model,
                    test,
                    data_config,
                    field=test.trajectories[:, 0],
                    potential=noisy_potential,
                    horizon=horizon,
                )
                for name, model in models.items()
            }
        key = str(level)
        field_summary = aggregate_measurements(field_per_seed, horizon)
        potential_summary = aggregate_measurements(potential_per_seed, horizon)
        field_noise[key] = {
            "level": level,
            "achieved_relative": sum(field_calibration) / len(field_calibration),
            "by_model": field_summary,
        }
        potential_noise[key] = {
            "level": level,
            "max_absolute": sum(potential_calibration) / len(potential_calibration),
            "by_model": potential_summary,
        }
        model_errors["field_noise"][key] = _numeric_model_errors(field_summary)
        model_errors["potential_noise"][key] = _numeric_model_errors(potential_summary)

    total_pairs = shards["train"].n_trajectories * (shards["train"].n_frames - 1)
    sample_efficiency = {}
    scale = field_scale(shards["train"], domain)
    for fraction in args.train_fractions:
        pair_budget = int(total_pairs * fraction)
        per_seed: dict[int, dict] = {}
        histories: dict[str, dict] = {}
        for seed in args.seeds:
            models = build_sample_efficiency_models(data_config, scale, seed)
            per_seed[seed] = {}
            for name, model in models.items():
                history = train_one_step(
                    model,
                    shards["train"],
                    shards["val"],
                    data_config,
                    replace(train_config, seed=seed, max_train_pairs=pair_budget),
                    verbose=False,
                )
                model.to(train_config.device)
                per_seed[seed][name] = evaluate_model(
                    model,
                    dict(shards),
                    data_config,
                    train_config,
                    checkpoints=(horizon,),
                )
                histories[f"{seed}:{name}"] = {
                    "converged": converged(history), "history": history.as_dict()
                }
        summary = aggregate_measurements(per_seed, horizon)
        key = str(fraction)
        sample_efficiency[key] = {
            "fraction": fraction,
            "max_train_pairs": pair_budget,
            "by_model": summary,
            "training": histories,
        }
        model_errors["sample_efficiency"][key] = _numeric_model_errors(summary)

    return {
        "separately_swept": "field and potential noise are never applied jointly; "
        "both retain clean targets for the errors reported here",
        "selected_horizon": horizon,
        "field_noise": field_noise,
        "potential_noise": potential_noise,
        "sample_efficiency": sample_efficiency,
        "model_errors": model_errors,
    }


def make_plots(payload: Mapping[str, object], output: Path) -> None:
    plots = output / "plots"
    resolution = payload["resolution_band_limited"], payload["resolution_new_high_k"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, (title, section) in zip(
        axes,
        (("band-limited upsample (not transfer evidence)", resolution[0]),
         ("new high-k (spectral extrapolation)", resolution[1])),
    ):
        values = section["by_model"]
        axis.bar(list(values), [values[name]["rollout_error"] for name in values])
        axis.set_title(title)
        axis.set_ylabel("relative rollout error")
        axis.grid(axis="y", alpha=0.3)
    figure.suptitle(f"Phase 8 resolution  [{payload['data_hash']}]")
    figure.tight_layout()
    figure.savefig(plots / "phase8_resolution.png", dpi=150)
    plt.close(figure)

    robustness = payload["robustness"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, key, title in (
        (axes[0], "field_noise", "field-noise rollout error"),
        (axes[1], "potential_noise", "potential-noise rollout error"),
    ):
        levels = sorted(float(value) for value in robustness[key])
        for name in MODEL_NAMES:
            axis.plot(
                levels,
                [robustness[key][str(level)]["by_model"][name]["rollout_error"] for level in levels],
                marker="o",
                label=name,
            )
        axis.set_title(title)
        axis.set_xlabel("noise level")
        axis.set_ylabel("relative rollout error")
        axis.grid(True, alpha=0.3)
        axis.legend()
    efficiency = robustness["sample_efficiency"]
    fractions = sorted(float(value) for value in efficiency)
    for name in MODEL_NAMES:
        axes[2].plot(
            fractions,
            [efficiency[str(fraction)]["by_model"][name]["rollout_error"] for fraction in fractions],
            marker="o",
            label=name,
        )
    axes[2].set_title("sample-efficiency rollout error")
    axes[2].set_xlabel("training fraction")
    axes[2].set_ylabel("relative rollout error")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    figure.suptitle(f"Phase 8 robustness  [{payload['data_hash']}]")
    figure.tight_layout()
    figure.savefig(plots / "phase8_robustness.png", dpi=150)
    plt.close(figure)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.001, 0.01, 0.05])
    parser.add_argument(
        "--train-fractions", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5, 1.0]
    )
    return parser.parse_args(argv)


def prepare_shards(data_config: DataConfig, *, quick: bool) -> dict[str, TrajectoryShard]:
    """Quick runs are generated in memory; production runs only read canonical shards."""

    if quick:
        return {split: generate_shard(data_config, split) for split in ("train", "val", "test")}
    return load_shards(data_config)


def main(argv=None) -> dict:
    args = parse_args(argv)
    if args.quick:
        # A smoke run verifies every arm without multiplying an already complete
        # measurement suite; its checkpoint-compatible coarse grid stays unchanged.
        args.noise = [0.0, 0.1]
        args.train_fractions = [0.1, 1.0]
    checkpoint_config = quick_checkpoint_config() if args.quick else DataConfig()
    data_config = quick_data_config() if args.quick else checkpoint_config
    device = pick_device(args.device)
    train_config = TrainConfig(
        epochs=1 if args.quick else args.epochs,
        device=device,
        patience=1 if args.quick else 5,
    )
    identifier = run_identifier(config_hash(data_config), f"N{args.grid}", quick=args.quick)
    shards = prepare_shards(data_config, quick=args.quick)
    models_by_seed = load_phase8_models(
        ROOT / "results",
        checkpoint_config,
        seeds=args.seeds,
        quick=args.quick,
        allow_budget_bound=args.quick,
    )

    payload: dict = {
        "phase": 8,
        "data_hash": config_hash(data_config),
        "checkpoint_data_hash": config_hash(checkpoint_config),
        "identifier": identifier,
        "device": device,
        "grid": args.grid,
        "quick": args.quick,
        "config": describe_config(data_config, train_config),
        "checkpoint_models": list(MODEL_NAMES),
    }
    payload.update(
        resolution_sections(models_by_seed, shards, data_config, args.grid, train_config)
    )
    payload["robustness"] = robustness_section(
        models_by_seed, shards, data_config, train_config, args
    )
    require_phase_arms(
        8,
        ("resolution_band_limited", "resolution_new_high_k", "robustness"),
        payload,
    )

    output = save_run("phase8", identifier, payload)
    make_plots(payload, output)
    print(f"{'phase':<22}8 -- resolution and robustness")
    print(f"{'data hash':<22}{payload['data_hash']}")
    print(f"{'checkpoint hash':<22}{payload['checkpoint_data_hash']}")
    print(f"{'identifier':<22}{identifier}")
    print(f"{'device':<22}{device}")
    print(f"{'grid':<22}{data_config.grid_size} -> {args.grid}")
    print(f"{'written to':<22}{output}")
    return payload


if __name__ == "__main__":
    main()
