"""Phases 2 and 3: the FNO baseline and the mass-projected FNO.

Trains three models on identical data with an identical objective, schedule, and
budget, so any difference is attributable to the architecture:

    A         unrestricted FNO
    B-post    the same trained core, with mass projection applied only at evaluation
    B-loop    an FNO trained with the projection inside the forward pass

A and B-post share weights by construction -- B-post *is* A with a projection bolted
on -- which isolates what the projection alone buys.  B-loop is what changes if the
core can learn to rely on it.

Usage::

    python scripts/run_phase23.py [--quick] [--seeds 0 1 2] [--device auto]
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from spno.config import DataConfig, config_hash
from spno.experiments import (
    describe_config,
    evaluate_model,
    arithmetic_mass_floor,
    load_shards,
    pick_device,
    save_run,
)
from spno.models.fno import FNOStepOperator
from spno.models.projected import MassProjectedOperator
from spno.train import TrainConfig, field_scale, train_one_step

CHECKPOINTS = (1, 10, 20, 50, 100, 200)


def build_fno(data_config: DataConfig, scale: float, seed: int) -> FNOStepOperator:
    torch.manual_seed(seed)
    return FNOStepOperator(
        data_config.domain,
        modes=16,
        width=64,
        n_layers=4,
        alpha_range=data_config.alpha_range,
        beta_range=data_config.beta_range,
        field_scale=scale,
        trained_dt=data_config.dt,
    )


def run_seed(
    seed: int, shards, data_config: DataConfig, train_config: TrainConfig, scale: float
) -> dict:
    config = dataclasses.replace(train_config, seed=seed)
    results = {}

    print(f"\n--- seed {seed}: Model A (FNO) ---")
    model_a = build_fno(data_config, scale, seed)
    history_a = train_one_step(
        model_a, shards["train"], shards["val"], data_config, config
    )
    results["A"] = {
        **evaluate_model(model_a, shards, data_config, config, checkpoints=CHECKPOINTS),
        "history": history_a.as_dict(),
    }

    print(f"--- seed {seed}: Model B-post (projection at evaluation only) ---")
    # Same weights as A: isolates the projection itself, with zero training difference.
    model_b_post = MassProjectedOperator(model_a)
    results["B-post"] = {
        **evaluate_model(
            model_b_post, shards, data_config, config, checkpoints=CHECKPOINTS
        ),
        "history": history_a.as_dict(),
        "note": "shares Model A's weights; projection applied only at evaluation",
    }

    print(f"--- seed {seed}: Model B-loop (projection inside the forward pass) ---")
    model_b_loop = MassProjectedOperator(build_fno(data_config, scale, seed))
    history_b = train_one_step(
        model_b_loop, shards["train"], shards["val"], data_config, config
    )
    results["B-loop"] = {
        **evaluate_model(
            model_b_loop, shards, data_config, config, checkpoints=CHECKPOINTS
        ),
        "history": history_b.as_dict(),
    }
    return results


def aggregate(per_seed: dict[int, dict]) -> dict:
    models = sorted(next(iter(per_seed.values())).keys())
    summary = {}
    for name in models:
        one_step = [per_seed[s][name]["one_step_test"] for s in per_seed]
        entry = {
            "one_step_mean": sum(one_step) / len(one_step),
            "one_step_min": min(one_step),
            "one_step_max": max(one_step),
            "parameters": per_seed[next(iter(per_seed))][name]["parameters"],
            "rollout": {},
        }
        for index, step in enumerate(
            per_seed[next(iter(per_seed))][name]["rollout"]["steps"]
        ):
            errors = [per_seed[s][name]["rollout"]["relative_error"][index] for s in per_seed]
            masses = [per_seed[s][name]["rollout"]["mass_drift"][index] for s in per_seed]
            energies = [per_seed[s][name]["rollout"]["energy_drift"][index] for s in per_seed]
            entry["rollout"][str(step)] = {
                "error_mean": sum(errors) / len(errors),
                "error_min": min(errors),
                "error_max": max(errors),
                "mass_drift_mean": sum(masses) / len(masses),
                "energy_drift_mean": sum(energies) / len(energies),
            }
        entry["energy_classification"] = [
            per_seed[s][name]["rollout"]["energy_classification"] for s in per_seed
        ]
        entry["diverged_at"] = [
            per_seed[s][name]["rollout"]["diverged_at"] for s in per_seed
        ]
        summary[name] = entry
    return summary


def make_plots(per_seed, summary, payload, output):
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    colors = {"A": "tab:blue", "B-post": "tab:orange", "B-loop": "tab:green"}

    for name, entry in summary.items():
        steps = [int(s) for s in entry["rollout"]]
        axes[0].plot(
            steps, [entry["rollout"][str(s)]["error_mean"] for s in steps],
            "o-", color=colors[name], label=name,
        )
        axes[1].plot(
            steps,
            [max(entry["rollout"][str(s)]["mass_drift_mean"], 1e-17) for s in steps],
            "o-", color=colors[name], label=name,
        )
        axes[2].plot(
            steps, [entry["rollout"][str(s)]["energy_drift_mean"] for s in steps],
            "o-", color=colors[name], label=name,
        )

    floor = payload["mass_floor"]["float64_cpu"]["mean"]
    axes[1].axhline(
        max(floor, 1e-16), color="k", ls="--",
        label=f"float64 solver floor ({floor:.1e})",
    )
    eps_split = payload.get("eps_split")
    if eps_split:
        axes[0].axhline(eps_split, color="k", ls=":", label=f"eps_split ({eps_split:.1e})")

    axes[0].set(xlabel="rollout step", ylabel="relative L2", title="Rollout error")
    axes[1].set(xlabel="rollout step", ylabel="relative mass drift", title="Mass drift")
    axes[2].set(xlabel="rollout step", ylabel="relative energy drift", title="Energy drift")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)

    figure.suptitle(
        f"Phases 2-3: FNO vs mass-projected FNO  [{payload['data_hash']}]  "
        f"{len(per_seed)} seeds"
    )
    figure.tight_layout()
    figure.savefig(output / "plots" / "phase23_rollout.png", dpi=150)
    plt.close(figure)


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    data_config = DataConfig()
    device = pick_device(args.device)
    train_config = TrainConfig(
        epochs=3 if args.quick else 25,
        batch_size=256,
        learning_rate=1e-3,
        patience=6,
        device=device,
    )
    seeds = args.seeds[:1] if args.quick else args.seeds

    shards = load_shards(data_config)
    scale = field_scale(shards["train"], data_config.domain)
    print(f"device={device}  field_scale={scale:.4f}  seeds={seeds}")

    per_seed = {
        seed: run_seed(seed, shards, data_config, train_config, scale) for seed in seeds
    }
    summary = aggregate(per_seed)

    payload = {
        **describe_config(data_config, train_config),
        "seeds": seeds,
        "field_scale": scale,
        "mass_floor": arithmetic_mass_floor(shards, data_config),
        "eps_split": 6.239e-05,  # measured in Phase 0
        "summary": summary,
        "per_seed": per_seed,
    }
    identifier = config_hash(data_config)
    output = save_run("phase23", identifier, payload)
    make_plots(per_seed, summary, payload, output)

    print(f"\n=== Phases 2-3 [{identifier}] ===")
    mf = payload["mass_floor"]
    print(
        f"solver mass-drift floor, 100 steps: "
        f"float64/cpu {mf['float64_cpu']['mean']:.2e}   float32/cpu {mf['float32_cpu']['mean']:.2e}"
    )
    print("rollout + invariants evaluated in float64 on cpu (weights widened)")
    print(f"eps_split (Phase 0)                        : {payload['eps_split']:.3e}\n")
    header = f"{'model':8s} {'params':>9s} {'1-step':>10s} {'roll@20':>10s} {'roll@100':>10s} {'mass@100':>10s} {'energy@100':>11s}"
    print(header)
    print("-" * len(header))
    for name, entry in summary.items():
        roll = entry["rollout"]
        print(
            f"{name:8s} {entry['parameters']:9,d} {entry['one_step_mean']:10.3e} "
            f"{roll['20']['error_mean']:10.3e} {roll['100']['error_mean']:10.3e} "
            f"{roll['100']['mass_drift_mean']:10.3e} "
            f"{roll['100']['energy_drift_mean']:11.3e}"
        )
    print(f"\nwritten to: {output}")
    return payload


if __name__ == "__main__":
    main()
