"""Phase 7a: the PINO baseline -- a soft PDE-residual penalty, swept over lambda.

**The confound, stated where the numbers are produced.**  The midpoint residual averages
the nonlinearity in the Delfour--Fortin--Payre form, which makes the scheme discretely
mass-conserving: for a plane wave the update factor is the Cayley transform
``(1 - i omega dt/2)/(1 + i omega dt/2)``, whose modulus is **exactly** one.  So this
physics loss smuggles in the very invariant the study compares methods on.  That is not
a reason to skip 7a -- it is precisely what the physics-loss / projection / architecture
distinction exists to expose -- but it must be written next to every number below.

**Report the whole sweep, never a tuned lambda.**  Which lambda wins is itself the
result; a single tuned value hides whether the physics term helped at all.  ``lambda=0``
is the control arm and is bit-identical to Phase 2-3's Model A.

Usage:
    python scripts/run_phase7.py [--quick] [--seeds 0 1 2] [--epochs 60]
                                 [--device auto] [--lambdas 0.0 0.01 0.1 1 10]

The explicit --stage workflow evaluates A, A+PDE, B-loop, C1 and C1+PDE.
The legacy invocation without --stage retains the A-only sweep.

Nothing is trained here without an explicit run; the deferred Phase 2-5 convergence debt
must be cleared first or every row describes the epoch cap rather than the model.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spno.config import DataConfig, config_hash
from spno.experiments import (
    budget_warning,
    converged,
    describe_config,
    evaluate_model,
    load_shards,
    pick_device,
    run_identifier,
    save_run,
)
from spno.models.fno import FNOStepOperator
from spno.train import TrainConfig, field_scale, train_pino

#: Measured in Phase 0 on the production config; the one-step floor for a split step.
EPS_SPLIT = 6.239e-05
#: Phase 2-3 Model A, for the reference lines. Budget-bound -- see the payload note.
MODEL_A_ONE_STEP = 6.94e-04
MODEL_A_ROLLOUT_100 = 2.77e-02


def build_pino_model(
    data_config: DataConfig, scale: float, seed: int
) -> FNOStepOperator:
    """Build the controlled PINO arm for ``seed`` with train-shard normalization."""

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


def aggregate_pino_seed_metrics(per_seed: list[dict], horizon: int = 100) -> dict:
    """Aggregate test metrics at the recorded horizon, with spread across seeds."""

    if not per_seed:
        raise ValueError("At least one seed is required for the PINO comparison")
    values = {"one_step": [], "rollout": [], "mass_drift": [], "energy_drift": []}
    for entry in per_seed:
        metrics = entry["metrics"]
        rollout = metrics["rollout"]
        if horizon not in rollout["steps"]:
            raise RuntimeError(f"Rollout did not reach step {horizon}; inspect per-seed divergence")
        index = rollout["steps"].index(horizon)
        values["one_step"].append(metrics["one_step_test"])
        for key, source in (("rollout", "relative_error"), ("mass_drift", "mass_drift"),
                            ("energy_drift", "energy_drift")):
            values[key].append(rollout[source][index])
    result = {"selected_horizon": horizon, "seed_count": len(per_seed),
              "converged": [entry["converged"] for entry in per_seed]}
    for key, series in values.items():
        if any(not math.isfinite(v) or v < 0 for v in series):
            raise RuntimeError(f"Invalid {key} metrics; cannot publish an incomplete comparison")
        result.update({f"{key}_mean": statistics.mean(series),
                       f"{key}_std": statistics.stdev(series) if len(series) > 1 else 0.,
                       f"{key}_min": min(series), f"{key}_max": max(series)})
    # Preserve historical keys only when they really describe horizon 100.
    if horizon == 100:
        result["mass_drift_100_mean"] = result["mass_drift_mean"]
        result["energy_drift_100_mean"] = result["energy_drift_mean"]
    return result


def make_comparison_plots(payload: dict, output) -> None:
    """Display both full sweeps, the projection baseline, and a reusable CSV table."""

    horizon = payload["selected_horizon"]
    panels = (("one_step", "One-step relative L2"),
              ("rollout", f"Rollout relative L2 @{horizon}"),
              ("mass_drift", f"Mass drift @{horizon}"),
              ("energy_drift", f"Energy drift @{horizon}"))
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    rows = []
    for name, sweep in payload["sweeps"].items():
        weights = sorted(float(w) for w in sweep)
        for weight in weights:
            entry = sweep[str(weight)]
            rows.append({"arm": name if weight == 0 else f"{name}+PDE",
                         "lambda": weight, "selected_horizon": horizon,
                         "seed_count": entry["seed_count"], "parameters": entry["parameter_count"],
                         "converged_seeds": sum(entry["converged"]),
                         **{f"{key}_{stat}": entry[f"{key}_{stat}"]
                            for key, _ in panels for stat in ("mean", "std")}})
        for axis, (key, title) in zip(axes.flat, panels):
            means = [sweep[str(w)][f"{key}_mean"] for w in weights]
            stds = [sweep[str(w)][f"{key}_std"] for w in weights]
            line, = axis.plot(weights, means, marker="o", label=f"{name} / {name}+PDE")
            axis.fill_between(weights, [max(0., m-s) for m, s in zip(means, stds)],
                              [m+s for m, s in zip(means, stds)], color=line.get_color(), alpha=.12)
    baseline = payload["baselines"]["B-loop"]
    rows.append({"arm": "B-loop", "lambda": 0., "selected_horizon": horizon,
                 "seed_count": baseline["seed_count"], "parameters": baseline["parameter_count"],
                 "converged_seeds": sum(baseline["converged"]),
                 **{f"{key}_{stat}": baseline[f"{key}_{stat}"]
                    for key, _ in panels for stat in ("mean", "std")}})
    positive = [w for w in weights if w > 0]
    for axis, (key, title) in zip(axes.flat, panels):
        mean, std = baseline[f"{key}_mean"], baseline[f"{key}_std"]
        axis.axhline(mean, color="tab:green", ls="--", label="B-loop (projection)")
        axis.axhspan(max(0., mean-std), mean+std, color="tab:green", alpha=.08)
        axis.set_xscale("symlog", linthresh=min(positive)/2 if positive else .01)
        axis.set_xticks(weights, [f"{w:g}" for w in weights])
        values = [entry[f"{key}_mean"] for sweep in payload["sweeps"].values() for entry in sweep.values()]
        positive_values = [v for v in values + [mean] if v > 0]
        upper = max(row[f"{key}_mean"] + row[f"{key}_std"] for row in rows)
        # Keep zero visible without adding meaningless negative-error decades or
        # stretching an accuracy panel all the way to machine precision.
        axis.set_yscale("symlog", linthresh=min(positive_values)/10 if positive_values else 1e-15)
        axis.set_ylim(0., upper * 1.5 if upper else 1e-15)
        axis.set_xlabel("PDE weight lambda (0 = no residual)")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=.3)
        axis.legend(fontsize=8)
    status = "EXPLORATORY" if payload["exploratory"] else "converged checkpoints"
    if not payload["comparison_complete"]:
        status += "; controls only — no positive lambda"
    figure.suptitle(f"Phase 7a: soft physics, projection, and structured architecture — {status}")
    figure.text(.5, .015, "Mean ± sample SD across seeds (one seed has no spread estimate). "
                "CN conserves mass at zero residual; finite-dt CN and C1 split-step dynamics differ.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .04, 1, .95))
    figure.savefig(output / "plots" / "phase7_lambda_sweep.png", dpi=150)
    plt.close(figure)
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(payload: dict, output) -> None:
    """One figure: error, mass drift and energy drift against lambda on a log axis."""

    if "sweeps" in payload:
        make_comparison_plots(payload, output)
        return

    sweep = payload["sweep"]
    lambdas = sorted(float(key) for key in sweep)
    if not lambdas:
        return

    def series(getter):
        values = []
        for value in lambdas:
            entry = sweep[str(value)]
            values.append(getter(entry))
        return values

    # log-x cannot show lambda=0; plot it at a decade below the smallest positive value.
    positive = [v for v in lambdas if v > 0]
    floor = min(positive) / 10 if positive else 1e-3
    x = [v if v > 0 else floor for v in lambdas]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = (
        ("one-step relative L2", lambda e: e.get("one_step_mean")),
        ("mass drift @100", lambda e: e.get("mass_drift_100_mean")),
        ("energy drift @100", lambda e: e.get("energy_drift_100_mean")),
    )
    for axis, (title, getter) in zip(axes, panels):
        values = series(getter)
        if any(v is not None for v in values):
            axis.plot(x, [v if v is not None else float("nan") for v in values],
                      marker="o", label="PINO")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("lambda  (leftmost point is lambda=0)")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.3)

    axes[0].axhline(EPS_SPLIT, color="k", ls=":", label=f"eps_split ({EPS_SPLIT:.1e})")
    baseline = payload.get("reference_lines", {}).get("model_A_one_step")
    if baseline is not None:
        axes[0].axhline(baseline, color="tab:red", ls="--", label=f"Model A ({baseline:.2e})")

    for axis in axes:
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
    if "selected_horizon" in payload:
        for axis, quantity in zip(axes[1:], ("mass", "energy")):
            axis.set_title(f"{quantity} drift @{payload['selected_horizon']}")
    figure.suptitle(
        f"Phase 7a: PINO lambda sweep  [{payload['data_hash']}]  "
        "(CN residual is itself mass-preserving -- see docstring)"
    )
    figure.tight_layout()
    figure.savefig(output / "plots" / "phase7_lambda_sweep.png", dpi=150)
    plt.close(figure)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--lambdas", type=float, nargs="+", default=[0.0, 0.01, 0.1, 1.0, 10.0]
    )
    from spno.phase_workflow import add_workflow_arguments, parse_workflow_args
    add_workflow_arguments(parser)
    return parse_workflow_args(parser, argv)


def run_sweep(shards, data_config: DataConfig, args, *, save=save_run) -> dict:
    """Run every lambda from the same per-seed initialization and save one payload."""

    device = pick_device(args.device)
    epochs = 2 if args.quick else args.epochs
    seeds = args.seeds[:1] if args.quick else args.seeds

    # Every lambda lands in ONE payload: the sweep is the deliverable. A per-lambda
    # identifier would scatter it across directories and invite quoting one value.
    identifier = run_identifier(config_hash(data_config), "pino", quick=args.quick)

    scale = field_scale(shards["train"], data_config.domain)

    payload: dict = {
        "phase": 7,
        "variant": "7a-discrete-midpoint",
        "data_hash": config_hash(data_config),
        "identifier": identifier,
        "device": device,
        "quick": args.quick,
        "field_scale": scale,
        "confound": (
            "the Crank-Nicolson residual is itself discretely mass-preserving -- its "
            "exact plane-wave update is the Cayley transform, |z| = 1 identically -- so "
            "this physics loss smuggles in the invariant under study. Report this "
            "beside every number here."
        ),
        "control_arm": "lambda=0 is bit-identical to train_one_step, i.e. Phase 2-3 "
        "Model A at the same budget",
        "reference_lines": {
            "eps_split": EPS_SPLIT,
            "model_A_one_step": MODEL_A_ONE_STEP,
            "model_A_rollout_100": MODEL_A_ROLLOUT_100,
            "caveat": "the Phase 2-3 Model A numbers were budget-bound (best_epoch == "
            "24 of 25); they are reference lines, not converged baselines",
        },
        "sweep": {},
    }

    by_lambda = {physics_weight: [] for physics_weight in args.lambdas}
    parameter_count = None
    for seed in seeds:
        baseline = build_pino_model(data_config, scale, seed)
        initial_state = {
            key: value.detach().clone()
            for key, value in baseline.state_dict().items()
        }
        for physics_weight in args.lambdas:
            train_config = TrainConfig(
                epochs=epochs, device=device, seed=seed,
                max_train_pairs=256 if args.quick else None,
            )
            model = build_pino_model(data_config, scale, seed)
            model.load_state_dict(initial_state)
            history = train_pino(
                model,
                shards["train"],
                shards["val"],
                data_config,
                train_config,
                physics_weight=physics_weight,
                verbose=not args.quick,
            )
            metrics = evaluate_model(model, shards, data_config, train_config)
            by_lambda[physics_weight].append(
                {
                    "seed": seed,
                    "history": history.as_dict(),
                    "metrics": metrics,
                    "converged": converged(history),
                    "config": describe_config(data_config, train_config),
                }
            )
            parameter_count = model.parameter_count()

    for physics_weight, per_seed in by_lambda.items():
        payload["sweep"][str(physics_weight)] = {
            "physics_weight": physics_weight,
            "per_seed": per_seed,
            "parameter_count": parameter_count,
            **aggregate_pino_seed_metrics(per_seed),
        }

    warning = budget_warning(
        {
            key: {"converged": [seed_result["converged"] for seed_result in entry["per_seed"]]}
            for key, entry in payload["sweep"].items()
        }
    )
    payload["budget_warning"] = warning

    output = save("phase7", identifier, payload)
    make_plots(payload, output)

    print(f"{'phase':<22}7a -- PINO discrete midpoint residual")
    print(f"{'data hash':<22}{payload['data_hash']}")
    print(f"{'identifier':<22}{identifier}")
    print(f"{'device':<22}{device}")
    print(f"{'lambdas':<22}{' '.join(str(v) for v in args.lambdas)}")
    print(f"{'seeds':<22}{' '.join(str(s) for s in seeds)}  epochs {epochs}")
    for key, entry in payload["sweep"].items():
        print(f"{'  lambda=' + key:<22}one-step {entry['one_step_mean']:.4e}")
    if warning:
        print(f"{'BUDGET BOUND':<22}{warning}")
    print(f"{'written to':<22}{output}")
    return payload


def main(argv=None) -> dict:
    args = parse_args(argv)
    if args.stage is not None:
        from spno.phase_workflow import run_stage
        return run_stage(7, args)

    data_config = DataConfig()
    shards = load_shards(data_config)
    return run_sweep(shards, data_config, args)


if __name__ == "__main__":
    main()
