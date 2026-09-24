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


PROBE_COLUMNS = ("arm", "lambda", "seeds", "one_step", "distance_to_cn", "rollout", "aligned",
                 "global_phase", "energy_drift_final", "energy_slope", "energy_trend",
                 "mass_drift_final", "diverged_seeds")
_NAN = float("nan")


def _median(values) -> float:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.median(finite) if finite else _NAN


def _probe_values(measured: dict) -> dict:
    """Last-checkpoint rollout values and final-step invariants for one seed or reference."""

    rollout, long = measured["rollout"], measured["long_horizon"]
    finished = rollout["diverged_at"] is None and bool(rollout["steps"])
    last = lambda key: rollout[key][-1] if finished else _NAN
    values = {"distance_to_cn": measured["distance_to_cn"], "rollout": last("relative_error"),
              "aligned": last("aligned_error"), "global_phase": last("global_phase"),
              "energy_drift_final": _NAN, "mass_drift_final": _NAN, "energy_slope": _NAN,
              "energy_trend": "", "diverged": not finished}
    if long is not None:
        complete = (long["diverged_at"] is None and bool(long["steps"])
                    and long["steps"][-1] == long["requested_steps"])
        values.update(energy_slope=long["energy_trend"]["slope"],
                      energy_trend=long["energy_trend"]["classification"],
                      diverged=values["diverged"] or not complete)
        if complete:
            values.update(energy_drift_final=long["energy_drift"][-1],
                          mass_drift_final=long["mass_drift"][-1])
    return values


def probe_summary_rows(payload: dict) -> list[dict]:
    """One row per reference operator and per (arm, lambda); seed medians for arms."""

    numeric = ("distance_to_cn", "rollout", "aligned", "global_phase", "energy_drift_final",
               "energy_slope", "mass_drift_final")
    rows = []
    for name, measured in payload["references"].items():
        values = _probe_values(measured)
        rows.append({"arm": name, "lambda": "", "seeds": "reference", "one_step": measured["one_step"],
                     "energy_trend": values["energy_trend"], "diverged_seeds": int(values["diverged"]),
                     **{key: values[key] for key in numeric}})
    for entry in payload["arms"]:
        per_seed = [_probe_values(m) for m in entry["per_seed"].values()]
        trends = [v["energy_trend"] for v in per_seed if v["energy_trend"]]
        rows.append({"arm": entry["arm"], "lambda": entry["lambda"], "seeds": len(per_seed),
                     "one_step": "", "diverged_seeds": sum(v["diverged"] for v in per_seed),
                     "energy_trend": ", ".join(f"{t}×{trends.count(t)}" for t in sorted(set(trends))),
                     **{key: _median(v[key] for v in per_seed) for key in numeric}})
    return rows


def probe_table_html(payload: dict) -> str:
    import html

    def cell(value):
        if isinstance(value, float):
            return "—" if math.isnan(value) else f"{value:.3e}"
        return str(value)

    header = "".join(f"<th>{html.escape(c)}</th>" for c in PROBE_COLUMNS)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(cell(row[c]))}</td>" for c in PROBE_COLUMNS)
                   + "</tr>" for row in probe_summary_rows(payload))
    settings = payload["settings"]
    caption = (f"distance_to_cn: one-step relative L2 to the exact CN step, first "
               f"{settings['cn_trajectories']} test trajectories. rollout / aligned / global_phase "
               f"(rad) at step {settings['checkpoints'][-1]}. Invariants at step "
               f"{settings['long_steps']} over {settings['long_batch']} initial conditions, only "
               f"for lambda in {settings['long_lambdas']}. Arms show seed medians; diverged seeds "
               "are excluded and counted.")
    return f"<p>{html.escape(caption)}</p><table><tr>{header}</tr>{body}</table>"


def make_probe_plots(payload: dict, output) -> None:
    """Distance to CN against lambda, long-horizon invariants, raw vs aligned rollout."""

    floor = 1e-17  # exact conservation has no log; draw it at the floor
    status = "EXPLORATORY" if payload["exploratory"] else "converged checkpoints"
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    arms = payload["arms"]
    references = payload["references"]
    styles = {"CN exact": ("black", "--"), "Strang 1-step": ("0.45", ":")}
    colour_of = {f"{e['arm']}|{e['lambda']:g}": colours[i % len(colours)] for i, e in enumerate(arms)}

    # 1. Does a larger lambda move the model toward the CN step?
    figure, axis = plt.subplots(figsize=(8, 5))
    weights = sorted({e["lambda"] for e in arms})
    for index, family in enumerate(sorted({e["family"] for e in arms})):
        entries = {e["lambda"]: e for e in arms if e["family"] == family}
        xs = [weights.index(w) for w in weights if w in entries]
        medians = [_median(m["distance_to_cn"] for m in entries[w]["per_seed"].values())
                   for w in weights if w in entries]
        colour = colours[index % len(colours)]
        axis.plot(xs, medians, color=colour, marker="o", label=f"{family} / {family}+PDE (median)")
        for w in entries:
            values = [m["distance_to_cn"] for m in entries[w]["per_seed"].values()]
            axis.plot([weights.index(w)] * len(values), values, ".", color=colour, alpha=.5)
    axis.axhline(references["CN exact"]["one_step"], color="black", ls="--",
                 label="substepped truth (= CN one-step error)")
    axis.axhline(references["Strang 1-step"]["distance_to_cn"], color="0.45", ls=":",
                 label="Strang 1-step")
    axis.set_xticks(range(len(weights)), [f"{w:g}" for w in weights])
    axis.set_yscale("log")
    axis.set_xlabel("PDE weight lambda")
    axis.set_ylabel("one-step relative L2 to the exact CN step")
    axis.set_title(f"Does the residual pull toward CN dynamics? — {status}")
    axis.grid(True, which="both", alpha=.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "plots" / "phase7_toward_cn.png", dpi=150)
    plt.close(figure)

    # 2. Long-horizon invariants, one line per seed.
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for axis, key, title in zip(axes, ("energy_drift", "mass_drift"), ("Energy drift", "Mass drift")):
        for entry in arms:
            colour = colour_of[f"{entry['arm']}|{entry['lambda']:g}"]
            longs = [m["long_horizon"] for m in entry["per_seed"].values() if m["long_horizon"]]
            for number, long in enumerate(longs):
                axis.plot(long["steps"], [max(v, floor) for v in long[key]], color=colour, lw=1.2,
                          alpha=.75, label=f"{entry['arm']} λ={entry['lambda']:g}" if number == 0 else None)
                if long["diverged_at"] is not None and long["steps"]:
                    axis.plot(long["steps"][-1], max(long[key][-1], floor), "x", color=colour)
        for name, measured in references.items():
            colour, dash = styles.get(name, ("0.3", "-."))
            long = measured["long_horizon"]
            axis.plot(long["steps"], [max(v, floor) for v in long[key]], color=colour, ls=dash,
                      lw=1.6, label=name)
        horizon = payload["settings"]["checkpoints"][-1]
        axis.axvline(horizon, color="0.6", lw=.8)
        axis.text(horizon, 1, " stored frames end", fontsize=7, color="0.4",
                  transform=axis.get_xaxis_transform(), va="top")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("step")
        axis.set_title(f"{title} (one line per seed; × = diverged)")
        axis.grid(True, which="both", alpha=.3)
    axes[0].legend(fontsize=7, ncol=2)
    figure.suptitle(f"Phase 7 probes: long-horizon invariants, float64 — {status}")
    figure.tight_layout(rect=(0, 0, 1, .94))
    figure.savefig(output / "plots" / "phase7_long_horizon.png", dpi=150)
    plt.close(figure)

    # 3. How much of the rollout error is a global phase?
    def median_curve(measurements, key):
        steps = payload["settings"]["checkpoints"]
        curves = [dict(zip(m["rollout"]["steps"], m["rollout"][key])) for m in measurements]
        return steps, [_median(c.get(s, _NAN) for c in curves) for s in steps]

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    series = [(f"{e['arm']} λ={e['lambda']:g}", list(e["per_seed"].values()),
               colour_of[f"{e['arm']}|{e['lambda']:g}"], "-") for e in arms]
    series += [(name, [m], *styles.get(name, ("0.3", "-."))) for name, m in references.items()]
    for label, measurements, colour, dash in series:
        steps, raw = median_curve(measurements, "relative_error")
        _, aligned = median_curve(measurements, "aligned_error")
        _, phase = median_curve(measurements, "global_phase")
        axes[0].plot(steps, raw, color=colour, ls=dash, lw=1, alpha=.45)
        axes[0].plot(steps, aligned, color=colour, ls=dash, lw=1.8, marker="o", ms=3, label=label)
        axes[1].plot(steps, phase, color=colour, ls=dash, lw=1.6, marker="o", ms=3, label=label)
    axes[0].set_title("Rollout relative L2: aligned (bold) vs raw (faint), seed median")
    axes[1].set_title("Mean |global phase| (rad), seed median")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("step")
        axis.grid(True, which="both", alpha=.3)
    axes[1].legend(fontsize=7, ncol=2)
    figure.suptitle(f"Phase 7 probes: how much rollout error is global phase — {status}")
    figure.tight_layout(rect=(0, 0, 1, .94))
    figure.savefig(output / "plots" / "phase7_phase_aligned.png", dpi=150)
    plt.close(figure)

    with (output / "probes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PROBE_COLUMNS))
        writer.writeheader()
        writer.writerows(probe_summary_rows(payload))


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
