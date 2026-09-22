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

Nothing is trained here without an explicit run; the deferred Phase 2-5 convergence debt
must be cleared first or every row describes the epoch cap rather than the model.
"""

from __future__ import annotations

import argparse

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
    """Aggregate real test metrics, locating horizon 100 by its recorded step."""

    one_step = [entry["metrics"]["one_step_test"] for entry in per_seed]
    mass_drift_100 = []
    energy_drift_100 = []
    for entry in per_seed:
        rollout = entry["metrics"]["rollout"]
        horizon_index = rollout["steps"].index(horizon)
        mass_drift_100.append(rollout["mass_drift"][horizon_index])
        energy_drift_100.append(rollout["energy_drift"][horizon_index])
    return {
        "one_step_mean": sum(one_step) / len(one_step),
        "one_step_min": min(one_step),
        "one_step_max": max(one_step),
        "mass_drift_100_mean": sum(mass_drift_100) / len(mass_drift_100),
        "energy_drift_100_mean": sum(energy_drift_100) / len(energy_drift_100),
        "converged": [entry["converged"] for entry in per_seed],
    }


def make_plots(payload: dict, output) -> None:
    """One figure: error, mass drift and energy drift against lambda on a log axis."""

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
