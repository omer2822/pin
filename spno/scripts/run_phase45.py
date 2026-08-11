"""Phases 4 and 5: the structured models and the full evaluation suite.

Trains the whole model family on identical data with an identical objective and budget,
then runs every evaluator: rollout, conservation with a fitted secular-vs-bounded
verdict, reversibility with its regime classification, and the spectral decomposition.

Models (guarantees hold for every theta, so they are asserted before training starts):

    A       FNO, unrestricted                     -- none
    B-loop  FNO + mass projection in the loop     -- mass
    C1      split step, pointwise phase in rho    -- mass, reversible, symplectic, U(1)
    C2      split step, FNO phase in rho          -- mass, reversible, U(1)
    C3      split step, phase in Re/Im psi        -- mass only (the control for B)

Two training modes, reported separately and never pooled: one-step, and short-horizon
rollout.  Rollout training can repair much of an unconstrained model's drift, so
averaging the modes would erase the contrast the study exists to measure.

Budget note: Phases 2-3 ran 25 epochs and early stopping never fired -- val loss was
still falling 26-50% over the last five epochs, so those baselines were **not
converged**.  The default here is larger, and ``--epochs`` should be raised until
``best_epoch`` stops landing on the final epoch before any of this reaches the thesis.

Usage::

    python scripts/run_phase45.py [--quick] [--seeds 0 1 2] [--epochs 60]
                                  [--mode one-step|rollout] [--kinetic K0|K1|K2]
"""

from __future__ import annotations

import argparse
import dataclasses
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from spno.config import DataConfig, config_hash
from spno.equations.nls import wrap_wavenumber
from spno.evaluation.conservation import evaluate_conservation
from spno.evaluation.reversibility import evaluate_reversibility
from spno.evaluation.spectral import evaluate_spectral
from spno.experiments import (
    arithmetic_mass_floor,
    describe_config,
    evaluate_model,
    load_shards,
    pick_device,
    rollout_inputs,
    save_run,
)
from spno.models.fno import FNOStepOperator
from spno.models.projected import MassProjectedOperator
from spno.models.split_learned import (
    DensityPhaseSplitStep,
    FieldDensityPhaseSplitStep,
    FullFieldPhaseSplitStep,
)
from spno.precision import widen_to_double
from spno.train import TrainConfig, field_scale, train_one_step, train_rollout

CHECKPOINTS = (1, 10, 20, 50, 100, 200)
EPS_SPLIT = 6.239e-05  # measured in Phase 0


def build_models(
    data_config: DataConfig, scale: float, seed: int, kinetic: str, local: str = "L0"
) -> dict:
    """Every model gets the same seed, so differences are architectural."""

    def fno():
        torch.manual_seed(seed)
        return FNOStepOperator(
            data_config.domain,
            modes=16, width=64, n_layers=4,
            alpha_range=data_config.alpha_range,
            beta_range=data_config.beta_range,
            field_scale=scale,
            trained_dt=data_config.dt,
        )

    def structured(kind, **kwargs):
        torch.manual_seed(seed)
        return kind(
            data_config.domain,
            kinetic_mode=kinetic,
            trained_dt=data_config.dt,
            **kwargs,
        )

    def structured_local(kind):
        torch.manual_seed(seed)
        # L1 supplies the product beta*rho as a pointwise *feature*, which only makes
        # sense for the pointwise net.  C2's phase comes from an FNO over channels, so
        # it falls back to L0 -- stated here rather than silently, because a table
        # labelled "L1" whose C2 column is really L0 would be misleading.
        mode = "L0" if (local == "L1" and kind is FieldDensityPhaseSplitStep) else local
        return kind(
            data_config.domain,
            kinetic_mode=kinetic,
            local_mode=mode,
            trained_dt=data_config.dt,
        )

    return {
        "A": fno(),
        "B-loop": MassProjectedOperator(fno()),
        "C1": structured_local(DensityPhaseSplitStep),
        "C2": structured_local(FieldDensityPhaseSplitStep),
        "C3": structured(FullFieldPhaseSplitStep),
    }


def assert_untrained_guarantees(models: dict, data_config: DataConfig) -> dict:
    """Structural claims hold for every theta, so verify them *before* training.

    If a guarantee fails here it is an implementation bug, and every downstream number
    would be meaningless.  Cheap enough to run every time.
    """

    from spno.domain import l2_mass

    domain = data_config.domain
    generator = torch.Generator().manual_seed(12345)
    field = torch.complex(
        torch.randn(4, *domain.shape, generator=generator, dtype=torch.float64),
        torch.randn(4, *domain.shape, generator=generator, dtype=torch.float64),
    )
    potential = torch.randn(4, *domain.shape, generator=generator, dtype=torch.float64)
    alpha = torch.rand(4, generator=generator, dtype=torch.float64) * 0.4 + 0.7
    beta = torch.rand(4, generator=generator, dtype=torch.float64) - 0.4

    report = {}
    for name, model in models.items():
        widened = widen_to_double(model)
        with torch.no_grad():
            output = widened(field, potential, alpha, beta, data_config.dt)
        drift = float(
            torch.abs(l2_mass(output, domain) / l2_mass(field, domain) - 1).max()
        )
        entry = {"mass_drift_untrained": drift}
        if name.startswith("C"):
            # `raise`, not `assert`: this is the only gate between an implementation bug
            # and a full run of plausible-but-fictional numbers, and `python -O` strips
            # asserts.
            if not drift < 1e-13:
                raise RuntimeError(
                    f"{name} violates mass preservation at untrained weights: "
                    f"{drift:.2e}. This is an implementation bug; every downstream "
                    "number would be meaningless."
                )
            metrics = evaluate_reversibility(
                widened, domain, field, potential, alpha, beta, data_config.dt, steps=5
            )
            entry["reversibility_regime"] = metrics.regime
            entry["reversibility_order"] = metrics.order
        report[name] = entry
    return report


def run_seed(seed, shards, data_config, train_config, scale, kinetic, mode, local="L0") -> dict:
    domain = data_config.domain
    results = {}
    for name, model in build_models(data_config, scale, seed, kinetic, local).items():
        print(f"\n--- seed {seed}: {name} ({mode}) ---")
        config = dataclasses.replace(train_config, seed=seed)
        trainer = train_rollout if mode == "rollout" else train_one_step
        history = trainer(model, shards["train"], shards["val"], data_config, config)

        entry = evaluate_model(
            model, shards, data_config, config, checkpoints=CHECKPOINTS
        )
        entry["history"] = history.as_dict()
        entry["converged"] = history.best_epoch < len(history.val_loss) - 1

        widened = widen_to_double(model, device="cpu").eval()
        inputs = rollout_inputs(shards["test"], data_config, "cpu", n=64, dtype=torch.complex128)
        entry["conservation"] = evaluate_conservation(
            widened, domain, inputs["initial"], inputs["potential"],
            inputs["alpha"], inputs["beta"], data_config.dt, steps=200, stride=10,
        ).as_dict()

        if getattr(widened, "supports_time_reversal", False):
            entry["reversibility"] = evaluate_reversibility(
                widened, domain, inputs["initial"], inputs["potential"],
                inputs["alpha"], inputs["beta"], data_config.dt, steps=20,
            ).as_dict()
        else:
            entry["reversibility"] = {
                "regime": "not-applicable",
                "reason": "model ignores dt; stepping at -dt would re-apply the forward map",
            }

        with torch.no_grad():
            state = inputs["initial"]
            for _ in range(100):
                state = widened(
                    state, inputs["potential"], inputs["alpha"], inputs["beta"],
                    data_config.dt,
                )
        entry["spectral_at_100"] = evaluate_spectral(
            state, inputs["trajectories"][:, 100], domain,
            band_edges=(
                float(data_config.initial_bandwidth),
                wrap_wavenumber(data_config.alpha_range[1], data_config.dt),
            ),
        ).as_dict()
        results[name] = entry
    return results


def aggregate(per_seed: dict) -> dict:
    names = list(next(iter(per_seed.values())))
    summary = {}
    for name in names:
        entries = [per_seed[s][name] for s in per_seed]
        one_step = [e["one_step_test"] for e in entries]
        summary[name] = {
            "parameters": entries[0]["parameters"],
            "one_step_mean": sum(one_step) / len(one_step),
            "one_step_min": min(one_step),
            "one_step_max": max(one_step),
            "one_step_over_eps_split": (sum(one_step) / len(one_step)) / EPS_SPLIT,
            "rollout": {
                str(step): {
                    "error_mean": sum(
                        e["rollout"]["relative_error"][i] for e in entries
                    ) / len(entries),
                    "mass_drift_mean": sum(
                        e["rollout"]["mass_drift"][i] for e in entries
                    ) / len(entries),
                }
                for i, step in enumerate(entries[0]["rollout"]["steps"])
            },
            "energy_slope_mean": sum(
                e["conservation"]["energy_trend"]["slope"] for e in entries
            ) / len(entries),
            "energy_classification": [
                e["conservation"]["energy_trend"]["classification"] for e in entries
            ],
            "reversibility_regime": [e["reversibility"]["regime"] for e in entries],
            "mean_phase_error": sum(
                e["spectral_at_100"]["mean_phase_error"] for e in entries
            ) / len(entries),
            "banded_error": entries[0]["spectral_at_100"]["banded"],
            "converged": [e["converged"] for e in entries],
        }
    return summary


def make_plots(summary, payload, output):
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for name, entry in summary.items():
        steps = sorted(int(s) for s in entry["rollout"])
        axes[0, 0].plot(
            steps, [entry["rollout"][str(s)]["error_mean"] for s in steps], "o-", label=name
        )
        axes[0, 1].plot(
            steps,
            [max(entry["rollout"][str(s)]["mass_drift_mean"], 1e-17) for s in steps],
            "o-", label=name,
        )
    axes[0, 0].axhline(EPS_SPLIT, color="k", ls=":", label=f"eps_split ({EPS_SPLIT:.1e})")
    axes[0, 1].axhline(
        payload["mass_floor"]["float64_cpu"]["mean"], color="k", ls="--", label="float64 floor"
    )
    axes[0, 0].set(xlabel="rollout step", ylabel="relative L2", title="Rollout error")
    axes[0, 1].set(xlabel="rollout step", ylabel="mass drift", title="Mass drift")
    for axis in axes[0]:
        axis.set_xscale("log")
        axis.set_yscale("log")

    names = list(summary)
    axes[1, 0].bar(names, [summary[n]["energy_slope_mean"] for n in names])
    axes[1, 0].axhline(0.25, color="k", ls="--", label="bounded / secular cut")
    axes[1, 0].set(ylabel="fitted log-log slope", title="Energy drift trend")

    axes[1, 1].bar(names, [summary[n]["mean_phase_error"] for n in names])
    axes[1, 1].set(ylabel="radians", title="Amplitude-weighted phase error @100")

    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle(f"Phases 4-5  [{payload['data_hash']}]  mode={payload['mode']}")
    figure.tight_layout()
    figure.savefig(output / "plots" / f"phase45_{payload['mode']}.png", dpi=150)
    plt.close(figure)


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--mode", choices=("one-step", "rollout"), default="one-step")
    parser.add_argument("--kinetic", choices=("K0", "K1", "K2"), default="K0")
    parser.add_argument("--local", choices=("L0", "L1", "L2"), default="L0")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    data_config = DataConfig()
    device = pick_device(args.device)
    train_config = TrainConfig(
        epochs=2 if args.quick else args.epochs,
        batch_size=256,
        learning_rate=1e-3,
        patience=8,
        device=device,
    )
    seeds = args.seeds[:1] if args.quick else args.seeds

    shards = load_shards(data_config)
    scale = field_scale(shards["train"], data_config.domain)
    print(f"device={device}  scale={scale:.4f}  seeds={seeds}  kinetic={args.kinetic}  local={args.local}")

    print("\nverifying architectural guarantees at untrained weights ...")
    guarantees = assert_untrained_guarantees(
        build_models(data_config, scale, seeds[0], args.kinetic, args.local), data_config
    )
    for name, entry in guarantees.items():
        print(f"  {name:8s} mass drift {entry['mass_drift_untrained']:.2e}"
              + (f"   reversibility {entry.get('reversibility_regime')}"
                 f" (order {entry.get('reversibility_order')})" if name.startswith("C") else ""))

    per_seed = {
        seed: run_seed(
            seed, shards, data_config, train_config, scale, args.kinetic, args.mode, args.local
        )
        for seed in seeds
    }
    summary = aggregate(per_seed)

    payload = {
        **describe_config(data_config, train_config),
        "seeds": seeds,
        "mode": args.mode,
        "kinetic_mode": args.kinetic,
        "local_mode": args.local,
        "eps_split": EPS_SPLIT,
        "mass_floor": arithmetic_mass_floor(shards, data_config),
        "untrained_guarantees": guarantees,
        "summary": summary,
        "per_seed": per_seed,
    }
    identifier = f"{config_hash(data_config)}-{args.mode}-{args.kinetic}{args.local}"
    output = save_run("phase45", identifier, payload)
    make_plots(summary, payload, output)

    print(f"\n=== Phases 4-5 [{identifier}] ===")
    header = (
        f"{'model':8s} {'params':>9s} {'1-step':>10s} {'/eps':>7s} {'roll@100':>10s} "
        f"{'mass@100':>10s} {'E slope':>8s} {'energy':>9s} {'reversible':>12s}"
    )
    print(header)
    print("-" * len(header))
    for name, entry in summary.items():
        print(
            f"{name:8s} {entry['parameters']:9,d} {entry['one_step_mean']:10.3e} "
            f"{entry['one_step_over_eps_split']:7.2f} "
            f"{entry['rollout']['100']['error_mean']:10.3e} "
            f"{entry['rollout']['100']['mass_drift_mean']:10.3e} "
            f"{entry['energy_slope_mean']:8.3f} "
            f"{entry['energy_classification'][0]:>9s} "
            f"{entry['reversibility_regime'][0]:>12s}"
        )
    unconverged = [n for n, e in summary.items() if not all(e["converged"])]
    if unconverged:
        print(f"\nWARNING: budget bound for {unconverged} -- raise --epochs before reporting")
    print(f"\nwritten to: {output}")
    return payload


if __name__ == "__main__":
    main()
