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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from spno.config import DataConfig, config_hash
from spno.experiments import (
    describe_config,
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
from spno.train import TrainConfig

import dataclasses
from pathlib import Path
from spno.experiments import load_shards, evaluate_model, DATA_ROOT
from spno.train import train_one_step
from spno.data.datasets import generate_shard, shard_paths, assert_no_leakage


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


def build_models(domain, data_config: DataConfig) -> dict:
    """All five models are retrained at every dial value."""

    common = dict(
        alpha_range=data_config.alpha_range,
        beta_range=data_config.beta_range,
        trained_dt=data_config.dt,
    )
    dt = data_config.dt
    return {
        "A": lambda: FNOStepOperator(domain, modes=16, **common),
        "B-loop": lambda: MassProjectedOperator(
            FNOStepOperator(domain, modes=16, **common)
        ),
        "C1": lambda: DensityPhaseSplitStep(domain, kinetic_mode="K0", trained_dt=dt),
        "C2": lambda: FieldDensityPhaseSplitStep(
            domain, kinetic_mode="K0", trained_dt=dt
        ),
        "C3": lambda: FullFieldPhaseSplitStep(domain, trained_dt=dt),
    }


def locate_crossover(by_dial: dict, model: str) -> dict:
    """Where a model's A-relative rollout error first reaches 1.0, or a bounded note.

    A located crossover is the result; a bounded one ("no crossover within the swept
    range") is also publishable.  The bounded form is emitted explicitly rather than
    leaving the key absent, so a reader cannot mistake "not found" for "not looked for".
    """

    dials = sorted(float(key) for key in by_dial)
    ratios = [by_dial[str(d)].get("relative_to_A", {}).get(model) for d in dials]
    
    if not ratios or any(r is None for r in ratios):
        raise ValueError(f"Missing evaluation data for {model}; cannot draw conclusions about crossover.")

    for dial, ratio in zip(dials, ratios):
        if ratio >= 1.0:
            return {"crossover": dial, "bounded": None}
    return {
        "crossover": None,
        "bounded": f"no crossover within [{min(dials, default=0)}, "
        f"{max(dials, default=0)}]: {model} still beats A at every swept value",
    }


def make_plots(payload: dict, output) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, dial in zip(axes, ("sigma", "gamma")):
        by_dial = payload["sweeps"].get(dial, {})
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
                        by_dial[str(v)]["relative_to_A"].get(model, float("nan"))
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
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    data_config = DataConfig()
    device = pick_device(args.device)
    epochs = 2 if args.quick else args.epochs
    seeds = args.seeds[:1] if args.quick else args.seeds
    identifier = run_identifier(config_hash(data_config), "misspec", quick=args.quick)

    gates = bitwise_gate(data_config)
    domain = data_config.domain
    builders = build_models(domain, data_config)

    payload: dict = {
        "phase": 9,
        "data_hash": config_hash(data_config),
        "identifier": identifier,
        "device": device,
        "quick": args.quick,
        "gates": gates,
        "parameter_counts": {name: build().parameter_count() for name, build in builders.items()},
        "dials": {
            "sigma": "nonlocal nonlinearity: breaks LOCALITY only. Still Hamiltonian, "
            "still U(1), still exactly mass-conserving, so B's projection stays correct "
            "and only C1's pointwise nu cannot represent the truth. Separates C1 from C2.",
            "gamma": "weak gain/loss: breaks CONSERVATION itself, so B's hard mass "
            "constraint becomes actively wrong.",
            "excluded": "saturable and quintic nonlinearities are NOT used: nu_theta is "
            "already a free function of rho and learns them easily, so they are not "
            "misspecifications at all.",
        },
        "config": describe_config(
            data_config, TrainConfig(epochs=epochs, device=device)
        ),
        "sweeps": {"sigma": {}, "gamma": {}},
        "crossover": {},
    }

    for dial, values in (("sigma", args.sigmas), ("gamma", args.gammas)):
        for value in values:
            spec = (
                MisspecificationConfig(nonlocal_sigma=value)
                if dial == "sigma"
                else MisspecificationConfig(gain_loss_gamma=value)
            )
            identifier = spec.identifier(data_config)
            
            # Load or generate shards
            try:
                shards = load_shards(data_config, identifier=identifier)
            except FileNotFoundError:
                paths = shard_paths(DATA_ROOT, identifier)
                paths["train"].parent.mkdir(parents=True, exist_ok=True)
                shards = {}
                reference = spec.reference(data_config)
                for split in ("train", "val", "test"):
                    print(f"generating {split} for {identifier} ...")
                    shard = generate_shard(
                        data_config,
                        split,
                        reference=reference,
                        reference_metadata=spec.provenance(data_config)
                    )
                    shard.save(paths[split])
                    shards[split] = shard
                assert_no_leakage(shards)
                
            # Train models and collect errors
            print(f"\n--- {dial}={value} ({identifier}) ---")
            train_config = TrainConfig(
                epochs=epochs,
                batch_size=256,
                learning_rate=1e-3,
                patience=6,
                device=device,
            )
            
            # We track the rollout error at step 100 for each model and seed
            errors = {name: [] for name in builders.keys()}
            
            for seed in seeds:
                config_for_seed = dataclasses.replace(train_config, seed=seed)
                for name, build_fn in builders.items():
                    print(f"  training {name} (seed {seed})...")
                    model = build_fn()
                    # Re-seed exactly like the models do to ensure consistency
                    torch.manual_seed(seed) 
                    train_one_step(model, shards["train"], shards["val"], data_config, config_for_seed)
                    evaluated = evaluate_model(model, shards, data_config, config_for_seed, checkpoints=(100,))
                    # Index 0 corresponds to checkpoint 100 since it's the only one
                    rollout_error = evaluated["rollout"]["relative_error"][0]
                    errors[name].append(rollout_error)
            
            avg_errors = {name: sum(errs)/len(errs) for name, errs in errors.items()}
            relative_to_A = {
                name: avg_errors[name] / avg_errors["A"] 
                for name in builders.keys() if name != "A"
            }
            
            payload["sweeps"][dial][str(value)] = {
                "value": value,
                "identifier": identifier,
                "reuses_production_shards": spec.is_exact,
                "config": spec.as_dict(),
                "seeds": seeds,
                "note": "retrain all five models on this shard set, evaluate, and "
                "record rollout error relative to A under 'relative_to_A'",
                "relative_to_A": relative_to_A,
            }

    for dial in ("sigma", "gamma"):
        payload["crossover"][dial] = {
            model: locate_crossover(payload["sweeps"][dial], model)
            for model in ("B-loop", "C1", "C2", "C3")
        }

    output = save_run("phase9", identifier, payload)
    make_plots(payload, output)

    print(f"{'phase':<22}9 -- misspecification sweep")
    print(f"{'data hash':<22}{payload['data_hash']}")
    print(f"{'identifier':<22}{identifier}")
    print(f"{'device':<22}{device}")
    print(f"{'bitwise gate':<22}passed at dial zero")
    print(f"{'sigmas':<22}{' '.join(str(v) for v in args.sigmas)}")
    print(f"{'gammas':<22}{' '.join(str(v) for v in args.gammas)}")
    for name, count in payload["parameter_counts"].items():
        print(f"{'  params ' + name:<22}{count:,}")
    for dial in ("sigma", "gamma"):
        for model, entry in payload["crossover"][dial].items():
            state = entry["crossover"] if entry["crossover"] is not None else "bounded"
            print(f"{'  crossover ' + dial + ' ' + model:<22}{state}")
    print(f"{'written to':<22}{output}")
    return payload


if __name__ == "__main__":
    main()
