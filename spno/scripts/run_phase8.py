"""Phase 8: resolution transfer and robustness.

**The distinction this runner refuses to blur.**  Upsampling a *band-limited* test set
and re-running an FNO barely changes its output, because the FNO only touches
``k <= n_modes`` and the resample leaves those modes alone.  That is **not** evidence of
resolution transfer, and the payload says so in the section itself.  The genuine test is
new energy above the training grid's Nyquist -- which is G4 (spectral extrapolation)
wearing a different hat, and is cross-referenced as such.  The two are reported in
separate payload sections and separate plot panels, never pooled.

Noise on psi and noise on V are swept **separately**: they probe state estimation versus
parameter estimation, and a joint sweep could not attribute a degradation to either.

Usage:
    python scripts/run_phase8.py [--quick] [--seeds 0 1 2] [--epochs 60]
                                 [--device auto] [--grid 128]
                                 [--noise 0.0 0.001 0.01 0.05]
                                 [--train-fractions 0.05 0.1 0.25 0.5 1.0]
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from spno.config import DataConfig, config_hash
from spno.data.corruption import add_field_noise, add_potential_noise
from spno.domain import PeriodicDomain
from spno.evaluation.resolution import grid_dependence, rebind_domain, spectral_resample
from spno.experiments import (
    describe_config,
    evaluate_model,
    load_shards,
    pick_device,
    run_identifier,
    save_run,
)
from spno.models.fno import FNOStepOperator
from spno.models.split_learned import DensityPhaseSplitStep
from spno.train import TrainConfig

BAND_LIMITED_CAVEAT = (
    "NOT evidence of resolution transfer: an FNO only touches k <= n_modes, and "
    "upsampling a band-limited field leaves those modes unchanged, so its output barely "
    "moves by construction. Quote this sentence beside the numbers."
)
NEW_HIGH_K_CAVEAT = (
    "the genuine test, and it is G4 in disguise: energy above the training grid's "
    "Nyquist is spectral extrapolation. Cross-reference the G4 bandwidth arms in "
    "Phase 6 rather than presenting this as an independent result."
)


def build_models(domain, data_config: DataConfig) -> dict:
    common = dict(
        alpha_range=data_config.alpha_range,
        beta_range=data_config.beta_range,
        trained_dt=data_config.dt,
    )
    return {
        "A": FNOStepOperator(domain, modes=16, **common),
        "C1": DensityPhaseSplitStep(domain, kinetic_mode="K0", trained_dt=data_config.dt),
    }


def resolution_sections(models: dict, shards, data_config: DataConfig, grid: int) -> dict:
    """Band-limited upsampling and new-high-k, kept apart deliberately."""

    coarse = data_config.domain
    fine = PeriodicDomain.periodic_1d(grid)

    band_limited = {}
    for name, model in models.items():
        resampled = spectral_resample(shards["test"].trajectories[:, 0], coarse, fine)
        rebind_domain(model, fine)
        band_limited[name] = {
            "grid": grid,
            "input_shape": list(resampled.shape),
            "grid_dependence": grid_dependence(model),
        }
        rebind_domain(model, coarse)  # leave the model as we found it

    return {
        "resolution_band_limited": {"caveat": BAND_LIMITED_CAVEAT, "by_model": band_limited},
        "resolution_new_high_k": {
            "caveat": NEW_HIGH_K_CAVEAT,
            "cross_reference": "Phase 6 arms G4-bandwidth-{12,16,20,24}",
            "note": "requires shards generated at N=%d with initial_bandwidth above the "
            "N=%d Nyquist; generation is a deferred run." % (grid, coarse.shape[0]),
        },
    }


def robustness_section(models: dict, shards, data_config: DataConfig, args) -> dict:
    """Noise on psi and on V separately, plus the sample-efficiency curve."""

    domain = data_config.domain
    field = shards["test"].trajectories[:, 0]
    potential = shards["test"].potential

    field_noise, potential_noise = {}, {}
    for level in args.noise:
        generator = torch.Generator().manual_seed(0)
        corrupted = add_field_noise(field, domain, level=level, generator=generator)
        field_noise[str(level)] = {
            "level": level,
            "achieved_relative": float(
                (
                    torch.abs(corrupted - field).pow(2).sum(-1).sqrt()
                    / torch.abs(field).pow(2).sum(-1).sqrt()
                ).mean()
            ),
        }
        generator = torch.Generator().manual_seed(0)
        corrupted_v = add_potential_noise(potential, level=level, generator=generator)
        potential_noise[str(level)] = {
            "level": level,
            "max_absolute": float((corrupted_v - potential).abs().max()),
        }

    total_pairs = shards["train"].n_trajectories * (shards["train"].n_frames - 1)
    sample_efficiency = {
        str(fraction): {
            "fraction": fraction,
            "max_train_pairs": int(total_pairs * fraction),
            "parameter_counts": {n: m.parameter_count() for n, m in models.items()},
        }
        for fraction in args.train_fractions
    }

    return {
        "separately_swept": "field and potential noise are never applied jointly: they "
        "probe state versus parameter estimation and a joint sweep could not attribute "
        "a degradation to either",
        "field_noise": field_noise,
        "potential_noise": potential_noise,
        "sample_efficiency": sample_efficiency,
        "parameter_counts": {n: m.parameter_count() for n, m in models.items()},
    }


def make_plots(payload: dict, output) -> None:
    plots = output / "plots"

    # Band-limited and new-high-k as SEPARATE panels -- never pooled.
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].set_title("band-limited upsample\n(not resolution transfer)")
    axes[1].set_title("new high-k\n(G4 in disguise)")
    for axis in axes:
        axis.set_xlabel("model")
        axis.grid(True, alpha=0.3)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
    figure.suptitle(f"Phase 8 resolution  [{payload['data_hash']}]")
    figure.tight_layout()
    figure.savefig(plots / "phase8_resolution.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    robustness = payload.get("robustness", {})
    noise = robustness.get("field_noise", {})
    if noise:
        levels = sorted(float(k) for k in noise)
        axes[0].plot(
            levels, [noise[str(v)]["achieved_relative"] for v in levels], marker="o",
            label="achieved relative noise",
        )
    axes[0].set_xlabel("requested level")
    axes[0].set_title("field noise calibration")

    efficiency = robustness.get("sample_efficiency", {})
    if efficiency:
        fractions = sorted(float(k) for k in efficiency)
        axes[1].plot(
            fractions, [efficiency[str(f)]["max_train_pairs"] for f in fractions],
            marker="o", label="pairs used",
        )
        axes[1].set_yscale("log")
    axes[1].set_xlabel("training fraction")
    axes[1].set_title("sample-efficiency budget")

    for axis in axes:
        axis.grid(True, alpha=0.3)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
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
    parser.add_argument(
        "--noise", type=float, nargs="+", default=[0.0, 0.001, 0.01, 0.05]
    )
    parser.add_argument(
        "--train-fractions", type=float, nargs="+",
        default=[0.05, 0.1, 0.25, 0.5, 1.0],
    )
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    data_config = DataConfig()
    device = pick_device(args.device)
    identifier = run_identifier(
        config_hash(data_config), f"N{args.grid}", quick=args.quick
    )

    shards = load_shards(data_config)
    domain = data_config.domain
    models = build_models(domain, data_config)

    payload: dict = {
        "phase": 8,
        "data_hash": config_hash(data_config),
        "identifier": identifier,
        "device": device,
        "grid": args.grid,
        "quick": args.quick,
        "config": describe_config(
            data_config, TrainConfig(epochs=args.epochs, device=device)
        ),
    }
    payload.update(resolution_sections(models, shards, data_config, args.grid))
    payload["robustness"] = robustness_section(models, shards, data_config, args)

    output = save_run("phase8", identifier, payload)
    make_plots(payload, output)

    print(f"{'phase':<22}8 -- resolution and robustness")
    print(f"{'data hash':<22}{payload['data_hash']}")
    print(f"{'identifier':<22}{identifier}")
    print(f"{'device':<22}{device}")
    print(f"{'grid':<22}{domain.shape[0]} -> {args.grid}")
    print(f"{'noise levels':<22}{' '.join(str(v) for v in args.noise)}")
    print(f"{'train fractions':<22}{' '.join(str(v) for v in args.train_fractions)}")
    for name, count in payload["robustness"]["parameter_counts"].items():
        print(f"{'  params ' + name:<22}{count:,}")
    print(f"{'band-limited':<22}reported separately -- {BAND_LIMITED_CAVEAT[:44]}...")
    print(f"{'written to':<22}{output}")
    return payload


if __name__ == "__main__":
    main()
