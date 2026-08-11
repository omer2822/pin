"""Phase 1: generate the trajectory dataset.

Writes ``data/nls1d-<hash>/{train,val,test}.pt`` and a summary figure.  Runs in
float64 on CPU; shards are stored in complex128 so the reference data is never the
limiting precision.

Usage::

    python scripts/run_phase1.py [--quick]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from spno.config import DataConfig, config_hash
from spno.data.datasets import (
    TrajectoryShard,
    assert_no_leakage,
    generate_shard,
    shard_paths,
)
from spno.data.generate import energy_fraction_above
from spno.domain import l2_mass

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
RESULTS_ROOT = ROOT / "results"


def summarize(shards: dict[str, TrajectoryShard], config: DataConfig) -> dict:
    domain = config.domain
    summary = {"config": dataclasses.asdict(config), "splits": {}}
    for split, shard in shards.items():
        initial = shard.trajectories[:, 0]
        realized_bandwidth = _realized_bandwidth(initial, domain)
        summary["splits"][split] = {
            **shard.metadata,
            "n_trajectories": shard.n_trajectories,
            "n_frames": shard.n_frames,
            "realized_bandwidth": realized_bandwidth,
            "n_pairs": shard.n_trajectories * (shard.n_frames - 1),
        }
    return summary


def _realized_bandwidth(field: torch.Tensor, domain) -> int:
    """Largest wave number carrying non-negligible energy in the initial conditions."""

    spectrum = torch.abs(torch.fft.fftn(field, dim=domain.spatial_axes)) ** 2
    per_mode = spectrum.mean(dim=0)
    magnitude = torch.sqrt(domain.wave_number_squared())
    significant = per_mode > 1e-20 * per_mode.max()
    return int(magnitude[significant].max())


def make_plots(shards: dict[str, TrajectoryShard], config: DataConfig, output: Path):
    domain = config.domain
    train = shards["train"]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))

    magnitude = torch.sqrt(domain.wave_number_squared())
    order = torch.argsort(magnitude)
    for split, shard in shards.items():
        spectrum = (
            torch.abs(torch.fft.fftn(shard.trajectories[:, 0], dim=domain.spatial_axes))
            ** 2
        ).mean(dim=0)
        axes[0, 0].semilogy(magnitude[order], spectrum[order] + 1e-40, ".-", label=split, alpha=0.7)
    axes[0, 0].axvline(config.initial_bandwidth, color="k", ls="--", label="k_train")
    axes[0, 0].set(xlabel="|k|", ylabel="mean energy", title="Initial-condition spectra")
    axes[0, 0].legend(fontsize=8)

    for index in range(4):
        axes[0, 1].plot(train.potential[index], alpha=0.8)
    axes[0, 1].set(xlabel="grid index", ylabel="V(x)", title="Potentials (train sample)")

    axes[0, 2].scatter(train.alpha, train.beta, s=6, alpha=0.5)
    axes[0, 2].set(xlabel="alpha", ylabel="beta", title="Parameter coverage")

    masses = l2_mass(train.trajectories[:, 0], domain)
    axes[1, 0].hist(masses.numpy(), bins=40)
    axes[1, 0].set(
        xlabel="mass", ylabel="count",
        title=f"Mass spread (ratio {float(masses.max()/masses.min()):.2f}x)",
    )

    amplitude = torch.abs(train.trajectories[0]).T
    image = axes[1, 1].imshow(
        amplitude.numpy(), aspect="auto", origin="lower", cmap="magma",
        extent=[0, train.n_frames - 1, 0, domain.shape[0]],
    )
    figure.colorbar(image, ax=axes[1, 1], label="|psi|")
    axes[1, 1].set(xlabel="step", ylabel="grid index", title="Trajectory 0")

    cascade = train.metadata
    axes[1, 2].semilogy(
        cascade["cascade_steps"], cascade["cascade_above_k_train"], "o-", label="> k_train"
    )
    axes[1, 2].semilogy(
        cascade["cascade_steps"], cascade["cascade_above_k_wrap"], "s-", label="> k_wrap"
    )
    axes[1, 2].set(
        xlabel="step", ylabel="max energy fraction",
        title="Nonlinear cascade (is G1 a clean control?)",
    )
    axes[1, 2].legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    figure.suptitle(f"Phase 1 dataset  [{config_hash(config)}]")
    figure.tight_layout()
    figure.savefig(output / "phase1_dataset.png", dpi=150)
    plt.close(figure)


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="tiny dataset for smoke runs")
    args = parser.parse_args()

    config = DataConfig()
    if args.quick:
        config = dataclasses.replace(
            config, n_train=32, n_val=8, n_test=8, steps=20
        )

    torch.set_default_dtype(torch.float64)
    identifier = config_hash(config)
    paths = shard_paths(DATA_ROOT, identifier)
    output = RESULTS_ROOT / f"phase1-{identifier}"
    (output / "plots").mkdir(parents=True, exist_ok=True)

    shards = {}
    for split in ("train", "val", "test"):
        print(f"generating {split} ...")
        shard = generate_shard(config, split)
        shard.save(paths[split])
        shards[split] = shard

    assert_no_leakage(shards)
    summary = summarize(shards, config)
    summary["config_hash"] = identifier
    (output / "metrics.json").write_text(json.dumps(summary, indent=2))
    make_plots(shards, config, output / "plots")

    train = summary["splits"]["train"]
    print(f"\n=== Phase 1 [{identifier}] ===")
    print(f"trajectories        : " + ", ".join(
        f"{s} {summary['splits'][s]['n_trajectories']}" for s in shards
    ))
    print(f"frames per traj     : {train['n_frames']}")
    print(f"one-step pairs      : {train['n_pairs']} (train)")
    print(f"requested bandwidth : {config.initial_bandwidth}")
    print(f"realized bandwidth  : {train['realized_bandwidth']}")
    print(f"mass ratio          : {train['mass_ratio']:.3f}x")
    print(f"alpha max gap       : {train['alpha_max_gap']:.2e}")
    print(f"leakage check       : passed (ids disjoint across splits)")
    print(
        "cascade > k_train   : "
        + " -> ".join(
            f"{s}:{v:.1e}"
            for s, v in zip(train["cascade_steps"], train["cascade_above_k_train"])
        )
    )
    print(
        "cascade > k_wrap    : "
        + " -> ".join(
            f"{s}:{v:.1e}"
            for s, v in zip(train["cascade_steps"], train["cascade_above_k_wrap"])
        )
    )
    total = sum(p.stat().st_size for p in paths.values()) / 1e6
    print(f"shards              : {total:.0f} MB in {paths['train'].parent}")
    return summary


if __name__ == "__main__":
    main()
