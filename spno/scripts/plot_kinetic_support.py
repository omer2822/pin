"""Figures for the kinetic support ladder, from saved records only (no new evaluation)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.run_kinetic_support import PRIMARY, as_knee

SEED_COLORS = ("#2b6cb0", "#d9822b", "#21866c", "#7655a4", "#bf4b45")
CELL_COLORS = {"K0": "#2b6cb0", "K1": "#d9822b", "K2": "#21866c"}


def _load(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    data = json.loads((root / "data-checks.json").read_text())
    options = manifest["options"]
    records = {name: [json.loads((root / "cells" / name / f"seed{s}.json").read_text()) for s in options["seeds"]]
               for name in manifest["cells"]}
    return root, manifest, summary, data, options, records


def _cell_color(name, cells):
    kinetic = cells[name]["kinetic"]
    k0 = sorted(n for n, c in cells.items() if c["kinetic"] == "K0")
    if kinetic != "K0" or len(k0) < 2:
        return CELL_COLORS[kinetic]
    return plt.cm.Blues(0.45 + 0.5 * k0.index(name) / (len(k0) - 1))


def export_plots(root):
    root, manifest, summary, data, options, records = _load(root)
    cells, base = manifest["cells"], manifest["base"]
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    status = "SMOKE" if options["smoke"] else ("EXPLORATORY / budget-bound" if manifest.get("exploratory") else "trained study")
    caption = f"{status} · {len(options['seeds'])} seeds per cell · {options['epochs']}-epoch Phase 6 protocol · run {manifest['run_id']}"
    wrap = summary["k_wrap"]
    nyquist = base["grid_size"] // 2
    alpha0 = str(options["knee_alphas"][0])
    threshold = str(PRIMARY["threshold"])
    exported = []

    def save(fig, name, title):
        fig.suptitle(title + "\n" + caption, fontsize=11)
        for extension in ("png", "pdf"):
            fig.savefig(figures / f"{name}.{extension}", dpi=160, bbox_inches="tight")
        exported.append(figures / f"{name}.png")
        plt.close(fig)

    def support(name):
        return data[cells[name]["data_hash"]]["support"]

    # 1. The learned rate itself, one panel per cell.
    names = list(cells)
    columns = min(3, len(names))
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 3.9 * rows), squeeze=False, layout="constrained")
    for ax, name in zip(axes.flat, names):
        for index, record in enumerate(records[name]):
            rate = record["kinetic_rate"][alpha0]
            ax.plot(rate["centered_kappa"], color=SEED_COLORS[index % 5], label=f"seed {record['seed']}")
        ax.plot(records[name][0]["kinetic_rate"][alpha0]["truth"], "k--", lw=1.4, label=f"−αk², α={alpha0}")
        ax.axvspan(wrap["alpha_max"], wrap["alpha_min"], color="#e6b0aa", alpha=.25, label="k_wrap(α) band")
        ax.axvline(cells[name]["bandwidth"], color="#555", ls=":", label="training bandwidth")
        ax.axvline(support(name)["1e-06"], color="#21866c", ls="-.", label="support edge (1e-6)")
        ax.set(title=name, xlabel="k", ylabel="κθ(k) − κθ(0)", xlim=(0, nyquist))
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    save(fig, "01_kinetic_rate", "Learned kinetic rate per cell (figure only: above k_wrap the data fixes κ mod 2π/dt)")

    # 2. Knee vs training bandwidth: the headline.
    fig, ax = plt.subplots(figsize=(7.5, 5.2), layout="constrained")
    k0 = sorted((n for n in names if cells[n]["kinetic"] == "K0"), key=lambda n: cells[n]["bandwidth"])
    for name in names:
        knees = [as_knee(r["knees"][alpha0][threshold], base["grid_size"]) for r in records[name]]
        x = cells[name]["bandwidth"] + {"K0": 0, "K1": -.25, "K2": .25}[cells[name]["kinetic"]]
        ax.scatter([x] * len(knees), knees, color=CELL_COLORS[cells[name]["kinetic"]], alpha=.6,
                   label=cells[name]["kinetic"] if name == next(n for n in names if cells[n]["kinetic"] == cells[name]["kinetic"]) else None)
        ax.plot(x, np.median(knees), "_", color="k", ms=18)
    if k0:
        bands = [cells[n]["bandwidth"] for n in k0]
        ax.plot(bands, [support(n)["1e-06"] for n in k0], "-.", color="#21866c", label="data support edge (1e-6)")
        ax.plot(bands, bands, ":", color="#555", label="knee = training bandwidth")
    ax.axhspan(wrap["alpha_max"], wrap["alpha_min"], color="#e6b0aa", alpha=.25, label="k_wrap(α) band")
    ax.axhline(nyquist + 1, color="#999", lw=.8)
    ax.text(ax.get_xlim()[0], nyquist + 1.2, "no knee up to Nyquist", fontsize=8, color="#666")
    ax.set(xlabel="training initial bandwidth", ylabel=f"knee: first k with map residual > {threshold} (α={alpha0})")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    save(fig, "02_knee_vs_bandwidth", "Does the knee follow the training data? (dots: seeds, bars: median)")

    # 3. Map residual and 4. plane-wave G5b, per cell.
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), layout="constrained")
    for name in names:
        color = _cell_color(name, cells)
        residual = np.median([r["map_residual"][alpha0] for r in records[name]], axis=0)
        axes[0].semilogy(np.arange(len(residual))[1:], residual[1:] + 1e-16, color=color, label=name)
        ks = sorted(int(k) for k in summary["cells"][name]["G5b"])
        axes[1].semilogy(ks, [summary["cells"][name]["G5b"][str(k)] for k in ks], "o-", color=color, label=name)
    for value in options["knee_thresholds"]:
        axes[0].axhline(value, color="#999", ls=":", lw=.8)
    for ax in axes:
        ax.axvspan(wrap["alpha_max"], wrap["alpha_min"], color="#e6b0aa", alpha=.25)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set(xlabel="k", ylabel="|e^{-iω_θ dt} − e^{-iω dt}| (median over seeds)",
                title=f"One-step plane-wave map residual, α={alpha0} (gauge- and branch-free)")
    axes[1].axhline(1, color="k", ls=":", lw=.8)
    axes[1].set(xlabel="k", ylabel="max relative error of d arg m / dα (mean over seeds)",
                title="G5b plane-wave α-derivative")
    save(fig, "03_residual_and_g5b", "Where each cell leaves the true dispersion")

    # 5. G4 rollouts.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    for name in names:
        g4 = summary["cells"][name]["G4"]
        bands = sorted(int(b) for b in g4)
        for ax, metric in zip(axes, ("state_error", "aligned_state_error")):
            ax.semilogy(bands, [g4[str(b)][metric] for b in bands], "o-", color=_cell_color(name, cells), label=name)
    for ax, metric in zip(axes, ("raw state error", "phase-aligned state error")):
        ax.set(xlabel="test initial bandwidth", ylabel=f"{metric} after {options['rollout_steps']} steps")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    save(fig, "04_g4_rollout", "G4 rollouts on shared ICs and a refined reference (mean over ICs and seeds)")

    # 6. Training curves and 7. training-data spectra.
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), layout="constrained")
    for name in names:
        color = _cell_color(name, cells)
        for index, record in enumerate(records[name]):
            axes[0].semilogy(record["history"]["val_loss"], color=color, alpha=.7, label=name if index == 0 else None)
    axes[0].set(xlabel="epoch", ylabel="validation relative L2", title="Training (best checkpoint is kept)")
    done = set()
    for name in names:
        identifier = cells[name]["data_hash"]
        if identifier in done:
            continue
        done.add(identifier)
        spectrum = np.asarray(data[identifier]["spectrum"])
        axes[1].semilogy(np.arange(len(spectrum)), spectrum + 1e-300, label=f"train bw {cells[name]['bandwidth']}",
                         color=_cell_color(name, cells) if cells[name]["kinetic"] == "K0" else "#777")
    for value in options["support_thresholds"]:
        axes[1].axhline(value, color="#999", ls=":", lw=.8)
    axes[1].set(xlabel="|k|", ylabel="mean energy fraction (all trajectories and frames)",
                title="Training-data spectral support", ylim=(1e-20, 2))
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    save(fig, "05_training_and_support", "Budget and data support")
    return exported
