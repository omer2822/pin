"""Publication exports from saved paired ablation records (no new evaluation)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.run_hybrid_ablation import read_unit
from spno.evaluation.component_ablation import MODEL_NAMES

LABELS = {"C1": "Full C1", "exactK_learnedL": "Exact kinetic + learned local",
          "learnedK_exactL": "Learned kinetic + exact local", "exact_split": "Exact split step"}
COLORS = {"C1": "#bf4b45", "exactK_learnedL": "#21866c", "learnedK_exactL": "#7655a4", "exact_split": "#252b33"}
PANEL_METRICS = ("state_error", "phase_rms", "spectrum_error", "mass_drift", "energy_drift", "aligned_state_error")


def export_plots(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    options = manifest["options"]
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    status = "SMOKE" if options["smoke"] else ("EXPLORATORY / budget-bound" if manifest["exploratory"] else "Frozen checkpoint study")
    caption = f"{status} · {len(options['training_seeds'])} training seeds × {len(options['probe_seeds'])} probe seeds × {options['batch']} ICs"
    exported = []

    def save(fig, name):
        fig.suptitle(fig._suptitle.get_text() + "\n" + caption, fontsize=11)
        for extension in ("png", "pdf"):
            fig.savefig(figures / f"{name}.{extension}", dpi=160, bbox_inches="tight")
        exported.append(figures / f"{name}.png")
        plt.close(fig)

    # Main question first: paired hybrid/C1 improvement on every random-field arm.
    chosen = [r for r in summary["paired"] if r["endpoint"] == "final" and r["metric"] == "state_error"]
    if chosen:
        fig, ax = plt.subplots(figsize=(10, max(5, len(chosen)*.29)), layout="constrained")
        labels = []
        for index, row in enumerate(chosen):
            ratio = row["hybrid_over_C1"]
            good_reference = all(c["time_refinement_pass"] and c["space_refinement_pass"]
                                 for c in summary["reference_checks"] if c["case"] == row["case"])
            color = COLORS["exactK_learnedL"] if good_reference else "#c38320"
            if ratio["mean"] is not None:
                ax.plot([ratio["low"], ratio["high"]], [index, index], color=color)
                ax.scatter(ratio["mean"], index, color=color, s=20)
            labels.append(f"{row['case']} / {row['cohort']}" + (" *" if not good_reference else ""))
        ax.axvline(1, color="black", ls="--", lw=1)
        ax.set(yticks=range(len(labels)), yticklabels=labels, xscale="log",
               xlabel="Final state error: hybrid / C1 (geometric mean; paired 95% bootstrap interval)")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
        fig.suptitle("Does the hybrid generalize?  < 1 favors exact kinetic + learned local\n* temporal or spatial reference check failed / unchecked")
        save(fig, "01_all_arms_hybrid_vs_C1")

    for prefix in ("G4", "G9"):
        cases = [c for c in manifest["cases"] if c["name"].startswith(prefix + "-bandwidth-")]
        cases.sort(key=lambda c: c["config"]["initial_bandwidth"])
        if not cases:
            continue
        fig, axs = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
        for ax, metric in zip(axs.flat, PANEL_METRICS):
            for model in MODEL_NAMES:
                entries = [next(r for r in summary["rows"] if r["case"] == c["name"] and
                                r["model"] == model and r["metric"] == metric and r["endpoint"] == "final") for c in cases]
                xs = [c["config"]["initial_bandwidth"] for c in cases]
                ys = np.asarray([r["mean"] for r in entries], dtype=float)
                ax.plot(xs, np.maximum(ys, 1e-16), "o-", color=COLORS[model], label=LABELS[model], ms=3)
                # Separate seed axes remain in CSV; these are descriptive ranges of probe means.
                low = np.asarray([min(r["per_probe_seed"]) if None not in r["per_probe_seed"] else np.nan for r in entries])
                high = np.asarray([max(r["per_probe_seed"]) if None not in r["per_probe_seed"] else np.nan for r in entries])
                ax.fill_between(xs, np.maximum(low, 1e-16), np.maximum(high, 1e-16), color=COLORS[model], alpha=.10)
            ax.axvline(manifest["data"]["initial_bandwidth"], color="#777", ls="--")
            ax.set(xlabel="Input support k_max", ylabel=metric.replace("_", " "), yscale="log")
            ax.grid(alpha=.15)
        axs.flat[0].legend(fontsize=7)
        fig.suptitle(f"{prefix}: paired component interventions vs input spectral support\nShading: range of probe-seed means; no ICs excluded; see reference_checks.csv")
        save(fig, f"02_{prefix}_bandwidth_sweep")

    dispersion = read_unit(root / "dispersion.json.gz")
    for cohort, entries in dispersion.items():
        first = next(iter(entries.values()))
        k = np.asarray(first["k"])
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.7), layout="constrained")
        alpha_index = len(first["alphas"])//2
        for seed, entry in entries.items():
            curve = entry["curves"][len(entry["curves"])//2]
            for ax, key in zip(axs[:2], ("omega", "omega_centered")):
                ax.plot(k, np.asarray(curve[key])[alpha_index], label=f"training seed {seed}")
            if curve["alpha_secant_error"] is not None:
                error = np.max(np.abs(curve["alpha_secant_error"]), axis=0)
                axs[2].plot(k, np.maximum(error, 1e-16), label=f"training seed {seed}")
        exact = np.asarray(first["curves"][0]["exact"])[alpha_index]
        for ax in axs[:2]:
            ax.plot(k, exact, "k--", label="Exact αk²")
        for ax in axs:
            ax.axvline(first["training_bandwidth"], color="#555", ls=":", label="Training bandwidth")
            ax.axvline(-first["training_bandwidth"], color="#555", ls=":")
            ax.set(xlabel="Signed Fourier mode k")
            ax.grid(alpha=.2)
        axs[0].set(title="G5a: raw learned kinetic ωθ(k)", ylabel="Frequency")
        axs[1].set(title="Offset diagnostic: ωθ(k) − ωθ(0)", ylabel="Frequency")
        axs[2].set(title="G5b: max in-range α-secant error", ylabel="|Δωθ/Δα − k²|", yscale="log")
        axs[0].legend(fontsize=7)
        fig.suptitle(f"{cohort}: direct generator readout over the full resolvable spectrum\nMiddle training α and β shown; every α/β curve saved; no truth-assisted unwrapping")
        save(fig, f"03_dispersion_{cohort}")
        fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
        for model in MODEL_NAMES:
            curves = np.asarray([np.asarray(e["curves"][len(e["curves"])//2]["map_residual"][model])[alpha_index]
                                 for e in entries.values()])
            ax.plot(k, np.maximum(curves.mean(0), 1e-16), label=LABELS[model], color=COLORS[model])
        ax.axvline(first["training_bandwidth"], color="#777", ls="--")
        ax.axvline(-first["training_bandwidth"], color="#777", ls="--")
        ax.set(yscale="log", xlabel="Signed Fourier mode k", ylabel="|learned multiplier − exact multiplier|")
        ax.legend(fontsize=8)
        fig.suptitle(f"G5a / {cohort}: effective plane-wave map for all four models")
        save(fig, f"04_G5_map_{cohort}")

    for case in manifest["cases"]:
        name = case["name"]
        reference_units = [read_unit(root / "cases" / name / f"probe-{p}" / "reference.json.gz") for p in options["probe_seeds"]]
        refs = [u["reference"]["records"] for u in reference_units]
        for cohort in case["cohorts"]:
            units = [read_unit(root / "cases" / name / f"probe-{p}" / f"{cohort}-{s}.json.gz")
                     for s in options["training_seeds"] for p in options["probe_seeds"]]
            times = [r["time"] for r in refs[0]]
            fig, axs = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
            for ax, metric in zip(axs.flat, PANEL_METRICS):
                for model in MODEL_NAMES:
                    # Deliberately propagate nonfinite failures; never silently drop them.
                    vals = np.asarray([[np.asarray(r[metric], dtype=float).mean() for r in u["by_model"][model]["records"]] for u in units])
                    ax.plot(times, np.maximum(vals.mean(0), 1e-16), color=COLORS[model], label=LABELS[model])
                    ax.fill_between(times, np.maximum(vals.min(0), 1e-16), np.maximum(vals.max(0), 1e-16), color=COLORS[model], alpha=.1)
                if metric in ("mass_drift", "energy_drift"):
                    vals = np.asarray([[np.asarray(r[metric], dtype=float).mean() for r in rs] for rs in refs])
                    ax.plot(times, np.maximum(vals.mean(0), 1e-16), "k:", label="Refined reference")
                ax.set(xlabel="Physical time", ylabel=metric.replace("_", " "), yscale="log")
                ax.grid(alpha=.15)
            axs.flat[0].legend(fontsize=7)
            fig.suptitle(f"{name} / {cohort}: error and conservation over time\nShading: range across training × probe means; values below 1e−16 clipped for log display")
            save(fig, f"rollout_{name}_{cohort}")

            fig, axs = plt.subplots(1, 3, figsize=(15, 4.7), layout="constrained")
            n = case["config"]["grid_size"]
            k = np.fft.fftfreq(n, d=1/n)
            order = np.argsort(k)
            refpower = np.asarray([np.asarray(rs[-1]["power_spectrum"]).mean(0) for rs in refs]).mean(0)
            axs[0].plot(k[order], np.maximum(refpower[order], 1e-20), "k:", lw=2, label="Refined reference")
            for model in MODEL_NAMES:
                for ax, key in zip(axs[:2], ("power_spectrum", "error_spectrum")):
                    values = np.asarray([np.asarray(u["by_model"][model]["records"][-1][key], dtype=float).mean(0) for u in units]).mean(0)
                    ax.plot(k[order], np.maximum(values[order], 1e-20), color=COLORS[model], label=LABELS[model])
                tails = np.asarray([[np.asarray(r["above_training_band"], dtype=float).mean() for r in u["by_model"][model]["records"]] for u in units]).mean(0)
                axs[2].plot(times, tails, color=COLORS[model], label=LABELS[model])
            reference_tail = np.asarray([[np.mean(r["above_training_band"]) for r in rs] for rs in refs]).mean(0)
            axs[2].plot(times, reference_tail, "k:", label="Refined reference")
            for ax in axs[:2]:
                ax.axvline(manifest["data"]["initial_bandwidth"], color="#777", ls="--")
                ax.axvline(-manifest["data"]["initial_bandwidth"], color="#777", ls="--")
                ax.set(xlabel="Signed Fourier mode k", yscale="log")
            axs[0].set(ylabel="Final |ψ̂(k)|² (orthonormal FFT)")
            axs[1].set(ylabel="Final |ψ̂_model(k) − ψ̂_ref(k)|²")
            axs[2].set(xlabel="Physical time", ylabel="Power fraction above training band")
            axs[0].legend(fontsize=7)
            fig.suptitle(f"{name} / {cohort}: full spectra, spectral error, and cascade")
            save(fig, f"spectrum_{name}_{cohort}")

    for filename, rows in (("metrics", summary["rows"]), ("paired_comparisons", summary["paired"]),
                           ("reference_checks", summary["reference_checks"])):
        if rows:
            with (root / f"{filename}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)
    return exported
