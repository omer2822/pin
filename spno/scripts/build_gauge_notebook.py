"""Build notebook 12: gauge-identifiable C1 (reciprocal ablation, local law, C1g).

Run as ``python -m scripts.build_gauge_notebook`` from spno/. Self-contained like
notebook 11: the experiment source is embedded, so Colab needs no
git clone or bundle upload. Nothing is trained here; C1g is trained by 00_training.
"""
from __future__ import annotations

import json
import textwrap

from scripts.build_hybrid_notebook import ROOT, embedded_source, setup_cell


def build_notebook():
    encoded, digest = embedded_source()
    cells = []

    def cell(kind, source):
        item = {"cell_type": kind, "metadata": {}, "id": f"gauge-{len(cells):02d}",
                "source": textwrap.dedent(source).strip().splitlines(keepends=True)}
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell("markdown", r'''
    # Gauge-identifiable C1 — where does the OOD failure live?

    **Self-contained: upload this `.ipynb` to Colab and run all cells.** Nothing is trained here.

    Notebook 11 found that C1's spectral-shift failure sits in the learned kinetic operator, but also
    that the two learned halves are only defined **up to a gauge**: (κ, ν) and (κ − c, ν + c) give the
    same C1, and the Phase 6 checkpoints carry κθ(0) ≈ −25. This notebook finishes that story:

    | Section | Experiment | Needs |
    |---|---|---|
    | (a) | Reciprocal ablation stats on the saved run: K_exact+L_θ vs K_θ+L_exact | Drive run `3bf81deae4ef1ff9` |
    | (b) | Port cross-check, then gauged swaps on K axes (G4 spectral, G2-α) and L axes (G2-β, G3 V) | Phase 6 C1 checkpoints |
    | (c) | Local law: is L_θ ≈ βρ − V once the gauge is fixed? | (b), plus C1g if trained |
    | (d) | The same ablation on **C1g**, trained with κθ(0) = 0 from the start | C1g from `00_training` (`C1G_LAMBDAS=[0, 0.01]`) |

    ## Pre-registered expectations (written before the C1g results)

    1. **The gauge fix is an exact reparameterization**, not a smaller model class: the local net's
       free bias absorbs any constant. C1g therefore tests *identifiability and optimization*, not
       expressivity. At the same seed, C1 and C1g start from bitwise-identical weights and use the
       same 40-epoch protocol. **But the λ=0 C1 is the Phase 6 import trained on Windows, while C1g
       is trained in Colab**, so small λ=0 differences may be hardware, not gauge. The clean paired
       contrast is **C1+PDE vs C1g+PDE at λ=0.01**: both go through the same workflow code path.
    2. **C1g is not expected to repair bandwidth-16 error of the full model.** The learned dispersion
       plateaus above |k| ≈ 9 because the training ICs have no energy there — a data-support limit
       that no gauge choice touches.
    3. **C1g is expected to** (i) close the raw-vs-aligned gap, (ii) make both component swaps
       meaningful *without* any post-hoc correction, and (iii) possibly move the effective
       nonlinearity from ≈ 0.88β toward β.
    4. **Double dissociation test.** The kinetic half reads (k², α, β); the local half reads (ρ, V, α, β).
       If the kinetic half is what fails under spectral shift, K_exact+L_θ should survive G4 and
       G2-α-only while K_θ+L_exact fails; under G2-β-only and G3 (V) the reverse pattern implicates
       the local half. The joint G2 moves α and β together and stresses **both** halves.
    ''')
    cell("markdown", '''
    ## 1. Settings

    `SMOKE=True` is a plumbing check on tiny settings, never a result. Section (b) and (d) write
    resumable runs under `OUTPUT_ROOT`; rerunning after a disconnect reuses completed units.
    ''')
    cell("code", '''
    from pathlib import Path
    import os, sys, json, importlib.util

    SOURCE_ROOT = os.environ.get("SPNO_SOURCE_ROOT", "")          # Phase 6 checkpoints (auto-located)
    CHECKPOINT_ARCHIVE = os.environ.get("SPNO_CHECKPOINT_ARCHIVE", "")
    SOURCE_CONFIG = os.environ.get("SPNO_SOURCE_CONFIG", "")      # optional JSON with original `data`
    OUTPUT_ROOT = Path(os.environ.get("SPNO_OUTPUT_ROOT", "/content/drive/MyDrive/spno/gauge-study"))
    HYBRID_RUN = Path(os.environ.get("SPNO_HYBRID_RUN",
                                     "/content/drive/MyDrive/spno/hybrid-ablation/3bf81deae4ef1ff9"))
    WORKFLOW_ROOT = Path(os.environ.get("SPNO_WORKFLOW_ROOT", "/content/drive/MyDrive/spno/workflow"))
    SMOKE = False
    SETTINGS = {
        "training_seeds": [0, 1, 2], "probe_seeds": [1000, 1001, 1002, 1003, 1004],
        "batch": 16, "bandwidths": [4, 8, 12, 16, 24],
        "short_steps": 200, "long_steps": 2000, "stride": 50,
        "reference_substeps": 32, "reference_tolerance": 1e-4,
        "spatial_samples": 2, "spatial_tolerance": 1e-3, "tail_threshold": 1e-6,
        "device": "cpu", "threads": 2, "allow_budget_bound": True, "bootstrap_draws": 2000,
        "gauge": True,
        # K axes: G4 spectral, G2-alpha. L axes: G2-beta, G3 (V). Joint G2; G1 control.
        "case_names": ["G1-interpolation", "G2-extrapolation", "G2-alpha-only",
                       "G2-beta-only", "G3-potential-strong",
                       "G3-potential-short", "G4-bandwidth-4", "G4-bandwidth-8",
                       "G4-bandwidth-12", "G4-bandwidth-16", "G4-bandwidth-24"],
    }
    # Section (b)/(c) probes: one probe batch, as in the 2026-09-23 Colab cells.
    PROBE = {"probe_seed": 1000, "batch": 16, "steps": 200, "reference_substeps": 64,
             "local_law_batch": 256}
    if SMOKE:
        SETTINGS.update(training_seeds=[0], probe_seeds=[1000, 1001], batch=2, short_steps=2,
                        long_steps=4, stride=1, reference_substeps=2, spatial_samples=1,
                        bootstrap_draws=50, smoke=True)
        PROBE.update(batch=2, steps=3, reference_substeps=4, local_law_batch=8)
    SETTINGS.update(json.loads(os.environ.get("SPNO_OPTIONS", "{}")))
    PROBE.update(json.loads(os.environ.get("SPNO_PROBE", "{}")))
    IN_COLAB = importlib.util.find_spec("google") is not None and importlib.util.find_spec("google.colab") is not None
    print(json.dumps({"settings": SETTINGS, "probe": PROBE}, indent=2))
    ''')
    cell("markdown", '''
    ## 2. Install the embedded code and locate the Phase 6 checkpoints
    ''')
    cell("code", setup_cell(digest, encoded))
    cell("markdown", r'''
    ## (a) Reciprocal ablation on the saved G1–G9 run — no new rollouts

    Run `3bf81deae4ef1ff9` already saved K_θ+L_exact next to K_exact+L_θ, but its summary only paired
    the hybrid with C1. `resummarize` rebuilds the summary from the saved units **without** a new
    run identity; the previous `summary.json` is kept as `summary.v1.json`.

    These swaps are **raw** (no gauge fix), so K_θ+L_exact carries the full κθ(0) ≈ −25 offset as a
    global phase. Read its raw error as "the gauge is wrong", not "the kinetic shape is wrong";
    the aligned error and section (b) separate the two.
    ''')
    cell("code", '''
    import html
    from IPython.display import display, Image, HTML
    from scripts.run_hybrid_ablation import (resummarize, run_study, load_c1_cohorts,
                                            load_c1g_cohort, clean_json)
    from scripts.plot_hybrid_ablation import export_plots
    from spno.artifacts import atomic_json

    def reciprocal_table(summary, metric="state_error", cohort="base"):
        rows = [r for r in summary["paired"] if r["endpoint"] == "final" and r["metric"] == metric
                and r["cohort"] == cohort and "model" in r]
        header = "<tr><th>case</th><th>model / baseline</th><th>ratio (geo. mean)</th><th>95% interval</th><th>status</th></tr>"
        body = ""
        for r in rows:
            ratio = r["ratio"]
            mean = "—" if ratio["mean"] is None else f"{ratio['mean']:.3g}"
            interval = "—" if ratio["low"] is None else f"[{ratio['low']:.3g}, {ratio['high']:.3g}]"
            body += (f"<tr><td>{html.escape(r['case'])}</td><td>{html.escape(r['model'])} / "
                     f"{html.escape(r['baseline'])}</td><td>{mean}</td><td>{interval}</td>"
                     f"<td>{html.escape(r['status'])}</td></tr>")
        caption = (f"<p>Final {metric.replace('_', ' ')}, paired ratio; &lt; 1 favors the first model. "
                   "Descriptive intervals, not multiplicity-adjusted.</p>")
        return caption + "<table>" + header + body + "</table>"

    def show_run(run, label):
        summary = json.loads((run / "summary.json").read_text())
        export_plots(run)
        print(label)
        for metric in ("state_error", "aligned_state_error"):
            display(HTML(reciprocal_table(summary, metric)))
        figure = run / "figures" / "07_reciprocal_ablation.png"
        if figure.exists():
            display(Image(filename=str(figure)))
        return summary

    if (HYBRID_RUN / "manifest.json").exists():
        resummarize(HYBRID_RUN)
        saved_summary = show_run(HYBRID_RUN, f"(a) saved run {HYBRID_RUN.name}")
    else:
        print(f"(a) skipped: no saved run at {HYBRID_RUN}")
    ''')
    cell("markdown", r'''
    ## (b) Port cross-check, then gauged swaps on C1

    The gauge-fixed swaps and the drift/δq probes were first run as ad hoc Colab cells on
    2026-09-23; they now live in `spno.evaluation.component_ablation`. **Before trusting any new
    number, this cell must reproduce the recorded ones:**

    | Check | Recorded 2026-09-23 |
    |---|---|
    | Gauge move on full C1 | identical to C1 at 1e-12 |
    | Gauge-fixed hybrid, bw 8, 200 steps | raw ≈ 6e-2, aligned ≈ 4e-3 |
    | Gauge-fixed hybrid, bw 16 | aligned ≈ 5.4e-3 (C1: 0.88) |
    | Drift predictor Σ dt⟨δq⟩_ρ | corr 1.000, slope 1.000, residual ≤ 0.3% |
    | δq(ρ) slope | ≈ −0.12β on every seed (effective β ≈ 0.88β) |
    ''')
    cell("code", '''
    import torch, numpy as np
    from spno.config import DataConfig
    from spno.evaluation.component_ablation import (
        ComponentSplitStep, component_models, delta_q_profile, local_law, predicted_phase_drift, probe_cases, sample_probe)
    from spno.solvers.split_step import SubsteppedReference

    torch.set_num_threads(SETTINGS["threads"])
    declared = DataConfig(**json.loads(Path(SOURCE_CONFIG).read_text())["data"]) if SOURCE_CONFIG else None
    data_config, c1_cohorts, _ = load_c1_cohorts(SOURCE_ROOT, declared, SETTINGS["training_seeds"],
                                                 allow_budget_bound=SETTINGS["allow_budget_bound"])
    dt = data_config.dt
    bands = [b for b in (8, 12, 16) if b <= data_config.grid_size // 2] or [data_config.initial_bandwidth]
    cases = {c.name: c for c in probe_cases(data_config, bands)}
    check_names = ["G1-interpolation"] + [f"G4-bandwidth-{b}" for b in bands]

    def errors(pred, ref):
        raw = ((pred - ref).norm(dim=-1) / ref.norm(dim=-1)).mean().item()
        inner = (ref.conj() * pred).sum(-1)
        aligned = pred * (inner / inner.abs()).conj().unsqueeze(-1)
        return raw, ((aligned - ref).norm(dim=-1) / ref.norm(dim=-1)).mean().item()

    port_check = {}
    rho_samples = sample_probe(cases["G1-interpolation"], PROBE["probe_seed"],
                               PROBE["local_law_batch"])[0][0].abs().square().numpy()
    with torch.inference_mode():
        for seed, c1 in c1_cohorts["base"].items():
            record = port_check.setdefault(str(seed), {})
            parts = component_models(c1, gauge=True)
            both = ComponentSplitStep(c1, exact_kinetic=False, exact_local=False, gauge="zero_mode")
            inputs, _ = sample_probe(cases["G1-interpolation"], PROBE["probe_seed"], PROBE["batch"])
            identity = (both(*inputs, dt) - c1(*inputs, dt)).abs().max().item()
            assert identity < 1e-12, identity
            record["identity_max_abs"] = identity
            for name in check_names:
                cfg = cases[name].config
                inputs, _ = sample_probe(cases[name], PROBE["probe_seed"], PROBE["batch"])
                x, V, a, b = inputs
                ops = {"reference": SubsteppedReference(cfg.domain, PROBE["reference_substeps"]), "C1": c1,
                       "raw hybrid": parts["exactK_learnedL"], "gauge-fixed hybrid": parts["exactK_learnedL_g"],
                       "raw reverse": parts["learnedK_exactL"], "gauge-fixed reverse": parts["learnedK_exactL_g"]}
                state = {k: x.clone() for k in ops}
                for _ in range(PROBE["steps"]):
                    state = {k: op(state[k], V, a, b, cfg.dt) for k, op in ops.items()}
                for k in list(ops)[1:]:
                    raw, al = errors(state[k], state["reference"])
                    record.setdefault(name, {})[k] = {"raw": raw, "aligned": al}
                    print(f"seed {seed} {name:18s} {k:20s} raw={raw:.3e} aligned={al:.3e}")
            drift_case = cases[check_names[1] if len(check_names) > 1 else check_names[0]]
            inputs, _ = sample_probe(drift_case, PROBE["probe_seed"], PROBE["batch"])
            drift = predicted_phase_drift(c1, inputs, drift_case.config.dt, steps=PROBE["steps"],
                                          reference_substeps=PROBE["reference_substeps"])
            record["drift"] = drift["stats"]
            for label, stats in drift["stats"].items():
                print(f"seed {seed} drift [{label:13s}] corr={stats['corr']:+.3f} slope={stats['slope']:.3f} "
                      f"residual={stats['relative_residual']:.3f}")
            profile = delta_q_profile(c1, rho_samples, shift=True)
            record["delta_q"] = profile["fits"]
            print(f"seed {seed} delta-q slope/beta:",
                  ", ".join(f"β={f['beta']:+.1f}: {f['slope_over_beta']:+.3f}" for f in profile["fits"]
                            if f["slope_over_beta"] is not None))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    # Recorded 2026-09-23 values, with explicit tolerances. CHECK means "look before trusting".
    verdicts = []
    for seed, record in port_check.items():
        hybrid = record.get("G4-bandwidth-8", {}).get("gauge-fixed hybrid")
        if hybrid:
            verdicts.append((seed, "bw8 gauged hybrid aligned ≈ 4e-3 (×2)", 2e-3 <= hybrid["aligned"] <= 8e-3))
            verdicts.append((seed, "bw8 gauged hybrid raw ≈ 6e-2 (×2)", 3e-2 <= hybrid["raw"] <= 1.2e-1))
        for label, stats in record["drift"].items():
            verdicts.append((seed, f"drift slope 1.000±0.01, corr>0.999 [{label}]",
                             abs(stats["slope"] - 1) < .01 and stats["corr"] > .999))
        ratios = [f["slope_over_beta"] for f in record["delta_q"] if f["slope_over_beta"] is not None]
        verdicts.append((seed, "δq slope/β in [−0.17, −0.07]", all(-.17 <= r <= -.07 for r in ratios)))
    for seed, label, ok in verdicts:
        print(f"{'PASS ' if ok else 'CHECK'} seed {seed}: {label}")
    port_check["verdicts"] = [{"seed": s, "check": l, "pass": bool(ok)} for s, l, ok in verdicts]
    atomic_json(OUTPUT_ROOT / "port_check.json", clean_json(port_check))
    ''')
    cell("markdown", r'''
    ### Gauged swaps on the selected axes (new resumable run)

    Same frozen C1 checkpoints, now with **six** operators per case: C1, both raw swaps, both
    gauge-fixed swaps, and the exact split step. The 2026-09-23 result predicts that the gauged
    K_θ+L_exact loses its huge raw error while keeping the kinetic *shape* error at high bandwidth.
    ''')
    cell("code", '''
    C1_RUN = run_study(SOURCE_ROOT, OUTPUT_ROOT / "c1", data=data_config, options=SETTINGS)
    c1_summary = show_run(C1_RUN, f"(b) C1 gauged swaps: {C1_RUN}")
    ''')
    cell("markdown", r'''
    ## (c) Local law — is L_θ ≈ βρ − V?

    Fit, on every grid point of one G1 probe batch (the data distribution), with ρ weights
    (the global-phase weighting):
    ν ≈ c_βρ·(βρ) + c_V·V + c₀ + c_α·(α − ᾱ). The truth is (1, −1, 0, 0).

    * **C1 raw** — the local net as trained; its intercept carries the gauge constant.
    * **C1 shifted** — ν + κθ(0), the post-hoc gauge fix. The intercept is removed by construction,
      so the informative comparison is **C1 shifted vs C1g** on c_βρ and c_V.
    * **C1g** — trained with κθ(0) = 0; no shift exists to apply.
    * **C1+PDE shifted vs C1g+PDE** (λ=0.01) — the headline pair: same workflow code path and runtime,
      same seeds and initialization, differing only in the gauge. Shown once both are trained.
    ''')
    cell("code", '''
    import matplotlib.pyplot as plt
    try:
        c1g_cohorts = load_c1g_cohort(WORKFLOW_ROOT, data_config, SETTINGS["training_seeds"])
    except ValueError as error:
        c1g_cohorts = None
        print("C1g not available yet:", error)
        print("Train it in 00_training.ipynb: TRAIN_PHASES=[7], C1G_LAMBDAS=[0.0, 0.01].")

    # The clean pair: C1+PDE and C1g+PDE at lambda=0.01, both trained by the same workflow path.
    pde_pair = {}
    for family in ("C1", "C1g"):
        try:
            pde_pair[family] = load_c1g_cohort(WORKFLOW_ROOT, data_config, SETTINGS["training_seeds"],
                                               name=family, weight=0.01)[0]["base"]
        except ValueError as error:
            print(f"{family}+PDE (λ=0.01) not available:", error)

    law_inputs, _ = sample_probe(cases["G1-interpolation"], PROBE["probe_seed"], PROBE["local_law_batch"])
    variants = {"C1 raw": (c1_cohorts["base"], False), "C1 shifted": (c1_cohorts["base"], True)}
    if c1g_cohorts:
        variants["C1g"] = (c1g_cohorts[0]["base"], False)
    if len(pde_pair) == 2:
        variants["C1+PDE shifted"] = (pde_pair["C1"], True)
        variants["C1g+PDE"] = (pde_pair["C1g"], False)
    laws, profiles = {}, {}
    with torch.inference_mode():
        for label, (by_seed, shift) in variants.items():
            for seed, model in by_seed.items():
                laws.setdefault(label, {})[str(seed)] = local_law(model, law_inputs, shift=shift)
                profiles.setdefault(label, {})[str(seed)] = delta_q_profile(model, rho_samples, shift=shift)
    keys = ("beta_rho", "potential", "intercept", "alpha", "relative_rmse_vs_truth", "unexplained_relative_rmse")
    header = "<tr><th>variant</th><th>seed</th>" + "".join(f"<th>{k}</th>" for k in keys) + "</tr>"
    body = "".join(f"<tr><td>{label}</td><td>{seed}</td>" + "".join(f"<td>{law[k]:+.4f}</td>" for k in keys) + "</tr>"
                   for label, by_seed in laws.items() for seed, law in by_seed.items())
    display(HTML("<p>Truth: beta_rho=+1, potential=−1, intercept=0, alpha=0.</p><table>" + header + body + "</table>"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), layout="constrained")
    colors = {"C1 raw": "#bf4b45", "C1 shifted": "#e39a8f", "C1g": "#21866c",
              "C1+PDE shifted": "#d9822b", "C1g+PDE": "#2b6cb0"}
    for label, by_seed in profiles.items():
        if label == "C1 raw":
            continue  # its intercept (the gauge constant) dwarfs the shape
        for number, (seed, profile) in enumerate(by_seed.items()):
            middle = len(profile["fits"]) // 2 + 1  # a nonzero beta
            axes[0].plot(profile["rho_grid"], profile["curves"][middle], color=colors[label],
                         label=f"{label} (β={profile['fits'][middle]['beta']:+.1f})" if number == 0 else None)
    axes[0].axhline(0, color="k", lw=.8)
    axes[0].set(xlabel="Density ρ", ylabel="δq = learned − βρ at V=0, α=0.9", title="Local-rate bias")
    labels = list(laws)
    for index, key in enumerate(("beta_rho", "potential")):
        means = [np.mean([law[key] for law in laws[l].values()]) for l in labels]
        spread = [np.ptp([law[key] for law in laws[l].values()]) / 2 for l in labels]
        axes[1].bar(np.arange(len(labels)) + .38 * index, means, .36, yerr=spread, capsize=3,
                    label={"beta_rho": "c_βρ (truth +1)", "potential": "c_V (truth −1)"}[key])
    axes[1].axhline(1, color="#555", ls=":"); axes[1].axhline(-1, color="#555", ls=":")
    axes[1].set_xticks(np.arange(len(labels)) + .19, labels)
    axes[1].set(title="Fitted local law (mean ± half-range over seeds)")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.savefig(OUTPUT_ROOT / "local_law.png", dpi=150)
    plt.show()
    atomic_json(OUTPUT_ROOT / "local_law.json", clean_json({"laws": laws, "profiles": profiles}))
    ''')
    cell("markdown", r'''
    ## (d) The same ablation on C1g

    Pre-registered: K_exact+L_θ and K_θ+L_exact of C1g should behave like the **gauged** swaps of C1
    without any correction (the gauged and raw swaps coincide, since κθ(0) = 0 already), and the
    hybrid's raw error should sit close to its aligned error. The full-model error at bandwidth 16 is
    **not** expected to improve (data-support limit). G6a/G7 rows are absent: those cohorts were not
    retrained.
    ''')
    cell("code", '''
    if c1g_cohorts:
        C1G_RUN = run_study(SOURCE_ROOT, OUTPUT_ROOT / "c1g", data=data_config, options=SETTINGS,
                            cohorts=c1g_cohorts)
        c1g_summary = show_run(C1G_RUN, f"(d) C1g: {C1G_RUN}")
        with torch.inference_mode():
            for seed, model in c1g_cohorts[0]["base"].items():
                drift_case = cases[check_names[1] if len(check_names) > 1 else check_names[0]]
                inputs, _ = sample_probe(drift_case, PROBE["probe_seed"], PROBE["batch"])
                stats = predicted_phase_drift(model, inputs, drift_case.config.dt, steps=PROBE["steps"],
                                              reference_substeps=PROBE["reference_substeps"])["stats"]
                print(f"C1g seed {seed} drift:", {k: round(v["slope"], 3) for k, v in stats.items()})
    else:
        print("(d) skipped: train C1g first (see section (c)).")
    ''')
    cell("markdown", r'''
    ## Reading the results

    * **Localization claim** — needs the dissociation: G4 hurts the swaps that keep K_θ, G2/G3 hurt
      the swaps that keep L_θ, in the *gauged* rows. Raw-row differences alone can be pure gauge.
    * **Gauge claim** — lead with C1+PDE vs C1g+PDE (λ=0.01, same code path). The λ=0 C1 vs C1g rows
      share initialization and protocol but not training hardware. A C1g win on raw error with an
      unchanged aligned error means the gauge was costing phase, not shape.
    * **Local-law claim** — compare C1 shifted with C1g on c_βρ. If C1g moves toward 1, gauge
      ambiguity was also hurting the *learning* of the nonlinearity; if both stay near 0.88, the
      bias is an L0 optimization limit (the L1/L2 rungs are the follow-up).
    * All source checkpoints are budget-bound (40 epochs): every number here is **exploratory**.
    ''')
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}, "colab": {"name": "12_gauge_identifiable_c1.ipynb", "provenance": []}},
        "nbformat": 4, "nbformat_minor": 5}
    output = ROOT / "notebooks/12_gauge_identifiable_c1.ipynb"
    output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    return output


if __name__ == "__main__":
    print(build_notebook())
