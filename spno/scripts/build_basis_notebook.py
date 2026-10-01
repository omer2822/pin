"""Build the self-contained Colab notebook for quick K0 polynomial surgery."""
from __future__ import annotations

import json
import textwrap

from scripts.build_hybrid_notebook import ROOT, embedded_source, setup_cell


def build_notebook():
    encoded, digest = embedded_source()
    cells = []

    def cell(kind, source):
        item = {'cell_type': kind, 'metadata': {}, 'id': f'basis-{len(cells):02d}',
                'source': textwrap.dedent(source).strip().splitlines(keepends=True)}
        if kind == 'code':
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell('markdown', r'''
    # Quick check: replace the saturating K0 head with a polynomial in k²

    **Upload this notebook directly to Colab and run top to bottom.** Source code is embedded.
    Put the existing Phase 6 checkpoint ZIP in MyDrive; the notebook finds it automatically.

    **Does it need retraining?** A new basis needs fitted coefficients. You cannot reinterpret
    tanh weights as polynomial weights, and merely swapping tanh for ReLU changes the function.
    This default experiment needs **no full-model retraining and no gradient optimization**:
    it distills the existing low-k K0 rates into a polynomial using one least-squares solve.
    That is a **post-hoc refit**, not a checkpoint-only architecture change. Set `head_epochs > 0`
    for optional short training of just the coefficient head against the original trajectories.

    The replacement is
    $$\kappa_{new}(k;\alpha,\beta)=\kappa_{old}(0;\alpha,\beta)
      +\sum_{j=1}^{d} c_j(\alpha,\beta)(k^2/k_{fit}^2)^j,\quad d\in\{1,3\}.$$
    Each coefficient is a **learned affine function of (α,β)**. The coefficient head receives
    only those parameters; no αk² feature, exact dispersion labels, or fixed physical coefficient
    is supplied. Multiplication by the k² basis happens at the output, as in the proposed model.
    The constant source rate is preserved because the frozen local head cancels that gauge offset.

    The degree-1 head imposes linear dependence on k² but learns its parameter dependence.
    Degree 3 allows curvature and can magnify tiny low-k fitting errors far outside support.
    **Removing a plateau alone does not establish accurate extrapolation.**
    ''')
    cell('markdown', '''
    ## 1. Settings

    Default: source seed 0, one fresh probe batch, 200-step G4/G9, degrees 1 and 3, fit only
    |k| ≤ 8 inside the original α/β box. This is an exploratory screen, not a matched training
    ablation. For a more reliable follow-up use seeds `[0,1,2]` and several fresh `probe_seeds`.
    `SMOKE=True` tests plumbing only and labels every verdict accordingly.

    Fine-tuning is OFF by default. To enable it, set `head_epochs=5` and use the full Phase 6
    artifacts containing train.pt and val.pt. Training uses only those splits; G4/G9 remain
    evaluation data. Checkpoint and source data hashes must match `SOURCE_CONFIG`.
    ''')
    cell('code', '''
    from pathlib import Path
    import os, sys, json, importlib.util

    IN_COLAB = importlib.util.find_spec("google") is not None and importlib.util.find_spec("google.colab") is not None
    SOURCE_ROOT = os.environ.get("SPNO_SOURCE_ROOT", "")
    CHECKPOINT_ARCHIVE = os.environ.get("SPNO_CHECKPOINT_ARCHIVE", "")
    SOURCE_CONFIG = os.environ.get("SPNO_SOURCE_CONFIG", "")  # JSON with original `data`, if non-default
    default_output = "/content/drive/MyDrive/spno/kinetic-basis" if IN_COLAB else "results/kinetic-basis"
    OUTPUT_ROOT = Path(os.environ.get("SPNO_OUTPUT_ROOT", default_output))
    SMOKE = False
    SETTINGS = {
        "seeds": [0], "probe_seeds": [1000], "probe_batch": 8,
        "degrees": [1, 3], "fit_k": 8, "parameter_points": 9,
        "head_epochs": 0, "head_pairs": 8192, "head_lr": 1e-4, "device": "cpu",
        "bandwidths": [8, 12, 16], "rollout_steps": 200, "stride": 25,
        "reference_substeps": 32, "reference_tolerance": 1e-4,
        "tail_threshold": 1e-6, "cascade_floor": 1e-8,
        "dispersion_tolerance": .1, "g4_tolerance": .1, "g9_tolerance": .25,
        "g1_ratio_guard": 1.1, "g1_absolute_floor": 1e-3,
    }
    if SMOKE:
        SETTINGS.update(probe_batch=2, rollout_steps=4, stride=1,
                        reference_substeps=2, smoke=True)
    SETTINGS.update(json.loads(os.environ.get("SPNO_OPTIONS", "{}")))
    if not IN_COLAB and not SOURCE_ROOT:
        candidates = [Path.cwd() / "results/phase6-standalone-artifacts",
                      Path.cwd() / "results/phase6-standalone-artifacts-quick"]
        SOURCE_ROOT = next((str(p) for p in candidates if (p / "checkpoints/phase6").is_dir()), "")
    print(json.dumps(SETTINGS, indent=2))
    print("Gradient training:", "HEAD ONLY" if SETTINGS["head_epochs"] else "OFF (least-squares refit only)")
    ''')
    cell('markdown', '## 2. Install embedded source and locate existing checkpoints')
    cell('code', setup_cell(digest, encoded))
    cell('markdown', r'''
    ## 3. Fit replacements and run paired G1 / G4 / G9 probes

    The source local head and zero-mode phase remain frozen. All candidates see the same
    initial conditions, parameters, and refined references. Report raw and phase-aligned G4
    errors, G9 spectrum and cascade-fraction errors, and the **centered unwrapped kinetic rate**.
    The exact −αk² curve is consulted only here, after fitting.

    Predeclared quick-screen thresholds:

    - Dispersion: ≤10% relative rate error above the fitted band through Nyquist, across
      a 3×3 α/β evaluation grid; the knee is the first nonzero integer mode above 10% error.
    - G4: aligned endpoint relative L2 ≤0.1 at every tested bandwidth above training support.
    - G1 control: raw and aligned errors ≤max(0.001, 1.1 × corresponding source K0 error).
    - G9: relative full-spectrum error and relative cascade-fraction error both ≤0.25.
    - References: agreement between 32 and 64 substeps ≤1e-4, spectral tail ≤1e-6,
      and reference cascade fraction ≥1e-8. Failure yields **inconclusive**.

    This checks time convergence and screens the spatial tail; it does not establish full
    spatial convergence. Values above k_wrap may have phase-alias ambiguity in training;
    unwrapped rates and one-step phase multipliers are different quantities.
    ''')
    cell('code', '''
    from dataclasses import fields
    from spno.config import DataConfig
    from scripts.run_kinetic_basis import run_check, plot_results

    BASE = DataConfig()
    if SOURCE_CONFIG:
        payload = json.loads(Path(SOURCE_CONFIG).read_text())
        data = payload.get("data", payload)
        allowed = {f.name for f in fields(DataConfig)}
        BASE = DataConfig(**{k: tuple(v) if k.endswith("_range") else v
                             for k, v in data.items() if k in allowed})
    RUN_ROOT, SUMMARY = run_check(SOURCE_ROOT, OUTPUT_ROOT, BASE, SETTINGS)
    FIGURE = plot_results(RUN_ROOT, SUMMARY)
    print("Saved:", RUN_ROOT)
    ''')
    cell('markdown', '## 4. Compare the candidates')
    cell('code', '''
    import html
    from IPython.display import display, HTML, Image
    display(Image(filename=str(FIGURE)))
    def fmt(value):
        return "—" if value is None else f"{value:.3g}"
    rows = []
    for row in SUMMARY["rows"]:
        baseline = next(r for r in SUMMARY["rows"] if r["seed"] == row["seed"]
                        and r["probe_seed"] == row["probe_seed"] and r["arm"] == "K0-tanh")
        g4 = " / ".join(f"bw{bw}: {fmt(v['raw'])}, {fmt(v['aligned'])}"
                         for bw, v in row["G4"].items())
        changes = []
        for bw, v in row["G4"].items():
            old = baseline["G4"][bw]["aligned"]
            if old > 1e-12:
                changes.append(f"bw{bw}: {v['aligned']/old:.2f}×")
        curves = SUMMARY["dispersion"][f"{row['seed']}/{row['arm']}"]["curves"]
        knees = ", ".join("none" if c["knee"] is None else str(c["knee"]) for c in curves)
        cells = [row["seed"], row["probe_seed"], row["arm"], knees, g4,
                 " / ".join(changes), fmt(row["G9_spectrum_error"]),
                 fmt(row["G9_fraction_error"]), row["verdict"]]
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in cells) + "</tr>")
    titles = ["seed", "probe", "arm", "rate knees (α/β grid)", "G1/G4 raw, aligned",
              "aligned / K0", "G9 spectrum", "G9 fraction", "screen verdict"]
    display(HTML("<table><tr>" + "".join(f"<th>{t}</th>" for t in titles)
                 + "</tr>" + "".join(rows) + "</table>"))
    print("Low-band fitting errors:", {name: value.get("fit_rmse") for name, value in SUMMARY["fits"].items()})
    print("Full metrics and provenance:", RUN_ROOT / "summary.json")
    ''')
    cell('markdown', '''
    ## Read the result

    A useful repair needs accurate high-k rates, lower G4/G9 errors, and a preserved G1
    control. Look at raw as well as aligned error and compare each row with the source
    checkpoint of the same training/probe seed. Passing thresholds is only a candidate
    result; inspect reference gates and the strength of the reference cascade.

    If degree 1 works and degree 3 blows up, the low-band teacher is not accurate enough to
    identify higher-order terms for long extrapolation. If neither works, that does **not**
    refute polynomial heads trained on trajectory data: distillation inherits teacher error.
    Try the head-only fine-tune before committing to a full retrain.

    If a candidate works, this supports **post-hoc engineering repair**. To claim that the
    original training plateau came from tanh's functional form, run paired from-scratch
    K0-tanh vs polynomial training with the same data, seeds, budget, and local head.
    Existing source checkpoints are never overwritten. Saved replacement `.pt` files require
    rebuilding the polynomial wrapper; they are not standard Phase 6 K0 checkpoints.
    ''')
    notebook = {'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
        'colab': {'name': '14_kinetic_basis_quick_check.ipynb', 'provenance': []}},
        'nbformat': 4, 'nbformat_minor': 5}
    output = ROOT / 'notebooks/14_kinetic_basis_quick_check.ipynb'
    output.write_text(json.dumps(notebook, indent=1) + '\n')
    return output


if __name__ == '__main__':
    print(build_notebook())
