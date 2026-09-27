"""Build notebook 13: the kinetic support ladder (training bandwidth x K0/K1/K2).

Run as ``python -m scripts.build_support_notebook`` from spno/. Self-contained like
notebooks 11/12 (the source is embedded), but unlike them it TRAINS: C1 variants with the
Phase 6 base protocol. The Phase 6 artifacts are optional here -- only the port check and
the bw8 data-reproduction check use them.
"""
from __future__ import annotations

import json
import textwrap

from scripts.build_hybrid_notebook import ROOT, embedded_source

#: Installs the embedded source; Phase 6 artifacts are looked up but never required.
SETUP = '''
    import base64, hashlib, io, subprocess, zipfile
    from pathlib import PurePosixPath

    EMBEDDED_SOURCE_SHA256 = "__DIGEST__"
    EMBEDDED_SOURCE = """__SOURCE__"""

    def safe_extract(archive, destination):
        destination = Path(destination).resolve()
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts or "\\\\" in member.filename:
                raise ValueError("Unsafe archive member: " + member.filename)
            if not (destination / member.filename).resolve().is_relative_to(destination):
                raise ValueError("Archive member escapes destination")
        archive.extractall(destination)

    if IN_COLAB:
        from google.colab import drive
        drive.mount("/content/drive")
    local_project = os.environ.get("SPNO_PROJECT_ROOT")
    if local_project:
        PROJECT_ROOT = Path(local_project)
    else:
        raw = base64.b64decode(EMBEDDED_SOURCE)
        assert hashlib.sha256(raw).hexdigest() == EMBEDDED_SOURCE_SHA256
        base_dir = Path("/content") if IN_COLAB else Path.cwd()
        PROJECT_ROOT = base_dir / ("spno-support-code-" + EMBEDDED_SOURCE_SHA256[:12])
        PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            assert archive.testzip() is None
            safe_extract(archive, PROJECT_ROOT)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(PROJECT_ROOT)])
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    # Optional: the Phase 6 artifacts, for the port check and the bw8 data reproduction.
    if not SOURCE_ROOT and USE_PHASE6_ARTIFACTS:
        existing = [Path("/content/pin/spno/results/phase6-standalone-artifacts")]
        existing += sorted(Path("/content").glob("spno-hybrid-artifacts-*/**/phase6-standalone-artifacts"))
        found = [p for p in existing if (p / "checkpoints" / "phase6").is_dir()]
        if found:
            SOURCE_ROOT = found[0]
        elif IN_COLAB:
            names = ["phase6-eval-only-bd4e108527-K0.zip", "phase6-standalone-artifacts-bd4e108527-K0.zip"]
            hits = [p for name in names for p in Path("/content/drive/MyDrive").rglob(name)]
            if hits:
                archive_path = sorted(hits)[0]
                print("Extracting", archive_path, "(one time per runtime)")
                extracted = Path("/content") / ("spno-hybrid-artifacts-" + archive_path.stem[-12:])
                with zipfile.ZipFile(archive_path) as archive:
                    safe_extract(archive, extracted)
                roots = [p.parent for p in extracted.rglob("checkpoints") if (p / "phase6").is_dir()]
                SOURCE_ROOT = roots[0] if len(roots) == 1 else ""
    SOURCE_ROOT = Path(SOURCE_ROOT) if SOURCE_ROOT else None
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("Embedded source:", EMBEDDED_SOURCE_SHA256)
    print("Phase 6 artifacts:", SOURCE_ROOT or "not found -- port check and bw8 reproduction will be skipped")
    print("Results (Drive):", OUTPUT_ROOT)
    print("Data (regenerable, this runtime):", DATA_ROOT)
'''


def build_notebook():
    encoded, digest = embedded_source()
    cells = []

    def cell(kind, source):
        item = {"cell_type": kind, "metadata": {}, "id": f"support-{len(cells):02d}",
                "source": textwrap.dedent(source).strip().splitlines(keepends=True)}
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell("markdown", r'''
    # Kinetic support ladder — why does C1's learned dispersion go flat at k ≈ 10?

    **Self-contained: upload this `.ipynb` to Colab and run all cells.** This notebook **trains**
    (18 small C1 runs by default). It resumes after a disconnect: checkpoints and per-epoch
    progress are saved to Drive.

    Every Phase 6 C1 checkpoint is K0 trained on data with initial bandwidth 8, and all of them
    reproduce −αk² up to |k| ≈ 9 and are flat (≈ −80) beyond. Two explanations were never tested:

    1. **Data support** — the kinetic MLP is fitted only where the training data has energy.
    2. **Functional form** — a bounded tanh MLP cannot follow an unbounded k².

    Each cell below changes **one** factor from Phase 6 C1 (L0 local, width 32, free gauge,
    40 epochs, batch 256, lr 1e-3, patience 8, 800 training trajectories):

    | cell | kinetic | train bandwidth | role |
    |---|---|---|---|
    | K0-bw8 | K0 | 8 | control; same init and data as the Windows C1 |
    | K0-bw10 | K0 | 10 | third point below the wrap limit |
    | **K0-bw12** | K0 | 12 | **primary contrast** with bw8 |
    | K0-bw16 | K0 | 16 | straddles k_wrap(α) = 16.9–21.2: secondary |
    | K1-bw8 | K1 (α·k² as a feature) | 8 | does the product feature help? |
    | K2-bw8 | K2 (κ = −αk²(1+MLP)) | 8 | form imposed — an ablation, not a headline |

    **The knee.** For a plane wave, the one-step map is e^{−iω_θ dt}. The knee is the first k whose
    residual |e^{−iω_θ dt} − e^{−iω dt}| exceeds 0.05 at α = 0.9, β = 0.3 (0.01/0.2 and α = 0.7 also
    reported). This is gauge-free (κ and ν offsets cancel) and branch-free (it only sees what the data
    constrains). The centered κθ(k) − κθ(0) is shown as a figure only: above k_wrap the one-step data
    fixes κ only modulo 2π/dt.

    ## Pre-registered decision rule (written before any result)

    * **Data support SUPPORTED** — the K0-bw12 knee ≥ the K0-bw8 knee + 2 (half the bandwidth increase)
      on ≥ 2 of 3 seeds (paired by seed), and the K0-bw10 median lies between.
    * **Architecture limit INDICATED** — every K0 knee within ±1 of the bw8 median, **and** K0-bw12 is
      not budget-bound (validation improving by > 5% over the last 5 epochs ⇒ *inconclusive, rerun with
      more epochs*).
    * Anything else is inconclusive. Cells whose training data fails the resolution checks are flagged.

    Expectations: K1-bw8 knee ≈ K0-bw8 (its output is still tanh-bounded); K2 accurate to Nyquist
    (it is told the law). All results are 40-epoch, budget-bound, and **exploratory**, like Phase 6.
    ''')
    cell("markdown", '''
    ## 1. Settings

    `SMOKE=True` is a plumbing check on a tiny dataset and 1 epoch — never a result.
    Edit `SETTINGS["cells"]` to add cells (for example `{"kinetic": "K1", "bandwidth": 16}`).
    Changing settings starts a new run directory; the device does not.
    ''')
    cell("code", '''
    from pathlib import Path
    import os, sys, json, importlib.util

    SMOKE = False
    USE_PHASE6_ARTIFACTS = True   # port check + bw8 data reproduction; skipped if not found
    SOURCE_ROOT = os.environ.get("SPNO_SOURCE_ROOT", "")
    OUTPUT_ROOT = Path(os.environ.get("SPNO_OUTPUT_ROOT", "/content/drive/MyDrive/spno/kinetic-support"))
    DATA_ROOT = Path(os.environ.get("SPNO_DATA_ROOT", "/content/spno-support-data"))
    SETTINGS = {
        "cells": [{"kinetic": "K0", "bandwidth": 8}, {"kinetic": "K0", "bandwidth": 10},
                  {"kinetic": "K0", "bandwidth": 12}, {"kinetic": "K0", "bandwidth": 16},
                  {"kinetic": "K1", "bandwidth": 8}, {"kinetic": "K2", "bandwidth": 8}],
        "seeds": [0, 1, 2], "epochs": 40, "device": "auto", "threads": 2,
    }
    BASE_OVERRIDES = {}  # DataConfig fields; empty = the Phase 6 training distribution
    if SMOKE:
        OUTPUT_ROOT = OUTPUT_ROOT / "smoke"
        BASE_OVERRIDES = {"n_train": 4, "n_val": 2, "n_test": 2, "steps": 4}
        SETTINGS.update(seeds=[0], epochs=1, rollout_steps=2, check_steps=2, check_trajectories=1,
                        probe_batch=2, reference_substeps=4, smoke=True)
    SETTINGS.update(json.loads(os.environ.get("SPNO_OPTIONS", "{}")))
    BASE_OVERRIDES.update(json.loads(os.environ.get("SPNO_BASE", "{}")))
    IN_COLAB = importlib.util.find_spec("google") is not None and importlib.util.find_spec("google.colab") is not None
    print(json.dumps({"settings": SETTINGS, "base_overrides": BASE_OVERRIDES}, indent=2))
    ''')
    cell("markdown", '''
    ## 2. Install the embedded code
    ''')
    cell("code", textwrap.dedent(SETUP).replace("__DIGEST__", digest)
         .replace("__SOURCE__", "\n" + "\n".join(textwrap.wrap(encoded, 120))))
    cell("markdown", r'''
    ## 3. Data: generate, check resolution, measure the spectral support

    Deterministic in the config, so it is regenerated (not stored) after a disconnect; the content
    hash is recorded and re-checked. For bw8 the config hash must be `bd4e108527`, and when the Phase 6
    archive is available the regenerated val/test tensors must equal the Windows ones bitwise.

    * **Support edge** — the largest |k| whose mean energy fraction over every training trajectory and
      frame exceeds 1e-6 (and 1e-8). The nonlinear cascade puts it above the initial bandwidth.
    * **Resolution** — tail above 0.75·Nyquist < 1e-6 and a stored-vs-2N (2× substeps) spot check < 1e-3.
    ''')
    cell("code", '''
    from dataclasses import replace
    import torch
    from spno.config import DataConfig, config_hash
    import scripts.run_kinetic_support as ks

    base = replace(DataConfig(), **BASE_OVERRIDES)
    if not BASE_OVERRIDES:
        assert config_hash(base) == ks.PHASE6_BASE_HASH, "Phase 6 training distribution changed"
    study = ks.prepare(OUTPUT_ROOT, DATA_ROOT, base=base, options=SETTINGS)
    data_checks = ks.run_data(study, source_root=SOURCE_ROOT)
    ''')
    cell("markdown", '''
    ## 4. Train (resumable)

    One C1 per cell and seed, seeded exactly as Phase 6 did (`torch.manual_seed(seed)` right before
    construction). Rerun this cell after a disconnect: finished runs are skipped and an interrupted
    run continues from its last epoch (optimizer, scheduler and RNG state included).
    ''')
    cell("code", '''
    ks.train_all(study)
    ''')
    cell("markdown", '''
    ## 5. Measure, summarize, plot

    Per checkpoint: knees, learned rate, output bound 2‖w_out‖₁ (K0/K1), plane-wave G5a/G5b
    (after the reference-solver gates pass), G4 rollouts on shared ICs against a refined reference,
    one-step error on its own test set and on the bw8 test set, and budget diagnostics.
    ''')
    cell("code", '''
    import html
    from IPython.display import display, Image, HTML
    from scripts.plot_kinetic_support import export_plots

    port = ks.measure_all(study, source_root=SOURCE_ROOT)
    summary = ks.summarize(study)
    figures = export_plots(study["root"])

    key = f"{ks.PRIMARY['alpha']}/{ks.PRIMARY['threshold']}"
    header = ("<tr><th>cell</th><th>support edge (1e-6)</th><th>knee per seed</th><th>plateau κ(N/2)−κ(0)</th>"
              "<th>2‖w_out‖₁</th><th>G5b err k=16</th><th>G4 aligned bw16</th><th>best epoch / run</th>"
              "<th>last-5 val gain</th><th>data ok</th></tr>")
    body = ""
    for name, c in summary["cells"].items():
        fmt = lambda v, f="{:.3g}": "—" if v is None else f.format(v)
        budgets = ", ".join(f"{b['best_epoch']}/{b['epochs_run']}" for b in c["budget"])
        gains = ", ".join(fmt(b["last5_improvement"], "{:.1%}") for b in c["budget"])
        body += (f"<tr><td>{name}</td><td>{c['support'].get('1e-06')}</td><td>{c['knees'][key]}</td>"
                 f"<td>{', '.join(fmt(v, '{:.0f}') for v in c['plateau'])}</td>"
                 f"<td>{', '.join(fmt(v, '{:.0f}') for v in c['output_bound'])}</td>"
                 f"<td>{fmt(c['G5b'].get('16'))}</td><td>{fmt(c['G4'].get('16', {}).get('aligned_state_error'))}</td>"
                 f"<td>{budgets}</td><td>{gains}</td><td>{'yes' if c['data_resolved'] else 'FLAG'}</td></tr>")
    display(HTML(f"<p>Knee = first k with map residual &gt; {ks.PRIMARY['threshold']} at α={ks.PRIMARY['alpha']} "
                 "(None = accurate to Nyquist).</p><table>" + header + body + "</table>"))
    verdict = summary["verdict"]
    display(HTML("<h3>Pre-registered verdict: " + html.escape(verdict["data_support"]) + "</h3><pre>"
                 + html.escape(json.dumps(verdict, indent=1)) + "</pre>"))
    print("Port check vs Windows Phase 6 C1:", json.dumps(port, indent=1))
    for figure in figures:
        display(Image(filename=str(figure)))
    print("Saved:", study["root"])
    ''')
    cell("markdown", r'''
    ## Reading the results

    * **Headline = K0-bw8 vs K0-bw12** (both below k_wrap). Figure 02 is the one to show: if the knee
      tracks the data support edge, the Phase 6 flat line is a data-coverage limit, not a statement about
      what C1 can represent.
    * **bw16** is secondary: part of its support lies above k_wrap(α) where one-step data only fixes κ
      mod 2π/dt, so a flat κ there can still give a small map residual. Read its map residual, not κ.
    * **K1 vs K0** at bw8 tests the product feature; **K2** is told the law, so its success says nothing
      about learning dispersion — it bounds what a correct form buys.
    * The port check compares Colab K0-bw8 with the Windows C1 (same data and init, different hardware);
      a CHECK there means the other rows need care before comparing with Phase 6 numbers.
    * Everything is 40-epoch budget-bound, as in Phase 6: **exploratory**.
    ''')
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}, "colab": {"name": "13_kinetic_support_ladder.ipynb", "provenance": []}},
        "nbformat": 4, "nbformat_minor": 5}
    output = ROOT / "notebooks/13_kinetic_support_ladder.ipynb"
    output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    return output


if __name__ == "__main__":
    print(build_notebook())
