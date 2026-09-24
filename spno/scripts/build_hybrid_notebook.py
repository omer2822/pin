"""Build a self-contained Colab notebook with a frozen, embedded source archive."""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build_notebook():
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    paths += [ROOT / "pyproject.toml", ROOT / "README.md"]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            entry = zipfile.ZipInfo(str(path.relative_to(ROOT)), date_time=(2026, 9, 23, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes())
    encoded = base64.b64encode(buffer.getvalue()).decode()
    digest = hashlib.sha256(buffer.getvalue()).hexdigest()
    cells = []

    def cell(kind, source):
        item = {"cell_type": kind, "metadata": {}, "id": f"hybrid-{len(cells):02d}",
                "source": textwrap.dedent(source).strip().splitlines(keepends=True)}
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell("markdown", r'''
    # Why learn the kinetic operator? — C1 hybrid study, G1–G9

    **This notebook is self-contained. Upload this `.ipynb` to Colab and run all cells.**
    It embeds the experiment source; no GitHub push, source ZIP, or separate script upload is needed.
    Keep your Phase 6 checkpoint archive in Google Drive, as for the preceding evaluation notebook.
    Nothing is trained. Existing checkpoints are read without modification.

    The primary experiment evaluates **exact kinetic + learned local across the full G1–G9 suite**.
    Every trajectory experiment compares the four literal, frozen component combinations:

    | Model | Kinetic flow | Local flow |
    |---|---|---|
    | Full C1 | Learned from saved C1 | Learned from that same C1 |
    | **Hybrid** | **Exact** | **Learned from saved C1** |
    | Reverse swap | Learned from saved C1 | Exact |
    | Exact split step | Exact | Exact |

    The exact split step is a **coarse-step comparator**, not ground truth. Targets use a finer,
    independently refined split-step integration. A frozen hybrid is a deployable operator, but this
    experiment does not establish how well a hybrid trained from scratch would perform.
    ''')
    cell("markdown", r'''
    ## 1. Protocol and editable settings

    Defaults: **3 training seeds × 5 probe seeds × 16 ICs**, 200 short steps and **2,000 long steps**
    at the original Δt (T=2 and T=20 for Δt=0.01). Training seeds share exactly the same probe ICs;
    they are not counted as additional independent trajectories. Probe seeds are fresh draws.
    G6 compares equal physical horizons across step sizes. G9 uses the original α=0.9, β=0.3, V=0.

    | Arm | What is tested |
    |---|---|
    | G1 | Fresh interpolation ICs and parameters |
    | G2 | Wider α/β range, matching the existing extrapolation arm |
    | G3 | Zero, cosine, well, short-correlation, and stronger potentials |
    | G4 | Input support k_max = 4, 6, 8, 10, 12, 16, 20, 24, 28, 32 |
    | G5a | Direct kinetic generator and effective plane-wave map, full signed spectrum |
    | G5b | In-range α sensitivity of the kinetic generator |
    | G6a | Multi-Δt C1 checkpoints and their component swaps, with base C1 controls |
    | G6b | Base-checkpoint transfer to Δt/2, Δt, 2Δt |
    | G7 | Fixed-α-trained vs varying-α-trained C1, on identical fixed-α ICs |
    | G8 | Long-rollout and conservation tests (new extension; previously unassigned) |
    | G9 | Nonlinear cascade, full errors and spectra, bandwidth sweep, long rollout |

    All trajectory arms save state, phase, full spectrum, mass, and true-Hamiltonian diagnostics.
    A full run is substantial; use the labeled smoke mode to check setup first. Results are saved
    per case/probe/training seed to Drive. Rerun after a disconnect; completed units are reused.
    Changes to settings, source, checkpoint bytes, or software environment create a new run identity.
    ''')
    cell("code", '''
    from pathlib import Path
    import os, sys, json, importlib.util

    # Optional: use an already extracted source directory or a specific checkpoint ZIP.
    SOURCE_ROOT = os.environ.get("SPNO_SOURCE_ROOT", "")
    CHECKPOINT_ARCHIVE = os.environ.get("SPNO_CHECKPOINT_ARCHIVE", "")
    OUTPUT_ROOT = Path(os.environ.get("SPNO_OUTPUT_ROOT", "/content/drive/MyDrive/spno/hybrid-ablation"))
    SOURCE_CONFIG = os.environ.get("SPNO_SOURCE_CONFIG", "")  # optional JSON with original `data`
    SMOKE = False  # True is only a plumbing check, never a research result.
    SETTINGS = {
        "training_seeds": [0, 1, 2],
        "probe_seeds": [1000, 1001, 1002, 1003, 1004],
        "batch": 16,
        "bandwidths": [4, 6, 8, 10, 12, 16, 20, 24, 28, 32],
        "short_steps": 200, "long_steps": 2000, "stride": 50,
        "reference_substeps": 32,   # compares 32 vs 64; uses 64 as target
        "reference_tolerance": 1e-4,
        "spatial_samples": 2,       # 2 ICs per probe also checked on a 2N grid
        "spatial_tolerance": 1e-3,
        "tail_threshold": 1e-6,
        "device": "cpu",           # float64 CPU default; "cuda" also supported
        "threads": 2,
        "allow_budget_bound": True, # retains the preceding frozen-checkpoint policy
        "bootstrap_draws": 2000,
    }
    if SMOKE:
        SETTINGS.update(training_seeds=[0], probe_seeds=[1000, 1001], batch=2,
                        bandwidths=[4, 8, 12], short_steps=2, long_steps=4, stride=1,
                        reference_substeps=2, spatial_samples=1, bootstrap_draws=50,
                        smoke=True)
    SETTINGS.update(json.loads(os.environ.get("SPNO_OPTIONS", "{}")))
    IN_COLAB = importlib.util.find_spec("google") is not None and importlib.util.find_spec("google.colab") is not None
    print(json.dumps(SETTINGS, indent=2))
    ''')
    cell("markdown", '''
    ## 2. Install the embedded experiment code and locate saved checkpoints

    Colab asks to mount Drive. The notebook searches for the existing Phase 6 eval-only or full
    artifact ZIP. Alternatively, set `SOURCE_ROOT` above to an extracted artifact directory, or
    `CHECKPOINT_ARCHIVE` to a ZIP containing the checkpoints. No training datasets are required.
    Checkpoint identity and convergence status are verified below; missing G6a/G7 weights stop the
    run instead of silently omitting those arms.
    ''')
    setup = '''
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
        base = Path("/content") if IN_COLAB else Path.cwd()
        PROJECT_ROOT = base / ("spno-hybrid-code-" + EMBEDDED_SOURCE_SHA256[:12])
        PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            assert archive.testzip() is None
            safe_extract(archive, PROJECT_ROOT)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(PROJECT_ROOT)])
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    if not SOURCE_ROOT:
        existing = Path("/content/pin/spno/results/phase6-standalone-artifacts")
        if existing.is_dir():
            SOURCE_ROOT = existing
        else:
            if not CHECKPOINT_ARCHIVE:
                drive_root = Path("/content/drive/MyDrive")
                names = ["phase6-eval-only-bd4e108527-K0.zip", "phase6-standalone-artifacts-bd4e108527-K0.zip"]
                hits = [p for name in names for p in drive_root.rglob(name)]
                if not hits:
                    raise FileNotFoundError("Place the Phase 6 artifact ZIP in MyDrive, or set SOURCE_ROOT / CHECKPOINT_ARCHIVE in cell 1.")
                CHECKPOINT_ARCHIVE = str(sorted(hits)[0])
            archive_path = Path(CHECKPOINT_ARCHIVE)
            archive_digest = hashlib.sha256()
            with archive_path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    archive_digest.update(block)
            known = {
                "phase6-eval-only-bd4e108527-K0.zip": "cc882810f6fec63f36d6923833c05c6dcde5b6d50b2a1df296634be755b1235d",
                "phase6-standalone-artifacts-bd4e108527-K0.zip": "01fd894349dc36dd90386d877693e51bbe3d2d28e98609ccefdf02608a0a0812",
            }
            if archive_path.name in known:
                assert archive_digest.hexdigest() == known[archive_path.name], "Checkpoint archive checksum mismatch"
            extracted = PROJECT_ROOT.parent / ("spno-hybrid-artifacts-" + archive_digest.hexdigest()[:12])
            with zipfile.ZipFile(archive_path) as archive:
                assert archive.testzip() is None, "Corrupt checkpoint archive"
                safe_extract(archive, extracted)
            candidates = [p.parent for p in extracted.rglob("checkpoints") if (p / "phase6").is_dir()]
            assert len(candidates) == 1, f"Expected one artifact root, found {candidates}"
            SOURCE_ROOT = candidates[0]
    SOURCE_ROOT = Path(SOURCE_ROOT)
    assert (SOURCE_ROOT / "checkpoints" / "phase6").is_dir(), SOURCE_ROOT
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("Embedded source:", EMBEDDED_SOURCE_SHA256)
    print("Checkpoint root:", SOURCE_ROOT)
    print("Results:", OUTPUT_ROOT)
    '''
    setup = textwrap.dedent(setup).replace("__DIGEST__", digest).replace("__SOURCE__", "\n" + "\n".join(textwrap.wrap(encoded, 120)))
    cell("code", setup)
    cell("markdown", r'''
    ## 3. Checkpoint inventory and operator sanity checks

    The local module is copied from **the same C1 seed and training cohort** as the kinetic module.
    No weights or phase offsets are fitted using evaluation data. Original convergence metadata is
    retained. Budget-bound checkpoints produce **exploratory** results, as in the preceding notebook.
    ''')
    cell("code", '''
    import torch, numpy as np
    from spno.config import DataConfig
    from spno.artifacts import atomic_json
    from spno.evaluation.component_ablation import (
        ComponentSplitStep, component_models, probe_cases, sample_probe, kinetic_dispersion)
    from spno.solvers.split_step import SplitStepNLSOperator
    from scripts.run_hybrid_ablation import load_c1_cohorts, run_study, DEFAULTS

    torch.set_num_threads(SETTINGS["threads"])
    declared = DataConfig(**json.loads(Path(SOURCE_CONFIG).read_text())["data"]) if SOURCE_CONFIG else None
    data_config, cohorts, checkpoint_inventory = load_c1_cohorts(
        SOURCE_ROOT, declared, SETTINGS["training_seeds"],
        allow_budget_bound=SETTINGS["allow_budget_bound"])
    print("Training data:", data_config)
    for row in checkpoint_inventory:
        print(row["name"], "seed", row["seed"], "converged", row["metadata"]["converged"], "sha256", row["sha256"][:16])
    test_case = probe_cases(data_config, [data_config.initial_bandwidth])[0]
    inputs, _ = sample_probe(test_case, SETTINGS["probe_seeds"][0], 2)
    x, potential, alpha, beta = inputs
    for cohort, by_seed in cohorts.items():
        for seed, model in by_seed.items():
            parts = component_models(model)
            exact = SplitStepNLSOperator(data_config.domain)(*inputs, data_config.dt)
            assert torch.allclose(parts["exact_split"](*inputs, data_config.dt), exact, atol=1e-12, rtol=1e-12)
            copied = ComponentSplitStep(model, exact_kinetic=False, exact_local=False)
            assert torch.allclose(copied(*inputs, data_config.dt), parts["C1"](*inputs, data_config.dt), atol=1e-12, rtol=1e-12)
            for name, operator in parts.items():
                y = operator(*inputs, data_config.dt)
                back = operator(y, potential, alpha, beta, -data_config.dt)
                assert torch.allclose(back, x, atol=1e-11, rtol=1e-11), (cohort, seed, name)
    print("PASS: exact control, copied C1, and reversibility for all four component combinations.")
    print("Active learned parameter counts (frozen for evaluation):")
    example = component_models(next(iter(cohorts["base"].values())))
    print({name: sum(p.numel() for p in model.parameters()) for name, model in example.items()})
    ''')
    cell("markdown", r'''
    ## 4. Dispersion preview — before the long experiment

    Read **ωθ(k) = −κθ(k², α, β)** directly from C1 at every resolvable signed Fourier mode.
    This is a generator diagnostic, not a frequency inferred from a wrapped one-step phase.
    The generator can contain a constant offset that cancels against the local rate in the full model.
    Both raw and k=0-centered curves are shown; **the actual swaps remain uncorrected**.
    A line at ± the training bandwidth separates supervised spectral support from extrapolation.
    ''')
    cell("code", '''
    import matplotlib.pyplot as plt
    from IPython.display import display, Image, Markdown
    preview = {str(seed): kinetic_dispersion(model, data_config) for seed, model in cohorts["base"].items()}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    for seed, entry in preview.items():
        curve = entry["curves"][len(entry["curves"])//2]
        mid = len(entry["alphas"])//2
        for ax, key in zip(axes, ("omega", "omega_centered")):
            ax.plot(entry["k"], curve[key][mid], label="C1 seed " + seed)
    for ax in axes:
        ax.plot(entry["k"], curve["exact"][mid], "k--", label="Exact αk²")
        ax.axvline(data_config.initial_bandwidth, color="#777", ls=":", label="Training bandwidth")
        ax.axvline(-data_config.initial_bandwidth, color="#777", ls=":")
        ax.set(xlabel="Signed Fourier mode k", ylabel="Kinetic frequency ω(k)")
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    axes[0].set_title("Raw kinetic generator")
    axes[1].set_title("ωθ(k) − ωθ(0): constant-offset diagnostic")
    plt.show()
    ''')
    cell("markdown", r'''
    ## 5. Run the paired G1–G9 experiment and save each unit to Drive

    **Metric definitions.** State error is ‖ψ_model−ψ_ref‖₂/‖ψ_ref‖₂. Aligned state error removes
    one best global phase for diagnosis. Phase RMS is the reference-density-weighted principal
    circular phase error in radians; nodes with undefined phase are excluded and their coverage is
    recorded. Global phase error is reported separately. Spectrum error is the relative L1 difference
    in full Fourier power; full power and complex-error spectra are also saved for each IC.

    Mass drift is |M(t)/M(0)−1|. Energy drift is |H_model(t)−H(0)|, normalized by the sum of absolute
    initial kinetic/potential/nonlinear energy terms to avoid division by nearly zero H(0).
    Energy error against H_ref(t), absolute drift, raw mass and raw energy are also retained.
    Nonfinite rollouts keep a failure step and missing metrics; failures are never silently averaged away.

    **Reference checks.** 32 vs 64 substeps at all stored times, plus N vs 2N on a fixed subset of ICs.
    Report unresolved cases explicitly; do not exclude them to make the hybrid look better.
    Broad-support/Nyquist tests primarily describe the same-grid discrete dynamics unless spatial
    convergence is established. A small high-k tail alone is not a convergence proof.
    ''')
    cell("code", '''
    RUN_ROOT = run_study(SOURCE_ROOT, OUTPUT_ROOT, data=data_config, options=SETTINGS)
    print("Saved run:", RUN_ROOT)
    ''')
    cell("markdown", r'''
    ## 6. Main result: does the hybrid preserve generalization and repair G4/G9?

    Generate PNG and vector PDF figures, a machine-readable summary, per-IC compressed JSON,
    and CSV tables. The main paired plot reports hybrid/C1 final state error for every arm and cohort.
    **Values below 1 favor the hybrid.** Bootstrap intervals resample training seeds and probe-seed
    clusters independently, with ICs resampled within probe clusters. IC selections stay paired across
    training seeds and models. With only three training seeds these are descriptive intervals;
    they are not simultaneous confidence bounds over all arms.

    G5 is a deterministic generator probe: it has a training-seed axis, not a fictitious probe-seed axis.
    G7 spectra should be inspected by band; fixing α changes the learning task, so an absolute-error
    improvement alone does not identify the α-conditioning mechanism.
    ''')
    cell("code", '''
    from scripts.plot_hybrid_ablation import export_plots
    figures = export_plots(RUN_ROOT)
    summary = json.loads((RUN_ROOT / "summary.json").read_text())
    manifest = json.loads((RUN_ROOT / "manifest.json").read_text())
    print("Status:", "SMOKE" if manifest["options"]["smoke"] else "EXPLORATORY" if manifest["exploratory"] else "FROZEN CHECKPOINT STUDY")
    print("Completed:", manifest["complete"], "| figures:", len(figures))
    main = RUN_ROOT / "figures/01_all_arms_hybrid_vs_C1.png"
    if main.exists(): display(Image(filename=str(main)))
    for name in ("02_G4_bandwidth_sweep", "02_G9_bandwidth_sweep", "03_dispersion_base",
                 "rollout_G8-long-rollout-extension_base", "rollout_G9-cascade-long_base",
                 "spectrum_G9-cascade-long_base"):
        path = RUN_ROOT / "figures" / (name + ".png")
        if path.exists(): display(Image(filename=str(path)))
    ''')
    cell("markdown", '''
    ## 7. Reference audit and interpretation

    Before making the design claim, inspect G3 potential generalization, G4/G9 spectral extrapolation,
    and G8 long-horizon state/phase/energy together. A mass-conserving but inaccurate rollout is a failure.
    A hybrid that only improves after phase alignment requires an offset explanation, not a claim of
    accurate raw dynamics. An exact split-step competitor that is already as good as the hybrid means
    the experiment has not established a benefit from learning the local term for this known equation.

    This notebook tests **whether replacing the frozen learned kinetic component helps**. It does not
    establish superiority for a freshly trained hybrid, an uncertain local law, or a continuum PDE on an
    unresolved grid. Report those as separate follow-up experiments if the present evidence supports them.
    ''')
    cell("code", '''
    bad = [c for c in summary["reference_checks"] if not c["time_refinement_pass"] or not c["space_refinement_pass"]]
    print(f"Temporal/spatial reference flags: {len(bad)} / {len(summary['reference_checks'])} probe batches")
    for check in bad:
        print(check["case"], "probe", check["probe_seed"],
              "Δt error", check["time_refinement_max"], "2N error", check["space_refinement_max"])
    print("High-Nyquist-tail flags:", sum(not c["tail_pass"] for c in summary["reference_checks"]))
    print("\\nPaired final-state hybrid / C1 comparisons:")
    for row in summary["paired"]:
        if row["metric"] == "state_error" and row["endpoint"] == "final":
            interval = row["hybrid_over_C1"]
            print(row["case"], row["cohort"], interval, row["status"])
    print("\\nArtifacts:", RUN_ROOT)
    print("metrics.csv | paired_comparisons.csv | reference_checks.csv | summary.json | figures/ | cases/")
    print("Settings, source hash, checkpoint hashes and convergence status: manifest.json")
    ''')
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}, "colab": {"name": "11_hybrid_kinetic_ablation.ipynb", "provenance": []}},
        "nbformat": 4, "nbformat_minor": 5}
    output = ROOT / "notebooks/11_hybrid_kinetic_ablation.ipynb"
    output.write_text(json.dumps(notebook, indent=1) + "\n")
    return output


if __name__ == "__main__":
    print(build_notebook())
