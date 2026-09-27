# Kinetic Support Ladder (Notebook 13) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A self-contained Colab notebook that retrains C1 while changing one factor at a time (training-data bandwidth 8/10/12/16 for K0; kinetic rung K1/K2 at bandwidth 8), then decides by a pre-registered rule whether C1's learned dispersion flattening at k ≈ 10 is a data-support limit or an architecture limit.

**Architecture:** Same pattern as notebooks 11/12: a runner module (`scripts/run_kinetic_support.py`) holding data generation, resumable training, measurement and the verdict; a plot module reading only saved records; a builder that embeds the source into `notebooks/13_kinetic_support_ladder.ipynb`. Training reuses the Phase 6 code path (`train_one_step`, `TrainingProgress`, `_metadata`, `_structured_architecture`); measurement reuses the Phase 6 probes (`run_gates`, `alpha_phase_derivative`, `dispersion_curve`) and notebook 11 helpers (`probe_cases`, `sample_probe`, `reference_frames`, `measure_rollout`).

**Tech Stack:** Python 3.14 (`spno/.venv`, run through `uv run`), PyTorch, NumPy, Matplotlib (Agg), pytest, nbformat/nbclient for the notebook test. Colab for all real compute.

**Spec:** `docs/PROMPT_PHASES_6_TO_9.md` § Phase 6, last paragraph ("Also run the K0/K1/K2 kinetic ladder here") plus the design below. The design was approved on 2026-09-27; its evidence base is `results/phase6-diagnosis-2026-09-22/` (G5b, `frozen-probes/kinetic-rate-bounds.json`) and notebook 11 run `3bf81deae4ef1ff9`.

### Design (what the code must do)

| cell | kinetic | train bandwidth | role |
|---|---|---|---|
| K0-bw8 | K0 | 8 | control; same init and data as the Windows Phase 6 C1 |
| K0-bw10 | K0 | 10 | third point below the wrap limit |
| K0-bw12 | K0 | 12 | **primary** data-support contrast with bw8 |
| K0-bw16 | K0 | 16 | straddles k_wrap(α) = 16.9–21.2; secondary |
| K1-bw8 | K1 | 8 | α·k² supplied as a feature |
| K2-bw8 | K2 | 8 | κ = −αk²(1+MLP); form imposed, an ablation |

- **Knee** = smallest k ≥ 1 whose plane-wave one-step map residual |e^{−iω_θ dt} − e^{−iω dt}| exceeds 0.05 at α = 0.9, β = 0.3 (0.01/0.2 and α = 0.7 also reported). Gauge-free and branch-free. Calibrated on the stored Windows G5a curves: knee = 10 for all 6 C1 checkpoints, in-band (k ≤ 8) residual ≤ 4.4e-4.
- **Pre-registered verdict:** *SUPPORTED* if the K0-bw12 knee ≥ K0-bw8 knee + 2 on ≥ ⌈2/3⌉ of seeds (paired) and the K0-bw10 median lies between; *ARCHITECTURE* if every K0 knee is within ±1 of the bw8 median and K0-bw12 improved ≤ 5 % over its last 5 epochs; otherwise *INCONCLUSIVE* (with an explicit "rerun with more epochs" variant).

## Global Constraints

- Training protocol = Phase 6 base: `epochs=40, batch_size=256, learning_rate=1e-3, patience=8`, AdamW/cosine/clip 1 via `train_one_step`; C1 = `DensityPhaseSplitStep(kinetic_mode=…, local_mode="L0", width=32, kinetic_gauge="free")`, seeded with `torch.manual_seed(seed)` immediately before construction.
- `DataConfig()` must keep hash `bd4e108527`; the bandwidth is varied only through `replace(DataConfig(), initial_bandwidth=b)`.
- Every measurement is float64 on CPU via `precision.widen_to_double` (it deep-copies); never `.double()`.
- `pyproject.toml` sets `filterwarnings = ["error::UserWarning"]`: any `UserWarning` in a test fails it.
- Compute and results live in Colab/Drive; on the Mac only tiny configs (grid 8) are run, never full-size shards.
- Regenerated data may differ from a previous generation at rounding level (~1e-12, cross-runtime FFT); that must never block resume. Cross-machine comparisons use a 1e-10 relative tolerance, not bitwise equality.
- Budget-bound (40-epoch) results are labelled exploratory everywhere.
- Run tests with `uv run pytest …` from `spno/`.

## Review Focus

1. **Knee instrument vs the real one-step map** — `plane_wave_rates` builds ω_θ analytically from κ and ν; if that ever disagrees with what the model actually does to a plane wave, every knee is wrong. Expect agreement with `one_step_multiplier` to ~1e-12 for any C1 rung. (Task 4, test a)
2. **Calibration drift of the decision threshold** — a future edit to `knee`/`map_residual` could silently move the bw8 knee off 10 and flip the verdict. Expect the stored Windows G5a curves to give knee 10 for every base C1 seed. (Task 4, test b)
3. **Colab reconnect on a different CPU** — regenerated shards are not bitwise identical; expect a printed note and training to continue, never a `RuntimeError`. (Task 4, test c)
4. **Phase 6 data reproduction across machines** — identical tensors → PASS, a 1e-6 perturbation → CHECK, missing archive → skipped; never FAIL on rounding. (Task 4, test d)
5. **Port check against Windows C1** — never exercised at tiny size because the tiny config never hashes to `bd4e108527`. Expect PASS for an identical checkpoint and CHECK for a different one. (Task 4, test e)

---

## File Structure

| file | responsibility | status |
|---|---|---|
| `scripts/run_kinetic_support.py` | settings, run identity, data + checks, resumable training, measurements, summary, verdict | exists |
| `scripts/plot_kinetic_support.py` | 5 figures (PNG+PDF) from saved records only | exists |
| `scripts/build_support_notebook.py` | builds notebook 13 with embedded source; optional Phase 6 artifacts | exists |
| `notebooks/13_kinetic_support_ladder.ipynb` | generated; never edited by hand | exists |
| `notebooks/README.md` | one row for notebook 13 | modified |
| `tests/test_kinetic_support.py` | unit, end-to-end, verdict, notebook-execution tests | exists; Task 4 extends |

---

### Task 1: Runner module — DONE (verify only)

**Files:**
- Create: `scripts/run_kinetic_support.py` (already written)
- Test: `tests/test_kinetic_support.py`

**Interfaces:**
- Produces:
  - `prepare(output_root, data_root, *, base: DataConfig | None = None, options: dict | None = None) -> dict` — keys `root, run_id, options, base, data_root, cells, device`; each cell is `{"kinetic", "bandwidth", "name", "config"}`.
  - `run_data(study, *, source_root=None) -> dict` (writes `data-checks.json`, keyed by `config_hash`).
  - `ensure_shards(cfg, data_root, expected=None) -> (dict[str, TrajectoryShard], dict[str, str])`.
  - `reproduce_phase6(shards, source_root) -> dict` with `status ∈ {"PASS","CHECK","skipped"}`.
  - `train_all(study) -> None`; `load_trained(study, cell, seed) -> (model, history_dict)`; `build_model(cell, seed)`.
  - `plane_wave_rates(model, cfg, alpha, beta) -> (k, omega_model, omega_true, centered_kappa)` (float64 tensors over k = 0..N/2).
  - `map_residual(model, cfg, alpha, beta) -> list[float]`; `knee(residual, threshold) -> int | None`; `as_knee(value, grid_size) -> int`.
  - `measure_all(study, *, source_root=None) -> dict` (port-check result); `port_check(study, source_root) -> dict`.
  - `summarize(study) -> dict` (writes `summary.json`, `knees.csv`); `verdict(study, cells) -> dict`.
  - Constants: `DEFAULTS`, `DEFAULT_CELLS`, `PRIMARY = {"threshold": 0.05, "alpha": 0.9}`, `PHASE6_BASE_HASH = "bd4e108527"`.

- [x] **Step 1: Implemented** — see the file; the tolerance-based data checks (Global Constraints) are already in `ensure_shards` and `reproduce_phase6`.
- [ ] **Step 2: Verify**

Run: `uv run pytest tests/test_kinetic_support.py -q`
Expected: `9 passed`

### Task 2: Plot module — DONE (verify only)

**Files:**
- Create: `scripts/plot_kinetic_support.py` (already written)

**Interfaces:**
- Consumes: run directory written by Task 1 (`manifest.json`, `summary.json`, `data-checks.json`, `cells/<cell>/seed<s>.json`); `PRIMARY`, `as_knee`.
- Produces: `export_plots(root) -> list[Path]` — `01_kinetic_rate`, `02_knee_vs_bandwidth`, `03_residual_and_g5b`, `04_g4_rollout`, `05_training_and_support` (PNG + PDF).

- [x] **Step 1: Implemented.**
- [ ] **Step 2: Verify** — covered by `test_full_run_resumes_without_retraining_and_writes_every_output` (asserts 5 figures exist).

### Task 3: Notebook builder, notebook, README — DONE (verify only)

**Files:**
- Create: `scripts/build_support_notebook.py`, `notebooks/13_kinetic_support_ladder.ipynb` (generated)
- Modify: `notebooks/README.md` (row for notebook 13)

**Interfaces:**
- Consumes: `embedded_source()` and `ROOT` from `scripts/build_hybrid_notebook.py` (unchanged, so notebooks 11/12 are untouched).
- Produces: notebook honouring env overrides `SPNO_PROJECT_ROOT, SPNO_OUTPUT_ROOT, SPNO_DATA_ROOT, SPNO_SOURCE_ROOT, SPNO_OPTIONS, SPNO_BASE`.

- [x] **Step 1: Implemented.**
- [ ] **Step 2: Verify the generated notebook is current**

Run: `uv run python -m scripts.build_support_notebook && git diff --stat notebooks/13_kinetic_support_ladder.ipynb`
Expected: the path is printed; no diff after the first commit (the embedded source hash only changes when `src/` or `scripts/` change — rebuild after any such edit).

### Task 4: Pin the Review Focus failure modes

**Files:**
- Modify: `tests/test_kinetic_support.py` (append the five tests below)
- Modify (only if a test fails): `scripts/run_kinetic_support.py`

**Interfaces:**
- Consumes: everything listed under Task 1; `one_step_multiplier` from `spno.evaluation.dispersion`; `probe_amplitude`; `shard_paths`, `TrajectoryShard`; `save_checkpoint`, `load_checkpoint_payload`; `config_hash`.

These pin behavior that already exists, so the "failing test" step is replaced by "run and confirm". If any test fails, the failure is a real defect in the listed function: fix the function, not the test.

- [ ] **Step 1: Append the tests**

```python
# --- Review Focus pins (plan 2026-09-27) --------------------------------------
import glob
import cmath
import shutil

from spno.checkpoints import save_checkpoint
from spno.config import config_hash
from spno.data.datasets import TrajectoryShard, shard_paths
from spno.evaluation.dispersion import one_step_multiplier, probe_amplitude
from spno.precision import widen_to_double


@pytest.mark.parametrize("kinetic", ["K0", "K1", "K2"])
def test_map_residual_matches_the_models_actual_one_step_multiplier(kinetic):
    cfg = replace(DataConfig(), grid_size=32)
    cell = {"kinetic": kinetic, "bandwidth": 8, "name": f"{kinetic}-bw8", "config": cfg}
    model = ks.build_model(cell, seed=4)
    with torch.no_grad():  # move every rung off its zero-initialized head
        for parameter in model.parameters():
            parameter.add_(.3 * torch.randn_like(parameter))
    k, omega, _, _ = ks.plane_wave_rates(model, cfg, .9, .3)
    probe = widen_to_double(model, device="cpu").eval()
    amplitude = probe_amplitude(cfg.domain, cfg.mass_range)
    for wave_number in (0, 3, 9, 16):
        m = one_step_multiplier(probe, cfg.domain, wave_number, alpha=.9, beta=.3,
                                amplitude=amplitude, potential_constant=0.0, dt=cfg.dt)
        predicted = cmath.exp(-1j * cfg.dt * float(omega[wave_number]))
        assert abs(m - predicted) < 1e-12


def test_knee_calibration_on_the_stored_windows_c1_curves():
    root = Path(__file__).resolve().parents[1] / "results/phase6-diagnosis-2026-09-22/frozen-probes"
    files = sorted(glob.glob(str(root / "base-C1-seed*.json")))
    assert len(files) == 3
    for path in files:
        record = json.loads(Path(path).read_text())
        curve = next(iter(record["experiments"]["G5a"]["curves"].values()))
        residual = [abs(cmath.exp(-1j * curve["dt"] * a) - cmath.exp(-1j * curve["dt"] * b))
                    for a, b in zip(curve["principal"], curve["truth"])]
        assert max(residual[1:9]) < 1e-3
        assert ks.knee(residual, ks.PRIMARY["threshold"]) == 10


def test_regenerated_data_that_differs_at_rounding_level_never_blocks_resume(tmp_path, capsys):
    options = {**OPTIONS, "cells": [{"kinetic": "K0", "bandwidth": 2}], "epochs": 1}
    study = ks.prepare(tmp_path / "out", tmp_path / "data", base=TINY, options=options)
    ks.run_data(study)
    checks_path = study["root"] / "data-checks.json"
    checks = json.loads(checks_path.read_text())
    for entry in checks.values():
        entry["digests"] = {split: "0" * 64 for split in entry["digests"]}
    checks_path.write_text(json.dumps(checks))
    shutil.rmtree(tmp_path / "data")
    ks.run_data(study)
    ks.train_all(study)
    assert "not bitwise identical" in capsys.readouterr().out
    assert (study["root"] / "checkpoints/K0-bw2/C1-seed0.pt").exists()


def test_phase6_data_reproduction_uses_a_tolerance(tmp_path, monkeypatch):
    shards = {split: TrajectoryShard.load(path)
              for split, path in _tiny_shards(tmp_path).items() if split != "train"}
    monkeypatch.setattr(ks, "PHASE6_BASE_HASH", "tinyhash00")
    assert ks.reproduce_phase6(shards, None)["status"] == "skipped"
    source = tmp_path / "phase6"
    paths = shard_paths(source / "data", "tinyhash00")
    for split in ("val", "test"):
        shards[split].save(paths[split])
    assert ks.reproduce_phase6(shards, source)["status"] == "PASS"
    shifted = TrajectoryShard.load(paths["val"])
    shifted.trajectories = shifted.trajectories * (1 + 1e-6)
    shifted.save(paths["val"])
    result = ks.reproduce_phase6(shards, source)
    assert result["status"] == "CHECK"
    assert result["max_relative_difference"]["val"] == pytest.approx(1e-6, rel=1e-3)


def _tiny_shards(tmp_path):
    ks.ensure_shards(TINY, tmp_path / "tiny-data")
    return shard_paths(tmp_path / "tiny-data", config_hash(TINY))


def test_port_check_passes_identical_weights_and_flags_different_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(ks, "PHASE6_BASE_HASH", config_hash(TINY))
    options = {**OPTIONS, "cells": [{"kinetic": "K0", "bandwidth": 2}]}
    study = ks.prepare(tmp_path / "out", tmp_path / "data", base=TINY, options=options)
    ks.run_data(study)
    ks.train_all(study)
    ks.measure_all(study)
    ours = study["root"] / "checkpoints/K0-bw2/C1-seed0.pt"
    windows = tmp_path / "phase6/checkpoints/phase6" / f"{config_hash(TINY)}-K0" / "C1-seed0.pt"
    windows.parent.mkdir(parents=True)
    shutil.copy(ours, windows)
    assert ks.port_check(study, tmp_path / "phase6")["status"] == "PASS"
    payload = load_checkpoint_payload(windows)
    model, _ = ks.load_trained(study, study["cells"][0], 0)
    torch.manual_seed(9)
    with torch.no_grad():
        model.kinetic.network.network[-1].weight.copy_(20 * torch.randn_like(model.kinetic.network.network[-1].weight))
    save_checkpoint(windows, model, payload.metadata)
    assert ks.port_check(study, tmp_path / "phase6")["status"] == "CHECK"
```

- [ ] **Step 2: Run and confirm**

Run: `uv run pytest tests/test_kinetic_support.py -q`
Expected: `16 passed` (9 existing + 3 parametrized + 4). If `test_map_residual_matches…` fails, the analytic ω in `plane_wave_rates` has the wrong sign or density convention — compare with `kinetic_dispersion` in `src/spno/evaluation/component_ablation.py:247`, whose `"C1"` entry is `omega - local` with `omega = -kappa`. If `test_port_check…` fails on the CHECK assertion because both knees and plateaus happen to agree, increase the perturbation factor rather than loosening the rule.

- [ ] **Step 3: Rebuild the notebook (embedded source changes only if runner code changed) and rerun the full suite**

Run: `uv run python -m scripts.build_support_notebook && uv run pytest tests -q`
Expected: all new tests pass. Two failures pre-exist on a clean tree and are out of scope: `test_notebooks.py::…[10_dirichlet.ipynb]` (no `scipy`) and `test_artifact_discovery.py::test_known_windows_archive_requires_its_recorded_checksum`.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_kinetic_support.py scripts/plot_kinetic_support.py scripts/build_support_notebook.py \
        notebooks/13_kinetic_support_ladder.ipynb notebooks/README.md tests/test_kinetic_support.py \
        docs/superpowers/plans/2026-09-27-kinetic-support-ladder.md
git commit -m "feat: notebook 13, kinetic support ladder (training bandwidth x K0/K1/K2)"
```

### Task 5: Run in Colab and record the outcome

**Files:**
- None in the repo (results stay in Drive: `MyDrive/spno/kinetic-support/<run-id>/`).
- Memory: `/Users/omermazal/.claude/projects/-Users-omermazal-dev-pin/memory/phase6-colab-handoff.md` (append a notebook 13 paragraph).

**Interfaces:**
- Consumes: the committed notebook from Task 4.

- [ ] **Step 1: Smoke run** — upload `notebooks/13_kinetic_support_ladder.ipynb` to Colab, GPU runtime, set `SMOKE = True`, Run all.
Expected: every cell finishes; a table with 6 rows; verdict `not evaluable` or `INCONCLUSIVE` (1 seed, 1 epoch — meaningless by design); files under `MyDrive/spno/kinetic-support/smoke/<run-id>/`.

- [ ] **Step 2: Full run** — set `SMOKE = False`, Run all. After any disconnect, Run all again (finished runs print `already trained`).
Expected output lines before training: `K0-bw8 data bd4e108527: … | Phase 6 data reproduction: PASS` (or `skipped` if the Phase 6 zip is not in Drive), tail/2N `ok` for bw8/10/12; bw16 may show `FLAG` — keep it, it is reported, not dropped.

- [ ] **Step 3: Read the results in this order**
  1. `Port check vs Windows Phase 6 C1` — must be `PASS` (or `skipped`); on `CHECK`, stop and compare `knee_colab/knee_windows` before trusting other rows.
  2. K0-bw8 knees should be `[10, 10, 10]` (calibration).
  3. The verdict header and `02_knee_vs_bandwidth.png`.
  4. `last-5 val gain` column for K0-bw12: > 5 % means the verdict cannot be ARCHITECTURE.

- [ ] **Step 4: Record** — append to the memory file: run id, verdict string, K0 knees per bandwidth, K1/K2 knees, port-check status, and whether bw16 data was flagged. Update the `MEMORY.md` hook line for `phase6-colab-handoff` if the summary changes.
