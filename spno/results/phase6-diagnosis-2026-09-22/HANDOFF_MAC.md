# Mac handoff: preserve the trained Phase 6 run

The Git branch carries the analysis, plots, original training metrics and script
snapshots. The weights and data are a separate transfer: 36 checkpoints and 26
datasets, approximately 1.33 GB, already excluded by the repository's `.gitignore`.
Git alone is not a backup of these files.

## Before leaving Windows

Copy both files below to an external drive or your chosen file-transfer storage:

- `phase6-standalone-artifacts-bd4e108527-K0.zip`
- `phase6-standalone-artifacts-bd4e108527-K0.zip.sha256`

They were prepared in
`C:/Users/omerm/Documents/Codex/2026-09-18/he/outputs/phase6-mac-transfer/`.
All 62 ZIP members were checked against the original SHA-256 manifest. The archive's
own checksum is recorded in [transfer-archive.json](transfer-archive.json).
Keep the Windows originals until the Mac verification passes. No transfer to a
cloud account or another computer has been performed automatically.

## Restore on the Mac

Use a fresh clone to avoid disturbing any uncommitted work in your existing Mac
checkout. Run this from the parent directory where you want the new clone:

```bash
git clone --branch analysis/phase6-k0-results-2026-09-22 --single-branch https://github.com/omer2822/pin.git pin-phase6-handoff
cd pin-phase6-handoff
```

Place the ZIP and `.sha256` file together in `~/Downloads` (adjust that location if
you used an external drive). Verify the copy before extraction:

```bash
(cd ~/Downloads && shasum -a 256 -c phase6-standalone-artifacts-bd4e108527-K0.zip.sha256)
```

Proceed only if that reports `OK`. In the fresh repository root, confirm the target
does not already exist, then extract. The ZIP already contains `spno/results/...`:

```bash
test ! -e spno/results/phase6-standalone-artifacts && python3 -m zipfile -e ~/Downloads/phase6-standalone-artifacts-bd4e108527-K0.zip .
python3 spno/results/phase6-diagnosis-2026-09-22/verify_artifacts.py --repo-root .
```

The verifier must report **36 checkpoints and 26 data files** with matching hashes.
If the target already exists, inspect it or use another fresh clone; do not overwrite
an existing artifact set blindly. Verification uses standard Python only; the
project itself requires Python 3.11 or newer.

If you will run Python on the Mac, create a Mac virtual environment rather than
copying the Windows `.venv`. Skip this installation if Colab will do the computation:

```bash
python3 -m venv spno/.venv
spno/.venv/bin/python -m pip install -e './spno[dev]'
```

## Using the existing Colab environment

The Mac can manage code and read results while Colab does the computation. The
shared notebook is [the existing Colab notebook](https://colab.research.google.com/drive/1H3LXFalF3y02RrcFxE-Q3vz1C37iCFTi).
Its cells could not be inspected because the available browser required sign-in;
do not assume its training/run-all cells already implement safe resume.

Keep the 36 checkpoints, 26 data files and new evaluation outputs in persistent
Google Drive storage. A notebook being saved in Drive does not by itself preserve
files held only in the temporary `/content` runtime. If the artifacts are already
in Drive, verify the 62 file hashes before uploading another copy. Otherwise copy
the verified Windows archive to your chosen Drive folder and keep the Windows
originals until the extracted files pass verification.

In a fresh Colab session, obtain this exact Git branch, install the project
dependencies into the Colab runtime, and restore or link the verified artifact
directory at `spno/results/phase6-standalone-artifacts` in that checkout. With that
layout, run `verify_artifacts.py --repo-root <checkout-root>` before evaluation.
Google Drive paths must be chosen from the actual existing folder layout; no Drive
folder was created or uploaded to by this task.

Run only setup/loading/evaluation cells after inspection. Do not use Run all or the
`--standalone` training command as a resume shortcut. A portable evaluation-only
runner is still the next code task. It should write each completed arm/seed result
to persistent Drive storage immediately. Then copy or commit the small metrics and
plots back to the repository. Keep invariant diagnostics on CPU float64; Colab GPU
availability is not needed merely to load weights and complete these measurements.

## Where to resume

Read [research-synthesis-he.md](research-synthesis-he.md) first. All 36 training jobs
finished. The original runner stopped at the `budget-bound` checkpoint gate before
evaluation. The new archive includes G5a/G5b plus held-out test/200-step rollouts for
27 checkpoints; the remaining G1-G4, full G6a sweep, G6b, G7 and G9 are still pending.

Do **not** rerun `run_phase6.py --standalone` to resume: it retrains. Removing
`--standalone` or adding `--quick` is not a correct resume method either. The next
code task is a portable evaluation-only loader that selects the existing standalone
paths independently of quick mode, retains the budget-bound metadata, and saves
each completed arm/seed incrementally. That CLI change is not implemented here.

The files in `analysis-scripts/` are exact local script snapshots with Windows paths,
not drop-in Mac executables. Adapt paths/imports before using them. Stored checkpoints
have weights and metadata but no optimizer/scheduler/RNG state; exact optimizer
continuation is unavailable, while frozen evaluation is fully supported by the weights.
Use CPU float64 for invariant measurements; the original evaluation did so even when
training could use an accelerator.

Suggested prompt for the next coding session:

> Read spno/results/phase6-diagnosis-2026-09-22/HANDOFF_MAC.md and the research synthesis.
> First verify all 62 transferred artifacts. Continue Phase 6 evaluation from the 36
> saved checkpoints without retraining or editing convergence flags. Implement a
> portable evaluation-only entry point, separating artifact paths, quick mode and
> budget acceptance, with incremental per-arm/per-seed saving. Prioritize G4 and G6b.
> The existing Colab environment is available; inspect its notebook before using
> run cells, and keep checkpoints and evaluation outputs in persistent Drive storage.

## Verification scope

Archive integrity, JSON parsing, Markdown links, source-script syntax and recorded
evaluation consistency were checked. The full project pytest suite was not run:
pytest was unavailable in the accessible Windows Python environment. No model or
training implementation is changed by this archive.
