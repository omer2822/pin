# Colab notebooks

| Notebook | Purpose |
|---|---|
| [00_training.ipynb](00_training.ipynb) | Discover Phase 6 artifacts; prepare, train, and resume explicitly selected experiments |
| [06_phase6_evaluation.ipynb](06_phase6_evaluation.ipynb) | Evaluate every Phase 6 arm from the transferred Windows checkpoints; self-contained (clones `master`, extracts the eval-only archive from Drive, saves each arm to Drive) |
| [07_pino_evaluation.ipynb](07_pino_evaluation.ipynb) | Compare A, A+PDE, B-loop, C1, and C1+PDE using saved checkpoints; both full residual sweeps, common metrics, CSV and plots |
| [08_resolution_robustness.ipynb](08_resolution_robustness.ipynb) | Evaluate resolution, noise robustness, and sample efficiency |
| [09_misspecification.ipynb](09_misspecification.ipynb) | Evaluate equation misspecification sweeps with a shared zero-perturbation baseline |
| [10_dirichlet.ipynb](10_dirichlet.ipynb) | Solve Laplace/Poisson problems on a disk and rectangle, without training |
| [11_hybrid_kinetic_ablation.ipynb](11_hybrid_kinetic_ablation.ipynb) | Self-contained Colab study of exact kinetic + learned local across G1–G9, paired component ablations, fresh probe seeds, bandwidth sweeps, dispersion, and long rollouts |
| [12_gauge_identifiable_c1.ipynb](12_gauge_identifiable_c1.ipynb) | Self-contained: reciprocal component ablation (K_exact+L_θ vs K_θ+L_exact), gauge-fixed swaps, the local law L_θ ≈ βρ − V, and the same ablation on C1g |

## Hybrid follow-up notebook

Upload **`11_hybrid_kinetic_ablation.ipynb` directly to Colab** and run top to bottom.
The experiment source is embedded in the notebook: no separate Python files,
source bundle, or GitHub update is needed. The existing Phase 6 eval-only/full
checkpoint ZIP should be in MyDrive, or set an extracted `SOURCE_ROOT` in cell 1.
Only the base, G6a, and G7 C1 checkpoints are required; no training data is needed.

The default study uses 3 training seeds, 5 independent probe seeds, 16 ICs per probe,
bandwidths 4–32, and a 2,000-step long horizon. G8 is the newly assigned long-rollout
and conservation arm. The four literal frozen component combinations share each
probe batch and refined reference. Results resume per case/probe/training seed and
include PNG/PDF figures, per-IC spectra and metrics, paired bootstrap comparisons,
and reference refinement checks. Budget-bound source checkpoints remain labeled
exploratory. `SMOKE=True` is a setup check, not a scientific result.

Maintainers: regenerate the embedded source after edits with
`python scripts/build_hybrid_notebook.py`. The builder deliberately embeds no
weights, datasets, outputs, or notebook copies.

## Getting started after Phase 6

1. Build the source bundle with `python scripts/build_colab_bundle.py` and download
   `dist/spno-colab-source.zip`. It contains the code and notebooks, **not** your
   trained model weights.
2. Upload the training notebook to Colab. The setup cell requests the source bundle
   if it is not already installed in the runtime. Code runs locally in the runtime;
   Drive stores results and training progress.
3. Set `SOURCE_ROOT` to your run's `phase6-standalone-artifacts` directory. It
   contains `checkpoints/phase6/...` and `data/...`. You do not need to rerun
   Phases 1–5.
4. If the artifacts are still under `/content` in the original runtime, also set
   `COPY_TO` to a directory in Drive. Copying verifies file contents and preserves
   the source. Complete this step before closing that runtime: a notebook in a
   different runtime cannot access the original runtime's temporary disk.
5. Check that the model inventory includes every seed from your run. `SOURCE_CONFIG`
   can point to a JSON file containing `data` and `train`. If the report does not
   include the epoch budget, fill in `SOURCE_TRAIN_CONFIG` from your original
   training command, **not** from `best_epoch`. Once the configuration is known,
   the notebook saves it to `OUTPUT_ROOT/source-config.json` for the evaluation
   notebooks.
6. Leave `TRAIN_PHASES=[]` to inspect the inventory without preparing any training.
   To prepare experiments, add phase numbers, select the sweeps, and then copy only
   the experiment IDs you want to run into `TRAIN_SELECTED_IDS`. Preparing Phase 9
   may generate datasets; training starts only for explicitly selected IDs.
7. Open the evaluation notebooks with the same `SOURCE_ROOT`, `OUTPUT_ROOT`, and
   `SOURCE_CONFIG`. They never start training. A missing-checkpoint error identifies
   the experiment you need to prepare in the training notebook.

An existing model that has not converged is retained rather than automatically
retrained. Research evaluation rejects it by default. `ALLOW_BUDGET_BOUND=True`
(or `--allow-budget-bound`) enables evaluation explicitly marked as exploratory; for
Phase 6 the report identifier also gains a `-budget-bound` suffix. Saved convergence
metadata is never modified. A `quick` run is always a plumbing
check, not a research result.

## Experiment identity and recovery

Training identity includes the bytes of the training and validation datasets,
architecture, preprocessing, seed, loss function, and `TrainConfig`. Changing the
epoch budget also changes the learning-rate schedule, so it creates a new
experiment; it does not automatically extend an existing schedule. Evaluation
settings do not change training identity. Evaluation records include hashes of
the weights and datasets they use.

`model.pt` contains the best weights. `latest.pt` stores training state at the end
of an epoch, including the optimizer, scheduler, and random-number-generator states.
`latest.pt.previous` provides a recovery copy if the latest saved file is corrupted.
Resume with the same epoch budget and configuration. A disconnect during an epoch
requires repeating the unfinished epoch from the last successful save. Bitwise
recovery was tested on CPU in the same software environment; changing hardware or
PyTorch versions may change numerical results.

Legacy artifacts contain weights but no optimizer state, so exact mid-training
recovery is unavailable for them. Missing history remains marked as missing; it is
not reconstructed from a single `best_epoch` value. `workflow.base_models()` builds
`B-post` around the same core used by A.

Each sample-efficiency fraction must select at least one training example. For a
tiny smoke test, choose suitable fractions, such as `[0.5, 1.0]`. Report the full
sweep when publishing research results.

## Local execution and CLI

Run these commands from the `spno` directory, using its Python environment:

```bash
uv sync --extra dev --extra dirichlet --extra notebooks
uv run --extra dev --extra dirichlet --extra notebooks pytest tests -q
python scripts/run_phase7.py --stage prepare \
  --source-root /path/to/phase6-standalone-artifacts \
  --output-root /path/to/workflow --source-config /path/to/source-config.json --seeds 0 1 2
python scripts/run_phase7.py --stage train \
  --source-root /path/to/phase6-standalone-artifacts \
  --output-root /path/to/workflow --source-config /path/to/source-config.json --seeds 0 1 2
python scripts/run_phase7.py --stage evaluate \
  --source-root /path/to/phase6-standalone-artifacts \
  --output-root /path/to/workflow --source-config /path/to/source-config.json --seeds 0 1 2
python scripts/run_phase6.py --stage evaluate \
  --source-root /path/to/phase6-standalone-artifacts \
  --output-root /path/to/workflow --arms G5a G5b
```

The same stages are available for Phases 8–9. Phase 6 supports inventory inspection
and evaluation of imported artifacts; use its original trainer for new Phase 6
training. Omitting `--stage` preserves the scripts' historical behavior, including
repeated training in the old `--standalone` workflow. **Always select
`--stage evaluate` when rerunning evaluations.**

In the new stages, the training protocol comes from the source run or
`--source-config`. The historical `--epochs` flag does not override it and raises
an error when explicitly supplied alongside `--stage`.

To change the requested protocol in notebooks, use the same `TRAIN_OVERRIDES` for
training and evaluation. In the CLI, pass `--train-config` with a JSON file
containing only the changes, such as `{"epochs": 100}`. This does not alter the
source protocol. In Python, use:

```python
workflow.prepare(
    ...,
    train_config=replace(workflow.train_config, epochs=...),
)
```

Notebook paths can also be configured through `SPNO_PROJECT_ROOT`,
`SPNO_SOURCE_ROOT`, `SPNO_OUTPUT_ROOT`, and `SPNO_SOURCE_CONFIG`. `SPNO_OPTIONS`
accepts experiment selections as JSON, for example
`{"seeds": [0], "lambdas": [0.0]}`. In local Jupyter, select a kernel from the
installed project environment.

## Phase 7 finalized comparison

`07_pino_evaluation.ipynb` evaluates **A, A+PDE, B-loop, C1, C1+PDE**.
B-loop is the single hard mass-projection baseline, trained with projection; it has
no PDE-weight sweep. Both A and C1 use the selected lambda values, with zero always
included as a matched control. The default `[0, 0.01, 0.1, 1, 10]` and three seeds
produce 33 experiments (9 controls that may be reused, plus 24 positive-weight runs).
C1 retains the source kinetic/local architecture. Positive-weight training starts
from the same seed initialization within each family, not from fitted base weights.

Use the full standalone source artifacts including train/validation/test shards and
the original training configuration. The Phase 6 evaluation-only transfer, which
omits training data, is insufficient. Prepare phase 7 in `00_training.ipynb`, train
explicitly selected missing IDs, then run notebook 07 with the same paths, seeds,
lambdas and training overrides. Evaluation checks all required checkpoints first
and never starts training. Nonconverged checkpoints require explicit exploratory
opt-in; controls-only selections are labelled incomplete.

The report saves `metrics.json`, `comparison.csv`, and a four-panel plot comparing
one-step error, rollout error, mass drift, and energy drift. All models share the
actual horizon `min(100, available steps)`; rollout measurements use float64 on CPU.
CSV and plots report seed means and sample standard deviations (a single seed has
no spread estimate). Full per-seed measurements and checkpoint provenance remain in
JSON. The notebook states the CN mass-conservation and finite-time-step bias caveats.
The historical CLI without `--stage` remains the A-only PINO sweep; use the explicit
`--stage prepare|train|evaluate` workflow for this five-family comparison.


Phase 7 and the training notebook now default to `SOURCE_ROOT=None`. They discover
full extracted Phase 6 artifacts under the project results directory, the Phase 6
Colab checkout (`/content/pin/spno/results`), and MyDrive. A unique full
`phase6-standalone-artifacts-*.zip` is unpacked into a separate local import directory.
You may set `SOURCE_ROOT` to an exact folder, an enclosing transfer folder, or a full
ZIP. Ambiguous runs require an explicit selection. Evaluation-only transfers produce
an actionable missing-`train.pt` message, rather than proceeding to training setup.
Use the updated source ZIP when prompted; the code bundle itself contains no weights.

## Gauge-identifiable C1 (C1g) follow-up

C1g is C1 with the kinetic zero mode pinned, κθ(k) = f(k²) − f(0). It is an exact
reparameterization of C1: same parameters, same seed initialization, same data and
40-epoch protocol, so C1 vs C1g is a paired comparison.

1. **Train C1g** in `00_training.ipynb`: `TRAIN_PHASES=[7]`, `C1G_LAMBDAS=[0.0, 0.01]`.
   This adds 6 jobs (C1g and C1g+PDE × 3 seeds). Every existing A / C1 / B-loop job keeps
   its identity and is reused.
2. **Extended Phase 7** in `07_pino_evaluation.ipynb` with `C1G_LAMBDAS=[0.0, 0.01]`:
   A, A+PDE, B-loop, C1, C1+PDE, C1g, C1g+PDE. The probes cell picks C1g up automatically.
   An empty `C1G_LAMBDAS` reproduces the original five-arm report identity.
3. **Upload `12_gauge_identifiable_c1.ipynb`** and run all cells. Section (a) re-summarizes
   the saved hybrid run in place, keeping the old summary as `summary.v1.json`. Section (b)
   must reproduce the 2026-09-23 numbers before any new result is trusted. Sections (c)
   and (d) include C1g once step 1 is done.

Maintainers: `python -m scripts.build_gauge_notebook` and
`python scripts/build_hybrid_notebook.py` re-embed the source after code edits.
