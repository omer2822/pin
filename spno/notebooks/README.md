# Colab notebooks

| Notebook | Purpose |
|---|---|
| [00_training.ipynb](00_training.ipynb) | Discover Phase 6 artifacts; prepare, train, and resume explicitly selected experiments |
| [06_phase6_evaluation.ipynb](06_phase6_evaluation.ipynb) | Evaluate every Phase 6 arm from the transferred Windows checkpoints; self-contained (clones `master`, extracts the eval-only archive from Drive, saves each arm to Drive) |
| [07_pino_evaluation.ipynb](07_pino_evaluation.ipynb) | Evaluate the PINO sweep using saved checkpoints |
| [08_resolution_robustness.ipynb](08_resolution_robustness.ipynb) | Evaluate resolution, noise robustness, and sample efficiency |
| [09_misspecification.ipynb](09_misspecification.ipynb) | Evaluate equation misspecification sweeps with a shared zero-perturbation baseline |
| [10_dirichlet.ipynb](10_dirichlet.ipynb) | Solve Laplace/Poisson problems on a disk and rectangle, without training |

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
