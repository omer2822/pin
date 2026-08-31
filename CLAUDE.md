# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo layout

This repo holds two mostly-independent Python projects, plus a few loose scratch files:

- **`spno/`** — the active project: a research study on structure-preserving neural operators for the parametric nonlinear Schrödinger equation. Almost all recent work happens here.
- **`pinn-neural-operators/`** — an earlier, self-contained PINN/FNO tutorial series that `spno` explicitly builds on and ports code from (see `spno/src/spno/domain.py`'s docstring).
- Root-level `schrodinger.py`, `split-step.py`, and `01_pytorch_pinns_to_fno_physicsnemo.ipynb` — standalone scratch/experimental scripts, not part of either package. Ignore these unless a task specifically concerns them.

## `spno/` — setup & commands

The package is `uv`-managed, `src`-layout, editable-installed into `spno/.venv` (Python 3.14). Dev dependencies (pytest) are not persisted into `.venv/bin`, so run tests via `uv run` rather than a bare `pytest`. All commands below are run from `spno/`.

```bash
uv run pytest tests -q                                      # whole suite
uv run pytest tests/test_split_learned.py                   # one file
uv run pytest tests/test_split_learned.py::test_name         # one test
```

`pyproject.toml` sets `filterwarnings = ["error::UserWarning"]` — any stray `UserWarning` fails the test that triggers it.

Each research phase has its own runner script under `scripts/`, following a common CLI pattern:

```bash
python scripts/run_phase45.py --epochs 80 --seeds 0 1 2
python scripts/run_phase6.py --quick          # smoke-test plumbing only
```

Common flags: `--quick` (2-3 epochs, plumbing check only), `--seeds`, `--epochs`, `--device auto`. Output lands in `results/<phase>-<confighash>[-mode-knob][-quick]/{metrics.json, plots/*.png}` — the directory name is derived from `config_hash()` so every result is traceable back to the exact config + seed that produced it. `--quick` runs are isolated into `-quick`-suffixed directories (also gitignored) so they can never be mistaken for real numbers.

**Phase dependency gotcha**: Phases 6-9 evaluate checkpoints trained by Phases 2-5; they do not train anything themselves. Per the runner scripts' own docstrings, the original Phase 2-5 baselines were budget-bound (25 epochs, early stopping never fired, val loss was still falling) — don't treat existing Phase 2-5 results as converged ground truth without checking `best_epoch` against the epoch budget first.

## `spno/` — architecture

**Config-hashing convention.** Experiment configuration is plain in-code frozen dataclasses, not YAML (despite `pyyaml` being a listed dependency): `DataConfig` and `Phase0Config` in `config.py`, plus `MisspecificationConfig` (Phase 9's data-generating-equation perturbation dials), which is deliberately kept in its own module rather than folded into `DataConfig` — merging it would change `config_hash(DataConfig())` and orphan already-generated data shards on disk. `config_hash()` (SHA256 of the dataclass, truncated) is used everywhere as the canonical identifier for result and data directories.

**One shared model interface.** Every learnable architecture, and even the reference solver, implements `StepOperator(nn.Module, ABC)` (`models/base.py`) with the call signature `(field, potential, alpha, beta, dt) -> field`:
- `models/fno.py` — Model A, the unrestricted FNO baseline.
- `models/projected.py` — Model B, wraps any `StepOperator` and rescales its output to conserve mass exactly.
- `models/split_learned.py` — the structure-preserving C-family: C1 (pointwise phase in ρ — exactly mass-conserving, symplectic, reversible, U(1)-equivariant by construction), C2 (FNO phase in ρ), C3 (phase in Re/Im ψ directly — a deliberate control that breaks U(1)/reversibility).

`StepOperator` also carries `supports_dt_transfer` / `supports_time_reversal` class attributes plus an `allow_dt_transfer()` context manager, so a model can only be evaluated outside its trained `dt` as a deliberate, scoped experiment rather than by accident.

**Module map / control flow.** A typical phase script wires these together in order:
`domain.py` (FFT/spectral primitives, `PeriodicDomain`, the mass invariant `l2_mass`) → `data/` (`generate.py` samplers, `datasets.py` shards, `corruption.py` noise, `shift.py`'s `SHIFT_SPECS` distribution-shift registry) → `models/` and `solvers/` (learned vs. deterministic reference operators — `solvers/split_step.py` for the exact case, `solvers/perturbed.py` for Phase 9's misspecified equations) → `losses/` (`relative_l2.py`, `pde_residual.py` for the PINO soft-physics term) → `train.py` (one shared training loop with early stopping, four modes: `train_one_step`, `train_rollout`, `train_multi_dt`, `train_pino`) → `evaluation/` (`rollout.py`, `conservation.py`, `dispersion.py`, `reversibility.py`, `resolution.py`, `spectral.py` probes) → `experiments.py` (shared bookkeeping — `load_shards`, `evaluate_model`, `run_identifier`/`save_run`) → `results/`.

**Precision-widening pattern.** Training runs in float32/complex64 for speed. Anything that measures an invariant (mass/energy drift, dispersion, reversibility) is first widened to float64/complex128 via `precision.widen_to_double` and evaluated on CPU, so float32 roundoff never masks a genuine architectural violation of a conservation law. Preserve this pattern when adding new evaluators.

For the research methodology, the full Phase 0-9 plan, and numbered global constraints (float64 discipline, warnings-as-errors, run-identifier hygiene, etc.), see `docs/PROMPT_PHASES_6_TO_9.md` and `docs/superpowers/plans/2026-08-11-phases-6-to-9.md` rather than relying on this summary.

## `pinn-neural-operators/` — tutorial series

Five self-contained PyTorch tutorials, no shared framework: `01`-`03` all solve the same 1D Burgers PDE via PINN, DeepONet, and FNO respectively so the approaches are directly comparable; `04_pinn_playground.py` is a reusable PINN harness across several registered PDEs (list them with `--list`); `05_3d_equations.py` generalizes the geometry/structure to multi-dimensional parametric PDE operators; `07_physicsnemo_heat_pinn.py` is a PhysicsNeMo-framework-backed companion (optional dependency, not in `requirements.txt`).

```bash
source pinn-neural-operators/venv./bin/activate   # note: dir is literally named "venv."
python common.py                                  # builds/checks the shared dataset
python 04_pinn_playground.py --pde schrodinger
python -m unittest test_pinn_playground -v         # run from inside pinn-neural-operators/
```

Read the docstring at the top of each numbered script before the code — that's where the explanation of what it does lives.
