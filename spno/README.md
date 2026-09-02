# spno

Structure-preserving neural operators for the parametric nonlinear Schrödinger equation.

## What this is

A research study asking one question: does baking physical structure — mass conservation,
symplecticity, time-reversibility, U(1) equivariance — directly into a neural operator's
*architecture* beat an unconstrained baseline, or softer alternatives like hard constraint
projection or a PINO-style soft-physics loss?

The task throughout is learning the one-step time-evolution operator

```
i psi_t + alpha * Laplacian(psi) + beta * |psi|^2 * psi - V(x) * psi = 0        (periodic domain)

(psi_n, V, alpha, beta)  |->  psi_{n+1}
```

for the parametric nonlinear Schrödinger equation, compared against an exact numerical
reference (Strang splitting). Four families of models are compared under an identical
training budget and data:

| Model | Structure it guarantees |
|---|---|
| **A** — plain FNO | none |
| **B** — FNO + mass projection | mass |
| **C1** — split-step, pointwise phase in ρ | mass, reversible, symplectic, U(1) |
| **C2** — split-step, FNO phase in ρ | mass, reversible, U(1) |
| **C3** — split-step, phase in Re/Im ψ (control) | mass only |

The work is organized as a ten-phase pipeline (Phase 0 through Phase 9 — reference
validation, data generation, baseline training, structured-model training and evaluation,
generalization/identifiability probes, PINO baseline, resolution/robustness transfer, and
equation-misspecification sweeps). See `docs/PROMPT_PHASES_6_TO_9.md` for the full research
plan and the exact question each phase answers.

## Quick start

Requires Python >= 3.11 and [`uv`](https://docs.astral.sh/uv/).

```bash
cd spno
uv sync --extra dev          # creates/updates .venv, installs spno + pytest
uv run pytest tests -q       # whole suite should pass
python scripts/run_phase0.py --quick   # smoke-test the reference solver end to end
```

`--quick` runs a couple of epochs purely to check plumbing — its numbers are not meaningful
and land in a separate `-quick`-suffixed results directory so they can never be mistaken for
real ones. Drop `--quick` (and raise `--epochs`) once you actually want numbers.

## Running an experiment phase

Each phase has its own runner under `scripts/`, sharing a common flag set:

```bash
python scripts/run_phase1.py --quick                      # generate the trajectory dataset
python scripts/run_phase45.py --epochs 80 --seeds 0 1 2   # train the structured C-family
python scripts/train_phase6_arms.py --epochs 80 --seeds 0 1 2  # train Phase 6 ablations
python scripts/run_phase6.py --device auto                # checkpoint-only probes
```

Common flags: `--quick`, `--seeds`, `--epochs`, `--device auto`. Phase 6 evaluates
checkpoints produced by Phases 2-5, while Phases 7-9 run their own controlled training or
retraining arms. Base checkpoints must therefore be converged before Phase 6, and every
newly trained arm must likewise be checked for budget-bound histories (`best_epoch` at the
epoch cap) before its numbers are reported.

Phase 6 has one additional dependency: run `train_phase6_arms.py` after the Phase 2-3
and matching Phase 4-5 one-step/K0L0 jobs. It trains `A-wide`, the fixed-alpha G7
ablation, and the multi-dt G6a arm. The evaluator restores every learned model strictly
from these checkpoint families; it never substitutes freshly initialized weights.

Phase 6 quick runs preserve the production grid but use two trajectories and two steps
for Phase 6-only shards. Their datasets live under `results/phase6-quick-artifacts/`, their
checkpoints are tagged `-quick`, and their metadata remains `converged=false`. They are
plumbing checks only: the explicit `--quick` evaluator override may restore them, but
they are never reportable scientific results.

Phase 9 quick runs use an independent 16-point, two-step distribution generated entirely
in memory. They never read, create, or overwrite production data shards; their metrics and
checkpoints use a `-misspec-quick` identifier. A minimal end-to-end smoke command is:

```bash
python scripts/run_phase9.py --quick --device cpu --seeds 0 --epochs 1 \
  --sigmas 0 0.1 --gammas 0 0.001
```

Every run writes to `results/<phase>-<config-hash>[-mode-knob][-quick]/`:

```
metrics.json     # the full config used, plus every measured result
plots/*.png      # generated figures
```

Model artifacts live separately under `results/checkpoints/<family>/<identifier>/`.
Phase 6-only tags use nested names such as `G7-alpha-fixed/A-seed0.pt` and
`G6a/C1-seed0.pt`; each checkpoint carries its exact architecture, preprocessing scale,
training dial, kinetic mode, seed, and convergence status.

The directory name is derived from `config_hash()` of the run's configuration, so any
reported number is traceable back to the exact config and seed that produced it. Bulk
checkpoint/array data (`*.pt`, `*.npz`) and `-quick` directories are gitignored; `metrics.json`
and plots are tracked.

## Testing

```bash
uv run pytest tests -q                                  # whole suite
uv run pytest tests/test_split_learned.py                # one file
uv run pytest tests/test_split_learned.py::test_name      # one test
```

`pyproject.toml` turns any stray `UserWarning` into a hard test failure
(`filterwarnings = ["error::UserWarning"]`).

## Architecture, in brief

- **Configs are code, not YAML.** Frozen dataclasses (`DataConfig`, `Phase0Config`,
  `MisspecificationConfig`) plus `config_hash()` make every result reproducible from a
  config + seed alone.
- **One shared model interface.** Every architecture — and the reference solver — implements
  `StepOperator` (`src/spno/models/base.py`) with the same
  `(field, potential, alpha, beta, dt) -> field` signature, so training, evaluation, and
  probing code is architecture-agnostic.
- **Precision discipline.** Training runs in float32/complex64 for speed; anything measuring
  a conservation law is explicitly widened to float64/complex128 and evaluated on CPU first,
  so roundoff never masks a genuine architectural violation.

The full module map and control flow are documented in the repository's top-level
[`CLAUDE.md`](../CLAUDE.md). For the research methodology, phase-by-phase goals, and the
numbered global constraints (float64 discipline, warnings-as-errors, run-identifier hygiene,
etc.), see [`docs/PROMPT_PHASES_6_TO_9.md`](docs/PROMPT_PHASES_6_TO_9.md) and
[`docs/superpowers/plans/2026-08-11-phases-6-to-9.md`](docs/superpowers/plans/2026-08-11-phases-6-to-9.md).
