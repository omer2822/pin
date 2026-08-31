# Phase 6–9 Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Phase 6–9 runners from partially descriptive manifests into reproducible scientific pipelines that load or train the intended models, compute every reported metric, reject incomplete evidence, and preserve the numerical guarantees already covered by unit tests.

**Architecture:** Introduce one typed checkpoint contract and one payload-completeness gate shared by all runners. Repair the three isolated numerical defects first, then make each phase independently executable: Phase 6 loads converged base and ablation checkpoints and evaluates real shift shards; Phase 7 performs a seeded, normalized PINO sweep; Phase 8 evaluates rebound models and trains sample-efficiency arms; Phase 9 generates auditable misspecified shards, retrains all models, and computes crossovers only from complete measurements.

**Tech Stack:** Python 3.11+, PyTorch 2.0+, NumPy, Matplotlib (Agg), pytest 7+, dataclasses, pathlib.

**Spec:** `docs/PROMPT_PHASES_6_TO_9.md`; this plan remediates the findings from the review of Git range `b8c1414..d9d7ccf` and supersedes incomplete runner portions of `docs/superpowers/plans/2026-08-11-phases-6-to-9.md`.

## Global Constraints

- Preserve `DataConfig()` hash `bd4e108527`; misspecification and shift knobs must remain outside `DataConfig`.
- Train in float32/complex64 on the selected device; perform invariant and scientific measurements in float64/complex128 on CPU via `precision.widen_to_double`.
- Never call `.double()` or `.to(torch.float64)` on models containing complex spectral parameters.
- Structural claims must be tested at random untrained weights with a paired negative.
- Report all seeds, ranges, parameter counts, convergence states, and preprocessing metadata.
- Do not pool one-step and rollout training modes.
- Phase 7b space-time PINO remains explicitly optional and outside this remediation.
- Phase 8 band-limited resampling and new-high-k evaluation remain separate results; the latter is cross-referenced to G4.
- A runner must raise before `save_run` when any selected arm lacks the metrics needed by its plots or conclusions.
- `--quick` runs use tiny budgets and isolated `-quick` result/checkpoint paths; their numbers are plumbing checks, not scientific results.
- Use `raise RuntimeError`, not `assert`, for runtime scientific gates.
- Add no third-party dependency.

---

## Diagnostic Summary

### Finding: G5b continuation rejects the configured anchor jump

**Severity:** Critical for Phase 6 execution.

**Understanding:** `run_phase6.py` prepends `0.0` to an in-range grid beginning at `0.7`; at `k=24`, `0.7 * 24**2 * 0.01 = 4.032 >= pi`, so `omega_by_alpha_continuation` correctly refuses the path.

**Root Cause:** The derivative grid and absolute-continuation path were modeled as one configuration even though they have different domains and density requirements.

**Impact:** Selecting G5b crashes before results are saved; the documented smoke path cannot complete.

**Solution Options:** (1) Remove continuation and retain only the in-range derivative—smallest change, but loses the separately labeled extrapolation diagnostic. (2) Use a dense anchor path from `0.0` to the terminal alpha—retains both claims at negligible cost. (3) Numerically unwrap a sparse path—more complex and weaker than satisfying the theorem's stated precondition.

**Recommended Fix:** Keep `ALPHA_DERIVATIVE_GRID=tuple(0.7 + 0.05*i for i in range(9))` and add `alpha_continuation_grid(target, max_k, dt)` whose step is strictly below `pi/(max_k**2*dt)`; use it for every continuation and record its terminal alpha and maximum phase increment.

**Prevention:** Add a runner-level test that executes G5b for `k=(24,32)` and a unit test that every supplied alpha sequence is finite, strictly monotone, and duplicate-free.

**Summary:** Separate local derivative sampling from anchored continuation; about 1 hour, low implementation risk.

### Finding: Phase 6 evaluates random models and leaves arms descriptive

**Severity:** Critical for scientific validity.

**Understanding:** `build_models` constructs random A/B/C/A-wide instances; G1–G4 and G7 serialize shift descriptions, G6a is absent, while G5/G6b/G9 operate on the random weights.

**Root Cause:** The earlier plan deferred long runs but the runner contract still claimed to consume checkpoints; no checkpoint format or load path was implemented, so descriptive scaffolding was mistaken for a deliverable.

**Impact:** The output cannot support any hypothesis-class or generalization conclusion.

**Solution Options:** (1) Train inside `run_phase6.py`—self-contained but mixes expensive training with evaluation and makes reruns wasteful. (2) Persist checkpoints from Phase 2–5 plus a dedicated Phase 6 arm trainer—clean separation and reproducible reuse. (3) Accept arbitrary checkpoint paths on every command—flexible but difficult to audit and compare.

**Recommended Fix:** Add a typed checkpoint artifact, save converged base models from Phase 2–5, train Phase 6-only arms in `scripts/train_phase6_arms.py`, and require `run_phase6.py` to load all selected checkpoints and evaluate generated/loaded shards.

**Prevention:** A payload schema gate must reject arm entries containing only `note`, `identifier`, or empty metric mappings.

**Summary:** Establish artifact-driven Phase 6 training/evaluation; 2–3 engineering days plus separate compute time, medium risk.

### Finding: Phase 7 raises `KeyError` after training

**Severity:** Critical for execution.

**Understanding:** `budget_warning` consumes `entry["converged"]`, but the runner supplies `entry["histories"]`.

**Root Cause:** The shared convergence API was integrated without a runner-level contract test.

**Impact:** Completed training is discarded before metrics and plots are saved.

**Solution Options:** (1) Teach `budget_warning` to infer histories—couples it to serialized history format. (2) Compute `converged(history)` where the live object exists—matches all other runners. (3) Remove the warning—would permit budget-bound claims.

**Recommended Fix:** Store `converged` per seed, aggregate it under the existing key, and add a quick sweep integration test.

**Prevention:** Type the sweep aggregation through a single helper and validate it before saving.

**Summary:** Supply the established convergence contract; under 1 hour, low risk.

### Finding: Phase 8 records metadata instead of model results

**Severity:** Critical for Phase 8 deliverables.

**Understanding:** Resolution transfer stores shapes, robustness stores corruption calibration, sample efficiency stores pair counts, and both resolution plot panels lack measured model error.

**Root Cause:** Data-transformation helpers were completed, but orchestration stopped before inference/training and no result schema required outcome metrics.

**Impact:** Phase 8 cannot answer resolution, robustness, or sample-efficiency questions.

**Solution Options:** (1) Narrow Phase 8 to only tested resampling primitives—honest but drops requested experiments. (2) Load checkpoints for inference arms and retrain only sample fractions—meets the spec with bounded compute. (3) Retrain every noise/resolution arm—unnecessary because noise is an evaluation perturbation.

**Recommended Fix:** Load A/C1 checkpoints, evaluate complete resampled trajectories, generate a fine-grid test set with modes above the coarse Nyquist, evaluate separate field/potential noise inputs, and retrain A/C1 only for each sample fraction and seed.

**Prevention:** Require `one_step_error`, `rollout_error`, and seed aggregates in every plotted Phase 8 entry.

**Summary:** Complete inference and bounded retraining paths; 1–2 engineering days plus compute, medium risk.

### Finding: Phase 9 infers bounded crossovers from absent ratios

**Severity:** Critical for scientific validity.

**Understanding:** Every `relative_to_A` mapping is empty, but `locate_crossover` interprets missing entries as “still beats A.”

**Root Cause:** Missing observations and observations below threshold share the same control path; the runner lacks a completeness invariant.

**Impact:** It emits a false publishable conclusion without generating data or fitting models.

**Solution Options:** (1) Return an `insufficient_data` status—safe but still leaves the phase incomplete. (2) Raise on any missing/non-finite ratio and complete the training pipeline—safe and meets the spec. (3) Treat missing as NaN and plot gaps—visual honesty but crossover text can still be misused.

**Recommended Fix:** Generate/load each dial's three shards, retrain all five models per seed, aggregate horizon-100 rollout error, populate finite ratios, and make `locate_crossover` reject incomplete data.

**Prevention:** Test that an empty or partial ratio matrix raises before `save_run`.

**Summary:** Replace manifest entries with measured sweeps and fail closed; 2–3 engineering days plus compute, medium-high risk.

### Finding: Phase 7 is not a controlled lambda comparison

**Severity:** Major.

**Understanding:** Models are constructed before the trainer seeds RNG, and `field_scale` defaults to `1.0`; therefore lambda arms do not share initialization or Phase 2–3 preprocessing.

**Root Cause:** Seeding and normalization responsibilities are split between model construction and training without a shared factory.

**Impact:** Differences can be attributed to initialization or scaling rather than lambda; lambda zero is not the claimed control.

**Solution Options:** (1) Seed immediately before each constructor and pass measured scale—minimal. (2) Clone one initialized state across lambdas—strongest paired comparison. (3) Average independent initializations—statistically valid at larger seed counts but noisier and more expensive.

**Recommended Fix:** Compute `field_scale` once, construct a seed-specific baseline state, deep-copy/load that state for every lambda, and assert lambda-zero history and weights match `train_one_step` in a tiny deterministic test.

**Prevention:** Centralize FNO construction and record the initialization fingerprint in checkpoint metadata.

**Summary:** Pair initialization and preprocessing across lambda; 2–3 hours, low risk.

### Finding: Phase 7 aggregates validation loss and omits invariant metrics

**Severity:** Major.

**Understanding:** `one_step` is `history.best_val`, while evaluated `one_step_test`, mass drift at 100, and energy drift at 100 are not promoted into sweep summaries.

**Root Cause:** Plot fields were defined before the evaluator payload was wired into aggregation.

**Impact:** The reported metric is mislabeled and two plot panels are empty.

**Solution Options:** (1) Read values inline from nested dictionaries—small but duplicates brittle indexing. (2) Add a typed `aggregate_pino_seed_metrics` helper—testable and explicit. (3) Change plots to use histories—would preserve the scientific mislabeling.

**Recommended Fix:** Aggregate `metrics.one_step_test` and the horizon-100 rollout `mass_drift`/`energy_drift`; record mean/min/max and retain every per-seed observation.

**Prevention:** Plot tests should assert each panel receives finite values from a complete payload fixture.

**Summary:** Aggregate evaluated test/invariant outcomes, not training history; 2 hours, low risk.

### Finding: Gain/loss local update is not the exact subflow

**Severity:** Major numerical-modeling defect.

**Understanding:** With gain/loss, density changes as `rho(t)=rho0*exp(2*gamma*t)`, so using `beta*rho0*dt` for nonlinear phase is not the exact local flow claimed by the operator.

**Root Cause:** The real gain factor was treated as commuting with the entire nonlinear local flow; it commutes with kinetic evolution, not with density-dependent phase evolution.

**Impact:** The truth generator has an avoidable first-order local splitting error and contaminates small-gamma comparisons.

**Solution Options:** (1) Retitle it as an approximate split and quantify convergence—honest but weakens the reference. (2) Use the analytic integral `expm1(2*gamma*dt)/(2*gamma)`—exact and inexpensive. (3) Substep gain/nonlinearity—more compute and still approximate.

**Recommended Fix:** Implement the analytic phase integral with a stable zero branch and test against the closed-form pure-local solution for positive, negative, and near-zero gamma.

**Prevention:** Every new “exact flow” docstring must have a closed-form oracle test, not only an invariant-rate test.

**Summary:** Replace the nonlinear phase duration by its analytic gain-weighted integral; 2 hours, low risk.

### Finding: Misspecification shards lack provenance

**Severity:** Major for reproducibility.

**Understanding:** Shard metadata does not record sigma, gamma, reference class, or misspecification identifier.

**Root Cause:** `generate_shard` accepts an opaque module and has no provenance argument.

**Impact:** A saved shard cannot be audited independently of its directory name.

**Solution Options:** (1) Introspect the module—fragile across wrappers. (2) Pass an explicit immutable provenance mapping—simple and auditable. (3) Put the whole `MisspecificationConfig` into `DataConfig`—breaks the production hash constraint.

**Recommended Fix:** Add `reference_metadata: Mapping[str, str | int | float | bool] | None`, merge it into metadata under `reference`, and have `MisspecificationConfig.provenance(data_config)` produce the canonical mapping.

**Prevention:** Validate loaded Phase 9 shard metadata against the requested identifier and dial before training.

**Summary:** Persist and verify canonical dial provenance; 2–3 hours, low risk.

### Finding: Odd-grid endpoint modes are treated as Nyquist

**Severity:** Minor but correctness-relevant.

**Understanding:** Odd grids have no self-conjugate Nyquist mode, yet index `n_source//2` is checked and the symmetric copy loop drops valid endpoint modes.

**Root Cause:** The even-grid FFT layout was generalized with integer division rather than explicit representable-wavenumber intersection.

**Impact:** Valid odd-grid band-limited fields can be rejected or truncated.

**Solution Options:** (1) Forbid odd grids—simple but unnecessarily narrows a general utility. (2) Branch copy bounds by parity—works but is easy to regress. (3) Map the intersection of integer FFT wavenumbers—clear and general.

**Recommended Fix:** Check Nyquist ambiguity only when `n_source % 2 == 0`; copy coefficients using explicit integer wavenumber sets from `torch.fft.fftfreq(n, d=1/n).round().int()`.

**Prevention:** Add odd→odd, odd→even, and even→odd round-trip tests using both positive and negative endpoint modes.

**Summary:** Make resampling parity-aware through exact wavenumber mapping; 2 hours, low risk.

### Finding: Duplicate/non-finite alpha samples are not rejected

**Severity:** Minor.

**Understanding:** A zero gap reaches slope division; NaN/Inf values can also bypass meaningful density checks.

**Root Cause:** Validation only checks list length and maximum absolute gap.

**Impact:** Callers receive division errors or non-finite scientific output instead of a diagnostic naming the invalid grid.

**Solution Options:** (1) Deduplicate/sort silently—changes caller intent. (2) Reject non-finite, duplicate, and non-monotone grids—fail-fast and auditable. (3) Permit decreasing grids—valid mathematically but not used by any runner and increases surface area.

**Recommended Fix:** Require finite, strictly increasing alpha values in both derivative and continuation functions.

**Prevention:** Parameterize validation tests over duplicate, descending, NaN, and Inf inputs.

**Summary:** Add one shared strict-alpha-grid validator; 1 hour, low risk.

---

## File Structure

**Create:**

- `src/spno/checkpoints.py` — checkpoint metadata, path construction, atomic save/load, compatibility validation.
- `src/spno/evaluation/payloads.py` — finite-value and selected-arm completeness gates used before serialization.
- `scripts/train_phase6_arms.py` — trains A-wide, fixed-alpha G7, and multi-dt G6a checkpoints without mixing training into evaluation.
- `tests/test_checkpoints.py` — checkpoint round-trip and incompatibility tests.
- `tests/test_phase_runners.py` — tiny dependency-injected runner and payload tests.

**Modify:**

- `src/spno/evaluation/dispersion.py`, `tests/test_dispersion.py` — strict alpha grids.
- `src/spno/evaluation/resolution.py`, `tests/test_resolution.py` — parity-correct spectral resampling.
- `src/spno/solvers/perturbed.py`, `src/spno/misspecification.py`, `src/spno/data/datasets.py`, `tests/test_perturbed.py` — exact gain/loss local flow and shard provenance.
- `scripts/run_phase23.py`, `scripts/run_phase45.py` — save converged checkpoints.
- `scripts/run_phase6.py` — checkpoint loading, dense continuation, actual G1–G7/G9 evaluation.
- `scripts/run_phase7.py` — controlled initialization, complete aggregation, convergence contract.
- `scripts/run_phase8.py` — checkpoint inference, resolution/noise metrics, sample-fraction training.
- `scripts/run_phase9.py` — shard generation/loading, training/evaluation, fail-closed crossovers.
- `README.md` — exact training/evaluation command sequence and artifact locations.

---

### Task 1: Shared checkpoint and payload contracts

**Files:**
- Create: `src/spno/checkpoints.py`
- Create: `src/spno/evaluation/payloads.py`
- Create: `tests/test_checkpoints.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- Produces: `CheckpointMetadata`, `checkpoint_path`, `save_checkpoint`, `load_checkpoint_payload`, `restore_checkpoint(model, payload, *, expected_data_hash: str, allow_budget_bound: bool = False)`.
- Produces: `require_phase_arms(phase: int, selected: Iterable[str], experiments: dict) -> None`.
- Consumed by: Tasks 4–7.

- [ ] **Step 1: Write failing checkpoint round-trip and mismatch tests**

```python
# tests/test_checkpoints.py
from dataclasses import replace
import pytest
import torch

from spno.checkpoints import (
    CheckpointMetadata, load_checkpoint_payload, restore_checkpoint, save_checkpoint,
)
from spno.config import DataConfig, config_hash
from spno.models.fno import FNOStepOperator


def _model(data, scale=0.5):
    return FNOStepOperator(
        data.domain, modes=4, width=8, n_layers=1,
        alpha_range=data.alpha_range, beta_range=data.beta_range,
        field_scale=scale, trained_dt=data.dt,
    )


def test_checkpoint_roundtrip_carries_preprocessing_and_convergence(tmp_path):
    data = replace(DataConfig(), grid_size=16)
    model = _model(data)
    meta = CheckpointMetadata(
        schema_version=1, model_name="A", data_hash=config_hash(data), seed=3,
        train_mode="one-step", field_scale=0.5, trained_dt=data.dt,
        architecture={"modes": 4, "width": 8, "n_layers": 1},
        converged=True, best_epoch=7,
    )
    path = tmp_path / "A-seed3.pt"
    save_checkpoint(path, model, meta)
    payload = load_checkpoint_payload(path)
    restored = _model(data)
    restore_checkpoint(restored, payload, expected_data_hash=config_hash(data))
    assert payload.metadata == meta
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def test_nonconverged_or_wrong_dataset_checkpoint_is_refused(tmp_path):
    data = replace(DataConfig(), grid_size=16)
    path = tmp_path / "bad.pt"
    meta = CheckpointMetadata(
        schema_version=1, model_name="A", data_hash="wrong", seed=0,
        train_mode="one-step", field_scale=0.5, trained_dt=data.dt,
        architecture={"modes": 4, "width": 8, "n_layers": 1},
        converged=False, best_epoch=1,
    )
    save_checkpoint(path, _model(data), meta)
    payload = load_checkpoint_payload(path)
    with pytest.raises(RuntimeError, match="dataset hash"):
        restore_checkpoint(_model(data), payload, expected_data_hash=config_hash(data))


def test_budget_bound_checkpoint_requires_an_explicit_quick_run_override(tmp_path):
    data = replace(DataConfig(), grid_size=16)
    path = tmp_path / "quick.pt"
    meta = CheckpointMetadata(
        schema_version=1, model_name="A", data_hash=config_hash(data), seed=0,
        train_mode="one-step", field_scale=0.5, trained_dt=data.dt,
        architecture={"modes": 4, "width": 8, "n_layers": 1},
        converged=False, best_epoch=0,
    )
    save_checkpoint(path, _model(data), meta)
    payload = load_checkpoint_payload(path)
    with pytest.raises(RuntimeError, match="budget-bound"):
        restore_checkpoint(_model(data), payload, expected_data_hash=config_hash(data))
    restore_checkpoint(_model(data), payload, expected_data_hash=config_hash(data),
                       allow_budget_bound=True)
```

- [ ] **Step 2: Run the tests and verify import failures**

Run: `.venv/bin/python -m pytest tests/test_checkpoints.py -q`

Expected: collection fails because `spno.checkpoints` does not exist.

- [ ] **Step 3: Implement the checkpoint contract**

```python
# src/spno/checkpoints.py
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import os
import torch


@dataclass(frozen=True)
class CheckpointMetadata:
    schema_version: int
    model_name: str
    data_hash: str
    seed: int
    train_mode: str
    field_scale: float | None
    trained_dt: float | None
    architecture: dict[str, Any]
    converged: bool
    best_epoch: int


@dataclass(frozen=True)
class CheckpointPayload:
    metadata: CheckpointMetadata
    state_dict: dict[str, torch.Tensor]


def checkpoint_path(root: Path, family: str, identifier: str, model: str, seed: int) -> Path:
    return root / "checkpoints" / family / identifier / f"{model}-seed{seed}.pt"


def save_checkpoint(path: Path, model, metadata: CheckpointMetadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"metadata": asdict(metadata), "state_dict": model.state_dict()}, temporary)
    os.replace(temporary, path)


def load_checkpoint_payload(path: Path) -> CheckpointPayload:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    metadata = CheckpointMetadata(**raw["metadata"])
    if metadata.schema_version != 1:
        raise RuntimeError(f"unsupported checkpoint schema {metadata.schema_version}")
    return CheckpointPayload(metadata, raw["state_dict"])


def restore_checkpoint(model, payload: CheckpointPayload, *, expected_data_hash: str,
                       allow_budget_bound: bool = False):
    if payload.metadata.data_hash != expected_data_hash:
        raise RuntimeError("checkpoint dataset hash does not match requested data")
    if not payload.metadata.converged and not allow_budget_bound:
        raise RuntimeError("checkpoint is budget-bound; raise the training epoch budget")
    model.load_state_dict(payload.state_dict, strict=True)
    return model
```

- [ ] **Step 4: Add payload completeness tests and implementation**

```python
# tests/test_evaluation.py
def test_phase_payload_gate_rejects_metadata_only_and_nonfinite_results():
    from spno.evaluation.payloads import require_phase_arms
    with pytest.raises(RuntimeError, match="G1"):
        require_phase_arms(6, ["G1"], {"G1": {"identifier": "shift"}})
    with pytest.raises(RuntimeError, match="finite"):
        require_phase_arms(9, ["sigma"], {
            "sigma": {"measurements": {"0.0": {"relative_to_A": {"C1": float("nan")}}}}
        })
```

Implement `require_phase_arms` with explicit per-phase required keys; recursively reject `None`, NaN, and Inf for required metric paths. Do not treat prose fields as evidence.

```python
# src/spno/evaluation/payloads.py
import math

REQUIRED_ARM_KEYS = {
    6: {"G1": ("measurements",), "G2": ("measurements",),
        "G3": ("measurements",), "G4": ("measurements",), "G5a": ("curves",),
        "G5b": ("alpha_derivative",), "G6a": ("by_model",),
        "G6b": ("by_model",), "G7": ("varying_alpha", "fixed_alpha"),
        "G9": ("reference",)},
    8: {"resolution_band_limited": ("by_model",),
        "resolution_new_high_k": ("by_model",), "robustness": ("model_errors",)},
    9: {"sigma": ("measurements",), "gamma": ("measurements",)},
}


def _has_finite_number(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return any(_has_finite_number(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_finite_number(item) for item in value)
    return False


def require_phase_arms(phase: int, selected, experiments: dict) -> None:
    requirements = REQUIRED_ARM_KEYS[phase]
    for arm in selected:
        if arm not in experiments or not experiments[arm]:
            raise RuntimeError(f"Phase {phase} arm {arm} has no measured result")
        for key in requirements[arm]:
            if key not in experiments[arm] or not _has_finite_number(experiments[arm][key]):
                raise RuntimeError(f"Phase {phase} arm {arm}.{key} needs a finite measurement")
        if phase == 6 and arm in {"G1", "G2", "G3", "G4"}:
            for shift, result in experiments[arm]["measurements"].items():
                if not result.get("by_model") or not _has_finite_number(result["by_model"]):
                    raise RuntimeError(f"Phase 6 arm {arm}/{shift} needs finite model metrics")
        if phase == 9:
            expected_models = {"B-loop", "C1", "C2", "C3"}
            for value, result in experiments[arm]["measurements"].items():
                ratios = result.get("relative_to_A", {})
                if set(ratios) != expected_models or not all(
                    isinstance(ratio, (int, float)) and math.isfinite(float(ratio))
                    for ratio in ratios.values()
                ):
                    raise RuntimeError(
                        f"Phase 9 arm {arm}/{value} needs complete finite A-relative ratios"
                    )
```

- [ ] **Step 5: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_checkpoints.py tests/test_evaluation.py -q`

Expected: all focused tests pass.

Run: `.venv/bin/python -m pytest tests -q`

Expected: 229 existing tests plus the new tests pass; five environment-dependent tests remain skipped.

- [ ] **Step 6: Commit**

```bash
git add src/spno/checkpoints.py src/spno/evaluation/payloads.py tests/test_checkpoints.py tests/test_evaluation.py
git commit -m "feat: add checkpoint and experiment payload contracts"
```

### Task 2: Repair dispersion and resolution edge cases

**Files:**
- Modify: `src/spno/evaluation/dispersion.py:239-355`
- Modify: `src/spno/evaluation/resolution.py:24-71`
- Modify: `tests/test_dispersion.py`
- Modify: `tests/test_resolution.py`

**Interfaces:**
- Produces: `_strict_alpha_grid(alphas, *, name) -> list[float]`.
- Preserves: `alpha_phase_derivative`, `omega_by_alpha_continuation`, `spectral_resample` public signatures.

- [ ] **Step 1: Write invalid-alpha tests**

```python
@pytest.mark.parametrize("alphas", [[0.7, 0.7], [0.8, 0.7], [0.7, float("nan")], [0.7, float("inf")]])
def test_alpha_estimators_require_a_finite_strictly_increasing_grid(alphas):
    domain = _domain()
    kwargs = dict(beta=BETA, amplitude=probe_amplitude(domain, MASS_RANGE),
                  potential_constant=V0, dt=DT)
    with pytest.raises(ValueError, match="strictly increasing"):
        alpha_phase_derivative(SubsteppedReference(domain, 32), domain, 8,
                               alphas=alphas, **kwargs)
```

- [ ] **Step 2: Write odd-grid resampling tests**

```python
@pytest.mark.parametrize("source_n,target_n,k", [(63, 127, 31), (63, 64, -31), (65, 63, 30)])
def test_resampling_preserves_valid_odd_grid_endpoint_modes(source_n, target_n, k):
    source = PeriodicDomain.periodic_1d(source_n)
    target = PeriodicDomain.periodic_1d(target_n)
    field = plane_wave(source, (k,), amplitude=0.7).unsqueeze(0)
    moved = spectral_resample(field, source, target)
    expected = plane_wave(target, (k,), amplitude=0.7).unsqueeze(0)
    assert float(torch.abs(moved - expected).max()) < 1e-12
```

- [ ] **Step 3: Run focused tests and verify failures**

Run: `.venv/bin/python -m pytest tests/test_dispersion.py tests/test_resolution.py -q`

Expected: invalid-alpha and odd-grid endpoint cases fail while existing even-grid Nyquist rejection continues to pass.

- [ ] **Step 4: Implement strict alpha validation**

```python
def _strict_alpha_grid(alphas, *, name: str) -> list[float]:
    values = [float(value) for value in alphas]
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} needs at least two finite, strictly increasing alphas")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(f"{name} alpha grid must be finite and strictly increasing")
    return values
```

Use this helper at the start of both estimators before calculating gaps.

- [ ] **Step 5: Implement parity-correct mode copying**

```python
def _integer_fft_modes(n: int, device) -> torch.Tensor:
    return torch.fft.fftfreq(n, d=1.0 / n, device=device).round().to(torch.int64)

source_modes = _integer_fft_modes(n_source, hat.device)
target_modes = _integer_fft_modes(n_target, hat.device)
if n_target > n_source and n_source % 2 == 0:
    nyquist_index = n_source // 2
    # retain the existing relative-energy ambiguity check
source_index = {int(k): i for i, k in enumerate(source_modes.tolist())}
for target_index, wave_number in enumerate(target_modes.tolist()):
    if wave_number in source_index:
        resampled[..., target_index] = hat[..., source_index[wave_number]]
```

Retain the `n_target/n_source` normalization and the current even-source Nyquist refusal.

- [ ] **Step 6: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_dispersion.py tests/test_resolution.py -q`

Expected: all pass.

Run: `.venv/bin/python -m pytest tests -q`

Expected: full suite passes.

- [ ] **Step 7: Commit**

```bash
git add src/spno/evaluation/dispersion.py src/spno/evaluation/resolution.py tests/test_dispersion.py tests/test_resolution.py
git commit -m "fix: validate alpha grids and resample odd grids exactly"
```

### Task 3: Make the misspecified truth exact and auditable

**Files:**
- Modify: `src/spno/solvers/perturbed.py:125-160`
- Modify: `src/spno/misspecification.py`
- Modify: `src/spno/data/datasets.py:91-190`
- Modify: `tests/test_perturbed.py`

**Interfaces:**
- Produces: `MisspecificationConfig.provenance(data_config) -> dict[str, str | int | float | bool]`.
- Extends: `generate_shard(config, split, *, potential_family="random", steps=None, reference=None, reference_metadata: Mapping[str, str | int | float | bool] | None = None)`.

- [ ] **Step 1: Add a closed-form local-flow test**

```python
@pytest.mark.parametrize("gamma", [1e-2, -1e-2, 1e-12])
def test_gain_loss_local_subflow_matches_the_analytic_solution(gamma):
    domain = PeriodicDomain.periodic_1d(16)
    field = torch.full((2, 16), 0.7 + 0.2j, dtype=torch.complex128)
    potential = torch.zeros(2, 16, dtype=torch.float64)
    alpha = torch.zeros(2, dtype=torch.float64)
    beta = torch.full((2,), 0.4, dtype=torch.float64)
    out = GainLossSplitStepNLSOperator(domain, gamma=gamma)(field, potential, alpha, beta, DT)
    rho0 = torch.abs(field) ** 2
    duration = DT if gamma == 0 else math.expm1(2 * gamma * DT) / (2 * gamma)
    gain = torch.exp(torch.tensor(gamma * DT, dtype=torch.float64))
    expected = field * gain * torch.exp(1j * beta[:, None] * rho0 * duration)
    assert float(torch.abs(out - expected).max()) < 1e-12
```

- [ ] **Step 2: Add shard provenance tests**

```python
def test_misspecified_shard_records_and_validates_reference_provenance():
    small = replace(DataConfig(), n_test=2, steps=2, grid_size=16)
    spec = MisspecificationConfig(nonlocal_sigma=0.5)
    shard = generate_shard(small, "test", reference=spec.reference(small),
                           reference_metadata=spec.provenance(small))
    assert shard.metadata["reference"]["identifier"] == spec.identifier(small)
    assert shard.metadata["reference"]["nonlocal_sigma"] == 0.5
    assert shard.metadata["reference"]["gain_loss_gamma"] == 0.0
```

- [ ] **Step 3: Verify the new tests fail**

Run: `.venv/bin/python -m pytest tests/test_perturbed.py -q`

Expected: the analytic phase and missing provenance assertions fail.

- [ ] **Step 4: Implement the exact local phase**

```python
gamma_dt = self.gamma * float(dt)
if abs(gamma_dt) < 1e-8:
    nonlinear_duration = float(dt) * (1.0 + gamma_dt + (2.0 / 3.0) * gamma_dt**2)
else:
    nonlinear_duration = math.expm1(2.0 * gamma_dt) / (2.0 * self.gamma)
local_phase = torch.exp(
    1j * (beta_grid * torch.abs(midpoint) ** 2 * nonlinear_duration
          - potential * float(dt))
)
amplified = midpoint * local_phase * math.exp(gamma_dt)
```

Keep the exact `gamma == 0.0` short-circuit so the bitwise dial-zero guarantee remains intact.

- [ ] **Step 5: Implement canonical provenance**

```python
def provenance(self, data_config: DataConfig) -> dict:
    return {
        "identifier": self.identifier(data_config),
        "reference_type": type(self.reference(data_config)).__name__,
        "nonlocal_sigma": self.nonlocal_sigma,
        "gain_loss_gamma": self.gain_loss_gamma,
        "substeps": data_config.substeps,
    }
```

Copy `dict(reference_metadata)` into `TrajectoryShard.metadata["reference"]`; do not mutate the caller's mapping. Phase 9 loaders must compare the stored mapping to `spec.provenance(data_config)` before training.

- [ ] **Step 6: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_perturbed.py tests/test_datasets.py -q`

Expected: all pass, including bitwise zero-dial and mass-rate tests.

Run: `.venv/bin/python -m pytest tests -q`

Expected: full suite passes.

- [ ] **Step 7: Commit**

```bash
git add src/spno/solvers/perturbed.py src/spno/misspecification.py src/spno/data/datasets.py tests/test_perturbed.py
git commit -m "fix: make misspecified references exact and self-describing"
```

### Task 4: Persist converged base checkpoints and complete Phase 7

**Files:**
- Modify: `scripts/run_phase23.py`
- Modify: `scripts/run_phase45.py`
- Modify: `scripts/run_phase7.py`
- Modify: `tests/test_phase_runners.py`

**Interfaces:**
- Consumes: Task 1 checkpoint contract.
- Produces: converged A/B-loop/C1/C2/C3 checkpoint files and `aggregate_pino_seed_metrics(per_seed) -> dict`.

- [ ] **Step 1: Write Phase 7 aggregation tests**

```python
def test_phase7_aggregates_test_and_horizon_100_metrics():
    from scripts.run_phase7 import aggregate_pino_seed_metrics
    per_seed = [{
        "converged": True,
        "metrics": {
            "one_step_test": 0.2,
            "rollout": {"steps": [1, 100], "mass_drift": [0.01, 0.03],
                        "energy_drift": [0.02, 0.04]},
        },
    }]
    result = aggregate_pino_seed_metrics(per_seed)
    assert result["one_step_mean"] == 0.2
    assert result["mass_drift_100_mean"] == 0.03
    assert result["energy_drift_100_mean"] == 0.04
    assert result["converged"] == [True]
```

- [ ] **Step 2: Write paired-initialization and runner smoke tests**

Extract `run_sweep(shards, data_config, args, *, save=save_run)` from `main`. In the test, use a two-trajectory in-memory shard, lambdas `[0.0, 0.1]`, one seed, one epoch, CPU, and a temporary `save` function. Assert both entries contain finite one-step/mass/energy metrics and `budget_warning` is a string rather than an exception. Add a deterministic test that factories for the same seed return identical state dictionaries before training.

- [ ] **Step 3: Verify the tests fail**

Run: `.venv/bin/python -m pytest tests/test_phase_runners.py -q`

Expected: missing aggregation helper and current `budget_warning` payload fail.

- [ ] **Step 4: Save checkpoints from Phase 2–5**

After each live `TrainHistory` is evaluated, construct `CheckpointMetadata` from the exact constructor arguments, `field_scale`, seed, train mode, `converged(history)`, and `history.best_epoch`. Save A and B-loop in Phase 2–3; save A, B-loop, C1, C2, C3 in Phase 4–5. Keep B-post represented by A plus projection rather than duplicating weights. Include the run identifier and `quick` suffix in the checkpoint path. Persist quick checkpoints with `converged=False`; only callers passing `allow_budget_bound=args.quick` may restore them.

```python
metadata = CheckpointMetadata(
    schema_version=1, model_name=name, data_hash=config_hash(data_config), seed=seed,
    train_mode=mode, field_scale=scale, trained_dt=data_config.dt,
    architecture={"kinetic_mode": kinetic, "local_mode": local},
    converged=converged(history), best_epoch=history.best_epoch,
)
path = checkpoint_path(RESULTS_ROOT, "phase45", identifier, name, seed)
save_checkpoint(path, model, metadata)
```

- [ ] **Step 5: Make Phase 7 a paired controlled sweep**

```python
scale = field_scale(shards["train"], domain)
for seed in seeds:
    torch.manual_seed(seed)
    baseline = FNOStepOperator(
        domain, modes=16, width=64, n_layers=4,
        alpha_range=data_config.alpha_range, beta_range=data_config.beta_range,
        field_scale=scale, trained_dt=data_config.dt,
    )
    initial_state = {key: value.detach().clone() for key, value in baseline.state_dict().items()}
    for physics_weight in args.lambdas:
        model = FNOStepOperator(
            domain, modes=16, width=64, n_layers=4,
            alpha_range=data_config.alpha_range, beta_range=data_config.beta_range,
            field_scale=scale, trained_dt=data_config.dt,
        )
        model.load_state_dict(initial_state)
```

Loop seed-first so every lambda for a seed starts from the same state. Store `converged(history)` alongside the live history and pass `{"converged": [seed_result["converged"] for seed_result in per_seed]}` to `budget_warning`.

- [ ] **Step 6: Implement metric aggregation**

`aggregate_pino_seed_metrics` must locate step `100` by index in `rollout["steps"]`, not assume a fixed array offset. Return mean/min/max for one-step error and mean for horizon-100 mass/energy drift, plus all convergence flags. Rename plot accessors to `one_step_mean`, `mass_drift_100_mean`, and `energy_drift_100_mean`.

- [ ] **Step 7: Verify lambda zero against `train_one_step`**

Add a tiny deterministic test that trains two identical models from the same state with `train_pino(model, train, val, data, config, physics_weight=0.0)` and `train_one_step(model, train, val, data, config)`, then asserts equal histories and bitwise-equal state dictionaries. This pins the claim that lambda zero is the Phase 2–3 objective.

- [ ] **Step 8: Run tests and quick command**

Run: `.venv/bin/python -m pytest tests/test_phase_runners.py tests/test_models.py tests/test_pde_residual.py -q`

Expected: all pass.

Run: `env MPLCONFIGDIR=/tmp/spno-mpl-cache .venv/bin/python -u scripts/run_phase7.py --quick --device cpu --seeds 0 --lambdas 0 0.1`

Expected: exits zero, writes three populated plot panels, and reports a budget-bound warning without raising.

- [ ] **Step 9: Commit**

```bash
git add scripts/run_phase23.py scripts/run_phase45.py scripts/run_phase7.py tests/test_phase_runners.py
git commit -m "fix: persist base checkpoints and complete the PINO sweep"
```

### Task 5: Train Phase 6-only arms and execute every selected Phase 6 experiment

**Files:**
- Create: `scripts/train_phase6_arms.py`
- Modify: `scripts/run_phase6.py`
- Modify: `tests/test_phase_runners.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 1 checkpoints; existing `train_one_step`, `train_multi_dt`, `SHIFT_SPECS`, `evaluate_model`, `evaluate_spectral`, `band_ratio`.
- Produces: checkpoints tagged `A-wide`, `G7-alpha-fixed/<model>`, `G6a/<model>` and complete G1–G7/G9 result mappings.

- [ ] **Step 1: Add a continuation-grid unit test**

```python
def test_phase6_continuation_grid_is_wrap_free_to_nyquist():
    from scripts.run_phase6 import alpha_continuation_grid
    values = alpha_continuation_grid(0.9, max_k=32, dt=0.01)
    assert values[0] == 0.0 and values[-1] == 0.9
    assert max(b - a for a, b in zip(values, values[1:])) * 32**2 * 0.01 < math.pi
```

- [ ] **Step 2: Add selected-arm completeness tests**

Using tiny checkpoints and generated `n_test=2, steps=2, grid_size=16` shift shards, call the extracted `run_selected_arms`. Assert:

```python
assert result["G1"]["measurements"]["G1-interpolation"]["by_model"]["A"]["one_step_test"] >= 0
assert result["G4"]["measurements"]["G4-bandwidth-12"]["by_model"]["C1"]["spectral"]
assert result["G5b"]["alpha_derivative"]["by_model"]
assert result["G6a"]["by_model"]
assert result["G7"]["varying_alpha"]["band_ratio"]
assert result["G7"]["fixed_alpha"]["band_ratio"]
assert result["G9"]["A"]["fraction_above_cutoff"]
```

- [ ] **Step 3: Implement the dense continuation path**

```python
def alpha_continuation_grid(target: float, *, max_k: int, dt: float) -> list[float]:
    maximum_gap = 0.95 * math.pi / (max_k**2 * dt)
    intervals = max(1, math.ceil(abs(target) / maximum_gap))
    return [target * index / intervals for index in range(intervals + 1)]
```

Use a path computed for `data_config.max_wave_number`; record `target`, `max_gap`, and `max_phase_increment` in G5b metadata.

- [ ] **Step 4: Implement `train_phase6_arms.py`**

The script accepts the same `--quick`, `--seeds`, `--epochs`, `--device`, and `--kinetic` arguments. It must:

1. Load production shards and measured `field_scale`.
2. Generate and save any missing G1–G4 test shard from `SHIFT_SPECS`, using `shift_identifier(spec)` and metadata containing `spec.name`, potential family, config hash, and seed. In quick mode, retain the production grid so the checkpoint domain remains compatible, replace each config with `n_test=2` and `steps=2`, and keep it under a quick-only artifact root.
3. Train A-wide (`modes=32`, width 64, four layers) with `train_one_step`.
4. Load or generate the three `G7-alpha-fixed` shards and train A, C1, and C2 with `train_one_step`.
5. Build per-dt shards for `dt=(0.005, 0.01, 0.02)`, wrap them in `MultiDtBatches`, construct C-family models with `trained_dt=None`, and call `train_multi_dt` for G6a. Record A as structurally unsupported because it ignores `dt`, rather than training a misleading arm.
6. Save every artifact with convergence, architecture, scale, dial, kinetic mode, and seed metadata.

- [ ] **Step 5: Load checkpointed models in `run_phase6.py`**

Replace `build_models` with `load_models(checkpoint_root, data_config, seeds, kinetic, *, allow_budget_bound=False)`. Load A and B-loop from the Phase 2–3 family, C1/C2/C3 from the matching Phase 4–5 one-step/K0L0 family, and A-wide from the Phase 6 arm family. Rebuild exact constructors from checkpoint metadata, restore state strictly, and return `dict[int, dict[str, StepOperator]]`. Missing or incompatible checkpoints must raise with the exact expected path and the command that creates them; budget-bound checkpoints are accepted only when `allow_budget_bound=args.quick`.

- [ ] **Step 6: Evaluate G1–G4 on real shift shards**

For each selected `ShiftSpec`, load `test.pt` by `shift_identifier(spec)` or generate a quick in-memory test shard. For every seed/model, call `evaluate_model` with checkpoints available within the shard horizon and call `evaluate_spectral` at the final shared horizon. Store each result under `experiments[arm]["measurements"][spec.name]`, aggregate mean/min/max across seeds, and include arm-specific `k_wrap`, parameter counts, and provenance.

- [ ] **Step 7: Execute G6a, G7, and G9**

- G6a loads multi-dt checkpoints and evaluates each dt-specific test shard separately.
- G6b remains a transfer test of production-trained C checkpoints under `allow_dt_transfer`; A/B stay explicitly unsupported.
- G7 loads varying-alpha production and fixed-alpha checkpoints, evaluates both on the same fixed-alpha test shard, and reports both absolute banded errors and `band_ratio`.
- G9 runs the existing cascade probe on loaded checkpoints and the reference.

- [ ] **Step 8: Gate and plot complete results**

Call `require_phase_arms(6, args.arms, payload["experiments"])` immediately before `save_run`. Extend plots so G1/G2 show error versus alpha, G3/G4 show per-arm errors, and G7 shows absolute band errors beside the normalized ratio. Every selected arm must contribute at least one finite measured series.

- [ ] **Step 9: Run focused tests and quick pipeline**

Run: `.venv/bin/python -m pytest tests/test_phase_runners.py tests/test_dispersion.py tests/test_shift.py -q`

Expected: all pass.

Run the tiny trainer, then evaluator:

```bash
env MPLCONFIGDIR=/tmp/spno-mpl-cache .venv/bin/python -u scripts/train_phase6_arms.py --quick --device cpu --seeds 0 --epochs 1 --kinetic K0
env MPLCONFIGDIR=/tmp/spno-mpl-cache .venv/bin/python -u scripts/run_phase6.py --quick --device cpu --seeds 0 --arms G5a G5b G6a G6b G7 G9 --kinetic K0
```

Expected: both exit zero and every selected arm passes the payload gate.

- [ ] **Step 10: Commit**

```bash
git add scripts/train_phase6_arms.py scripts/run_phase6.py tests/test_phase_runners.py README.md
git commit -m "fix: execute Phase 6 from trained checkpoints"
```

### Task 6: Complete Phase 8 resolution, robustness, and sample efficiency

**Files:**
- Modify: `scripts/run_phase8.py`
- Modify: `tests/test_phase_runners.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: production A/C1 checkpoints, `spectral_resample`, `rebind_domain`, `evaluate_rollout`, `train_one_step`.
- Produces: finite per-model error metrics for band-limited transfer, fine-grid new-high-k data, field noise, potential noise, and every sample fraction.

- [ ] **Step 1: Write band-limited inference tests**

Create a tiny trajectory shard, resample every trajectory frame and potential to a fine grid, rebind a checkpointed model, and assert `evaluate_resolution_transfer` returns `one_step_error` and `rollout_error` for each model. Assert the model is rebound to its original domain in a `finally` block even when evaluation raises.

- [ ] **Step 2: Write robustness and sample-efficiency aggregation tests**

Inject a deterministic evaluator/trainer into extracted helpers. Assert every field-noise and potential-noise level has model error, and every fraction has per-seed error plus mean/min/max—not only achieved corruption or pair counts.

- [ ] **Step 3: Verify tests fail**

Run: `.venv/bin/python -m pytest tests/test_phase_runners.py -q`

Expected: missing Phase 8 evaluation helpers and metric fields fail.

- [ ] **Step 4: Implement complete spectral transfer evaluation**

Resample `trajectories` and `potential` along their final spatial axis; copy alpha, beta, IDs, dt, split, and metadata into a fine-grid `TrajectoryShard`. For each loaded checkpoint model:

```python
try:
    rebind_domain(model, fine)
    metrics = evaluate_model(model, fine_shards, fine_config, train_config,
                             checkpoints=(1, 10, 20, 50, 100))
finally:
    rebind_domain(model, coarse)
```

Use `dataclasses.replace(data_config, grid_size=grid)` only for evaluation geometry; do not use its hash as a production dataset identifier.

- [ ] **Step 5: Evaluate a genuine new-high-k fine-grid dataset**

Create a separate evaluation-only config with `grid_size=args.grid`, `initial_bandwidth=min(args.grid // 2 - 1, data_config.grid_size // 2 + 8)`, `n_train=0`, `n_val=0`, `n_test=100`, and a dedicated seed. Generate its test shard on the fine grid, evaluate the rebound A/C1 checkpoints, and store model errors under `resolution_new_high_k["by_model"]`. Record both Nyquists and the occupied bandwidth. Label the section “G4 methodology on a finer grid” and never pool it with band-limited resampling.

In quick mode retain the production coarse grid and requested fine grid so checkpoint domains stay compatible, but use two test trajectories and two steps; set `initial_bandwidth=data_config.grid_size // 2 + 8` so the same semantic condition—energy above the coarse Nyquist—is exercised cheaply.

- [ ] **Step 6: Evaluate field and potential noise separately**

For each level, corrupt only the selected input while retaining clean targets. Compute one-step and rollout errors per model/seed. Preserve `achieved_relative`/`max_absolute` as calibration metadata, not the plotted outcome. The plot y-axis is relative model error.

- [ ] **Step 7: Train sample-fraction arms**

For each fraction and seed, seed before constructing A and C1, use the same `max_train_pairs=int(total_pairs*fraction)`, call `train_one_step`, evaluate on the unchanged clean test shard, and record convergence. Quick mode uses fractions `[0.1, 1.0]`, one seed, and one epoch but retains the same result schema.

- [ ] **Step 8: Gate, test, and smoke-run**

Call `require_phase_arms(8, ("resolution_band_limited", "resolution_new_high_k", "robustness"), payload)` before saving.

Run: `.venv/bin/python -m pytest tests/test_phase_runners.py tests/test_resolution.py tests/test_shift.py -q`

Expected: all pass.

Run: `env MPLCONFIGDIR=/tmp/spno-mpl-cache .venv/bin/python -u scripts/run_phase8.py --quick --device cpu --seeds 0 --epochs 1 --noise 0 0.01 --train-fractions 0.1 1.0`

Expected: exits zero with populated resolution, noise, and sample-efficiency plots.

- [ ] **Step 9: Commit**

```bash
git add scripts/run_phase8.py tests/test_phase_runners.py README.md
git commit -m "fix: measure Phase 8 transfer and robustness outcomes"
```

### Task 7: Execute Phase 9 end to end and fail closed on crossovers

**Files:**
- Modify: `scripts/run_phase9.py`
- Modify: `tests/test_phase_runners.py`
- Modify: `tests/test_perturbed.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 3 provenance, `generate_shard`, `train_one_step`, `evaluate_model`.
- Produces: `ensure_complete_ratios`, strict `locate_crossover`, measured per-dial seed results.

- [ ] **Step 1: Write fail-closed crossover tests**

```python
@pytest.mark.parametrize("by_dial", [
    {"0.0": {"relative_to_A": {}}},
    {"0.0": {"relative_to_A": {"C1": float("nan")}}},
    {"0.0": {"relative_to_A": {"C1": 0.8}}, "0.5": {"relative_to_A": {}}},
])
def test_crossover_refuses_missing_or_nonfinite_ratios(by_dial):
    from scripts.run_phase9 import locate_crossover
    with pytest.raises(RuntimeError, match="complete finite"):
        locate_crossover(by_dial, "C1")


def test_crossover_distinguishes_located_from_bounded_complete_data():
    located = {"0.0": {"relative_to_A": {"C1": 0.8}},
               "0.5": {"relative_to_A": {"C1": 1.1}}}
    bounded = {"0.0": {"relative_to_A": {"C1": 0.8}},
               "0.5": {"relative_to_A": {"C1": 0.9}}}
    assert locate_crossover(located, "C1")["crossover"] == 0.5
    assert locate_crossover(bounded, "C1")["bounded"] is not None
```

- [ ] **Step 2: Write a tiny dial-sweep integration test**

Use `DataConfig(grid_size=16, n_train=2, n_val=1, n_test=2, steps=2)`, sigma values `[0.0,0.1]`, gamma values `[0.0,0.001]`, one seed, one epoch, CPU, and a temporary artifact root. Assert every dial has all five model metrics and four finite non-A ratios before crossover calculation.

- [ ] **Step 3: Verify tests fail**

Run: `.venv/bin/python -m pytest tests/test_phase_runners.py tests/test_perturbed.py -q`

Expected: current missing-ratio behavior and absent training pipeline fail.

- [ ] **Step 4: Generate or validate each shard set**

For an exact spec, reuse production shards after verifying the base data hash. For a nonzero spec, generate train/val/test with `reference=spec.reference(data_config)` and `reference_metadata=spec.provenance(data_config)`, save under `data/<spec.identifier>/`, then load and validate every shard's `metadata["reference"]`. Quick mode uses an in-memory reduced config and never overwrites production paths.

- [ ] **Step 5: Train all five models with paired seeds**

Compute `field_scale` on each dial's training shard. Seed immediately before every constructor; use identical TrainConfig and pair budget across models. Train A, B-loop, C1, C2, and C3 with `train_one_step`; evaluate horizon-100 where available, otherwise the maximum common quick horizon. Store checkpoint metadata with the misspecification identifier and convergence flag.

- [ ] **Step 6: Aggregate finite A-relative ratios**

For every dial/model, aggregate per-seed rollout error at the selected horizon. Compute `relative_to_A[model] = model_error_mean / A_error_mean`; require a positive finite A denominator and finite positive model errors. Retain absolute mean/min/max errors so the ratio is auditable.

- [ ] **Step 7: Make crossover computation strict**

At the start of `locate_crossover`, build the ratio list with direct indexing. Raise `RuntimeError` if the dial set is empty, any model key is absent, or any value is non-finite. Only a complete list entirely below `1.0` may produce the bounded statement.

- [ ] **Step 8: Gate, test, and smoke-run**

Store each dial value under `payload["sweeps"][dial]["measurements"][str(value)]`. Call `require_phase_arms(9, ("sigma", "gamma"), payload["sweeps"])` before crossover calculation and again before saving; pass the nested `measurements` mapping into `locate_crossover`.

Run: `.venv/bin/python -m pytest tests/test_phase_runners.py tests/test_perturbed.py tests/test_datasets.py -q`

Expected: all pass.

Run: `env MPLCONFIGDIR=/tmp/spno-mpl-cache .venv/bin/python -u scripts/run_phase9.py --quick --device cpu --seeds 0 --epochs 1 --sigmas 0 0.1 --gammas 0 0.001`

Expected: exits zero, every ratio is finite, and crossover text is derived only from complete measurements.

- [ ] **Step 9: Commit**

```bash
git add scripts/run_phase9.py tests/test_phase_runners.py tests/test_perturbed.py README.md
git commit -m "fix: execute Phase 9 misspecification sweeps end to end"
```

### Task 8: Full verification and documentation handoff

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-08-18-phase6-9-review-remediation.md` (checkboxes only during execution)

**Interfaces:**
- Consumes: all preceding tasks.
- Produces: verified command sequence and clean review boundary.

- [ ] **Step 1: Run the complete unit suite**

Run: `.venv/bin/python -m pytest tests -q`

Expected: all non-environment-dependent tests pass; only the existing five dependency/platform skips remain.

- [ ] **Step 2: Run all quick pipelines in dependency order**

```bash
env MPLCONFIGDIR=/tmp/spno-mpl-cache .venv/bin/python -u scripts/run_phase23.py --quick --device cpu --seeds 0 --epochs 1
env MPLCONFIGDIR=/tmp/spno-mpl-cache .venv/bin/python -u scripts/run_phase45.py --quick --device cpu --seeds 0 --epochs 1 --mode one-step --kinetic K0 --local L0
env MPLCONFIGDIR=/tmp/spno-mpl-cache .venv/bin/python -u scripts/train_phase6_arms.py --quick --device cpu --seeds 0 --epochs 1 --kinetic K0
env MPLCONFIGDIR=/tmp/spno-mpl-cache .venv/bin/python -u scripts/run_phase6.py --quick --device cpu --seeds 0 --kinetic K0
env MPLCONFIGDIR=/tmp/spno-mpl-cache .venv/bin/python -u scripts/run_phase7.py --quick --device cpu --seeds 0 --epochs 1 --lambdas 0 0.1
env MPLCONFIGDIR=/tmp/spno-mpl-cache .venv/bin/python -u scripts/run_phase8.py --quick --device cpu --seeds 0 --epochs 1 --noise 0 0.01 --train-fractions 0.1 1
env MPLCONFIGDIR=/tmp/spno-mpl-cache .venv/bin/python -u scripts/run_phase9.py --quick --device cpu --seeds 0 --epochs 1 --sigmas 0 0.1 --gammas 0 0.001
```

Expected: every command exits zero; each runner passes its completeness gate and writes only to a `-quick` path.

- [ ] **Step 3: Inspect artifact schemas**

Run a read-only script that loads every quick `metrics.json` and asserts no selected arm contains empty mappings, no required numerical field is null/non-finite, every model table includes parameter counts and seeds, and every Phase 9 perturbed shard includes canonical provenance.

- [ ] **Step 4: Update README execution order**

Document:

1. Base checkpoint creation with Phases 2–5.
2. Phase 6-only arm training.
3. Evaluation commands for Phases 6–9.
4. Checkpoint and dataset artifact layout.
5. The rule that quick/budget-bound artifacts are not reportable.
6. Phase 7b remains optional and unimplemented.

- [ ] **Step 5: Review the final diff**

Run: `git diff --check && git status --short && git diff --stat`

Expected: no whitespace errors; only intended source, test, documentation, and plan files differ.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/superpowers/plans/2026-08-18-phase6-9-review-remediation.md
git commit -m "docs: document verified Phase 6-9 execution workflow"
```

---

## Acceptance Criteria

- `run_phase6.py --quick --arms G5a G5b` no longer raises on the continuation grid.
- No Phase 6 scientific measurement is performed on a freshly initialized model.
- G1–G4, G6a, G7, and G9 contain finite model outcomes rather than descriptions.
- Phase 7 lambda zero is deterministically equivalent to the one-step baseline objective; all three plotted metrics are finite and the budget warning cannot raise `KeyError`.
- Spectral resampling preserves valid odd-grid endpoint modes and still refuses ambiguous even-grid Nyquist upsampling.
- Gain/loss truth matches its analytic local flow while retaining bitwise zero-dial behavior.
- Every nonzero misspecification shard records and validates canonical provenance.
- Phase 8 plots model errors for resolution, noise, and sample fraction rather than transformation calibration alone.
- Phase 9 cannot emit either a located or bounded crossover from missing/non-finite ratios.
- All quick runners complete end to end and the full pytest suite passes.
- Phase 7b remains explicitly optional/planned and is not silently implemented under the one-step protocol.
