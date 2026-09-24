"""Resumable frozen-component G1–G9 study used by the Colab notebook."""
from __future__ import annotations

from dataclasses import asdict, replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from spno.artifacts import atomic_json, file_digest
from spno.checkpoints import load_checkpoint_payload
from spno.config import DataConfig, config_hash
from spno.evaluation.component_ablation import (
    GAUGED_MODEL_NAMES, MODEL_NAMES, METRICS, component_models, crossed_interval, kinetic_dispersion,
    measure_rollout, probe_cases, reference_frames, sample_probe,
)
from spno.precision import widen_to_double
from scripts.run_phase6 import _model_from_checkpoint, multi_dt_data_hash


DEFAULTS = {
    "training_seeds": [0, 1, 2], "probe_seeds": [1000, 1001, 1002, 1003, 1004],
    "batch": 16, "bandwidths": [4, 6, 8, 10, 12, 16, 20, 24, 28, 32],
    "short_steps": 200, "long_steps": 2000, "stride": 50,
    "reference_substeps": 32, "reference_tolerance": 1e-4,
    "spatial_samples": 2, "spatial_tolerance": 1e-3,
    "tail_threshold": 1e-6, "device": "cpu", "threads": 2,
    "allow_budget_bound": True, "bootstrap_draws": 2000,
    "case_names": None, "smoke": False, "gauge": False,
}

#: Paired contrasts, (model, baseline). The first is the original hybrid-vs-C1 question;
#: the others are the reciprocal swap. Gauged pairs are added when the run has them.
PAIRS = (("exactK_learnedL", "C1"), ("learnedK_exactL", "C1"),
         ("exactK_learnedL", "learnedK_exactL"))
GAUGED_PAIRS = (("exactK_learnedL_g", "C1"), ("learnedK_exactL_g", "C1"),
                ("exactK_learnedL_g", "learnedK_exactL_g"))

#: Exp 1 selection for new rollouts: the spectral axis (stresses K), beta/alpha and V
#: shifts (stress L), and the in-distribution control.
RECIPROCAL_CASES = ("G1-interpolation", "G2-extrapolation", "G3-potential-strong",
                    "G3-potential-short", "G4-bandwidth-4", "G4-bandwidth-8",
                    "G4-bandwidth-12", "G4-bandwidth-16", "G4-bandwidth-24")


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_unit(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf8") as stream:
        json.dump(clean_json(value), stream, allow_nan=False)
    os.replace(temporary, path)


def read_unit(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        return json.load(stream)


def source_digest():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src").rglob("*.py")) + sorted((root / "scripts").glob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_c1_cohorts(source_root, data, seeds, *, allow_budget_bound):
    # This study needs C1 only; do not require unrelated FNO weights or train.pt.
    paths = sorted((Path(source_root) / "checkpoints/phase6").rglob("C1-seed*.pt"))
    inventory = []
    for path in paths:
        payload = load_checkpoint_payload(path)
        meta = payload.metadata
        inventory.append({"name": meta.model_name, "seed": meta.seed, "path": str(path),
                          "metadata": asdict(meta), "converged": meta.converged,
                          "sha256": file_digest(path), "quick": "-quick" in str(path.relative_to(source_root))})
    base_hashes = {r["metadata"]["data_hash"] for r in inventory if r["name"] == "C1"}
    if data is None:
        candidates = [DataConfig(), replace(DataConfig(), n_train=2, n_val=2, n_test=2, steps=2)]
        matches = [c for c in candidates if base_hashes == {config_hash(c)}]
        if len(matches) != 1:
            raise ValueError("Provide SOURCE_CONFIG with the original training DataConfig")
        data = matches[0]
    models, provenance = {}, []
    for cohort in ("base", "G6a", "G7-alpha-fixed"):
        models[cohort] = {}
        name = "C1" if cohort == "base" else f"{cohort}/C1"
        for seed in seeds:
            rows = [r for r in inventory if r["name"] == name and r["seed"] == seed]
            if len(rows) != 1:
                raise ValueError(f"Need exactly one {name}, training seed {seed}; found {len(rows)}")
            row = rows[0]
            expected = config_hash(data)
            if cohort == "G6a":
                expected = multi_dt_data_hash(data, quick=row["quick"])
            elif cohort == "G7-alpha-fixed":
                expected = config_hash(replace(data, alpha_range=(.9, .9), seed=120))
            if row["metadata"]["data_hash"] != expected:
                raise ValueError(f"Data identity mismatch for {name}, seed {seed}")
            if not row["converged"] and not allow_budget_bound:
                raise ValueError(f"{name}, seed {seed} is budget-bound; explicit exploratory opt-in required")
            payload = load_checkpoint_payload(Path(row["path"]))
            model = _model_from_checkpoint(payload, data, expected_name=name)
            model.load_state_dict(payload.state_dict, strict=True)
            models[cohort][seed] = widen_to_double(model, device="cpu").eval()
            provenance.append({k: row[k] for k in ("name", "seed", "sha256", "metadata", "quick")})
    return data, models, provenance


def load_c1g_cohort(workflow_root, data, seeds):
    """C1g lambda=0 checkpoints trained by the Phase 7 workflow, as a base cohort."""
    from spno.workflow import stable_hash
    records = {}
    for path in sorted(Path(workflow_root).glob("experiments/*/record.json")):
        record = json.loads(path.read_text())
        spec = record["spec"]
        if spec["name"] != "C1g" or spec["physics_weight"] != 0 or spec["seed"] not in seeds:
            continue
        if stable_hash(spec) != path.parent.name:
            raise RuntimeError(f"Record identity mismatch: {path}")
        if spec["data_hash"] != config_hash(data):
            raise ValueError(f"C1g seed {spec['seed']} was trained on other data")
        if spec["seed"] in records:
            raise ValueError(f"Ambiguous C1g checkpoints for seed {spec['seed']}; select one workflow")
        records[spec["seed"]] = (path.parent / "model.pt", record)
    missing = sorted(set(seeds) - set(records))
    if missing:
        raise ValueError(f"C1g checkpoints missing for seeds {missing}; train them in 00_training")
    models, provenance = {}, []
    for seed, (path, record) in sorted(records.items()):
        if file_digest(path) != record["sha256"]:
            raise RuntimeError(f"Checkpoint corrupt: {path}")
        payload = load_checkpoint_payload(path)
        model = _model_from_checkpoint(payload, data, expected_name="C1g")
        model.load_state_dict(payload.state_dict, strict=True)
        models[seed] = widen_to_double(model, device="cpu").eval()
        provenance.append({"name": "C1g", "seed": seed, "sha256": record["sha256"],
                           "metadata": asdict(payload.metadata), "quick": bool(record["spec"].get("quick"))})
    return {"base": models}, provenance


def prolong(field):
    """Fourier interpolate a complex periodic field from N to 2N."""
    n = field.shape[-1]
    spectrum = torch.zeros(*field.shape[:-1], 2 * n, dtype=torch.complex128, device=field.device)
    modes = torch.fft.fftfreq(n, d=1/n, device=field.device).long()
    spectrum[..., modes % (2*n)] = 2 * torch.fft.fft(field)
    result = torch.fft.ifft(spectrum)
    return result if field.is_complex() else result.real


@torch.no_grad()
def reference_bundle(inputs, case, options, steps, stride):
    cfg = case.config
    substeps = options["reference_substeps"]
    coarse = reference_frames(inputs, cfg, steps=steps, stride=stride, substeps=substeps)
    fine = reference_frames(inputs, cfg, steps=steps, stride=stride, substeps=2*substeps)
    checks = []
    for step in fine:
        error = ((coarse[step] - fine[step]).norm(dim=-1) /
                 fine[step].norm(dim=-1).clamp_min(1e-30))
        checks.append({"step": step, "state_error": error.cpu().tolist()})
    ncheck = min(options["spatial_samples"], len(inputs[0]))
    spatial = []
    if ncheck:
        high_inputs = (prolong(inputs[0][:ncheck]), prolong(inputs[1][:ncheck]),
                       inputs[2][:ncheck], inputs[3][:ncheck])
        high = reference_frames(high_inputs, replace(cfg, grid_size=cfg.grid_size*2),
                                steps=steps, stride=stride, substeps=2*substeps)
        for step in high:
            error = ((high[step][..., ::2] - fine[step][:ncheck]).norm(dim=-1) /
                     fine[step][:ncheck].norm(dim=-1).clamp_min(1e-30))
            spatial.append({"step": step, "state_error": error.cpu().tolist()})
    temporal_max = max(max(r["state_error"]) for r in checks)
    spatial_max = max((max(r["state_error"]) for r in spatial), default=None)
    return fine, {"substeps": [substeps, 2*substeps], "time_refinement": checks,
                  "time_refinement_max": temporal_max,
                  "time_refinement_pass": temporal_max <= options["reference_tolerance"],
                  "spatial_samples": ncheck, "space_refinement": spatial,
                  "space_refinement_max": spatial_max,
                  "space_refinement_pass": None if spatial_max is None else spatial_max <= options["spatial_tolerance"],
                  "scope": "Same-grid semi-discrete NLS; 2N check is a subset diagnostic, not a convergence proof."}


def validate_options(options, data):
    for key in ("training_seeds", "probe_seeds", "bandwidths"):
        if not options[key] or len(set(options[key])) != len(options[key]):
            raise ValueError(f"{key} must be nonempty and contain no duplicates")
    for key in ("batch", "short_steps", "long_steps", "stride", "reference_substeps", "bootstrap_draws"):
        if not isinstance(options[key], int) or options[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if options["long_steps"] < options["short_steps"]:
        raise ValueError("long_steps must be at least short_steps")
    if options["device"] not in ("cpu", "cuda"):
        raise ValueError("Use cpu or cuda; MPS cannot evaluate float64 invariants")
    if options["spatial_samples"] < 0:
        raise ValueError("spatial_samples must be nonnegative")
    if not options["smoke"] and (len(options["probe_seeds"]) < 3 or len(options["training_seeds"]) < 3):
        raise ValueError("Use at least 3 seeds on each axis, or mark smoke=True")


def run_study(source_root, output_root, *, data=None, options=None, cohorts=None):
    """``cohorts=(models, provenance)`` substitutes other C1-family checkpoints (C1g).

    Cases that need a cohort the substitute lacks (G6a, G7) keep only their base rows.
    """
    options = {**DEFAULTS, **(options or {})}
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    source_root, output_root = Path(source_root), Path(output_root)
    if output_root.resolve().is_relative_to(source_root.resolve()):
        raise ValueError("Keep outputs separate from source checkpoints")
    torch.set_num_threads(options["threads"])
    if cohorts is None:
        data, cohorts, provenance = load_c1_cohorts(source_root, data, options["training_seeds"],
                                                 allow_budget_bound=options["allow_budget_bound"])
    else:
        if data is None:
            raise ValueError("Substituted cohorts need their training DataConfig")
        cohorts, provenance = cohorts
        if any(set(models) != set(options["training_seeds"]) for models in cohorts.values()):
            raise ValueError("Substituted cohorts must cover exactly the training seeds")
    validate_options(options, data)
    model_names = MODEL_NAMES + (GAUGED_MODEL_NAMES if options["gauge"] else ())
    cases = [replace(c, cohorts=tuple(k for k in c.cohorts if k in cohorts))
             for c in probe_cases(data, options["bandwidths"])]
    cases = [c for c in cases if c.cohorts]
    if options["case_names"] is not None:
        known = {c.name for c in cases}
        if set(options["case_names"]) - known:
            raise ValueError("Unknown case_names")
        cases = [c for c in cases if c.name in options["case_names"]]
    identity = {"schema": 1, "data": asdict(data), "options": options,
                "checkpoints": provenance, "source_sha256": source_digest(),
                "torch": str(torch.__version__), "numpy": str(np.__version__),
                "python": platform.python_version(),
                "hardware": torch.cuda.get_device_name(0) if options["device"] == "cuda" else platform.machine()}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    root = output_root / run_id
    root.mkdir(parents=True, exist_ok=True)
    exploratory = any(not p["metadata"]["converged"] or p["quick"] for p in provenance)
    manifest = {**identity, "run_id": run_id, "exploratory": exploratory,
                "complete": False, "cases": [asdict(c) for c in cases], "models": list(model_names),
                "checkpoint_family": sorted({p["name"].rsplit("/", 1)[-1] for p in provenance}),
                "interpretation": "Frozen checkpoint interventions; no hybrid retraining or test-set tuning.",
                "G8": "New long-rollout/conservation extension authorized by user."}
    atomic_json(root / "manifest.json", manifest)
    print(f"Run {run_id}: {len(cases)} cases × {len(options['probe_seeds'])} probe batches; "
          f"{len(options['training_seeds'])} training seeds; exploratory={exploratory}", flush=True)
    dispersion_path = root / "dispersion.json.gz"
    if not dispersion_path.exists():
        save_unit(dispersion_path, {cohort: {str(seed): kinetic_dispersion(model,
                                          replace(data, alpha_range=(.9, .9)) if cohort == "G7-alpha-fixed" else data)
                                          for seed, model in models.items()}
                                    for cohort, models in cohorts.items()})
    prepared = {cohort: {seed: {name: model.to(options["device"])
                                for name, model in component_models(c1, gauge=options["gauge"]).items()}
                        for seed, c1 in models.items()} for cohort, models in cohorts.items()}
    for case in cases:
        # Fix physical T across G6's different dt values.
        duration = data.dt * options["long_steps" if case.long else "short_steps"]
        steps = max(1, round(duration / case.config.dt))
        stride = max(1, round(options["stride"] * data.dt / case.config.dt))
        for probe_seed in options["probe_seeds"]:
            folder = root / "cases" / case.name / f"probe-{probe_seed}"
            paths = {f"{cohort}-{seed}": folder / f"{cohort}-{seed}.json.gz"
                     for cohort in case.cohorts for seed in options["training_seeds"]}
            reference_path = folder / "reference.json.gz"
            if reference_path.exists() and all(p.exists() for p in paths.values()):
                # Read gzip CRC + embedded identity before trusting a completed unit.
                for path in [reference_path, *paths.values()]:
                    saved = read_unit(path)
                    if saved["run_id"] != run_id:
                        raise RuntimeError(f"Result identity mismatch: {path}")
                print(f"{case.name} / probe {probe_seed}: already done", flush=True)
                continue
            started = time.monotonic()
            inputs, input_hash = sample_probe(case, probe_seed, options["batch"])
            inputs = tuple(x.to(options["device"]) for x in inputs)
            truth, convergence = reference_bundle(inputs, case, options, steps, stride)
            common = {"run_id": run_id, "case": case.name, "probe_seed": probe_seed,
                      "input_sha256": input_hash, "dt": case.config.dt,
                      "bandwidth": case.config.initial_bandwidth,
                      "alpha": inputs[2].cpu().tolist(), "beta": inputs[3].cpu().tolist()}
            # An indexed map returns the saved fine solution without recomputing it.
            class SavedReference:
                def __init__(self): self.step = 0
                def __call__(self, state, *args):
                    self.step += 1
                    return truth.get(self.step, state)
            reference = measure_rollout(SavedReference(), inputs, case.config, truth,
                                        training_bandwidth=data.initial_bandwidth)
            tail_max = max(max(r["nyquist_tail"]) for r in reference["records"])
            convergence["tail_max"] = tail_max
            convergence["tail_pass"] = tail_max <= options["tail_threshold"]
            save_unit(reference_path, {**common, "convergence": convergence, "reference": reference})
            for cohort in case.cohorts:
                for seed in options["training_seeds"]:
                    path = paths[f"{cohort}-{seed}"]
                    if path.exists():
                        continue
                    measurements = {name: measure_rollout(model, inputs, case.config, truth,
                                                          training_bandwidth=data.initial_bandwidth)
                                    for name, model in prepared[cohort][seed].items()}
                    save_unit(path, {**common, "cohort": cohort, "training_seed": seed,
                                     "by_model": measurements})
            print(f"{case.name} / probe {probe_seed}: {time.monotonic()-started:.1f}s; "
                  f"reference Δt error={convergence['time_refinement_max']:.2e}, "
                  f"2N error={convergence['space_refinement_max']}", flush=True)
    summary = summarize(root, manifest)
    atomic_json(root / "summary.json", clean_json(summary))
    manifest["complete"] = True
    atomic_json(root / "manifest.json", manifest)
    return root


def resummarize(run_root):
    """Rebuild summary.json from saved units without touching run identity.

    ``run_study`` would hash the current source into a new run_id and start over; this
    reads the existing manifest instead. A pre-existing summary is kept as summary.v1.json.
    """
    root = Path(run_root)
    manifest = json.loads((root / "manifest.json").read_text())
    previous, backup = root / "summary.json", root / "summary.v1.json"
    if previous.exists() and not backup.exists():
        atomic_json(backup, json.loads(previous.read_text()))
    summary = summarize(root, manifest)
    atomic_json(previous, clean_json(summary))
    return summary


def summarize(root, manifest):
    options = manifest["options"]
    models = tuple(manifest.get("models", MODEL_NAMES))  # pre-gauge manifests: four models
    pairs = [p for p in PAIRS + GAUGED_PAIRS if set(p) <= set(models)]
    rows, paired, checks = [], [], []
    for case in manifest["cases"]:
        name = case["name"]
        for probe in options["probe_seeds"]:
            ref = read_unit(root / "cases" / name / f"probe-{probe}" / "reference.json.gz")
            checks.append({"case": name, "probe_seed": probe, **ref["convergence"]})
        for cohort in case["cohorts"]:
            units = [[read_unit(root / "cases" / name / f"probe-{p}" / f"{cohort}-{s}.json.gz")
                      for p in options["probe_seeds"]] for s in options["training_seeds"]]
            for model in models:
                for metric in METRICS:
                    for endpoint in (1, -1):
                        def values(u, m):
                            rs = u["by_model"][m]["records"]
                            return (next(r for r in rs if r["step"] == 1) if endpoint == 1 else rs[-1])[metric]
                        array = np.asarray([[values(u, model) for u in group] for group in units], dtype=float)
                        failed = sum(v is not None for group in units for u in group
                                     for v in u["by_model"][model]["failed_at"])
                        rows.append({"case": name, "cohort": cohort, "model": model, "metric": metric,
                                     "endpoint": "one_step" if endpoint == 1 else "final",
                                     "mean": float(array.mean()), "min": float(array.min()),
                                     "max": float(array.max()), "failed_rollouts": failed,
                                     "per_training_seed": array.mean(axis=(1, 2)).tolist(),
                                     "per_probe_seed": array.mean(axis=(0, 2)).tolist()})
                        for first, second in pairs:
                            if model != first:
                                continue
                            baseline = np.asarray([[values(u, second) for u in group] for group in units], dtype=float)
                            difference = crossed_interval(array - baseline, draws=options["bootstrap_draws"])
                            ratio = crossed_interval(np.log((array + 1e-15) / (baseline + 1e-15)),
                                                     draws=options["bootstrap_draws"])
                            row = {"case": name, "cohort": cohort, "metric": metric,
                                   "endpoint": "one_step" if endpoint == 1 else "final",
                                   "model": first, "baseline": second, "difference": difference,
                                   "ratio": {k: math_exp(ratio[k]) for k in ("mean", "low", "high")},
                                   "status": ratio["status"]}
                            if (first, second) == PAIRS[0]:  # historical keys, read by notebook 11
                                row["hybrid_minus_C1"], row["hybrid_over_C1"] = difference, row["ratio"]
                            paired.append(row)
    return {"rows": rows, "paired": paired, "reference_checks": checks,
            "note": "Crossed training/probe bootstrap, ICs resampled within probes; paired 95% intervals are descriptive and not multiplicity-adjusted. Ratios are geometric means with a 1e-15 numerical floor. Exact comparator repetitions across training seeds are duplicates, not additional independent samples."}


def math_exp(value):
    return None if value is None else float(np.exp(value))
