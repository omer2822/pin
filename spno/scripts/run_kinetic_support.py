"""Kinetic support ladder: does C1's learned dispersion follow the training spectrum?

Phase 6 C1 (K0, training bandwidth 8) reproduces -alpha k^2 up to |k| ~ 9 and is flat
beyond. This study trains C1 variants that differ in ONE factor -- the training-data
bandwidth, or the kinetic rung K0/K1/K2 -- with the Phase 6 base protocol, and measures
where each one leaves the true dispersion. Used by notebook 13.

The knee is read from the full plane-wave map residual |e^{-i w_model dt} - e^{-i w dt}|,
which is gauge-free (kappa and nu offsets cancel) and branch-free (it only sees the
one-step multiplier, exactly what the data constrains). The centered kinetic rate is kept
for figures only: above k_wrap the data fixes kappa just modulo 2 pi / dt.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch

from spno.artifacts import atomic_json
from spno.checkpoints import load_checkpoint_payload, save_checkpoint
from spno.config import DataConfig, config_hash
from spno.data.datasets import OneStepBatches, TrajectoryShard, generate_shard, shard_paths
from spno.equations.nls import wrap_wavenumber
from spno.evaluation.component_ablation import (
    ProbeCase, measure_rollout, reference_frames, sample_probe,
)
from spno.evaluation.dispersion import alpha_phase_derivative, dispersion_curve, probe_amplitude
from spno.experiments import converged, pick_device
from spno.models.split_learned import DensityPhaseSplitStep
from spno.precision import widen_to_double
from spno.train import TrainConfig, TrainHistory, evaluate_one_step, field_scale, train_one_step
from spno.training_progress import TrainingProgress
from scripts.run_hybrid_ablation import clean_json, prolong, source_digest
from scripts.run_phase6 import ALPHA_GRID, PROBE_ALPHA, PROBE_BETA, PROBE_V0, _model_from_checkpoint, run_gates
from scripts.train_phase6_arms import _metadata, _structured_architecture


DEFAULT_CELLS = ([{"kinetic": "K0", "bandwidth": b} for b in (8, 10, 12, 16)]
                 + [{"kinetic": "K1", "bandwidth": 8}, {"kinetic": "K2", "bandwidth": 8}])

DEFAULTS = {
    "cells": DEFAULT_CELLS, "seeds": [0, 1, 2],
    # Phase 6 base protocol (train_phase6_arms.py): 40 epochs, batch 256, lr 1e-3, patience 8.
    "epochs": 40, "batch_size": 256, "learning_rate": 1e-3, "patience": 8,
    "device": "auto", "threads": 2,
    "knee_thresholds": [0.01, 0.05, 0.2], "knee_alphas": [0.9, 0.7], "knee_beta": PROBE_BETA,
    "probe_k": [8, 10, 12, 14, 16, 18, 20, 24, 32],
    "support_thresholds": [1e-6, 1e-8],
    "rollout_bandwidths": [8, 12, 16], "rollout_steps": 200, "probe_seed": 1000, "probe_batch": 16,
    "reference_substeps": 64,
    "check_trajectories": 4, "check_steps": 200, "tail_threshold": 1e-6, "spatial_tolerance": 1e-3,
    "smoke": False,
}
#: Settings that change what is computed; device/threads only change where.
IDENTITY_KEYS = tuple(k for k in DEFAULTS if k not in ("device", "threads"))
PRIMARY = {"threshold": 0.05, "alpha": 0.9}
PHASE6_BASE_HASH = "bd4e108527"


def cell_name(cell):
    return f"{cell['kinetic']}-bw{cell['bandwidth']}"


def cell_config(base, cell):
    return replace(base, initial_bandwidth=int(cell["bandwidth"]))


def validate(options, base):
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    names = [cell_name(c) for c in options["cells"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("cells must be nonempty and unique")
    for cell in options["cells"]:
        if cell["kinetic"] not in ("K0", "K1", "K2"):
            raise ValueError(f"unknown kinetic rung {cell['kinetic']!r}")
        if not 1 <= int(cell["bandwidth"]) <= base.grid_size // 2:
            raise ValueError("bandwidth must lie in the resolvable spectrum")
    if not options["seeds"] or len(set(options["seeds"])) != len(options["seeds"]):
        raise ValueError("seeds must be nonempty and unique")
    if not options["smoke"] and len(options["seeds"]) < 3:
        raise ValueError("Use at least 3 seeds, or mark smoke=True")
    if options["device"] == "mps":
        raise ValueError("MPS has no float64; use cpu or cuda")


def prepare(output_root, data_root, *, base=None, options=None):
    """Resolve settings and run identity; nothing is generated or trained yet."""
    options = {**DEFAULTS, **(options or {})}
    base = DataConfig() if base is None else base
    validate(options, base)
    identity = {"schema": 1, "base": asdict(base), "options": {k: options[k] for k in IDENTITY_KEYS}}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    root = Path(output_root) / run_id
    root.mkdir(parents=True, exist_ok=True)
    device = pick_device(options["device"])
    if device == "mps":
        device = "cpu"
    cells = [{**c, "name": cell_name(c), "config": cell_config(base, c)} for c in options["cells"]]
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update(identity, run_id=run_id, source_sha256=source_digest(), torch=str(torch.__version__),
                    python=platform.python_version(), device=device,
                    cells={c["name"]: {"kinetic": c["kinetic"], "bandwidth": c["bandwidth"],
                                       "data_hash": config_hash(c["config"])} for c in cells},
                    interpretation="C1 retrained with the Phase 6 base protocol; one factor varied per cell. "
                                   "40-epoch budget-bound results are exploratory.")
    manifest.setdefault("complete", False)
    atomic_json(manifest_path, manifest)
    print(f"Run {run_id}: {len(cells)} cells x {len(options['seeds'])} seeds on {device}", flush=True)
    return {"root": root, "run_id": run_id, "options": options, "base": base,
            "data_root": Path(data_root), "cells": cells, "device": device}


# ---------------------------------------------------------------- data

def shard_digest(shard):
    digest = hashlib.sha256()
    for tensor in (shard.trajectories, shard.potential, shard.alpha, shard.beta):
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def ensure_shards(cfg, data_root, expected=None):
    """Generate (deterministically) or load train/val/test.

    A regenerated shard is compared with the recorded digest but never rejected: a new
    Colab runtime may round FFTs differently (~1e-12), which changes the digest and not
    the science. The config hash is what identifies the data.
    """
    identifier = config_hash(cfg)
    shards, digests = {}, {}
    for split, path in shard_paths(Path(data_root), identifier).items():
        if not path.exists():
            print(f"  generating {split} for {identifier} (bandwidth {cfg.initial_bandwidth})", flush=True)
            generate_shard(cfg, split).save(path)
        shards[split] = TrajectoryShard.load(path)
        digests[split] = shard_digest(shards[split])
        if expected and expected.get(split) not in (None, digests[split]):
            print(f"  note: regenerated {split} for {identifier} is not bitwise identical to the first "
                  "generation (runtime rounding); continuing", flush=True)
    return shards, digests


def power_by_k(field):
    """Mean normalized power per |k| over every trajectory and frame; index = |k|."""
    n = field.shape[-1]
    power = torch.fft.fft(field).abs().square()
    power = power / power.sum(-1, keepdim=True).clamp_min(1e-300)
    power = power.reshape(-1, n).mean(0)
    k = torch.fft.fftfreq(n, d=1 / n).abs().long()
    folded = torch.zeros(n // 2 + 1, dtype=power.dtype).index_add_(0, k, power)
    return folded.tolist()


def support_edge(spectrum, threshold):
    above = [k for k, value in enumerate(spectrum) if value > threshold]
    return max(above) if above else 0


@torch.no_grad()
def resolution_check(cfg, shard, options):
    """Tail below 0.75 Nyquist and an N-vs-2N (with 2x substeps) spot check."""
    n = cfg.grid_size
    k = torch.fft.fftfreq(n, d=1 / n).abs()
    power = torch.fft.fft(shard.trajectories).abs().square()
    tail = ((power * (k > .75 * (n // 2))).sum(-1) / power.sum(-1).clamp_min(1e-300)).max().item()
    m = min(options["check_trajectories"], shard.n_trajectories)
    steps = min(options["check_steps"], shard.n_frames - 1)
    fine = replace(cfg, grid_size=2 * n)
    inputs = (prolong(shard.trajectories[:m, 0]), prolong(shard.potential[:m]),
              shard.alpha[:m], shard.beta[:m])
    high = reference_frames(inputs, fine, steps=steps, stride=steps, substeps=2 * cfg.substeps)[steps]
    stored = shard.trajectories[:m, steps]
    spatial = ((high[..., ::2] - stored).norm(dim=-1) / stored.norm(dim=-1)).max().item()
    return {"tail_max": tail, "tail_pass": tail <= options["tail_threshold"],
            "refined_error_max": spatial, "refined_pass": spatial <= options["spatial_tolerance"],
            "check_trajectories": m, "check_steps": steps,
            "scope": "stored trajectory vs 2N grid with 2x substeps; a subset diagnostic, not a proof"}


def reproduce_phase6(shards, source_root):
    """Bitwise tensor comparison with the Windows Phase 6 val/test shards, if present."""
    if source_root is None:
        return {"status": "skipped", "reason": "no Phase 6 artifacts"}
    paths = shard_paths(Path(source_root) / "data", PHASE6_BASE_HASH)
    result = {}
    for split in ("val", "test"):
        if not paths[split].exists():
            result[split] = None
            continue
        saved = TrajectoryShard.load(paths[split])
        ours = shards[split]
        result[split] = max(float(((getattr(saved, f) - getattr(ours, f)).abs().max()
                                   / getattr(saved, f).abs().max().clamp_min(1e-300)))
                            for f in ("trajectories", "potential", "alpha", "beta"))
    found = [v for v in result.values() if v is not None]
    status = "skipped" if not found else "PASS" if max(found) <= 1e-10 else "CHECK"
    return {"status": status, "max_relative_difference": result,
            "rule": "PASS if every tensor agrees to 1e-10 relative (cross-machine float64 rounding)"}


def run_data(study, *, source_root=None):
    """Generate every cell's data, check resolution, measure the spectral support."""
    root, options = study["root"], study["options"]
    path = root / "data-checks.json"
    record = json.loads(path.read_text()) if path.exists() else {}
    for cell in study["cells"]:
        cfg = cell["config"]
        identifier = config_hash(cfg)
        entry = record.get(identifier, {})
        shards, digests = ensure_shards(cfg, study["data_root"], entry.get("digests"))
        if "support" not in entry:
            spectrum = power_by_k(shards["train"].trajectories)
            entry = {"bandwidth": cfg.initial_bandwidth, "digests": digests, "spectrum": spectrum,
                     "support": {str(t): support_edge(spectrum, t) for t in options["support_thresholds"]},
                     "resolution": resolution_check(cfg, shards["train"], options),
                     "field_scale": field_scale(shards["train"], cfg.domain)}
            if identifier == PHASE6_BASE_HASH:
                entry["phase6_reproduction"] = reproduce_phase6(shards, source_root)
            record[identifier] = entry
            atomic_json(path, clean_json(record))
        res = entry["resolution"]
        print(f"{cell['name']:8s} data {identifier}: support edge "
              + ", ".join(f"{t}: k={v}" for t, v in entry["support"].items())
              + f" | tail {res['tail_max']:.1e} ({'ok' if res['tail_pass'] else 'FLAG'})"
              + f" | 2N error {res['refined_error_max']:.1e} ({'ok' if res['refined_pass'] else 'FLAG'})"
              + (f" | Phase 6 data reproduction: {entry['phase6_reproduction']['status']}"
                 if "phase6_reproduction" in entry else ""), flush=True)
    return record


# ---------------------------------------------------------------- training

def build_model(cell, seed):
    cfg = cell["config"]
    # Seed immediately before construction, as Phase 6 did: K0 inits match the Windows C1.
    torch.manual_seed(seed)
    return DensityPhaseSplitStep(cfg.domain, kinetic_mode=cell["kinetic"], local_mode="L0",
                                 width=32, trained_dt=cfg.dt)


def budget(history):
    val = history["val_loss"]
    last5 = (val[-6] - val[-1]) / val[-6] if len(val) >= 6 and val[-6] > 0 else None
    return {"epochs_run": len(val), "best_epoch": history["best_epoch"], "best_val": history["best_val"],
            "converged_gate": converged(TrainHistory(**history)), "last5_improvement": last5,
            "seconds": history["seconds"]}


def train_all(study):
    root, options = study["root"], study["options"]
    data = json.loads((root / "data-checks.json").read_text())
    torch.set_num_threads(options["threads"])
    for cell in study["cells"]:
        cfg = cell["config"]
        shards = None
        for seed in options["seeds"]:
            checkpoint = root / "checkpoints" / cell["name"] / f"C1-seed{seed}.pt"
            history_path = checkpoint.with_suffix(".history.json")
            if checkpoint.exists() and history_path.exists():
                print(f"{cell['name']} seed {seed}: already trained", flush=True)
                continue
            if shards is None:
                shards, _ = ensure_shards(cfg, study["data_root"], data[config_hash(cfg)]["digests"])
            model = build_model(cell, seed)
            train_config = TrainConfig(epochs=options["epochs"], batch_size=options["batch_size"],
                                       learning_rate=options["learning_rate"], patience=options["patience"],
                                       seed=seed, device=study["device"], log_every=5)
            progress = TrainingProgress(root / "progress" / cell["name"] / f"seed{seed}.pt",
                                        f"{study['run_id']}/{cell['name']}/{seed}")
            print(f"{cell['name']} seed {seed}: training ({options['epochs']} epochs max)", flush=True)
            history = train_one_step(model, shards["train"], shards["val"], cfg, train_config,
                                     progress=progress)
            save_checkpoint(checkpoint, model.cpu(), _metadata(
                name="C1", data_hash=config_hash(cfg), seed=seed, train_mode="one-step",
                scale=data[config_hash(cfg)]["field_scale"], trained_dt=cfg.dt,
                architecture=_structured_architecture("C1", cfg, cell["kinetic"], "support-ladder"),
                history=history, quick=options["smoke"]))
            atomic_json(history_path, clean_json(history.as_dict()))
            info = budget(history.as_dict())
            print(f"  best val {info['best_val']:.3e} at epoch {info['best_epoch']} of {info['epochs_run']}"
                  f" ({history.seconds:.0f}s)", flush=True)


def load_trained(study, cell, seed):
    checkpoint = study["root"] / "checkpoints" / cell["name"] / f"C1-seed{seed}.pt"
    payload = load_checkpoint_payload(checkpoint)
    model = _model_from_checkpoint(payload, cell["config"], expected_name="C1")
    model.load_state_dict(payload.state_dict, strict=True)
    history = json.loads(checkpoint.with_suffix(".history.json").read_text())
    return model.eval(), history


# ---------------------------------------------------------------- measurements

@torch.no_grad()
def plane_wave_rates(model, cfg, alpha, beta):
    """(k, model frequency, true frequency, centered kappa) on k = 0..N/2 for a plane wave."""
    model = widen_to_double(model, device="cpu").eval()
    n = cfg.grid_size
    amplitude = probe_amplitude(cfg.domain, cfg.mass_range)
    parameters = torch.tensor([[alpha, beta]], dtype=torch.float64)
    kappa = model.kinetic(parameters)[0, : n // 2 + 1]
    rho = torch.full((1, n), amplitude ** 2, dtype=torch.float64)
    nu = model.local(rho, torch.zeros_like(rho), parameters[:, 0], parameters[:, 1])[0, 0]
    k = torch.arange(n // 2 + 1, dtype=torch.float64)
    return k, -(kappa + nu), alpha * k.square() - beta * amplitude ** 2, kappa - kappa[0]


def map_residual(model, cfg, alpha, beta):
    k, omega, truth, _ = plane_wave_rates(model, cfg, alpha, beta)
    return (torch.exp(-1j * cfg.dt * omega) - torch.exp(-1j * cfg.dt * truth)).abs().tolist()


def knee(residual, threshold):
    """Smallest k >= 1 whose one-step map residual exceeds ``threshold``; None if none does."""
    return next((k for k, value in enumerate(residual) if k and value > threshold), None)


def as_knee(value, grid_size):
    """A missing knee means the model is accurate to Nyquist: rank it past the grid."""
    return grid_size // 2 + 1 if value is None else value


def output_bound(model):
    """2 ||w_out||_1 bounds |kappa(x) - kappa(y)| for K0/K1 (tanh features in [-1, 1])."""
    if model.kinetic.mode == "K2":
        return None
    return 2 * float(model.kinetic.network.network[-1].weight.detach().abs().sum())


@torch.no_grad()
def centered_secant(model, cfg, beta):
    """In-range alpha secant of the centered kinetic rate vs the truth -k^2 (gauge-free)."""
    model = widen_to_double(model, device="cpu").eval()
    alphas = torch.tensor(ALPHA_GRID, dtype=torch.float64)
    parameters = torch.stack([alphas, torch.full_like(alphas, beta)], -1)
    kappa = model.kinetic(parameters)[:, : cfg.grid_size // 2 + 1]
    centered = kappa - kappa[:, :1]
    slope = torch.diff(centered, dim=0) / torch.diff(alphas)[:, None]
    k2 = torch.arange(cfg.grid_size // 2 + 1, dtype=torch.float64).square()
    error = ((slope + k2).abs() / k2.clamp_min(1)).amax(0)
    error[0] = 0.0
    return error.tolist()


def g5_plane_wave(model, cfg, probe_k):
    domain = cfg.domain
    kwargs = dict(beta=PROBE_BETA, amplitude=probe_amplitude(domain, cfg.mass_range),
                  potential_constant=PROBE_V0, dt=cfg.dt, widen=True)
    derivative = {}
    for k in probe_k:
        if k <= cfg.grid_size // 2:
            result = alpha_phase_derivative(model, domain, k, alphas=ALPHA_GRID, **kwargs).as_dict()
            derivative[str(k)] = {key: result[key] for key in ("estimate", "truth", "max_relative_error")}
    curve = dispersion_curve(model, domain, range(cfg.grid_size // 2 + 1), alpha=PROBE_ALPHA, **kwargs).as_dict()
    return {"G5b": derivative, "G5a": curve}


def rollout_cases(base, options):
    return {b: ProbeCase(f"G4-bandwidth-{b}", replace(base, initial_bandwidth=b))
            for b in options["rollout_bandwidths"] if b <= base.grid_size // 2}


def rollout_references(study):
    """One refined reference per test bandwidth, shared by every model; cached on disk."""
    options, root = study["options"], study["root"]
    references = {}
    for bandwidth, case in rollout_cases(study["base"], options).items():
        path = root / "references" / f"G4-bandwidth-{bandwidth}.pt"
        if path.exists():
            saved = torch.load(path, weights_only=False)
        else:
            inputs, digest = sample_probe(case, options["probe_seed"], options["probe_batch"])
            truth = reference_frames(inputs, case.config, steps=options["rollout_steps"],
                                     stride=options["rollout_steps"], substeps=options["reference_substeps"])
            saved = {"inputs": inputs, "truth": truth, "sha256": digest}
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(saved, path)
        references[bandwidth] = (case, saved)
    return references


@torch.no_grad()
def g4_rollouts(model, references, training_bandwidth):
    model = widen_to_double(model, device="cpu").eval()
    result = {}
    for bandwidth, (case, saved) in references.items():
        measured = measure_rollout(model, saved["inputs"], case.config, saved["truth"],
                                   training_bandwidth=training_bandwidth)
        final = measured["records"][-1]
        result[str(bandwidth)] = {
            metric: float(np.nanmean(np.asarray(final[metric], dtype=float)))
            for metric in ("state_error", "aligned_state_error", "phase_rms", "mass_drift", "energy_drift")}
        result[str(bandwidth)]["failed"] = sum(v is not None for v in measured["failed_at"])
    return result


def one_step_errors(model, cell, study, own_test, common_test):
    cfg = cell["config"]
    device = study["device"]
    evaluation = TrainConfig(batch_size=study["options"]["batch_size"], device=device)
    model = model.float().to(device)
    own = evaluate_one_step(model, OneStepBatches(own_test, device=device), cfg.domain, cfg.dt, evaluation)
    common = evaluate_one_step(model, OneStepBatches(common_test, device=device), cfg.domain, cfg.dt, evaluation)
    model.cpu()
    return {"own_test": own, "bw8_test": common}


def measure_model(model, cfg, options):
    """Everything that depends only on the weights (shared with the Phase 6 port check)."""
    knees, residuals, rates = {}, {}, {}
    for alpha in options["knee_alphas"]:
        residual = map_residual(model, cfg, alpha, options["knee_beta"])
        residuals[str(alpha)] = residual
        knees[str(alpha)] = {str(t): knee(residual, t) for t in options["knee_thresholds"]}
        k, _, _, centered = plane_wave_rates(model, cfg, alpha, options["knee_beta"])
        rates[str(alpha)] = {"centered_kappa": centered.tolist(), "truth": (-alpha * k.square()).tolist()}
    return {"knees": knees, "map_residual": residuals, "kinetic_rate": rates,
            "plateau": rates[str(options["knee_alphas"][0])]["centered_kappa"][-1],
            "output_bound": output_bound(model),
            "centered_secant_error": centered_secant(model, cfg, options["knee_beta"])}


def measure_all(study, *, source_root=None):
    root, options = study["root"], study["options"]
    gates = run_gates(study["base"].domain, study["base"])
    atomic_json(root / "gates.json", clean_json(gates))
    print("Probe gates passed on the reference solver", flush=True)
    references = rollout_references(study)
    common_test, _ = ensure_shards(study["base"], study["data_root"])
    common_test = common_test["test"]
    for cell in study["cells"]:
        own_test = None
        for seed in options["seeds"]:
            path = root / "cells" / cell["name"] / f"seed{seed}.json"
            if path.exists():
                continue
            if own_test is None:
                own_test = ensure_shards(cell["config"], study["data_root"])[0]["test"]
            started = time.monotonic()
            model, history = load_trained(study, cell, seed)
            record = {"run_id": study["run_id"], "cell": cell["name"], "kinetic": cell["kinetic"],
                      "bandwidth": cell["bandwidth"], "seed": seed, "budget": budget(history),
                      "history": history, **measure_model(model, cell["config"], options),
                      **g5_plane_wave(model, cell["config"], options["probe_k"]),
                      "G4": g4_rollouts(model, references, cell["bandwidth"]),
                      "one_step": one_step_errors(model, cell, study, own_test, common_test)}
            atomic_json(path, clean_json(record))
            primary = record["knees"][str(PRIMARY["alpha"])][str(PRIMARY["threshold"])]
            print(f"{cell['name']} seed {seed}: knee {primary}, plateau {record['plateau']:.1f} "
                  f"({time.monotonic() - started:.1f}s)", flush=True)
    port = port_check(study, source_root)
    atomic_json(root / "port-check.json", clean_json(port))
    return port


def port_check(study, source_root):
    """Colab K0-bw8 against the Windows Phase 6 C1 of the same seed and init."""
    cell = next((c for c in study["cells"] if c["kinetic"] == "K0"
                 and config_hash(c["config"]) == PHASE6_BASE_HASH), None)
    if source_root is None or cell is None:
        return {"status": "skipped", "reason": "no Phase 6 checkpoints or no K0-bw8 cell"}
    options, rows = study["options"], []
    for seed in options["seeds"]:
        path = Path(source_root) / "checkpoints/phase6" / f"{PHASE6_BASE_HASH}-K0" / f"C1-seed{seed}.pt"
        if not path.exists():
            continue
        payload = load_checkpoint_payload(path)
        windows = _model_from_checkpoint(payload, cell["config"], expected_name="C1")
        windows.load_state_dict(payload.state_dict, strict=True)
        theirs = measure_model(windows.eval(), cell["config"], options)
        ours = json.loads((study["root"] / "cells" / cell["name"] / f"seed{seed}.json").read_text())
        key = (str(PRIMARY["alpha"]), str(PRIMARY["threshold"]))
        knee_ours, knee_theirs = ours["knees"][key[0]][key[1]], theirs["knees"][key[0]][key[1]]
        ok = (knee_ours is not None and knee_theirs is not None and abs(knee_ours - knee_theirs) <= 1
              and abs(ours["plateau"] - theirs["plateau"]) <= .1 * abs(theirs["plateau"]))
        rows.append({"seed": seed, "knee_colab": knee_ours, "knee_windows": knee_theirs,
                     "plateau_colab": ours["plateau"], "plateau_windows": theirs["plateau"], "pass": ok})
    if not rows:
        return {"status": "skipped", "reason": "Phase 6 C1 checkpoints not found"}
    return {"status": "PASS" if all(r["pass"] for r in rows) else "CHECK", "rows": rows,
            "rule": "knee within 1 and plateau within 10% (different hardware, same init and data)"}


# ---------------------------------------------------------------- summary

def summarize(study):
    root, options = study["root"], study["options"]
    data = json.loads((root / "data-checks.json").read_text())
    cells, rows = {}, []
    for cell in study["cells"]:
        records = [json.loads((root / "cells" / cell["name"] / f"seed{s}.json").read_text())
                   for s in options["seeds"]]
        checks = data[config_hash(cell["config"])]
        entry = {"kinetic": cell["kinetic"], "bandwidth": cell["bandwidth"], "support": checks["support"],
                 "data_resolved": checks["resolution"]["tail_pass"] and checks["resolution"]["refined_pass"],
                 "knees": {}, "plateau": [r["plateau"] for r in records],
                 "output_bound": [r["output_bound"] for r in records],
                 "budget": [r["budget"] for r in records],
                 "G5b": {k: float(np.mean([r["G5b"][k]["max_relative_error"] for r in records]))
                         for k in records[0]["G5b"]},
                 "G4": {b: {m: float(np.mean([r["G4"][b][m] for r in records]))
                            for m in ("state_error", "aligned_state_error")} for b in records[0]["G4"]},
                 "one_step": {k: float(np.mean([r["one_step"][k] for r in records])) for k in ("own_test", "bw8_test")}}
        for alpha in options["knee_alphas"]:
            for threshold in options["knee_thresholds"]:
                values = [r["knees"][str(alpha)][str(threshold)] for r in records]
                entry["knees"][f"{alpha}/{threshold}"] = values
                rows.append({"cell": cell["name"], "kinetic": cell["kinetic"], "bandwidth": cell["bandwidth"],
                             "alpha": alpha, "threshold": threshold,
                             **{f"seed{s}": v for s, v in zip(options["seeds"], values)},
                             "support_edge_1e-6": checks["support"].get("1e-06")})
        cells[cell["name"]] = entry
    summary = {"run_id": study["run_id"], "cells": cells, "verdict": verdict(study, cells),
               "k_wrap": {"alpha_max": wrap_wavenumber(study["base"].alpha_range[1], study["base"].dt),
                          "alpha_min": wrap_wavenumber(study["base"].alpha_range[0], study["base"].dt)},
               "primary_knee": PRIMARY}
    atomic_json(root / "summary.json", clean_json(summary))
    columns = list(rows[0]) if rows else []
    lines = [",".join(columns)] + [",".join("" if r.get(c) is None else str(r.get(c)) for c in columns) for r in rows]
    (root / "knees.csv").write_text("\n".join(lines) + "\n")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["complete"] = True
    manifest["exploratory"] = any(not b["converged_gate"] for c in cells.values() for b in c["budget"])
    atomic_json(root / "manifest.json", manifest)
    return summary


def verdict(study, cells):
    """The pre-registered decision rule of notebook 13, applied mechanically."""
    key = f"{PRIMARY['alpha']}/{PRIMARY['threshold']}"
    base = study["base"]
    names = {c["name"]: c for c in study["cells"]}
    if "K0-bw8" not in cells or "K0-bw12" not in cells:
        return {"data_support": "not evaluable", "reason": "needs cells K0-bw8 and K0-bw12"}
    k8 = [as_knee(v, base.grid_size) for v in cells["K0-bw8"]["knees"][key]]
    k12 = [as_knee(v, base.grid_size) for v in cells["K0-bw12"]["knees"][key]]
    moved = sum(b >= a + 2 for a, b in zip(k8, k12))
    ordered = True
    if "K0-bw10" in cells:
        k10 = [as_knee(v, base.grid_size) for v in cells["K0-bw10"]["knees"][key]]
        ordered = float(np.median(k8)) <= float(np.median(k10)) <= float(np.median(k12))
    k0 = [c for n, c in cells.items() if names[n]["kinetic"] == "K0"]
    centre = float(np.median(k8))
    flat = all(abs(as_knee(v, base.grid_size) - centre) <= 1 for c in k0 for v in c["knees"][key])
    budget_bound = any((b["last5_improvement"] or 0) > .05 for b in cells["K0-bw12"]["budget"])
    unresolved = [n for n, c in cells.items() if not c["data_resolved"]]
    if ordered and moved >= math.ceil(2 * len(k8) / 3):
        outcome = "data support SUPPORTED"
    elif flat and not budget_bound:
        outcome = "architecture limit INDICATED"
    elif flat:
        outcome = "INCONCLUSIVE: knees did not move but K0-bw12 was still improving (>5% over the last 5 epochs); rerun with more epochs"
    else:
        outcome = "INCONCLUSIVE"
    return {"data_support": outcome, "k8": k8, "k12": k12, "seeds_moved_by_2": moved,
            "bw10_between": ordered, "all_K0_within_1": flat, "bw12_budget_bound": budget_bound,
            "unresolved_data_cells": unresolved,
            "rule": "SUPPORTED: K0-bw12 knee >= K0-bw8 knee + 2 on >= 2/3 seeds (paired) and bw10 median between. "
                    "ARCHITECTURE: every K0 knee within 1 of the bw8 median and bw12 not budget-bound. "
                    f"Knee = first k with one-step map residual > {PRIMARY['threshold']} at alpha={PRIMARY['alpha']}."}
