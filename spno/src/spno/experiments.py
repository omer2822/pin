"""Shared plumbing for the phase scripts: load shards, evaluate, record results.

Every run writes ``results/<phase>-<config hash>/`` containing the config, the metrics,
and the plots, so a number in the thesis can be traced back to the run that produced it.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch

from .config import DataConfig, config_hash
from .data.datasets import TrajectoryShard, shard_paths
from .evaluation.rollout import evaluate_rollout
from .precision import widen_to_double
from .solvers.split_step import SplitStepNLSOperator, SubsteppedReference
from .train import TrainConfig, TrainHistory, evaluate_one_step
from .data.datasets import OneStepBatches

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
RESULTS_ROOT = ROOT / "results"

Tensor = torch.Tensor


def load_shards(config: DataConfig) -> dict[str, TrajectoryShard]:
    paths = shard_paths(DATA_ROOT, config_hash(config))
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "dataset shards not found; run scripts/run_phase1.py first.\n  missing: "
            + "\n  ".join(missing)
        )
    return {split: TrajectoryShard.load(path) for split, path in paths.items()}


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def rollout_inputs(
    shard: TrajectoryShard,
    data_config: DataConfig,
    device: str,
    *,
    n: int = 100,
    dtype: torch.dtype = torch.complex64,
) -> dict[str, Tensor]:
    """Prepare a fixed evaluation batch shared by every model, for comparability."""

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    return {
        "initial": shard.trajectories[:n, 0].to(device=device, dtype=dtype),
        "trajectories": shard.trajectories[:n].to(device=device, dtype=dtype),
        "potential": shard.potential[:n].to(device=device, dtype=real_dtype),
        "alpha": shard.alpha[:n].to(device=device, dtype=real_dtype),
        "beta": shard.beta[:n].to(device=device, dtype=real_dtype),
    }


def evaluate_model(
    model,
    shards: dict[str, TrajectoryShard],
    data_config: DataConfig,
    train_config: TrainConfig,
    *,
    checkpoints: tuple[int, ...] = (1, 10, 20, 50, 100, 200),
    n_rollout: int = 100,
) -> dict:
    """One-step test error plus a rollout with invariant drift."""

    domain = data_config.domain
    device = train_config.device
    test_batches = OneStepBatches(shards["test"], device=device)
    one_step = evaluate_one_step(model, test_batches, domain, data_config.dt, train_config)

    # Rollout and invariants are measured in float64 on CPU with the trained weights
    # widened.  Otherwise float32 arithmetic drift (~1.2e-7 per step, accumulating
    # linearly) sits on top of every conservation number and would mask an
    # architectural violation of comparable size.  The weights are unchanged; only the
    # arithmetic used to iterate them is.
    # widen_to_double deep-copies, so the original stays on its training device.
    # (``model.to("cpu")`` mutates in place and would silently strand a shared core --
    # Model B-post reuses Model A's weights.)
    evaluation_model = widen_to_double(model, device="cpu").eval()
    inputs = rollout_inputs(
        shards["test"], data_config, "cpu", n=n_rollout, dtype=torch.complex128
    )
    metrics = evaluate_rollout(
        evaluation_model,
        domain,
        inputs["initial"],
        inputs["trajectories"],
        inputs["potential"],
        inputs["alpha"],
        inputs["beta"],
        data_config.dt,
        checkpoints=checkpoints,
    )
    return {
        "one_step_test": one_step,
        "parameters": model.parameter_count(),
        "rollout": metrics.as_dict(),
        "rollout_precision": "float64/cpu",
    }


def arithmetic_mass_floor(
    shards: dict[str, TrajectoryShard],
    data_config: DataConfig,
    *,
    steps: int = 100,
    n: int = 32,
) -> dict:
    """Mass drift of the *exact* solver taking one split step per model step.

    Every "exact mass preservation" claim is quoted against this.  Two subtleties, both
    found by measurement rather than assumption:

    * The floor must use **one** split step per model step, not the 32 substeps the
      data generator uses.  Drift accumulates roughly *linearly* at ~1.2e-7 per split
      step (float32 eps), not as a random walk, so the substepped generator drifts
      ~30x more than any model would.  Quoting that as a model's floor would hide
      genuine violations an order of magnitude in size.
    * MPS is about 3x worse than CPU at the same precision, so invariant measurements
      are taken on CPU even when training ran on MPS.
    """

    from .domain import l2_mass

    domain = data_config.domain
    results = {}
    for label, device, dtype in (
        ("float32_cpu", "cpu", torch.complex64),
        ("float64_cpu", "cpu", torch.complex128),
    ):
        inputs = rollout_inputs(shards["test"], data_config, device, n=n, dtype=dtype)
        solver = SplitStepNLSOperator(domain).to(device)
        with torch.no_grad():
            state = inputs["initial"]
            initial_mass = l2_mass(state, domain)
            for _ in range(steps):
                state = solver(
                    state,
                    inputs["potential"],
                    inputs["alpha"],
                    inputs["beta"],
                    data_config.dt,
                )
            drift = torch.abs(l2_mass(state, domain) / initial_mass - 1)
        results[label] = {"mean": float(drift.mean()), "max": float(drift.max())}
    results["steps"] = steps
    return results


def converged(history: TrainHistory) -> bool:
    """True when the best epoch was not the last one -- i.e. early stopping fired.

    A run whose ``best_val`` lands on the final epoch is *budget bound*: validation loss
    was still falling when the epoch cap hit, so its numbers describe the budget rather
    than the model.  Phases 2-3 shipped six such runs (``best_epoch == 24`` for all six
    at a 25-epoch cap, val loss still dropping 26-50% over the last five epochs).

    This is a necessary condition, not a sufficient one: it detects "the cap bound the
    run", not "the model reached its floor".  A run that early-stops on a plateau it
    would have escaped later still reads as converged.
    """

    return history.best_epoch < len(history.val_loss) - 1


def budget_warning(summary: dict[str, dict]) -> str | None:
    """The warning for any model with an unconverged seed, or None when all are clean.

    A single budget-bound seed condemns the model's row: averaging it with converged
    seeds would launder the cap into the reported mean.
    """

    unconverged = [
        name for name, entry in summary.items() if not all(entry["converged"])
    ]
    if not unconverged:
        return None
    return f"WARNING: budget bound for {unconverged} -- raise --epochs before reporting"


def run_identifier(base: str, *parts: str, quick: bool = False) -> str:
    """Assemble a results-directory identifier from every knob that moves the numbers.

    ``TrainConfig`` is not hashed -- ``describe_config`` records ``data_hash`` only --
    so anything that changes the result and is *not* in the data config has to appear
    here or two runs silently share a directory and the later one wins.

    ``quick`` is such a knob, and the reason this helper exists: a ``--quick`` smoke run
    is a handful of epochs on a truncated seed list, and it used to write to the same
    directory as the real run.  Running the documented smoke test therefore destroyed
    the Phases 2-3 metrics it was meant to leave alone.  Tagging the directory makes
    that structurally impossible rather than a thing to remember.
    """

    pieces = [base, *(str(part) for part in parts if part)]
    if quick:
        pieces.append("quick")
    return "-".join(pieces)


def save_run(
    phase: str, identifier: str, payload: dict, *, subdirectories: tuple[str, ...] = ("plots",)
) -> Path:
    output = RESULTS_ROOT / f"{phase}-{identifier}"
    for name in subdirectories:
        (output / name).mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    return output


def describe_config(data_config: DataConfig, train_config: TrainConfig) -> dict:
    return {
        "data": dataclasses.asdict(data_config),
        "train": dataclasses.asdict(train_config),
        "data_hash": config_hash(data_config),
    }
