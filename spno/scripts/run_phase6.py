"""Phase 6: parameter and spectral generalization -- the thesis spine.

Runs the G1-G9 arms against trained checkpoints and writes one payload plus four
figures.  The centerpiece is G5: at fixed alpha the one-step map determines omega only
modulo ``2 pi / dt`` above ``k_wrap`` (**theorem**, and the probe reproduces it on the
reference solver to 3.1e-15), while the alpha-derivative ``d arg m / d alpha = -k^2 dt``
is wrap-free.  G5b asks which hypothesis classes actually extract that.

**The matched-bandwidth arm is not optional.**  ``SpectralConv1d`` keeps
``min(modes, n//2+1)`` modes, so at ``modes=16`` Model A has no learned mode coupling
above k=15 -- while ``k_wrap`` is 16.9-21.2.  A's failure above ``k_wrap`` would then be
over-determined by truncation alone, and G5 could not separate "the FNO cannot route
alpha-dependence to high k" from "the FNO has no weights there at all".  ``A-wide``
(``modes=32``, matched to Nyquist) removes that confound.  It is **not** capacity-matched
to C2; ``A`` remains the capacity-matched comparison.

Usage:
    python scripts/run_phase6.py [--quick] [--seeds 0 1 2] [--epochs 60]
                                 [--device auto] [--arms G1 G2 ...] [--kinetic K0]

By default this evaluates checkpoints produced by Phases 2-5 and the Phase 6 arm
trainer. Use --standalone to generate isolated datasets and train every required
checkpoint first; no previous phases are needed. Production evaluation still requires
converged checkpoints. --standalone --quick is a small end-to-end plumbing check.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # before pyplot, so a headless run cannot pick an interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch

from spno.checkpoints import (
    CheckpointPayload,
    checkpoint_path,
    load_checkpoint_payload,
    restore_checkpoint,
)
from spno.config import DataConfig, config_hash
from spno.data.datasets import TrajectoryShard, generate_shard, shard_paths
from spno.data.shift import SHIFT_SPECS, ShiftSpec, shift_identifier
from spno.equations.nls import alpha_sampling_is_dense_enough, wrap_wavenumber
from spno.evaluation.dispersion import (
    alpha_phase_derivative,
    dispersion_curve,
    omega_by_alpha_continuation,
    probe_amplitude,
    validate_probe,
)
from spno.evaluation.payloads import require_phase_arms
from spno.evaluation.spectral import band_ratio, cascade_series, evaluate_spectral
from spno.experiments import (
    DATA_ROOT,
    RESULTS_ROOT,
    evaluate_model,
    pick_device,
    run_identifier,
    save_run,
)
from spno.losses.relative_l2 import relative_l2_per_sample
from spno.models.base import allow_dt_transfer
from spno.models.fno import FNOStepOperator
from spno.models.projected import MassProjectedOperator
from spno.models.split_learned import (
    DensityPhaseSplitStep,
    FieldDensityPhaseSplitStep,
    FullFieldPhaseSplitStep,
)
from spno.precision import widen_to_double
from spno.solvers.split_step import SubsteppedReference
from spno.train import TrainConfig

#: The probe needs a *constant* potential -- a plane wave is an exact solution only then.
PROBE_V0 = 0.0
PROBE_ALPHA = 0.9
PROBE_BETA = 0.3
#: Spacing chosen to satisfy the G5b precondition at k = Nyquist; see the gate below.
ALPHA_GRID = tuple(0.7 + 0.05 * index for index in range(9))

ALL_ARMS = ("G1", "G2", "G3", "G4", "G5a", "G5b", "G6a", "G6b", "G7", "G9")
DT_VALUES = (0.005, 0.01, 0.02)
ROLLOUT_CHECKPOINTS = (1, 10, 20, 50, 100, 200)


def alpha_continuation_grid(target: float, *, max_k: int, dt: float) -> list[float]:
    """Connect zero to ``target`` without crossing a phase wrap at Nyquist."""

    maximum_gap = 0.95 * math.pi / (max_k**2 * dt)
    intervals = max(1, math.ceil(abs(target) / maximum_gap))
    return [target * index / intervals for index in range(intervals + 1)]


def quick_data_config(config: DataConfig) -> DataConfig:
    """Use tiny counts/horizons without changing the production spatial domain."""

    return replace(config, n_train=2, n_val=2, n_test=2, steps=2)


def multi_dt_identifier(config: DataConfig) -> str:
    return f"g6a-{config_hash(config)}"


def multi_dt_data_hash(data_config: DataConfig, *, quick: bool) -> str:
    """Hash the complete ordered collection of datasets used by G6a."""

    hashes = []
    for dt in DT_VALUES:
        config = replace(data_config, dt=dt)
        if quick:
            config = quick_data_config(config)
        hashes.append(config_hash(config))
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()[:10]


def phase6_artifact_root(*, quick: bool) -> Path:
    return RESULTS_ROOT / "phase6-quick-artifacts" if quick else DATA_ROOT


def _phase6_identifier(data_config: DataConfig, kinetic: str, *, quick: bool) -> str:
    return run_identifier(config_hash(data_config), kinetic, quick=quick)


def _model_from_checkpoint(
    payload: CheckpointPayload, data_config: DataConfig, *, expected_name: str
):
    """Rebuild exactly the constructor recorded alongside a checkpoint."""

    metadata = payload.metadata
    if metadata.model_name != expected_name:
        raise RuntimeError(
            f"checkpoint names {metadata.model_name!r}, expected {expected_name!r}"
        )
    architecture = metadata.architecture
    model_name = expected_name.rsplit("/", 1)[-1]
    trained_dt = metadata.trained_dt
    domain = data_config.domain

    if model_name in {"A", "A-wide", "B-loop"}:
        if metadata.field_scale is None:
            raise RuntimeError("FNO checkpoint is missing field_scale")
        core = FNOStepOperator(
            domain,
            modes=int(architecture["modes"]),
            width=int(architecture["width"]),
            n_layers=int(architecture["n_layers"]),
            alpha_range=tuple(architecture.get("alpha_range", data_config.alpha_range)),
            beta_range=tuple(architecture.get("beta_range", data_config.beta_range)),
            field_scale=float(metadata.field_scale),
            use_coordinate_channel=bool(
                architecture.get("use_coordinate_channel", False)
            ),
            trained_dt=trained_dt,
        )
        return MassProjectedOperator(core) if model_name == "B-loop" else core

    kinetic = architecture.get("kinetic_mode", "K0")
    if model_name == "C1":
        return DensityPhaseSplitStep(
            domain,
            kinetic_mode=kinetic,
            local_mode=architecture.get("local_mode") or "L0",
            width=int(architecture.get("width", 32)),
            trained_dt=trained_dt,
        )
    if model_name == "C2":
        return FieldDensityPhaseSplitStep(
            domain,
            kinetic_mode=kinetic,
            local_mode=architecture.get("local_mode") or "L0",
            modes=int(architecture.get("modes", 16)),
            width=int(architecture.get("width", 64)),
            n_layers=int(architecture.get("n_layers", 4)),
            trained_dt=trained_dt,
        )
    if model_name == "C3":
        return FullFieldPhaseSplitStep(
            domain,
            kinetic_mode=kinetic,
            modes=int(architecture.get("modes", 16)),
            width=int(architecture.get("width", 64)),
            n_layers=int(architecture.get("n_layers", 4)),
            trained_dt=trained_dt,
        )
    raise RuntimeError(f"unsupported checkpoint model {expected_name!r}")


def _load_model(
    path: Path,
    data_config: DataConfig,
    *,
    expected_name: str,
    expected_data_hash: str,
    allow_budget_bound: bool,
    creation_command: str,
):
    if not path.exists():
        raise RuntimeError(
            f"required checkpoint is missing: {path}\nCreate it with: {creation_command}"
        )
    try:
        payload = load_checkpoint_payload(path)
        model = _model_from_checkpoint(payload, data_config, expected_name=expected_name)
        return restore_checkpoint(
            model,
            payload,
            expected_data_hash=expected_data_hash,
            allow_budget_bound=allow_budget_bound,
        )
    except Exception as error:
        raise RuntimeError(
            f"checkpoint is incompatible: {path}\n{error}\n"
            f"Recreate it with: {creation_command}"
        ) from error


def load_models(
    checkpoint_root,
    data_config: DataConfig,
    seeds,
    kinetic: str,
    *,
    allow_budget_bound: bool = False,
    standalone: bool = False,
) -> dict[int, dict[str, object]]:
    """Load the Phase 2-5 family plus the Phase 6 A-wide ablation."""

    quick = allow_budget_bound
    phase23_identifier = run_identifier(config_hash(data_config), quick=quick)
    phase45_identifier = run_identifier(
        config_hash(data_config), "one-step", f"{kinetic}L0", quick=quick
    )
    phase6_identifier = _phase6_identifier(data_config, kinetic, quick=quick)
    commands = {
        "phase23": (
            "python scripts/run_phase23.py "
            f"{'--quick ' if quick else ''}--seeds {' '.join(map(str, seeds))}"
        ),
        "phase45": (
            "python scripts/run_phase45.py "
            f"{'--quick ' if quick else ''}--mode one-step --kinetic {kinetic} "
            f"--local L0 --seeds {' '.join(map(str, seeds))}"
        ),
        "phase6": (
            "python scripts/train_phase6_arms.py "
            f"{'--quick ' if quick else ''}--kinetic {kinetic} "
            f"--seeds {' '.join(map(str, seeds))}"
        ),
    }
    if standalone:
        command = (
            "python scripts/run_phase6.py --standalone "
            f"{'--quick ' if quick else ''}--kinetic {kinetic} "
            f"--seeds {' '.join(map(str, seeds))} --epochs <larger-budget>"
        )
        commands = dict.fromkeys(commands, command)

    loaded = {}
    for seed in seeds:
        models = {}
        for name in ("A", "B-loop"):
            models[name] = _load_model(
                checkpoint_path(
                    Path(checkpoint_root),
                    "phase6" if standalone else "phase23",
                    phase6_identifier if standalone else phase23_identifier, name, seed
                ),
                data_config,
                expected_name=name,
                expected_data_hash=config_hash(data_config),
                allow_budget_bound=allow_budget_bound,
                creation_command=commands["phase23"],
            )
        for name in ("C1", "C2", "C3"):
            models[name] = _load_model(
                checkpoint_path(
                    Path(checkpoint_root),
                    "phase6" if standalone else "phase45",
                    phase6_identifier if standalone else phase45_identifier, name, seed
                ),
                data_config,
                expected_name=name,
                expected_data_hash=config_hash(data_config),
                allow_budget_bound=allow_budget_bound,
                creation_command=commands["phase45"],
            )
            if models[name].kinetic.mode != kinetic:
                raise RuntimeError(
                    f"checkpoint kinetic mode for {name} is "
                    f"{models[name].kinetic.mode}, expected {kinetic}"
                )
        models["A-wide"] = _load_model(
            checkpoint_path(
                Path(checkpoint_root), "phase6", phase6_identifier, "A-wide", seed
            ),
            data_config,
            expected_name="A-wide",
            expected_data_hash=config_hash(data_config),
            allow_budget_bound=allow_budget_bound,
            creation_command=commands["phase6"],
        )
        loaded[int(seed)] = models
    return loaded


def _load_phase6_arm_models(
    checkpoint_root,
    data_config: DataConfig,
    seeds,
    kinetic: str,
    arm: str,
    model_names,
    *,
    allow_budget_bound: bool,
    expected_data_hash: str,
):
    quick = allow_budget_bound
    identifier = _phase6_identifier(data_config, kinetic, quick=quick)
    command = (
        "python scripts/train_phase6_arms.py "
        f"{'--quick ' if quick else ''}--kinetic {kinetic} "
        f"--seeds {' '.join(map(str, seeds))}"
    )
    loaded = {}
    for seed in seeds:
        loaded[int(seed)] = {
            name: _load_model(
                checkpoint_path(
                    Path(checkpoint_root),
                    "phase6",
                    identifier,
                    f"{arm}/{name}",
                    seed,
                ),
                data_config,
                expected_name=f"{arm}/{name}",
                expected_data_hash=expected_data_hash,
                allow_budget_bound=allow_budget_bound,
                creation_command=command,
            )
            for name in model_names
        }
    return loaded


def run_gates(domain, data_config: DataConfig) -> dict:
    """Two hard gates, before any model is touched.

    ``raise RuntimeError``, never ``assert``: ``python -O`` strips asserts, and these
    gates are the only thing between an instrument bug and a run of plausible-but-
    fictional omega curves.
    """

    amplitude = probe_amplitude(domain, data_config.mass_range)
    report = validate_probe(
        domain,
        dt=data_config.dt,
        alpha=PROBE_ALPHA,
        beta=PROBE_BETA,
        amplitude=amplitude,
        potential_constant=PROBE_V0,
        wave_numbers=range(0, domain.shape[0] // 2 + 1),
        substeps=data_config.substeps,
        alpha_grid=ALPHA_GRID,
    )

    # The precondition is asserted on the PROBE's alpha grid, not the dataset's. The
    # dataset gap (3.47e-3) passes with enormous margin; the probe picks its own
    # spacing and fails once the gap exceeds 0.307 at k=32.
    gap = max(abs(b - a) for a, b in zip(ALPHA_GRID, ALPHA_GRID[1:]))
    highest = data_config.max_wave_number
    if not alpha_sampling_is_dense_enough(gap, highest, data_config.dt):
        raise RuntimeError(
            f"probe alpha grid gap {gap:g} violates the G5b precondition at "
            f"k={highest}: needs gap < "
            f"{math.pi / (highest ** 2 * data_config.dt):.3g}"
        )
    return {"probe": report, "alpha_gap": gap, "probe_amplitude": amplitude}


def k_wrap_for(alpha_range: tuple[float, float], dt: float) -> tuple[float, float]:
    """``k_wrap`` is alpha-dependent, so each arm reports its own span."""

    return (wrap_wavenumber(alpha_range[1], dt), wrap_wavenumber(alpha_range[0], dt))


def dispersion_arms(models: dict, domain, data_config: DataConfig) -> dict:
    """G5a (fixed alpha) and G5b (alpha-varying), stored under separate keys.

    **Do not merge them.**  ``alpha_phase_derivative`` queries a *local* alpha grid
    inside the training range and is the identifiability measurement.
    ``omega_by_alpha_continuation`` anchors at alpha=0, far outside training -- an FNO's
    rescaled alpha channel sits near -4.5 there -- so its failure is evidence about
    alpha-extrapolation, not about identifiability.  Each carries a ``claim`` field
    naming which result it supports.
    """

    amplitude = probe_amplitude(domain, data_config.mass_range)
    probe_kwargs = dict(
        beta=PROBE_BETA,
        amplitude=amplitude,
        potential_constant=PROBE_V0,
        dt=data_config.dt,
        widen=True,
    )
    wave_numbers = list(range(0, domain.shape[0] // 2 + 1))
    maximum_k = data_config.max_wave_number
    probe_k = tuple(
        sorted({k for k in (8, 16, 20, 24, 30, 32, maximum_k) if k <= maximum_k})
    )
    continuation_grid = alpha_continuation_grid(
        PROBE_ALPHA, max_k=maximum_k, dt=data_config.dt
    )
    continuation_gap = max(
        abs(right - left)
        for left, right in zip(continuation_grid, continuation_grid[1:])
    )

    g5a, derivative, continuation = {}, {}, {}
    for name, model in models.items():
        g5a[name] = dispersion_curve(
            model, domain, wave_numbers, alpha=PROBE_ALPHA, **probe_kwargs
        ).as_dict()
        derivative[name] = {
            str(k): alpha_phase_derivative(
                model, domain, k, alphas=ALPHA_GRID, **probe_kwargs
            ).as_dict()
            for k in probe_k
        }
        continuation[name] = {
            str(k): omega_by_alpha_continuation(
                model,
                domain,
                k,
                alphas=continuation_grid,
                **probe_kwargs,
            )
            for k in probe_k
        }

    return {
        "G5a": {
            "claim": "theorem: at fixed alpha, omega is determined only mod 2pi/dt "
            "above k_wrap -- for every model",
            "curves": g5a,
        },
        "G5b": {
            "alpha_derivative": {
                "claim": "identifiability: d arg m/d alpha = -k^2 dt is wrap-free, so "
                "the information IS present. Does this class extract it?",
                "by_model": derivative,
            },
            "alpha_continuation": {
                "claim": "alpha-EXTRAPOLATION, not identifiability: anchoring at "
                "alpha=0 queries far outside training",
                "target": PROBE_ALPHA,
                "max_gap": continuation_gap,
                "max_phase_increment": (
                    continuation_gap * maximum_k**2 * data_config.dt
                ),
                "by_model": continuation,
            },
        },
    }


def transfer_arms(models: dict, domain, data_config: DataConfig) -> dict:
    """G6b: dt transfer, meaningful only where dt multiplies a learned rate."""

    amplitude = probe_amplitude(domain, data_config.mass_range)
    by_model, unsupported = {}, {}
    for name, model in models.items():
        if isinstance(model, (FNOStepOperator, MassProjectedOperator)):
            unsupported[name] = {
                "supported": False,
                "reason": "the FNO ignores its dt argument entirely; a transfer number "
                "would be an artefact of the harness, not a property of the model",
            }
            continue
        curves = {}
        with allow_dt_transfer(model):
            for dt in (0.005, 0.01, 0.02):
                curves[str(dt)] = dispersion_curve(
                    model,
                    domain,
                    list(range(0, domain.shape[0] // 2 + 1)),
                    alpha=PROBE_ALPHA,
                    beta=PROBE_BETA,
                    amplitude=amplitude,
                    potential_constant=PROBE_V0,
                    dt=dt,
                    widen=True,
                ).as_dict()
        by_model[name] = {"by_dt": curves}
    return {"by_model": by_model, "unsupported": unsupported}


def cascade_arm(
    models: dict,
    domain,
    data_config: DataConfig,
    *,
    steps: int = 200,
    stride: int = 25,
) -> dict:
    """G9: does the model reproduce the nonlinear cascade the reference produces?"""

    from spno.data.generate import sample_initial_conditions

    generator = torch.Generator().manual_seed(0)
    initial = sample_initial_conditions(
        domain, 8, data_config.initial_bandwidth, data_config.mass_range, generator
    )
    potential = torch.zeros_like(initial.real)
    alpha = torch.full((8,), PROBE_ALPHA, dtype=torch.float64)
    beta = torch.full((8,), PROBE_BETA, dtype=torch.float64)

    series = {
        "reference": cascade_series(
            SubsteppedReference(domain, data_config.substeps),
            domain, initial, potential, alpha, beta, data_config.dt,
            steps=steps,
            stride=stride,
            cutoff=float(data_config.initial_bandwidth),
        )
    }
    for name, model in models.items():
        series[name] = cascade_series(
            widen_to_double(model, device="cpu"),
            domain, initial, potential, alpha, beta, data_config.dt,
            steps=steps,
            stride=stride,
            cutoff=float(data_config.initial_bandwidth),
        )
    return series


def _aggregate_records(records: list):
    """Elementwise arithmetic mean for same-shaped measurement records."""

    first = records[0]
    if isinstance(first, bool) or first is None or isinstance(first, str):
        return first
    if isinstance(first, (int, float)):
        return sum(float(value) for value in records) / len(records)
    if isinstance(first, list):
        return [
            _aggregate_records([record[index] for record in records])
            for index in range(len(first))
        ]
    if isinstance(first, dict):
        return {
            key: _aggregate_records([record[key] for record in records])
            for key in first
        }
    raise TypeError(f"cannot aggregate measurement value {type(first).__name__}")


def _extreme_records(records: list, chooser):
    """Elementwise minimum or maximum for same-shaped measurement records."""

    first = records[0]
    if isinstance(first, bool) or first is None or isinstance(first, str):
        return first
    if isinstance(first, (int, float)):
        return float(chooser(records))
    if isinstance(first, list):
        return [
            _extreme_records([record[index] for record in records], chooser)
            for index in range(len(first))
        ]
    if isinstance(first, dict):
        return {
            key: _extreme_records([record[key] for record in records], chooser)
            for key in first
        }
    raise TypeError(f"cannot reduce measurement value {type(first).__name__}")


def _mean_min_max(values) -> dict[str, float]:
    numeric = [float(value) for value in values]
    return {
        "mean": sum(numeric) / len(numeric),
        "min": min(numeric),
        "max": max(numeric),
    }


def _available_checkpoints(shard: TrajectoryShard) -> tuple[int, ...]:
    horizon = shard.n_frames - 1
    selected = [step for step in ROLLOUT_CHECKPOINTS if step <= horizon]
    if horizon not in selected:
        selected.append(horizon)
    return tuple(selected)


@torch.no_grad()
def _final_prediction(model, shard: TrajectoryShard, data_config: DataConfig):
    model = widen_to_double(model, device="cpu").eval()
    state = shard.trajectories[:, 0]
    for _ in range(shard.n_frames - 1):
        state = model(
            state,
            shard.potential,
            shard.alpha,
            shard.beta,
            data_config.dt,
        )
    return state


def _evaluate_seed_model(
    model, shard: TrajectoryShard, data_config: DataConfig, *, device: str
) -> dict:
    model.to(device)
    train_config = TrainConfig(batch_size=256, device=device)
    metrics = evaluate_model(
        model,
        {"test": shard},
        data_config,
        train_config,
        checkpoints=_available_checkpoints(shard),
        n_rollout=shard.n_trajectories,
    )
    prediction = _final_prediction(model, shard, data_config)
    target = shard.trajectories[:, -1]
    high_edge = wrap_wavenumber(data_config.alpha_range[1], data_config.dt)
    metrics["spectral"] = evaluate_spectral(
        prediction,
        target,
        data_config.domain,
        band_edges=(float(data_config.initial_bandwidth), high_edge),
    ).as_dict()
    one_step = _final_prediction(
        model,
        TrajectoryShard(
            trajectories=shard.trajectories[:, :2],
            potential=shard.potential,
            alpha=shard.alpha,
            beta=shard.beta,
            trajectory_ids=shard.trajectory_ids,
            dt=shard.dt,
            split=shard.split,
            metadata=shard.metadata,
        ),
        replace(data_config, steps=1),
    )
    metrics["alpha_curve"] = {
        "alpha": shard.alpha.tolist(),
        "relative_error": relative_l2_per_sample(
            one_step, shard.trajectories[:, 1], data_config.domain
        ).tolist(),
    }
    return metrics


def _aggregate_model_evaluations(per_seed: dict[int, dict[str, dict]]) -> dict:
    model_names = list(next(iter(per_seed.values())))
    aggregated = {}
    for name in model_names:
        entries = [per_seed[seed][name] for seed in per_seed]
        one_step = _mean_min_max([entry["one_step_test"] for entry in entries])
        alpha_errors = [entry["alpha_curve"]["relative_error"] for entry in entries]
        aggregated[name] = {
            "one_step_test": one_step["mean"],
            "one_step_test_min": one_step["min"],
            "one_step_test_max": one_step["max"],
            "parameters": entries[0]["parameters"],
            "rollout": _aggregate_records([entry["rollout"] for entry in entries]),
            "rollout_min": _extreme_records(
                [entry["rollout"] for entry in entries], min
            ),
            "rollout_max": _extreme_records(
                [entry["rollout"] for entry in entries], max
            ),
            "spectral": _aggregate_records([entry["spectral"] for entry in entries]),
            "spectral_min": _extreme_records(
                [entry["spectral"] for entry in entries], min
            ),
            "spectral_max": _extreme_records(
                [entry["spectral"] for entry in entries], max
            ),
            "alpha_curve": {
                "alpha": entries[0]["alpha_curve"]["alpha"],
                "relative_error": _aggregate_records(alpha_errors),
                "relative_error_min": [
                    min(values) for values in zip(*alpha_errors)
                ],
                "relative_error_max": [
                    max(values) for values in zip(*alpha_errors)
                ],
            },
            "per_seed": {
                str(seed): per_seed[seed][name] for seed in per_seed
            },
        }
    return aggregated


def _shift_config(spec: ShiftSpec, *, quick: bool) -> DataConfig:
    return quick_data_config(spec.config) if quick else spec.config


def _load_shift_shard(
    root: Path, spec: ShiftSpec, split: str, *, quick: bool
) -> tuple[TrajectoryShard, DataConfig]:
    config = _shift_config(spec, quick=quick)
    path = shard_paths(Path(root), shift_identifier(spec))[split]
    if path.exists():
        return TrajectoryShard.load(path), config
    if quick:
        return (
            generate_shard(
                config, split, potential_family=spec.potential_family
            ),
            config,
        )
    raise RuntimeError(
        f"required Phase 6 shift shard is missing: {path}\n"
        "Create it with: python scripts/train_phase6_arms.py"
    )


def _evaluate_shift(
    spec: ShiftSpec,
    models_by_seed: dict[int, dict],
    *,
    device: str,
    quick: bool,
    shift_root: Path,
) -> dict:
    shard, config = _load_shift_shard(shift_root, spec, "test", quick=quick)
    per_seed = {
        seed: {
            name: _evaluate_seed_model(model, shard, config, device=device)
            for name, model in models.items()
        }
        for seed, models in models_by_seed.items()
    }
    return {
        "identifier": shift_identifier(spec),
        "potential_family": spec.potential_family,
        "note": spec.note,
        "config_hash": config_hash(config),
        "seed": config.seed,
        "k_wrap": list(k_wrap_for(config.alpha_range, config.dt)),
        "horizon": shard.n_frames - 1,
        "shard_metadata": shard.metadata,
        "by_model": _aggregate_model_evaluations(per_seed),
    }


def _run_g6a(
    checkpoint_root,
    shift_root: Path,
    data_config: DataConfig,
    seeds,
    kinetic: str,
    *,
    device: str,
    quick: bool,
) -> dict:
    models_by_seed = _load_phase6_arm_models(
        checkpoint_root,
        data_config,
        seeds,
        kinetic,
        "G6a",
        ("C1", "C2", "C3"),
        allow_budget_bound=quick,
        expected_data_hash=multi_dt_data_hash(data_config, quick=quick),
    )
    per_model: dict[str, dict[int, dict[str, dict]]] = {
        name: {} for name in ("C1", "C2", "C3")
    }
    for dt in DT_VALUES:
        config = replace(data_config, dt=dt)
        if quick:
            config = quick_data_config(config)
        path = shard_paths(shift_root, multi_dt_identifier(config))["test"]
        if path.exists():
            shard = TrajectoryShard.load(path)
        elif quick:
            shard = generate_shard(config, "test")
        else:
            raise RuntimeError(
                f"required G6a shard is missing: {path}\n"
                "Create it with: python scripts/train_phase6_arms.py"
            )
        for seed, models in models_by_seed.items():
            for name, model in models.items():
                per_model[name].setdefault(seed, {})[str(dt)] = _evaluate_seed_model(
                    model, shard, config, device=device
                )

    by_model = {}
    for name, per_seed in per_model.items():
        measurements = {}
        for dt in map(str, DT_VALUES):
            entries = [per_seed[seed][dt] for seed in per_seed]
            stats = _mean_min_max([entry["one_step_test"] for entry in entries])
            measurements[dt] = {
                "one_step_test": stats["mean"],
                "one_step_test_min": stats["min"],
                "one_step_test_max": stats["max"],
                "spectral": _aggregate_records(
                    [entry["spectral"] for entry in entries]
                ),
            }
        by_model[name] = {
            "parameters": next(iter(models_by_seed.values()))[name].parameter_count(),
            "measurements": measurements,
        }
    return {
        "dt_values": list(DT_VALUES),
        "data_hash": multi_dt_data_hash(data_config, quick=quick),
        "by_model": by_model,
        "unsupported": {
            "A": {
                "supported": False,
                "reason": "the FNO ignores dt, so multi-dt supervision cannot be represented",
            }
        },
    }


def _g7_measurement(models_by_seed, shard, config, *, device: str) -> dict:
    ratios: dict[str, list[float]] = {}
    banded: dict[str, list[dict[str, float]]] = {}
    by_model = {}
    low_edge = float(config.initial_bandwidth)
    high_edge = wrap_wavenumber(config.alpha_range[1], config.dt)
    for seed, models in models_by_seed.items():
        for name, model in models.items():
            model.to(device)
            prediction = _final_prediction(model, shard, config)
            target = shard.trajectories[:, -1]
            spectral = evaluate_spectral(
                prediction,
                target,
                config.domain,
                band_edges=(low_edge, high_edge),
            ).as_dict()
            ratios.setdefault(name, []).append(
                band_ratio(
                    prediction,
                    target,
                    config.domain,
                    low_edge=low_edge,
                    high_edge=high_edge,
                )
            )
            banded.setdefault(name, []).append(spectral["banded"])
    for name in ratios:
        ratio_stats = _mean_min_max(ratios[name])
        by_model[name] = {
            "parameters": next(iter(models_by_seed.values()))[name].parameter_count(),
            "band_ratio": ratio_stats,
            "banded": _aggregate_records(banded[name]),
        }
    return {
        "band_ratio": {name: _mean_min_max(values)["mean"] for name, values in ratios.items()},
        "banded": {
            name: _aggregate_records(records) for name, records in banded.items()
        },
        "by_model": by_model,
    }


def _run_g7(
    base_models,
    checkpoint_root,
    shift_root: Path,
    data_config: DataConfig,
    seeds,
    kinetic: str,
    spec: ShiftSpec,
    *,
    device: str,
    quick: bool,
) -> dict:
    shard, fixed_config = _load_shift_shard(
        shift_root, spec, "test", quick=quick
    )
    fixed_models = _load_phase6_arm_models(
        checkpoint_root,
        data_config,
        seeds,
        kinetic,
        "G7-alpha-fixed",
        ("A", "C1", "C2"),
        allow_budget_bound=quick,
        expected_data_hash=config_hash(fixed_config),
    )
    varying = {
        seed: {name: models[name] for name in ("A", "C1", "C2")}
        for seed, models in base_models.items()
    }
    return {
        "shift_spec": spec.name,
        "test_identifier": shift_identifier(spec),
        "varying_alpha": _g7_measurement(varying, shard, fixed_config, device=device),
        "fixed_alpha": _g7_measurement(
            fixed_models, shard, fixed_config, device=device
        ),
    }


def run_selected_arms(
    selected,
    data_config: DataConfig,
    *,
    seeds,
    kinetic: str,
    device: str,
    quick: bool,
    checkpoint_root=RESULTS_ROOT,
    shift_root: Path | None = None,
    shift_specs: dict[str, ShiftSpec] = SHIFT_SPECS,
    models_by_seed: dict[int, dict] | None = None,
) -> dict:
    """Execute every selected Phase 6 arm exclusively from restored checkpoints."""

    selected = tuple(selected)
    shift_root = phase6_artifact_root(quick=quick) if shift_root is None else Path(shift_root)
    base_models = models_by_seed
    if base_models is None:
        base_models = load_models(
            checkpoint_root,
            data_config,
            seeds,
            kinetic,
            allow_budget_bound=quick,
        )
    experiments = {}

    for arm in ("G1", "G2", "G3", "G4"):
        if arm not in selected:
            continue
        experiments[arm] = {
            "measurements": {
                name: _evaluate_shift(
                    spec,
                    base_models,
                    device=device,
                    quick=quick,
                    shift_root=shift_root,
                )
                for name, spec in shift_specs.items()
                if name.startswith(arm)
            }
        }

    if "G5a" in selected or "G5b" in selected:
        per_seed = [
            dispersion_arms(models, data_config.domain, data_config)
            for models in base_models.values()
        ]
        dispersion = _aggregate_records(per_seed)
        if "G5a" in selected:
            experiments["G5a"] = dispersion["G5a"]
        if "G5b" in selected:
            experiments["G5b"] = dispersion["G5b"]

    if "G6a" in selected:
        experiments["G6a"] = _run_g6a(
            checkpoint_root,
            shift_root,
            data_config,
            seeds,
            kinetic,
            device=device,
            quick=quick,
        )
    if "G6b" in selected:
        per_seed = [
            transfer_arms(models, data_config.domain, data_config)
            for models in base_models.values()
        ]
        experiments["G6b"] = _aggregate_records(per_seed)
    if "G7" in selected:
        if "G7-alpha-fixed" not in shift_specs:
            raise RuntimeError("G7-alpha-fixed is missing from the shift registry")
        experiments["G7"] = _run_g7(
            base_models,
            checkpoint_root,
            shift_root,
            data_config,
            seeds,
            kinetic,
            shift_specs["G7-alpha-fixed"],
            device=device,
            quick=quick,
        )
    if "G9" in selected:
        per_seed = [
            cascade_arm(
                models,
                data_config.domain,
                data_config,
                steps=2 if quick else 200,
                stride=1 if quick else 25,
            )
            for models in base_models.values()
        ]
        experiments["G9"] = _aggregate_records(per_seed)
    return experiments


def make_plots(payload: dict, output) -> None:
    """Four figures.  Every legend is guarded -- an empty one raises under
    ``filterwarnings = ["error::UserWarning"]``."""

    plots = output / "plots"

    # 1. dispersion: the centerpiece
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    curves = payload["experiments"].get("G5a", {}).get("curves", {})
    for name, curve in curves.items():
        axes[0].plot(curve["wave_numbers"], curve["principal"], label=name, lw=1.2)
    if curves:
        any_curve = next(iter(curves.values()))
        axes[0].plot(
            any_curve["wave_numbers"], any_curve["truth"], "k--", label="truth", lw=1.5
        )
        axes[0].axvline(
            any_curve["k_wrap"], color="crimson", ls=":",
            label=f"k_wrap={any_curve['k_wrap']:.1f}",
        )
    axes[0].set_title("G5a: fixed alpha")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("omega")

    derivative = (
        payload["experiments"].get("G5b", {}).get("alpha_derivative", {}).get("by_model", {})
    )
    for name, by_k in derivative.items():
        ks = sorted(int(k) for k in by_k)
        axes[1].semilogy(
            ks,
            [max(by_k[str(k)]["max_relative_error"], 1e-18) for k in ks],
            marker="o", label=name,
        )
    axes[1].set_title("G5b: |rel. error| of d arg m/d alpha")
    axes[1].set_xlabel("k")

    for name, entry in (
        payload["experiments"].get("G6b", {}).get("by_model", {}).items()
    ):
        for dt, curve in entry["by_dt"].items():
            axes[2].semilogy(
                curve["wave_numbers"],
                [abs(r) + 1e-18 for r in curve["residual"]],
                label=f"{name} dt={dt}", lw=1.0,
            )
    axes[2].set_title("G6: dt transfer residual")
    axes[2].set_xlabel("k")

    for axis in axes:
        axis.grid(True, alpha=0.3)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=7)
    figure.suptitle(f"Phase 6 dispersion  [{payload['data_hash']}]")
    figure.tight_layout()
    figure.savefig(plots / "phase6_dispersion.png", dpi=150)
    plt.close(figure)

    # 2. E(k) heatmap, mode x model
    spectra = payload["experiments"].get("G9", {})
    if spectra:
        names = [n for n in spectra if n != "reference"]
        if names:
            matrix = torch.tensor(
                [[max(v, 1e-30) for v in spectra[n]["spectra"][-1]] for n in names]
            )
            figure, axis = plt.subplots(figsize=(10, 0.6 * len(names) + 2))
            mesh = axis.pcolormesh(matrix.numpy(), norm=LogNorm(), cmap="viridis")
            axis.set_yticks([i + 0.5 for i in range(len(names))])
            axis.set_yticklabels(names)
            axis.set_xlabel("mode index (sorted by |k|)")
            figure.colorbar(mesh, ax=axis, label="E(k)")
            figure.suptitle(f"Phase 6 final spectra  [{payload['data_hash']}]")
            figure.tight_layout()
            figure.savefig(plots / "phase6_spectra.png", dpi=150)
            plt.close(figure)

    # 3. error vs alpha, training range shaded
    figure, axis = plt.subplots(figsize=(7, 4.5))
    low, high = payload["alpha_train_range"]
    axis.axvspan(low, high, alpha=0.15, color="tab:green", label="training range")
    for name, by_k in derivative.items():
        entry = by_k.get("8")
        if entry:
            alpha_midpoints = [
                (left + right) / 2
                for left, right in zip(entry["alphas"], entry["alphas"][1:])
            ]
            axis.semilogy(
                alpha_midpoints,
                [abs(s - entry["truth"]) + 1e-18 for s in entry["slopes"]],
                marker=".", label=name,
            )
    axis.set_xlabel("alpha")
    axis.set_ylabel("|slope - truth| at k=8")
    axis.grid(True, alpha=0.3)
    if axis.get_legend_handles_labels()[0]:
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(plots / "phase6_parameters.png", dpi=150)
    plt.close(figure)

    # 4. cascade vs time
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for name, series in spectra.items():
        axis.plot(
            series["steps"], series["fraction_above_cutoff"],
            marker="o" if name == "reference" else None,
            ls="--" if name == "reference" else "-",
            label=name,
        )
    axis.set_xlabel("step")
    axis.set_ylabel(f"energy fraction above k={payload.get('cascade_cutoff')}")
    axis.set_yscale("log")
    axis.grid(True, alpha=0.3)
    if axis.get_legend_handles_labels()[0]:
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(plots / "phase6_cascade.png", dpi=150)
    plt.close(figure)

    # 5. distribution shifts: alpha curves for G1/G2, per-arm bars for G3/G4
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, arm in zip(axes[0], ("G1", "G2")):
        measurements = payload["experiments"].get(arm, {}).get("measurements", {})
        for measurement in measurements.values():
            for name, metrics in measurement["by_model"].items():
                curve = metrics["alpha_curve"]
                axis.semilogy(
                    curve["alpha"],
                    [max(value, 1e-18) for value in curve["relative_error"]],
                    ".",
                    label=name,
                )
        axis.set(xlabel="alpha", ylabel="one-step relative L2", title=f"{arm}: error vs alpha")
    for axis, arm in zip(axes[1], ("G3", "G4")):
        measurements = payload["experiments"].get(arm, {}).get("measurements", {})
        for name in sorted(
            {
                model
                for measurement in measurements.values()
                for model in measurement["by_model"]
            }
        ):
            labels = list(measurements)
            axis.plot(
                labels,
                [measurements[label]["by_model"][name]["one_step_test"] for label in labels],
                "o-",
                label=name,
            )
        axis.set(ylabel="one-step relative L2", title=f"{arm}: per-arm error")
        axis.tick_params(axis="x", rotation=25)
        axis.set_yscale("log")
    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(plots / "phase6_generalization.png", dpi=150)
    plt.close(figure)

    # 6. G7: absolute band errors next to the normalized high/low ratio
    g7 = payload["experiments"].get("G7", {})
    if g7:
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for training_arm, style in (("varying_alpha", "-"), ("fixed_alpha", "--")):
            entry = g7[training_arm]
            for name, bands in entry["banded"].items():
                axes[0].plot(
                    list(bands),
                    [max(value, 1e-18) for value in bands.values()],
                    marker="o",
                    ls=style,
                    label=f"{training_arm}/{name}",
                )
            names = list(entry["band_ratio"])
            offset = -0.18 if training_arm == "varying_alpha" else 0.18
            axes[1].bar(
                [index + offset for index in range(len(names))],
                [entry["band_ratio"][name] for name in names],
                width=0.36,
                label=training_arm,
            )
            axes[1].set_xticks(range(len(names)), names)
        axes[0].set_yscale("log")
        axes[0].set(xlabel="spectral band", ylabel="relative L2", title="G7 absolute band errors")
        axes[1].set(ylabel="high/low error", title="G7 normalized band ratio")
        for axis in axes:
            axis.grid(True, alpha=0.3)
            axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(plots / "phase6_g7.png", dpi=150)
        plt.close(figure)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--standalone", action="store_true",
        help="generate isolated data and train all models before evaluation; no prior phases needed",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="standalone training budget (default: 60, or 1 with --quick)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--arms", nargs="+", default=list(ALL_ARMS), choices=ALL_ARMS)
    parser.add_argument("--kinetic", default="K0", choices=("K0", "K1", "K2"))
    from spno.phase_workflow import add_workflow_arguments, parse_workflow_args
    add_workflow_arguments(parser)
    return parse_workflow_args(parser, argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    if args.stage is not None:
        from spno.phase_workflow import run_stage
        return run_stage(6, args)

    data_config = DataConfig()
    if args.standalone and args.quick:
        data_config = quick_data_config(data_config)
    epochs = args.epochs if args.epochs is not None else (1 if args.quick else 60)
    if epochs < 1:
        raise ValueError("--epochs must be at least 1")
    domain = data_config.domain
    device = pick_device(args.device)
    identifier = run_identifier(
        config_hash(data_config), args.kinetic,
        "standalone" if args.standalone else "", quick=args.quick
    )

    seeds = args.seeds[:1] if args.quick else args.seeds
    gates = run_gates(domain, data_config)
    checkpoint_root = RESULTS_ROOT
    shift_root = phase6_artifact_root(quick=args.quick)
    training = None
    if args.standalone:
        if __package__:
            from scripts.train_phase6_arms import train_phase6_arms
        else:
            from train_phase6_arms import train_phase6_arms

        checkpoint_root = RESULTS_ROOT / run_identifier(
            "phase6-standalone-artifacts", quick=args.quick
        )
        shift_root = checkpoint_root / "data"
        print(f"Preparing standalone Phase 6 artifacts in {checkpoint_root}", flush=True)
        training = train_phase6_arms(
            data_config,
            seeds=seeds,
            epochs=epochs,
            device=device,
            kinetic=args.kinetic,
            quick=args.quick,
            standalone=True,
            data_root=shift_root,
            checkpoint_root=checkpoint_root,
            artifact_root=shift_root,
        )
        save_run("phase6-training", identifier, training)
    models_by_seed = load_models(
        checkpoint_root,
        data_config,
        seeds,
        args.kinetic,
        allow_budget_bound=args.quick,
        standalone=args.standalone,
    )
    parameter_counts = {
        name: model.parameter_count()
        for name, model in next(iter(models_by_seed.values())).items()
    }

    payload: dict = {
        "phase": 6,
        "data_hash": config_hash(data_config),
        "identifier": identifier,
        "device": device,
        "seeds": seeds,
        "epochs": epochs,
        "standalone": args.standalone,
        "training": training,
        "kinetic_mode": args.kinetic,
        "quick": args.quick,
        "gates": gates,
        "alpha_train_range": list(data_config.alpha_range),
        "k_wrap_training": list(k_wrap_for(data_config.alpha_range, data_config.dt)),
        "cascade_cutoff": data_config.initial_bandwidth,
        "parameter_counts": parameter_counts,
        "notes": {
            "A-wide": "modes=32 matched to Nyquist; removes the truncation confound. "
            "NOT capacity-matched to C2 -- A remains the capacity-matched arm.",
            "prediction": "at modes=16, omega_A(k) above k=15 comes only from k-blind "
            "pointwise operations, so it should NOT grow like k^2. omega_A-wide may.",
            "plane_waves_are_OOD": "these probe the learned operator; models were "
            "trained on random band-limited fields, so this is not a generalization "
            "claim about the training distribution.",
            "eps_split": "do NOT quote eps_split beside a probe residual: on a plane "
            "wave in a constant potential the Strang step is exact, so probe residuals "
            "floor at float64 roundoff instead.",
        },
        "experiments": run_selected_arms(
            args.arms,
            data_config,
            seeds=seeds,
            kinetic=args.kinetic,
            device=device,
            quick=args.quick,
            checkpoint_root=checkpoint_root,
            shift_root=shift_root,
            models_by_seed=models_by_seed,
        ),
    }
    require_phase_arms(6, args.arms, payload["experiments"])
    output = save_run("phase6", identifier, payload)
    make_plots(payload, output)

    print(f"{'phase':<22}6 -- parameter and spectral generalization")
    print(f"{'data hash':<22}{payload['data_hash']}")
    print(f"{'identifier':<22}{identifier}")
    print(f"{'device':<22}{device}")
    print(f"{'k_wrap (training)':<22}{payload['k_wrap_training'][0]:.2f} - "
          f"{payload['k_wrap_training'][1]:.2f}")
    print(f"{'probe alpha gap':<22}{gates['alpha_gap']:.4g}")
    print(f"{'arms':<22}{' '.join(args.arms)}")
    for name, count in payload["parameter_counts"].items():
        print(f"{'  params ' + name:<22}{count:,}")
    print(f"{'written to':<22}{output}")
    return payload


if __name__ == "__main__":
    main()
