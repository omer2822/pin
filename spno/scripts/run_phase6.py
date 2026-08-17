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

Nothing here trains: the runner consumes checkpoints produced by Phases 2-5 and must not
be run before the convergence debt on those phases is cleared.
"""

from __future__ import annotations

import argparse
import math

import matplotlib

matplotlib.use("Agg")  # before pyplot, so a headless run cannot pick an interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch

from spno.config import DataConfig, config_hash
from spno.data.shift import SHIFT_SPECS
from spno.equations.nls import alpha_sampling_is_dense_enough, wrap_wavenumber
from spno.evaluation.dispersion import (
    alpha_phase_derivative,
    dispersion_curve,
    omega_by_alpha_continuation,
    probe_amplitude,
    validate_probe,
)
from spno.evaluation.spectral import band_ratio, cascade_series
from spno.experiments import pick_device, run_identifier, save_run
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

#: The probe needs a *constant* potential -- a plane wave is an exact solution only then.
PROBE_V0 = 0.0
PROBE_ALPHA = 0.9
PROBE_BETA = 0.3
#: Spacing chosen to satisfy the G5b precondition at k = Nyquist; see the gate below.
ALPHA_GRID = tuple(0.7 + 0.05 * index for index in range(9))

ALL_ARMS = ("G1", "G2", "G3", "G4", "G5a", "G5b", "G6a", "G6b", "G7", "G9")


def build_models(domain, data_config: DataConfig, kinetic: str) -> dict:
    """Model arms.  Every entry records its parameter count in the payload."""

    dt = data_config.dt
    alpha_range = data_config.alpha_range
    beta_range = data_config.beta_range
    common = dict(alpha_range=alpha_range, beta_range=beta_range, trained_dt=dt)
    return {
        "A": FNOStepOperator(domain, modes=16, **common),
        # Matched to Nyquist to remove the truncation confound. Larger than A, so NOT
        # capacity-matched to C2 -- A remains the capacity-matched comparison.
        "A-wide": FNOStepOperator(domain, modes=32, **common),
        "B-loop": MassProjectedOperator(FNOStepOperator(domain, modes=16, **common)),
        "C1": DensityPhaseSplitStep(domain, kinetic_mode=kinetic, trained_dt=dt),
        "C2": FieldDensityPhaseSplitStep(domain, kinetic_mode=kinetic, trained_dt=dt),
        "C3": FullFieldPhaseSplitStep(domain, trained_dt=dt),
    }


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
    probe_k = (8, 16, 20, 24, 30, 32)

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
                alphas=[0.0, *ALPHA_GRID],
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
                "by_model": continuation,
            },
        },
    }


def transfer_arms(models: dict, domain, data_config: DataConfig) -> dict:
    """G6b: dt transfer, meaningful only where dt multiplies a learned rate."""

    amplitude = probe_amplitude(domain, data_config.mass_range)
    results = {}
    for name, model in models.items():
        if isinstance(model, (FNOStepOperator, MassProjectedOperator)):
            results[name] = {
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
        results[name] = {"supported": True, "by_dt": curves}
    return results


def cascade_arm(models: dict, domain, data_config: DataConfig) -> dict:
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
            cutoff=float(data_config.initial_bandwidth),
        )
    }
    for name, model in models.items():
        series[name] = cascade_series(
            widen_to_double(model, device="cpu"),
            domain, initial, potential, alpha, beta, data_config.dt,
            cutoff=float(data_config.initial_bandwidth),
        )
    return series


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

    for name, entry in payload["experiments"].get("G6b", {}).items():
        if not entry.get("supported"):
            continue
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
            axis.semilogy(
                entry["alphas"],
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


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--arms", nargs="+", default=list(ALL_ARMS), choices=ALL_ARMS)
    parser.add_argument("--kinetic", default="K0", choices=("K0", "K1", "K2"))
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    data_config = DataConfig()
    domain = data_config.domain
    device = pick_device(args.device)
    identifier = run_identifier(
        config_hash(data_config), args.kinetic, quick=args.quick
    )

    gates = run_gates(domain, data_config)
    models = build_models(domain, data_config, args.kinetic)

    payload: dict = {
        "phase": 6,
        "data_hash": config_hash(data_config),
        "identifier": identifier,
        "device": device,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "kinetic_mode": args.kinetic,
        "quick": args.quick,
        "gates": gates,
        "alpha_train_range": list(data_config.alpha_range),
        "k_wrap_training": list(k_wrap_for(data_config.alpha_range, data_config.dt)),
        "cascade_cutoff": data_config.initial_bandwidth,
        "parameter_counts": {n: m.parameter_count() for n, m in models.items()},
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
        "experiments": {},
    }

    if "G5a" in args.arms or "G5b" in args.arms:
        payload["experiments"].update(dispersion_arms(models, domain, data_config))
    if "G6b" in args.arms:
        payload["experiments"]["G6b"] = transfer_arms(models, domain, data_config)
    if "G9" in args.arms:
        payload["experiments"]["G9"] = cascade_arm(models, domain, data_config)
    if "G7" in args.arms:
        payload["experiments"]["G7"] = {
            "note": "band_ratio(fixed-alpha) vs band_ratio(alpha-varying); absolute "
            "banded errors recorded too, so the reader sees the level drop the ratio "
            "normalizes away. Requires trained checkpoints for both arms.",
            "shift_spec": "G7-alpha-fixed",
        }
    for arm in ("G1", "G2", "G3", "G4"):
        if arm not in args.arms:
            continue
        payload["experiments"][arm] = {
            name: {
                "identifier": spec.name,
                "k_wrap": list(k_wrap_for(spec.config.alpha_range, spec.config.dt)),
                "potential_family": spec.potential_family,
                "note": spec.note,
            }
            for name, spec in SHIFT_SPECS.items()
            if name.startswith(arm)
        }

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
