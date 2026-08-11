"""Phase 0: validate the numerical reference and quantify the splitting floor.

Runs entirely in float64 on CPU and writes ``results/phase0-<hash>/``.

Deliverables:

1. **Convergence order** -- confirm the Strang splitting is second order in time.
2. **eps_split** -- the relative error of one split step at ``dt`` against the
   substepped reference.  This is the analytic lower bound on the one-step error of
   any model whose hypothesis class is a single split step, and it is the number that
   stops "structure beats FNO" from being a tautology.
3. **Long-run invariants** -- mass and energy over 1000 steps, classifying energy
   drift as bounded or secular.
4. **Resolution screen** -- spectral tail energy across the (alpha, beta) box, so an
   under-resolved corner is narrowed *before* any model is trained on it.

Usage::

    python scripts/run_phase0.py [--quick]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from spno.config import Phase0Config, config_hash
from spno.data.generate import (
    energy_fraction_above,
    sample_initial_conditions,
    sample_parameters,
    sample_potentials,
    spectral_tail_fraction,
)
from spno.domain import l2_mass
from spno.equations.nls import nls_hamiltonian, wrap_wavenumber
from spno.seeding import seed_everything
from spno.solvers.split_step import (
    SplitStepNLSOperator,
    SubsteppedReference,
    fitted_splitting_floor,
    relative_l2,
)

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"


def _sample_problem(config: Phase0Config, batch: int, generator: torch.Generator):
    data = config.data
    domain = data.domain
    field = sample_initial_conditions(
        domain, batch, data.initial_bandwidth, data.mass_range, generator
    )
    potential = sample_potentials(
        domain,
        batch,
        data.potential_amplitude_range,
        data.potential_correlation_length,
        generator,
    )
    alpha, beta = sample_parameters(
        batch, data.alpha_range, data.beta_range, generator
    )
    return domain, field, potential, alpha, beta


def measure_convergence_order(config: Phase0Config, generator: torch.Generator) -> dict:
    """Halving dt must quarter the error; report the observed exponents."""

    domain, field, potential, alpha, beta = _sample_problem(config, 8, generator)
    horizon = config.convergence_horizon
    truth = SubsteppedReference(domain, 1024)(field, potential, alpha, beta, horizon)

    errors = []
    for substeps in config.convergence_substeps:
        evolved = SubsteppedReference(domain, substeps)(
            field, potential, alpha, beta, horizon
        )
        errors.append(float(relative_l2(evolved, truth, domain).mean()))
    orders = [math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    return {
        "substeps": list(config.convergence_substeps),
        "errors": errors,
        "orders": orders,
        "finest_order": orders[-1],
    }


def measure_splitting_floor(config: Phase0Config, generator: torch.Generator) -> dict:
    """eps_split, plus the reference's own convergence gap (M=32 vs M=64)."""

    data = config.data
    domain, field, potential, alpha, beta = _sample_problem(config, 256, generator)

    coarse = SplitStepNLSOperator(domain)(field, potential, alpha, beta, data.dt)
    reference = SubsteppedReference(domain, data.substeps)(
        field, potential, alpha, beta, data.dt
    )
    finer = SubsteppedReference(domain, 2 * data.substeps)(
        field, potential, alpha, beta, data.dt
    )

    floor = relative_l2(coarse, reference, domain)
    gap = relative_l2(reference, finer, domain)

    # The bound that actually applies to a *learned* split step: the best a rescaled
    # pair of generators can do, which may sit below the exact-rate error.
    fitted, scales = fitted_splitting_floor(
        domain, field, potential, alpha, beta, data.dt, substeps=data.substeps
    )
    # The richer K2 family Phase 4 actually gives Model C1.
    fitted_k2, _ = fitted_splitting_floor(
        domain,
        field,
        potential,
        alpha,
        beta,
        data.dt,
        substeps=data.substeps,
        per_mode=True,
        iterations=200,
    )
    return {
        "eps_split_k2_mean": float(fitted_k2.mean()),
        "k2_over_exact": float(fitted_k2.mean() / floor.mean()),
        "eps_split_exact_mean": float(floor.mean()),
        "eps_split_exact_median": float(floor.median()),
        "eps_split_exact_min": float(floor.min()),
        "eps_split_exact_max": float(floor.max()),
        "eps_split_fitted_mean": float(fitted.mean()),
        "eps_split_fitted_median": float(fitted.median()),
        "fitted_over_exact": float(fitted.mean() / floor.mean()),
        "fitted_kinetic_scale": float(scales[0]),
        "fitted_local_scale": float(scales[1]),
        "reference_convergence_gap_max": float(gap.max()),
        "gap_over_fitted_floor": float(gap.max() / fitted.mean()),
        "substeps": data.substeps,
        "dt": data.dt,
    }


def measure_long_run_invariants(
    config: Phase0Config, generator: torch.Generator
) -> dict:
    """Mass and energy over a long rollout; classify energy drift bounded vs secular."""

    data = config.data
    domain, field, potential, alpha, beta = _sample_problem(config, 16, generator)
    operator = SplitStepNLSOperator(domain)

    initial_mass = l2_mass(field, domain)
    initial_energy = nls_hamiltonian(field, potential, domain, alpha, beta)

    times, mass_drift, energy_drift = [], [], []
    evolved = field
    stride = max(config.long_run_steps // 100, 1)
    for step in range(config.long_run_steps):
        evolved = operator(evolved, potential, alpha, beta, data.dt)
        if (step + 1) % stride == 0:
            times.append((step + 1) * data.dt)
            mass_drift.append(
                float(torch.abs(l2_mass(evolved, domain) / initial_mass - 1).max())
            )
            energy = nls_hamiltonian(evolved, potential, domain, alpha, beta)
            energy_drift.append(
                float(torch.abs(energy / initial_energy - 1).max())
            )

    # Secular drift would make the second half systematically worse than the first.
    half = len(energy_drift) // 2
    early, late = max(energy_drift[:half]), max(energy_drift[half:])
    return {
        "times": times,
        "mass_drift": mass_drift,
        "energy_drift": energy_drift,
        "max_mass_drift": max(mass_drift),
        "max_energy_drift": max(energy_drift),
        "energy_late_over_early": late / max(early, 1e-30),
        "energy_classification": "bounded" if late < 5 * early else "secular",
    }


def run_resolution_screen(config: Phase0Config, generator: torch.Generator) -> dict:
    """Spectral tail energy across the (alpha, beta) box after a full trajectory.

    Focusing (beta > 0) NLS steepens fields; if the tail grows, the grid is the
    limiting factor and any "generalization failure" measured there is really
    under-resolution.
    """

    data = config.data
    n = config.screen_samples
    domain, field, potential, alpha, beta = _sample_problem(config, n, generator)
    reference = SubsteppedReference(domain, data.substeps)
    k_wrap_min = wrap_wavenumber(data.alpha_range[1], data.dt)

    # Track the nonlinear cascade: how much energy climbs out of the training band, and
    # how much crosses the identifiability horizon, during an in-distribution rollout.
    checkpoints = {0: field}
    evolved = field
    for step in range(data.steps):
        evolved = reference(evolved, potential, alpha, beta, data.dt)
        if (step + 1) in (50, 100, data.steps):
            checkpoints[step + 1] = evolved
    cascade = {
        "steps": sorted(checkpoints),
        "above_k_train": [
            float(
                energy_fraction_above(
                    checkpoints[s], domain, data.initial_bandwidth
                ).max()
            )
            for s in sorted(checkpoints)
        ],
        "above_k_wrap": [
            float(energy_fraction_above(checkpoints[s], domain, k_wrap_min).max())
            for s in sorted(checkpoints)
        ],
        "k_train": data.initial_bandwidth,
        "k_wrap_min": k_wrap_min,
    }

    tail = spectral_tail_fraction(evolved, domain, config.tail_band_start)
    over = tail > config.tail_fraction_threshold
    worst_index = int(torch.argmax(tail))
    return {
        "threshold": config.tail_fraction_threshold,
        "band_start_fraction": config.tail_band_start,
        "max_tail_fraction": float(tail.max()),
        "median_tail_fraction": float(tail.median()),
        "n_over_threshold": int(over.sum()),
        "n_samples": n,
        "worst_alpha": float(alpha[worst_index]),
        "worst_beta": float(beta[worst_index]),
        "cascade": cascade,
        "alpha": alpha.tolist(),
        "beta": beta.tolist(),
        "tail_fraction": tail.tolist(),
        "verdict": (
            "resolved"
            if int(over.sum()) == 0
            else f"{int(over.sum())}/{n} trajectories under-resolved -- narrow the box"
        ),
    }


def make_plots(results: dict, config: Phase0Config, output: Path) -> None:
    data = config.data
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))

    convergence = results["convergence"]
    dts = [config.convergence_horizon / s for s in convergence["substeps"]]
    axes[0, 0].loglog(dts, convergence["errors"], "o-", label="measured")
    reference_curve = [
        convergence["errors"][-1] * (dt / dts[-1]) ** 2 for dt in dts
    ]
    axes[0, 0].loglog(dts, reference_curve, "k--", label="second order")
    axes[0, 0].set(xlabel="dt", ylabel="relative L2", title="Strang convergence")
    axes[0, 0].set_xticks(dts)
    axes[0, 0].set_xticklabels([f"{dt:.4g}" for dt in dts], rotation=45, fontsize=8)
    axes[0, 0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0, 0].legend()
    axes[0, 0].grid(True, which="both", alpha=0.3)

    invariants = results["long_run"]
    axes[0, 1].semilogy(invariants["times"], invariants["mass_drift"], label="mass")
    axes[0, 1].semilogy(invariants["times"], invariants["energy_drift"], label="energy")
    axes[0, 1].set(
        xlabel="t",
        ylabel="max relative drift",
        title=f"Invariants ({invariants['energy_classification']} energy)",
    )
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    screen = results["screen"]
    scatter = axes[1, 0].scatter(
        screen["alpha"],
        screen["beta"],
        c=[max(t, 1e-18) for t in screen["tail_fraction"]],
        norm=matplotlib.colors.LogNorm(),
        cmap="viridis",
        s=18,
    )
    figure.colorbar(scatter, ax=axes[1, 0], label="tail energy fraction")
    axes[1, 0].set(
        xlabel="alpha", ylabel="beta", title=f"Resolution screen: {screen['verdict']}"
    )

    floor = results["splitting_floor"]
    axes[1, 1].axis("off")
    k_wrap_low = wrap_wavenumber(data.alpha_range[1], data.dt)
    k_wrap_high = wrap_wavenumber(data.alpha_range[0], data.dt)
    summary = [
        f"eps_split exact rates   {floor['eps_split_exact_mean']:.3e}",
        f"eps_split fitted rates  {floor['eps_split_fitted_mean']:.3e}"
        f"  ({floor['fitted_over_exact']:.2f}x)",
        f"eps_split K2 per-mode    {floor['eps_split_k2_mean']:.3e}"
        f"  ({floor['k2_over_exact']:.2f}x)",
        f"fitted (c_kin, c_loc)   ({floor['fitted_kinetic_scale']:.6f},"
        f" {floor['fitted_local_scale']:.6f})",
        f"reference gap M32/M64   {floor['reference_convergence_gap_max']:.3e}",
        "",
        f"observed order (finest) {convergence['finest_order']:.4f}",
        f"max mass drift          {invariants['max_mass_drift']:.3e}",
        f"max energy drift        {invariants['max_energy_drift']:.3e}",
        "",
        f"k_wrap over alpha box   {k_wrap_low:.1f} .. {k_wrap_high:.1f}",
        f"k_nyquist               {data.max_wave_number}",
        f"training bandwidth      {data.initial_bandwidth}",
        "",
        f"cascade above k_train   {screen['cascade']['above_k_train'][-1]:.2e}",
        f"cascade above k_wrap    {screen['cascade']['above_k_wrap'][-1]:.2e}",
    ]
    axes[1, 1].text(
        0.0, 0.95, "\n".join(summary), family="monospace", va="top", fontsize=10
    )
    axes[1, 1].set_title("Phase 0 summary", loc="left")

    figure.suptitle(f"Phase 0 reference validation  [{results['config_hash']}]")
    figure.tight_layout()
    figure.savefig(output / "phase0_summary.png", dpi=150)
    plt.close(figure)


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="smaller sweep for a fast smoke run"
    )
    args = parser.parse_args()

    config = Phase0Config()
    if args.quick:
        config = dataclasses.replace(
            config,
            data=dataclasses.replace(config.data, steps=40),
            long_run_steps=200,
            screen_samples=16,
            convergence_substeps=(4, 8, 16, 32),
        )

    torch.set_default_dtype(torch.float64)
    generator = seed_everything(config.data.seed)
    identifier = config_hash(config)
    output = RESULTS_ROOT / f"phase0-{identifier}"
    (output / "plots").mkdir(parents=True, exist_ok=True)

    results = {"config_hash": identifier, "config": dataclasses.asdict(config)}
    print("measuring convergence order ...")
    results["convergence"] = measure_convergence_order(config, generator)
    print("measuring splitting floor ...")
    results["splitting_floor"] = measure_splitting_floor(config, generator)
    print("measuring long-run invariants ...")
    results["long_run"] = measure_long_run_invariants(config, generator)
    print("running resolution screen ...")
    results["screen"] = run_resolution_screen(config, generator)

    make_plots(results, config, output / "plots")
    (output / "metrics.json").write_text(json.dumps(results, indent=2))

    floor, convergence = results["splitting_floor"], results["convergence"]
    invariants, screen = results["long_run"], results["screen"]
    print(f"\n=== Phase 0 [{identifier}] ===")
    print(f"observed order (finest ratio) : {convergence['finest_order']:.4f}")
    print(f"eps_split (exact rates)       : {floor['eps_split_exact_mean']:.4e}")
    print(
        f"eps_split (fitted rates)      : {floor['eps_split_fitted_mean']:.4e}"
        f"  ({floor['fitted_over_exact']:.3f}x exact)"
    )
    print(
        f"fitted scales (kin, loc)      : "
        f"({floor['fitted_kinetic_scale']:.6f}, {floor['fitted_local_scale']:.6f})"
    )
    print(
        f"eps_split (K2 per-mode)       : {floor['eps_split_k2_mean']:.4e}"
        f"  ({floor['k2_over_exact']:.3f}x exact)"
    )
    print(f"reference gap / fitted floor  : {floor['gap_over_fitted_floor']:.4e}")
    print(f"max mass drift over long run  : {invariants['max_mass_drift']:.3e}")
    print(
        f"energy drift                  : {invariants['max_energy_drift']:.3e} "
        f"({invariants['energy_classification']})"
    )
    print(f"resolution screen             : {screen['verdict']}")
    cascade = screen["cascade"]
    print(
        f"cascade above k_train={cascade['k_train']}      : "
        + " -> ".join(
            f"{s}steps {v:.1e}" for s, v in zip(cascade["steps"], cascade["above_k_train"])
        )
    )
    print(
        f"cascade above k_wrap={cascade['k_wrap_min']:.1f}   : "
        + " -> ".join(
            f"{s}steps {v:.1e}" for s, v in zip(cascade["steps"], cascade["above_k_wrap"])
        )
    )
    print(f"written to                    : {output}")
    return results


if __name__ == "__main__":
    main()
