"""Paired, frozen C1 component interventions and trajectory diagnostics.

The coarse exact split step is a comparator, never the ground truth. All
reported vectors retain the IC axis; training seeds do not create new ICs.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import math

import numpy as np
import torch

from ..config import DataConfig
from ..data.generate import (sample_initial_conditions, sample_parameters,
                             sample_potentials, energy_fraction_above)
from ..domain import l2_mass, spectral_gradient
from ..equations.nls import nls_hamiltonian
from ..models.split_learned import DensityPhaseSplitStep, ExactSplitStep
from ..precision import widen_to_double
from ..solvers.split_step import SubsteppedReference


MODEL_NAMES = ("C1", "exactK_learnedL", "learnedK_exactL", "exact_split")
METRICS = ("state_error", "aligned_state_error", "phase_rms", "global_phase",
           "spectrum_error", "mass_drift", "energy_drift", "energy_error")


class ComponentSplitStep(ExactSplitStep):
    """Independent copy of the retained modules; no fitting or gauge correction."""

    def __init__(self, c1, *, exact_kinetic: bool, exact_local: bool):
        super().__init__(c1.domain)
        self.kinetic = None if exact_kinetic else deepcopy(c1.kinetic)
        self.local = None if exact_local else deepcopy(c1.local)

    def _kinetic_half(self, field, parameters, dt):
        if self.kinetic is None:
            return super()._kinetic_half(field, parameters, dt)
        rate = self.kinetic(parameters)
        return torch.fft.ifft(torch.fft.fft(field) * torch.exp(.5j * dt * rate))

    def local_phase(self, field, potential, alpha, beta):
        if self.local is None:
            return super().local_phase(field, potential, alpha, beta)
        return self.local(field.abs().square(), potential, alpha, beta)


def component_models(c1):
    if not isinstance(c1, DensityPhaseSplitStep):
        raise TypeError("The paired intervention requires a C1 checkpoint")
    models = {"C1": deepcopy(c1)}
    for name, exact_k, exact_l in (("exactK_learnedL", True, False),
                                   ("learnedK_exactL", False, True),
                                   ("exact_split", True, True)):
        models[name] = ComponentSplitStep(c1, exact_kinetic=exact_k, exact_local=exact_l)
    for model in models.values():
        model.eval().requires_grad_(False)
        # Explicit evaluation of dt transfer, including the original C1.
        model.supports_dt_transfer = True
    return models


@dataclass(frozen=True)
class ProbeCase:
    name: str
    config: DataConfig
    family: str = "random"
    cohorts: tuple[str, ...] = ("base",)
    long: bool = False


def probe_cases(data, bandwidths, *, include_g8=True):
    """Preserve the existing G-arm definitions; G8 is explicitly a new extension."""
    cases = [ProbeCase("G1-interpolation", data),
             ProbeCase("G2-extrapolation", replace(data, alpha_range=(.5, 1.5), beta_range=(-.5, .8)))]
    for tag, family, changes in (
        ("zero", "zero", {}), ("cosine", "cosine", {}),
        ("well", "gaussian_well", {}),
        ("short", "random", {"potential_correlation_length": .3}),
        ("strong", "random", {"potential_amplitude_range": (0., 1.)}),
    ):
        cases.append(ProbeCase(f"G3-potential-{tag}", replace(data, **changes), family))
    for bandwidth in bandwidths:
        if not 1 <= bandwidth <= data.grid_size // 2:
            raise ValueError("Bandwidth must lie in the resolvable spectrum")
        cases.append(ProbeCase(f"G4-bandwidth-{bandwidth}", replace(data, initial_bandwidth=bandwidth)))
    for factor in (.5, 1., 2.):
        cfg = replace(data, dt=data.dt * factor)
        cases.append(ProbeCase(f"G6a-multidt-{factor:g}", cfg, cohorts=("base", "G6a")))
        cases.append(ProbeCase(f"G6b-transfer-{factor:g}", cfg))
    cases.append(ProbeCase("G7-alpha-fixed", replace(data, alpha_range=(.9, .9)),
                           cohorts=("base", "G7-alpha-fixed")))
    if include_g8:
        cases.append(ProbeCase("G8-long-rollout-extension", data, long=True))
    cascade = replace(data, alpha_range=(.9, .9), beta_range=(.3, .3))
    cases.append(ProbeCase("G9-cascade-long", cascade, "zero", long=True))
    for bandwidth in bandwidths:
        cases.append(ProbeCase(f"G9-bandwidth-{bandwidth}",
                               replace(cascade, initial_bandwidth=bandwidth), "zero"))
    return cases


def sample_probe(case, seed, batch):
    """Same random draw across bandwidths and model seeds, independent across probes."""
    cfg, domain = case.config, case.config.domain
    gen = torch.Generator().manual_seed(int(seed))
    initial = sample_initial_conditions(domain, batch, cfg.initial_bandwidth, cfg.mass_range, gen)
    alpha, beta = sample_parameters(batch, cfg.alpha_range, cfg.beta_range, gen)
    potential = sample_potentials(domain, batch, cfg.potential_amplitude_range,
                                 cfg.potential_correlation_length, gen, family=case.family)
    inputs = (initial, potential, alpha, beta)
    digest = hashlib.sha256()
    for tensor in inputs:
        digest.update(tensor.numpy().tobytes())
    return inputs, digest.hexdigest()


def energy_scale(initial, potential, domain, alpha, beta):
    """Sum of absolute Hamiltonian terms avoids cancellation near zero total H."""
    rho = initial.abs().square()
    gradient = spectral_gradient(initial, domain).abs().square().sum(dim=1)
    return ((alpha.abs()[:, None] * gradient + potential.abs() * rho
             + .5 * beta.abs()[:, None] * rho.square()).sum(-1)
            * domain.cell_volume).clamp_min(1e-30)


def state_metrics(prediction, target, initial, potential, alpha, beta, domain):
    def rel(x, y):
        return (x - y).norm(dim=-1) / y.norm(dim=-1).clamp_min(1e-30)
    cross = (target.conj() * prediction).sum(-1)
    phase = cross.angle()
    aligned = prediction * torch.exp(-1j * phase[:, None])
    weights = target.abs().square()
    # Exclude nodes where either field has undefined phase, and report coverage.
    mask = ((weights > 1e-12 * weights.amax(-1, keepdim=True)) &
            (prediction.abs().square() > 1e-12 * weights.amax(-1, keepdim=True)))
    phase_weight = weights * mask
    local_phase = (prediction * target.conj()).angle()
    phase_rms = ((local_phase.square() * phase_weight).sum(-1) /
                 phase_weight.sum(-1).clamp_min(1e-30)).sqrt()
    phase_rms = torch.where(phase_weight.sum(-1) > 0, phase_rms, torch.nan)
    initial_energy = nls_hamiltonian(initial, potential, domain, alpha, beta)
    pred_energy = nls_hamiltonian(prediction, potential, domain, alpha, beta)
    true_energy = nls_hamiltonian(target, potential, domain, alpha, beta)
    scale = energy_scale(initial, potential, domain, alpha, beta)
    ppower = torch.fft.fft(prediction).abs().square()
    tpower = torch.fft.fft(target).abs().square()
    return {
        "state_error": rel(prediction, target), "aligned_state_error": rel(aligned, target),
        "phase_rms": phase_rms, "global_phase": phase.abs(),
        "phase_coverage": phase_weight.sum(-1) / weights.sum(-1).clamp_min(1e-30),
        "spectrum_error": (ppower - tpower).abs().sum(-1) / tpower.sum(-1).clamp_min(1e-30),
        "mass_drift": (l2_mass(prediction, domain) / l2_mass(initial, domain) - 1).abs(),
        "energy_drift": (pred_energy - initial_energy).abs() / scale,
        "energy_error": (pred_energy - true_energy).abs() / scale,
        "energy_drift_absolute": (pred_energy - initial_energy).abs(),
        "mass": l2_mass(prediction, domain), "energy": pred_energy,
    }


def sampled_steps(horizon, stride):
    return sorted({0, 1, horizon, *range(stride, horizon + 1, stride)})


@torch.no_grad()
def reference_frames(inputs, cfg, *, steps, stride, substeps):
    state, potential, alpha, beta = inputs
    solver = SubsteppedReference(cfg.domain, substeps)
    frames = {0: state.clone()}
    selected = set(sampled_steps(steps, stride))
    for step in range(1, steps + 1):
        state = solver(state, potential, alpha, beta, cfg.dt)
        if not torch.isfinite(state).all():
            raise RuntimeError(f"Nonfinite reference at step {step}")
        if step in selected:
            frames[step] = state.clone()
    return frames


@torch.no_grad()
def measure_rollout(model, inputs, cfg, truth, *, training_bandwidth):
    state, potential, alpha, beta = inputs
    initial = state.clone()
    alive = torch.ones(len(initial), dtype=torch.bool, device=initial.device)
    failures = [None] * len(initial)
    records = []
    for step in range(max(truth) + 1):
        if step:
            state = model(state, potential, alpha, beta, cfg.dt)
            finite = torch.isfinite(state).all(-1)
            for index in torch.where(alive & ~finite)[0].tolist():
                failures[index] = step
            alive &= finite
            state = torch.where(alive[:, None], state, torch.zeros_like(state))
        if step in truth:
            metrics = state_metrics(state, truth[step], initial, potential, alpha, beta, cfg.domain)
            metrics["above_training_band"] = energy_fraction_above(state, cfg.domain, training_bandwidth)
            metrics["nyquist_tail"] = energy_fraction_above(state, cfg.domain, .75 * (cfg.grid_size // 2))
            values = {name: torch.where(alive, value, torch.nan).cpu().tolist()
                      for name, value in metrics.items()}
            # Power AND complex-error spectrum, retaining individual ICs.
            values["power_spectrum"] = torch.where(alive[:, None],
                torch.fft.fft(state, norm="ortho").abs().square(), torch.nan).cpu().tolist()
            values["error_spectrum"] = torch.where(alive[:, None],
                torch.fft.fft(state - truth[step], norm="ortho").abs().square(), torch.nan).cpu().tolist()
            records.append({"step": step, "time": step * cfg.dt, **values})
    return {"records": records, "failed_at": failures}


@torch.no_grad()
def kinetic_dispersion(c1, cfg):
    """Raw generator readout, never a truth-assisted phase lift.

    Constant kinetic/local offsets can cancel. Store both raw omega and
    omega(k)-omega(0) and quantify the effective plane-wave map separately.
    """
    model = widen_to_double(c1, device="cpu").eval()
    k = torch.fft.fftfreq(cfg.grid_size, d=1 / cfg.grid_size).double()
    order = k.argsort()
    alphas = torch.linspace(*cfg.alpha_range, 9, dtype=torch.float64)
    betas = sorted(set([cfg.beta_range[0], sum(cfg.beta_range) / 2, cfg.beta_range[1]]))
    amplitude = math.sqrt(sum(cfg.mass_range) / (2 * cfg.domain.volume))
    curves = []
    for beta in betas:
        parameters = torch.stack([alphas, torch.full_like(alphas, beta)], -1)
        omega = -model.kinetic(parameters)
        exact = alphas[:, None] * k.square()
        rho = torch.full((len(alphas), cfg.grid_size), amplitude**2, dtype=torch.float64)
        local = model.local(rho, torch.zeros_like(rho), alphas, parameters[:, 1])
        exact_local = beta * amplitude**2
        total = {"C1": omega - local, "exactK_learnedL": exact - local,
                 "learnedK_exactL": omega - exact_local, "exact_split": exact - exact_local}
        truth = exact - exact_local
        curves.append({"beta": beta, "omega": omega[:, order].tolist(),
                       "omega_centered": (omega - omega[:, :1])[:, order].tolist(),
                       "exact": exact[:, order].tolist(),
                       "kinetic_offset": omega[:, 0].tolist(),
                       "local_offset": (local[:, 0] - exact_local).tolist(),
                       "map_residual": {name: (torch.exp(-1j * cfg.dt * value) -
                           torch.exp(-1j * cfg.dt * truth)).abs()[:, order].tolist()
                           for name, value in total.items()},
                       "alpha_secant_error": None if cfg.alpha_range[0] == cfg.alpha_range[1] else
                           ((omega[1:] - omega[:-1]) / torch.diff(alphas)[:, None]
                            - k.square())[..., order].tolist()})
    return {"k": k[order].tolist(), "alphas": alphas.tolist(), "curves": curves,
            "training_bandwidth": cfg.initial_bandwidth, "dt": cfg.dt,
            "note": "G5a raw kinetic generator; G5b in-range alpha secants. No phase unwrapping."}


def crossed_interval(values, *, draws=2000, seed=731):
    """Crossed training/probe bootstrap with IC resampling shared across models.

    Input is training_seed x probe_seed x IC (already paired model differences
    or log ratios). Resample training seeds and probe clusters independently;
    resample ICs once per selected probe and use them for every training seed.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or not np.isfinite(values).all():
        return {"mean": None, "low": None, "high": None, "status": "incomplete-or-nonfinite"}
    rng = np.random.default_rng(seed)
    nt, npb, ni = values.shape
    means = []
    for _ in range(draws):
        ts = rng.integers(nt, size=nt)
        ps = rng.integers(npb, size=npb)
        means.append(np.mean([values[ts[:, None], p, rng.integers(ni, size=ni)[None, :]].mean() for p in ps]))
    low, high = np.quantile(means, [.025, .975])
    return {"mean": float(values.mean()), "low": float(low), "high": float(high),
            "status": "descriptive-bootstrap" if nt > 1 and npb > 1 else "insufficient-seed-replication"}
