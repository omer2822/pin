"""Phase 7 probes: CN oracle, phase-aligned rollout, long-horizon invariants."""

from __future__ import annotations

import math

import pytest
import torch

from spno.domain import PeriodicDomain, l2_mass
from spno.equations.nls import exact_dispersion, plane_wave
from spno.evaluation.phase7_probes import (
    CrankNicolsonStep,
    aligned_rollout,
    long_horizon_invariants,
    record_steps,
    reference_one_step,
)
from spno.losses.pde_residual import crank_nicolson_frequency, midpoint_residual
from spno.solvers.split_step import SplitStepNLSOperator

N, DT = 64, 0.01


def _smooth_batch(domain, batch=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    spectrum = torch.zeros(batch, N, dtype=torch.complex128)
    modes = torch.randn(batch, 9, 2, generator=generator, dtype=torch.float64)
    spectrum[:, :9] = torch.complex(modes[..., 0], modes[..., 1])
    field = torch.fft.ifft(spectrum, dim=-1) * N / 3
    potential = 0.3 * torch.cos(domain.mesh(dtype=torch.float64)[0]).expand(batch, N).clone()
    alpha = torch.linspace(0.5, 1.5, batch, dtype=torch.float64)
    beta = torch.linspace(-1.0, 1.0, batch, dtype=torch.float64)
    return field, potential, alpha, beta


def test_cn_step_solves_the_midpoint_residual_and_conserves_mass():
    domain = PeriodicDomain.periodic_1d(N)
    field, potential, alpha, beta = _smooth_batch(domain)
    step = CrankNicolsonStep(domain)(field, potential, alpha, beta, DT)
    residual = midpoint_residual(field, step, potential, domain, alpha, beta, DT)
    scale = field.abs().amax() / DT
    assert float(residual.abs().amax() / scale) < 1e-12
    drift = (l2_mass(step, domain) / l2_mass(field, domain) - 1).abs()
    assert float(drift.max()) < 1e-13


def test_cn_step_advances_a_nonlinear_plane_wave_by_the_cayley_factor():
    domain = PeriodicDomain.periodic_1d(N)
    alpha, beta, v0, amplitude, k = 0.9, 0.3, 0.2, 0.5642, 5
    field = plane_wave(domain, (k,), amplitude=amplitude).unsqueeze(0)
    potential = torch.full_like(field.real, v0)
    omega = float(exact_dispersion((k,), alpha=alpha, beta=beta, amplitude=amplitude,
                                   potential_constant=v0))
    theta = crank_nicolson_frequency(omega, DT) * DT
    expected = field * complex(math.cos(theta), -math.sin(theta))
    step = CrankNicolsonStep(domain)(field, potential, torch.tensor([alpha], dtype=torch.float64),
                                     torch.tensor([beta], dtype=torch.float64), DT)
    assert float((step - expected).abs().amax()) < 1e-12


def test_cn_step_refuses_to_return_an_unconverged_iterate():
    domain = PeriodicDomain.periodic_1d(N)
    field, potential, alpha, beta = _smooth_batch(domain)
    with pytest.raises(RuntimeError, match="did not converge"):
        CrankNicolsonStep(domain, max_iter=2)(field, potential, alpha, beta, DT)


def test_reference_one_step_is_zero_for_the_generating_operator():
    domain = PeriodicDomain.periodic_1d(N)
    field, potential, alpha, beta = _smooth_batch(domain, batch=3)
    solver = SplitStepNLSOperator(domain)
    frames = [field]
    for _ in range(5):
        frames.append(solver(frames[-1], potential, alpha, beta, DT))
    trajectories = torch.stack(frames, dim=1)
    assert reference_one_step(solver, domain, trajectories, potential, alpha, beta, DT,
                              chunk=2) < 1e-14
    cn = reference_one_step(CrankNicolsonStep(domain), domain, trajectories, potential,
                            alpha, beta, DT)
    assert 0 < cn < 1e-2
    cn_step = CrankNicolsonStep(domain)
    assert reference_one_step(cn_step, domain, trajectories, potential, alpha, beta, DT,
                              against=cn_step) < 1e-14
    assert reference_one_step(solver, domain, trajectories, potential, alpha, beta, DT,
                              against=cn_step) == pytest.approx(cn, rel=0.2)


class _PhaseSlip(torch.nn.Module):
    """The exact solver followed by a constant global phase kick per step."""

    def __init__(self, domain, kick):
        super().__init__()
        self.solver, self.kick = SplitStepNLSOperator(domain), kick

    def forward(self, field, potential, alpha, beta, dt):
        return self.solver(field, potential, alpha, beta, dt) * complex(math.cos(self.kick),
                                                                        math.sin(self.kick))


def test_aligned_rollout_removes_a_pure_global_phase_drift():
    domain = PeriodicDomain.periodic_1d(N)
    field, potential, alpha, beta = _smooth_batch(domain)
    solver = SplitStepNLSOperator(domain)
    frames = [field]
    for _ in range(20):
        frames.append(solver(frames[-1], potential, alpha, beta, DT))
    trajectories = torch.stack(frames, dim=1)
    result = aligned_rollout(_PhaseSlip(domain, 0.03), domain, field, trajectories, potential,
                             alpha, beta, DT, checkpoints=(1, 10, 20))
    assert result["steps"] == [1, 10, 20]
    assert result["relative_error"][-1] > 0.5
    assert max(result["aligned_error"]) < 1e-12
    assert result["global_phase"] == pytest.approx([0.03, 0.3, 0.6], rel=1e-9)
    assert result["diverged_at"] is None


def test_record_steps_are_log_spaced_and_hit_both_ends():
    steps = record_steps(5000, 40)
    assert steps[0] == 1 and steps[-1] == 5000
    assert steps == sorted(set(steps)) and len(steps) <= 40
    assert record_steps(3, 60) == [1, 2, 3]


class _Leaky(torch.nn.Module):
    def forward(self, field, potential, alpha, beta, dt):
        return field * 1.001


class _Explodes(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, field, potential, alpha, beta, dt):
        self.calls += 1
        return field * float("nan") if self.calls == 7 else field


def test_long_horizon_separates_exact_conservation_secular_growth_and_divergence():
    domain = PeriodicDomain.periodic_1d(N)
    field, potential, alpha, beta = _smooth_batch(domain)
    exact = long_horizon_invariants(SplitStepNLSOperator(domain), domain, field, potential,
                                    alpha, beta, DT, steps=400, n_records=30, fit_from=50)
    assert max(exact["mass_drift"]) < 1e-13
    assert exact["mass_trend"]["classification"] == "roundoff"
    assert 100 in exact["steps"]
    assert exact["energy_trend"]["classification"] == "bounded"
    assert exact["steps"][-1] == 400 and exact["diverged_at"] is None

    leaky = long_horizon_invariants(_Leaky(), domain, field, potential, alpha, beta, DT,
                                    steps=400, n_records=30, fit_from=50)
    assert leaky["mass_trend"]["classification"] == "secular"
    assert leaky["mass_trend"]["slope"] > 0.9

    broken = long_horizon_invariants(_Explodes(), domain, field, potential, alpha, beta, DT,
                                     steps=50, n_records=50)
    assert broken["diverged_at"] == 7 and broken["steps"][-1] < 7
