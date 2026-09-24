"""Phase 7 follow-up probes: the CN oracle, phase-aligned rollout, long-horizon invariants.

Three measurements the Phase 7a table cannot make on its own:

* **The Crank--Nicolson oracle.**  The PDE penalty is the midpoint residual, so the model
  it pulls toward is the *exact* CN step.  Solving that step and scoring it against the
  substepped reference gives the error floor a large-lambda model is being dragged onto.
  If the lambda >= 1 arms sit at this floor, the degradation is the residual targeting
  different discrete dynamics, not the physics loss failing to train.
* **Phase-aligned rollout error.**  Raw relative L2 charges a pure global-phase drift as
  full error.  Removing the best global phase ``argmax_phi Re<target, e^{i phi} pred>``
  separates "wrong shape" from "right shape, clock running slightly fast".
* **Long-horizon invariants.**  Mass and energy need no reference trajectory, so the
  horizon is not capped by the stored frames.  The symplecticity prediction -- bounded,
  oscillatory energy error for the split-step family, secular growth otherwise -- is a
  statement about the late-time trend, so the drift slope is fitted beyond ``fit_from``
  rather than over the transient where every model's drift starts at zero.

Every probe here is an evaluator: callers pass float64 models on CPU (``widen_to_double``)
so float32 roundoff never masks an architectural violation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..domain import PeriodicDomain, batch_parameter, l2_mass
from ..equations.nls import nls_hamiltonian
from ..losses.relative_l2 import relative_l2_per_sample
from .conservation import classify_drift

Tensor = torch.Tensor


class CrankNicolsonStep(nn.Module):
    """The exact solution of the midpoint residual in ``losses/pde_residual.py``.

    Rearranging ``R = 0`` in Fourier space, with ``g = (beta rho_bar - V) psi_bar``:

        (1 + i dt alpha k^2 / 2) psi_{n+1}^ = (1 - i dt alpha k^2 / 2) psi_n^ + i dt g^

    The nonlinear term depends on ``psi_{n+1}``, so the step is a fixed-point iteration
    to roundoff.  It is discretely mass-conserving (the Delfour--Fortin--Payre average
    makes the iteration's fixed point unitary), which is exactly the confound Phase 7a
    reports.  Same call signature as every ``StepOperator``.
    """

    def __init__(self, domain: PeriodicDomain, *, tol: float = 1e-13, max_iter: int = 200) -> None:
        super().__init__()
        self.domain = domain
        self.tol = tol
        self.max_iter = max_iter

    def forward(self, field: Tensor, potential: Tensor, alpha, beta, dt: float) -> Tensor:
        self.domain.validate_batched_field(field)
        if potential.shape != field.shape:
            raise ValueError("field and potential must have shape (batch, *domain.shape)")
        if not field.is_complex() or potential.is_complex():
            raise ValueError("field must be complex and potential must be real")
        axes = self.domain.spatial_axes
        alpha_grid = batch_parameter(alpha, field.shape[0], self.domain, field, "alpha")
        beta_grid = batch_parameter(beta, field.shape[0], self.domain, field, "beta")
        k_squared = self.domain.wave_number_squared(device=field.device, dtype=field.real.dtype)
        half = 0.5 * float(dt) * alpha_grid * k_squared
        explicit = torch.fft.fftn(field, dim=axes) * (1 - 1j * half)
        implicit = 1 + 1j * half
        # Relative tolerance no tighter than the working precision can resolve.
        tol = max(self.tol, 64 * torch.finfo(field.real.dtype).eps)
        state = field
        for _ in range(self.max_iter):
            average = 0.5 * (state + field)
            density = 0.5 * (state.abs().square() + field.abs().square())
            forcing = (beta_grid * density - potential) * average
            update = torch.fft.ifftn(
                (explicit + 1j * float(dt) * torch.fft.fftn(forcing, dim=axes)) / implicit,
                dim=axes,
            )
            change = float((update - state).abs().amax() / update.abs().amax().clamp_min(1e-300))
            state = update
            if change <= tol:
                return state
        raise RuntimeError(
            f"Crank-Nicolson fixed point did not converge in {self.max_iter} iterations "
            f"(last relative change {change:.2e}); dt * beta * |psi|^2 is too large"
        )


@torch.no_grad()
def reference_one_step(operator, domain: PeriodicDomain, trajectories: Tensor, potential: Tensor,
                       alpha: Tensor, beta: Tensor, dt: float, *, chunk: int = 16,
                       against=None) -> float:
    """Mean one-step relative L2 over every consecutive frame pair.

    The same statistic as ``train.evaluate_one_step`` (batch mean of the per-sample
    relative L2 over all test pairs), but in whatever precision the inputs carry.  With
    ``against`` set, the target is that operator's step from the same input instead of
    the stored next frame -- e.g. a model's distance to the CN step, which is what the
    penalty actually pulls toward.
    """

    total, count = 0.0, 0
    pairs = trajectories.shape[1] - 1
    for start in range(0, trajectories.shape[0], chunk):
        block = trajectories[start:start + chunk]
        n = block.shape[0]
        field = block[:, :-1].reshape(n * pairs, *block.shape[2:])
        repeat = lambda t: t[start:start + chunk].repeat_interleave(pairs, dim=0)
        arguments = (repeat(potential), repeat(alpha), repeat(beta), dt)
        if against is None:
            target = block[:, 1:].reshape(n * pairs, *block.shape[2:])
        else:
            target = against(field, *arguments)
        prediction = operator(field, *arguments)
        error = relative_l2_per_sample(prediction, target, domain)
        total += float(error.sum())
        count += error.numel()
    return total / max(count, 1)


def global_phase(prediction: Tensor, target: Tensor, domain: PeriodicDomain) -> Tensor:
    """Per-sample phase ``phi`` maximizing ``Re <target, e^{-i phi} prediction>``."""

    return (target.conj() * prediction).sum(dim=domain.spatial_axes).angle()


@torch.no_grad()
def aligned_rollout(model, domain: PeriodicDomain, initial: Tensor, trajectories: Tensor,
                    potential: Tensor, alpha: Tensor, beta: Tensor, dt: float, *,
                    checkpoints: tuple[int, ...] = (1, 10, 20, 50, 100, 200)) -> dict:
    """Raw and global-phase-aligned rollout error against the stored frames.

    ``aligned_error <= relative_error`` always; the gap is the part of the raw error a
    single global phase explains.  ``global_phase`` is the mean ``|phi|`` in radians.
    """

    horizon = min(max(checkpoints), trajectories.shape[1] - 1)
    wanted = {step for step in checkpoints if step <= horizon}
    result = {"steps": [], "relative_error": [], "aligned_error": [], "global_phase": [],
              "diverged_at": None}
    state = initial
    for step in range(1, horizon + 1):
        state = model(state, potential, alpha, beta, dt)
        if not torch.isfinite(state).all():
            result["diverged_at"] = step
            break
        if step in wanted:
            target = trajectories[:, step]
            phase = global_phase(state, target, domain)
            shape = (-1,) + (1,) * len(domain.spatial_axes)
            aligned = state * torch.exp(-1j * phase).reshape(shape)
            result["steps"].append(step)
            result["relative_error"].append(float(relative_l2_per_sample(state, target, domain).mean()))
            result["aligned_error"].append(float(relative_l2_per_sample(aligned, target, domain).mean()))
            result["global_phase"].append(float(phase.abs().mean()))
    return result


def record_steps(steps: int, n_records: int = 60) -> list[int]:
    """Roughly log-spaced step indices from 1 to ``steps`` inclusive."""

    if steps < 1:
        raise ValueError("steps must be positive")
    count = max(2, n_records)
    return sorted({max(1, min(steps, round(math.exp(math.log(steps) * i / (count - 1)))))
                   for i in range(count)} | {steps})


@torch.no_grad()
def long_horizon_invariants(model, domain: PeriodicDomain, initial: Tensor, potential: Tensor,
                            alpha: Tensor, beta: Tensor, dt: float, *, steps: int = 5000,
                            n_records: int = 60, fit_from: int = 100,
                            always: tuple[int, ...] = (10, 100, 1000),
                            roundoff_floor: float = 1e-10) -> dict:
    """Mass and energy drift at log-spaced steps, with late-time trend fits.

    Drifts use the same definitions as ``evaluate_rollout`` (mean over samples of
    ``|M/M0 - 1|`` and ``|H - H0| / |H0|``), so step 100 here is directly comparable to
    the Phase 7 table (step 100 is always recorded).  A non-finite state stops the
    rollout and sets ``diverged_at``.
    """

    recorded = set(record_steps(steps, n_records)) | {s for s in always if 1 <= s <= steps}
    reference_mass = l2_mass(initial, domain)
    reference_energy = nls_hamiltonian(initial, potential, domain, alpha, beta)
    out = {"steps": [], "mass_drift": [], "energy_drift": [], "energy_drift_max": [],
           "diverged_at": None, "requested_steps": steps}
    state = initial
    for step in range(1, steps + 1):
        state = model(state, potential, alpha, beta, dt)
        if not torch.isfinite(state).all():
            out["diverged_at"] = step
            break
        if step in recorded:
            energy = nls_hamiltonian(state, potential, domain, alpha, beta)
            relative = ((energy - reference_energy) / reference_energy.abs().clamp_min(1e-12)).abs()
            mass = (l2_mass(state, domain) / reference_mass - 1).abs()
            if not (torch.isfinite(relative).all() and torch.isfinite(mass).all()):
                out["diverged_at"] = step
                break
            out["steps"].append(step)
            out["mass_drift"].append(float(mass.mean()))
            out["energy_drift"].append(float(relative.mean()))
            out["energy_drift_max"].append(float(relative.max()))
    fit_from = min(fit_from, max(1, steps // 10))
    late = [i for i, s in enumerate(out["steps"]) if s >= fit_from]
    out["fit_from"] = fit_from
    for key in ("mass", "energy"):
        series = [out[f"{key}_drift"][i] for i in late]
        trend = classify_drift([out["steps"][i] for i in late], series).as_dict()
        # float64 roundoff accumulates linearly, so an exactly conserving scheme fits a
        # slope near 1; below the floor the slope describes arithmetic, not dynamics.
        if series and max(series) < roundoff_floor:
            trend["classification"] = "roundoff"
        out[f"{key}_trend"] = trend
    return out
