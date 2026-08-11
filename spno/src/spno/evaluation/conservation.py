"""Invariant drift over a rollout, and the secular-versus-bounded classification.

This carries the falsifiable prediction of the symplecticity theorem.  A learned split
step with a rho-only local phase is an exact symplectic integrator of a *learned*
Hamiltonian, so its energy error should be governed by ``||H_theta - H_true||`` -- a
fixed model-error term, therefore **bounded and oscillatory**.  A model with no such
structure has nothing preventing **secular** growth.

Phases 2-3 measured secular energy drift for the FNO and the mass-projected FNO in 9/9
seed-model runs, so there is a real baseline to falsify against.

The classification is a fitted log-log slope with a confidence interval, not an
eyeball: "bounded" and "secular" differ by whether drift keeps growing with the
horizon, and that is a statement about a trend.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..domain import PeriodicDomain, l2_mass
from ..equations.nls import nls_hamiltonian

Tensor = torch.Tensor


@dataclass
class DriftTrend:
    slope: float
    slope_stderr: float
    classification: str
    n_points: int

    def as_dict(self) -> dict:
        return {
            "slope": self.slope,
            "slope_stderr": self.slope_stderr,
            "classification": self.classification,
            "n_points": self.n_points,
        }


def classify_drift(
    steps: list[int], drift: list[float], *, bounded_slope: float = 0.25
) -> DriftTrend:
    """Fit ``log(drift) ~ slope * log(step)`` and classify by the slope.

    A conserved-up-to-model-error quantity plateaus (slope ~ 0).  Linear accumulation
    gives slope ~ 1.  The cut at 0.25 is deliberately generous to "bounded", so a
    borderline case is reported as secular rather than flattering the structured model.

    Points at or below zero drift are dropped: they are exact conservation, which has no
    slope, and a log of them would be undefined.
    """

    pairs = [(s, d) for s, d in zip(steps, drift) if s > 0 and d > 0]
    if len(pairs) < 3:
        return DriftTrend(float("nan"), float("nan"), "insufficient-data", len(pairs))

    xs = [math.log(s) for s, _ in pairs]
    ys = [math.log(d) for _, d in pairs]
    n = len(pairs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        return DriftTrend(float("nan"), float("nan"), "insufficient-data", n)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    if n > 2:
        variance = sum(r**2 for r in residuals) / (n - 2)
        stderr = math.sqrt(variance / sxx)
    else:
        stderr = float("nan")

    classification = "bounded" if slope < bounded_slope else "secular"
    return DriftTrend(slope, stderr, classification, n)


@dataclass
class ConservationMetrics:
    steps: list[int]
    mass_drift: list[float]
    energy_drift: list[float]
    mass_trend: DriftTrend
    energy_trend: DriftTrend

    def as_dict(self) -> dict:
        return {
            "steps": self.steps,
            "mass_drift": self.mass_drift,
            "energy_drift": self.energy_drift,
            "mass_trend": self.mass_trend.as_dict(),
            "energy_trend": self.energy_trend.as_dict(),
        }


@torch.no_grad()
def evaluate_conservation(
    model,
    domain: PeriodicDomain,
    initial: Tensor,
    potential: Tensor,
    alpha: Tensor,
    beta: Tensor,
    dt: float,
    *,
    steps: int = 200,
    stride: int = 10,
) -> ConservationMetrics:
    """Track mass and energy along a rollout and fit their trends."""

    reference_mass = l2_mass(initial, domain)
    reference_energy = nls_hamiltonian(initial, potential, domain, alpha, beta)

    recorded, mass_drift, energy_drift = [], [], []
    state = initial
    for step in range(1, steps + 1):
        state = model(state, potential, alpha, beta, dt)
        if not torch.isfinite(state).all():
            break
        if step % stride == 0:
            recorded.append(step)
            mass_drift.append(
                float(torch.abs(l2_mass(state, domain) / reference_mass - 1).mean())
            )
            energy = nls_hamiltonian(state, potential, domain, alpha, beta)
            energy_drift.append(
                float(
                    torch.abs(
                        (energy - reference_energy)
                        / reference_energy.abs().clamp_min(1e-12)
                    ).mean()
                )
            )

    return ConservationMetrics(
        steps=recorded,
        mass_drift=mass_drift,
        energy_drift=energy_drift,
        mass_trend=classify_drift(recorded, mass_drift),
        energy_trend=classify_drift(recorded, energy_drift),
    )
