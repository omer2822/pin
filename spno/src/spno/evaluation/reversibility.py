"""Time-reversal error: evolve forward by T, back by T, compare with the start.

Only meaningful for models where ``dt`` genuinely enters the map as a multiplier.  An
FNO ignores its ``dt`` argument, so stepping it at ``-dt`` re-applies the *forward* map
and any number produced that way measures nothing -- :func:`evaluate_reversibility`
refuses rather than returning a misleading figure.

Phase 4 established that the interesting distinction is not reversible/irreversible but
**exact versus second-order**: rho-only phase sharing gives machine precision
independent of ``dt``, while a phase that reads ``Re psi``/``Im psi`` misses by
``O(dt^2)``.  :func:`reversibility_order` measures which regime a model is in instead
of thresholding, since at small ``dt`` an O(dt^2) violation is small enough to pass for
exactness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..domain import PeriodicDomain, l2_mass
from ..losses.relative_l2 import relative_l2_per_sample

Tensor = torch.Tensor


@dataclass
class ReversibilityMetrics:
    steps: int
    dt: float
    relative_error: float
    mass_drift: float
    regime: str
    order: float | None

    def as_dict(self) -> dict:
        return {
            "steps": self.steps,
            "dt": self.dt,
            "relative_error": self.relative_error,
            "mass_drift": self.mass_drift,
            "regime": self.regime,
            "order": self.order,
        }


@torch.no_grad()
def reversibility_error(
    model,
    domain: PeriodicDomain,
    initial: Tensor,
    potential: Tensor,
    alpha: Tensor,
    beta: Tensor,
    dt: float,
    steps: int,
) -> tuple[float, float]:
    """Forward ``steps``, backward ``steps``; return (relative error, mass drift)."""

    if not getattr(model, "supports_time_reversal", False):
        raise ValueError(
            f"{type(model).__name__} does not support time reversal: its dt argument is "
            "not a multiplier on a learned rate, so stepping at -dt re-applies the "
            "forward map and the resulting number would be an artefact."
        )
    state = initial
    for _ in range(steps):
        state = model(state, potential, alpha, beta, dt)
    for _ in range(steps):
        state = model(state, potential, alpha, beta, -dt)
    error = float(relative_l2_per_sample(state, initial, domain).max())
    drift = float(
        torch.abs(l2_mass(state, domain) / l2_mass(initial, domain) - 1).max()
    )
    return error, drift


@torch.no_grad()
def reversibility_order(
    model,
    domain: PeriodicDomain,
    initial: Tensor,
    potential: Tensor,
    alpha: Tensor,
    beta: Tensor,
    *,
    dts: tuple[float, ...] = (0.01, 0.02, 0.04),
    steps: int = 1,
    exact_threshold: float = 1e-12,
) -> tuple[str, float | None, list[float]]:
    """Classify the reversal regime by how the error scales with ``dt``.

    Returns ``(regime, order, errors)`` where regime is ``"exact"`` (machine precision
    and flat in dt), ``"approximate"`` (a clean power law), or ``"inconsistent"``.

    Note this varies ``dt`` away from the trained value on purpose, so it is a probe of
    the *architecture*, not a claim about prediction accuracy at those step sizes.
    """

    original_dt = getattr(model, "trained_dt", None)
    errors = []
    try:
        for dt in dts:
            if original_dt is not None:
                model.trained_dt = dt
            error, _ = reversibility_error(
                model, domain, initial, potential, alpha, beta, dt, steps
            )
            errors.append(error)
    finally:
        if original_dt is not None:
            model.trained_dt = original_dt

    if max(errors) < exact_threshold and max(errors) <= 4 * max(min(errors), 1e-300):
        return "exact", None, errors
    orders = [
        math.log2(errors[i + 1] / max(errors[i], 1e-300)) for i in range(len(errors) - 1)
    ]
    spread = max(orders) - min(orders)
    if spread < 0.25:
        return "approximate", sum(orders) / len(orders), errors
    return "inconsistent", sum(orders) / len(orders), errors


@torch.no_grad()
def evaluate_reversibility(
    model,
    domain: PeriodicDomain,
    initial: Tensor,
    potential: Tensor,
    alpha: Tensor,
    beta: Tensor,
    dt: float,
    *,
    steps: int = 20,
    measure_order: bool = True,
) -> ReversibilityMetrics:
    error, drift = reversibility_error(
        model, domain, initial, potential, alpha, beta, dt, steps
    )
    regime, order = "not-measured", None
    if measure_order:
        regime, order, _ = reversibility_order(
            model, domain, initial, potential, alpha, beta
        )
    return ReversibilityMetrics(
        steps=steps, dt=dt, relative_error=error, mass_drift=drift,
        regime=regime, order=order,
    )
