"""Autoregressive rollout: error growth and invariant drift over long horizons.

The central measurement of the study.  One-step error says how well a model fits the
map; rollout says whether iterating it stays on the solution manifold.  Divergence is
recorded as a metric in its own right rather than being allowed to poison the means --
a model that produces NaN at step 37 has told you something specific, and averaging
that into an error is a way of hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..domain import PeriodicDomain, l2_mass
from ..equations.nls import nls_hamiltonian
from ..losses.relative_l2 import relative_l2_per_sample

Tensor = torch.Tensor


@dataclass
class RolloutMetrics:
    steps: list[int]
    relative_error: list[float]
    mass_drift: list[float]
    energy_drift: list[float]
    diverged_at: int | None
    energy_classification: str

    def as_dict(self) -> dict:
        return {
            "steps": self.steps,
            "relative_error": self.relative_error,
            "mass_drift": self.mass_drift,
            "energy_drift": self.energy_drift,
            "diverged_at": self.diverged_at,
            "energy_classification": self.energy_classification,
        }

    def error_at(self, step: int) -> float:
        return self.relative_error[self.steps.index(step)]

    def mass_drift_at(self, step: int) -> float:
        return self.mass_drift[self.steps.index(step)]


@torch.no_grad()
def evaluate_rollout(
    model,
    domain: PeriodicDomain,
    initial: Tensor,
    trajectories: Tensor,
    potential: Tensor,
    alpha: Tensor,
    beta: Tensor,
    dt: float,
    *,
    checkpoints: tuple[int, ...] = (1, 10, 20, 50, 100, 200),
) -> RolloutMetrics:
    """Roll ``model`` forward and compare against stored reference frames.

    ``trajectories`` has shape ``(batch, frames, *shape)`` with frame 0 equal to
    ``initial``, exactly as the shards store it.
    """

    domain.validate_batched_field(initial)
    horizon = min(max(checkpoints), trajectories.shape[1] - 1)
    checkpoints = tuple(step for step in checkpoints if step <= horizon)

    reference_mass = l2_mass(initial, domain)
    reference_energy = nls_hamiltonian(initial, potential, domain, alpha, beta)

    steps, errors, mass_drift, energy_drift = [], [], [], []
    diverged_at: int | None = None
    state = initial
    for step in range(1, horizon + 1):
        state = model(state, potential, alpha, beta, dt)
        if not torch.isfinite(state).all():
            diverged_at = step
            break
        if step in checkpoints:
            target = trajectories[:, step]
            steps.append(step)
            errors.append(
                float(relative_l2_per_sample(state, target, domain).mean())
            )
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

    classification = _classify(energy_drift)
    return RolloutMetrics(
        steps=steps,
        relative_error=errors,
        mass_drift=mass_drift,
        energy_drift=energy_drift,
        diverged_at=diverged_at,
        energy_classification=classification,
    )


def _classify(energy_drift: list[float]) -> str:
    """Secular drift grows with the horizon; bounded drift oscillates around a level.

    The distinction is the falsifiable prediction of the symplecticity theorem, so it
    gets computed rather than eyeballed.
    """

    if len(energy_drift) < 4:
        return "insufficient-data"
    half = len(energy_drift) // 2
    early = max(energy_drift[:half])
    late = max(energy_drift[half:])
    if early <= 0:
        return "bounded" if late <= 0 else "secular"
    return "bounded" if late < 5 * early else "secular"
