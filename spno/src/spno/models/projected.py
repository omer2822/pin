"""Model B: an FNO whose output is rescaled to the input's mass.

This is the "constraint projection" rung of the three-way distinction the thesis keeps
explicit -- physics-informed *loss*, constraint *projection*, physics encoded in the
*architecture*.  It enforces exactly one scalar invariant and says nothing about
phase, energy, reversibility, or the dispersion relation.  If it behaves like the
unconstrained FNO on rollout, that is the intended lesson rather than a bug.

Two training modes, and they are not equivalent:

``projection in the loop``
    the projection is part of the forward pass, so gradients flow through it and the
    core learns to produce fields whose *shape* is right, with amplitude handled
    downstream.

``post hoc``
    the core is trained unconstrained and the projection is applied only at
    evaluation.  The core still spends capacity on getting the amplitude right.
"""

from __future__ import annotations

import torch

from ..domain import PeriodicDomain, l2_mass, project_to_mass
from .base import StepOperator

Tensor = torch.Tensor


class MassProjectedOperator(StepOperator):
    """Wrap any :class:`StepOperator` and rescale its output to the input mass.

    Mass is preserved exactly in exact arithmetic.  In float32 the FFT round trips
    inside the core accumulate roundoff, so the measured drift floors near 1e-6 over
    100 steps; that floor is reported next to every "exact" claim rather than being
    rounded away.
    """

    def __init__(self, core: StepOperator, *, enabled: bool = True) -> None:
        super().__init__(core.domain, core.trained_dt)
        self.core = core
        self.enabled = enabled
        self.supports_dt_transfer = core.supports_dt_transfer

    def step(
        self,
        field: Tensor,
        potential: Tensor,
        alpha: Tensor,
        beta: Tensor,
        dt: float,
    ) -> Tensor:
        raw = self.core.step(field, potential, alpha, beta, dt)
        if not self.enabled:
            return raw
        return project_to_mass(raw, field, self.domain)

    def parameter_count(self) -> int:
        # The projection adds no parameters; report the core's count so the model
        # comparison table is not misleading about capacity.
        return self.core.parameter_count()


@torch.no_grad()
def mass_drift(field: Tensor, reference: Tensor, domain: PeriodicDomain) -> Tensor:
    """Relative mass drift ``|M(psi)/M(ref) - 1|``, per sample."""

    return torch.abs(
        l2_mass(field, domain) / l2_mass(reference, domain).clamp_min(1e-30) - 1
    )
