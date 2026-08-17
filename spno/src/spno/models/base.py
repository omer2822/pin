"""The one interface every model in the comparison implements.

Signature is deliberately identical to :class:`SplitStepNLSOperator`, so the reference
solver is itself a valid ``StepOperator`` and every evaluator can be sanity-checked by
running it on the ground truth.

``dt`` handling is explicit rather than implicit.  A model trained at a single ``dt``
has no basis for stepping at another one: it absorbs the O(dt^2) splitting correction
into whatever it learned, making its effective rates dt-dependent.  Such a model
declares ``supports_dt_transfer = False`` and raises instead of silently returning a
wrong answer when Phase 6 asks it to extrapolate in time.
"""

from __future__ import annotations

import abc
import contextlib

import torch
import torch.nn as nn

from ..domain import PeriodicDomain

Tensor = torch.Tensor


class StepOperator(nn.Module, abc.ABC):
    """Advance a complex field by one step of size ``dt``."""

    #: Whether the model may be evaluated at a ``dt`` other than the one it was trained at.
    supports_dt_transfer: bool = False

    #: Whether stepping at ``-dt`` is meaningful.  True only when ``dt`` genuinely enters
    #: the map as a multiplier on a learned rate, as in the split-step family.  An FNO
    #: ignores its ``dt`` argument entirely, so asking it for ``-dt`` returns the forward
    #: map and any "reversibility" measured that way would be an artefact.
    supports_time_reversal: bool = False

    def __init__(self, domain: PeriodicDomain, trained_dt: float | None = None) -> None:
        super().__init__()
        self.domain = domain
        self.trained_dt = trained_dt

    @abc.abstractmethod
    def step(
        self,
        field: Tensor,
        potential: Tensor,
        alpha: Tensor,
        beta: Tensor,
        dt: float,
    ) -> Tensor:
        """Implement the actual map; inputs are already validated."""

    def forward(
        self,
        field: Tensor,
        potential: Tensor,
        alpha: Tensor,
        beta: Tensor,
        dt: float,
    ) -> Tensor:
        self.domain.validate_batched_field(field)
        if not field.is_complex():
            raise ValueError("StepOperator consumes and returns complex fields")
        if potential.shape != field.shape:
            raise ValueError("potential must have shape (batch, *domain.shape)")
        self._check_dt(dt)
        return self.step(field, potential, alpha, beta, dt)

    def _check_dt(self, dt: float) -> None:
        if self.supports_dt_transfer or self.trained_dt is None:
            return
        tolerance = 1e-12 * max(1.0, abs(self.trained_dt))
        if self.supports_time_reversal and abs(abs(float(dt)) - abs(self.trained_dt)) <= tolerance:
            return
        if abs(float(dt) - self.trained_dt) > tolerance:
            raise ValueError(
                f"{type(self).__name__} was trained at dt={self.trained_dt} and does not "
                f"support dt transfer; got dt={dt}. Its learned rates absorb the O(dt^2) "
                "splitting correction and are therefore dt-specific."
            )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@contextlib.contextmanager
def allow_dt_transfer(model: "StepOperator"):
    """Temporarily license evaluation at a ``dt`` the model was not trained at.

    An explicit, scoped opt-in rather than a permanent flag, because the refusal in
    :meth:`StepOperator._check_dt` is a *feature*: the learned rates absorb the O(dt^2)
    splitting correction and are therefore dt-specific.  Measuring transfer is
    legitimate; doing it by flipping a flag that then stays flipped for the rest of the
    process is not.  Mirrors how ``reversibility_order`` overrides ``trained_dt``.

    ``supports_dt_transfer`` is declared as a **class** attribute, so assigning to the
    instance shadows it.  The restore therefore deletes the instance attribute when
    there was not one to begin with, rather than writing the class value onto the
    instance -- otherwise every model that passed through this manager would carry a
    lingering instance attribute no longer bound to the class default.

    G6b is meaningful only for the split-step family, where ``dt`` multiplies a learned
    rate.  An FNO ignores its ``dt`` argument entirely, so a transfer number measured on
    one is an artefact, not a result.
    """

    had_instance_attribute = "supports_dt_transfer" in vars(model)
    original = model.supports_dt_transfer
    model.supports_dt_transfer = True
    try:
        yield model
    finally:
        if had_instance_attribute:
            model.supports_dt_transfer = original
        else:
            del model.supports_dt_transfer
