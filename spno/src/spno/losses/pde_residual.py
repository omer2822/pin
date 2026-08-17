"""The discrete midpoint (Crank--Nicolson) PDE residual: the soft-physics baseline.

    R = i (psi_{n+1} - psi_n)/dt
      + alpha * laplacian((psi_{n+1} + psi_n)/2)
      + beta * (|psi_{n+1}|^2 + |psi_n|^2)/2 * (psi_{n+1} + psi_n)/2
      - V * (psi_{n+1} + psi_n)/2

*The confound, stated up front.*  The nonlinear term is averaged in the Delfour--Fortin
--Payre form, which makes the scheme discretely mass-conserving; for a plane wave the
update factor is the Cayley transform ``(1 - i omega dt/2)/(1 + i omega dt/2)``, whose
modulus is **exactly** one.  So this physics loss smuggles in the very invariant the
study compares methods on.  That is not a reason to avoid it -- it is precisely what the
physics-loss / projection / architecture distinction exists to expose, and it must be
said in the thesis next to every 7a number.

*Order (numerical observation, derived and measured).*  On the exact solution
``psi_{n+1} = exp(-i omega dt) psi_n`` the residual is

    Re(R)/psi =  omega^3 dt^2 / 12 + O(dt^4)
    Im(R)/psi = -omega^4 dt^3 / 24 + O(dt^5)

so ``|R| = |psi| omega^3 dt^2 / 12`` up to a *relative* correction of order
``(omega dt)^2``.  The residual is second order; the leading coefficient is recovered
only in the small-``dt`` limit.  ``tests/test_pde_residual.py`` asserts both parts.
"""

from __future__ import annotations

import math

import torch

from ..domain import PeriodicDomain, batch_parameter, spectral_laplacian

Tensor = torch.Tensor


def crank_nicolson_frequency(omega: float, dt: float) -> float:
    """The frequency the CN residual is exactly solved by: ``(2/dt) arctan(omega dt/2)``.

    Inverting the Cayley transform.  A test that feeds the residual the *analytic* plane
    wave ``exp(-i omega dt)`` and expects zero is testing the wrong thing: the scheme's
    own exact solution advances at this shifted frequency, and the gap between the two
    is the scheme's O(dt^2) truncation error.
    """

    return 2.0 / float(dt) * math.atan(float(omega) * float(dt) / 2.0)


def midpoint_residual(
    field: Tensor,
    next_field: Tensor,
    potential: Tensor,
    domain: PeriodicDomain,
    alpha: Tensor | float,
    beta: Tensor | float,
    dt: float,
) -> Tensor:
    """Pointwise residual; zero exactly when the pair solves the CN scheme."""

    domain.validate_batched_field(field)
    if next_field.shape != field.shape or potential.shape != field.shape:
        raise ValueError("field, next_field and potential must have identical shapes")
    batch = field.shape[0]
    alpha_grid = batch_parameter(alpha, batch, domain, field, "alpha")
    beta_grid = batch_parameter(beta, batch, domain, field, "beta")

    average = 0.5 * (next_field + field)
    density = 0.5 * (torch.abs(next_field) ** 2 + torch.abs(field) ** 2)
    return (
        1j * (next_field - field) / float(dt)
        + alpha_grid * spectral_laplacian(average, domain)
        + beta_grid * density * average
        - potential * average
    )


def residual_loss(
    field: Tensor,
    next_field: Tensor,
    potential: Tensor,
    domain: PeriodicDomain,
    alpha: Tensor | float,
    beta: Tensor | float,
    dt: float,
) -> Tensor:
    """Batch-mean residual norm, normalized by ``||psi_n|| / dt``.

    The normalization makes ``lambda`` comparable across ``dt`` and across mass, which
    varies by design.  An unnormalized residual weight is uninterpretable the moment the
    multi-dt arm exists, and would quietly become a mass-weighted objective.
    """

    residual = midpoint_residual(field, next_field, potential, domain, alpha, beta, dt)
    numerator = torch.sqrt(torch.sum(torch.abs(residual) ** 2, dim=domain.spatial_axes))
    scale = torch.sqrt(
        torch.sum(torch.abs(field) ** 2, dim=domain.spatial_axes)
    ) / float(dt)
    return (numerator / scale.clamp_min(1e-30)).mean()
