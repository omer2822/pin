"""Symmetric (Strang) split-step integrator for the parametric NLS.

Ported from ``pinn-neural-operators/05_3d_equations.py`` (``SplitStepNLSOperator``),
which is covered by plane-wave, mass, and reversibility tests to 1e-10.

What this scheme guarantees, and what it does not:

============================  ==========================================================
mass conservation             **exact** -- every substep is a modulus-one multiplier, and
                              the FFT round trip preserves the discrete l2 norm (Parseval)
time reversibility            **exact** -- ``Phi_{-dt} . Phi_{dt} = Id``; the local phase
                              reads ``|psi|^2``, which its own multiplier leaves unchanged
symplecticity                 **exact** -- a composition of the exact flows of the kinetic
                              and local sub-Hamiltonians
Hamiltonian conservation      **not exact** -- ``O(dt^2)``, bounded rather than secular
order in time                 2
============================  ==========================================================

:class:`SubsteppedReference` exists to defuse the central methodological confound of
the thesis: if training targets were produced by a single Strang step at the same
``dt`` the models take, the true one-step map would sit *inside* the learned
split-step hypothesis class, and "structure beats FNO" would be a tautology.
Targets must therefore come from a converged, substepped reference.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..domain import PeriodicDomain, batch_parameter

Tensor = torch.Tensor

class SplitStepNLSOperator(nn.Module):
    """One symmetric unitary Strang step for NLS with an arbitrary real potential."""

    def __init__(self, domain: PeriodicDomain) -> None:
        super().__init__()
        self.domain = domain

    def _kinetic(self, field: Tensor, alpha: Tensor, dt: float) -> Tensor:
        k_squared = self.domain.wave_number_squared(
            device=field.device, dtype=field.real.dtype
        )
        transformed = torch.fft.fftn(field, dim=self.domain.spatial_axes)
        return torch.fft.ifftn(
            transformed * torch.exp(-1j * alpha * k_squared * dt),
            dim=self.domain.spatial_axes,
        )

    def forward(
        self,
        field: Tensor,
        potential: Tensor,
        alpha: Tensor | float,
        beta: Tensor | float,
        dt: float,
    ) -> Tensor:
        self.domain.validate_batched_field(field)
        self.domain.validate_field(potential)
        if potential.shape != field.shape:
            raise ValueError("field and potential must have shape (batch, *domain.shape)")
        if not field.is_complex() or potential.is_complex():
            raise ValueError("field must be complex and potential must be real")
        alpha_grid = batch_parameter(alpha, field.shape[0], self.domain, field, "alpha")
        beta_grid = batch_parameter(beta, field.shape[0], self.domain, field, "beta")
        midpoint = self._kinetic(field, alpha_grid, 0.5 * float(dt))
        local_phase = torch.exp(
            1j * (beta_grid * torch.abs(midpoint) ** 2 - potential) * float(dt)
        )
        return self._kinetic(midpoint * local_phase, alpha_grid, 0.5 * float(dt))


class SubsteppedReference(nn.Module):
    """Advance by ``dt`` using ``substeps`` symmetric Strang steps of ``dt/substeps``.

    This is the trajectory generator for every dataset in the project.  With
    ``substeps=1`` it degenerates to :class:`SplitStepNLSOperator`, which is exactly
    the configuration the study must avoid for target generation.
    """

    def __init__(self, domain: PeriodicDomain, substeps: int = 32) -> None:
        super().__init__()
        if not isinstance(substeps, int) or substeps < 1:
            raise ValueError("substeps must be a positive integer")
        self.domain = domain
        self.substeps = substeps
        self.step = SplitStepNLSOperator(domain)

    def forward(
        self,
        field: Tensor,
        potential: Tensor,
        alpha: Tensor | float,
        beta: Tensor | float,
        dt: float,
    ) -> Tensor:
        sub_dt = float(dt) / self.substeps
        evolved = field
        for _ in range(self.substeps):
            evolved = self.step(evolved, potential, alpha, beta, sub_dt)
        return evolved

def split_step_solver(
    psi0: np.ndarray,
    x: np.ndarray,
    dt: float,
    steps: int,
    kappa: float,
) -> np.ndarray:
    """Solve the 1D cubic NLS with a NumPy split-step method.

    This is a small, unbatched NumPy convenience solver.  The Torch operators below
    are the package's differentiable, batched reference implementation; this helper
    is intentionally kept separate because it has a different API and does not use
    :class:`PeriodicDomain`.

    Args:
        psi0: Initial complex-valued field with shape ``(N,)``.
        x: Uniform spatial grid with shape ``(N,)``.
        dt: Time step.
        steps: Number of time steps.
        kappa: Cubic nonlinear coefficient.

    Returns:
        The field after ``steps`` split steps.
    """

    psi = np.asarray(psi0, dtype=complex)
    grid = np.asarray(x)
    if psi.ndim != 1 or grid.ndim != 1 or psi.shape != grid.shape:
        raise ValueError("psi0 and x must be one-dimensional arrays of equal length")
    if grid.size < 2:
        raise ValueError("x must contain at least two grid points")
    if not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")

    n = grid.size
    length = grid[-1] - grid[0]
    k = 2 * np.pi * np.fft.fftfreq(n, d=length / n)
    operator_dispersion = np.exp(-1j * (k**2) * (dt / 2))

    for _ in range(steps):
        # fft the field 
        psi_freq = np.fft.fft(psi)
        psi = np.fft.ifft(psi_freq * operator_dispersion)

        operator_nonlinear = np.exp(-1j * kappa * np.abs(psi) ** 2 * dt)
        psi *= operator_nonlinear

        psi_freq = np.fft.fft(psi)
        psi = np.fft.ifft(psi_freq * operator_dispersion)

    return psi

@torch.no_grad()
def integrate(
    operator: nn.Module,
    field: Tensor,
    potential: Tensor,
    alpha: Tensor | float,
    beta: Tensor | float,
    dt: float,
    steps: int,
    *,
    store_every: int = 1,
) -> Tensor:
    """Roll ``operator`` forward and stack the trajectory as ``(batch, frames, *shape)``.

    Frame 0 is always the initial condition, so ``frames = steps // store_every + 1``.
    """

    if steps < 0 or store_every < 1:
        raise ValueError("steps must be nonnegative and store_every must be positive")
    frames = [field]
    evolved = field
    for index in range(steps):
        evolved = operator(evolved, potential, alpha, beta, dt)
        if (index + 1) % store_every == 0:
            frames.append(evolved)
    return torch.stack(frames, dim=1)


@torch.no_grad()
def splitting_floor(
    domain: PeriodicDomain,
    field: Tensor,
    potential: Tensor,
    alpha: Tensor | float,
    beta: Tensor | float,
    dt: float,
    *,
    substeps: int = 32,
) -> Tensor:
    """Relative L2 error of one Strang step at ``dt`` against the substepped reference.

    This is ``eps_split``: the analytic lower bound on the one-step error of any model
    whose hypothesis class is a single split step at ``dt``.  A structured model that
    trains down to this value has recovered the physics; a black-box model that beats
    it is exploiting non-split structure the splitting cannot represent -- which is a
    finding about expressivity, not a failure.
    """

    coarse = SplitStepNLSOperator(domain)(field, potential, alpha, beta, dt)
    fine = SubsteppedReference(domain, substeps)(field, potential, alpha, beta, dt)
    return relative_l2(coarse, fine, domain)


def fitted_splitting_floor(
    domain: PeriodicDomain,
    field: Tensor,
    potential: Tensor,
    alpha: Tensor | float,
    beta: Tensor | float,
    dt: float,
    *,
    substeps: int = 32,
    iterations: int = 80,
    per_mode: bool = False,
) -> tuple[Tensor, Tensor]:
    """The floor a *learned* split step can reach, not the one the exact rates reach.

    :func:`splitting_floor` measures the error of the Strang map built from the true
    rates ``kappa = -alpha|k|^2`` and ``nu = beta|psi|^2 - V``.  A learned split-step
    model is not obliged to use those: it can pick *effective* rates that absorb part
    of the O(dt^2) commutator.  The quantity that actually lower-bounds Model C is
    therefore ``inf_theta || Phi^theta_dt - Phi_ref ||``, which is at most, and may be
    well below, the exact-rate error.

    Two nested families are available, both shared across the whole batch and hence
    across all ``(alpha, beta)`` -- a single model must serve every parameter value:

    ``per_mode=False``
        one scalar on each generator,
        ``kappa = -c0 * alpha|k|^2``, ``nu = c1 * (beta|psi|^2 - V)``.

    ``per_mode=True``
        a free per-wavenumber correction on the kinetic rate,
        ``kappa = -alpha|k|^2 * g(|k|^2)`` with ``g`` a free vector initialized at 1.
        This is exactly the ``K2`` parameterization Phase 4 gives Model C1, so the
        error it achieves is the target that C1-K2 should be measured against.
        Note the correction multiplies ``alpha|k|^2`` rather than replacing it: a free
        function of ``|k|^2`` alone could not represent the required alpha-scaling.

    Returns the achieved per-sample error together with the fitted coefficients.
    """

    domain.validate_batched_field(field)
    reference = SubsteppedReference(domain, substeps)(field, potential, alpha, beta, dt)
    k_squared = domain.wave_number_squared(
        device=field.device, dtype=field.real.dtype
    )
    alpha_grid = batch_parameter(alpha, field.shape[0], domain, field, "alpha")
    beta_grid = batch_parameter(beta, field.shape[0], domain, field, "beta")
    kinetic_size = k_squared.numel() if per_mode else 1
    scales = torch.ones(kinetic_size + 1, dtype=field.real.dtype, requires_grad=True)

    def parameterized_step(coefficients: Tensor) -> Tensor:
        kinetic_scale = coefficients[:kinetic_size].reshape(
            k_squared.shape if per_mode else ()
        )

        def kinetic(state: Tensor) -> Tensor:
            transformed = torch.fft.fftn(state, dim=domain.spatial_axes)
            multiplier = torch.exp(
                -1j * kinetic_scale * alpha_grid * k_squared * (0.5 * dt)
            )
            return torch.fft.ifftn(transformed * multiplier, dim=domain.spatial_axes)

        midpoint = kinetic(field)
        local_phase = torch.exp(
            1j
            * coefficients[-1]
            * (beta_grid * torch.abs(midpoint) ** 2 - potential)
            * dt
        )
        return kinetic(midpoint * local_phase)

    optimizer = torch.optim.LBFGS(
        [scales], max_iter=iterations, line_search_fn="strong_wolfe"
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = relative_l2(parameterized_step(scales), reference, domain).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        achieved = relative_l2(parameterized_step(scales), reference, domain)
    return achieved.detach(), scales.detach()


def relative_l2(prediction: Tensor, target: Tensor, domain: PeriodicDomain) -> Tensor:
    """Per-sample relative L2 error over the spatial axes."""

    domain.validate_field(prediction)
    domain.validate_field(target)
    numerator = torch.sqrt(
        torch.sum(torch.abs(prediction - target) ** 2, dim=domain.spatial_axes)
    )
    denominator = torch.sqrt(
        torch.sum(torch.abs(target) ** 2, dim=domain.spatial_axes)
    )
    return numerator / torch.clamp(denominator, min=1e-30)
