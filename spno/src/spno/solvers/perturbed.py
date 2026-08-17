"""Phase 9: two dials on the *data-generating* equation, each recovering NLS at 0.

The misspecification sweep is the only phase that makes "structure beats FNO" a
question rather than a tautology, because it moves the truth *outside* the constrained
model class.  Each dial breaks a different assumption:

===================  ==========================================================
``sigma``            **nonlocal nonlinearity** ``nu = beta (W_sigma * rho)``.
                     Breaks *locality*.  Still Hamiltonian, still U(1)-equivariant,
                     still exactly mass-conserving -- so B's projection stays *correct*
                     and only C1's pointwise ``nu`` becomes unable to represent the
                     truth.  This is what separates C1 from C2.
``gamma``            **weak gain/loss** ``+ i gamma psi``.  Breaks *conservation
                     itself*, so B's hard mass constraint becomes actively **wrong**.
===================  ==========================================================

Deliberately **not** a saturable or quintic nonlinearity: ``nu_theta`` is already a free
function of ``rho``, so the C family learns those easily and they are not
misspecifications at all.

**The dial-zero short-circuit is load-bearing, not an optimization.**  A spectral
Gaussian with ``sigma=0`` has multiplier exactly 1, but ``ifft(fft(rho) * 1)`` is *not*
bitwise ``rho`` -- the FFT round trip is accurate, not exact.  Phase 9's opening gate
asserts the dial-zero generator reproduces the unperturbed solver **bitwise**, so both
operators branch to :meth:`SplitStepNLSOperator.forward` at zero.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..domain import PeriodicDomain, batch_parameter
from .split_step import SplitStepNLSOperator

Tensor = torch.Tensor


class NonlocalSplitStepNLSOperator(SplitStepNLSOperator):
    """Strang split step whose nonlinearity sees a *smoothed* density.

    The local substep applies ``exp(i dt nu)`` with

        ``nu = beta * (W_sigma * rho) - V``,   ``W_hat(k) = exp(-sigma^2 |k|^2 / 2)``.

    *Architectural guarantee.*  ``nu`` is real, so the substep is a modulus-one
    multiplier: discrete mass is preserved exactly and global U(1) equivariance
    survives, for every ``sigma``.

    *Theorem.*  The dynamics remain Hamiltonian.  A symmetric kernel makes ``nu`` the
    variational derivative of ``-(beta/2) int int rho(x) W(x-y) rho(y)``, so the local
    substep is still the exact flow of a sub-Hamiltonian and the composition is still
    symplectic.  What is broken is **locality**, nothing else.

    ``W_hat(0) = 1`` exactly, so the kernel has unit integral at every ``sigma`` and
    never rescales the mean density; ``sigma -> 0`` is a delta by construction.
    """

    def __init__(self, domain: PeriodicDomain, *, sigma: float) -> None:
        super().__init__(domain)
        if not float(sigma) >= 0.0:
            raise ValueError("sigma must be non-negative")
        self.sigma = float(sigma)
        # Built in float64 and narrowed by the caller if needed -- never constructed
        # natively at float32, which lands 1.2e-4 from the exact integers at N=64.
        k_squared = domain.wave_number_squared()
        self.register_buffer("kernel", torch.exp(-0.5 * self.sigma**2 * k_squared))

    def forward(
        self,
        field: Tensor,
        potential: Tensor,
        alpha: Tensor | float,
        beta: Tensor | float,
        dt: float,
    ) -> Tensor:
        if self.sigma == 0.0:
            # Bitwise identity with the unperturbed solver: see the module docstring.
            return super().forward(field, potential, alpha, beta, dt)

        self.domain.validate_batched_field(field)
        self.domain.validate_field(potential)
        if potential.shape != field.shape:
            raise ValueError("field and potential must have shape (batch, *domain.shape)")
        if not field.is_complex() or potential.is_complex():
            raise ValueError("field must be complex and potential must be real")

        alpha_grid = batch_parameter(alpha, field.shape[0], self.domain, field, "alpha")
        beta_grid = batch_parameter(beta, field.shape[0], self.domain, field, "beta")

        midpoint = self._kinetic(field, alpha_grid, 0.5 * float(dt))
        density = torch.abs(midpoint) ** 2
        smoothed = torch.fft.ifftn(
            torch.fft.fftn(density, dim=self.domain.spatial_axes)
            * self.kernel.to(density.dtype),
            dim=self.domain.spatial_axes,
        ).real
        local_phase = torch.exp(1j * (beta_grid * smoothed - potential) * float(dt))
        return self._kinetic(midpoint * local_phase, alpha_grid, 0.5 * float(dt))


class GainLossSplitStepNLSOperator(SplitStepNLSOperator):
    """Strang split step with weak linear gain (``gamma > 0``) or loss (``gamma < 0``).

    **Sign convention, stated explicitly because it is easy to invert.**  The perturbed
    equation is

        ``i psi_t + alpha Lap psi + beta |psi|^2 psi - V psi = +i gamma psi``

    hence ``psi_t = +gamma psi + i(...)``, the amplitude multiplies by
    ``exp(gamma dt)`` each step, and

        ``d ln M / dt = 2 gamma``   *(theorem, asserted as a rate in the tests)*.

    Writing ``-i gamma psi`` on the right would give decay instead; the tests pin the
    convention by measuring the rate rather than trusting this paragraph.

    **This dial breaks conservation itself**, which is the point: Model B's hard mass
    projection becomes actively *wrong*, not merely unhelpful, so a crossover here is
    evidence about when a hard invariant stops helping.

    The gain factor is a real scalar and therefore commutes with the kinetic operator,
    so applying it inside the local substep is **exact** rather than an additional
    O(dt^2) splitting error -- the dial does not contaminate ``eps_split``.
    """

    def __init__(self, domain: PeriodicDomain, *, gamma: float) -> None:
        super().__init__(domain)
        self.gamma = float(gamma)

    def forward(
        self,
        field: Tensor,
        potential: Tensor,
        alpha: Tensor | float,
        beta: Tensor | float,
        dt: float,
    ) -> Tensor:
        if self.gamma == 0.0:
            return super().forward(field, potential, alpha, beta, dt)

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
        # exp(gamma dt) is real and commutes with the kinetic multiplier: exact.
        amplified = midpoint * local_phase * torch.exp(
            torch.tensor(self.gamma * float(dt), dtype=midpoint.real.dtype)
        )
        return self._kinetic(amplified, alpha_grid, 0.5 * float(dt))


class SubsteppedOperator(nn.Module):
    """Advance by ``dt`` using ``substeps`` steps of an injected inner operator.

    :class:`~spno.solvers.split_step.SubsteppedReference` hardcodes the unperturbed
    step.  Phase 9's generators need the *same* ``substeps=32`` treatment, because a
    single perturbed Strang step at ``dt`` would confound the dial with the O(dt^2)
    splitting error and every located crossover would then be partly a measurement of
    ``eps_split``.
    """

    def __init__(self, inner: nn.Module, substeps: int = 32) -> None:
        super().__init__()
        if not isinstance(substeps, int) or substeps < 1:
            raise ValueError("substeps must be a positive integer")
        self.inner = inner
        self.substeps = substeps

    @property
    def domain(self) -> PeriodicDomain:
        return self.inner.domain

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
            evolved = self.inner(evolved, potential, alpha, beta, sub_dt)
        return evolved
