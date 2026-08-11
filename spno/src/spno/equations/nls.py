"""The parametric nonlinear Schrodinger equation: invariants and exact solutions.

    i psi_t + alpha * laplacian(psi) + beta * |psi|^2 psi - V(x) psi = 0
    <=> psi_t = i * (alpha * laplacian(psi) + beta * |psi|^2 psi - V psi)

on ``[0, 2*pi)^d`` with periodic boundary conditions.

Sign conventions (verified against the tutorial solver and its plane-wave test):

* kinetic Fourier multiplier   ``exp(-i * alpha * |k|^2 * dt)``
* local physical-space phase   ``exp(+i * (beta*|psi|^2 - V) * dt)``
* plane-wave dispersion        ``omega = alpha*|k|^2 - beta*|A|^2 + V0``

``H`` below is the generator: ``i psi_t = delta H / delta conj(psi)``.  ``H`` is
invariant under the global gauge rotation ``psi -> exp(i*theta) psi``, so Noether's
theorem gives conservation of the mass ``M = integral |psi|^2``.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from ..domain import PeriodicDomain, batch_parameter, spectral_gradient

Tensor = torch.Tensor


def nls_hamiltonian(
    field: Tensor,
    potential: Tensor,
    domain: PeriodicDomain,
    alpha: Tensor | float,
    beta: Tensor | float,
) -> Tensor:
    """Discrete NLS energy ``integral(alpha|grad psi|^2 + V|psi|^2 - beta|psi|^4/2)``.

    Satisfies ``i psi_t = delta H / delta conj(psi)`` for the equation above.  The
    split-step solver conserves this only to ``O(dt^2)``; conservation of *mass* is
    what is exact.
    """

    domain.validate_batched_field(field)
    domain.validate_field(potential)
    if potential.shape != field.shape:
        raise ValueError("field and potential must have shape (batch, *domain.shape)")
    alpha_grid = batch_parameter(alpha, field.shape[0], domain, field, "alpha")
    beta_grid = batch_parameter(beta, field.shape[0], domain, field, "beta")
    spatial_gradient = spectral_gradient(field, domain)
    kinetic_density = torch.sum(torch.abs(spatial_gradient) ** 2, dim=1)
    density = torch.abs(field) ** 2
    energy_density = (
        alpha_grid * kinetic_density
        + potential * density
        - 0.5 * beta_grid * density**2
    )
    return torch.sum(energy_density, dim=domain.spatial_axes) * domain.cell_volume


def exact_dispersion(
    wave_numbers: Sequence[int] | Tensor,
    *,
    alpha: float | Tensor,
    beta: float | Tensor = 0.0,
    amplitude: float | Tensor = 1.0,
    potential_constant: float | Tensor = 0.0,
) -> Tensor | float:
    """``omega = alpha|k|^2 - beta|A|^2 + V0`` for a plane wave with constant potential.

    This is the ground truth that ``evaluation/dispersion.py`` probes every trained
    model against, black-box models included.
    """

    if isinstance(wave_numbers, torch.Tensor):
        k_squared: Tensor | float = torch.sum(wave_numbers.double() ** 2, dim=-1)
    else:
        k_squared = float(sum(float(k) ** 2 for k in wave_numbers))
    return alpha * k_squared - beta * amplitude**2 + potential_constant


def plane_wave(
    domain: PeriodicDomain,
    wave_numbers: Sequence[int],
    *,
    amplitude: float = 1.0,
    time: float = 0.0,
    alpha: float = 0.5,
    beta: float = 0.0,
    potential_constant: float = 0.0,
    dtype: torch.dtype = torch.complex128,
) -> Tensor:
    """ND NLS plane wave, an exact solution when the potential is constant."""

    if len(wave_numbers) != domain.dim:
        raise ValueError("wave_numbers must contain one integer per dimension")
    if amplitude <= 0:
        raise ValueError("amplitude must be positive")
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    coordinates = domain.mesh(dtype=real_dtype)
    phase = sum(k * x for k, x in zip(wave_numbers, coordinates))
    omega = exact_dispersion(
        wave_numbers,
        alpha=alpha,
        beta=beta,
        amplitude=amplitude,
        potential_constant=potential_constant,
    )
    return (amplitude * torch.exp(1j * (phase - omega * time))).to(dtype)


def plane_wave_mass(domain: PeriodicDomain, amplitude: float) -> float:
    """Mass of a plane wave: ``|A|^2 * volume``.

    The dispersion probe must choose ``amplitude`` so this lands inside the training
    mass distribution, or it confounds spectral extrapolation with mass extrapolation.
    """

    return amplitude**2 * math.prod(domain.lengths)


def wrap_wavenumber(alpha: float, dt: float) -> float:
    """``k_wrap = sqrt(pi / (alpha*dt))``: the single-alpha identifiability horizon.

    For ``|k| > k_wrap`` the one-step kinetic phase ``alpha*k^2*dt`` exceeds ``pi``, so
    the observed Fourier multiplier ``exp(-i*alpha*k^2*dt)`` determines ``omega`` only
    modulo ``2*pi/dt``.  Multi-step data at the same ``dt`` adds nothing, since the
    n-step multiplier is a function of the one-step multiplier.

    This bound is *conditional on fixed alpha*.  Across an alpha-family the derivative
    ``d(arg m)/d(alpha) = -k^2 dt`` is wrap-free whenever ``d(alpha)*k^2*dt < pi``, so
    the dispersion relation is identifiable in principle at every ``k``.  See
    :func:`alpha_sampling_is_dense_enough`.
    """

    if alpha <= 0 or dt <= 0:
        raise ValueError("alpha and dt must be positive")
    return math.sqrt(math.pi / (alpha * dt))


def alpha_sampling_is_dense_enough(
    alpha_gap: float, max_wave_number: int, dt: float
) -> bool:
    """Whether ``d(arg m)/d(alpha)`` is recoverable without phase unwrapping.

    Requires ``alpha_gap * k^2 * dt < pi`` at the largest wave number of interest.
    This is a *checked precondition* of the alpha-varying identifiability arm (G5b),
    not an assumption.
    """

    if alpha_gap <= 0 or dt <= 0 or max_wave_number <= 0:
        raise ValueError("alpha_gap, max_wave_number, and dt must be positive")
    return alpha_gap * max_wave_number**2 * dt < math.pi
