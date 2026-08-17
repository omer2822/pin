"""Phase 7: the discrete midpoint residual, checked against closed-form answers.

The Crank-Nicolson residual admits an exact plane-wave solution, so every claim here is
against analytic truth rather than a tolerance chosen to make it pass.

**The expansion.**  For ``psi_{n+1} = z psi_n`` with a plane wave in a constant
potential the residual collapses to ``psi_n [ i(z-1)/dt - omega(1+z)/2 ]``.  Putting
``z = exp(-i omega dt)`` and ``theta = omega dt``:

    Re(R)/psi =  (1/dt)[sin(theta) - (theta/2)(1 + cos(theta))] =  omega^3 dt^2 / 12 + ...
    Im(R)/psi =  (1/dt)[cos(theta) - 1 + (theta/2) sin(theta)]   = -omega^4 dt^3 / 24 + ...

so the *magnitude* is ``|psi| omega^3 dt^2 / 12`` only up to a relative correction of
order ``(omega dt)^2``.  That correction is 0.6% at ``dt=0.01`` here -- far above a
``rel=1e-3`` tolerance -- so the leading coefficient is asserted tightly at the smallest
``dt`` and its *deviation* is separately asserted to vanish at order 2.  Asserting the
coefficient at every ``dt`` with one tight tolerance would be testing the truncation
error, not the residual.
"""

from __future__ import annotations

import math

import pytest
import torch

from spno.domain import PeriodicDomain
from spno.equations.nls import exact_dispersion, plane_wave
from spno.losses.pde_residual import (
    crank_nicolson_frequency,
    midpoint_residual,
    residual_loss,
)

N, DT = 64, 0.01
ALPHA, BETA, V0, A = 0.9, 0.3, 0.2, 0.5642
WAVE_NUMBER = 5


def _omega() -> float:
    return float(
        exact_dispersion(
            (WAVE_NUMBER,), alpha=ALPHA, beta=BETA, amplitude=A, potential_constant=V0
        )
    )


def _plane_wave_pair(dt, frequency):
    domain = PeriodicDomain.periodic_1d(N)
    field = plane_wave(
        domain, (WAVE_NUMBER,), amplitude=A, dtype=torch.complex128
    ).unsqueeze(0)
    return domain, field, field * complex(
        math.cos(frequency * dt), -math.sin(frequency * dt)
    )


def _residual_magnitude(dt, frequency) -> float:
    domain, field, nxt = _plane_wave_pair(dt, frequency)
    potential = torch.full_like(field.real, V0)
    residual = midpoint_residual(
        field,
        nxt,
        potential,
        domain,
        torch.tensor([ALPHA], dtype=torch.float64),
        torch.tensor([BETA], dtype=torch.float64),
        dt,
    )
    return float(torch.abs(residual).max())


def test_the_residual_vanishes_on_its_own_exact_plane_wave_solution():
    """The Cayley frequency is the one the scheme solves exactly, not omega."""

    assert _residual_magnitude(DT, crank_nicolson_frequency(_omega(), DT)) < 1e-12


def test_the_cn_update_is_exactly_unit_modulus_which_is_the_confound():
    """The physics loss of 7a smuggles in mass conservation: |z| = 1 exactly.

    Not "to within tolerance" -- the Cayley transform of a real number has modulus one
    identically.  Stating this is the point of running 7a at all.
    """

    for omega in (0.5, 7.0, 120.0, -33.0):
        theta = crank_nicolson_frequency(omega, DT) * DT
        z = complex(math.cos(theta), -math.sin(theta))
        assert abs(abs(z) - 1.0) < 1e-15


def test_the_residual_on_the_exact_solution_is_second_order():
    """|R| ~ |psi| omega^3 dt^2 / 12: the *order* holds at every dt in the ladder."""

    omega = _omega()
    errors = [_residual_magnitude(dt, omega) for dt in (0.0025, 0.005, 0.01)]
    orders = [math.log2(errors[i + 1] / errors[i]) for i in range(len(errors) - 1)]
    assert all(o == pytest.approx(2.0, abs=0.05) for o in orders), f"{orders=}"


def test_the_leading_coefficient_is_exact_in_the_small_dt_limit():
    """The coefficient omega^3/12 is asserted where the O((omega dt)^2) correction is
    below the tolerance, and its deviation is asserted to vanish at order 2.

    Together these test the leading coefficient *and* the structure of the next term --
    strictly more than asserting the coefficient at one dt with a loose tolerance.
    """

    omega = _omega()
    dts = (0.0025, 0.005, 0.01)
    predicted = [A * abs(omega) ** 3 * dt**2 / 12 for dt in dts]
    measured = [_residual_magnitude(dt, omega) for dt in dts]

    assert measured[0] == pytest.approx(predicted[0], rel=1e-3)

    deviations = [abs(m / p - 1.0) for m, p in zip(measured, predicted)]
    orders = [
        math.log2(deviations[i + 1] / deviations[i]) for i in range(len(deviations) - 1)
    ]
    assert all(o == pytest.approx(2.0, abs=0.05) for o in orders), f"{deviations=}"


def test_the_residual_is_large_on_an_unrelated_pair():
    """The paired negative: a residual small for everything measures nothing."""

    domain = PeriodicDomain.periodic_1d(N)
    torch.manual_seed(0)
    field = torch.randn(2, N, dtype=torch.complex128)
    nxt = torch.randn(2, N, dtype=torch.complex128)
    potential = torch.zeros(2, N, dtype=torch.float64)
    loss = residual_loss(
        field,
        nxt,
        potential,
        domain,
        torch.full((2,), ALPHA, dtype=torch.float64),
        torch.full((2,), BETA, dtype=torch.float64),
        DT,
    )
    assert float(loss) > 1.0


def test_the_residual_rejects_mismatched_shapes():
    domain = PeriodicDomain.periodic_1d(N)
    field = torch.randn(2, N, dtype=torch.complex128)
    potential = torch.zeros(2, N, dtype=torch.float64)
    with pytest.raises(ValueError, match="identical shapes"):
        midpoint_residual(
            field,
            torch.randn(3, N, dtype=torch.complex128),
            potential,
            domain,
            torch.full((2,), ALPHA, dtype=torch.float64),
            torch.full((2,), BETA, dtype=torch.float64),
            DT,
        )
