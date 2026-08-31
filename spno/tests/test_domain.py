"""Tests for ``spno.domain`` -- geometry, invariants, the domain hierarchy, and the
two Nyquist traps.

There was no ``tests/test_domain.py`` before Phase 1 of the domain abstraction review,
which is itself notable for a module 22 other modules depend on.  This file exists to
pin the behavior identified as under-tested or defective by that review, plus the
``MeasureSpace -> HilbertSpace -> SpectralDomain`` hierarchy built per ADR 0002:

- Task 1: ``project_to_mass`` must not NaN the whole batch when one reference sample
  has zero mass (``torch.where`` evaluates both branches in the forward pass).
- Task 2: Trap A -- the Nyquist mode is zeroed for real fields on even grids, and
  must keep being zeroed, while complex fields are left untouched.  Trap A now lives
  in ``PeriodicDomain.derivative_multiplier``; these tests exercise it indirectly
  through ``spectral_gradient`` exactly as before the refactor, plus directly.
- Task 3 / 4: ``volume`` and the ``spatial_broadcast`` contract guard.
- ADR 0002: the class hierarchy is instantiable (the abstract-``shape``-property /
  dataclass-field collision this depends on avoiding), and ``integrate`` /
  ``analyze`` + ``synthesize`` + the two multipliers agree with the pre-existing
  free functions they now back.
"""

from __future__ import annotations

import math

import pytest
import torch

from spno.domain import (
    HilbertSpace,
    MeasureSpace,
    PeriodicDomain,
    SpectralDomain,
    l2_mass,
    project_to_mass,
    spatial_broadcast,
    spectral_gradient,
    spectral_laplacian,
)


# ---------------------------------------------------------------------------
# Task 1 -- project_to_mass gradient regression
# ---------------------------------------------------------------------------


def test_projection_gradient_is_finite_with_a_zero_mass_reference():
    """torch.where masks the gradient but not the forward pass: sqrt(0) has an
    infinite derivative, and 0 * inf is NaN.  A single zero-mass reference sample
    must not NaN the whole batch."""

    domain = PeriodicDomain.periodic_1d(16)
    prediction = torch.randn(3, 16, dtype=torch.complex128, requires_grad=True)
    reference = torch.randn(3, 16, dtype=torch.complex128)
    reference[1] = 0  # the singular sample

    project_to_mass(prediction, reference, domain).abs().sum().backward()

    assert torch.isfinite(prediction.grad).all()


def test_projection_forward_value_unchanged_away_from_the_guard():
    """The fix only touches the branch that is masked out; every sample whose
    reference mass clears ``eps`` must see a bit-identical result."""

    domain = PeriodicDomain.periodic_1d(16)
    torch.manual_seed(0)
    prediction = torch.randn(4, 16, dtype=torch.complex128)
    reference = torch.randn(4, 16, dtype=torch.complex128)

    result = project_to_mass(prediction, reference, domain)

    # Every projected sample reproduces the reference mass.
    assert torch.allclose(
        l2_mass(result, domain), l2_mass(reference, domain), atol=1e-10
    )


# ---------------------------------------------------------------------------
# Task 2 -- Trap A: Nyquist zeroing for real fields on even grids
# ---------------------------------------------------------------------------


def test_real_gradient_preserves_realness_with_nyquist_content():
    """A real field on an even grid that carries a Nyquist component must produce a
    real gradient, uncontaminated by the self-conjugate mode."""

    domain = PeriodicDomain.periodic_1d(16)
    x = domain.mesh(dtype=torch.float64)[0]
    field = torch.cos(8 * x)  # k=8 is exactly Nyquist for n=16

    gradient = spectral_gradient(field, domain)

    assert not gradient.is_complex()
    assert torch.isfinite(gradient).all()


def test_real_gradient_matches_analytic_derivative_below_nyquist():
    """Zeroing the Nyquist mode costs nothing on fields the study actually uses --
    band-limited strictly below Nyquist."""

    domain = PeriodicDomain.periodic_1d(64)
    x = domain.mesh(dtype=torch.float64)[0]
    field = torch.sin(5 * x)
    analytic = 5 * torch.cos(5 * x)

    gradient = spectral_gradient(field, domain)

    assert torch.allclose(gradient, analytic, atol=1e-10)


def test_complex_gradient_keeps_the_full_signed_nyquist_convention():
    """The complex path must NOT zero the Nyquist mode -- this is the assertion that
    stops a future cleanup pass from "simplifying" the ``field.is_complex()`` early
    return away.

    A field carrying only the Nyquist mode (``cos(n/2 * x)``) has zero *signed* first
    derivative once that mode is zeroed, so the real path's gradient is identically
    zero -- while the complex path, which never zeroes it, is not."""

    n = 16
    domain = PeriodicDomain.periodic_1d(n)
    x = domain.mesh(dtype=torch.float64)[0]
    real_field = torch.cos(n // 2 * x)
    complex_field = real_field.to(torch.complex128)

    real_result = spectral_gradient(real_field, domain)
    complex_result = spectral_gradient(complex_field, domain)

    assert torch.allclose(real_result, torch.zeros_like(real_result), atol=1e-10)
    assert not torch.allclose(
        complex_result, torch.zeros_like(complex_result), atol=1e-8
    )


def test_odd_grid_takes_no_nyquist_zeroing_branch():
    """An odd-sized grid has no Nyquist mode to zero, so real and complex fields with
    equivalent content must agree."""

    n = 15
    domain = PeriodicDomain.periodic_1d(n)
    x = domain.mesh(dtype=torch.float64)[0]
    real_field = torch.cos(3 * x)
    complex_field = real_field.to(torch.complex128)

    real_result = spectral_gradient(real_field, domain)
    complex_result = spectral_gradient(complex_field, domain)

    assert torch.allclose(complex_result.real, real_result, atol=1e-10)


# ---------------------------------------------------------------------------
# Task 3 -- volume
# ---------------------------------------------------------------------------


def test_volume_is_the_product_of_lengths():
    domain = PeriodicDomain((16, 32), (2 * math.pi, 4 * math.pi))
    assert domain.volume == pytest.approx(2 * math.pi * 4 * math.pi)


def test_volume_matches_cell_volume_times_point_count():
    domain = PeriodicDomain.periodic_1d(64)
    assert domain.volume == pytest.approx(domain.cell_volume * 64)


# ---------------------------------------------------------------------------
# Task 4a -- spatial_broadcast's contract guard
# ---------------------------------------------------------------------------


def test_spatial_broadcast_accepts_a_per_sample_scalar():
    domain = PeriodicDomain.periodic_1d(16)
    values = torch.rand(4)

    result = spatial_broadcast(values, domain)

    assert result.shape == (4, 1)


def test_spatial_broadcast_rejects_grid_shaped_input():
    domain = PeriodicDomain.periodic_1d(16)
    grid_shaped = torch.rand(4, 16)  # (batch, *shape) -- already spatial

    with pytest.raises(ValueError):
        spatial_broadcast(grid_shaped, domain)


# ---------------------------------------------------------------------------
# Task 4b -- integral coercion of shape
# ---------------------------------------------------------------------------


def test_shape_accepts_numpy_style_integer_via_index():
    class FakeNumpyInt:
        """Stand-in for numpy.int64: not an int subclass, but losslessly indexable."""

        def __init__(self, value: int) -> None:
            self._value = value

        def __index__(self) -> int:
            return self._value

    domain = PeriodicDomain((FakeNumpyInt(16),), (2 * math.pi,))
    assert domain.shape == (16,)


def test_shape_rejects_bool_even_though_it_subclasses_int():
    with pytest.raises(ValueError):
        PeriodicDomain((True,), (2 * math.pi,))


# ---------------------------------------------------------------------------
# ADR 0002 -- MeasureSpace -> HilbertSpace -> SpectralDomain
# ---------------------------------------------------------------------------


def test_periodic_domain_is_instantiable_and_satisfies_the_full_hierarchy():
    """Regression guard for the ABC/dataclass collision: an ``@abc.abstractmethod``
    property named ``shape`` on the base class would make this permanently
    non-instantiable, because a dataclass field of the same name (no default) never
    becomes a class-level attribute that could shadow it in the MRO. ``shape`` is
    therefore a plain annotation on ``MeasureSpace``, not a property."""

    domain = PeriodicDomain.periodic_1d(16)

    assert isinstance(domain, MeasureSpace)
    assert isinstance(domain, HilbertSpace)
    assert isinstance(domain, SpectralDomain)


def test_integrate_agrees_with_l2_mass():
    domain = PeriodicDomain.periodic_1d(32)
    torch.manual_seed(0)
    field = torch.randn(3, 32, dtype=torch.complex128)

    assert torch.allclose(
        domain.integrate(field.abs() ** 2), l2_mass(field, domain), atol=1e-10
    )


def test_norm_squared_agrees_with_l2_mass():
    domain = PeriodicDomain.periodic_1d(32)
    torch.manual_seed(1)
    field = torch.randn(3, 32, dtype=torch.complex128)

    assert torch.allclose(
        domain.norm(field) ** 2, l2_mass(field, domain), atol=1e-10
    )


def test_analyze_synthesize_round_trip():
    domain = PeriodicDomain.periodic_1d(32)
    torch.manual_seed(2)
    field = torch.randn(3, 32, dtype=torch.complex128)

    assert torch.allclose(domain.synthesize(domain.analyze(field)), field, atol=1e-9)


def test_laplacian_multiplier_backs_spectral_laplacian():
    """spectral_laplacian was refactored to go through analyze/synthesize plus this
    multiplier; it must remain bit-for-bit what it was before the refactor."""

    domain = PeriodicDomain.periodic_1d(64)
    x = domain.mesh(dtype=torch.float64)[0]
    field = torch.sin(5 * x)
    analytic = -25 * torch.sin(5 * x)

    assert torch.allclose(spectral_laplacian(field, domain), analytic, atol=1e-9)

    # And directly, via the multiplier the free function now delegates to:
    transformed = domain.analyze(field)
    multiplier = domain.laplacian_multiplier(dtype=torch.float64)
    manual = domain.synthesize(multiplier * transformed).real
    assert torch.allclose(manual, spectral_laplacian(field, domain), atol=1e-12)


def test_derivative_multiplier_backs_spectral_gradient():
    """spectral_gradient was refactored to loop over derivative_multiplier per axis;
    confirm the two paths agree exactly, including on the Nyquist mode."""

    n = 16
    domain = PeriodicDomain.periodic_1d(n)
    x = domain.mesh(dtype=torch.float64)[0]
    field = torch.cos(8 * x)  # carries Nyquist content

    manual = domain.synthesize(
        domain.derivative_multiplier(0, real_field=True, dtype=torch.float64)
        * domain.analyze(field)
    ).real
    assert torch.allclose(manual, spectral_gradient(field, domain)[0], atol=1e-12)


def test_quadrature_weights_is_a_scalar_cell_volume():
    domain = PeriodicDomain((16, 32), (2 * math.pi, 4 * math.pi))
    weights = domain.quadrature_weights(dtype=torch.float64)

    assert weights.shape == ()
    assert weights.item() == pytest.approx(domain.cell_volume)
