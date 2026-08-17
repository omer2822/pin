"""Phase 8: which parameters are grid-independent, and which only look like it.

The headline trap: ``KineticPhase`` normalizes ``k^2`` by the *grid* maximum (1024 at
N=64, 4096 at N=128), so naively rebuilding its buffer on a finer grid sends the same
physical ``k`` to a different network input and silently changes ``kappa`` everywhere
without touching a weight.  Freezing the normalizer fixes that, at the honest cost that
new modes enter outside the range the MLP ever saw -- which :func:`grid_dependence`
reports rather than hides.
"""

from __future__ import annotations

import pytest
import torch

from spno.domain import PeriodicDomain
from spno.equations.nls import plane_wave
from spno.evaluation.resolution import grid_dependence, rebind_domain, spectral_resample
from spno.models.fno import FNOStepOperator, SpectralConv1d
from spno.models.projected import MassProjectedOperator
from spno.models.split_learned import DensityPhaseSplitStep
from spno.precision import widen_to_double

COARSE = PeriodicDomain.periodic_1d(64)
FINE = PeriodicDomain.periodic_1d(128)


# ---------------------------------------------------------------------------
# spectral_resample
# ---------------------------------------------------------------------------


def test_resampling_a_band_limited_field_up_and_back_is_the_identity():
    torch.manual_seed(0)
    hat = torch.zeros(2, 64, dtype=torch.complex128)
    hat[:, :9] = torch.randn(2, 9, dtype=torch.complex128)
    hat[:, -8:] = torch.randn(2, 8, dtype=torch.complex128)
    field = torch.fft.ifftn(hat, dim=(-1,))
    roundtrip = spectral_resample(spectral_resample(field, COARSE, FINE), FINE, COARSE)
    assert float(torch.abs(roundtrip - field).max()) < 1e-14


def test_resampling_preserves_a_plane_wave_amplitude():
    field = plane_wave(COARSE, (5,), amplitude=0.7).unsqueeze(0)
    upsampled = spectral_resample(field, COARSE, FINE)
    assert float(torch.abs(upsampled).max()) == pytest.approx(0.7, rel=1e-12)


def test_resampling_refuses_a_field_with_energy_at_the_source_nyquist():
    """The Nyquist mode is self-conjugate; upsampling it has no unambiguous answer."""

    field = plane_wave(COARSE, (32,), amplitude=0.7).unsqueeze(0)
    with pytest.raises(ValueError, match="Nyquist"):
        spectral_resample(field, COARSE, FINE)


def test_resampling_to_the_same_grid_is_the_identity():
    torch.manual_seed(0)
    field = torch.randn(2, 64, dtype=torch.complex128)
    assert float(torch.abs(spectral_resample(field, COARSE, COARSE) - field).max()) < 1e-14


# ---------------------------------------------------------------------------
# rebind_domain
# ---------------------------------------------------------------------------


def test_rebinding_c1_keeps_the_learned_kinetic_rate_at_the_same_physical_k():
    """The trap: k^2 is normalized by the *grid* maximum, so a naive rebuild of the
    buffer would change kappa at every physical k without changing a single weight."""

    torch.manual_seed(0)
    model = widen_to_double(DensityPhaseSplitStep(COARSE, trained_dt=0.01))
    parameters = torch.tensor([[0.9, 0.3]], dtype=torch.float64)
    before = model.kinetic(parameters)[0, 8].item()  # physical k = 8
    rebind_domain(model, FINE)
    after = model.kinetic(parameters)[0, 8].item()  # same physical k = 8
    assert after == pytest.approx(before, rel=1e-12)


def test_a_naive_rebuild_would_have_changed_it():
    """The paired negative.  Without freezing the normalizer this test's `after` moves,
    so the positive above is not vacuous."""

    torch.manual_seed(0)
    model = widen_to_double(DensityPhaseSplitStep(COARSE, trained_dt=0.01))
    parameters = torch.tensor([[0.9, 0.3]], dtype=torch.float64)
    before = model.kinetic(parameters)[0, 8].item()

    model.kinetic.domain = FINE
    fresh = FINE.wave_number_squared().to(model.kinetic.k_squared.dtype)
    model.kinetic.register_buffer("k_squared", fresh)
    model.kinetic.register_buffer("k_squared_scale", fresh.max().clamp_min(1.0))
    naive = model.kinetic(parameters)[0, 8].item()

    assert abs(naive - before) > 1e-9


def test_rebinding_preserves_the_buffer_dtype_and_exact_integers():
    """The amendment: freeze the normalizer's VALUE, not its tensor.

    A widened model carries float64 buffers; a rebind that recomputed them at the
    default dtype would reintroduce the float64-on-float32 mismatch that barred the C
    family from MPS.
    """

    torch.manual_seed(0)
    model = widen_to_double(DensityPhaseSplitStep(COARSE, trained_dt=0.01))
    rebind_domain(model, FINE)
    assert model.kinetic.k_squared.dtype == torch.float64
    assert model.kinetic.k_squared_scale.dtype == torch.float64
    assert torch.equal(model.kinetic.k_squared, torch.round(model.kinetic.k_squared))


def test_rebinding_matches_constructing_at_the_new_grid_bitwise():
    """The invariant the dtype path exists to protect.

    Building ``2*pi*fftfreq`` straight into a float64 buffer lands ~9.1e-13 from the
    exact integers at N=128, while ``__init__`` builds in float64 and *narrows*, which
    snaps onto them.  A rebind that skipped the narrow would leave a rebound model
    carrying error a freshly-constructed one does not have -- above the 1e-13 bounds the
    structural suite asserts.
    """

    torch.manual_seed(0)
    rebound = widen_to_double(DensityPhaseSplitStep(COARSE, trained_dt=0.01))
    rebind_domain(rebound, FINE)

    torch.manual_seed(0)
    constructed = widen_to_double(DensityPhaseSplitStep(FINE, trained_dt=0.01))

    assert torch.equal(rebound.kinetic.k_squared, constructed.kinetic.k_squared)


def test_rebinding_keeps_a_float32_model_free_of_float64_buffers():
    """Global Constraint 3: one float64 buffer bars the model from MPS entirely."""

    torch.manual_seed(0)
    model = DensityPhaseSplitStep(COARSE, trained_dt=0.01)
    rebind_domain(model, FINE)
    assert model.kinetic.k_squared.dtype == torch.get_default_dtype()
    assert not any(b.dtype == torch.float64 for b in model.buffers())


def test_rebinding_reports_the_extrapolation_it_introduces():
    torch.manual_seed(0)
    model = widen_to_double(DensityPhaseSplitStep(COARSE, trained_dt=0.01))
    rebind_domain(model, FINE)
    assert "outside" in grid_dependence(model)["kinetic.k_squared"]


def test_grid_dependence_is_quiet_when_no_extrapolation_was_introduced():
    """Paired negative: the word 'outside' must not appear for an unmoved model."""

    torch.manual_seed(0)
    model = widen_to_double(DensityPhaseSplitStep(COARSE, trained_dt=0.01))
    assert "outside" not in grid_dependence(model)["kinetic.k_squared"]


def test_the_fno_spectral_weights_are_indexed_by_mode_and_need_no_rebinding():
    torch.manual_seed(0)
    model = widen_to_double(FNOStepOperator(COARSE, modes=16, trained_dt=0.01))
    weights = model.spectral[0].weight.clone()
    rebind_domain(model, FINE)
    assert torch.equal(model.spectral[0].weight, weights)
    assert model.domain == FINE


def test_rebinding_the_fno_beyond_its_mode_budget_is_refused():
    torch.manual_seed(0)
    model = FNOStepOperator(COARSE, modes=16, trained_dt=0.01)
    with pytest.raises(ValueError, match="modes"):
        rebind_domain(model, PeriodicDomain.periodic_1d(16))


def test_rebinding_recurses_through_the_mass_projection():
    torch.manual_seed(0)
    model = MassProjectedOperator(FNOStepOperator(COARSE, modes=16, trained_dt=0.01))
    rebind_domain(model, FINE)
    assert model.domain == FINE
    assert model.core.domain == FINE


def test_rebinding_an_unknown_model_names_the_class():
    class Mystery:
        pass

    with pytest.raises(NotImplementedError, match="Mystery"):
        rebind_domain(Mystery(), FINE)


# ---------------------------------------------------------------------------
# The 06_fno_core.py borrowings: dtype guard and mode budget
# ---------------------------------------------------------------------------


def test_spectral_conv_rejects_a_dtype_mismatch_instead_of_promoting():
    """Borrowed from 06_fno_core.py:110,:187.  Global Constraint 1 documents that
    ``.to(torch.float64)`` casts complex spectral weights to REAL, destroying the
    operator with only a warning.  Without a guard the failure is silent."""

    torch.manual_seed(0)
    layer = SpectralConv1d(2, 2, modes=4)  # cfloat weights
    with pytest.raises(ValueError, match="dtype"):
        layer(torch.randn(1, 2, 32, dtype=torch.float64))


def test_spectral_conv_rejects_real_weights_the_destroyed_operator_case():
    """The exact silent failure Constraint 1 names, now an exception.

    The real-weight state is constructed directly rather than via ``.to(torch.float64)``
    so the test asserts *this* guard rather than torch's warning behaviour -- whose
    UserWarning is deduplicated by the interpreter's registry once another test in the
    session has already triggered it, making a ``pytest.warns`` here order-dependent.
    """

    torch.manual_seed(0)
    layer = SpectralConv1d(2, 2, modes=4)
    layer.weight = torch.nn.Parameter(layer.weight.real.clone())  # the destroyed state
    with pytest.raises(ValueError, match="complex"):
        layer(torch.randn(1, 2, 32))


def test_spectral_conv_still_accepts_matched_dtypes():
    """The paired positive: the guard must not break the normal paths."""

    torch.manual_seed(0)
    single = SpectralConv1d(2, 2, modes=4)
    assert single(torch.randn(1, 2, 32)).shape == (1, 2, 32)

    double = widen_to_double(SpectralConv1d(2, 2, modes=4))
    assert double(torch.randn(1, 2, 32, dtype=torch.float64)).shape == (1, 2, 32)
