"""Phase 9: dials on the data-generating equation.  Each recovers the exact case at 0.

    nonlocal:   nu = beta (W_sigma * rho)   breaks locality; still Hamiltonian, still
                U(1), still exactly mass-conserving.  Separates C1 from C2.
    gain/loss:  + i gamma psi               breaks conservation itself, so B's hard
                constraint becomes actively *wrong*.

**Sign convention, derived not assumed.**  Gain means the equation

    i psi_t + alpha Lap psi + beta |psi|^2 psi - V psi = +i gamma psi

so ``psi_t = +gamma psi + i(...)``, the amplitude multiplies by ``exp(gamma dt)`` each
step, and ``d ln M / dt = 2 gamma``.  Writing ``-i gamma psi`` on the right instead
gives ``psi_t = -gamma psi + i(...)`` -- decay, and ``d ln M/dt = -2 gamma``.  The
tests below assert the *rate*, so the convention is pinned by measurement rather than
by the docstring.
"""

from __future__ import annotations

import math

import pytest
import torch

from dataclasses import replace

from spno.config import DataConfig, config_hash
from spno.data.datasets import generate_shard
from spno.domain import PeriodicDomain, l2_mass
from spno.misspecification import MisspecificationConfig
from spno.solvers.perturbed import (
    GainLossSplitStepNLSOperator,
    NonlocalSplitStepNLSOperator,
    SubsteppedOperator,
)
from spno.solvers.split_step import SplitStepNLSOperator, SubsteppedReference

N, DT = 64, 0.01


def _inputs(batch=4, seed=0):
    domain = PeriodicDomain.periodic_1d(N)
    torch.manual_seed(seed)
    field = torch.randn(batch, N, dtype=torch.complex128)
    potential = torch.randn(batch, N, dtype=torch.float64) * 0.3
    alpha = torch.full((batch,), 0.9, dtype=torch.float64)
    beta = torch.full((batch,), 0.4, dtype=torch.float64)
    return domain, field, potential, alpha, beta


@pytest.mark.parametrize(
    "build",
    [
        lambda d: NonlocalSplitStepNLSOperator(d, sigma=0.0),
        lambda d: GainLossSplitStepNLSOperator(d, gamma=0.0),
    ],
)
def test_at_dial_zero_the_perturbed_generator_is_bitwise_the_unperturbed_one(build):
    """Bitwise, not approximately.  A spectral convolution with a sigma=0 kernel has
    multiplier 1, but the FFT round-trip is accurate rather than exact -- so the
    short-circuit at zero is required, not an optimization."""

    domain, field, potential, alpha, beta = _inputs()
    exact = SplitStepNLSOperator(domain)(field, potential, alpha, beta, DT)
    assert torch.equal(build(domain)(field, potential, alpha, beta, DT), exact)


def test_the_zero_dial_short_circuit_is_load_bearing():
    """The paired negative for the bitwise test: without the short-circuit, running the
    sigma=0 kernel through the FFT round-trip is accurate but NOT bitwise identical."""

    domain, field, potential, alpha, beta = _inputs()
    rho = torch.abs(field) ** 2
    roundtrip = torch.fft.ifftn(
        torch.fft.fftn(rho, dim=domain.spatial_axes) * 1.0, dim=domain.spatial_axes
    ).real
    assert not torch.equal(roundtrip, rho)
    assert float(torch.abs(roundtrip - rho).max()) < 1e-12


@pytest.mark.parametrize("sigma", [0.1, 0.5, 1.0])
def test_the_nonlocal_dial_departs_and_still_conserves_mass(sigma):
    """The nonlocal nonlinearity is still a phase, so mass stays exact -- which is why
    it isolates *locality* rather than conservation."""

    domain, field, potential, alpha, beta = _inputs()
    exact = SplitStepNLSOperator(domain)(field, potential, alpha, beta, DT)
    out = NonlocalSplitStepNLSOperator(domain, sigma=sigma)(
        field, potential, alpha, beta, DT
    )
    assert float(torch.abs(out - exact).max()) > 1e-6
    drift = torch.abs(l2_mass(out, domain) / l2_mass(field, domain) - 1)
    assert float(drift.max()) < 1e-13


def test_the_nonlocal_departure_grows_with_sigma():
    domain, field, potential, alpha, beta = _inputs()
    exact = SplitStepNLSOperator(domain)(field, potential, alpha, beta, DT)
    departures = [
        float(
            torch.abs(
                NonlocalSplitStepNLSOperator(domain, sigma=s)(
                    field, potential, alpha, beta, DT
                )
                - exact
            ).max()
        )
        for s in (0.1, 0.3, 1.0)
    ]
    assert departures[0] < departures[1] < departures[2], departures


def test_the_nonlocal_kernel_has_unit_integral_at_every_sigma():
    """W_hat(0) = 1 exactly, so the kernel never rescales the mean density and
    sigma -> 0 is a delta by construction rather than by limit."""

    domain = PeriodicDomain.periodic_1d(N)
    for sigma in (0.1, 0.5, 1.0, 4.0):
        operator = NonlocalSplitStepNLSOperator(domain, sigma=sigma)
        assert float(operator.kernel.reshape(-1)[0]) == pytest.approx(1.0, abs=0.0)


@pytest.mark.parametrize("gamma", [1e-3, 1e-2])
def test_the_gain_dial_grows_mass_at_exactly_the_declared_rate(gamma):
    """Convention: i psi_t + ... = +i gamma psi, so psi_t = ... + gamma psi and
    d ln M / dt = 2 gamma.  The *rate* is tested, not a threshold."""

    domain, field, potential, alpha, beta = _inputs()
    state = GainLossSplitStepNLSOperator(domain, gamma=gamma)(
        field, potential, alpha, beta, DT
    )
    ratio = float((l2_mass(state, domain) / l2_mass(field, domain)).mean())
    assert math.log(ratio) / DT == pytest.approx(2 * gamma, rel=1e-10)


def test_the_gain_dial_rate_is_exact_over_many_steps():
    domain, field, potential, alpha, beta = _inputs()
    gamma = 5e-3
    operator = GainLossSplitStepNLSOperator(domain, gamma=gamma)
    state = field
    for _ in range(50):
        state = operator(state, potential, alpha, beta, DT)
    ratio = float((l2_mass(state, domain) / l2_mass(field, domain)).mean())
    assert math.log(ratio) / (50 * DT) == pytest.approx(2 * gamma, rel=1e-9)


def test_a_negative_gamma_is_loss():
    """The dial is signed: the same operator covers gain and loss."""

    domain, field, potential, alpha, beta = _inputs()
    state = GainLossSplitStepNLSOperator(domain, gamma=-2e-3)(
        field, potential, alpha, beta, DT
    )
    ratio = float((l2_mass(state, domain) / l2_mass(field, domain)).mean())
    assert math.log(ratio) / DT == pytest.approx(-4e-3, rel=1e-10)


def test_a_negative_sigma_is_refused():
    domain = PeriodicDomain.periodic_1d(N)
    with pytest.raises(ValueError, match="non-negative"):
        NonlocalSplitStepNLSOperator(domain, sigma=-0.1)


def test_the_substepped_wrapper_reproduces_the_reference_on_the_exact_inner_step():
    """SubsteppedOperator must be the same loop SubsteppedReference runs, or Phase 9
    would compare a substepped exact case against single-step perturbed ones and read
    the splitting error as a misspecification effect."""

    domain, field, potential, alpha, beta = _inputs()
    wrapped = SubsteppedOperator(SplitStepNLSOperator(domain), 32)
    reference = SubsteppedReference(domain, 32)
    assert torch.equal(
        wrapped(field, potential, alpha, beta, DT),
        reference(field, potential, alpha, beta, DT),
    )


# ---------------------------------------------------------------------------
# Task 15: MisspecificationConfig and generator injection
# ---------------------------------------------------------------------------


def test_the_exact_case_reuses_the_production_identifier():
    """Dial zero must not orphan the 206 MB of shards already on disk."""

    data = DataConfig()
    assert MisspecificationConfig().identifier(data) == config_hash(data)


def test_every_nonzero_dial_gets_its_own_identifier():
    data = DataConfig()
    identifiers = {
        MisspecificationConfig(nonlocal_sigma=s, gain_loss_gamma=g).identifier(data)
        for s, g in [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.0, 1e-3), (0.0, 1e-2)]
    }
    assert len(identifiers) == 5


def test_turning_two_dials_at_once_is_refused():
    """A simultaneous perturbation cannot be attributed to either broken assumption."""

    # The plan's snippet paired the message "turn one dial at a time" with
    # match="one at a time", which cannot match it -- "dial" sits between.
    with pytest.raises(ValueError, match="one dial at a time"):
        MisspecificationConfig(nonlocal_sigma=0.5, gain_loss_gamma=1e-3)


def test_the_exact_dial_reference_is_the_substepped_reference_itself():
    assert type(MisspecificationConfig().reference(DataConfig())) is SubsteppedReference


def test_a_shard_generated_at_dial_zero_is_bitwise_the_production_shard():
    """The end-to-end bitwise test: injection must not perturb anything."""

    small = replace(DataConfig(), n_test=2, steps=3, grid_size=32)
    baseline = generate_shard(small, "test")
    injected = generate_shard(
        small, "test", reference=MisspecificationConfig().reference(small)
    )
    assert torch.equal(injected.trajectories, baseline.trajectories)


def test_a_dialled_shard_actually_differs():
    """The paired negative: if injection changed nothing, Phase 9 would sweep noise."""

    small = replace(DataConfig(), n_test=2, steps=3, grid_size=32)
    baseline = generate_shard(small, "test")
    perturbed = generate_shard(
        small,
        "test",
        reference=MisspecificationConfig(nonlocal_sigma=0.5).reference(small),
    )
    assert not torch.equal(perturbed.trajectories, baseline.trajectories)


def test_a_negative_sigma_is_refused_by_the_config():
    with pytest.raises(ValueError, match="non-negative"):
        MisspecificationConfig(nonlocal_sigma=-0.1)
