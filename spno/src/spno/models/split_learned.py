"""Model C: structure-preserving learned split-step operators.

The step is

    Phi^theta_dt = K^theta_{dt/2} . N^theta_dt . K^theta_{dt/2}

with ``K^theta: psi_hat -> exp(i dt kappa_theta(|k|^2, alpha, beta)) psi_hat`` and
``N^theta: psi -> exp(i dt nu_theta(rho, V, alpha, beta)) psi``, ``rho = |psi|^2``, and
both ``kappa_theta`` and ``nu_theta`` real-valued.

**Theorem.** For every theta -- untrained weights included:

1. ``Phi^theta`` preserves the discrete mass exactly.  Each substep is a modulus-one
   multiplier and the FFT round trip is unitary (Parseval).
2. ``Phi^theta_{-dt} . Phi^theta_dt = Id`` exactly.  ``N`` leaves ``rho`` pointwise
   unchanged, so the backward step reads the *same* phase field and cancels it.
3. ``Phi^theta`` is the Strang splitting of a learned Hamiltonian
   ``H_theta = H_K,theta + H_N,theta`` with ``H_K = -int kappa(|k|^2)|psi_hat|^2`` and
   ``H_N = -int F(rho, V)``, ``dF/drho = nu``.  An antiderivative of ``nu`` in the
   scalar ``rho`` always exists, and ``rho`` is constant along the local flow, so each
   substep is an *exact* flow of its sub-Hamiltonian.  Hence ``Phi^theta`` is
   symplectic and time-symmetric.
4. ``Phi^theta(e^{i c} psi) = e^{i c} Phi^theta(psi)``: global U(1) equivariance, since
   ``rho`` and ``|psi_hat|`` are phase-blind.

(3) is the sharp prediction the study tests: the model does not conserve the *true*
Hamiltonian, but it exactly-symplectically integrates a learned one, so its energy
error should be **bounded** (governed by ``||H_theta - H_true||``, a model-error term)
rather than **secular**.  Phases 2-3 measured secular energy drift for the FNO and for
the mass-projected FNO, so this is falsifiable against a real baseline.

**What breaks the guarantees.**  (2), (3), and (4) hold only because ``nu`` sees
``psi`` through ``rho`` alone.  :class:`FullFieldPhaseSplitStep` (C3) feeds ``Re psi``
and ``Im psi`` to the phase net instead: mass survives, U(1) is destroyed outright, and
reversibility degrades from *exact* to *second order* -- measured at order 2.00, so at
small ``dt`` its violation (~1e-9) is small enough that an absolute threshold would call
it exact.  The defensible claim is therefore "rho-only parameter sharing makes
reversibility exact rather than second-order", not "C3 is irreversible".

C3 exists as the control that isolates "mass alone" -- the same guarantee Model B has --
from the rest of the structure, and as a guard against the theorem tests being vacuous.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..domain import PeriodicDomain
from .base import StepOperator
from .fno import SpectralConv1d

Tensor = torch.Tensor

KineticMode = Literal["K0", "K1", "K2"]
LocalMode = Literal["L0", "L1", "L2"]


class PointwisePhaseMLP(nn.Module):
    """Real scalar field from pointwise features plus broadcast PDE parameters.

    Follows ``ParametricPhaseMLP`` in the tutorial toolkit: features are
    ``(batch, ..., feature_dim)``, parameters are ``(batch, parameter_dim)``, and the
    output is one **real** value per point.
    """

    def __init__(
        self, feature_dim: int, parameter_dim: int, width: int = 32, depth: int = 3
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or parameter_dim <= 0 or width <= 0 or depth < 2:
            raise ValueError("feature_dim, parameter_dim, width > 0 and depth >= 2")
        self.feature_dim = feature_dim
        self.parameter_dim = parameter_dim
        layers: list[nn.Module] = [nn.Linear(feature_dim + parameter_dim, width), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: Tensor, parameters: Tensor) -> Tensor:
        if features.ndim < 2 or features.shape[-1] != self.feature_dim:
            raise ValueError("features must have shape (batch, ..., feature_dim)")
        if parameters.shape != (features.shape[0], self.parameter_dim):
            raise ValueError("parameters must have shape (batch, parameter_dim)")
        expanded = parameters.reshape(
            parameters.shape[0], *((1,) * (features.ndim - 2)), self.parameter_dim
        ).expand(*features.shape[:-1], self.parameter_dim)
        return self.network(torch.cat((features, expanded), dim=-1)).squeeze(-1)


class FieldPhaseFNO(nn.Module):
    """Real phase field from real input channels, via the Model A backbone.

    Used by C2 so the structured model and the black-box baseline differ only in
    *where* the network output enters -- as a field, or as a phase -- rather than in
    capacity.
    """

    def __init__(
        self, in_channels: int, *, modes: int = 16, width: int = 64, n_layers: int = 4
    ) -> None:
        super().__init__()
        self.lift = nn.Linear(in_channels, width)
        self.spectral = nn.ModuleList(
            SpectralConv1d(width, width, modes) for _ in range(n_layers)
        )
        self.local = nn.ModuleList(nn.Conv1d(width, width, 1) for _ in range(n_layers))
        self.project = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, channels: Tensor) -> Tensor:
        """``channels``: ``(batch, n, in_channels)`` real -> ``(batch, n)`` real."""

        hidden = self.lift(channels).permute(0, 2, 1)
        for index, (spectral, local) in enumerate(zip(self.spectral, self.local)):
            updated = spectral(hidden) + local(hidden)
            hidden = F.gelu(updated) if index < len(self.spectral) - 1 else updated
        return self.project(hidden.permute(0, 2, 1)).squeeze(-1)


class KineticPhase(nn.Module):
    """Learned kinetic rate ``kappa(|k|^2, alpha, beta)``; the true value is ``-alpha|k|^2``.

    Three rungs, because how much of the product ``alpha * |k|^2`` is handed to the
    network determines whether a win above ``k_wrap`` means the model *learned* the
    dispersion relation or was *told* it:

    ``K0``  free MLP on ``(k^2, alpha, beta)`` -- must discover the product itself.
    ``K1``  the product ``alpha * k^2`` supplied as an extra feature.
    ``K2``  ``kappa = -alpha k^2 (1 + MLP(...))`` -- functional form imposed, network
            only corrects it.  Nearly oracle, so it is an ablation and not the headline.

    ``k^2`` is normalized by its grid maximum before entering the network; unnormalized
    it reaches 1024 at N=64 and saturates every tanh.
    """

    def __init__(
        self,
        domain: PeriodicDomain,
        *,
        mode: KineticMode = "K0",
        width: int = 32,
        parameter_dim: int = 2,
    ) -> None:
        super().__init__()
        if mode not in ("K0", "K1", "K2"):
            raise ValueError("mode must be one of K0, K1, K2")
        self.domain = domain
        self.mode = mode
        k_squared = domain.wave_number_squared()
        self.register_buffer("k_squared", k_squared)
        self.register_buffer("k_squared_scale", k_squared.max().clamp_min(1.0))
        feature_dim = 2 if mode == "K1" else 1
        self.network = PointwisePhaseMLP(feature_dim, parameter_dim, width=width)
        if mode == "K2":
            # Start as the exact rate: the correction is zero at initialization.
            nn.init.zeros_(self.network.network[-1].weight)
            nn.init.zeros_(self.network.network[-1].bias)

    def forward(self, parameters: Tensor) -> Tensor:
        """``parameters``: ``(batch, parameter_dim)`` with alpha first -> ``(batch, n)``."""

        batch = parameters.shape[0]
        k_squared = self.k_squared.to(parameters.dtype)
        normalized = (k_squared / self.k_squared_scale).reshape(1, -1, 1).expand(
            batch, -1, 1
        )
        alpha = parameters[:, :1]
        if self.mode == "K1":
            product = normalized.squeeze(-1) * alpha
            features = torch.stack((normalized.squeeze(-1), product), dim=-1)
        else:
            features = normalized
        raw = self.network(features, parameters)
        if self.mode == "K2":
            exact = -alpha * k_squared.reshape(1, -1)
            return exact * (1 + raw)
        return raw


class LocalPhaseLadder(nn.Module):
    """Learned local rate ``nu(rho, V, alpha, beta)``; the true value is ``beta*rho - V``.

    The mirror image of :class:`KineticPhase`, and it exists for the same reason.  The
    true rate is a *bilinear* form -- a product ``beta*rho`` plus a linear ``-V`` -- and
    a tanh MLP given ``(rho, V)`` as features with ``(alpha, beta)`` as parameters has
    to discover that product on its own.  Measured on the production distribution, it
    does not: a free MLP plateaus at **2.1% relative RMSE** on ``nu`` (and a wider,
    deeper one at 2.6% -- an optimization limit, not a capacity one).

    Measured end-to-end (kinetic held at the exact rate, local net fitted, one-step
    error against the substepped reference), the cost is real but modest:

        L0   1.18 x eps_split
        L1   1.03 x eps_split
        L2   0.93 x eps_split

    So the approximation floor inflates C1's one-step error by ~18% at L0, not by the
    factor of 3 a back-of-the-envelope ``dt * |nu| * e`` estimate suggests -- the phase
    error is partly incoherent across the grid and does not accumulate as that bound
    assumes.  The Phase 4 acceptance criterion ("C1 reaches ~eps_split") therefore
    remains testable at L0.

    L2 landing slightly *below* the floor is expected, not anomalous: it matches the
    Phase 0 finding that a free correction to the generators can absorb a few percent
    of the Strang commutator.

    ``L0``  free MLP on ``(rho, V)`` -- must discover ``beta*rho`` itself.
    ``L1``  the product ``beta*rho`` supplied as an extra feature.
    ``L2``  ``nu = beta*rho - V + MLP(...)`` with the correction zero-initialized.

    Report ``L0`` as the headline for the same reason ``K0`` is the headline: handing
    the model the answer makes a win above ``k_wrap`` uninterpretable.  But report the
    ladder, because if ``L0`` cannot reach the floor that is itself a finding about how
    hard the *learning* problem is, distinct from how expressive the class is.
    """

    def __init__(
        self, *, mode: LocalMode = "L0", width: int = 32, parameter_dim: int = 2
    ) -> None:
        super().__init__()
        if mode not in ("L0", "L1", "L2"):
            raise ValueError("mode must be one of L0, L1, L2")
        self.mode = mode
        feature_dim = 3 if mode == "L1" else 2
        self.network = PointwisePhaseMLP(feature_dim, parameter_dim, width=width)
        if mode == "L2":
            nn.init.zeros_(self.network.network[-1].weight)
            nn.init.zeros_(self.network.network[-1].bias)

    def forward(
        self, density: Tensor, potential: Tensor, alpha: Tensor, beta: Tensor
    ) -> Tensor:
        dtype = density.dtype
        beta_grid = beta.to(dtype).reshape(-1, 1)
        parameters = torch.stack((alpha.to(dtype), beta.to(dtype)), dim=-1)
        if self.mode == "L1":
            features = torch.stack(
                (density, potential, beta_grid * density), dim=-1
            )
        else:
            features = torch.stack((density, potential), dim=-1)
        raw = self.network(features, parameters)
        if self.mode == "L2":
            return beta_grid * density - potential + raw
        return raw


class LearnedSplitStep(StepOperator):
    """Symmetric split step with learned real phase rates.

    Mass preservation, reversibility, symplecticity, and U(1) equivariance are
    properties of the *architecture*, so they hold at random initialization.  The tests
    assert them on untrained weights for exactly that reason.
    """

    supports_time_reversal = True

    def __init__(
        self,
        domain: PeriodicDomain,
        *,
        kinetic_mode: KineticMode | None = "K0",
        trained_dt: float | None = None,
    ) -> None:
        super().__init__(domain, trained_dt)
        if domain.dim != 1:
            raise NotImplementedError("the learned split step is implemented for 1D")
        # ``None`` means the subclass supplies its own kinetic half-step.  Skipping
        # construction is not the same as constructing and then deleting: ``del
        # self.kinetic`` would drop the module from ``_modules`` and leave a state_dict
        # incompatible with its siblings, breaking checkpoint round-trips across the
        # family.
        self.kinetic = (
            KineticPhase(domain, mode=kinetic_mode) if kinetic_mode is not None else None
        )

    def local_phase(
        self, field: Tensor, potential: Tensor, alpha: Tensor, beta: Tensor
    ) -> Tensor:
        raise NotImplementedError

    def _kinetic_half(self, field: Tensor, parameters: Tensor, dt: float) -> Tensor:
        rate = self.kinetic(parameters)
        transformed = torch.fft.fftn(field, dim=self.domain.spatial_axes)
        return torch.fft.ifftn(
            transformed * torch.exp(1j * (0.5 * float(dt)) * rate),
            dim=self.domain.spatial_axes,
        )

    def step(
        self,
        field: Tensor,
        potential: Tensor,
        alpha: Tensor,
        beta: Tensor,
        dt: float,
    ) -> Tensor:
        parameters = torch.stack(
            (alpha.to(field.real.dtype), beta.to(field.real.dtype)), dim=-1
        )
        midpoint = self._kinetic_half(field, parameters, dt)
        rate = self.local_phase(midpoint, potential, alpha, beta)
        if rate.is_complex():
            raise ValueError("the local phase rate must be real")
        if rate.shape != field.shape:
            raise ValueError("the local phase rate must have one value per grid point")
        rotated = midpoint * torch.exp(1j * float(dt) * rate)
        return self._kinetic_half(rotated, parameters, dt)


class DensityPhaseSplitStep(LearnedSplitStep):
    """C1: pointwise local phase ``nu(rho, V, alpha, beta)``.

    The strongest inductive bias in the comparison: mass, reversibility, symplecticity,
    and U(1) equivariance all hold by construction.  Roughly a thousand parameters
    against the FNO's ~288k, so the parameter counts must be reported next to any claim
    about which model is better.
    """

    def __init__(
        self,
        domain: PeriodicDomain,
        *,
        kinetic_mode: KineticMode = "K0",
        local_mode: LocalMode = "L0",
        width: int = 32,
        trained_dt: float | None = None,
    ) -> None:
        super().__init__(domain, kinetic_mode=kinetic_mode, trained_dt=trained_dt)
        self.local = LocalPhaseLadder(mode=local_mode, width=width)

    def local_phase(
        self, field: Tensor, potential: Tensor, alpha: Tensor, beta: Tensor
    ) -> Tensor:
        return self.local(torch.abs(field) ** 2, potential, alpha, beta)


class FieldDensityPhaseSplitStep(LearnedSplitStep):
    """C2: nonlocal local phase, an FNO reading ``(rho, V, alpha, beta)``.

    Expressivity-matched to Model A -- same backbone, same hyperparameters, comparable
    parameter count -- so ``A vs C2`` is the fair fight: identical capacity, differing
    only in whether the network output is the field itself or a phase applied to it.

    Mass, reversibility, and U(1) equivariance survive, because the phase still depends
    on ``psi`` only through ``rho``.  Symplecticity does **not**: a nonlocal ``nu`` is
    the variational derivative of a density functional only for a symmetric kernel, and
    a free FNO is not one.  Note that symmetric-and-reversible methods already give
    bounded energy via reversible-KAM, so C1 and C2 are unlikely to separate on energy
    drift -- frame that comparison as sample efficiency versus expressivity.
    """

    def __init__(
        self,
        domain: PeriodicDomain,
        *,
        kinetic_mode: KineticMode = "K0",
        local_mode: LocalMode = "L0",
        modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
        trained_dt: float | None = None,
    ) -> None:
        super().__init__(domain, kinetic_mode=kinetic_mode, trained_dt=trained_dt)
        if local_mode == "L1":
            raise ValueError("L1 is a pointwise-feature rung; C2 supports L0 and L2")
        self.local_mode = local_mode
        self.local = FieldPhaseFNO(4, modes=modes, width=width, n_layers=n_layers)

    def local_phase(
        self, field: Tensor, potential: Tensor, alpha: Tensor, beta: Tensor
    ) -> Tensor:
        density = torch.abs(field) ** 2
        batch, n = density.shape
        channels = torch.stack(
            (
                density,
                potential,
                alpha.to(density.dtype).reshape(batch, 1).expand(batch, n),
                beta.to(density.dtype).reshape(batch, 1).expand(batch, n),
            ),
            dim=-1,
        )
        raw = self.local(channels)
        if self.local_mode == "L2":
            return beta.to(density.dtype).reshape(-1, 1) * density - potential + raw
        return raw


class FullFieldPhaseSplitStep(LearnedSplitStep):
    """C3 (control): the phase net reads ``Re psi`` and ``Im psi`` instead of ``rho``.

    Deliberately weaker.  Mass is still exact -- the multiplier is still modulus one --
    but U(1) equivariance is destroyed outright, and reversibility degrades from
    *exact* to *second-order*: the local step rotates ``psi`` by ``dt*nu``, so the
    backward step reads a slightly different input and the round trip misses by
    ``O(dt^2)`` (measured: order 2.00, ~1e-9 at ``dt=0.01`` with untrained weights,
    against ~7e-16 and dt-independent for C1/C2).

    Stating that carefully matters.  "C3 is not reversible" would be wrong -- at small
    ``dt`` its violation is tiny, and an absolute threshold would call it exact.  The
    claim the thesis can defend is that ``rho``-only parameter sharing makes
    reversibility **exact rather than second-order**.

    Two jobs: it is the direct control for Model B (both preserve mass and nothing
    else, at very different capacities), and it is the negative case proving the C1/C2
    reversibility tests are not vacuous.
    """

    supports_time_reversal = True  # the step is well-defined at -dt; it just is not an inverse

    def __init__(
        self,
        domain: PeriodicDomain,
        *,
        kinetic_mode: KineticMode = "K0",
        modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
        trained_dt: float | None = None,
    ) -> None:
        super().__init__(domain, kinetic_mode=kinetic_mode, trained_dt=trained_dt)
        self.local = FieldPhaseFNO(5, modes=modes, width=width, n_layers=n_layers)

    def local_phase(
        self, field: Tensor, potential: Tensor, alpha: Tensor, beta: Tensor
    ) -> Tensor:
        batch, n = field.shape
        real_dtype = field.real.dtype
        channels = torch.stack(
            (
                field.real,
                field.imag,
                potential,
                alpha.to(real_dtype).reshape(batch, 1).expand(batch, n),
                beta.to(real_dtype).reshape(batch, 1).expand(batch, n),
            ),
            dim=-1,
        )
        return self.local(channels)


class ExactSplitStep(LearnedSplitStep):
    """The true rates in the learned architecture's shape: a correctness fixture.

    If this does not reproduce :class:`SplitStepNLSOperator` to machine precision, the
    learned models are wired wrong -- sign convention, half-step placement, or
    normalization -- and every downstream number would be meaningless.
    """

    def __init__(self, domain: PeriodicDomain, *, trained_dt: float | None = None) -> None:
        super().__init__(domain, kinetic_mode=None, trained_dt=trained_dt)

    def _kinetic_half(self, field: Tensor, parameters: Tensor, dt: float) -> Tensor:
        alpha = parameters[:, :1]
        k_squared = self.domain.wave_number_squared(
            device=field.device, dtype=field.real.dtype
        ).reshape(1, -1)
        transformed = torch.fft.fftn(field, dim=self.domain.spatial_axes)
        rate = -alpha * k_squared
        return torch.fft.ifftn(
            transformed * torch.exp(1j * (0.5 * float(dt)) * rate),
            dim=self.domain.spatial_axes,
        )

    def local_phase(
        self, field: Tensor, potential: Tensor, alpha: Tensor, beta: Tensor
    ) -> Tensor:
        density = torch.abs(field) ** 2
        return beta.to(density.dtype).reshape(-1, 1) * density - potential
