"""Structure-preserving neural operators for parametric PDEs.

This final tutorial is deliberately a ``# %%`` cell-based Python file: it can
be read interactively like a notebook and executed or tested like a script.
It starts with geometry because every later constraint depends on knowing what
the spatial axes mean.
"""

from __future__ import annotations

# %% Imports and periodic geometry

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


@dataclass(frozen=True)
class PeriodicDomain:
    """Uniform endpoint-free rectangular periodic grid in any dimension."""

    shape: tuple[int, ...]
    lengths: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "lengths", tuple(float(v) for v in self.lengths))
        if not self.shape:
            raise ValueError("shape must contain at least one spatial dimension")
        if len(self.shape) != len(self.lengths):
            raise ValueError("shape and lengths must have the same dimension")
        if any(not isinstance(n, int) or n <= 0 for n in self.shape):
            raise ValueError("every grid size must be a positive integer")
        if any(not math.isfinite(length) or length <= 0 for length in self.lengths):
            raise ValueError("every domain length must be positive and finite")

    @property
    def dim(self) -> int:
        return len(self.shape)

    @property
    def spacing(self) -> tuple[float, ...]:
        return tuple(length / n for n, length in zip(self.shape, self.lengths))

    @property
    def cell_volume(self) -> float:
        return math.prod(self.spacing)

    @property
    def spatial_axes(self) -> tuple[int, ...]:
        return tuple(range(-self.dim, 0))

    def mesh(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> tuple[Tensor, ...]:
        axes = [
            torch.arange(n, device=device, dtype=dtype) * dx
            for n, dx in zip(self.shape, self.spacing)
        ]
        return tuple(torch.meshgrid(*axes, indexing="ij"))

    def wave_vectors(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> tuple[Tensor, ...]:
        axes = [
            2 * math.pi * torch.fft.fftfreq(n, d=dx, device=device, dtype=dtype)
            for n, dx in zip(self.shape, self.spacing)
        ]
        return tuple(torch.meshgrid(*axes, indexing="ij"))

    def validate_field(self, field: Tensor) -> None:
        if field.ndim < self.dim or tuple(field.shape[-self.dim :]) != self.shape:
            raise ValueError(
                f"field must end in spatial shape {self.shape}; got {tuple(field.shape)}"
            )


# %% Continuous coordinate derivatives: PINN/autograd geometry


def _validate_coordinate_values(values: Tensor, coordinates: Tensor) -> None:
    if coordinates.ndim != 2 or coordinates.shape[1] == 0:
        raise ValueError("coordinates must have shape (points, dimensions)")
    if values.ndim != 2 or values.shape[0] != coordinates.shape[0]:
        raise ValueError("values and coordinates must have aligned point axes")
    if not coordinates.requires_grad or not values.requires_grad:
        raise ValueError("values must be differentiable functions of coordinates")


def gradient(y: Tensor, coordinates: Tensor) -> Tensor:
    """Gradient of aligned scalar samples: ``(N,1) -> (N,D)``."""

    _validate_coordinate_values(y, coordinates)
    if y.shape[1] != 1:
        raise ValueError("gradient expects one scalar value per point")
    result = torch.autograd.grad(
        y,
        coordinates,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
    )[0]
    if result is None:
        raise ValueError("values are not connected to coordinates")
    return result


def spatial_jacobian(field: Tensor, coordinates: Tensor) -> Tensor:
    """Jacobian of vector samples, shaped ``(N, components, dimensions)``."""

    _validate_coordinate_values(field, coordinates)
    columns = [gradient(field[:, c : c + 1], coordinates) for c in range(field.shape[1])]
    return torch.stack(columns, dim=1)


def divergence(field: Tensor, coordinates: Tensor) -> Tensor:
    """Divergence of an ND vector field sampled at independent coordinates."""

    _validate_coordinate_values(field, coordinates)
    if field.shape[1] != coordinates.shape[1]:
        raise ValueError("divergence requires one field component per dimension")
    jacobian = spatial_jacobian(field, coordinates)
    return torch.diagonal(jacobian, dim1=1, dim2=2).sum(dim=1, keepdim=True)


def curl(field: Tensor, coordinates: Tensor) -> Tensor:
    """Scalar 2D curl or vector 3D curl from ``spatial_jacobian``."""

    _validate_coordinate_values(field, coordinates)
    dim = coordinates.shape[1]
    if field.shape[1] != dim or dim not in (2, 3):
        raise ValueError("curl requires a two- or three-dimensional vector field")
    jacobian = spatial_jacobian(field, coordinates)
    if dim == 2:
        return (jacobian[:, 1, 0] - jacobian[:, 0, 1]).unsqueeze(1)
    return torch.stack(
        (
            jacobian[:, 2, 1] - jacobian[:, 1, 2],
            jacobian[:, 0, 2] - jacobian[:, 2, 0],
            jacobian[:, 1, 0] - jacobian[:, 0, 1],
        ),
        dim=1,
    )


def laplacian(y: Tensor, coordinates: Tensor) -> Tensor:
    """Sum of the pure second derivatives of scalar coordinate samples."""

    first = gradient(y, coordinates)
    terms = []
    for axis in range(coordinates.shape[1]):
        component = first[:, axis : axis + 1]
        if component.requires_grad:
            second = torch.autograd.grad(
                component,
                coordinates,
                grad_outputs=torch.ones_like(component),
                create_graph=True,
                allow_unused=True,
            )[0]
        else:
            second = None
        if second is None:
            terms.append(torch.zeros_like(y))
        else:
            terms.append(second[:, axis : axis + 1])
    return torch.stack(terms, dim=0).sum(dim=0)


# %% Uniform periodic finite differences


def periodic_gradient_fd(field: Tensor, domain: PeriodicDomain) -> Tensor:
    """Second-order centered gradient; components precede spatial axes."""

    domain.validate_field(field)
    derivatives = []
    for axis, dx in zip(domain.spatial_axes, domain.spacing):
        derivatives.append(
            (torch.roll(field, -1, axis) - torch.roll(field, 1, axis)) / (2 * dx)
        )
    component_axis = field.ndim - domain.dim
    return torch.stack(derivatives, dim=component_axis)


def periodic_laplacian_fd(field: Tensor, domain: PeriodicDomain) -> Tensor:
    """Second-order centered Laplacian on a periodic uniform grid."""

    domain.validate_field(field)
    result = torch.zeros_like(field)
    for axis, dx in zip(domain.spatial_axes, domain.spacing):
        result = result + (
            torch.roll(field, -1, axis) - 2 * field + torch.roll(field, 1, axis)
        ) / dx**2
    return result


# %% Spectral derivatives on periodic grids


def _first_derivative_wave_vectors(
    field: Tensor, domain: PeriodicDomain
) -> tuple[Tensor, ...]:
    """Wave vectors compatible with real fields on even-sized grids.

    A real grid's Nyquist mode is self-conjugate and therefore has no signed
    first derivative.  Setting that component to zero preserves Hermitian
    symmetry; complex fields retain the full signed Fourier convention.
    """

    wave_vectors = list(
        domain.wave_vectors(device=field.device, dtype=field.real.dtype)
    )
    if field.is_complex():
        return tuple(wave_vectors)
    for axis, n in enumerate(domain.shape):
        if n % 2 == 0:
            index = [slice(None)] * domain.dim
            index[axis] = n // 2
            wave_vectors[axis] = wave_vectors[axis].clone()
            wave_vectors[axis][tuple(index)] = 0
    return tuple(wave_vectors)


def spectral_gradient(field: Tensor, domain: PeriodicDomain) -> Tensor:
    """FFT gradient, exact for modes represented on the periodic grid."""

    domain.validate_field(field)
    transformed = torch.fft.fftn(field, dim=domain.spatial_axes)
    wave_vectors = _first_derivative_wave_vectors(field, domain)
    derivatives = [
        torch.fft.ifftn(1j * k * transformed, dim=domain.spatial_axes)
        for k in wave_vectors
    ]
    if not field.is_complex():
        derivatives = [value.real for value in derivatives]
    component_axis = field.ndim - domain.dim
    return torch.stack(derivatives, dim=component_axis)


def spectral_laplacian(field: Tensor, domain: PeriodicDomain) -> Tensor:
    """FFT Laplacian using the multiplier ``-|k|^2``."""

    domain.validate_field(field)
    transformed = torch.fft.fftn(field, dim=domain.spatial_axes)
    wave_vectors = domain.wave_vectors(device=field.device, dtype=field.real.dtype)
    k_squared = sum(k**2 for k in wave_vectors)
    result = torch.fft.ifftn(-k_squared * transformed, dim=domain.spatial_axes)
    return result if field.is_complex() else result.real


# %% Invariants and projections


def l2_mass(field: Tensor, domain: PeriodicDomain) -> Tensor:
    """Spatial integral of ``|field|^2``, retaining any leading batch axes."""

    domain.validate_field(field)
    return torch.sum(torch.abs(field) ** 2, dim=domain.spatial_axes) * domain.cell_volume


def _spatial_broadcast(values: Tensor, domain: PeriodicDomain) -> Tensor:
    return values.reshape(*values.shape, *((1,) * domain.dim))


def project_to_mass(
    prediction: Tensor,
    reference: Tensor,
    domain: PeriodicDomain,
    *,
    eps: float = 1e-14,
) -> Tensor:
    """Rescale each predicted field to the corresponding reference mass."""

    domain.validate_field(prediction)
    domain.validate_field(reference)
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference must have identical shapes")
    predicted_mass = l2_mass(prediction, domain)
    reference_mass = l2_mass(reference, domain)
    if torch.any((predicted_mass <= eps) & (reference_mass > eps)):
        raise ValueError("cannot project a zero field to positive mass")
    scale = torch.where(
        reference_mass <= eps,
        torch.zeros_like(reference_mass),
        torch.sqrt(reference_mass / torch.clamp(predicted_mass, min=eps)),
    )
    return prediction * _spatial_broadcast(scale, domain)


def positive_unit_mass(
    logits: Tensor,
    domain: PeriodicDomain,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """Map arbitrary real logits to a strictly positive unit-integral density."""

    domain.validate_field(logits)
    if logits.is_complex():
        raise ValueError("density logits must be real")
    density = F.softplus(logits) + eps
    integral = torch.sum(density, dim=domain.spatial_axes) * domain.cell_volume
    return density / _spatial_broadcast(integral, domain)


def _vector_component_axis(field: Tensor, domain: PeriodicDomain) -> int:
    component_axis = field.ndim - domain.dim - 1
    if component_axis < 0 or field.shape[component_axis] != domain.dim:
        raise ValueError(
            "vector field must have one component axis immediately before spatial axes"
        )
    if tuple(field.shape[-domain.dim :]) != domain.shape:
        raise ValueError(f"vector field must end in spatial shape {domain.shape}")
    return component_axis


def spectral_divergence(field: Tensor, domain: PeriodicDomain) -> Tensor:
    """Divergence of a grid vector field shaped ``(..., D, *domain.shape)``."""

    component_axis = _vector_component_axis(field, domain)
    transformed = torch.fft.fftn(field, dim=domain.spatial_axes)
    wave_vectors = _first_derivative_wave_vectors(field, domain)
    components = transformed.unbind(component_axis)
    divergence_hat = sum(1j * k * value for k, value in zip(wave_vectors, components))
    result = torch.fft.ifftn(divergence_hat, dim=domain.spatial_axes)
    return result if field.is_complex() else result.real


def helmholtz_project(field: Tensor, domain: PeriodicDomain) -> Tensor:
    """Orthogonally project an ND periodic vector field onto zero divergence."""

    component_axis = _vector_component_axis(field, domain)
    transformed = torch.fft.fftn(field, dim=domain.spatial_axes)
    components = transformed.unbind(component_axis)
    wave_vectors = _first_derivative_wave_vectors(field, domain)
    k_squared = sum(k**2 for k in wave_vectors)
    longitudinal = sum(k * value for k, value in zip(wave_vectors, components))
    coefficient = torch.where(
        k_squared > 0,
        longitudinal / torch.where(k_squared > 0, k_squared, torch.ones_like(k_squared)),
        torch.zeros_like(longitudinal),
    )
    projected_hat = torch.stack(
        [value - k * coefficient for k, value in zip(wave_vectors, components)],
        dim=component_axis,
    )
    result = torch.fft.ifftn(projected_hat, dim=domain.spatial_axes)
    return result if field.is_complex() else result.real


def canonical_hamiltonian_field(
    hamiltonian: Tensor, q: Tensor, p: Tensor
) -> tuple[Tensor, Tensor]:
    """Return ``J grad(H) = (dH/dp, -dH/dq)`` for canonical variables."""

    if hamiltonian.numel() != 1:
        raise ValueError("hamiltonian must be scalar")
    if q.shape != p.shape or not q.requires_grad or not p.requires_grad:
        raise ValueError("q and p must be aligned differentiable tensors")
    d_h_d_q, d_h_d_p = torch.autograd.grad(
        hamiltonian, (q, p), create_graph=True
    )
    return d_h_d_p, -d_h_d_q


# %% Parametric structure-preserving operator blocks


class ParametricPhaseMLP(nn.Module):
    """Small real network used for either phase rates or raw decay rates."""

    def __init__(
        self, feature_dim: int, parameter_dim: int, width: int = 16
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or parameter_dim <= 0 or width <= 0:
            raise ValueError("feature_dim, parameter_dim, and width must be positive")
        self.feature_dim = feature_dim
        self.parameter_dim = parameter_dim
        self.network = nn.Sequential(
            nn.Linear(feature_dim + parameter_dim, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, features: Tensor, parameters: Tensor) -> Tensor:
        if features.ndim < 2 or features.shape[-1] != self.feature_dim:
            raise ValueError("features must have shape (batch, ..., feature_dim)")
        if parameters.ndim != 2 or parameters.shape != (
            features.shape[0],
            self.parameter_dim,
        ):
            raise ValueError("parameters must have shape (batch, parameter_dim)")
        expanded_parameters = parameters.reshape(
            parameters.shape[0],
            *((1,) * (features.ndim - 2)),
            self.parameter_dim,
        ).expand(*features.shape[:-1], self.parameter_dim)
        return self.network(torch.cat((features, expanded_parameters), dim=-1)).squeeze(-1)


class _ParametricSpectralOperator(nn.Module):
    def __init__(self, domain: PeriodicDomain, rate_model: nn.Module) -> None:
        super().__init__()
        self.domain = domain
        self.rate_model = rate_model

    def _rates(self, field: Tensor, parameters: Tensor) -> Tensor:
        self.domain.validate_field(field)
        if field.ndim != self.domain.dim + 1:
            raise ValueError("operator fields must have one leading batch axis")
        if parameters.ndim != 2 or parameters.shape[0] != field.shape[0]:
            raise ValueError("parameters must have shape (batch, parameter_dim)")
        wave_vectors = self.domain.wave_vectors(
            device=field.device, dtype=field.real.dtype
        )
        k_squared = sum(k**2 for k in wave_vectors)
        features = k_squared.unsqueeze(0).expand(field.shape[0], *self.domain.shape)
        rates = self.rate_model(features.unsqueeze(-1), parameters)
        if rates.shape != field.shape or rates.is_complex():
            raise ValueError("rate model must return one real value per spatial point")
        return rates


class UnitarySpectralOperator(_ParametricSpectralOperator):
    """Parametric Fourier multiplier with unit modulus for every weight value."""

    def forward(self, field: Tensor, parameters: Tensor, dt: float) -> Tensor:
        if not field.is_complex():
            raise ValueError("unitary spectral fields must be complex")
        rates = self._rates(field, parameters)
        transformed = torch.fft.fftn(field, dim=self.domain.spatial_axes)
        multiplier = torch.exp(1j * float(dt) * rates)
        return torch.fft.ifftn(transformed * multiplier, dim=self.domain.spatial_axes)


class DissipativeSpectralOperator(_ParametricSpectralOperator):
    """Parametric Fourier multiplier whose magnitude can never exceed one."""

    def forward(self, field: Tensor, parameters: Tensor, dt: float) -> Tensor:
        if dt < 0:
            raise ValueError("dissipative evolution requires nonnegative dt")
        rates = F.softplus(self._rates(field, parameters))
        transformed = torch.fft.fftn(field, dim=self.domain.spatial_axes)
        result = torch.fft.ifftn(
            transformed * torch.exp(-float(dt) * rates),
            dim=self.domain.spatial_axes,
        )
        return result if field.is_complex() else result.real


class ProjectedOperator(nn.Module):
    """Apply an arbitrary core, then enforce a hard output constraint."""

    def __init__(
        self,
        core: nn.Module,
        projector: Callable[[Tensor, Tensor], Tensor],
    ) -> None:
        super().__init__()
        self.core = core
        self.projector = projector

    def forward(self, state: Tensor, *args, **kwargs) -> Tensor:
        raw = self.core(state, *args, **kwargs)
        return self.projector(raw, state)


# %% Schrödinger/NLS: plane waves, split stepping, and Hamiltonian energy


def plane_wave(
    domain: PeriodicDomain,
    wave_numbers: Sequence[int],
    *,
    amplitude: float = 1.0,
    time: float = 0.0,
    alpha: float = 0.5,
    beta: float = 0.0,
    potential_constant: float = 0.0,
) -> Tensor:
    """ND NLS plane wave with ``omega=alpha|k|^2-beta|A|^2+V0``."""

    if len(wave_numbers) != domain.dim:
        raise ValueError("wave_numbers must contain one integer per dimension")
    if amplitude <= 0:
        raise ValueError("amplitude must be positive")
    coordinates = domain.mesh()
    phase = sum(k * x for k, x in zip(wave_numbers, coordinates))
    omega = (
        alpha * sum(k**2 for k in wave_numbers)
        - beta * amplitude**2
        + potential_constant
    )
    return amplitude * torch.exp(1j * (phase - omega * time))


def _batch_parameter(
    value: Tensor | float,
    batch: int,
    domain: PeriodicDomain,
    reference: Tensor,
    name: str,
) -> Tensor:
    tensor = torch.as_tensor(value, device=reference.device, dtype=reference.real.dtype)
    if tensor.ndim == 0:
        tensor = tensor.expand(batch)
    if tensor.shape != (batch,):
        raise ValueError(f"{name} must be scalar or have shape (batch,)")
    return tensor.reshape(batch, *((1,) * domain.dim))


class SplitStepNLSOperator(nn.Module):
    """Symmetric unitary step for NLS with an arbitrary real potential field."""

    def __init__(self, domain: PeriodicDomain) -> None:
        super().__init__()
        self.domain = domain

    def _kinetic(self, field: Tensor, alpha: Tensor, dt: float) -> Tensor:
        wave_vectors = self.domain.wave_vectors(
            device=field.device, dtype=field.real.dtype
        )
        k_squared = sum(k**2 for k in wave_vectors)
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
        self.domain.validate_field(field)
        self.domain.validate_field(potential)
        if field.ndim != self.domain.dim + 1 or potential.shape != field.shape:
            raise ValueError("field and potential must have shape (batch, *domain.shape)")
        if not field.is_complex() or potential.is_complex():
            raise ValueError("field must be complex and potential must be real")
        alpha_grid = _batch_parameter(
            alpha, field.shape[0], self.domain, field, "alpha"
        )
        beta_grid = _batch_parameter(beta, field.shape[0], self.domain, field, "beta")
        midpoint = self._kinetic(field, alpha_grid, 0.5 * float(dt))
        local_phase = torch.exp(
            1j * (beta_grid * torch.abs(midpoint) ** 2 - potential) * float(dt)
        )
        return self._kinetic(midpoint * local_phase, alpha_grid, 0.5 * float(dt))


def nls_hamiltonian(
    field: Tensor,
    potential: Tensor,
    domain: PeriodicDomain,
    alpha: Tensor | float,
    beta: Tensor | float,
) -> Tensor:
    """Discrete NLS energy ``integral(alpha|grad psi|^2+V|psi|^2-beta|psi|^4/2)``."""

    domain.validate_field(field)
    domain.validate_field(potential)
    if field.ndim != domain.dim + 1 or potential.shape != field.shape:
        raise ValueError("field and potential must have shape (batch, *domain.shape)")
    
    alpha_grid = _batch_parameter(alpha, field.shape[0], domain, field, "alpha")
    beta_grid = _batch_parameter(beta, field.shape[0], domain, field, "beta")

    spatial_gradient = spectral_gradient(field, domain)

    kinetic_density = torch.sum(torch.abs(spatial_gradient) ** 2, dim=1)

    density = torch.abs(field) ** 2

    energy_density = (
        alpha_grid * kinetic_density
        + potential * density
        - 0.5 * beta_grid * density**2
    )
    return torch.sum(energy_density, dim=domain.spatial_axes) * domain.cell_volume


# %% Small, no-training demonstrations


@dataclass
class DemoResult:
    """Metrics plus two scalar fields used by the rank-aware summary figure."""

    name: str
    metrics: dict[str, float]
    initial: Tensor
    final: Tensor


def _relative_l2(actual: Tensor, expected: Tensor) -> float:
    denominator = torch.linalg.vector_norm(expected.reshape(-1))
    numerator = torch.linalg.vector_norm((actual - expected).reshape(-1))
    return float(numerator / torch.clamp(denominator, min=1e-15))


def _finite_difference_refinement_ratio() -> float:
    errors = []
    for n in (32, 64):
        domain = PeriodicDomain((n,), (2 * math.pi,))
        (x,) = domain.mesh()
        actual = periodic_gradient_fd(torch.sin(3 * x), domain)[0]
        exact = 3 * torch.cos(3 * x)
        errors.append(torch.sqrt(torch.mean((actual - exact) ** 2)))
    return float(errors[1] / errors[0])


def run_derivative_demo(domain: PeriodicDomain) -> DemoResult:
    """Compare autograd, finite differences, and FFT differentiation."""

    samples = torch.stack(
        [torch.linspace(0.15, 1.05, 9) + 0.07 * axis for axis in range(domain.dim)],
        dim=1,
    ).requires_grad_(True)
    scalar_samples = torch.sin(samples).sum(dim=1, keepdim=True)
    autograd_error = float(
        torch.max(
            torch.abs(laplacian(scalar_samples, samples) + scalar_samples)
        ).detach()
    )

    coordinates = domain.mesh()
    phase = sum(coordinates)
    field = torch.sin(phase)
    exact_gradient = torch.stack([torch.cos(phase)] * domain.dim, dim=0)
    fd_gradient = periodic_gradient_fd(field, domain)
    fd_error = float(torch.sqrt(torch.mean((fd_gradient - exact_gradient) ** 2)))
    spectral_result = spectral_laplacian(field, domain)
    spectral_error = float(torch.max(torch.abs(spectral_result + domain.dim * field)))
    refinement_ratio = _finite_difference_refinement_ratio()

    assert autograd_error < 1e-9
    assert spectral_error < 1e-9
    assert refinement_ratio < 0.3
    return DemoResult(
        "derivatives",
        {
            "autograd_laplacian_max_error": autograd_error,
            "fd_gradient_rms_error": fd_error,
            "spectral_laplacian_max_error": spectral_error,
            "fd_refinement_ratio": refinement_ratio,
        },
        field,
        spectral_result,
    )


def run_schrodinger_demo(
    domain: PeriodicDomain, *, steps: int, dt: float
) -> DemoResult:
    """Exercise unitary neural phases, plane waves, and split-step NLS."""

    alpha_value, beta_value = 0.7, 0.35
    alpha = torch.tensor([alpha_value])
    beta = torch.tensor([beta_value])
    coordinates = domain.mesh()
    potential_scalar = sum(0.15 * torch.cos(x) for x in coordinates) / domain.dim
    potential = potential_scalar.unsqueeze(0)

    first_mode = (1,) + (0,) * (domain.dim - 1)
    second_mode = (-1,) + (0,) * (domain.dim - 1)
    initial_scalar = plane_wave(domain, first_mode) + 0.35 * plane_wave(
        domain, second_mode
    )
    initial = initial_scalar.unsqueeze(0)
    split_step = SplitStepNLSOperator(domain)
    evolved = initial
    for _ in range(steps):
        evolved = split_step(evolved, potential, alpha, beta, dt)

    initial_mass = l2_mass(initial, domain)
    final_mass = l2_mass(evolved, domain)
    mass_drift = float(torch.max(torch.abs(final_mass / initial_mass - 1)))

    recovered = evolved
    for _ in range(steps):
        recovered = split_step(recovered, potential, alpha, beta, -dt)
    reversibility_error = _relative_l2(recovered, initial)

    phase_model = ParametricPhaseMLP(1, 2, width=8)
    unitary = UnitarySpectralOperator(domain, phase_model)
    neural_output = unitary(initial, torch.tensor([[alpha_value, beta_value]]), dt)
    neural_mass_drift = float(
        torch.max(
            torch.abs(l2_mass(neural_output, domain) / initial_mass - 1)
        ).detach()
    )

    plane_mode = (1,) * domain.dim
    plane_amplitude = 1.1
    potential_value = 0.2
    plane_initial = plane_wave(
        domain, plane_mode, amplitude=plane_amplitude
    ).unsqueeze(0)
    constant_potential = torch.full_like(plane_initial.real, potential_value)
    plane_actual = split_step(
        plane_initial, constant_potential, alpha, beta, dt
    )[0]
    plane_expected = plane_wave(
        domain,
        plane_mode,
        amplitude=plane_amplitude,
        time=dt,
        alpha=alpha_value,
        beta=beta_value,
        potential_constant=potential_value,
    )
    plane_error = float(torch.max(torch.abs(plane_actual - plane_expected)))

    q = initial.real.detach().clone().requires_grad_(True)
    p = initial.imag.detach().clone().requires_grad_(True)
    hamiltonian = nls_hamiltonian(
        q + 1j * p, potential, domain, alpha, beta
    ).sum()
    # Take the raw gradients independently so the check below actually exercises the
    # sign convention of ``canonical_hamiltonian_field``.  Expressing the same sum
    # purely in terms of ``q_dot``/``p_dot`` collapses to ``0`` algebraically and
    # would hold for any pair of tensors, testing nothing.
    grad_q, grad_p = torch.autograd.grad(
        hamiltonian, (q, p), create_graph=True, retain_graph=True
    )
    q_dot, p_dot = canonical_hamiltonian_field(hamiltonian, q, p)
    orthogonality = float(
        torch.abs(torch.sum(grad_q * q_dot + grad_p * p_dot)).detach()
    )
    energy_initial = nls_hamiltonian(initial, potential, domain, alpha, beta)
    energy_final = nls_hamiltonian(evolved, potential, domain, alpha, beta)
    energy_drift = float(
        torch.max(
            torch.abs(energy_final - energy_initial)
            / torch.clamp(torch.abs(energy_initial), min=1e-15)
        )
    )

    assert mass_drift < 1e-10
    assert neural_mass_drift < 1e-10
    assert reversibility_error < 1e-10
    assert plane_error < 1e-9
    assert orthogonality < 1e-10
    return DemoResult(
        "schrodinger",
        {
            "plane_wave_max_error": plane_error,
            "split_mass_relative_drift": mass_drift,
            "unitary_nn_mass_relative_drift": neural_mass_drift,
            "reversibility_relative_error": reversibility_error,
            "hamiltonian_orthogonality": orthogonality,
            "energy_relative_drift": energy_drift,
        },
        torch.abs(initial_scalar),
        torch.abs(evolved[0]),
    )


def heat_spectral_step(
    field: Tensor,
    domain: PeriodicDomain,
    diffusivity: float,
    dt: float,
) -> Tensor:
    """Exact-in-time Fourier step for ``u_t = diffusivity * Laplacian(u)``."""

    domain.validate_field(field)
    if diffusivity < 0 or dt < 0:
        raise ValueError("diffusivity and dt must be nonnegative")
    wave_vectors = domain.wave_vectors(device=field.device, dtype=field.real.dtype)
    k_squared = sum(k**2 for k in wave_vectors)
    transformed = torch.fft.fftn(field, dim=domain.spatial_axes)
    result = torch.fft.ifftn(
        transformed * torch.exp(-float(diffusivity) * k_squared * float(dt)),
        dim=domain.spatial_axes,
    )
    return result if field.is_complex() else result.real


def run_heat_demo(domain: PeriodicDomain, *, steps: int, dt: float) -> DemoResult:
    """Show exact mode decay and learned-by-construction contraction."""

    diffusivity = 0.3
    phase = sum(domain.mesh())
    initial_scalar = torch.cos(phase)
    evolved = initial_scalar
    for _ in range(steps):
        evolved = heat_spectral_step(evolved, domain, diffusivity, dt)
    expected = initial_scalar * math.exp(-diffusivity * domain.dim * steps * dt)
    analytic_error = float(torch.max(torch.abs(evolved - expected)))
    mass_ratio = float(l2_mass(evolved, domain) / l2_mass(initial_scalar, domain))

    model = ParametricPhaseMLP(1, 1, width=8)
    neural_operator = DissipativeSpectralOperator(domain, model)
    neural_input = initial_scalar.unsqueeze(0)
    neural_output = neural_operator(
        neural_input, torch.tensor([[diffusivity]]), dt
    )
    contraction_ratio = float(
        (
            l2_mass(neural_output, domain)[0]
            / l2_mass(neural_input, domain)[0]
        ).detach()
    )

    assert analytic_error < 1e-9
    assert mass_ratio <= 1 + 1e-12
    assert contraction_ratio <= 1 + 1e-12
    return DemoResult(
        "heat",
        {
            "analytic_mode_max_error": analytic_error,
            "mass_ratio": mass_ratio,
            "neural_contraction_ratio": contraction_ratio,
        },
        initial_scalar,
        evolved,
    )


def run_incompressible_demo(domain: PeriodicDomain) -> DemoResult | None:
    """Project an arbitrary periodic vector field and verify its divergence."""

    if domain.dim == 1:
        print("incompressible: skipped (nontrivial demo requires 2D or 3D)")
        return None
    vector = torch.randn(domain.dim, *domain.shape)
    projected = helmholtz_project(vector, domain)
    divergence_before = float(torch.max(torch.abs(spectral_divergence(vector, domain))))
    divergence_after = float(
        torch.max(torch.abs(spectral_divergence(projected, domain)))
    )

    coordinates = torch.randn(11, domain.dim, requires_grad=True)
    x = coordinates[:, 0:1]
    y = coordinates[:, 1:2]
    if domain.dim == 2:
        analytic = torch.cat((-y, x), dim=1)
    else:
        z = coordinates[:, 2:3]
        analytic = torch.cat((-y, x, 0 * z**2), dim=1)
    autograd_divergence = float(
        torch.max(torch.abs(divergence(analytic, coordinates))).detach()
    )

    assert divergence_after < 1e-9
    assert autograd_divergence < 1e-12
    return DemoResult(
        "incompressible",
        {
            "divergence_before": divergence_before,
            "divergence_after": divergence_after,
            "autograd_divergence": autograd_divergence,
        },
        torch.sqrt(torch.sum(vector**2, dim=0)),
        torch.sqrt(torch.sum(projected**2, dim=0)),
    )


def run_density_demo(domain: PeriodicDomain) -> DemoResult:
    """Turn arbitrary operator logits into a positive probability density."""

    coordinates = domain.mesh()
    logits = sum(torch.sin((axis + 1) * x) for axis, x in enumerate(coordinates))
    logits = logits + 0.15 * torch.randn(domain.shape)
    density = positive_unit_mass(logits, domain)
    minimum = float(torch.min(density))
    mass_error = float(torch.abs(torch.sum(density) * domain.cell_volume - 1))

    assert minimum > 0
    assert mass_error < 1e-10
    return DemoResult(
        "density",
        {"minimum": minimum, "mass_error": mass_error},
        logits,
        density,
    )


# %% Rank-aware plotting and CLI


DEMO_NAMES = (
    "derivatives",
    "schrodinger",
    "heat",
    "incompressible",
    "density",
)


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", choices=("all", *DEMO_NAMES), default="all")
    parser.add_argument("--dim", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--grid-size", type=_positive_int)
    parser.add_argument("--steps", type=_positive_int, default=8)
    parser.add_argument("--dt", type=_positive_float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def _central_view(field: Tensor, dim: int) -> Tensor:
    value = torch.abs(field) if field.is_complex() else field
    while value.ndim > dim:
        value = value[0]
    if dim == 3:
        value = value[:, :, value.shape[2] // 2]
    return value.detach().cpu()


def save_summary_figure(
    results: dict[str, DemoResult], domain: PeriodicDomain, output: Path
) -> None:
    import os
    import tempfile

    cache = Path(tempfile.gettempdir()) / "pinn-neural-operators-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("cannot plot an empty result set")
    figure, axes = plt.subplots(
        2,
        len(results),
        figsize=(4.1 * len(results), 6.0),
        squeeze=False,
    )
    x_axis = domain.mesh()[0].reshape(-1) if domain.dim == 1 else None
    for column, result in enumerate(results.values()):
        for row, (label, field) in enumerate(
            (("input", result.initial), ("structured output", result.final))
        ):
            axis = axes[row, column]
            view = _central_view(field, domain.dim)
            if domain.dim == 1:
                axis.plot(x_axis.cpu(), view.reshape(-1), color=f"C{column % 10}")
                axis.set_xlabel("x")
            else:
                image = axis.imshow(view, origin="lower", cmap="viridis")
                figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
                axis.set_xticks([])
                axis.set_yticks([])
            axis.set_title(f"{result.name}: {label}")
    figure.suptitle(f"Structure-preserving PDE toolkit ({domain.dim}D)")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _run_named_demo(
    name: str, domain: PeriodicDomain, steps: int, dt: float
) -> DemoResult | None:
    if name == "derivatives":
        return run_derivative_demo(domain)
    if name == "schrodinger":
        return run_schrodinger_demo(domain, steps=steps, dt=dt)
    if name == "heat":
        return run_heat_demo(domain, steps=steps, dt=dt)
    if name == "incompressible":
        return run_incompressible_demo(domain)
    if name == "density":
        return run_density_demo(domain)
    raise ValueError(f"unknown demo {name!r}")


def _run_cli(args: argparse.Namespace) -> dict[str, DemoResult]:
    torch.manual_seed(args.seed)
    defaults = {1: 128, 2: 32, 3: 12}
    grid_size = args.grid_size or defaults[args.dim]
    domain = PeriodicDomain(
        (grid_size,) * args.dim,
        (2 * math.pi,) * args.dim,
    )
    names = DEMO_NAMES if args.demo == "all" else (args.demo,)
    results: dict[str, DemoResult] = {}

    print(
        f"structure-preserving toolkit | dim={args.dim} grid={domain.shape} "
        f"steps={args.steps} dt={args.dt:g}"
    )
    for name in names:
        result = _run_named_demo(name, domain, args.steps, args.dt)
        if result is None:
            continue
        results[name] = result
        metrics = "  ".join(f"{key}={value:.3e}" for key, value in result.metrics.items())
        print(f"{name}: {metrics}")

    if not args.no_plot and results:
        output = args.output or Path(f"figures/05_structure_toolkit_{args.dim}d.png")
        save_summary_figure(results, domain, output)
        print(f"saved {output}")
    return results


def main(argv: Sequence[str] | None = None) -> dict[str, DemoResult]:
    args = parse_args(argv)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        return _run_cli(args)
    finally:
        torch.set_default_dtype(previous_dtype)


if __name__ == "__main__":
    main()
