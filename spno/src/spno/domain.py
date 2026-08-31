"""Periodic spectral geometry, invariants, and projections.

Ported from the validated tutorial toolkit ``pinn-neural-operators/05_3d_equations.py``
(``PeriodicDomain``, ``spectral_gradient``, ``spectral_laplacian``, ``l2_mass``,
``project_to_mass``).  Only the pieces the NLS study needs are carried over; the
incompressible-flow and density helpers stay behind in the tutorial.

Everything here is grid-resolution agnostic in the sense that it derives its wave
vectors from the domain, so the same code serves N=64 and N=128 unchanged.

``MeasureSpace -> HilbertSpace -> SpectralDomain`` is the generalisation hierarchy
from the Phase 1 domain-abstraction review (Candidate B), built per ADR 0002 on
explicit request -- see that ADR for why this supersedes ADR 0001's "not yet".
``PeriodicDomain`` is the sole concrete implementer; it keeps its existing name,
module path and full public API unchanged (constraint 4), and gains the hierarchy
as additional base classes plus additional methods, not replacements.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
import math

import torch

Tensor = torch.Tensor


def _as_grid_size(n: object) -> int:
    """Coerce a grid-size entry to ``int``, integrally.

    ``bool`` is rejected explicitly even though it subclasses ``int`` (``True`` would
    otherwise silently yield a one-point grid).  Anything else with a lossless
    ``__index__`` -- ``numpy.int64`` included -- is accepted, so array libraries that
    hand back their own integer scalar types do not need an explicit ``int(...)`` cast
    at the call site.
    """

    if isinstance(n, bool):
        raise ValueError(f"every grid size must be a positive integer; got bool {n!r}")
    if isinstance(n, int):
        return n
    index = getattr(n, "__index__", None)
    if index is None:
        raise ValueError(f"every grid size must be a positive integer; got {n!r}")
    return index()


class MeasureSpace(abc.ABC):
    """A discrete set of points with a quadrature rule. Tensor-free by contract.

    ``shape`` is declared as a plain annotation, not an ``@property``: a dataclass
    field of the same name in a concrete subclass has no class-level descriptor to
    override, so making this an ``@abc.abstractmethod`` property would make every
    such subclass permanently non-instantiable (the field shadows nothing, and the
    base class's abstract property survives the MRO lookup). See ADR 0002.

    Nothing here is cached: every space in this project derives its geometry per call
    with explicit ``device``/``dtype`` keywords (constraint 2 of the Phase 1 review),
    so the same space serves float32 training and float64 invariant evaluation
    without ever building a stale buffer.
    """

    shape: tuple[int, ...]

    @property
    def dim(self) -> int:
        return len(self.shape)

    @property
    def spatial_axes(self) -> tuple[int, ...]:
        return tuple(range(-self.dim, 0))

    @abc.abstractmethod
    def validate_field(self, field: Tensor) -> None:
        """Raise ``ValueError`` unless ``field`` ends in this space's spatial shape."""

    @abc.abstractmethod
    def quadrature_weights(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        """Per-point quadrature weight(s) for :meth:`integrate`, derived per call."""

    def integrate(self, density: Tensor) -> Tensor:
        """Spatial integral of ``density`` against this space's quadrature rule.

        Retains any leading batch axes, exactly as :func:`l2_mass` does for the
        special case ``density = |field|**2``.
        """

        self.validate_field(density)
        weights = self.quadrature_weights(
            device=density.device, dtype=density.real.dtype
        )
        return torch.sum(density * weights, dim=self.spatial_axes)


class HilbertSpace(MeasureSpace):
    """A measure space with the L^2 inner product ``SPNO-Mathematics.tex`` writes."""

    def inner(self, u: Tensor, v: Tensor) -> Tensor:
        return self.integrate(u.conj() * v)

    def norm(self, u: Tensor) -> Tensor:
        return torch.sqrt(self.inner(u, u).real)

    # Deliberately NOT `project`.  project_to_mass (module-level, below) is a
    # nonlinear radial rescale onto a fixed-mass sphere, not a linear projection --
    # renaming it into this vocabulary would launder that away (constraint 6 / ADR
    # 0001's dissent, preserved by ADR 0002).


class SpectralDomain(HilbertSpace):
    """A Hilbert space with a spectral basis: analysis/synthesis plus the elementwise
    multipliers that make differentiation and the Laplacian algebraic in that basis.
    """

    @abc.abstractmethod
    def analyze(self, field: Tensor) -> Tensor:
        """Field -> spectral coefficients."""

    @abc.abstractmethod
    def synthesize(self, coefficients: Tensor) -> Tensor:
        """Spectral coefficients -> field.

        Leaves realness casting to the caller, exactly as the module-level
        :func:`spectral_gradient` and :func:`spectral_laplacian` always have --
        ``torch.fft.ifftn`` returns a complex tensor regardless of input dtype.
        """

    @abc.abstractmethod
    def laplacian_multiplier(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        """The elementwise multiplier ``m`` such that ``synthesize(m * analyze(u))``
        is the Laplacian of ``u``."""

    @abc.abstractmethod
    def derivative_multiplier(
        self,
        axis: int,
        *,
        real_field: bool,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        """The elementwise multiplier for the first derivative along ``axis``.

        Trap A lives here (its public, named, tested home per ADR 0002): a real
        field's Nyquist mode on an even grid is self-conjugate and has no signed
        first derivative, so ``real_field=True`` zeroes it; ``real_field=False``
        keeps the full signed Fourier convention.
        """


@dataclass(frozen=True)
class PeriodicDomain(SpectralDomain):
    """Uniform endpoint-free rectangular periodic grid in any dimension."""

    shape: tuple[int, ...]
    lengths: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "shape", tuple(_as_grid_size(n) for n in self.shape)
        )
        object.__setattr__(self, "lengths", tuple(float(v) for v in self.lengths))
        if not self.shape:
            raise ValueError("shape must contain at least one spatial dimension")
        if len(self.shape) != len(self.lengths):
            raise ValueError("shape and lengths must have the same dimension")
        if any(not isinstance(n, int) or n <= 0 for n in self.shape):
            raise ValueError("every grid size must be a positive integer")
        if any(not math.isfinite(length) or length <= 0 for length in self.lengths):
            raise ValueError("every domain length must be positive and finite")

    @classmethod
    def periodic_1d(cls, n: int) -> "PeriodicDomain":
        """The thesis default: ``n`` points on ``[0, 2*pi)``."""

        return cls((n,), (2 * math.pi,))

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
    def volume(self) -> float:
        """Total measure of the domain: the ``|Omega|`` that normalises a mass density."""

        return math.prod(self.lengths)

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

    def wave_number_squared(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        """``|k|^2`` on the grid: the only spatial feature the kinetic step needs."""

        return sum(k**2 for k in self.wave_vectors(device=device, dtype=dtype))

    def validate_field(self, field: Tensor) -> None:
        if field.ndim < self.dim or tuple(field.shape[-self.dim :]) != self.shape:
            raise ValueError(
                f"field must end in spatial shape {self.shape}; got {tuple(field.shape)}"
            )

    def validate_batched_field(self, field: Tensor) -> None:
        """Enforce the ``(batch, *shape)`` layout every operator in this project uses."""

        self.validate_field(field)
        if field.ndim != self.dim + 1:
            raise ValueError(
                f"field must have exactly one leading batch axis; got {tuple(field.shape)}"
            )

    # -- SpectralDomain / HilbertSpace / MeasureSpace -----------------------------

    def quadrature_weights(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        """Uniform grid: every point carries the same weight, the cell volume."""

        return torch.as_tensor(self.cell_volume, device=device, dtype=dtype)

    def analyze(self, field: Tensor) -> Tensor:
        return torch.fft.fftn(field, dim=self.spatial_axes)

    def synthesize(self, coefficients: Tensor) -> Tensor:
        return torch.fft.ifftn(coefficients, dim=self.spatial_axes)

    def laplacian_multiplier(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        return -self.wave_number_squared(device=device, dtype=dtype)

    def derivative_multiplier(
        self,
        axis: int,
        *,
        real_field: bool,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> Tensor:
        k = self.wave_vectors(device=device, dtype=dtype)[axis]
        n = self.shape[axis]
        if real_field and n % 2 == 0:
            k = k.clone()
            index = [slice(None)] * self.dim
            index[axis] = n // 2
            k[tuple(index)] = 0
        return 1j * k


def spectral_gradient(field: Tensor, domain: PeriodicDomain) -> Tensor:
    """FFT gradient, exact for modes represented on the periodic grid."""

    domain.validate_field(field)
    transformed = domain.analyze(field)
    real_field = not field.is_complex()
    derivatives = [
        domain.synthesize(
            domain.derivative_multiplier(
                axis,
                real_field=real_field,
                device=field.device,
                dtype=field.real.dtype,
            )
            * transformed
        )
        for axis in range(domain.dim)
    ]
    if real_field:
        derivatives = [value.real for value in derivatives]
    component_axis = field.ndim - domain.dim
    return torch.stack(derivatives, dim=component_axis)


def spectral_laplacian(field: Tensor, domain: PeriodicDomain) -> Tensor:
    """FFT Laplacian using the multiplier ``-|k|^2``."""

    domain.validate_field(field)
    transformed = domain.analyze(field)
    multiplier = domain.laplacian_multiplier(device=field.device, dtype=field.real.dtype)
    result = domain.synthesize(multiplier * transformed)
    return result if field.is_complex() else result.real


def l2_mass(field: Tensor, domain: PeriodicDomain) -> Tensor:
    """Spatial integral of ``|field|^2``, retaining any leading batch axes."""

    domain.validate_field(field)
    return (
        torch.sum(torch.abs(field) ** 2, dim=domain.spatial_axes) * domain.cell_volume
    )


def spatial_broadcast(values: Tensor, domain: PeriodicDomain) -> Tensor:
    """Append singleton spatial axes so per-sample scalars broadcast over the grid.

    ``values`` must carry **no** spatial axes -- it is a per-sample (or per-sample,
    per-frame) scalar.  Passing a grid-shaped tensor would silently produce a
    rank-inflated result that broadcasts to ``(batch, *shape, *shape)``.
    """

    if (
        math.prod(domain.shape) > 1
        and values.ndim >= domain.dim
        and tuple(values.shape[-domain.dim :]) == domain.shape
    ):
        raise ValueError(
            f"values must not carry spatial axes; got {tuple(values.shape)}, which "
            f"already ends in the spatial shape {domain.shape}"
        )
    return values.reshape(*values.shape, *((1,) * domain.dim))


def project_to_mass(
    prediction: Tensor,
    reference: Tensor,
    domain: PeriodicDomain,
    *,
    eps: float = 1e-14,
) -> Tensor:
    """Rescale each predicted field to the corresponding reference mass.

    This is Model B's entire physics content: it enforces one scalar invariant and
    says nothing about phase, energy, or reversibility.  Not a ``HilbertSpace``
    method: see the note on :class:`HilbertSpace`.
    """

    domain.validate_field(prediction)
    domain.validate_field(reference)
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference must have identical shapes")
    predicted_mass = l2_mass(prediction, domain)
    reference_mass = l2_mass(reference, domain)
    if torch.any((predicted_mass <= eps) & (reference_mass > eps)):
        raise ValueError("cannot project a zero field to positive mass")
    safe_reference = torch.where(
        reference_mass <= eps, torch.ones_like(reference_mass), reference_mass
    )
    scale = torch.where(
        reference_mass <= eps,
        torch.zeros_like(reference_mass),
        torch.sqrt(safe_reference / torch.clamp(predicted_mass, min=eps)),
    )
    return prediction * spatial_broadcast(scale, domain)


# Historical home: batch_parameter is precision policy, not geometry -- it now
# lives in .precision, re-exported here so no importer's path has to change.
from .precision import batch_parameter as batch_parameter  # noqa: E402
