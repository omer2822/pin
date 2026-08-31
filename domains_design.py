"""A design spike: the abstract-space ladder, with three concrete domains.

NOT PRODUCTION CODE.  This file lives outside ``spno/`` on purpose: the Phase 1 gate
forbids touching ``src/``, and nothing here is approved.  It exists to answer one
question honestly -- *does the abstraction hold up when you build it from more than one
example?* -- because the review's case against building it was precisely that a shape
inferred from a single inhabitant is a guess.

Run it: ``python spikes/abstract_domain.py``.  It self-checks.

────────────────────────────────────────────────────────────────────────────────
THE LADDER, and what each rung actually buys

    MeasureSpace          (X, Sigma, mu)         -> integrate.  This alone is enough
                                                    for the mass invariant.  No metric,
                                                    no basis, no linearity.
    HilbertSpace          + inner product        -> inner, norm, Parseval, adjoints.
                                                    Guarantees *some* orthonormal basis
                                                    exists; never says which.
    SpectralOperatorDomain + a distinguished     -> a computable eigenbasis, hence
                            self-adjoint A         functional calculus f(A).  This is
                                                    the rung a "spectral method" lives
                                                    on, and it covers periodic,
                                                    Dirichlet, Neumann, Chebyshev,
                                                    the sphere and a graph Laplacian
                                                    with one API.

Two structural decisions worth defending, both learned from the second and third
inhabitant rather than the first:

1.  ``integrate`` is abstract as an OPERATION, not as a ``quadrature_weights()``
    accessor.  A uniform grid implements it as ``sum * cell_volume`` and never
    materialises a weight tensor; a Chebyshev grid materialises per-point weights on
    demand.  An accessor returning a tensor would put a tensor-shaped hole in an object
    whose entire value is being tensor-free -- the exact shape of the caching mistake
    the device/dtype discipline forbids.

2.  There is NO ``gradient`` on any ABC.  This is a mathematical obstruction, not an
    omission: the sine basis is not closed under d/dx (the derivative of a Dirichlet
    eigenfunction is a Neumann eigenfunction), so a first derivative cannot be a
    spectral multiplier in that basis.  ``spectral_gradient`` is therefore
    periodic-only, and hoisting it would have been a bug that only a second inhabitant
    reveals.  The Laplacian, being even in k, generalises fine.

3.  ``rescale_to_mass`` is a FREE FUNCTION and keeps a verb that says it rescales.  It
    is the *metric* projection onto a fixed-mass sphere -- nonlinear.  Naming it
    ``HilbertSpace.project`` would launder a nonlinear map into linear-operator
    vocabulary.

The class named ``PeriodicDomain`` below is the spike counterpart of
``spno.domain.PeriodicDomain``.  The production class is untouched.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass

import torch

Tensor = torch.Tensor


# ══════════════════════════════════════════════════════════════════════════════
# Rung 1 — a set with a measure.  Enough to integrate; enough for the invariant.
# ══════════════════════════════════════════════════════════════════════════════


class MeasureSpace(abc.ABC):
    """A finite point set carrying a quadrature rule.

    Tensor-free by contract: every method that produces tensors takes keyword-only
    ``device``/``dtype`` and derives them per call.  Nothing is cached, nothing is
    stored, and no subclass may hold a tensor attribute -- that is what lets one
    instance serve MPS-float32 training and CPU-float64 invariant evaluation.
    """

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Trailing spatial shape every field on this space must end in."""

    @property
    def dim(self) -> int:
        return len(self.shape)

    @property
    def spatial_axes(self) -> tuple[int, ...]:
        return tuple(range(-self.dim, 0))

    @abc.abstractmethod
    def integrate(self, density: Tensor) -> Tensor:
        """Integrate a scalar density over the space, keeping leading axes.

        Abstract as an *operation*, deliberately -- not as a weights accessor.
        """

    # ── shared machinery: the real code reuse the ladder buys ──────────────────

    def validate_field(self, field: Tensor) -> None:
        """Permissive: any number of leading axes, so trajectories (B, T, *shape) pass."""

        if field.ndim < self.dim or tuple(field.shape[-self.dim :]) != self.shape:
            raise ValueError(
                f"field must end in spatial shape {self.shape}; got {tuple(field.shape)}"
            )

    def validate_batched_field(self, field: Tensor) -> None:
        """Strict: exactly one leading batch axis.  Belongs at the operator boundary."""

        self.validate_field(field)
        if field.ndim != self.dim + 1:
            raise ValueError(
                f"field must have exactly one leading batch axis; got {tuple(field.shape)}"
            )

    def broadcast_scalars(self, values: Tensor) -> Tensor:
        """Append singleton spatial axes so per-sample scalars broadcast over the space.

        ``values`` must carry NO spatial axes.  Guarded, unlike the production version.
        """

        if (
            math.prod(self.shape) > 1
            and values.ndim >= self.dim
            and tuple(values.shape[-self.dim :]) == self.shape
        ):
            raise ValueError(
                f"values must not carry spatial axes; got {tuple(values.shape)}, which "
                f"already ends in the spatial shape {self.shape}"
            )
        return values.reshape(*values.shape, *((1,) * self.dim))


# ══════════════════════════════════════════════════════════════════════════════
# Rung 2 — an inner product.  Norms, orthogonality, Parseval.  Still no basis.
# ══════════════════════════════════════════════════════════════════════════════


class HilbertSpace(MeasureSpace):
    """Adds the inner product induced by the measure.  No basis is implied."""

    def inner(self, u: Tensor, v: Tensor) -> Tensor:
        self.validate_field(u)
        self.validate_field(v)
        return self.integrate(u.conj() * v)

    def norm(self, u: Tensor) -> Tensor:
        return torch.sqrt(self.inner(u, u).real)

    def mass(self, u: Tensor) -> Tensor:
        """The squared L2 norm.  For NLS this is the conserved mass."""

        self.validate_field(u)
        return self.integrate(torch.abs(u) ** 2)

    # Deliberately absent: `project`.  See the module docstring, decision 3.


# ══════════════════════════════════════════════════════════════════════════════
# Rung 3 — a distinguished self-adjoint operator, hence functional calculus.
# ══════════════════════════════════════════════════════════════════════════════


class SpectralOperatorDomain(HilbertSpace):
    """A Hilbert space plus one self-adjoint ``A`` whose eigenbasis is computable.

    Everything the study does to a field spatially is ``f(A)`` for some ``f``:

        Laplacian            f(lam) = -lam
        kinetic propagator   f(lam) = exp(-i * alpha * lam * dt)
        Model C's KineticPhase  f = a learned scalar function        <-- the interesting one

    That last line is the abstraction that actually has two inhabitants today
    (C1's pointwise phase and C2's FNO phase).  It is a statement about the
    *operator*, not the domain.
    """

    @abc.abstractmethod
    def eigenvalues(self, *, device=None, dtype: torch.dtype = torch.float64) -> Tensor:
        """Spectrum of ``A = -Laplacian``, laid out to match ``analyze``'s coefficients."""

    @abc.abstractmethod
    def analyze(self, field: Tensor) -> Tensor:
        """Field -> coefficients in the eigenbasis of ``A``."""

    @abc.abstractmethod
    def synthesize(self, coefficients: Tensor) -> Tensor:
        """Coefficients -> field.  Must invert ``analyze`` exactly."""

    def apply_function_of_operator(self, field: Tensor, fn) -> Tensor:
        """``f(A) field``, evaluated by functional calculus.  The whole point of rung 3."""

        self.validate_field(field)
        lam = self.eigenvalues(device=field.device, dtype=field.real.dtype)
        result = self.synthesize(fn(lam) * self.analyze(field))
        return result if field.is_complex() else result.real

    def laplacian(self, field: Tensor) -> Tensor:
        return self.apply_function_of_operator(field, lambda lam: -lam)

    def kinetic_propagator(self, field: Tensor, alpha: float, dt: float) -> Tensor:
        """``exp(-i alpha A dt)``, the linear half of a split step."""

        return self.apply_function_of_operator(
            field, lambda lam: torch.exp(-1j * alpha * lam * dt)
        )


# ══════════════════════════════════════════════════════════════════════════════
# Child 1 — the torus.  A finite abelian group with Haar measure.
# ══════════════════════════════════════════════════════════════════════════════


def self_conjugate_characters(n: int) -> tuple[int, ...]:
    """The 2-torsion of the dual group Z_n^ ~ Z_n: characters fixed by k -> -k.

    Always contains 0; contains n/2 exactly when n is even.  This single fact is
    both Nyquist traps: the mode at n/2 is its own conjugate, so it has no signed
    first derivative (real fields), and no unambiguous lift to a finer group.
    """

    return (0, n // 2) if n % 2 == 0 else (0,)


@dataclass(frozen=True)
class PeriodicDomain(SpectralOperatorDomain):
    """Uniform endpoint-free rectangular periodic grid: (Z_N)^d with Haar measure.

    ``fftn`` is the Fourier transform on that group; the wave vectors are its
    characters; O(N log N) is the group being a product of cyclic factors; and
    "the Laplacian is diagonal" is "translation-invariant operators are diagonal
    in the character basis".
    """

    grid: tuple[int, ...]
    lengths: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "grid", tuple(int(n) for n in self.grid))
        object.__setattr__(self, "lengths", tuple(float(v) for v in self.lengths))
        if not self.grid:
            raise ValueError("grid must contain at least one spatial dimension")
        if len(self.grid) != len(self.lengths):
            raise ValueError("grid and lengths must have the same dimension")
        if any(isinstance(n, bool) or n <= 0 for n in self.grid):
            raise ValueError("every grid size must be a positive integer")
        if any(not math.isfinite(v) or v <= 0 for v in self.lengths):
            raise ValueError("every domain length must be positive and finite")

    @classmethod
    def periodic_1d(cls, n: int) -> "PeriodicDomain":
        return cls((n,), (2 * math.pi,))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.grid

    @property
    def spacing(self) -> tuple[float, ...]:
        return tuple(L / n for n, L in zip(self.grid, self.lengths))

    @property
    def cell_volume(self) -> float:
        return math.prod(self.spacing)

    @property
    def volume(self) -> float:
        return math.prod(self.lengths)

    def integrate(self, density: Tensor) -> Tensor:
        self.validate_field(density)
        return torch.sum(density, dim=self.spatial_axes) * self.cell_volume

    def mesh(self, *, device=None, dtype: torch.dtype = torch.float64):
        axes = [
            torch.arange(n, device=device, dtype=dtype) * dx
            for n, dx in zip(self.grid, self.spacing)
        ]
        return tuple(torch.meshgrid(*axes, indexing="ij"))

    def wave_vectors(self, *, device=None, dtype: torch.dtype = torch.float64):
        axes = [
            2 * math.pi * torch.fft.fftfreq(n, d=dx, device=device, dtype=dtype)
            for n, dx in zip(self.grid, self.spacing)
        ]
        return tuple(torch.meshgrid(*axes, indexing="ij"))

    def eigenvalues(self, *, device=None, dtype: torch.dtype = torch.float64) -> Tensor:
        """|k|^2 -- the spectrum of -Laplacian on the torus."""

        return sum(k**2 for k in self.wave_vectors(device=device, dtype=dtype))

    def analyze(self, field: Tensor) -> Tensor:
        return torch.fft.fftn(field, dim=self.spatial_axes)

    def synthesize(self, coefficients: Tensor) -> Tensor:
        return torch.fft.ifftn(coefficients, dim=self.spatial_axes)

    # ── periodic-only: the first derivative IS a multiplier here, and only here ──

    def derivative_multipliers(self, field: Tensor):
        """``i k`` per axis, with the self-conjugate character zeroed for real fields.

        Complex fields keep the full signed convention.  See
        :func:`self_conjugate_characters`.
        """

        ks = list(self.wave_vectors(device=field.device, dtype=field.real.dtype))
        if field.is_complex():
            return tuple(ks)
        for axis, n in enumerate(self.grid):
            for k_index in self_conjugate_characters(n):
                if k_index == 0:
                    continue  # the trivial character already has zero derivative
                index = [slice(None)] * self.dim
                index[axis] = k_index
                ks[axis] = ks[axis].clone()
                ks[axis][tuple(index)] = 0
        return tuple(ks)

    def gradient(self, field: Tensor) -> Tensor:
        self.validate_field(field)
        hat = self.analyze(field)
        parts = [self.synthesize(1j * k * hat) for k in self.derivative_multipliers(field)]
        if not field.is_complex():
            parts = [p.real for p in parts]
        return torch.stack(parts, dim=field.ndim - self.dim)


# ══════════════════════════════════════════════════════════════════════════════
# Child 2 — a Dirichlet box.  Non-periodic BCs; the plausible successor.
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DirichletBox(SpectralOperatorDomain):
    """psi = 0 at both walls: the sine basis, via DST-I.  A trapped condensate.

    Fields live on the ``interior`` points x_j = j*L/(interior+1), j = 1..interior.
    Eigenfunctions sin(k pi x / L) with eigenvalues (k pi / L)^2, k = 1..interior.

    Note what does NOT come along: there is no ``gradient`` here, because
    d/dx sin(k pi x / L) is a *cosine*, i.e. lives in the Neumann basis.  The
    first derivative is not a multiplier in this basis.  That obstruction is
    invisible with only the torus in hand.
    """

    interior: int
    length: float = math.pi

    def __post_init__(self) -> None:
        object.__setattr__(self, "interior", int(self.interior))
        object.__setattr__(self, "length", float(self.length))
        if self.interior < 1:
            raise ValueError("interior must be a positive integer")
        if not math.isfinite(self.length) or self.length <= 0:
            raise ValueError("length must be positive and finite")

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.interior,)

    @property
    def _intervals(self) -> int:
        return self.interior + 1

    @property
    def spacing(self) -> float:
        return self.length / self._intervals

    def integrate(self, density: Tensor) -> Tensor:
        """Trapezoid, exact here: the boundary terms vanish by the BC."""

        self.validate_field(density)
        return torch.sum(density, dim=self.spatial_axes) * self.spacing

    def eigenvalues(self, *, device=None, dtype: torch.dtype = torch.float64) -> Tensor:
        k = torch.arange(1, self.interior + 1, device=device, dtype=dtype)
        return (k * math.pi / self.length) ** 2

    def analyze(self, field: Tensor) -> Tensor:
        """DST-I by odd extension: a_k = sum_j f_j sin(pi j k / N)."""

        n_int = self.interior
        zero = torch.zeros_like(field[..., :1])
        odd = torch.cat([zero, field, zero, -field.flip(-1)], dim=-1)  # length 2N
        hat = torch.fft.fft(odd, dim=-1)
        coefficients = 1j * hat[..., 1 : n_int + 1] / 2
        return coefficients if field.is_complex() else coefficients.real

    def synthesize(self, coefficients: Tensor) -> Tensor:
        """Inverse DST-I: f_j = (2/N) sum_k a_k sin(pi j k / N)."""

        n_int = self.interior
        zero = torch.zeros_like(coefficients[..., :1])
        odd = torch.cat([zero, coefficients, zero, -coefficients.flip(-1)], dim=-1)
        field = (-2j * torch.fft.ifft(odd, dim=-1))[..., 1 : n_int + 1]
        return field if coefficients.is_complex() else field.real


# ══════════════════════════════════════════════════════════════════════════════
# Child 3 — Chebyshev–Lobatto.  Stops at rung 2, on purpose.
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ChebyshevLobatto(HilbertSpace):
    """Non-uniform grid with per-point quadrature weights.  A HilbertSpace only.

    This child exists to make two points:

    1.  It is why ``integrate`` is an operation rather than a ``quadrature_weights``
        accessor.  Here the weights genuinely vary per point, so the uniform grid's
        scalar ``cell_volume`` was a Fourier artefact all along.

    2.  It inhabits rung 2 and *declines* rung 3.  The ladder is a precision
        mechanism, not decoration: a domain claims only the structure it can honestly
        implement.  A ``SpectralOperatorDomain`` here would need the full Chebyshev
        differentiation matrix and a boundary-condition story; absent that, claiming
        the rung would be a lie the type system would happily accept.
    """

    points: int
    interval: tuple[float, float] = (-1.0, 1.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", int(self.points))
        object.__setattr__(self, "interval", (float(self.interval[0]), float(self.interval[1])))
        if self.points < 2:
            raise ValueError("points must be at least 2")
        if not self.interval[1] > self.interval[0]:
            raise ValueError("interval must be non-degenerate and increasing")

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.points,)

    def nodes(self, *, device=None, dtype: torch.dtype = torch.float64) -> Tensor:
        """Gauss-Lobatto points, mapped to the interval and sorted increasingly."""

        j = torch.arange(self.points, device=device, dtype=dtype)
        reference = -torch.cos(j * math.pi / (self.points - 1))
        a, b = self.interval
        return a + (b - a) * (reference + 1) / 2

    def quadrature_weights(
        self, *, device=None, dtype: torch.dtype = torch.float64
    ) -> Tensor:
        """Interpolatory weights, by an exact Vandermonde solve.

        Obvious correctness over speed: a production implementation would use the
        FFT-based Clenshaw-Curtis (Waldvogel) recurrence.  Derived per call, never
        stored -- the constraint that forbids caching applies to non-uniform weights
        exactly as it does to wave vectors.
        """

        x = self.nodes(device=device, dtype=torch.float64)
        n = self.points
        powers = torch.arange(n, device=device, dtype=torch.float64)
        vandermonde = x.unsqueeze(0) ** powers.unsqueeze(1)  # V[i, j] = x_j ** i
        a, b = self.interval
        moments = (b ** (powers + 1) - a ** (powers + 1)) / (powers + 1)
        weights = torch.linalg.solve(vandermonde, moments)
        return weights.to(dtype)

    def integrate(self, density: Tensor) -> Tensor:
        self.validate_field(density)
        w = self.quadrature_weights(device=density.device, dtype=density.real.dtype)
        return torch.sum(density * w, dim=self.spatial_axes)


# ══════════════════════════════════════════════════════════════════════════════
# Free functions — nonlinear maps stay outside the linear vocabulary.
# ══════════════════════════════════════════════════════════════════════════════


def rescale_to_mass(
    prediction: Tensor, reference: Tensor, space: HilbertSpace, *, eps: float = 1e-14
) -> Tensor:
    """Radially rescale each field to its reference mass.

    The *metric* projection onto the fixed-mass sphere: the closest mass-correct
    field, and nonlinear.  A free function with a verb that says so.

    Needs rung 2 only -- it works unchanged on all three children, which is the one
    piece of genuine polymorphism the ladder delivers today.

    The double ``where`` is deliberate: ``torch.where`` evaluates both branches in the
    forward pass, so ``sqrt(0)`` in the unselected branch would contribute an infinite
    local derivative and the mask would form ``0 * inf = NaN``.
    """

    space.validate_field(prediction)
    space.validate_field(reference)
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference must have identical shapes")

    predicted = space.mass(prediction)
    target = space.mass(reference)
    if torch.any((predicted <= eps) & (target > eps)):
        raise ValueError("cannot rescale a zero field to positive mass")

    safe_target = torch.where(target <= eps, torch.ones_like(target), target)
    scale = torch.where(
        target <= eps,
        torch.zeros_like(target),
        torch.sqrt(safe_target / torch.clamp(predicted, min=eps)),
    )
    return prediction * space.broadcast_scalars(scale)


# ══════════════════════════════════════════════════════════════════════════════
# Self-check.  This is the part that decides whether the abstraction is real.
# ══════════════════════════════════════════════════════════════════════════════


def _check(label: str, condition: bool) -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}")
    if not condition:
        raise AssertionError(label)


def main() -> None:
    torch.manual_seed(0)

    print("\nchild 1 — PeriodicDomain  (Z_N)^d, Haar measure")
    torus = PeriodicDomain.periodic_1d(64)
    x = torus.mesh()[0]
    wave = torch.exp(3j * x).unsqueeze(0)

    _check("analyze/synthesize round-trip", torch.allclose(
        torus.synthesize(torus.analyze(wave)), wave, atol=1e-12))
    _check("Parseval: mass == coeff energy / N", torch.allclose(
        torus.mass(wave),
        (torch.abs(torus.analyze(wave)) ** 2).sum(-1) * torus.cell_volume / 64,
        atol=1e-10))
    _check("laplacian of exp(3ix) == -9 exp(3ix)  [functional calculus]", torch.allclose(
        torus.laplacian(wave), -9.0 * wave, atol=1e-9))
    _check("kinetic propagator is norm-preserving", torch.allclose(
        torus.mass(torus.kinetic_propagator(wave, 0.9, 0.01)), torus.mass(wave), atol=1e-12))
    _check("volume == 2 pi", abs(torus.volume - 2 * math.pi) < 1e-12)
    _check("frozen and hashable", isinstance(hash(torus), int))
    _check("tensor-free instance", not any(
        isinstance(v, torch.Tensor) for v in vars(torus).values()))

    print("\n  the 2-torsion fact — both Nyquist traps, one line of algebra")
    _check("self-conjugate characters of Z_64 == (0, 32)",
           self_conjugate_characters(64) == (0, 32))
    _check("self-conjugate characters of Z_65 == (0,)  [odd: no Nyquist]",
           self_conjugate_characters(65) == (0,))

    even = PeriodicDomain.periodic_1d(8)
    xs = even.mesh()[0]
    real_nyquist = torch.cos(4 * xs).unsqueeze(0)              # k = N/2 = 4
    _check("real field: derivative multiplier at k=N/2 is zeroed",
           float(even.derivative_multipliers(real_nyquist)[0][4].abs()) == 0.0)
    _check("real field: gradient of the Nyquist mode is identically zero",
           float(even.gradient(real_nyquist).abs().max()) < 1e-12)
    complex_nyquist = real_nyquist.to(torch.complex128)
    _check("complex field: full signed convention retained",
           float(even.derivative_multipliers(complex_nyquist)[0][4].abs()) > 0.0)
    _check("gradient is exact below Nyquist", torch.allclose(
        torus.gradient(torch.sin(5 * x).unsqueeze(0))[:, 0],
        5 * torch.cos(5 * x).unsqueeze(0), atol=1e-10))

    print("\nchild 2 — DirichletBox  sine basis, non-periodic BCs")
    box = DirichletBox(interior=63, length=math.pi)
    xj = torch.arange(1, 64, dtype=torch.float64) * box.spacing
    mode = torch.sin(4 * xj).unsqueeze(0)                       # k = 4, L = pi

    _check("DST-I round-trip", torch.allclose(
        box.synthesize(box.analyze(mode)), mode, atol=1e-11))
    _check("laplacian of sin(4x) == -16 sin(4x)  [exact eigenvalue]", torch.allclose(
        box.laplacian(mode), -16.0 * mode, atol=1e-9))
    _check("eigenvalues are (k pi / L)^2", torch.allclose(
        box.eigenvalues()[:3],
        torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64), atol=1e-12))
    _check("kinetic propagator is norm-preserving", torch.allclose(
        box.mass(box.kinetic_propagator(mode.to(torch.complex128), 0.9, 0.01)),
        box.mass(mode), atol=1e-11))
    _check("no gradient method — sine basis is not closed under d/dx",
           not hasattr(box, "gradient"))

    print("\nchild 3 — ChebyshevLobatto  non-uniform measure, rung 2 only")
    cheb = ChebyshevLobatto(points=12, interval=(-1.0, 1.0))
    nodes = cheb.nodes()
    ones = torch.ones(1, 12, dtype=torch.float64)

    _check("integrate(1) == 2", abs(float(cheb.integrate(ones)) - 2.0) < 1e-12)
    _check("integrate(x^2) == 2/3  [interpolatory exactness]",
           abs(float(cheb.integrate((nodes**2).unsqueeze(0))) - 2 / 3) < 1e-12)
    _check("weights genuinely vary per point",
           float(cheb.quadrature_weights().std()) > 1e-3)
    _check("declines rung 3", not isinstance(cheb, SpectralOperatorDomain))
    _check("still a HilbertSpace", isinstance(cheb, HilbertSpace))
    _check("tensor-free instance", not any(
        isinstance(v, torch.Tensor) for v in vars(cheb).values()))

    print("\npolymorphism — one nonlinear map, three spaces, rung 2 only")
    for name, space, field in (
        ("PeriodicDomain ", torus, wave),
        ("DirichletBox   ", box, mode.to(torch.complex128)),
        ("ChebyshevLobatto", cheb, (nodes * (1 - nodes**2)).unsqueeze(0).to(torch.complex128)),
    ):
        perturbed = field * 1.7
        rescaled = rescale_to_mass(perturbed, field, space)
        _check(f"{name}: rescale_to_mass hits the reference mass exactly",
               torch.allclose(space.mass(rescaled), space.mass(field), rtol=1e-12))

    print("\ngradient safety — the NaN the production code still has")
    g = torch.randn(3, 64, dtype=torch.complex128, requires_grad=True)
    ref = torch.randn(3, 64, dtype=torch.complex128)
    ref[1] = 0
    rescale_to_mass(g, ref, torus).abs().sum().backward()
    _check("finite gradient with a zero-mass reference sample",
           bool(torch.isfinite(g.grad).all()))

    print("\nall checks passed — 3 inhabitants, 2 of them on rung 3\n")


if __name__ == "__main__":
    main()
