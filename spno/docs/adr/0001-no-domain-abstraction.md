# ADR 0001 — No domain abstraction until a second domain exists

- **Status:** proposed (awaiting approval of the Phase 1 review)
- **Scope:** `src/spno/domain.py` and the 22 modules that import it
- **Supersedes:** nothing
- **Related:** `docs/SPNO-Mathematics.tex` §`sec:projection`; `CLAUDE.md`
  (config-hashing convention); `docs/prompts/domain-abstraction-review.md`

## Context

`PeriodicDomain` is the geometric foundation of the study. Twenty-two modules under
`src/spno/` import from `domain.py`; nine test modules and six phase scripts construct
domains directly. It is a constructor argument on every `StepOperator`, and
`DataConfig.domain` is a `@property` that returns a freshly built instance on each access.

The module holds four *named* concepts: geometry (`shape`, `lengths`, `spacing`,
`cell_volume`, `spatial_axes`, `mesh`, `validate_field`, `validate_batched_field`), a
Fourier basis and its calculus (`wave_vectors`, `wave_number_squared`,
`spectral_gradient`, `spectral_laplacian`), an invariant (`l2_mass`), and a nonlinear map
(`project_to_mass`).

The question raised was whether to generalise it, along one of: non-Fourier basis
(Chebyshev), non-periodic boundary conditions (Dirichlet/Neumann), non-uniform
measure/quadrature, or unstructured geometry. Three designs were specified concretely — a
`typing.Protocol`, an ABC hierarchy (`MeasureSpace → HilbertSpace → SpectralDomain`), and
composition (a domain *has* a grid, a measure, a basis).

Note that **dimension is not the missing generality**: `shape: tuple[int, ...]` is already
n-dimensional and `spatial_axes` derives from `dim`.

## Decision

**Build none of the three.** Keep `PeriodicDomain` as the single concrete class, at its
current name and module path, with its current public API. Land only additive
improvements (see *Consequences*).

## Rationale

**1. There is no second inhabitant, and none is planned.** `docs/` was searched for
*chebyshev, dirichlet, neumann, non-periodic, non-uniform, unstructured, sphere, torus,
2D, two-dimensional* across `SPNO-Mathematics.tex`, `PROMPT_PHASES_6_TO_9.md`, both
`superpowers/plans/` documents and the remediation log. The only hits are inside the
review brief itself. The Phase 0–9 plan is one equation on one periodic interval. A
hierarchy with one inhabitant is speculative generality.

**2. The stronger reason: the domain is already more general than its consumers.**
`PeriodicDomain` supports any dimension. Four of its consumers refuse anything but one:

| Site | Refusal |
| --- | --- |
| `models/fno.py:107` | `raise NotImplementedError("FNOStepOperator is implemented for 1D domains")` |
| `models/split_learned.py:320` | `if domain.dim != 1:` |
| `evaluation/resolution.py:47` | `raise NotImplementedError("spectral_resample is implemented for 1D domains")` |
| `data/generate.py:116` | `if family != "random" and domain.dim != 1:` |

The only non-1-D domain anywhere in the repository is a single `(8, 8)` grid in
`tests/test_datasets.py:138`. The generality that already exists is unexercised. Adding a
second, wider layer beneath it does not remove the four refusals — it gives the project
two unexercised abstractions instead of one.

**3. The abstraction's own test story would require fabricating its justification.**
Testing a hierarchy meaningfully means writing a second domain to test it against. That
second domain would exist only to validate the abstraction whose absence of demand is the
reason not to build it. Phase 8's honest-reporting discipline exists precisely to stop the
study claiming transfer it has not demonstrated; an abstraction with one inhabitant makes
the same kind of claim about the code.

**4. Composition (design C) additionally fails the additive constraint.** Preserving the
current public surface requires ~11 delegating properties, at which point the composition
is invisible to every caller. The honest version — callers writing
`domain.basis.wave_vectors(...)` — is a rename across 22 modules. There is also a
modelling objection: on a uniform periodic grid the measure and the basis are *not*
independent choices (Parseval ties `Δx` to the `1/N` convention), so a design that lets
you mix them freely lets you construct silently inconsistent spaces. Composition sells
configurability the mathematics does not have.

## The dissent, recorded

On taste, **design B** is the attractive one. `MeasureSpace → HilbertSpace →
SpectralDomain` is the vocabulary `SPNO-Mathematics.tex` already speaks; `l2_mass` becomes
`space.norm(ψ)**2`, Parseval becomes statable in code, and Trap A (below) gets a public,
obviously-load-bearing home instead of being a private helper.

It is still the wrong thing to build now, for three reasons. Its shape would be inferred
from a single example, and the most likely inference is wrong (see *When to revisit*). Its
main benefit is *expository*, and a page of the thesis explaining the correspondence costs
nothing and risks nothing, while the hierarchy puts a tensor-shaped hole in a tensor-free
object and two dispatch layers in front of every FFT. And its test story is the
fabrication described above.

## When to revisit, and what to build

**Trigger:** a second concrete domain named in a phase plan **with an owner** — not a
hypothetical and not a test fixture. Realistically a Dirichlet box for a trapped
condensate, which is the only physically motivated NLS successor. Build the abstraction
*from two examples*, not one.

**And expect a narrower shape than the sketch in the review.** The three-level hierarchy
has a specific flaw worth recording so it is not rediscovered:

```python
class MeasureSpace(abc.ABC):
    shape: tuple[int, ...]

    @property
    def spatial_axes(self) -> tuple[int, ...]: ...

    @abc.abstractmethod
    def integrate(self, density: Tensor) -> Tensor:
        """The operation, not the weights."""


class HilbertSpace(MeasureSpace):
    def inner(self, u: Tensor, v: Tensor) -> Tensor:
        return self.integrate(u.conj() * v)

    def norm(self, u: Tensor) -> Tensor:
        return torch.sqrt(self.inner(u, u).real)
```

Two levels, not three, and `integrate` abstract as an **operation** rather than as a
`quadrature_weights()` accessor:

- A weights accessor returning a tensor puts a tensor-shaped hole in an object whose
  entire value is being tensor-free — it is the exact shape of the caching mistake the
  device/dtype discipline forbids. `integrate` lets a uniform grid implement it as
  `sum * cell_volume` and never materialise a weight tensor, while a Gauss–Lobatto grid
  materialises one per call with explicit `device`/`dtype`.
- A scalar-shaped weight is a Fourier-uniform artefact. Any non-uniform successor wants
  per-point weights, so a `quadrature_weights` signature changes on first contact with
  the second inhabitant — which is precisely the failure mode of single-example
  generalisation.
- `SpectralDomain` stays out. `analyze`/`synthesize` as methods is where `torch.compile`
  loses sight of `torch.fft.fftn`, and free functions taking a domain — which is what
  `spectral_gradient` and `spectral_laplacian` already are — cost nothing and read fine.

**`project_to_mass` must not become `HilbertSpace.project`.** It is the *metric*
projection onto the fixed-mass sphere: a radial rescale, nonlinear
(`SPNO-Mathematics.tex`, `prop:metricproj`). Naming it `project` launders a nonlinear map
into linear-operator vocabulary. It stays a free function with a verb that says it
rescales.

## Invariants that must not be "cleaned up"

Both look like dead code and are not. Anyone touching `domain.py` or
`evaluation/resolution.py` should read these first.

**Trap A — `domain.py:113–132`, `_first_derivative_wave_vectors`.** For *real* fields on
even-sized grids the Nyquist mode is zeroed before the first-derivative multiplier is
applied. That mode is self-conjugate — it represents `cos(Nx/2)` with no way to
distinguish `+k` from `−k` — so it has no signed first derivative, and zeroing it is what
preserves Hermitian symmetry of the result. Complex fields take the early return and keep
the full signed convention; deleting that branch is not a simplification.

**Trap B — `evaluation/resolution.py:57–66`, `spectral_resample`.** Upsampling a field
carrying energy at the *source* Nyquist is **refused**, not handled. Splitting a
self-conjugate mode between the two target modes on the finer grid is a choice rather than
a fact, and silently making one would put an arbitrary phase into every downstream number.
Pinned by `tests/test_resolution.py:49`.

## Consequences

**Positive.** Zero blast radius across 22 modules. `config_hash` provably unmoved, so no
data shard or results directory is orphaned and every Phase 2–5 checkpoint stays loadable
by Phases 6–9 (`checkpoints.py:57` compares `metadata.data_hash` against
`config_hash(DataConfig())` — the hash and the checkpoints are one constraint, not two).
The device/dtype discipline that lets one domain serve MPS-float32 training and CPU-float64
invariant evaluation is untouched. Net *fewer* untested behaviours, because Trap A gains
the test it never had.

**Negative.** `domain.py` keeps one genuine cohesion defect: `project_to_mass` is
model-specific and nonlinear, used by exactly one module, and cannot be moved to
`models/projected.py` without an import cycle (`models` imports `domain`; the same cycle
is documented at `evaluation/resolution.py:93`). One concept plus one stowaway. The
docstring should say so out loud rather than listing it as if it belonged. The correct
eventual home is a third module both can import — deferred, because the move buys nothing
today.

**Also negative.** The expository win of design B is forgone. Mitigation: the
correspondence between the code and the mathematics chapter is written in prose in the
thesis, which costs nothing and risks nothing.

## Additive work approved alongside this decision

1. Fix the `torch.where` NaN gradient in `project_to_mass` (`domain.py:230–235`) — the
   unselected branch evaluates `sqrt(0)`, whose infinite derivative the mask turns into
   `0 × ∞ = NaN`, poisoning the whole batch's gradient under projection-in-the-loop.
2. Add the missing test pinning Trap A.
3. Add a `volume` property, deduplicating `math.prod(domain.lengths)` at
   `equations/nls.py:123` and `evaluation/dispersion.py:59`.
4. Enforce `spatial_broadcast`'s shape contract, and normalise `shape` integrally in
   `__post_init__` (accept `numpy` integers, reject `bool`).
5. Optionally relocate `batch_parameter`'s precision policy to `precision.py` behind a
   re-export — it is policy, not geometry.
6. This ADR.

Full specifications, proposed diffs and verification commands:
`PHASE1_TASKS.md` in the handoff bundle.
