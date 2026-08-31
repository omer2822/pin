# 2. Build the domain abstraction (Candidate B), overriding ADR 0001

## Status

Accepted, superseding [ADR 0001](0001-no-domain-abstraction.md).

## Context

ADR 0001 recorded the Phase 1 review's recommendation: build none of the three
generalisations it evaluated (`typing.Protocol`, an ABC hierarchy, or composition),
because at the time there was exactly one concrete domain in the codebase and the
consumers of `PeriodicDomain` (`models/fno.py:107`, `models/split_learned.py:320`,
`data/generate.py:116`, `evaluation/resolution.py:47`) were already hard-limited to
`dim == 1`. Its stated trigger condition for revisiting was: *"a second concrete
domain is written down in a phase plan with an owner."* That condition was not met
before this ADR.

The owner explicitly requested the abstraction be built anyway, in this session,
pointing at Candidate B from the review (`MeasureSpace -> HilbertSpace ->
SpectralDomain`) by name -- the ABC hierarchy the review's own author said they
would build "if choosing on taste," because it speaks the same `L^2` / measure /
Hilbert-space vocabulary `docs/SPNO-Mathematics.tex` already uses.

## Decision

Build Candidate B as a strict, additive superset of the existing module:

- `MeasureSpace(abc.ABC)` -- a discrete point set with a quadrature rule
  (`quadrature_weights`, `integrate`). Tensor-free by contract, matching constraint 2
  of the original review (derived per call, never cached).
- `HilbertSpace(MeasureSpace)` -- adds `inner` and `norm`. Deliberately does **not**
  add a `project` method: `project_to_mass` stays a free function with a verb that
  says it is a nonlinear radial rescale, not a linear projection (constraint 6 /
  ADR 0001's dissent -- preserved here, not relitigated).
- `SpectralDomain(HilbertSpace)` -- adds `analyze`, `synthesize`,
  `laplacian_multiplier`, and `derivative_multiplier`. The last of these is Trap A's
  new public, named, tested home, replacing the private
  `_first_derivative_wave_vectors` helper the original review flagged as "one
  cleanup pass away from silently breaking."
- `PeriodicDomain` becomes `SpectralDomain`'s sole concrete implementer. Its name,
  module path, and every pre-existing public method keep their exact signatures and
  behaviour (constraint 4). The free functions `spectral_gradient` and
  `spectral_laplacian` were refactored to delegate to the new methods rather than
  duplicate their logic, and are verified bit-identical to the pre-refactor
  implementation by `tests/test_domain.py`.

One mechanical finding not in the review's sketch: `MeasureSpace.shape` **cannot**
be declared as `@property @abc.abstractmethod`, as the review's own code block
shows it. A `dataclass` field of the same name with no default never becomes a
class-level attribute, so it has nothing to shadow the base class's abstract
property with in the MRO -- `PeriodicDomain` would be permanently
non-instantiable (`TypeError: Can't instantiate abstract class ... with abstract
method shape`). `shape` is declared as a plain annotation instead, which every
concrete subclass's dataclass field then satisfies with no runtime enforcement
beyond a type checker. This is now a regression-tested invariant
(`test_periodic_domain_is_instantiable_and_satisfies_the_full_hierarchy`).

## Consequences

- `spno.__all__` is unchanged; the new classes are importable from `spno.domain`
  but not re-exported at the package top level, so existing import graphs are
  untouched.
- `config_hash(DataConfig())`, `config_hash(Phase0Config())`, and
  `FNOStepOperator`'s registered buffer set are all unchanged before/after (checked
  via `git stash`), so no results directory, data shard, or checkpoint is orphaned.
- The full suite (291 tests, up from 284 pre-abstraction / 273 pre-Phase-1) passes,
  plus the Phase 6 quick-plumbing acceptance check loading an existing checkpoint.
- ADR 0001's trigger condition -- a second concrete domain named in a phase plan
  with an owner -- is **still not met**. This ADR does not claim it was; it records
  that the owner chose to build ahead of that trigger, on their own authority, and
  why the specific shape (Candidate B) was chosen. If a second domain never
  materialises, the cost of this decision is an unused-but-tested abstraction layer,
  not a correctness risk: every existing call site is untouched.
- The two Nyquist traps (Trap A in `derivative_multiplier`; Trap B in
  `evaluation/resolution.py`) remain load-bearing invariants, now with Trap A in a
  public, tested location rather than a private helper.
