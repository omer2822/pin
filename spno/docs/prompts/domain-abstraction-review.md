You are acting as engineering manager, staff SWE, and PM at once on the `spno/` project.
Two jobs, in order, with a hard gate between them.

## Ground rules before you read anything

Read these files. Do not "explore the codebase" — the targets are named:

- `spno/src/spno/domain.py`            — the subject
- `spno/src/spno/config.py`            — `config_hash`, `DataConfig.domain`
- `spno/src/spno/models/base.py`       — `StepOperator.__init__` stores the domain
- `spno/src/spno/precision.py`         — the widen-to-float64 pattern
- `spno/src/spno/evaluation/resolution.py` — `spectral_resample`, Nyquist refusal
- `spno/docs/SPNO-Mathematics.tex`     — §projection, the mass-sphere proposition
- `CLAUDE.md`                          — repo conventions

Then run, and paste the output into your notes:

    cd spno && grep -rn --include="*.py" -E "PeriodicDomain|spectral_gradient|spectral_laplacian|l2_mass|project_to_mass|batch_parameter|spatial_broadcast|wave_number_squared|wave_vectors|cell_volume|spatial_axes" src tests scripts

22 modules depend on `domain.py`. That number is the whole story of this task.

## The constraint contract — six hard constraints, not preferences

A design that violates any of these is **disqualified**, not "scored lower."

1. **`config_hash` must not move.** `config_hash()` is `sha256(json.dumps(dataclasses.asdict(config)))`.
   `DataConfig.domain` is a `@property`, so the domain is *not* in the hash payload today.
   Putting any domain/space object into a config dataclass **as a field** changes every hash
   and orphans every data shard and results dir on disk. CLAUDE.md records this exact hazard
   for `MisspecificationConfig`. Record `config_hash(DataConfig())` and `config_hash(Phase0Config())`
   BEFORE you touch anything; assert both unchanged after.

2. **The space object stays tensor-free.** Wave vectors are derived per call with explicit
   `device`/`dtype` on purpose — that is what lets one domain serve MPS/CUDA-float32 training
   and CPU-float64 invariant evaluation. A `HilbertSpace` that *caches* a basis breaks the
   precision-widening pattern, breaks frozen-dataclass hashability, and makes `widen_to_double`'s
   `deepcopy` drag tensors around. No caching, no stored tensors, no `.to()` on the space.

3. **Checkpoints cross phases.** `PeriodicDomain` is a constructor arg on every `StepOperator`.
   Phases 6-9 load checkpoints trained by Phases 2-5. Acceptance is NOT "tests pass" — it is
   *an existing checkpoint under `results/` still loads and reproduces its recorded metrics*.

4. **Phase 1 is additive only.** `PeriodicDomain` keeps its name, its module path, and its full
   current public API. `__init__.py` keeps exporting the same five names. Any general layer goes
   *underneath*. Blast radius across those 22 modules must be zero.

5. **Two Nyquist traps must survive.** `_first_derivative_wave_vectors` zeroes the Nyquist mode
   for *real* fields on even grids (domain.py:113-132). `spectral_resample` *refuses* to upsample
   a field carrying energy at the source Nyquist (evaluation/resolution.py). Both are deliberate
   correctness decisions that look like dead code to a cleanup pass. Name them as invariants and
   point at the tests that pin them.

6. **`project_to_mass` is not an orthogonal projection.** It is a radial rescale — the *metric*
   projection onto a fixed-mass sphere, nonlinear (see the proposition in SPNO-Mathematics.tex).
   If your abstraction names it `HilbertSpace.project`, you have laundered a nonlinear map into
   linear-operator vocabulary. Do not.

Tests: `uv run pytest tests -q` from `spno/`. `filterwarnings = ["error::UserWarning"]` — a stray
warning fails the test that raises it.

---

## PART 1 — Review `PeriodicDomain`

Not a style pass. Answer, with evidence from actual call sites:

- **What earns its keep.** For each member (`shape`, `lengths`, `spacing`, `cell_volume`,
  `spatial_axes`, `mesh`, `wave_vectors`, `wave_number_squared`, `validate_field`,
  `validate_batched_field`, `periodic_1d`): who calls it, how often, and would removing it hurt?
- **What is incidental.** Which parts are essential geometry vs. conveniences that happened to
  land here? Is `batch_parameter`'s float32-rejection policy geometry at all, or does it belong
  elsewhere?
- **Cohesion.** `domain.py` currently holds geometry + differentiation + an invariant + a
  nonlinear projection. Is that one concept or four? Argue it, don't assert it.
- **The frozen-dataclass choice.** `__post_init__` mutates through `object.__setattr__` to
  normalize tuples. Is frozen-plus-normalize the right call, or is there a cleaner construction?
- **Correctness.** Anything actually wrong — the `spatial_broadcast` shape contract, the `eps`
  handling in `project_to_mass`, `validate_field`'s leading-axis permissiveness vs.
  `validate_batched_field`'s strictness.

## PART 2 — Three candidate generalizations, then a recommendation

State the axis first. `PeriodicDomain` is **already** n-dimensional (`shape: tuple[int, ...]`,
`spatial_axes` derives from `dim`). So dimension is not the missing generality. The real axes are:
non-Fourier basis (Chebyshev), non-periodic BCs (Dirichlet/Neumann), non-uniform measure/quadrature,
unstructured geometry. Say which axis you are generalizing along and why *that* one.

Design three, concretely — with the actual Python signatures, not prose:

- **A — `typing.Protocol`**, structural typing, no inheritance
- **B — ABC hierarchy**, e.g. `MeasureSpace` (quadrature, `integrate`) → `HilbertSpace`
  (`inner`, `norm`) → `SpectralDomain` (basis, transforms), with `PeriodicDomain` as one instance
- **C — composition**, a domain *has* a measure, a basis, and an inner product as collaborators

Score each on a table: blast radius across the 22 modules · `config_hash` stability ·
device/dtype discipline (constraint 2) · checkpoint compatibility · `torch.compile` / autograd
friendliness · test cost · how it reads to a reviewer of the thesis.

Then recommend one — **or recommend none of them.** "None" is a legal, and possibly correct,
outcome: there is exactly one concrete inhabitant today, and a hierarchy with one inhabitant is
speculative generality. To recommend a refactor you must point at a *second concrete domain that
is actually on the roadmap*. I have already grepped `spno/docs/` for chebyshev / dirichlet /
neumann / non-periodic / non-uniform / unstructured / sphere — nothing came back. If you still
recommend building the abstraction, that evidence gap is yours to close, explicitly. Write the
YAGNI case against your own favourite before you pick it.

Close with a migration path for the recommended option: what lands in Phase 1 (additive, zero
blast radius), what would only be justified when a second domain actually arrives, and the exact
verification commands including the `config_hash` before/after check.

---

## Deliverable and the gate

Load the `artifact-design` skill before writing any HTML. Then write the review as a single
HTML file under `spno/docs/` and publish it as an Artifact. Title it as a short noun phrase.
It should carry: the review findings, the constraint contract, the three designs with real
signatures, the comparison table, the recommendation with its dissent, and the migration path.
A diagram is worth it only if it shows the actual dependency structure — if you draw one, load
`artifact-diagramming` first.

**Then STOP.** Hand me the link and wait. Write no production code, change no file under
`src/`, run no refactor until I explicitly approve the recommendation. Phase 2 is a separate
instruction from me.

If Part 1 turns up a real bug in the current code, say so in one line at the top of your
terminal response rather than burying it in the page.
