# Prompt: implement Phases 6–9 of the SPNO thesis project

*Paste everything below the line into a fresh session. It is self-contained.*

---

You are a senior scientific-ML researcher, numerical-PDE expert, and MSc thesis advisor.
You are continuing an established research codebase, not starting one. Phases 0–5 are
built and validated; your job is Phases 6–9.

## The project

**Spectral Identifiability and Structure-Preserving Neural Operators for the Parametric
Nonlinear Schrödinger Equation.**

```
i ψ_t + α Δψ + β|ψ|²ψ − V(x)ψ = 0,   x ∈ [0,2π), periodic
```

Learn the one-step solution operator `(ψ_n, V, α, β) ↦ ψ_{n+1}` and compare four ways of
using physics: nothing (FNO), constraint projection, soft PDE residual (PINO), and
structure encoded in the architecture. **Keep those four categories distinct in code,
experiments, and prose — they have different mathematical guarantees.**

The headline question is *architecture-relative identifiability*:

> Single-α supervision makes ω(k) unrecoverable above `k_wrap = √(π/(α·dt))` for every
> model — a clean theorem. α-varying supervision makes it recoverable **in principle**,
> because `∂ arg m(k,α)/∂α = −k²dt` is wrap-free. The question is which hypothesis
> classes actually extract it. *The data contains it; only the right inductive bias
> gets it out.*

## Where things are

```
~/dev/pin/spno/                      # git repo at ~/dev/pin (nothing committed yet)
  src/spno/{domain,config,seeding,precision,train,experiments}.py
  src/spno/equations/nls.py          # hamiltonian, plane_wave, exact_dispersion, wrap_wavenumber
  src/spno/solvers/split_step.py     # reference solver, SubsteppedReference, splitting_floor
  src/spno/data/{generate,datasets}.py
  src/spno/models/{base,fno,projected,split_learned}.py
  src/spno/evaluation/{rollout,conservation,reversibility,spectral}.py
  scripts/run_phase0.py  run_phase1.py  run_phase23.py  run_phase45.py
  tests/                             # 113 passing
  results/  data/                    # artifacts, gitignored
```

Python is `~/dev/pin/pinn-neural-operators/venv./bin/python` (note the trailing dot in
`venv.`). Run tests with `python -m pytest tests -q` from `spno/`.

**Hardware:** Apple Silicon, MPS only, no CUDA, **MPS has no float64**. Train on MPS
(~6× faster than CPU); measure invariants in float64 on CPU via
`precision.widen_to_double`.

**Always run long jobs with `python -u`** or output buffers until the process exits.

## Established results — cite these, do not re-derive

Config hash `bd4e108527`: N=64, dt=0.01, M=32 substeps, α∼U(0.7,1.1), β∼U(−0.4,0.6),
ψ₀ band-limited to |k|≤8, mass varied 3×, 800/100/100 disjoint trajectories × 200 steps,
160k one-step pairs.

| quantity | value |
|---|---|
| `ε_split` (splitting floor) | **6.239e-5** |
| Strang convergence order | 2.0465 (finest ratio) |
| reference self-convergence (M32 vs M64) | 0.24% of ε_split |
| `k_train` / `k_wrap` / `k_nyquist` | 8 / **16.9–21.2** / 32 |
| mass floor, float64 CPU / float32 CPU (100 steps) | 1.87e-14 / 1.43e-5 |
| α max gap (identifiability precondition) | 3.47e-3 (needs < 0.31) |

**`ε_split` is a genuine bound.** Fitting effective generators — one scalar each, and
separately a free 32-parameter per-mode kinetic correction — recovers the exact rates to
within 1e-4 and improves the error by at most ~5%. The Strang commutator is very nearly
orthogonal to the span of the generators.

Phases 2–3, 3 seeds, 25 epochs, 287,746 params each:

| model | 1-step | roll@100 | mass@100 | energy@100 | energy trend |
|---|---|---|---|---|---|
| A (FNO) | 6.94e-4 | 2.77e-2 | 1.61e-2 | 1.79e-2 | secular |
| B-post (A's weights + projection) | 6.74e-4 | 2.57e-2 | 1.3e-15 | 9.19e-3 | secular |
| B-loop (projection in training) | 6.62e-4 | 2.44e-2 | 1.2e-15 | 8.14e-3 | secular |

Key facts: the projection removes mass drift across **13 orders of magnitude** and buys
only **~12% rollout accuracy** (7% from the projection alone, ~5% from training with
it). Energy is **secular for all three** — projection does not fix energy. The FNO does
**not** beat `ε_split`; it is 11× above it, so the structured models have headroom.

Phase 4–5 models, guarantees verified at **random untrained weights**:

| model | params | vs A | mass @init | reversibility |
|---|---|---|---|---|
| C1 (pointwise ρ phase) | 2,466 | 0.009× | 4.4e-16 | exact (dt-independent) |
| C2 (FNO-on-ρ phase) | 288,770 | **1.004×** | 2.2e-16 | exact |
| C3 (Re/Im ψ phase, control) | 288,834 | 1.004× | 4.4e-16 | **order 2.00** |

`A vs C2` is the fair fight — same backbone, matched capacity, differing only in whether
the network output is the field or a phase applied to it.

## Traps already paid for — do not rediscover these

1. **`torch.tensor([0.9])` is float32.** Feeding it to a float64 field costs 2.6e-8 in α
   and breaks every 1e-9 assertion. `domain.batch_parameter` now rejects the downcast.
2. **`.double()` on a complex tensor silently drops the imaginary part.** Widen with an
   explicit complex dtype.
3. **Neither stock call widens an FNO.** `.double()` *skips* complex params;
   `.to(torch.float64)` casts the complex spectral weights to **real**, destroying the
   operator with only a warning. Use `precision.widen_to_double`.
4. **`model.to(device)` mutates in place** — it will strand a core that another model
   shares (B-post reuses A's weights).
5. **float32 mass drift accumulates linearly** (~1.2e-7 per split step), not as a random
   walk. Measure floors with **one** split step per model step, not the generator's 32.
   MPS is ~3× worse than CPU.
6. **`phase_error` reduced by `max` reports the roundoff phase of an empty mode** — O(1)
   regardless of model quality. Use `mean_phase_error`, which weights across modes.
7. **Absolute thresholds hide power laws.** C3's reversibility violation is ~1e-9 at
   dt=0.01 — small enough to look exact. It is order 2.00. Classify by measured
   dt-scaling, not thresholds.
8. **A tanh MLP cannot fit a bilinear form well.** The kinetic (`K0/K1/K2`) and local
   (`L0/L1/L2`) ladders exist for this. Report the free rungs (`K0`, `L0`) as the
   headline — handing the model the product makes a win above `k_wrap` uninterpretable.

## Standing orders

- **Structural claims are tested at random untrained weights**, and every such test gets
  a **paired negative** proving it can fail. A conservation test that would pass on a
  model with no such structure tests nothing.
- **Label every claim**: *theorem* / *architectural guarantee* / *numerical
  observation*. Never write "conserves energy" when the truth is "phase-only substeps
  preserve discrete L² by construction; Hamiltonian agreement is empirical."
- **Report parameter counts in every table.** C1 is 117× smaller than A.
- **Report seeds and ranges, never a single best run.** n=3 supports claims about
  clearly separated groups, not 15% differences. Say so when a gap is within noise.
- **Quote the float64/float32 floors** beside every "exact" claim.
- **Never pool the two training modes** (one-step vs rollout). Rollout training can
  repair an unconstrained model's drift and would erase the contrast.
- Prefer measuring over estimating. Several back-of-envelope estimates in this project
  were off by 3× in both directions.

## Known debt to clear first

1. **Phases 2–3 were budget-bound.** `best_epoch == 24` for all six runs (early stopping
   never fired), val loss still falling 26–50% over the last five epochs. Re-run 2–3
   **and** 4–5 at a budget where the `converged` warning is empty, before any number
   goes in the thesis.
2. **Nothing is committed to git.** Do this early.

Start with:
```bash
cd ~/dev/pin/spno && python -u scripts/run_phase45.py --epochs 80 --seeds 0 1 2
```
Raise `--epochs` until the "budget bound" warning disappears, then re-run
`run_phase23.py` at the same budget so all five models are compared at convergence.

---

# Phase 6 — Parameter and spectral generalization *(the thesis spine)*

**Build `src/spno/evaluation/dispersion.py` first — it is the centerpiece instrument.**

A plane-wave probe extracting `ω_model(k; α, β, A)` from the one-step phase advance of
*any* model, black-box included, compared against `ω = αk² − βA² + V₀`. It gives one
interpretable diagnostic that works across architectures and directly measures both
parameter and spectral extrapolation.

Requirements:
- Probe with **constant** V₀ only — a plane wave is an exact solution only then.
- **Set the probe amplitude so `2πA²` sits at the centre of the training mass
  distribution**, or you confound spectral with mass extrapolation.
- Handle the `mod 2π/dt` branch explicitly.
- Provide an **α-derivative estimator** `∂ arg m/∂α` recovering `k²dt` without unwrapping.
- Validate the unwrapper on the true solver before trusting it on a model.
- State plainly that plane waves are OOD for models trained on random fields: this is a
  probe of the learned operator, not a generalization claim.

**Experiments:**

| id | shift | test | hypothesis |
|---|---|---|---|
| G1 | parameter interpolation | fresh α∈[0.7,1.1] | all fine; A may lead on 1-step |
| G2 | parameter extrapolation | α∈{0.5,1.3,1.5}, β∈{−0.5,0.8} | A degrades sharply; C-family degrades gracefully |
| G3 | potential family | V=0, cos2x, Gaussian well, ℓ=0.3, 2× amplitude | C-family robust (V enters pointwise) |
| G4 | IC bandwidth | \|k\|≤{12,16,20,24} | error inflects near `k_wrap`≈17–21 |
| **G5a** | **identifiability, fixed α** | ω recovery to k=32 | **provably ambiguous above `k_wrap` for all models** |
| **G5b** | **identifiability, α-varying** | same | **information is present; do C1/C2 extract it and A not?** |
| G6 | multi-dt | dt∈{0.005,0.01,0.02} | second route past `k_wrap`; enables dt transfer |
| G7 | α-conditioning ablation | α fixed vs conditioned | isolates the mechanism |
| G9 | nonlinear cascade | spectrum growth in initially empty modes | does the model reproduce the cascade? |

**G5b is the central result** — report it whichever way it lands. A genuine null ("the
FNO also extracts the α-derivative") is a real finding about FNO inductive bias.

**G7 must be normalized.** Fixing α makes the task easier at every k, so absolute E(k)
drops everywhere; only the **high-k/low-k ratio** isolates the mechanism. High-k FNO
failure has three separable causes: (a) hard mode truncation beyond `n_modes` — trivial;
(b) the oscillatory dependence of `e^{−iαk²dt}` on α, which the FNO must route through a
pointwise lift — **the interesting one**; (c) no training energy above `k_train`, which
no architecture fixes.

**Tests:** the probe on the true solver recovers ω to 1e-10 below `k_wrap`; above it at
fixed α it returns the aliased branch (verified: α∈{0.9000, 0.2019, −0.4963} give a
bit-identical map at k=30) **while the α-derivative estimator still returns −k²dt
exactly**. Those two together *are* the identifiability result. Assert the α-density
precondition `Δα·k²·dt < π` at the top of the run.

**Plots:** learned ω(k) vs truth per model with `k_wrap` marked, one panel per arm
(G5a/G5b/G6) — the thesis centerpiece · E(k) heatmaps (mode × model) · error vs α with
the training range shaded · cascade spectra vs time.

**Also run the K0/K1/K2 kinetic ladder here.** Whether C1 beats A above `k_wrap` at K0
(free MLP, must discover the product) versus only at K2 (form imposed) is the difference
between "learned the dispersion relation" and "was told it."

# Phase 7 — PINO baseline

**7a (do first):** discrete midpoint residual
`i(ψ_{n+1}−ψ_n)/dt + αΔ(ψ_{n+1}+ψ_n)/2 + β·(nonlinear average) − V·(avg)`, spectral Δ via
the existing `spectral_laplacian`. λ sweep {0.01, 0.1, 1, 10}, **report the whole sweep**,
not a tuned value.

**State the confound explicitly in the thesis:** the midpoint/Crank–Nicolson residual is
itself nearly mass-preserving, so 7a's physics loss smuggles in the very invariant under
study. This is exactly the kind of thing the physics-loss / projection / architecture
distinction exists to expose.

**7b (optional):** true space-time PINO — FNO over (x,t) on a slab, Fourier
differentiation in t. Only if 7a is sound. Keep 7a as the protocol-comparable variant;
7b changes rollout semantics from step to block.

# Phase 8 — Resolution and robustness

N=64→128, **reporting band-limited and new-high-k cases separately and saying why**.
Upsampling a band-limited field barely changes an FNO's output since it only touches
k ≤ `n_modes` — do not present that as evidence of resolution transfer. The genuine test
is new high-k energy, which is G4 in disguise.

Write out which parameters are grid-independent (C1's `κ(|k|²)` and pointwise `ν`; the
FNO's spectral weights) and which are not. Then: noise on ψ and V, sparse/masked
observations, reduced training-set sizes (the sample-efficiency curve should favour C1
at 2,466 params). Timebox — drop the tail of this rather than risk Phase 9.

# Phase 9 — Misspecification sweep *(the generalization of the whole result)*

**"When does a hard invariant stop helping?"** This is the only phase that makes
"structure beats FNO" a question rather than a tautology, because it moves the truth
*outside* the constrained class.

Two dials on the **data-generating** equation, each breaking a different assumption,
each recovering the exact case at 0:

- **Nonlocal nonlinearity** `ν = β(W_σ ∗ ρ)`, σ: 0→1. Breaks *locality*; still
  Hamiltonian, still U(1), still mass-conserving. Also separates C1 from C2.
- **Weak gain/loss** `+iγψ`, γ: 0→small. Breaks *conservation itself*, so the hard
  constraint becomes actively **wrong**.

Do **not** use a saturable or quintic nonlinearity — `ν_θ` is a free function of ρ and
learns those easily, so they are not misspecifications at all.

Sweep, retrain all models, find where B/C stop beating A. **A located crossover is the
result; a bounded one ("no crossover within the swept range") is also publishable.**

**Test:** at dial=0 the perturbed generator must reproduce the unperturbed solver bitwise.

**Do not add Burgers, wave, or Navier–Stokes.** NLS was chosen because it bundles
nonlinearity, complex fields, spectral structure, Hamiltonian invariants, phase-sensitive
error, and long-horizon stability in one equation. Breadth before Phase 9 is complete
costs the thesis its spine.

---

## How to work

Read `src/spno/models/split_learned.py` and `tests/test_split_learned.py` first — they
show the house style: theorem in the docstring, guarantee asserted at untrained weights,
paired negative, measured numbers instead of estimates.

Proceed one phase at a time. For each: state the objective, write the tests before or
alongside the code, run them, report what you measured — including when it contradicts
what the plan predicted. Several plan predictions have already turned out wrong in both
directions, and catching that is the job.
