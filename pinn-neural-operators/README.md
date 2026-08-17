# PINNs, Neural Operators, and FNOs — a working quickstart

Five self-contained PyTorch tutorials, no framework, no magic. The first three attack
the **same PDE** — 1D viscous Burgers — so you can see exactly what changes between
the approaches and what it costs. A fourth playground reuses the PINN machinery
across several equations; the fifth generalises the geometry and structural ideas to
parametric PDE operators in multiple spatial dimensions.

```
du/dt + u du/dx = nu d2u/dx2      x in [-1,1) periodic,  nu = 0.02,  t in [0,1]
```

Burgers is the smallest PDE that is actually interesting: the nonlinear term steepens
the wave into a shock, the viscous term smooths it out. Everything you learn here
carries over to Navier–Stokes.

---

## Setup

```bash
pip install torch numpy matplotlib
python common.py            # sanity-checks the reference solver, builds the dataset (~15s)
python 01_pinn_burgers.py   # ~70s
python 02_deeponet.py       # ~30s
python 03_fno1d.py          # ~55s
python 05_3d_equations.py   # no training; runs the 2D toolkit demonstrations
```

CPU is fine — that's what all the timings below are from (4 cores). Figures land in
`figures/`, cached data in `data/`.

Read the docstring at the top of each script before the code. That's where the
explanation lives.

---

## Reusable PINN playground

`04_pinn_playground.py` separates the nested-autograd machinery, neural network,
training loop, and equation definition. List the available problems with:

```bash
python 04_pinn_playground.py --list
```

Then train any registered PDE with the same CLI:

```bash
python 04_pinn_playground.py --pde burgers
python 04_pinn_playground.py --pde heat
python 04_pinn_playground.py --pde reaction-diffusion
python 04_pinn_playground.py --pde schrodinger
```

These examples cover Dirichlet, Neumann, and periodic boundary conditions. Heat and
the default zero-potential Schrödinger setup have analytic solutions, so their figures
and console output report the relative L2 error rather than only the training residual. Figures are saved as
`figures/04_pinn_<pde>.png`; override the path with `--output`.

The Schrödinger example solves

```text
i*psi_t + alpha*psi_xx + beta*|psi|^2*psi - V*psi = 0
```

without requiring complex-valued PyTorch layers. The network emits two real channels,
`psi = a + i*b`. It uses autograd for `psi_t`, then evaluates every collocation time
on a full endpoint-free periodic spatial grid and computes `psi_xx` with an FFT:

```text
psi_hat = FFT(psi)
psi_xx = IFFT(-k^2 * psi_hat)
L_PDE = mean(|i*psi_t + alpha*psi_xx + beta*|psi|^2*psi - V*psi|^2)
```

Its inputs are `(sin(x), cos(x), t)`, making the prediction periodic in space by
construction. The boundary loss additionally matches both the value and spatial
derivative at `-pi` and `pi`. The default is `alpha=0.5`, `beta=0`, and `V=0`; create
the problem programmatically to use a nonlinear term or a periodic potential callable:

```python
problem = SchrodingerProblem(
    alpha=0.7,
    beta=0.35,
    potential=lambda x: 0.2 * torch.cos(x),
    spectral_grid_size=128,
)
```

The callable potential API is intentionally programmatic rather than a CLI string
option, so it can describe arbitrary differentiable periodic potentials.

For a quick wiring check rather than a converged solution:

```bash
python 04_pinn_playground.py --pde schrodinger --steps 2 --lbfgs-steps 0 \
  --collocation 16 --initial 8 --boundary 8 --width 8 --depth 2
```

The defaults are readable starting points, not universally optimal hyperparameters.
Burgers' steepening front, reaction-diffusion dynamics, and an oscillatory complex
wave generally benefit from different sampling budgets and loss weights.

The mathematical residual assertions and tiny optimization smoke test use only the
standard library test runner:

```bash
python -m unittest test_pinn_playground -v
```

---

## PhysicsNeMo v2 heat-PINN playground

`07_physicsnemo_heat_pinn.py` is the framework-backed companion to the
from-scratch playground. It follows the current PhysicsNeMo v2 style: an
explicit PyTorch training loop, a symbolic `physicsnemo.sym.PDE`, and a
`PhysicsInformer` rather than the archived `Solver` / `Domain` API.

It solves the same analytic 1D heat setup used by the general playground:

```text
u_t - alpha*u_xx = 0,  x in [-1, 1],  t in [0, 1]
u(x, 0) = cos(pi*x/2),  u(-1, t) = u(1, t) = 0
```

PhysicsNeMo computes the symbolic spatial derivative `u_xx`; PyTorch autograd
computes `u_t`, because time is not a spatial `PhysicsInformer` coordinate. The
default 10,000-step, 4,096-point budget is intended for an NVIDIA GPU. Install
the optional dependency in this repository's existing virtual environment:

```bash
source pinn-neural-operators/venv./bin/activate
python -m pip install --upgrade "nvidia-physicsnemo[sym]"
```

Then run the GPU-oriented example or a short CPU wiring check:

```bash
python 07_physicsnemo_heat_pinn.py
python 07_physicsnemo_heat_pinn.py --device cpu --steps 2 \
  --collocation 16 --initial 8 --boundary 8 --width 8 --depth 2 --log-every 0
```

It writes prediction, analytic solution, and absolute-error panels to
`figures/07_physicsnemo_heat.png`. PhysicsNeMo is intentionally not included in
`requirements.txt`, so the earlier tutorials remain lightweight.

---

## ND structure-preserving operator toolkit

`05_3d_equations.py` is both a normal CLI script and a `# %%` cell-based notebook.
It does **not train again**. Instead, it isolates reusable mathematical constraints
whose guarantees hold for every parameter value and even for random neural-network
weights.

```bash
python 05_3d_equations.py                         # every demo, 2D
python 05_3d_equations.py --dim 1 --demo heat
python 05_3d_equations.py --dim 2 --demo incompressible
python 05_3d_equations.py --dim 3 --grid-size 10 --steps 4
python 05_3d_equations.py --demo schrodinger --no-plot
```

The CLI visualises 1D–3D fields, while the underlying grid, derivative, FFT, and
projection helpers accept any spatial rank. Grids are uniform periodic rectangles;
irregular meshes need finite elements, graph operators, or coordinate mappings and
are intentionally outside this small tutorial.

### One toolkit, four kinds of structure

| PDE family | Structure to preserve | Construction in script 05 |
|---|---|---|
| Schrödinger / Hamiltonian waves | mass, unitary flow, reversibility | real learned phases inside `exp(i*phase)` and symmetric split stepping |
| Heat / diffusion | non-increasing L2 energy | nonnegative spectral rates inside `exp(-dt*softplus(rate))` |
| Incompressible flow | zero divergence | Fourier-space Helmholtz projection |
| Density / Fokker–Planck-like fields | positivity and unit integral | `softplus` followed by exact mass normalisation |

The Schrödinger demonstration uses the parametric nonlinear equation

```text
i*psi_t + alpha*Laplacian(psi) + beta*|psi|^2*psi - V(x)*psi = 0
```

where both scalar coefficients and the potential field can vary. Its ND plane wave

```text
psi(x,t) = A exp(i(k dot x - omega*t))
omega = alpha*|k|^2 - beta*|A|^2 + V0
```

is an exact assertion when the potential is constant. A nonconstant potential then
shows the more important point: unit-modulus split steps preserve mass even when no
analytic solution is available.

The first cells also put the derivative choices side by side:

- autograd on coordinate networks, including `spatial_jacobian` for vector fields;
- centered finite differences on periodic grids, with measured second-order convergence;
- spectral FFT derivatives, exact for represented Fourier modes.

Run all mathematical, structural, CLI, and figure smoke tests with:

```bash
python -m unittest test_3d_equations -v
```

The crucial limitation is deliberate: **preserving a structure does not guarantee an
accurate PDE solution**. A random unitary operator preserves mass perfectly and can
still predict the wrong phase. Structure narrows the hypothesis space and prevents a
known class of failures; data or physics losses must still learn the correct dynamics.

---

## The one distinction that matters

Almost all confusion in this area comes from mixing up two different jobs.

|                     | **PINN** (script 01)               | **Neural operator** (02, 03)                 |
| ------------------- | ---------------------------------- | -------------------------------------------- |
| Learns              | one function `u(x,t)`              | a map between function spaces `u0 ↦ u(·,T)`  |
| Trained on          | the equation (physics loss)        | input/output pairs (data)                    |
| Needs a solver?     | no                                 | yes — to generate training data              |
| New initial condition | retrain from scratch             | one forward pass                             |
| Analogous to        | a solver                           | a *surrogate for* a solver                   |

A PINN is a **solver**. A neural operator is a **learned replacement for a solver**,
amortising a large offline cost into near-free online evaluation. They are not
competitors; you can even combine them (physics-informed neural operators, PINO).

---

## What actually happened when I ran these

| | params | train time | error (rel. L2) | per-solve after training |
|---|---|---|---|---|
| Spectral solver (`common.py`) | — | — | reference | 140 ms |
| **01 PINN** | 12.7k | 67 s | **0.08 %** vs reference | 67 s (retrain) |
| **02 DeepONet** | 408k | 28 s | **3.5 %** mean on 200 unseen ICs | 5 µs |
| **03 FNO** | 163k | 53 s | **0.27 %** mean on 200 unseen ICs | 115 µs |

Three things worth staring at:

1. **The PINN is accurate and useless here.** 0.08% error, and it took 67 seconds to do
   what the classical solver does in 0.14 s — 500× slower, for one initial condition.
   For a plain forward problem, classical numerics wins and it isn't close. PINNs earn
   their keep on *inverse* problems (unknown `nu`, infer it from sparse noisy sensors),
   irregular geometries, and high-dimensional PDEs where meshing is hopeless.

2. **The operators pay once and then are ~100–2000× faster than the solver, forever.**
   That's the actual value proposition. It only pays off if you need many solves —
   uncertainty quantification, design optimisation, real-time control, ensemble
   forecasting.

3. **FNO beats DeepONet by 13× on the same data and budget.** That gap is real and it
   shows up across the literature on this kind of problem. See below for why.

---

## Why FNO wins

DeepONet compresses the entire input function through a dense layer into `p` numbers.
Local structure — where the shock is — has to survive that bottleneck, and largely
doesn't. Look at `figures/02_deeponet.png`: the worst cases are the ones with sharp
fronts.

FNO never compresses. Each layer is

```
u  <-  sigma( W u  +  IFFT( R · FFT(u) ) )
```

a **global convolution** implemented as a pointwise multiply in Fourier space, which is
the natural form of a Green's function for a translation-invariant problem. Three
consequences:

- **Global receptive field in one layer.** A CNN needs depth to move information across
  the domain; the FFT touches everything at once.
- **Mode truncation is free regularisation.** Keep the lowest 16 frequencies, discard
  the rest. PDE solutions are smooth-ish, so that's where the physics is.
- **Discretisation invariance.** The weights index *frequencies*, not grid points.
  Script 03 trains at N=128 and evaluates zero-shot at N=256 with the error **unchanged**
  (0.267% → 0.266%). That's the headline property, and DeepONet's fixed-size branch net
  structurally cannot do it.

This is what makes FNO a *neural operator* in the strict sense: it approximates a map
between infinite-dimensional function spaces, and the discretisation is an
implementation detail rather than part of the model.

---

## Failure modes you will hit

**PINNs**

- *ReLU anywhere in the network.* The loss contains `d2u/dx2`, and ReLU's second
  derivative is zero almost everywhere. Use tanh, sin, or gelu. This kills more
  first attempts than anything else.
- *Forgetting `create_graph=True`* in `autograd.grad`. You get no gradient from the
  residual term and the network just fits the initial condition.
- *Skipping L-BFGS.* Adam plateaus around 1e-4. In script 01, L-BFGS took the loss
  from 2.7e-3 to 1.7e-5 in a few extra seconds. Almost every tutorial omits it.
- *Loss weighting.* Equal weights work here. On stiffer problems the residual term
  swamps the boundary terms and the network converges to a smooth wrong answer. Fixes:
  gradient-norm balancing, NTK-based weights, self-adaptive weights, or hard-constraining
  the BCs into the ansatz (e.g. `u = u_bc + B(x,t)·net(x,t)` where `B` vanishes on the
  boundary).
- *Long time horizons.* PINNs degrade badly as `T` grows. Look up causal training and
  time-marching / domain decomposition (XPINN, cPINN).

**Neural operators**

- *Your test error is meaningless if the test inputs come from a different distribution
  than training.* An operator is only defined relative to a measure on the input space
  — here, the Gaussian random field in `common.py`. Change `alpha`/`tau` and watch the
  error jump. This is the single most over-claimed thing in the field.
- *Training on MSE instead of relative L2.* Relative L2 stops high-amplitude samples
  from dominating. Both scripts here use it.
- *Too many Fourier modes.* Overfits and costs parameters. Too few and the model can't
  represent sharp features — that's what the 1×1 conv path is for.
- *Data cost.* 1000 training solves is cheap in 1D and brutally expensive in 3D. Ask
  whether you can afford the dataset before you design the network.

---

## Where to go next

**Papers, in the order I'd read them**

1. Raissi, Perdikaris, Karniadakis (2019), *Physics-Informed Neural Networks* — the
   founding PINN paper; the Burgers example in script 01 is theirs (with `nu = 0.01/pi`).
2. Lu, Jin, Karniadakis (2019), *DeepONet* — the branch/trunk idea, plus the operator
   universal-approximation theorem it rests on (Chen & Chen, 1995).
3. Li et al. (2020), *Fourier Neural Operator for Parametric PDEs* — script 03.
4. Kovachki et al. (2021), *Neural Operator: Learning Maps Between Function Spaces* —
   the unifying framework; read this once the first three make sense.
5. Wang, Teng, Perdikaris (2020), *Understanding and Mitigating Gradient Flow Pathologies
   in Physics-Informed Neural Networks* — the paper that explains why your PINN won't
   converge, and where adaptive loss weighting comes from.
6. Li et al. (2021), *Physics-Informed Neural Operator (PINO)* — combines both halves
   of this repo.

**Libraries, once you've written it yourself**

- [`neuraloperator`](https://github.com/neuraloperator/neuraloperator) — the authors' own
  FNO library. Tensorised FNO, 2D/3D, the standard benchmark datasets.
- [`DeepXDE`](https://github.com/lululxvi/deepxde) — PINNs and DeepONets with a high-level
  API; multiple backends.
- [`NVIDIA PhysicsNeMo`](https://github.com/NVIDIA/physicsnemo) (formerly Modulus) — the
  industrial-scale option.

**Exercises that will teach you the most**

1. **Inverse problem.** In script 01, make `nu` a learnable `nn.Parameter`, add ~50 noisy
   observations of `u` to the loss, and recover it. This is where PINNs genuinely beat
   classical methods, and it's a ten-line change.
2. **Sharpen the shock.** Set `NU = 0.01/np.pi` in `common.py` and rerun everything. The
   PINN needs more collocation points and iterations; the FNO needs more modes. Watching
   what breaks first is instructive.
3. **Break the FNO.** Train it on GRFs with `alpha=3` and test on `alpha=2` (rougher
   inputs). Measure how far out-of-distribution generalisation actually goes.
4. **Time-dependent operator.** Learn `u0 ↦ u(·, t)` for all `t`, not just `t=1` — either
   by adding `t` as an FNO input channel or by autoregressive rollout. Then look at how
   errors compound.
5. **Go to 2D.** `SpectralConv1d` → `SpectralConv2d` is a small change: use `rfft2`, and
   keep *two* weight blocks instead of one — `[:m1, :m2]` and `[-m1:, :m2]` — because
   `rfft2` halves only the last axis, so the first axis still carries both ± frequencies.
   Darcy flow is the standard benchmark.
