# Reusable PINN Playground Design

## Goal

Add a readable, standalone PINN playground beside the existing Burgers tutorial.
It must demonstrate how the same automatic-differentiation and training loop can
solve several PDEs without obscuring `01_pinn_burgers.py`, whose value is its
single-purpose explanation.

## User interface

The new `04_pinn_playground.py` script accepts a PDE name on the command line:

```bash
python 04_pinn_playground.py --pde burgers
python 04_pinn_playground.py --pde heat
python 04_pinn_playground.py --pde reaction-diffusion
python 04_pinn_playground.py --pde schrodinger
```

Training-budget flags allow short experiments and automated smoke tests without
changing source constants. Each run prints component losses, evaluates the model
on a grid, and saves a PDE-specific figure under `figures/`.

## Architecture

The script has three intentionally small layers:

1. A smooth MLP and reusable automatic-differentiation helper calculate arbitrary
   output components and their first or second derivatives.
2. A `PDEProblem` interface owns the domain, output width, input transform,
   residual, initial values, boundary loss, and optional exact solution.
3. A shared trainer samples collocation/initial/boundary points and optimizes the
   sum of PDE, initial-condition, and boundary-condition losses.

Concrete problem definitions are registered by CLI name. Burgers, heat, and
reaction-diffusion use one real output. Schrödinger uses two outputs for the real
and imaginary components of the wave function.

## Equations and constraints

- Burgers: `u_t + u*u_x - nu*u_xx = 0`, `u(x,0) = -sin(pi*x)`, homogeneous
  Dirichlet boundaries on `[-1, 1]`.
- Heat: `u_t - alpha*u_xx = 0`, sinusoidal initial state and homogeneous
  Dirichlet boundaries on `[-1, 1]`; it has an analytic solution for grading.
- Reaction-diffusion: `u_t - D*u_xx - k*u*(1-u) = 0`, a smooth bounded initial
  state and no-flux (Neumann) boundaries on `[-1, 1]`.
- Schrödinger: `i*psi_t = -(1/2)*psi_xx` on `[-pi, pi]`, represented as coupled
  real residuals. A plane wave supplies the initial condition and exact solution.
  Inputs use `(sin(x), cos(x), t)` so predictions are periodic by construction;
  the boundary loss also matches spatial derivatives at both ends to teach the
  periodic-boundary condition explicitly.

## Error handling

CLI choices reject unknown PDE names. Numeric training arguments must be positive.
Derivative requests validate tensor shapes and orders. Non-finite losses stop with
an actionable error instead of silently producing a corrupt model.

## Tests and assertions

Tests import the script without launching training and exercise real PyTorch code.
They assert:

- first and second automatic derivatives of a known polynomial;
- each PDE residual on a manufactured or analytic solution;
- Schrödinger's real/imaginary decomposition and periodic input transform;
- problem registry and invalid-argument behavior;
- a tiny end-to-end optimization/evaluation smoke run remains finite and produces
  outputs with the expected shape.

The full training defaults remain educational rather than benchmark-oriented;
tests use small explicit budgets and do not assert convergence to a fragile error
threshold.

## Scope

This change adds the playground, focused tests, and README usage notes. It does not
refactor the original three tutorials, add a GUI, implement inverse coefficient
discovery, or promise one set of hyperparameters is optimal for every PDE.
