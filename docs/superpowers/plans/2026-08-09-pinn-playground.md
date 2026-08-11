# Reusable PINN Playground Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested CLI playground that trains the same PINN core on Burgers, heat, reaction-diffusion, and complex Schrödinger equations.

**Architecture:** Keep the existing focused tutorials unchanged. Put a smooth MLP, nested-autograd helper, PDE problem definitions, shared trainer, evaluator, plotter, and CLI in one readable tutorial script; use a separate standard-library `unittest` module for mathematical assertions and a tiny end-to-end smoke run.

**Tech Stack:** Python 3.9+, PyTorch 2.x, NumPy, Matplotlib, `argparse`, and `unittest`.

## Global Constraints

- Preserve `01_pinn_burgers.py`, `02_deeponet.py`, and `03_fno1d.py` unchanged.
- Use smooth `tanh` networks because every equation contains a second spatial derivative.
- Represent Schrödinger's complex wave function with two real output channels.
- Keep full training educational; automated tests use tiny budgets and do not assert a fragile convergence threshold.
- This workspace has no Git metadata, so the commit steps normally required by the planning workflow are replaced by explicit file/status checkpoints.

---

### Task 1: Autodiff core and model

**Files:**
- Create: `pinn-neural-operators/04_pinn_playground.py`
- Create: `pinn-neural-operators/test_pinn_playground.py`

**Interfaces:**
- Produces: `differentiate(y: Tensor, x: Tensor, order: int = 1) -> Tensor`
- Produces: `MLP(input_dim: int, output_dim: int, width: int = 64, depth: int = 4)`
- Produces: `PINN(problem: PDEProblem, width: int = 64, depth: int = 4)` whose `forward(x, t)` applies the problem's input transform.

- [ ] **Step 1: Write failing derivative/model tests**

```python
def test_differentiate_first_and_second_order(self):
    x = torch.linspace(-1, 1, 9).reshape(-1, 1).requires_grad_(True)
    y = x**3 + 2*x
    self.assertTrue(torch.allclose(pg.differentiate(y, x), 3*x**2 + 2))
    self.assertTrue(torch.allclose(pg.differentiate(y, x, 2), 6*x))

def test_differentiate_rejects_unsupported_order(self):
    x = torch.ones(1, 1, requires_grad=True)
    with self.assertRaisesRegex(ValueError, "order must be 1 or 2"):
        pg.differentiate(x**2, x, 3)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest test_pinn_playground.PlaygroundTests.test_differentiate_first_and_second_order -v`

Expected: import/file failure because `04_pinn_playground.py` does not exist.

- [ ] **Step 3: Implement minimal differentiable core**

Implement `differentiate` with `torch.autograd.grad(..., create_graph=True)`, an Xavier-initialized tanh MLP, and `PINN.forward` delegating coordinate features to `problem.input_features(x, t)`.

```python
def differentiate(y, x, order=1):
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    first = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
    if order == 1:
        return first
    return torch.autograd.grad(
        first, x, torch.ones_like(first), create_graph=True
    )[0]
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `python -m unittest test_pinn_playground.PlaygroundTests.test_differentiate_first_and_second_order test_pinn_playground.PlaygroundTests.test_differentiate_rejects_unsupported_order -v`

Expected: two passing tests.

- [ ] **Step 5: Record checkpoint**

Run: `ls -l 04_pinn_playground.py test_pinn_playground.py`

Expected: both files exist and no existing tutorial file was modified.

---

### Task 2: Four PDE definitions and mathematical assertions

**Files:**
- Modify: `pinn-neural-operators/04_pinn_playground.py`
- Modify: `pinn-neural-operators/test_pinn_playground.py`

**Interfaces:**
- Consumes: `differentiate`, `PINN`
- Produces: `PDEProblem` base interface with `input_features`, `initial_target`, `residual`, `boundary_loss`, and optional `exact_solution`.
- Produces: `BurgersProblem`, `HeatProblem`, `ReactionDiffusionProblem`, `SchrodingerProblem`
- Produces: `PROBLEMS: dict[str, type[PDEProblem]]` and `make_problem(name: str) -> PDEProblem`

- [ ] **Step 1: Write failing equation tests**

Add small analytic `nn.Module` fixtures and assertions:

```python
def test_heat_exact_solution_has_zero_residual(self):
    problem = pg.HeatProblem(alpha=0.1)
    model = HeatExact(alpha=0.1)
    x, t = sample_coordinates()
    self.assertLess(problem.residual(model, x, t).abs().max().item(), 1e-5)

def test_reaction_logistic_solution_has_zero_residual(self):
    problem = pg.ReactionDiffusionProblem(diffusion=0.05, rate=2.0)
    model = LogisticExact(rate=2.0)
    x, t = sample_coordinates()
    self.assertLess(problem.residual(model, x, t).abs().max().item(), 1e-5)

def test_schrodinger_plane_wave_has_zero_two_channel_residual(self):
    problem = pg.SchrodingerProblem(wavenumber=2)
    x, t = sample_coordinates(-math.pi, math.pi)
    residual = problem.residual(SchrodingerExact(2), x, t)
    self.assertEqual(residual.shape[1], 2)
    self.assertLess(residual.abs().max().item(), 1e-5)
```

Also assert Burgers' residual for `u=x**2+t` equals `1 + 2*x*(x**2+t) - 2*nu`, Schrödinger features agree at `-pi` and `pi`, and unknown registry names raise `ValueError` listing valid names.

- [ ] **Step 2: Run equation tests and verify RED**

Run: `python -m unittest test_pinn_playground.PlaygroundTests.test_heat_exact_solution_has_zero_residual test_pinn_playground.PlaygroundTests.test_reaction_logistic_solution_has_zero_residual test_pinn_playground.PlaygroundTests.test_schrodinger_plane_wave_has_zero_two_channel_residual -v`

Expected: failures because the problem classes do not exist.

- [ ] **Step 3: Implement the four problems**

Use these residuals:

```python
# Burgers
r = u_t + u*u_x - nu*u_xx

# Heat
r = u_t - alpha*u_xx

# Fisher/KPP reaction-diffusion
r = u_t - diffusion*u_xx - rate*u*(1-u)

# i*psi_t + 0.5*psi_xx = 0, psi = real + i*imag
r_real = -imag_t + 0.5*real_xx
r_imag = real_t + 0.5*imag_xx
```

Use Dirichlet losses for Burgers/heat, derivative-zero Neumann loss for reaction-diffusion, and value-plus-derivative periodic matching for Schrödinger. Use `(sin(x), cos(x), scaled_t)` as Schrödinger input features.

- [ ] **Step 4: Run all mathematical tests and verify GREEN**

Run: `python -m unittest test_pinn_playground.PlaygroundTests -v`

Expected: every core and equation test passes.

- [ ] **Step 5: Record checkpoint**

Run: `python 04_pinn_playground.py --list`

Expected output lists exactly `burgers`, `heat`, `reaction-diffusion`, and `schrodinger`.

---

### Task 3: Shared trainer, evaluator, plotting, and CLI

**Files:**
- Modify: `pinn-neural-operators/04_pinn_playground.py`
- Modify: `pinn-neural-operators/test_pinn_playground.py`

**Interfaces:**
- Consumes: the problem registry and `PINN`
- Produces: `TrainConfig`, `train(problem, config, device) -> tuple[PINN, list[dict[str, float]]]`
- Produces: `evaluate(model, problem, nx, nt, device) -> Evaluation`
- Produces: `save_figure(evaluation, problem, output_path) -> Path`
- Produces: `parse_args(argv=None) -> argparse.Namespace` and `main(argv=None) -> int`

- [ ] **Step 1: Write failing validation and smoke tests**

```python
def test_train_config_rejects_nonpositive_counts(self):
    with self.assertRaisesRegex(ValueError, "collocation must be positive"):
        pg.TrainConfig(collocation=0)

def test_tiny_training_and_evaluation_are_finite(self):
    config = pg.TrainConfig(steps=2, lbfgs_steps=0, collocation=16,
                            initial=8, boundary=8, width=8, depth=2,
                            log_every=0, seed=7)
    model, history = pg.train(pg.HeatProblem(), config, torch.device("cpu"))
    evaluation = pg.evaluate(model, pg.HeatProblem(), 11, 7, torch.device("cpu"))
    self.assertEqual(evaluation.prediction.shape, (7, 11, 1))
    self.assertTrue(np.isfinite(evaluation.prediction).all())
    self.assertTrue(all(math.isfinite(value) for value in history[-1].values()))
```

- [ ] **Step 2: Run smoke tests and verify RED**

Run: `python -m unittest test_pinn_playground.PlaygroundTests.test_train_config_rejects_nonpositive_counts test_pinn_playground.PlaygroundTests.test_tiny_training_and_evaluation_are_finite -v`

Expected: failures because `TrainConfig`, `train`, and `evaluate` do not exist.

- [ ] **Step 3: Implement training and CLI**

Validate every numeric budget in `TrainConfig.__post_init__`; permit only `lbfgs_steps` and `log_every` to be zero. In each Adam iteration, resample all three point sets, compute named PDE/IC/BC losses, reject non-finite totals with `FloatingPointError`, and append scalar histories. Run optional full-batch L-BFGS after Adam. Evaluate without input gradients on a mesh, and grade against an exact solution when available.

Add CLI flags for `--pde`, `--steps`, `--lbfgs-steps`, `--collocation`, `--initial`, `--boundary`, `--width`, `--depth`, `--seed`, `--device`, `--output`, and `--list`.

- [ ] **Step 4: Run the entire test module and verify GREEN**

Run: `python -m unittest test_pinn_playground -v`

Expected: all tests pass with no warnings or tracebacks.

- [ ] **Step 5: Smoke each CLI problem**

Run each with `--steps 1 --lbfgs-steps 0 --collocation 8 --initial 8 --boundary 8 --width 8 --depth 2 --nx 11 --nt 7` and a temporary output path.

Expected: exit code zero, finite printed losses, and four non-empty PNG files.

---

### Task 4: Documentation and final verification

**Files:**
- Modify: `pinn-neural-operators/README.md`
- Modify: `pinn-neural-operators/requirements.txt`

**Interfaces:**
- Consumes: CLI implemented in Task 3.
- Produces: copy-paste usage and concise explanation of the coupled real Schrödinger residual.

- [ ] **Step 1: Write a failing documentation assertion**

Add a test that reads `README.md` and asserts it contains `04_pinn_playground.py`, all four CLI choices, and `i*psi_t + 0.5*psi_xx = 0`.

- [ ] **Step 2: Run documentation test and verify RED**

Run: `python -m unittest test_pinn_playground.PlaygroundTests.test_readme_documents_every_playground_equation -v`

Expected: assertion failure because the README lacks the playground section.

- [ ] **Step 3: Document the playground**

Add a README section showing `--list`, one normal command per PDE, a tiny smoke command, output locations, complex-to-two-real-channel explanation, and a note that one default budget is not equally optimal for every PDE. Keep runtime dependencies at the existing three packages; do not add pytest.

- [ ] **Step 4: Run complete verification**

Run:

```bash
python -m unittest test_pinn_playground -v
python -m py_compile 04_pinn_playground.py test_pinn_playground.py
python 04_pinn_playground.py --list
```

Expected: all tests pass, compilation exits zero, and all four names are listed.

- [ ] **Step 5: Inspect final scope**

Run: `find . -maxdepth 2 -type f -newer docs/superpowers/specs/2026-08-09-pinn-playground-design.md -print | sort`

Expected project changes are limited to the new playground/test files, README, and the design/plan documents, plus generated smoke-test figures in a temporary directory.
