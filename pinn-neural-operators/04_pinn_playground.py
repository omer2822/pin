"""Reusable physics-informed neural-network playground.

The original ``01_pinn_burgers.py`` intentionally spells out one equation.  This
script keeps the same idea but separates the neural-network machinery from the
equation, so several PDEs can share one trainer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


Tensor = torch.Tensor


def differentiate(y: Tensor, x: Tensor, order: int = 1) -> Tensor:
    """Differentiate batched scalar values once or twice with respect to ``x``.

    ``create_graph=True`` is the PINN trick: the returned derivative remains a
    differentiable expression, allowing both a second input derivative and a
    later backward pass through the PDE residual to the network parameters.
    """

    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if (
        y.ndim != 2
        or x.ndim != 2
        or y.shape[1] != 1
        or x.shape[1] != 1
        or y.shape[0] != x.shape[0]
    ):
        raise ValueError("y and x must be aligned single-column tensors")

    first = torch.autograd.grad(
        y, x, torch.ones_like(y), create_graph=True
    )[0]
    if order == 1:
        return first
    return torch.autograd.grad(
        first, x, torch.ones_like(first), create_graph=True
    )[0]


def mean_square(values: Tensor) -> Tensor:
    """Mean squared magnitude, valid for real and complex residuals."""

    return torch.mean(torch.abs(values) ** 2)


def periodic_second_derivative(field: Tensor, length: float) -> Tensor:
    """Return the FFT second derivative along the final periodic spatial axis."""

    if field.ndim < 1 or field.shape[-1] < 2:
        raise ValueError("field must have a periodic spatial axis of length at least two")
    if not math.isfinite(length) or length <= 0:
        raise ValueError("length must be positive and finite")

    count = field.shape[-1]
    spacing = length / count
    wave_numbers = 2 * math.pi * torch.fft.fftfreq(
        count, d=spacing, device=field.device, dtype=field.real.dtype
    )
    transformed = torch.fft.fft(field, dim=-1)
    return torch.fft.ifft(-(wave_numbers**2) * transformed, dim=-1)


class PDEProblem:
    """The equation-specific contract consumed by the shared PINN trainer."""

    name = "abstract"
    equation = ""
    x_min = -1.0
    x_max = 1.0
    t_final = 1.0
    input_dim = 2
    output_dim = 1
    component_names = ("u",)
    has_exact_solution = False

    def input_features(self, x: Tensor, t: Tensor) -> Tensor:
        x_scaled = 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
        t_scaled = 2 * t / self.t_final - 1
        return torch.cat((x_scaled, t_scaled), dim=1)

    def initial_target(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def sample_collocation(
        self, count: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Draw coordinate pairs for a PDE residual evaluation."""

        return _sample_points(self, count, device)

    def residual(self, model: nn.Module, x: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError

    def boundary_loss(self, model: nn.Module, t: Tensor) -> Tensor:
        raise NotImplementedError

    def exact_solution(self, x: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError(f"{self.name} has no analytic grading solution")

    def _boundary_coordinates(self, t: Tensor) -> tuple[Tensor, Tensor]:
        x_left = torch.full_like(t, self.x_min).requires_grad_(True)
        x_right = torch.full_like(t, self.x_max).requires_grad_(True)
        return x_left, x_right


class BurgersProblem(PDEProblem):
    name = "burgers"
    equation = "u_t + u*u_x - nu*u_xx = 0"

    def __init__(self, nu: float = 0.02) -> None:
        if nu <= 0:
            raise ValueError("nu must be positive")
        self.nu = nu

    def initial_target(self, x: Tensor) -> Tensor:
        return -torch.sin(math.pi * x)

    def residual(self, model: nn.Module, x: Tensor, t: Tensor) -> Tensor:
        x.requires_grad_(True)
        t.requires_grad_(True)
        u = model(x, t)
        u_t = differentiate(u, t)
        u_x = differentiate(u, x)
        u_xx = differentiate(u, x, 2)
        return u_t + u * u_x - self.nu * u_xx

    def boundary_loss(self, model: nn.Module, t: Tensor) -> Tensor:
        x_left, x_right = self._boundary_coordinates(t)
        return mean_square(model(x_left, t)) + mean_square(model(x_right, t))


class HeatProblem(PDEProblem):
    name = "heat"
    equation = "u_t - alpha*u_xx = 0"
    has_exact_solution = True

    def __init__(self, alpha: float = 0.1) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        self.wavenumber = math.pi / 2

    def initial_target(self, x: Tensor) -> Tensor:
        return torch.cos(self.wavenumber * x)

    def residual(self, model: nn.Module, x: Tensor, t: Tensor) -> Tensor:
        x.requires_grad_(True)
        t.requires_grad_(True)
        u = model(x, t)
        return differentiate(u, t) - self.alpha * differentiate(u, x, 2)

    def boundary_loss(self, model: nn.Module, t: Tensor) -> Tensor:
        x_left, x_right = self._boundary_coordinates(t)
        return mean_square(model(x_left, t)) + mean_square(model(x_right, t))

    def exact_solution(self, x: Tensor, t: Tensor) -> Tensor:
        return torch.cos(self.wavenumber * x) * torch.exp(
            -self.alpha * self.wavenumber**2 * t
        )


class ReactionDiffusionProblem(PDEProblem):
    name = "reaction-diffusion"
    equation = "u_t - D*u_xx - k*u*(1-u) = 0"

    def __init__(self, diffusion: float = 0.05, rate: float = 2.0) -> None:
        if diffusion <= 0:
            raise ValueError("diffusion must be positive")
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.diffusion = diffusion
        self.rate = rate

    def initial_target(self, x: Tensor) -> Tensor:
        return 0.25 + 0.1 * torch.cos(math.pi * x)

    def residual(self, model: nn.Module, x: Tensor, t: Tensor) -> Tensor:
        x.requires_grad_(True)
        t.requires_grad_(True)
        u = model(x, t)
        return (
            differentiate(u, t)
            - self.diffusion * differentiate(u, x, 2)
            - self.rate * u * (1 - u)
        )

    def boundary_loss(self, model: nn.Module, t: Tensor) -> Tensor:
        x_left, x_right = self._boundary_coordinates(t)
        left_gradient = differentiate(model(x_left, t), x_left)
        right_gradient = differentiate(model(x_right, t), x_right)
        return mean_square(left_gradient) + mean_square(right_gradient)


class SchrodingerProblem(PDEProblem):
    name = "schrodinger"
    equation = "i*psi_t + alpha*psi_xx + beta*|psi|^2*psi - V*psi = 0"
    x_min = -math.pi
    x_max = math.pi
    input_dim = 3
    output_dim = 2
    component_names = ("Re(psi)", "Im(psi)")
    has_exact_solution = True

    def __init__(
        self,
        wavenumber: int = 2,
        alpha: float = 0.5,
        beta: float = 0.0,
        potential: Callable[[Tensor], Tensor] | None = None,
        spectral_grid_size: int = 128,
    ) -> None:
        if wavenumber <= 0:
            raise ValueError("wavenumber must be positive")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be positive and finite")
        if not math.isfinite(beta):
            raise ValueError("beta must be finite")
        if not isinstance(spectral_grid_size, int) or spectral_grid_size < 2:
            raise ValueError("spectral_grid_size must be an integer of at least two")
        if potential is not None and not callable(potential):
            raise ValueError("potential must be a callable or None")
        self.wavenumber = wavenumber
        self.alpha = alpha
        self.beta = beta
        self.potential = potential
        self.spectral_grid_size = spectral_grid_size
        self.has_exact_solution = potential is None

    def input_features(self, x: Tensor, t: Tensor) -> Tensor:
        t_scaled = 2 * t / self.t_final - 1
        return torch.cat((torch.sin(x), torch.cos(x), t_scaled), dim=1)

    def _plane_wave(self, x: Tensor, t: Tensor) -> Tensor:
        omega = self.alpha * self.wavenumber**2 - self.beta
        phase = self.wavenumber * x - omega * t
        return torch.cat((torch.cos(phase), torch.sin(phase)), dim=1)

    def initial_target(self, x: Tensor) -> Tensor:
        return self._plane_wave(x, torch.zeros_like(x))

    def sample_collocation(
        self, count: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Sample full endpoint-free periodic grids at random collocation times."""

        if count <= 0:
            raise ValueError("count must be positive")
        time_count = math.ceil(count / self.spectral_grid_size)
        length = self.x_max - self.x_min
        grid = self.x_min + length * torch.arange(
            self.spectral_grid_size, device=device, dtype=torch.get_default_dtype()
        ) / self.spectral_grid_size
        x = grid.repeat(time_count).reshape(-1, 1)
        sampled_times = self.t_final * torch.rand(
            time_count, 1, device=device, dtype=grid.dtype
        )
        t = sampled_times.repeat_interleave(self.spectral_grid_size, dim=0)
        return x, t.requires_grad_(True)

    def _potential_values(self, x: Tensor) -> Tensor:
        if self.potential is None:
            return torch.zeros_like(x)
        values = self.potential(x)
        if not isinstance(values, Tensor) or values.shape != x.shape:
            raise ValueError("potential must return a tensor with the same shape as x")
        if values.is_complex():
            raise ValueError("potential must return real values")
        return values.to(device=x.device, dtype=x.dtype)

    def residual(self, model: nn.Module, x: Tensor, t: Tensor) -> Tensor:
        if x.ndim != 2 or t.ndim != 2 or x.shape != t.shape or x.shape[1] != 1:
            raise ValueError("spectral NLS coordinates must be aligned single-column tensors")
        if x.shape[0] % self.spectral_grid_size:
            raise ValueError("spectral NLS points must contain complete spatial grids")
        t.requires_grad_(True)
        prediction = model(x, t)
        if prediction.shape != (x.shape[0], self.output_dim):
            raise ValueError("Schrodinger models must return aligned real and imaginary columns")
        real, imaginary = prediction[:, 0:1], prediction[:, 1:2]
        psi = torch.complex(real, imaginary)
        psi_t = torch.complex(differentiate(real, t), differentiate(imaginary, t))

        time_count = x.shape[0] // self.spectral_grid_size
        psi_grid = psi.reshape(time_count, self.spectral_grid_size)
        psi_t_grid = psi_t.reshape(time_count, self.spectral_grid_size)
        potential_grid = self._potential_values(x).reshape(
            time_count, self.spectral_grid_size
        )
        psi_xx = periodic_second_derivative(psi_grid, self.x_max - self.x_min)
        residual = (
            1j * psi_t_grid
            + self.alpha * psi_xx
            + self.beta * torch.abs(psi_grid) ** 2 * psi_grid
            - potential_grid * psi_grid
        )
        return residual.reshape(-1)

    def boundary_loss(self, model: nn.Module, t: Tensor) -> Tensor:
        x_left, x_right = self._boundary_coordinates(t)
        left = model(x_left, t)
        right = model(x_right, t)
        value_loss = mean_square(left - right)

        gradient_losses = []
        for channel in range(self.output_dim):
            left_gradient = differentiate(left[:, channel : channel + 1], x_left)
            right_gradient = differentiate(
                right[:, channel : channel + 1], x_right
            )
            gradient_losses.append(mean_square(left_gradient - right_gradient))
        return value_loss + sum(gradient_losses)

    def exact_solution(self, x: Tensor, t: Tensor) -> Tensor:
        return self._plane_wave(x, t)


PROBLEMS: dict[str, type[PDEProblem]] = {
    problem.name: problem
    for problem in (
        BurgersProblem,
        HeatProblem,
        ReactionDiffusionProblem,
        SchrodingerProblem,
    )
}


def make_problem(name: str) -> PDEProblem:
    """Construct a registered equation and show all choices on bad input."""

    try:
        return PROBLEMS[name]()
    except KeyError as error:
        valid = ", ".join(PROBLEMS)
        raise ValueError(f"unknown PDE {name!r}; choose one of: {valid}") from error


class MLP(nn.Module):
    """A smooth MLP suitable for PDEs with second derivatives."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int = 64,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if min(input_dim, output_dim, width, depth) <= 0:
            raise ValueError("network dimensions and depth must be positive")

        layers: list[nn.Module] = [nn.Linear(input_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.Tanh()))
        layers.append(nn.Linear(width, output_dim))
        self.net = nn.Sequential(*layers)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, values: Tensor) -> Tensor:
        return self.net(values)


class PINN(nn.Module):
    """An MLP whose coordinate representation belongs to its PDE problem."""

    def __init__(
        self, problem: PDEProblem, width: int = 64, depth: int = 4
    ) -> None:
        super().__init__()
        self.problem = problem
        self.mlp = MLP(problem.input_dim, problem.output_dim, width, depth)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.mlp(self.problem.input_features(x, t))


@dataclass
class TrainConfig:
    """Training budget shared by every registered PDE."""

    steps: int = 2_500
    lbfgs_steps: int = 100
    collocation: int = 2_000
    initial: int = 200
    boundary: int = 200
    width: int = 64
    depth: int = 4
    learning_rate: float = 1e-3
    log_every: int = 250
    seed: int = 0

    def __post_init__(self) -> None:
        positive = (
            "steps",
            "collocation",
            "initial",
            "boundary",
            "width",
            "depth",
            "learning_rate",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("lbfgs_steps", "log_every"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass
class Evaluation:
    x: np.ndarray
    t: np.ndarray
    prediction: np.ndarray
    exact: Optional[np.ndarray]
    relative_l2: Optional[float]


def _sample_points(
    problem: PDEProblem, count: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    x = torch.rand(count, 1, device=device)
    x = problem.x_min + (problem.x_max - problem.x_min) * x
    t = problem.t_final * torch.rand(count, 1, device=device)
    return x, t


def _losses_at_samples(
    model: PINN,
    problem: PDEProblem,
    field_points: tuple[Tensor, Tensor],
    initial_x: Tensor,
    boundary_t: Tensor,
) -> dict[str, Tensor]:
    
    x_field, t_field = field_points
    residual = problem.residual(model, x_field, t_field)

    t_initial = torch.zeros_like(initial_x)
    initial_error = model(initial_x, t_initial) - problem.initial_target(initial_x)

    losses = {
        "pde": mean_square(residual),
        "initial": mean_square(initial_error),
        "boundary": problem.boundary_loss(model, boundary_t),
    }
    losses["total"] = losses["pde"] + losses["initial"] + losses["boundary"]
    if not torch.isfinite(losses["total"]):
        values = {name: value.detach().item() for name, value in losses.items()}
        raise FloatingPointError(f"non-finite PINN loss: {values}")
    return losses


def _sample_training_batch(
    problem: PDEProblem, config: TrainConfig, device: torch.device
) -> tuple[tuple[Tensor, Tensor], Tensor, Tensor]:
    field_points = problem.sample_collocation(config.collocation, device)
    initial_x, _ = _sample_points(problem, config.initial, device)
    _, boundary_t = _sample_points(problem, config.boundary, device)
    return field_points, initial_x, boundary_t


def _history_row(step: int, losses: dict[str, Tensor]) -> dict[str, float]:
    row = {"step": float(step)}
    row.update({name: value.detach().item() for name, value in losses.items()})
    return row


def _print_history(row: dict[str, float], stage: str) -> None:
    print(
        f"  {stage:5s} {int(row['step']):5d}  total={row['total']:.3e}  "
        f"pde={row['pde']:.2e} ic={row['initial']:.2e} "
        f"bc={row['boundary']:.2e}"
    )


def train(
    problem: PDEProblem, config: TrainConfig, device: torch.device
) -> tuple[PINN, list[dict[str, float]]]:
    
    """Fit one solution of ``problem`` using physics, initial, and boundary loss."""

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = PINN(problem, width=config.width, depth=config.depth).to(device)
    history: list[dict[str, float]] = []

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for step in range(1, config.steps + 1):

        optimizer.zero_grad()

        batch = _sample_training_batch(problem, config, device)
        losses = _losses_at_samples(model, problem, *batch)

        losses["total"].backward()
        optimizer.step()

        row = _history_row(step, losses)
        history.append(row)
        if config.log_every and (step == 1 or step % config.log_every == 0):
            _print_history(row, "adam")

    if config.lbfgs_steps:
        batch = _sample_training_batch(problem, config, device)
        optimizer2 = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=config.lbfgs_steps,
            history_size=min(50, config.lbfgs_steps),
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            optimizer2.zero_grad()
            closure_losses = _losses_at_samples(model, problem, *batch)
            closure_losses["total"].backward()
            return closure_losses["total"]

        optimizer2.step(closure)
        final_losses = _losses_at_samples(model, problem, *batch)
        row = _history_row(config.steps + config.lbfgs_steps, final_losses)
        history.append(row)
        if config.log_every:
            _print_history(row, "lbfgs")

    return model, history


def evaluate(
    model: PINN,
    problem: PDEProblem,
    nx: int,
    nt: int,
    device: torch.device,
) -> Evaluation:
    """Evaluate a trained solution and its optional analytic reference on a grid."""

    if nx <= 1 or nt <= 1:
        raise ValueError("nx and nt must both be greater than one")
    x = np.linspace(problem.x_min, problem.x_max, nx, dtype=np.float32)
    t = np.linspace(0.0, problem.t_final, nt, dtype=np.float32)
    xx, tt = np.meshgrid(x, t, indexing="xy")
    x_tensor = torch.from_numpy(xx.reshape(-1, 1)).to(device)
    t_tensor = torch.from_numpy(tt.reshape(-1, 1)).to(device)

    model.eval()
    with torch.no_grad():
        prediction = model(x_tensor, t_tensor).cpu().numpy()
        prediction = prediction.reshape(nt, nx, problem.output_dim)
        exact = None
        relative_l2 = None
        if problem.has_exact_solution:
            exact = problem.exact_solution(x_tensor, t_tensor).cpu().numpy()
            exact = exact.reshape(nt, nx, problem.output_dim)
            denominator = np.linalg.norm(exact)
            relative_l2 = float(np.linalg.norm(prediction - exact) / denominator)

    return Evaluation(x, t, prediction, exact, relative_l2)


def save_figure(
    evaluation: Evaluation, problem: PDEProblem, output_path: Path
) -> Path:
    """Save a compact space-time view of the learned field."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if problem.output_dim == 2:
        real = evaluation.prediction[:, :, 0]
        imaginary = evaluation.prediction[:, :, 1]
        panels = (
            (real, "Re(psi)", "RdBu_r"),
            (imaginary, "Im(psi)", "RdBu_r"),
            (real**2 + imaginary**2, "|psi|^2", "viridis"),
        )
    elif evaluation.exact is not None:
        panels = (
            (evaluation.prediction[:, :, 0], "PINN", "RdBu_r"),
            (evaluation.exact[:, :, 0], "exact", "RdBu_r"),
            (
                np.abs(evaluation.prediction[:, :, 0] - evaluation.exact[:, :, 0]),
                "absolute error",
                "magma",
            ),
        )
    else:
        panels = ((evaluation.prediction[:, :, 0], "PINN", "RdBu_r"),)

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for axis, (values, title, color_map) in zip(axes, panels):
        image = axis.imshow(
            values,
            extent=(problem.x_min, problem.x_max, 0, problem.t_final),
            origin="lower",
            aspect="auto",
            cmap=color_map,
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("t")
        fig.colorbar(image, ax=axis, pad=0.02)
    score = (
        ""
        if evaluation.relative_l2 is None
        else f" | relative L2 = {evaluation.relative_l2:.2%}"
    )
    fig.suptitle(f"{problem.name}: {problem.equation}{score}")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def available_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return torch.device(name)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pde", choices=tuple(PROBLEMS), default="burgers")
    parser.add_argument("--list", action="store_true", help="list available PDEs")
    parser.add_argument("--steps", type=int, default=2_500)
    parser.add_argument("--lbfgs-steps", type=int, default=100)
    parser.add_argument("--collocation", type=int, default=2_000)
    parser.add_argument("--initial", type=int, default=200)
    parser.add_argument("--boundary", type=int, default=200)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--nx", type=int, default=201)
    parser.add_argument("--nt", type=int, default=101)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.list:
        print("\n".join(PROBLEMS))
        return 0

    problem = make_problem(args.pde)
    config = TrainConfig(
        steps=args.steps,
        lbfgs_steps=args.lbfgs_steps,
        collocation=args.collocation,
        initial=args.initial,
        boundary=args.boundary,
        width=args.width,
        depth=args.depth,
        learning_rate=args.learning_rate,
        log_every=args.log_every,
        seed=args.seed,
    )
    device = available_device(args.device)
    print(f"PDE: {problem.name} | {problem.equation}")
    print(f"device={device} | Adam={config.steps} | L-BFGS={config.lbfgs_steps}")
    started = time.perf_counter()

    model, history = train(problem, config, device)

    elapsed = time.perf_counter() - started
    evaluation = evaluate(model, problem, args.nx, args.nt, device)

    output = args.output or Path("figures") / f"04_pinn_{problem.name}.png"
    saved = save_figure(evaluation, problem, output)
    print(f"trained in {elapsed:.1f}s | final loss={history[-1]['total']:.3e}")
    if evaluation.relative_l2 is not None:
        print(f"relative L2 vs exact solution: {evaluation.relative_l2:.3%}")
    print(f"saved {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
