"""Transient 1D heat-equation PINN using the current PhysicsNeMo v2 workflow.

This is deliberately separate from ``04_pinn_playground.py``.  That script is
the from-scratch PyTorch explanation; this one demonstrates the maintained
PhysicsNeMo v2 style: an explicit PyTorch loop, a symbolic ``PDE``, and a
``PhysicsInformer`` for spatial derivatives.

The equation is ``u_t - alpha * u_xx = 0`` on ``x in [-1, 1]`` and
``t in [0, 1]``, with ``u(x, 0) = cos(pi*x/2)`` and zero endpoint values.
PhysicsInformer computes ``u_xx``.  Time is not a spatial PhysicsInformer
axis, so ``u_t`` remains an explicit PyTorch-autograd derivative.

Install the optional dependency in this repository's virtual environment:

    source pinn-neural-operators/venv./bin/activate
    python -m pip install --upgrade "nvidia-physicsnemo[sym]"

The defaults target convergence on an NVIDIA GPU.  For a quick wiring check:

    python 07_physicsnemo_heat_pinn.py --device cpu --steps 2 \\
        --collocation 16 --initial 8 --boundary 8 --width 8 --depth 2 \\
        --log-every 0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Optional, Sequence


INSTALL_HINT = '''PhysicsNeMo is required to train this playground.

Install it in the project virtual environment:
  source pinn-neural-operators/venv./bin/activate
  python -m pip install --upgrade "nvidia-physicsnemo[sym]"'''


class DependencyError(RuntimeError):
    """Raised when the optional PhysicsNeMo runtime is unavailable."""


@dataclass(frozen=True)
class Runtime:
    torch: Any
    fully_connected: type
    pde: type
    physics_informer: type


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 10_000
    collocation: int = 4_096
    initial: int = 512
    boundary: int = 512
    width: int = 128
    depth: int = 6
    alpha: float = 0.1
    learning_rate: float = 1e-3
    log_every: int = 500
    seed: int = 0
    device_name: str = "auto"

    def __post_init__(self) -> None:
        for name in ("steps", "collocation", "initial", "boundary", "width", "depth"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.alpha <= 0 or not math.isfinite(self.alpha):
            raise ValueError("alpha must be positive and finite")
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be positive and finite")
        if self.log_every < 0:
            raise ValueError("log_every must be non-negative")


@dataclass(frozen=True)
class Evaluation:
    x: Any
    t: Any
    prediction: Any
    exact: Any
    relative_l2: float

    @property
    def is_finite(self) -> bool:
        import numpy as np

        return bool(np.isfinite(self.prediction).all() and np.isfinite(self.exact).all())


def load_runtime() -> Runtime:
    """Load optional imports only when the user starts a training run."""

    try:
        import torch
        from physicsnemo.models.mlp import FullyConnected
        from physicsnemo.sym.eq.pde import PDE
        from physicsnemo.sym.eq.phy_informer import PhysicsInformer
    except ModuleNotFoundError as error:
        raise DependencyError(INSTALL_HINT) from error
    return Runtime(torch, FullyConnected, PDE, PhysicsInformer)


def exact_solution(x: Any, t: Any, alpha: float) -> Any:
    """Analytic heat solution used only for validation and visualization."""

    import torch

    wavenumber = math.pi / 2
    return torch.cos(wavenumber * x) * torch.exp(-alpha * wavenumber**2 * t)


def available_device(torch: Any, name: str) -> Any:
    """Choose CUDA first, then Apple MPS, then CPU when ``name`` is auto."""

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return torch.device(name)


def make_spatial_informer(runtime: Runtime, device: Any) -> Any:
    """Create the symbolic spatial operator that PhysicsInformer evaluates."""

    from sympy import Function, Symbol

    class SpatialSecondDerivative(runtime.pde):
        def __init__(self) -> None:
            self.dim = 1
            x = Symbol("x")
            u = Function("u")(x)
            self.equations = {"u_xx": u.diff(x, 2)}

    return runtime.physics_informer(
        required_outputs=["u_xx"],
        equations=SpatialSecondDerivative(),
        grad_method="autodiff",
        device=device,
    )


def residual(model: Any, informer: Any, x: Any, t: Any, alpha: float, torch: Any) -> Any:
    """Compose explicit ``u_t`` with PhysicsNeMo's symbolic ``u_xx``."""

    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    u = model(torch.cat((x, t), dim=1))
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_xx = informer.forward({"coordinates": x, "u": u})["u_xx"]
    return u_t - alpha * u_xx


def _sample_interior(config: TrainConfig, device: Any, torch: Any) -> tuple[Any, Any]:
    x = -1.0 + 2.0 * torch.rand(config.collocation, 1, device=device)
    t = torch.rand(config.collocation, 1, device=device)
    return x, t


def _losses(model: Any, informer: Any, config: TrainConfig, device: Any, torch: Any) -> dict[str, Any]:
    x, t = _sample_interior(config, device, torch)
    physics = torch.mean(residual(model, informer, x, t, config.alpha, torch) ** 2)

    initial_x = -1.0 + 2.0 * torch.rand(config.initial, 1, device=device)
    initial = torch.mean((model(torch.cat((initial_x, torch.zeros_like(initial_x)), dim=1)) - exact_solution(initial_x, torch.zeros_like(initial_x), config.alpha)) ** 2)

    boundary_t = torch.rand(config.boundary, 1, device=device)
    left = torch.full_like(boundary_t, -1.0)
    right = torch.full_like(boundary_t, 1.0)
    boundary = torch.mean(model(torch.cat((left, boundary_t), dim=1)) ** 2)
    boundary = boundary + torch.mean(model(torch.cat((right, boundary_t), dim=1)) ** 2)

    total = physics + initial + boundary
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite PINN loss")
    return {"total": total, "pde": physics, "initial": initial, "boundary": boundary}


def train(config: TrainConfig) -> tuple[Any, Any, Any, Runtime]:
    """Train the PhysicsNeMo-backed MLP with physics, initial, and boundary loss."""

    runtime = load_runtime()
    torch = runtime.torch
    torch.manual_seed(config.seed)
    device = available_device(torch, config.device_name)
    model = runtime.fully_connected(
        in_features=2,
        out_features=1,
        layer_size=config.width,
        num_layers=config.depth,
        activation_fn="silu",
    ).to(device)
    informer = make_spatial_informer(runtime, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for step in range(1, config.steps + 1):
        optimizer.zero_grad()
        losses = _losses(model, informer, config, device, torch)
        losses["total"].backward()
        optimizer.step()
        if config.log_every and (step == 1 or step % config.log_every == 0):
            print(
                f"step={step:5d} total={losses['total'].item():.3e} "
                f"pde={losses['pde'].item():.3e} ic={losses['initial'].item():.3e} "
                f"bc={losses['boundary'].item():.3e}"
            )
    return model, informer, device, runtime


def evaluate(model: Any, informer: Any, alpha: float, nx: int, nt: int, device: Any) -> Evaluation:
    """Evaluate the learned field and analytic solution on a regular grid."""

    del informer  # The trained model needs no residual graph for inference.
    if nx <= 1 or nt <= 1:
        raise ValueError("nx and nt must both be greater than one")
    import numpy as np
    import torch

    x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
    t = np.linspace(0.0, 1.0, nt, dtype=np.float32)
    xx, tt = np.meshgrid(x, t, indexing="xy")
    x_tensor = torch.from_numpy(xx.reshape(-1, 1)).to(device)
    t_tensor = torch.from_numpy(tt.reshape(-1, 1)).to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(torch.cat((x_tensor, t_tensor), dim=1)).cpu().numpy().reshape(nt, nx)
        exact = exact_solution(x_tensor, t_tensor, alpha).cpu().numpy().reshape(nt, nx)
    relative_l2 = float(np.linalg.norm(prediction - exact) / np.linalg.norm(exact))
    return Evaluation(x, t, prediction, exact, relative_l2)


def save_figure(evaluation: Evaluation, alpha: float, output_path: Path) -> Path:
    """Save prediction, exact solution, and absolute error panels."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = (
        (evaluation.prediction, "PhysicsNeMo PINN", "RdBu_r"),
        (evaluation.exact, "exact", "RdBu_r"),
        (abs(evaluation.prediction - evaluation.exact), "absolute error", "magma"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, (values, title, color_map) in zip(axes, panels):
        image = axis.imshow(values, extent=(-1, 1, 0, 1), origin="lower", aspect="auto", cmap=color_map)
        axis.set(title=title, xlabel="x", ylabel="t")
        fig.colorbar(image, ax=axis, pad=0.02)
    fig.suptitle(f"1D heat equation: u_t - {alpha:g} u_xx = 0 | relative L2 = {evaluation.relative_l2:.2%}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--collocation", type=int, default=4_096)
    parser.add_argument("--initial", type=int, default=512)
    parser.add_argument("--boundary", type=int, default=512)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--nx", type=int, default=201, help="evaluation points in space")
    parser.add_argument("--nt", type=int, default=101, help="evaluation points in time")
    parser.add_argument("--output", type=Path, default=Path("figures/07_physicsnemo_heat.png"))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        config = TrainConfig(
            steps=args.steps, collocation=args.collocation, initial=args.initial,
            boundary=args.boundary, width=args.width, depth=args.depth, alpha=args.alpha,
            learning_rate=args.learning_rate, log_every=args.log_every, seed=args.seed,
            device_name=args.device,
        )
        model, informer, device, _ = train(config)
    except DependencyError as error:
        print(error)
        return 2
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error

    started = time.perf_counter()
    evaluation = evaluate(model, informer, config.alpha, args.nx, args.nt, device)
    output = save_figure(evaluation, config.alpha, args.output)
    elapsed = time.perf_counter() - started
    print(f"device={device} | relative L2={evaluation.relative_l2:.3%} | evaluation+plot={elapsed:.1f}s")
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
