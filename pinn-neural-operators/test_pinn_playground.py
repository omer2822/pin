"""Mathematical and smoke tests for the reusable PINN playground."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "04_pinn_playground.py"


def load_playground():
    spec = importlib.util.spec_from_file_location("pinn_playground", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pg = load_playground()


def sample_coordinates(
    x_min: float = -1.0, x_max: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(x_min + 0.1, x_max - 0.1, 11).reshape(-1, 1)
    t = torch.linspace(0.05, 0.95, 11).reshape(-1, 1)
    return x.requires_grad_(True), t.requires_grad_(True)


class BurgersPolynomial(nn.Module):
    def forward(self, x, t):
        return x**2 + t


class HeatExact(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, t):
        wavenumber = math.pi / 2
        return torch.cos(wavenumber * x) * torch.exp(
            -self.alpha * wavenumber**2 * t
        )


class LogisticExact(nn.Module):
    def __init__(self, rate: float, initial: float = 0.25):
        super().__init__()
        self.rate = rate
        self.ratio = (1 - initial) / initial

    def forward(self, x, t):
        # The quadratic zero keeps the exact spatial second derivative in the
        # autograd graph while leaving this manufactured solution x-independent.
        return 1 / (1 + self.ratio * torch.exp(-self.rate * t)) + 0 * x**2


class SchrodingerExact(nn.Module):
    def __init__(self, wavenumber: int):
        super().__init__()
        self.wavenumber = wavenumber

    def forward(self, x, t):
        phase = (
            self.wavenumber * x
            - 0.5 * self.wavenumber**2 * t
        )
        return torch.cat((torch.cos(phase), torch.sin(phase)), dim=1)


class FullNLSPlaneWave(nn.Module):
    def __init__(
        self,
        wave_number: int,
        alpha: float,
        beta: float,
        potential_value: float,
    ) -> None:
        super().__init__()
        self.wave_number = wave_number
        self.omega = alpha * wave_number**2 - beta + potential_value

    def forward(self, x, t):
        phase = self.wave_number * x - self.omega * t
        return torch.cat((torch.cos(phase), torch.sin(phase)), dim=1)


class PlaygroundTests(unittest.TestCase):
    def test_periodic_second_derivative_is_exact_for_represented_fourier_mode(self):
        grid_size, wave_number = 64, 7
        x = -math.pi + 2 * math.pi * torch.arange(grid_size, dtype=torch.float64) / grid_size
        psi = torch.exp(1j * wave_number * x)

        actual = pg.periodic_second_derivative(psi, 2 * math.pi)

        self.assertLess(
            (actual + wave_number**2 * psi).abs().max().item(),
            1e-5,
        )

    def test_differentiate_first_and_second_order(self):
        x = torch.linspace(-1, 1, 9).reshape(-1, 1).requires_grad_(True)
        y = x**3 + 2 * x

        self.assertTrue(
            torch.allclose(pg.differentiate(y, x), 3 * x**2 + 2)
        )
        self.assertTrue(torch.allclose(pg.differentiate(y, x, 2), 6 * x))

    def test_differentiate_rejects_unsupported_order(self):
        x = torch.ones(1, 1, requires_grad=True)

        with self.assertRaisesRegex(ValueError, "order must be 1 or 2"):
            pg.differentiate(x**2, x, 3)

    def test_differentiate_rejects_non_column_shapes(self):
        x = torch.ones(3, 1, requires_grad=True)

        with self.assertRaisesRegex(ValueError, "single-column tensors"):
            pg.differentiate(torch.cat((x, x), dim=1), x)

    def test_burgers_polynomial_matches_hand_calculated_residual(self):
        problem = pg.BurgersProblem(nu=0.07)
        x, t = sample_coordinates()

        actual = problem.residual(BurgersPolynomial(), x, t)
        expected = 1 + 2 * x * (x**2 + t) - 2 * problem.nu

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_heat_exact_solution_has_zero_residual(self):
        problem = pg.HeatProblem(alpha=0.1)
        x, t = sample_coordinates()

        residual = problem.residual(HeatExact(alpha=0.1), x, t)

        self.assertLess(residual.abs().max().item(), 1e-5)

    def test_reaction_logistic_solution_has_zero_residual(self):
        problem = pg.ReactionDiffusionProblem(diffusion=0.05, rate=2.0)
        x, t = sample_coordinates()

        residual = problem.residual(LogisticExact(rate=2.0), x, t)

        self.assertLess(residual.abs().max().item(), 1e-5)

    def test_schrodinger_plane_wave_has_zero_spectral_residual(self):
        problem = pg.SchrodingerProblem(wavenumber=2)
        x = (-math.pi + 2 * math.pi * torch.arange(128, dtype=torch.float64) / 128).reshape(-1, 1)
        t = torch.full_like(x, 0.25, requires_grad=True)

        residual = problem.residual(SchrodingerExact(2), x, t)

        self.assertTrue(residual.is_complex())
        self.assertEqual(residual.shape, (x.shape[0],))
        self.assertLess(residual.abs().max().item(), 1e-5)

    def test_spectral_nls_plane_wave_has_zero_residual(self):
        alpha, beta, potential_value = 0.7, 0.35, 0.2
        problem = pg.SchrodingerProblem(
            wavenumber=3,
            alpha=alpha,
            beta=beta,
            potential=lambda x: torch.full_like(x, potential_value),
            spectral_grid_size=64,
        )
        x = (-math.pi + 2 * math.pi * torch.arange(64, dtype=torch.float64) / 64).reshape(-1, 1)
        t = torch.full_like(x, 0.25, requires_grad=True)

        residual = problem.residual(
            FullNLSPlaneWave(3, alpha, beta, potential_value), x, t
        )

        self.assertTrue(residual.is_complex())
        self.assertLess(residual.abs().max().item(), 1e-5)

    def test_schrodinger_collocation_uses_endpoint_free_periodic_grids(self):
        problem = pg.SchrodingerProblem(spectral_grid_size=8)

        x, t = problem.sample_collocation(15, torch.device("cpu"))

        expected = -math.pi + 2 * math.pi * torch.arange(8) / 8
        self.assertEqual(x.shape, (16, 1))
        self.assertEqual(t.shape, (16, 1))
        self.assertTrue(torch.allclose(x.reshape(-1, 8)[0], expected))
        self.assertTrue(torch.allclose(x.reshape(-1, 8)[1], expected))
        self.assertTrue(torch.allclose(t.reshape(-1, 8)[:, 0:1], t.reshape(-1, 8)))

    def test_schrodinger_rejects_potential_callable_with_wrong_shape(self):
        problem = pg.SchrodingerProblem(
            potential=lambda x: torch.zeros(x.shape[0]), spectral_grid_size=8
        )
        x, t = problem.sample_collocation(8, torch.device("cpu"))

        with self.assertRaisesRegex(ValueError, "potential"):
            problem.residual(SchrodingerExact(2), x, t)

    def test_schrodinger_features_are_periodic(self):
        problem = pg.SchrodingerProblem()
        x = torch.tensor([[-math.pi], [math.pi]])
        t = torch.full_like(x, 0.4)

        features = problem.input_features(x, t)

        self.assertTrue(torch.allclose(features[0], features[1], atol=1e-6))

    def test_problem_registry_rejects_unknown_name(self):
        with self.assertRaisesRegex(
            ValueError,
            "burgers.*heat.*reaction-diffusion.*schrodinger",
        ):
            pg.make_problem("maxwell")

    def test_train_config_rejects_nonpositive_counts(self):
        with self.assertRaisesRegex(ValueError, "collocation must be positive"):
            pg.TrainConfig(collocation=0)

    def test_tiny_training_and_evaluation_are_finite(self):
        problem = pg.HeatProblem()
        config = pg.TrainConfig(
            steps=2,
            lbfgs_steps=0,
            collocation=16,
            initial=8,
            boundary=8,
            width=8,
            depth=2,
            log_every=0,
            seed=7,
        )

        model, history = pg.train(problem, config, torch.device("cpu"))
        evaluation = pg.evaluate(
            model, problem, nx=11, nt=7, device=torch.device("cpu")
        )

        self.assertEqual(evaluation.prediction.shape, (7, 11, 1))
        self.assertEqual(evaluation.exact.shape, (7, 11, 1))
        self.assertTrue(np.isfinite(evaluation.prediction).all())
        self.assertTrue(
            all(math.isfinite(value) for value in history[-1].values())
        )

    def test_readme_documents_every_playground_equation(self):
        readme = (HERE / "README.md").read_text(encoding="utf-8")

        for text in (
            "04_pinn_playground.py",
            "--pde burgers",
            "--pde heat",
            "--pde reaction-diffusion",
            "--pde schrodinger",
            "i*psi_t + alpha*psi_xx + beta*|psi|^2*psi - V*psi = 0",
            "FFT",
            "SchrodingerProblem(",
        ):
            with self.subTest(text=text):
                self.assertIn(text, readme)


if __name__ == "__main__":
    unittest.main()
