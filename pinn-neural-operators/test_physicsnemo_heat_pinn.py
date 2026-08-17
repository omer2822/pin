"""Contracts for the optional PhysicsNeMo v2 heat-PINN playground."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest


HERE = Path(__file__).resolve().parent
SCRIPT_PATH = HERE / "07_physicsnemo_heat_pinn.py"


def load_playground():
    spec = importlib.util.spec_from_file_location("physicsnemo_heat", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pg = load_playground()


class PhysicsNeMoHeatCliTests(unittest.TestCase):
    def test_cli_defaults_to_a_gpu_convergence_budget(self):
        args = pg.parse_args([])

        self.assertEqual(args.steps, 10_000)
        self.assertEqual(args.collocation, 4_096)
        self.assertEqual(args.initial, 512)
        self.assertEqual(args.boundary, 512)
        self.assertEqual(args.width, 128)
        self.assertEqual(args.depth, 6)
        self.assertEqual(args.device, "auto")

    def test_help_works_without_importing_optional_runtime_dependencies(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--collocation", result.stdout)
        self.assertIn("--device", result.stdout)


@unittest.skipUnless(
    importlib.util.find_spec("physicsnemo") is not None,
    "PhysicsNeMo is optional; install nvidia-physicsnemo[sym] to run integration tests",
)
class PhysicsNeMoHeatIntegrationTests(unittest.TestCase):
    def test_exact_solution_satisfies_initial_and_dirichlet_conditions(self):
        import torch

        x = torch.tensor([[-1.0], [0.0], [1.0]])
        t = torch.zeros_like(x)
        actual = pg.exact_solution(x, t, alpha=0.1)

        self.assertTrue(torch.allclose(actual[:, 0], torch.tensor([0.0, 1.0, 0.0])))

    def test_tiny_training_produces_finite_evaluation_and_figure(self):
        import tempfile

        config = pg.TrainConfig(
            steps=2,
            collocation=16,
            initial=8,
            boundary=8,
            width=8,
            depth=2,
            log_every=0,
        )
        model, informer, device, _ = pg.train(config)
        evaluation = pg.evaluate(model, informer, config.alpha, 11, 7, device)

        self.assertEqual(evaluation.prediction.shape, (7, 11))
        self.assertTrue(evaluation.is_finite)
        with tempfile.TemporaryDirectory() as directory:
            output = pg.save_figure(
                evaluation, config.alpha, Path(directory) / "heat.png"
            )
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
