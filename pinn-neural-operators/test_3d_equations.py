"""Behavioral tests for the ND structure-preserving operator toolkit."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "05_3d_equations.py"


def load_toolkit():
    spec = importlib.util.spec_from_file_location("structure_toolkit", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sp = load_toolkit()


class DoublePrecisionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    @classmethod
    def tearDownClass(cls):
        torch.set_default_dtype(cls.previous_default_dtype)


class GeometryAndDerivativeTests(DoublePrecisionTestCase):
    def test_periodic_domain_builds_endpoint_free_mesh_and_wave_vectors(self):
        domain = sp.PeriodicDomain((4, 6), (2 * math.pi, 3.0))

        x, y = domain.mesh()
        kx, ky = domain.wave_vectors()

        self.assertEqual(domain.dim, 2)
        self.assertEqual(domain.spacing, (math.pi / 2, 0.5))
        self.assertAlmostEqual(domain.cell_volume, math.pi / 4)
        self.assertEqual(x.shape, (4, 6))
        self.assertEqual(y.shape, (4, 6))
        self.assertAlmostEqual(x[-1, 0].item(), 3 * math.pi / 2)
        self.assertAlmostEqual(y[0, -1].item(), 2.5)
        self.assertTrue(torch.equal(kx[:, 0], torch.tensor([0.0, 1.0, -2.0, -1.0])))
        self.assertTrue(
            torch.allclose(
                ky[0],
                2 * math.pi * torch.fft.fftfreq(6, d=0.5),
            )
        )

    def test_periodic_domain_rejects_invalid_geometry(self):
        invalid = (
            ((), ()),
            ((8, 8), (2 * math.pi,)),
            ((8, 0), (2 * math.pi, 2 * math.pi)),
            ((8,), (-1.0,)),
        )

        for shape, lengths in invalid:
            with self.subTest(shape=shape, lengths=lengths):
                with self.assertRaises(ValueError):
                    sp.PeriodicDomain(shape, lengths)

    def test_autograd_geometry_matches_hand_calculated_values(self):
        coordinates = torch.tensor(
            [[0.2, -0.4], [0.7, 0.3], [-0.5, 0.8]], requires_grad=True
        )
        x = coordinates[:, 0:1]
        y = coordinates[:, 1:2]
        scalar = x**2 + 3 * y**2
        vector = torch.cat((-y, x), dim=1)

        actual_gradient = sp.gradient(scalar, coordinates)
        actual_jacobian = sp.spatial_jacobian(vector, coordinates)

        expected_jacobian = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        self.assertTrue(
            torch.allclose(actual_gradient, torch.cat((2 * x, 6 * y), dim=1))
        )
        self.assertTrue(
            torch.allclose(
                actual_jacobian,
                expected_jacobian.expand(len(coordinates), -1, -1),
            )
        )
        self.assertTrue(
            torch.allclose(sp.laplacian(scalar, coordinates), torch.full_like(x, 8.0))
        )
        self.assertTrue(
            torch.allclose(sp.divergence(vector, coordinates), torch.zeros_like(x))
        )
        self.assertTrue(
            torch.allclose(sp.curl(vector, coordinates), torch.full_like(x, 2.0))
        )

    def test_three_dimensional_curl_has_vector_shape_and_orientation(self):
        coordinates = torch.randn(7, 3, requires_grad=True)
        x = coordinates[:, 0:1]
        y = coordinates[:, 1:2]
        z = coordinates[:, 2:3]
        rotational = torch.cat((-y, x, 0 * z**2), dim=1)

        actual = sp.curl(rotational, coordinates)
        expected = torch.cat((0 * x, 0 * y, 0 * z + 2), dim=1)

        self.assertEqual(actual.shape, (7, 3))
        self.assertTrue(torch.allclose(actual, expected))

    def test_coordinate_derivatives_reject_misaligned_shapes(self):
        coordinates = torch.ones(4, 2, requires_grad=True)

        with self.assertRaises(ValueError):
            sp.gradient(torch.ones(5, 1, requires_grad=True), coordinates)
        with self.assertRaises(ValueError):
            sp.spatial_jacobian(torch.ones(4, 2), coordinates)
        with self.assertRaises(ValueError):
            sp.divergence(torch.ones(4, 3), coordinates)

    def test_centered_finite_difference_shows_second_order_convergence(self):
        errors = []
        for n in (32, 64):
            domain = sp.PeriodicDomain((n,), (2 * math.pi,))
            (x,) = domain.mesh()
            field = torch.sin(3 * x)
            exact = 3 * torch.cos(3 * x)
            actual = sp.periodic_gradient_fd(field, domain)[0]
            errors.append(torch.sqrt(torch.mean((actual - exact) ** 2)).item())

        self.assertLess(errors[1], 0.3 * errors[0])

    def test_periodic_finite_difference_laplacian_matches_low_frequency_mode(self):
        domain = sp.PeriodicDomain((128,), (2 * math.pi,))
        (x,) = domain.mesh()
        field = torch.cos(2 * x)

        actual = sp.periodic_laplacian_fd(field, domain)

        self.assertLess(
            torch.max(torch.abs(actual + 4 * field)).item(),
            0.004,
        )

    def test_spectral_derivatives_are_exact_for_nd_plane_wave(self):
        domain = sp.PeriodicDomain((12, 10), (2 * math.pi, 2 * math.pi))
        x, y = domain.mesh()
        wave = torch.exp(1j * (2 * x - 3 * y))

        gradient = sp.spectral_gradient(wave, domain)
        laplacian = sp.spectral_laplacian(wave, domain)

        self.assertTrue(torch.allclose(gradient[0], 2j * wave, atol=1e-10))
        self.assertTrue(torch.allclose(gradient[1], -3j * wave, atol=1e-10))
        self.assertTrue(torch.allclose(laplacian, -13 * wave, atol=1e-10))


class ExactKineticPhase(nn.Module):
    def forward(self, features, parameters):
        return -parameters[:, 0].reshape(-1, *([1] * (features.ndim - 2))) * features[..., 0]


class ScalingCore(nn.Module):
    def forward(self, state):
        return 3 * state


class StructurePreservationTests(DoublePrecisionTestCase):
    def test_mass_projection_matches_reference_for_each_batch_item(self):
        domain = sp.PeriodicDomain((12,), (2 * math.pi,))
        reference = torch.randn(3, 12, dtype=torch.complex128)
        prediction = torch.randn(3, 12, dtype=torch.complex128)

        projected = sp.project_to_mass(prediction, reference, domain)

        self.assertTrue(
            torch.allclose(
                sp.l2_mass(projected, domain),
                sp.l2_mass(reference, domain),
                atol=1e-12,
            )
        )

    def test_mass_projection_rejects_zero_prediction_for_nonzero_reference(self):
        domain = sp.PeriodicDomain((8,), (2 * math.pi,))
        with self.assertRaises(ValueError):
            sp.project_to_mass(torch.zeros(8), torch.ones(8), domain)

    def test_positive_projection_is_strictly_positive_and_normalized(self):
        domain = sp.PeriodicDomain((9, 7), (2 * math.pi, 3.0))
        logits = torch.randn(4, 9, 7)

        density = sp.positive_unit_mass(logits, domain)

        self.assertGreater(density.min().item(), 0.0)
        self.assertTrue(
            torch.allclose(sp.l2_mass(torch.sqrt(density), domain), torch.ones(4))
        )

    def test_helmholtz_projection_removes_divergence_and_preserves_mean(self):
        for dim, n in ((2, 10), (3, 6)):
            domain = sp.PeriodicDomain((n,) * dim, (2 * math.pi,) * dim)
            vector = torch.randn(2, dim, *((n,) * dim))

            projected = sp.helmholtz_project(vector, domain)
            actual_divergence = sp.spectral_divergence(projected, domain)

            component_axis = vector.ndim - dim - 1
            spatial_axes = tuple(range(-dim, 0))
            self.assertLess(actual_divergence.abs().max().item(), 1e-10)
            self.assertTrue(
                torch.allclose(
                    projected.mean(dim=spatial_axes),
                    vector.mean(dim=spatial_axes),
                    atol=1e-12,
                )
            )
            self.assertEqual(projected.shape[component_axis], dim)

    def test_unitary_neural_operator_preserves_mass_in_one_to_three_dimensions(self):
        torch.manual_seed(4)
        for dim in (1, 2, 3):
            n = 8 if dim < 3 else 5
            domain = sp.PeriodicDomain((n,) * dim, (2 * math.pi,) * dim)
            phase_model = sp.ParametricPhaseMLP(1, 2, width=7)
            operator = sp.UnitarySpectralOperator(domain, phase_model)
            field = torch.randn(2, *((n,) * dim), dtype=torch.complex128)
            parameters = torch.tensor([[0.7, -0.2], [1.4, 0.5]])

            output = operator(field, parameters, dt=0.03)

            self.assertTrue(
                torch.allclose(
                    sp.l2_mass(output, domain),
                    sp.l2_mass(field, domain),
                    atol=1e-11,
                )
            )

    def test_exact_unitary_phase_advances_free_schrodinger_plane_wave(self):
        domain = sp.PeriodicDomain((16, 12), (2 * math.pi, 2 * math.pi))
        alpha = 0.8
        dt = 0.07
        operator = sp.UnitarySpectralOperator(domain, ExactKineticPhase())
        initial = sp.plane_wave(domain, (2, -3), time=0.0)

        output = operator(initial.unsqueeze(0), torch.tensor([[alpha]]), dt)[0]
        expected = sp.plane_wave(domain, (2, -3), time=dt, alpha=alpha)

        self.assertTrue(torch.allclose(output, expected, atol=1e-10))

    def test_dissipative_neural_operator_contracts_l2_norm(self):
        domain = sp.PeriodicDomain((10, 8), (2 * math.pi, 2 * math.pi))
        rate_model = sp.ParametricPhaseMLP(1, 1, width=5)
        operator = sp.DissipativeSpectralOperator(domain, rate_model)
        field = torch.randn(3, 10, 8)
        parameters = torch.tensor([[0.2], [0.5], [1.0]])

        output = operator(field, parameters, dt=0.1)

        self.assertTrue(
            torch.all(sp.l2_mass(output, domain) <= sp.l2_mass(field, domain) + 1e-12)
        )
        with self.assertRaises(ValueError):
            operator(field, parameters, dt=-0.1)

    def test_projected_operator_enforces_mass_on_arbitrary_core(self):
        domain = sp.PeriodicDomain((11,), (2 * math.pi,))
        state = torch.randn(2, 11, dtype=torch.complex128)
        operator = sp.ProjectedOperator(
            ScalingCore(),
            lambda raw, reference: sp.project_to_mass(raw, reference, domain),
        )

        output = operator(state)

        self.assertTrue(
            torch.allclose(sp.l2_mass(output, domain), sp.l2_mass(state, domain))
        )

    def test_split_step_nls_matches_constant_potential_plane_wave(self):
        domain = sp.PeriodicDomain((12, 10), (2 * math.pi, 2 * math.pi))
        operator = sp.SplitStepNLSOperator(domain)
        alpha = torch.tensor([0.7])
        beta = torch.tensor([0.4])
        potential_value = 0.3
        amplitude = 1.2
        dt = 0.04
        initial = sp.plane_wave(
            domain, (2, -1), amplitude=amplitude, time=0.0
        ).unsqueeze(0)
        potential = torch.full(domain.shape, potential_value).unsqueeze(0)

        output = operator(initial, potential, alpha, beta, dt)
        expected = sp.plane_wave(
            domain,
            (2, -1),
            amplitude=amplitude,
            time=dt,
            alpha=alpha.item(),
            beta=beta.item(),
            potential_constant=potential_value,
        )

        self.assertTrue(torch.allclose(output[0], expected, atol=1e-10))
        self.assertTrue(
            torch.allclose(sp.l2_mass(output, domain), sp.l2_mass(initial, domain))
        )

    def test_split_step_nls_is_reversible_with_nonconstant_potential(self):
        domain = sp.PeriodicDomain((10, 8), (2 * math.pi, 2 * math.pi))
        x, y = domain.mesh()
        initial = torch.randn(2, *domain.shape, dtype=torch.complex128)
        potential = (0.2 * torch.cos(x) - 0.1 * torch.sin(2 * y)).expand(2, -1, -1)
        alpha = torch.tensor([0.6, 1.1])
        beta = torch.tensor([0.3, -0.2])
        operator = sp.SplitStepNLSOperator(domain)

        forward = operator(initial, potential, alpha, beta, 0.03)
        recovered = operator(forward, potential, alpha, beta, -0.03)

        self.assertTrue(torch.allclose(recovered, initial, atol=1e-10))
        self.assertTrue(
            torch.allclose(sp.l2_mass(forward, domain), sp.l2_mass(initial, domain))
        )

    def test_nls_hamiltonian_matches_plane_wave_energy(self):
        domain = sp.PeriodicDomain((14,), (2 * math.pi,))
        alpha, beta, potential_value, amplitude = 0.8, 0.3, 0.2, 1.1
        wave = sp.plane_wave(domain, (3,), amplitude=amplitude).unsqueeze(0)
        potential = torch.full((1, *domain.shape), potential_value)

        actual = sp.nls_hamiltonian(
            wave, potential, domain, torch.tensor([alpha]), torch.tensor([beta])
        )
        expected_density = (
            alpha * 9 * amplitude**2
            + potential_value * amplitude**2
            - 0.5 * beta * amplitude**4
        )

        self.assertAlmostEqual(actual.item(), 2 * math.pi * expected_density, places=9)

    def test_canonical_hamiltonian_field_is_energy_orthogonal(self):
        q = torch.tensor([0.4, -0.7], requires_grad=True)
        p = torch.tensor([1.2, 0.3], requires_grad=True)
        hamiltonian = 0.5 * torch.sum(q**2 + p**2)

        q_dot, p_dot = sp.canonical_hamiltonian_field(hamiltonian, q, p)
        directional_derivative = torch.dot(q, q_dot) + torch.dot(p, p_dot)

        self.assertTrue(torch.allclose(q_dot, p))
        self.assertTrue(torch.allclose(p_dot, -q))
        self.assertAlmostEqual(directional_derivative.item(), 0.0, places=12)


class DemoAndCliTests(DoublePrecisionTestCase):
    def test_derivative_demo_emits_no_tensor_conversion_warning(self):
        domain = sp.PeriodicDomain((8,), (2 * math.pi,))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sp.run_derivative_demo(domain)

        self.assertEqual(caught, [])

    def test_schrodinger_demo_emits_no_tensor_conversion_warning(self):
        domain = sp.PeriodicDomain((8,), (2 * math.pi,))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sp.run_schrodinger_demo(domain, steps=1, dt=0.02)

        self.assertEqual(caught, [])

    def test_heat_demo_emits_no_tensor_conversion_warning(self):
        domain = sp.PeriodicDomain((8,), (2 * math.pi,))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sp.run_heat_demo(domain, steps=1, dt=0.02)

        self.assertEqual(caught, [])

    def test_incompressible_demo_emits_no_tensor_conversion_warning(self):
        domain = sp.PeriodicDomain((5, 5, 5), (2 * math.pi,) * 3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sp.run_incompressible_demo(domain)

        self.assertEqual(caught, [])

    def test_main_restores_callers_default_dtype(self):
        torch.set_default_dtype(torch.float32)
        try:
            sp.main(
                [
                    "--demo",
                    "density",
                    "--dim",
                    "1",
                    "--grid-size",
                    "8",
                    "--no-plot",
                ]
            )
            self.assertEqual(torch.get_default_dtype(), torch.float32)
        finally:
            torch.set_default_dtype(torch.float64)

    def test_heat_spectral_step_matches_analytic_nd_mode(self):
        domain = sp.PeriodicDomain((12, 10), (2 * math.pi, 2 * math.pi))
        x, y = domain.mesh()
        initial = torch.cos(2 * x - y)
        diffusivity = 0.3
        dt = 0.08

        output = sp.heat_spectral_step(initial, domain, diffusivity, dt)
        expected = initial * math.exp(-diffusivity * 5 * dt)

        self.assertTrue(torch.allclose(output, expected, atol=1e-10))
        with self.assertRaises(ValueError):
            sp.heat_spectral_step(initial, domain, diffusivity, -dt)

    def test_each_demo_reports_its_guaranteed_structure(self):
        domain = sp.PeriodicDomain((8, 8), (2 * math.pi, 2 * math.pi))

        derivatives = sp.run_derivative_demo(domain)
        schrodinger = sp.run_schrodinger_demo(domain, steps=2, dt=0.02)
        heat = sp.run_heat_demo(domain, steps=2, dt=0.02)
        incompressible = sp.run_incompressible_demo(domain)
        density = sp.run_density_demo(domain)

        self.assertLess(derivatives.metrics["spectral_laplacian_max_error"], 1e-10)
        self.assertLess(derivatives.metrics["fd_refinement_ratio"], 0.3)
        self.assertLess(schrodinger.metrics["plane_wave_max_error"], 1e-10)
        self.assertLess(schrodinger.metrics["split_mass_relative_drift"], 1e-10)
        self.assertLess(schrodinger.metrics["unitary_nn_mass_relative_drift"], 1e-10)
        self.assertLess(schrodinger.metrics["reversibility_relative_error"], 1e-10)
        self.assertLess(schrodinger.metrics["hamiltonian_orthogonality"], 1e-10)
        self.assertLess(heat.metrics["analytic_mode_max_error"], 1e-10)
        self.assertLessEqual(heat.metrics["neural_contraction_ratio"], 1.0 + 1e-12)
        self.assertIsNotNone(incompressible)
        self.assertLess(incompressible.metrics["divergence_after"], 1e-10)
        self.assertLess(incompressible.metrics["autograd_divergence"], 1e-10)
        self.assertGreater(density.metrics["minimum"], 0.0)
        self.assertLess(density.metrics["mass_error"], 1e-10)

    def test_incompressible_demo_is_explicitly_skipped_in_one_dimension(self):
        domain = sp.PeriodicDomain((12,), (2 * math.pi,))
        self.assertIsNone(sp.run_incompressible_demo(domain))

    def test_cli_smokes_every_demo_without_plotting(self):
        for demo in (
            "derivatives",
            "schrodinger",
            "heat",
            "incompressible",
            "density",
        ):
            with self.subTest(demo=demo):
                results = sp.main(
                    [
                        "--demo",
                        demo,
                        "--dim",
                        "2",
                        "--grid-size",
                        "6",
                        "--steps",
                        "2",
                        "--no-plot",
                    ]
                )
                self.assertIn(demo, results)

    def test_all_demo_runs_in_three_dimensions(self):
        results = sp.main(
            [
                "--demo",
                "all",
                "--dim",
                "3",
                "--grid-size",
                "5",
                "--steps",
                "1",
                "--no-plot",
            ]
        )
        self.assertEqual(
            set(results),
            {"derivatives", "schrodinger", "heat", "incompressible", "density"},
        )

    def test_cli_generates_nonempty_summary_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.png"

            results = sp.main(
                [
                    "--demo",
                    "density",
                    "--dim",
                    "1",
                    "--grid-size",
                    "12",
                    "--output",
                    str(output),
                ]
            )

            self.assertIn("density", results)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
