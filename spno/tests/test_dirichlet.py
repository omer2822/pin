import pytest
import torch

pytest.importorskip("scipy")

from spno.domain import HilbertSpace


@pytest.mark.parametrize('geometry', ['rectangle', 'disk'])
def test_harmonic_linear_solution_has_exact_boundary_and_small_residual(geometry):
    from spno.dirichlet import rectangle_mesh, disk_mesh, solve_dirichlet
    mesh = rectangle_mesh(8, 6) if geometry == 'rectangle' else disk_mesh(4, 24)
    assert isinstance(mesh, HilbertSpace)
    x, y = mesh.points.T
    exact = x + 2 * y
    result = solve_dirichlet(mesh, torch.zeros_like(x), exact)
    assert torch.equal(result.values[mesh.boundary], exact[mesh.boundary])
    assert torch.max(torch.abs(result.values - exact)) < 1e-12
    assert result.relative_residual < 1e-12
    assert torch.all(mesh.quadrature_weights() > 0)


@pytest.mark.parametrize('geometry', ['rectangle', 'disk'])
def test_manufactured_poisson_solution_converges_under_refinement(geometry):
    from spno.dirichlet import rectangle_mesh, disk_mesh, solve_dirichlet
    errors = []
    for n in (4, 8, 16):
        mesh = rectangle_mesh(n, n) if geometry == 'rectangle' else disk_mesh(n, 8 * n)
        x, y = mesh.points.T
        exact = torch.sin(x) * torch.cos(y)
        result = solve_dirichlet(mesh, 2 * exact, exact)
        errors.append(float(mesh.norm(result.values - exact)))
    assert errors[1] < errors[0] * .6
    assert errors[2] < errors[1] * .6


def test_mesh_rejects_degenerate_input_and_invalid_rhs():
    from spno.dirichlet import rectangle_mesh, disk_mesh, solve_dirichlet
    with pytest.raises(ValueError):
        rectangle_mesh(0, 3)
    with pytest.raises(ValueError):
        disk_mesh(3, 2)
    mesh = rectangle_mesh(3, 3)
    with pytest.raises(ValueError):
        solve_dirichlet(mesh, torch.zeros(4), torch.zeros(mesh.shape))
