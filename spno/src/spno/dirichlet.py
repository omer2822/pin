"""P1 finite elements for -Laplacian(u)=f, u=g on a bounded planar mesh.

Nodal fields form a HilbertSpace with lumped-mass quadrature. This is not a
Fourier SpectralDomain: first derivatives do not act by diagonal multipliers.
SciPy is imported only when assembling/solving the optional Dirichlet extension.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .domain import HilbertSpace


class TriangleDomain(HilbertSpace):
    geometric_dim = 2

    def __init__(self, points, triangles, boundary):
        self.points = torch.as_tensor(points, dtype=torch.float64).clone()
        self.triangles = torch.as_tensor(triangles, dtype=torch.int64).clone()
        self.boundary = torch.as_tensor(boundary, dtype=torch.bool).clone()
        if self.points.ndim != 2 or self.points.shape[1] != 2 or not torch.isfinite(self.points).all():
            raise ValueError('points must be finite (n, 2) coordinates')
        self.shape = (len(self.points),)
        if self.boundary.shape != self.shape or not self.boundary.any():
            raise ValueError('boundary must mark boundary nodes')
        if (self.triangles.ndim != 2 or self.triangles.shape[1] != 3 or not len(self.triangles)
                or self.triangles.min() < 0 or self.triangles.max() >= len(self.points)):
            raise ValueError('triangles must contain valid node triples')
        vertices = self.points[self.triangles]
        a, b = vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
        self._determinant = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        if torch.any(self._determinant.abs() <= torch.finfo(torch.float64).eps):
            raise ValueError('degenerate triangle')
        self.areas = self._determinant.abs() / 2
        self._weights = torch.zeros(self.shape, dtype=torch.float64)
        self._weights.scatter_add_(0, self.triangles.flatten(), self.areas.repeat_interleave(3) / 3)
        if torch.any(self._weights <= 0):
            raise ValueError('every node must belong to a triangle')

    def validate_field(self, field):
        if field.ndim < 1 or field.shape[-1:] != self.shape:
            raise ValueError(f'expected nodal field ending in {self.shape}')

    def quadrature_weights(self, *, device=None, dtype=torch.float64):
        return self._weights.to(device=device, dtype=dtype).clone()

    def stiffness(self):
        from scipy.sparse import coo_matrix
        vertices = self.points[self.triangles]
        x, y = vertices[:, :, 0], vertices[:, :, 1]
        gradients = torch.stack((
            torch.stack((y[:, 1] - y[:, 2], x[:, 2] - x[:, 1]), dim=-1),
            torch.stack((y[:, 2] - y[:, 0], x[:, 0] - x[:, 2]), dim=-1),
            torch.stack((y[:, 0] - y[:, 1], x[:, 1] - x[:, 0]), dim=-1),
        ), dim=1) / self._determinant[:, None, None]
        local = self.areas[:, None, None] * torch.einsum('tik,tjk->tij', gradients, gradients)
        rows = self.triangles[:, :, None].expand(-1, 3, 3).reshape(-1).numpy()
        cols = self.triangles[:, None, :].expand(-1, 3, 3).reshape(-1).numpy()
        return coo_matrix((local.flatten().numpy(), (rows, cols)), shape=(self.shape[0],) * 2).tocsr()


def _count(value, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'mesh count must be an integer >= {minimum}')
    return value


def rectangle_mesh(nx=16, ny=16, *, lengths=(1., 1.)) -> TriangleDomain:
    nx, ny = _count(nx, 1), _count(ny, 1)
    if len(lengths) != 2 or any(not math.isfinite(v) or v <= 0 for v in lengths):
        raise ValueError('rectangle lengths must be finite and positive')
    x, y = torch.meshgrid(torch.linspace(0, lengths[0], nx + 1, dtype=torch.float64),
                          torch.linspace(0, lengths[1], ny + 1, dtype=torch.float64), indexing='ij')
    points = torch.stack((x.flatten(), y.flatten()), dim=-1)
    triangles = []
    for i in range(nx):
        for j in range(ny):
            a = i * (ny + 1) + j
            b, c, d = a + ny + 1, a + 1, a + ny + 2
            triangles.extend(((a, b, d), (a, d, c)))
    boundary = (x == 0) | (y == 0) | (x == lengths[0]) | (y == lengths[1])
    return TriangleDomain(points, triangles, boundary.flatten())


def disk_mesh(rings=8, angles=64, *, radius=1.) -> TriangleDomain:
    rings, angles = _count(rings, 1), _count(angles, 3)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('radius must be finite and positive')
    points = [(0., 0.)]
    for ring in range(1, rings + 1):
        r = radius * ring / rings
        points.extend((r * math.cos(2 * math.pi * j / angles),
                       r * math.sin(2 * math.pi * j / angles)) for j in range(angles))
    triangles = [(0, 1 + j, 1 + (j + 1) % angles) for j in range(angles)]
    for ring in range(1, rings):
        inner, outer = 1 + (ring - 1) * angles, 1 + ring * angles
        for j in range(angles):
            k = (j + 1) % angles
            triangles.extend(((inner + j, outer + j, outer + k),
                              (inner + j, outer + k, inner + k)))
    boundary = torch.zeros(len(points), dtype=torch.bool)
    boundary[-angles:] = True
    return TriangleDomain(points, triangles, boundary)


@dataclass(frozen=True)
class DirichletResult:
    values: torch.Tensor
    relative_residual: float


def solve_dirichlet(domain: TriangleDomain, rhs, boundary_values) -> DirichletResult:
    from scipy.sparse.linalg import spsolve
    rhs = torch.as_tensor(rhs, dtype=torch.float64).cpu()
    boundary_values = torch.as_tensor(boundary_values, dtype=torch.float64).cpu()
    for field in (rhs, boundary_values):
        if field.shape != domain.shape or not torch.isfinite(field).all():
            raise ValueError(f'provide finite real nodal values of shape {domain.shape}')
    matrix = domain.stiffness()
    # Consistent P1 load: area/12 * [[2,1,1],[1,2,1],[1,1,2]] @ nodal f.
    samples = rhs[domain.triangles]
    local_load = domain.areas[:, None] / 12 * (samples + samples.sum(dim=1, keepdim=True))
    load = torch.zeros(domain.shape, dtype=torch.float64)
    load.scatter_add_(0, domain.triangles.flatten(), local_load.flatten())
    fixed = domain.boundary.numpy()
    free = ~fixed
    result = boundary_values.numpy().copy()
    residual = 0.
    if free.any():
        reduced = matrix[free][:, free]
        b = load.numpy()[free] - matrix[free][:, fixed] @ result[fixed]
        result[free] = spsolve(reduced, b)
        residual = float(np.linalg.norm(reduced @ result[free] - b) / max(np.linalg.norm(b), 1.))
    if not np.isfinite(result).all():
        raise RuntimeError('Dirichlet solve produced non-finite values')
    return DirichletResult(torch.from_numpy(result), residual)


def demo(*, refinements=(4, 8, 16), output=None):
    """Manufactured solutions and refinement plots; small CPU-only notebook entry point."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    report = {}
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for row, geometry in enumerate(('rectangle', 'disk')):
        errors, harmonic_errors = [], []
        for n in refinements:
            domain = rectangle_mesh(n, n) if geometry == 'rectangle' else disk_mesh(n, n * 8)
            x, y = domain.points.T
            exact = torch.sin(x) * torch.cos(y)
            result = solve_dirichlet(domain, 2 * exact, exact)
            errors.append(float(domain.norm(result.values - exact)))
            harmonic = x + 2 * y
            harmonic_result = solve_dirichlet(domain, torch.zeros_like(x), harmonic)
            harmonic_errors.append(float(torch.max(torch.abs(harmonic_result.values - harmonic))))
        axes[row, 0].triplot(x, y, domain.triangles, linewidth=.3)
        axes[row, 0].set_title(f'{geometry}: mesh')
        plotted = axes[row, 1].tripcolor(x, y, domain.triangles, result.values, shading='gouraud')
        figure.colorbar(plotted, ax=axes[row, 1])
        axes[row, 1].set_title('Poisson solution')
        axes[row, 2].loglog(refinements, errors, 'o-', label='weighted L2 error')
        axes[row, 2].set(xlabel='refinement', ylabel='error', title='Convergence')
        axes[row, 2].grid(True)
        for axis in axes[row, :2]:
            axis.set_aspect('equal')
        report[geometry] = {'refinements': list(refinements), 'errors': errors,
                            'harmonic_errors': harmonic_errors, 'relative_residual': result.relative_residual}
    figure.tight_layout()
    if output:
        from .artifacts import atomic_json
        path = Path(output)
        path.mkdir(parents=True, exist_ok=True)
        figure.savefig(path / 'dirichlet.png', dpi=140)
        atomic_json(path / 'metrics.json', report)
    return report, figure
