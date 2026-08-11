"""Structure-preserving neural operators for the parametric NLS equation."""

from .domain import (
    PeriodicDomain,
    l2_mass,
    project_to_mass,
    spectral_gradient,
    spectral_laplacian,
)

__all__ = [
    "PeriodicDomain",
    "l2_mass",
    "project_to_mass",
    "spectral_gradient",
    "spectral_laplacian",
]
