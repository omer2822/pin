"""Phase 8 robustness: additive noise on the field and on the potential.

Kept separate from :mod:`spno.data.generate` because this corrupts *observations*, not
the physics.  The trajectories are still exact solutions of the exact equation; what
degrades is what the model is allowed to see.  Phase 9's dials are the opposite --
there the equation itself changes and the trajectories are genuinely different.

The two noise channels are swept **separately, never jointly**.  Field noise and
potential noise probe different failure modes (state estimation versus parameter
estimation), and a joint sweep could not attribute a degradation to either.

Both take an explicit :class:`torch.Generator`, so a sweep is reproducible and two
noise levels drawn in the same run do not share a global RNG position.
"""

from __future__ import annotations

import torch

from ..domain import PeriodicDomain

Tensor = torch.Tensor


def add_field_noise(
    field: Tensor, domain: PeriodicDomain, *, level: float, generator: torch.Generator
) -> Tensor:
    """Complex Gaussian noise scaled per sample so ``||noise|| / ||field|| == level``.

    Per-sample rather than global, because mass varies across the dataset by design; a
    single global scale would make the effective noise level depend on a trajectory's
    mass and confound the robustness sweep with the mass distribution.

    ``level == 0.0`` returns the input **unchanged**, not merely almost unchanged, so
    the zero rung of a sweep is bitwise the uncorrupted arm.
    """

    domain.validate_field(field)
    if level < 0:
        raise ValueError("level must be non-negative")
    if level == 0.0:
        return field
    if not field.is_complex():
        raise ValueError("add_field_noise expects a complex field")

    real_dtype = field.real.dtype
    noise = torch.complex(
        torch.randn(field.shape, generator=generator, dtype=real_dtype),
        torch.randn(field.shape, generator=generator, dtype=real_dtype),
    )
    field_norm = torch.sqrt(
        torch.sum(torch.abs(field) ** 2, dim=domain.spatial_axes, keepdim=True)
    )
    noise_norm = torch.sqrt(
        torch.sum(torch.abs(noise) ** 2, dim=domain.spatial_axes, keepdim=True)
    ).clamp_min(1e-30)
    return field + noise * (level * field_norm / noise_norm)


def add_potential_noise(
    potential: Tensor, *, level: float, generator: torch.Generator
) -> Tensor:
    """Real Gaussian noise scaled to ``level * max|V|``.

    Scaled by the amplitude rather than the norm because ``V`` is a *parameter field*
    whose natural scale is its amplitude range, and because ``V = 0`` is a legitimate
    arm (the G3 zero-potential family) where a norm-relative level would be undefined.
    """

    if level < 0:
        raise ValueError("level must be non-negative")
    if level == 0.0:
        return potential
    if potential.is_complex():
        raise ValueError("add_potential_noise expects a real potential")

    scale = torch.abs(potential).amax()
    noise = torch.randn(
        potential.shape, generator=generator, dtype=potential.dtype
    )
    return potential + noise * (level * scale)
