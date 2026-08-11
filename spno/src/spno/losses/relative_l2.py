"""Relative L2, the primary objective and the primary metric.

Relative rather than absolute because mass varies across samples by design: an
absolute L2 would weight high-mass trajectories more heavily and quietly turn the
objective into "fit the big ones".
"""

from __future__ import annotations

import torch

from ..domain import PeriodicDomain

Tensor = torch.Tensor


def relative_l2_per_sample(
    prediction: Tensor, target: Tensor, domain: PeriodicDomain
) -> Tensor:
    """Per-sample relative L2 over the spatial axes; works for complex fields."""

    domain.validate_field(prediction)
    domain.validate_field(target)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    numerator = torch.sqrt(
        torch.sum(torch.abs(prediction - target) ** 2, dim=domain.spatial_axes)
    )
    denominator = torch.sqrt(
        torch.sum(torch.abs(target) ** 2, dim=domain.spatial_axes)
    )
    return numerator / denominator.clamp_min(1e-12)


def relative_l2_loss(
    prediction: Tensor, target: Tensor, domain: PeriodicDomain
) -> Tensor:
    """Batch mean of the per-sample relative L2."""

    return relative_l2_per_sample(prediction, target, domain).mean()
