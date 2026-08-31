"""Precision handling for models that carry complex parameters.

Neither stock PyTorch call does the right thing to an FNO:

* ``model.double()`` converts float32 parameters but **skips** complex ones, leaving a
  mixed-dtype module that fails at the first matmul.
* ``model.to(torch.float64)`` converts the complex spectral weights to *real*,
  discarding the imaginary part -- silently, with only a warning.  That destroys the
  operator while leaving a module that still runs.

Both were observed on this model.  :func:`widen_to_double` maps float32 -> float64 and
complex64 -> complex128 explicitly, which is what invariant evaluation needs: rollout
and conservation are measured in float64 so that float32 arithmetic drift (~1.2e-7 per
split step, accumulating roughly linearly) does not sit on top of every conservation
number and mask an architectural violation of comparable size.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .domain import PeriodicDomain

Tensor = torch.Tensor


def widen_to_double(
    model: nn.Module, *, device: str = "cpu", clone: bool = True
) -> nn.Module:
    """Return ``model`` on ``device`` with float32 -> float64, complex64 -> complex128.

    The learned values are unchanged; only the arithmetic used to iterate them is.

    Order matters and is handled here: the copy is moved *before* widening, because MPS
    has no float64 and would raise.  Cloning by default keeps the original on its
    training device -- ``model.to(...)`` mutates in place, which would strand a core
    that another model shares.
    """

    target = copy.deepcopy(model) if clone else model
    target = target.to(device)

    def convert(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_complex():
            return tensor.to(torch.complex128)
        if tensor.is_floating_point():
            return tensor.to(torch.float64)
        return tensor

    return target._apply(convert)


def parameter_dtypes(model: nn.Module) -> set[str]:
    return {str(p.dtype) for p in model.parameters()}


def batch_parameter(
    value: Tensor | float,
    batch: int,
    domain: PeriodicDomain,
    reference: Tensor,
    name: str,
) -> Tensor:
    """Normalize a scalar-or-per-sample PDE parameter to a grid-broadcastable tensor.

    Rejects a silent precision downcast.  ``torch.tensor([0.9])`` is float32, and
    feeding it to a float64 field costs ~2.6e-8 in alpha -- enough to move the
    one-step phase at k=30 by 5e-7 and to invalidate every 1e-10 assertion in the
    Phase 0 suite.  Pass a python float or an explicitly-typed tensor instead.

    Historical home: ``spno.domain`` -- this is precision policy, not geometry, and
    touches ``domain`` only to append singleton spatial axes via
    :func:`spno.domain.spatial_broadcast`.  Re-exported from ``spno.domain`` so no
    importer's path had to change (see ADR 0001).
    """

    if isinstance(value, Tensor) and value.is_floating_point():
        target_dtype = reference.real.dtype
        if torch.finfo(value.dtype).bits < torch.finfo(target_dtype).bits:
            raise ValueError(
                f"{name} is {value.dtype} but the field is {target_dtype}; this would "
                f"silently lose precision. Pass a python float or a {target_dtype} tensor."
            )
    tensor = torch.as_tensor(
        value, device=reference.device, dtype=reference.real.dtype
    )
    if tensor.ndim == 0:
        tensor = tensor.expand(batch)
    if tensor.shape != (batch,):
        raise ValueError(f"{name} must be scalar or have shape (batch,)")
    return tensor.reshape(batch, *((1,) * domain.dim))

