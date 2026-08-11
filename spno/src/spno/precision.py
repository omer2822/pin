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
