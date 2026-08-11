"""Reproducible seeding.

Every dataset shard and every training run records the seed it was produced with, so
a result can be regenerated from its config alone.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> torch.Generator:
    """Seed python/numpy/torch and return a generator for explicit sampling."""

    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
