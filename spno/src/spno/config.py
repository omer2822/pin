"""Frozen configuration objects.

Every result in this project must be regenerable from a config plus a seed.  The
config hash is recorded alongside metrics and printed into figure captions.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

from .domain import PeriodicDomain


@dataclass(frozen=True)
class DataConfig:
    """The training distribution.

    ``alpha_range``/``beta_range`` are provisional and are narrowed by the Phase 0
    resolution screen if any corner of the box produces under-resolved dynamics.
    """

    grid_size: int = 64
    dt: float = 0.01
    substeps: int = 32
    steps: int = 200
    alpha_range: tuple[float, float] = (0.7, 1.1)
    beta_range: tuple[float, float] = (-0.4, 0.6)
    initial_bandwidth: int = 8
    potential_amplitude_range: tuple[float, float] = (0.0, 0.5)
    potential_correlation_length: float = 1.0
    mass_range: tuple[float, float] = (1.0, 3.0)
    n_train: int = 800
    n_val: int = 100
    n_test: int = 100
    seed: int = 0

    @property
    def domain(self) -> PeriodicDomain:
        return PeriodicDomain.periodic_1d(self.grid_size)

    @property
    def max_wave_number(self) -> int:
        return self.grid_size // 2


@dataclass(frozen=True)
class Phase0Config:
    """Reference-validation sweep."""

    data: DataConfig = field(default_factory=DataConfig)
    convergence_substeps: tuple[int, ...] = (4, 8, 16, 32, 64)
    convergence_horizon: float = 0.16
    long_run_steps: int = 1000
    screen_samples: int = 64
    tail_fraction_threshold: float = 1e-6
    tail_band_start: float = 0.75


def config_hash(config: Any) -> str:
    """Stable 10-character hash of a dataclass config, used to name result directories."""

    payload = json.dumps(dataclasses.asdict(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:10]
