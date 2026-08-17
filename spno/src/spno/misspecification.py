"""Phase 9 configuration: which dial is turned, and what that names on disk.

Deliberately **not** fields on :class:`~spno.config.DataConfig`.  Folding them in would
change ``config_hash(DataConfig())`` away from ``bd4e108527``, orphaning the 206 MB of
shards already on disk and every artifact path cited in the thesis -- and regenerating
them is a deferred run.  The identifier below instead *collapses* to the bare data hash
at dial zero, so the exact case reuses those shards untouched while every perturbed arm
gets its own directory.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from .config import DataConfig, config_hash
from .solvers.perturbed import (
    GainLossSplitStepNLSOperator,
    NonlocalSplitStepNLSOperator,
    SubsteppedOperator,
)
from .solvers.split_step import SubsteppedReference


@dataclass(frozen=True)
class MisspecificationConfig:
    """Dials on the data-generating equation.  Both recover the exact case at 0."""

    nonlocal_sigma: float = 0.0
    gain_loss_gamma: float = 0.0

    def __post_init__(self) -> None:
        if self.nonlocal_sigma < 0:
            raise ValueError("nonlocal_sigma must be non-negative")
        if self.nonlocal_sigma != 0.0 and self.gain_loss_gamma != 0.0:
            raise ValueError(
                "turn one dial at a time: two simultaneous perturbations cannot be "
                "attributed to either broken assumption"
            )

    @property
    def is_exact(self) -> bool:
        return self.nonlocal_sigma == 0.0 and self.gain_loss_gamma == 0.0

    def identifier(self, data_config: DataConfig) -> str:
        """Run/dataset identifier; the bare data hash when no dial is turned."""

        base = config_hash(data_config)
        if self.is_exact:
            return base
        return f"{base}-s{self.nonlocal_sigma:g}-g{self.gain_loss_gamma:g}"

    def reference(self, data_config: DataConfig) -> nn.Module:
        """The substepped generator carrying this dial.

        At dial zero this returns a plain :class:`SubsteppedReference` -- the *same*
        class the production shards were generated with -- rather than a wrapper around
        a zero-dial operator.  Phase 9's opening gate asserts bitwise reproduction, and
        routing the exact case through a different object risks it in a way that would
        make every crossover in the sweep uninterpretable.
        """

        domain = data_config.domain
        if self.is_exact:
            return SubsteppedReference(domain, data_config.substeps)
        if self.nonlocal_sigma != 0.0:
            inner: nn.Module = NonlocalSplitStepNLSOperator(
                domain, sigma=self.nonlocal_sigma
            )
        else:
            inner = GainLossSplitStepNLSOperator(domain, gamma=self.gain_loss_gamma)
        return SubsteppedOperator(inner, data_config.substeps)

    def as_dict(self) -> dict:
        return {
            "nonlocal_sigma": self.nonlocal_sigma,
            "gain_loss_gamma": self.gain_loss_gamma,
            "is_exact": self.is_exact,
        }
