"""Named distribution-shift arms for Phase 6 (G1-G4 and G7).

Each arm is a :class:`~spno.config.DataConfig` produced by ``dataclasses.replace``,
paired with the potential family :func:`~spno.data.datasets.generate_shard` should draw.
Naming them here rather than passing kwargs at the call site buys two things:

* **Reproducibility.** ``shift_identifier`` derives the run directory from the config
  itself, so an edited range cannot silently overwrite another arm's results.
* **Auditability.** The shift is visible as data, so "which distribution produced this
  number" is answerable from the registry instead of from a runner's argv.

**k_wrap is arm-dependent.**  ``k_wrap = sqrt(pi / (alpha*dt))`` is a function of alpha,
so widening the alpha range moves the identifiability horizon.  On the production config
(``alpha`` in ``[0.7, 1.1]``, ``dt=0.01``) it spans **16.9-21.2**; the G2 extrapolation
arm (``alpha`` in ``[0.5, 1.5]``) spans **14.5-25.1**.  Every plot must therefore draw
its own arm's horizon rather than a single global line, and :func:`banded_error` edges
must be recomputed per arm.

All arms except ``G7-alpha-fixed`` are **evaluation sets**: ``n_train = n_val = 0``.
G7 is the only arm that trains, because it is an ablation on the training distribution
(alpha held fixed) rather than a shifted test set.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..config import DataConfig, config_hash
from .generate import PotentialFamily


@dataclass(frozen=True)
class ShiftSpec:
    """One named distribution-shift arm."""

    name: str
    config: DataConfig
    potential_family: PotentialFamily = "random"
    note: str = ""


def shift_identifier(spec: ShiftSpec) -> str:
    """Run identifier for an arm: the config hash plus the potential family.

    The family is appended because it is *not* a ``DataConfig`` field -- two arms can
    share a config and differ only in the potential they draw, and pooling those into
    one directory would silently overwrite one with the other.
    """

    return f"{config_hash(spec.config)}-{spec.potential_family}"


def _evaluation(**changes) -> DataConfig:
    """An evaluation-only config: no training or validation trajectories."""

    return replace(DataConfig(), n_train=0, n_val=0, n_test=100, **changes)


SHIFT_SPECS: dict[str, ShiftSpec] = {
    "G1-interpolation": ShiftSpec(
        name="G1-interpolation",
        config=_evaluation(seed=101),
        note="fresh draw inside the training ranges: the control",
    ),
    "G2-extrapolation": ShiftSpec(
        name="G2-extrapolation",
        config=_evaluation(
            alpha_range=(0.5, 1.5), beta_range=(-0.5, 0.8), seed=102
        ),
        note="alpha/beta outside training on both sides; k_wrap moves to 14.5-25.1",
    ),
    "G3-potential-zero": ShiftSpec(
        name="G3-potential-zero",
        config=_evaluation(seed=103),
        potential_family="zero",
        note="V=0 restores translation invariance, so momentum is conserved here",
    ),
    "G3-potential-cosine": ShiftSpec(
        name="G3-potential-cosine",
        config=_evaluation(seed=104),
        potential_family="cosine",
    ),
    "G3-potential-well": ShiftSpec(
        name="G3-potential-well",
        config=_evaluation(seed=105),
        potential_family="gaussian_well",
    ),
    "G3-potential-short": ShiftSpec(
        name="G3-potential-short",
        config=_evaluation(potential_correlation_length=0.3, seed=106),
        note="short correlation length puts potential energy at higher k",
    ),
    "G3-potential-strong": ShiftSpec(
        name="G3-potential-strong",
        config=_evaluation(potential_amplitude_range=(0.0, 1.0), seed=107),
        note="2x the training potential amplitude",
    ),
    "G7-alpha-fixed": ShiftSpec(
        name="G7-alpha-fixed",
        config=replace(
            DataConfig(),
            alpha_range=(0.9, 0.9),
            n_train=800,
            n_val=100,
            n_test=100,
            seed=120,
        ),
        note=(
            "the only TRAINING arm: alpha held fixed to isolate the alpha-conditioning "
            "mechanism.  Compare via band_ratio, never absolute E(k) -- fixing alpha "
            "makes the task easier at every k"
        ),
    ),
}

for _bandwidth in (12, 16, 20, 24):
    SHIFT_SPECS[f"G4-bandwidth-{_bandwidth}"] = ShiftSpec(
        name=f"G4-bandwidth-{_bandwidth}",
        config=_evaluation(initial_bandwidth=_bandwidth, seed=110 + _bandwidth),
        note="error should inflect near k_wrap (16.9-21.2 on the production config)",
    )
del _bandwidth
