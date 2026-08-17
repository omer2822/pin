"""Phase 6: the distribution-shift arms are specified as data, not ad-hoc kwargs.

Every G1-G4/G7 arm is a ``dataclasses.replace`` on :class:`DataConfig` plus a potential
family.  Naming them in a registry is what keeps an arm reproducible: the identifier is
derived from the config itself, so a silently edited range cannot reuse another arm's
result directory.
"""

from __future__ import annotations

import pytest

from spno.config import DataConfig, config_hash
from spno.data.shift import SHIFT_SPECS, shift_identifier

BASE = DataConfig()


def test_every_arm_that_changes_the_distribution_changes_the_identifier():
    identifiers = {name: shift_identifier(spec) for name, spec in SHIFT_SPECS.items()}
    assert len(set(identifiers.values())) == len(identifiers), identifiers


def test_the_interpolation_arm_keeps_the_training_parameter_ranges():
    spec = SHIFT_SPECS["G1-interpolation"]
    assert spec.config.alpha_range == BASE.alpha_range
    assert spec.config.beta_range == BASE.beta_range
    assert spec.config.seed != BASE.seed, "a fresh draw needs a fresh seed"


def test_the_extrapolation_arm_leaves_the_training_range_on_both_sides():
    low, high = SHIFT_SPECS["G2-extrapolation"].config.alpha_range
    assert low < BASE.alpha_range[0] and high > BASE.alpha_range[1]


@pytest.mark.parametrize("bandwidth", [12, 16, 20, 24])
def test_the_bandwidth_arms_straddle_k_wrap(bandwidth):
    spec = SHIFT_SPECS[f"G4-bandwidth-{bandwidth}"]
    assert spec.config.initial_bandwidth == bandwidth
    assert spec.config.initial_bandwidth <= spec.config.max_wave_number


def test_the_fixed_alpha_arm_has_zero_width():
    low, high = SHIFT_SPECS["G7-alpha-fixed"].config.alpha_range
    assert low == high


def test_no_arm_collides_with_the_production_dataset_hash():
    production = config_hash(BASE)
    for name, spec in SHIFT_SPECS.items():
        assert shift_identifier(spec) != production, name


def test_only_the_g7_arm_carries_a_training_split():
    """The G1-G4 arms are evaluation sets; a nonzero n_train there would silently
    generate 800 unused trajectories per arm."""

    for name, spec in SHIFT_SPECS.items():
        if name == "G7-alpha-fixed":
            assert spec.config.n_train > 0
        else:
            assert spec.config.n_train == 0, name
            assert spec.config.n_val == 0, name


def test_every_arm_names_a_potential_family_the_generator_accepts():
    from spno.data.generate import PotentialFamily
    from typing import get_args

    valid = set(get_args(PotentialFamily))
    for name, spec in SHIFT_SPECS.items():
        assert spec.potential_family in valid, (name, spec.potential_family)


def test_a_spec_reaches_the_generator_and_its_family_is_recorded():
    """Integration check at tiny size: the family must survive the call, not just
    typecheck.  A spec whose family is dropped would silently evaluate G3 on random
    potentials and report it as a potential-family result."""

    from dataclasses import replace

    from spno.data.datasets import generate_shard

    spec = SHIFT_SPECS["G3-potential-cosine"]
    shard = generate_shard(
        replace(spec.config, n_test=2, steps=3),
        "test",
        potential_family=spec.potential_family,
    )
    assert shard.trajectories.shape == (2, 4, 64)
    assert shard.metadata["potential_family"] == "cosine"
