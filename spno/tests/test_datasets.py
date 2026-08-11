"""Phase 1: the dataset must have the properties the experiment design assumes.

Every test here guards a specific confound named in the plan.  A dataset that quietly
violates one of these produces plausible-looking numbers that answer the wrong
question, which is the most expensive kind of bug in this project.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from spno.config import DataConfig
from spno.data.datasets import (
    OneStepDataset,
    RolloutDataset,
    as_channels,
    as_complex,
    assert_no_leakage,
    generate_shard,
)
from spno.data.generate import energy_fraction_above, sample_potentials
from spno.domain import PeriodicDomain, l2_mass
from spno.equations.nls import alpha_sampling_is_dense_enough, wrap_wavenumber

SMALL = DataConfig(n_train=12, n_val=6, n_test=6, steps=8)


@pytest.fixture(scope="module")
def shards():
    return {split: generate_shard(SMALL, split) for split in ("train", "val", "test")}


# --------------------------------------------------------------------------------
# Split hygiene
# --------------------------------------------------------------------------------


def test_splits_share_no_trajectories(shards):
    assert_no_leakage(shards)


def test_leakage_check_actually_catches_a_collision(shards):
    """Guard against a vacuous leakage test."""

    colliding = dict(shards)
    duplicate = dataclasses.replace(shards["val"], split="test")
    duplicate.trajectory_ids = shards["train"].trajectory_ids[: duplicate.n_trajectories]
    colliding["test"] = duplicate

    with pytest.raises(AssertionError, match="appears in both"):
        assert_no_leakage(colliding)


def test_splits_have_distinct_initial_conditions(shards):
    """Different seeds must produce genuinely different fields, not just different ids."""

    train_initial = shards["train"].trajectories[:, 0]
    val_initial = shards["val"].trajectories[:, 0]
    closest = torch.cdist(
        as_channels(train_initial).flatten(1), as_channels(val_initial).flatten(1)
    ).min()
    assert float(closest) > 1e-3


# --------------------------------------------------------------------------------
# Distribution properties the generalization design depends on
# --------------------------------------------------------------------------------


def test_initial_conditions_are_exactly_band_limited(shards):
    """G4 compares training band against test bands; leaked high-k energy voids it."""

    domain = SMALL.domain
    initial = shards["train"].trajectories[:, 0]

    above = energy_fraction_above(initial, domain, SMALL.initial_bandwidth)

    assert float(above.max()) < 1e-28


def test_mass_varies_across_samples(shards):
    """Constant mass would make conservation memorizable and Model B's projection a no-op."""

    masses = l2_mass(shards["train"].trajectories[:, 0], SMALL.domain)

    assert float(masses.max() / masses.min()) > 1.5
    assert float(masses.min()) >= SMALL.mass_range[0] - 1e-9
    assert float(masses.max()) <= SMALL.mass_range[1] + 1e-9


def test_alpha_sampling_is_dense_enough_for_the_identifiability_arm():
    """G5b needs d(arg m)/d(alpha) to be recoverable without unwrapping at every k."""

    config = DataConfig()
    shard = generate_shard(dataclasses.replace(config, n_train=400, steps=1), "train")
    gap = float(torch.diff(shard.alpha.sort().values).max())

    assert alpha_sampling_is_dense_enough(gap, config.max_wave_number, config.dt)


def test_wrap_horizon_sits_inside_the_grid():
    """The design requires k_train < k_wrap < k_nyquist so G4 can straddle the horizon."""

    config = DataConfig()
    k_wrap = wrap_wavenumber(config.alpha_range[1], config.dt)

    assert config.initial_bandwidth < k_wrap < config.max_wave_number


def test_potential_families_are_distinguishable():
    domain = PeriodicDomain.periodic_1d(32)
    generator = torch.Generator().manual_seed(0)
    families = {
        name: sample_potentials(domain, 4, (0.5, 0.5), 1.0, generator, family=name)
        for name in ("zero", "cosine", "gaussian_well", "harmonic", "random")
    }

    assert torch.all(families["zero"] == 0)
    for name, values in families.items():
        if name != "zero":
            peak = values.abs().amax(dim=-1)
            assert torch.allclose(peak, torch.full_like(peak, 0.5), atol=1e-9)


def test_non_random_potential_families_refuse_higher_dimensions():
    """Silently dropping the y-dependence would be worse than an error."""

    domain = PeriodicDomain((8, 8), (6.283185307179586, 6.283185307179586))
    generator = torch.Generator().manual_seed(0)

    with pytest.raises(NotImplementedError):
        sample_potentials(domain, 2, (0.1, 0.5), 1.0, generator, family="cosine")
    sample_potentials(domain, 2, (0.1, 0.5), 1.0, generator, family="random")


# --------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------


def test_regeneration_is_bit_identical():
    first = generate_shard(SMALL, "val")
    second = generate_shard(SMALL, "val")

    assert torch.equal(first.trajectories, second.trajectories)
    assert torch.equal(first.potential, second.potential)
    assert torch.equal(first.alpha, second.alpha)


def test_changing_the_seed_changes_the_data():
    other = generate_shard(dataclasses.replace(SMALL, seed=SMALL.seed + 1), "val")
    base = generate_shard(SMALL, "val")

    assert not torch.equal(base.trajectories, other.trajectories)


def test_shard_round_trips_through_disk(tmp_path, shards):
    path = tmp_path / "val.pt"
    shards["val"].save(path)

    from spno.data.datasets import TrajectoryShard

    loaded = TrajectoryShard.load(path)

    assert torch.equal(loaded.trajectories, shards["val"].trajectories)
    assert loaded.metadata == shards["val"].metadata


# --------------------------------------------------------------------------------
# Dataset views
# --------------------------------------------------------------------------------


def test_one_step_dataset_pairs_consecutive_frames(shards):
    shard = shards["val"]
    dataset = OneStepDataset(shard, dtype=torch.float64)

    assert len(dataset) == shard.n_trajectories * (shard.n_frames - 1)

    sample = dataset[shard.n_frames - 1]  # first pair of trajectory 1
    assert torch.allclose(as_complex(sample["psi"]), shard.trajectories[1, 0])
    assert torch.allclose(as_complex(sample["target"]), shard.trajectories[1, 1])
    assert sample["trajectory_id"] == shard.trajectory_ids[1]


def test_one_step_dataset_never_crosses_a_trajectory_boundary(shards):
    """The last frame of trajectory i must not be paired with the first of i+1."""

    shard = shards["val"]
    dataset = OneStepDataset(shard, dtype=torch.float64)

    for index in range(len(dataset)):
        sample = dataset[index]
        trajectory = index // (shard.n_frames - 1)
        assert sample["trajectory_id"] == shard.trajectory_ids[trajectory]


def test_rollout_dataset_windows_are_consecutive(shards):
    shard = shards["val"]
    horizon = 3
    dataset = RolloutDataset(shard, horizon, dtype=torch.float64)

    sample = dataset[0]
    assert sample["targets"].shape[0] == horizon
    for step in range(horizon):
        assert torch.allclose(
            as_complex(sample["targets"][step]), shard.trajectories[0, step + 1]
        )


def test_rollout_horizon_must_fit_in_the_trajectory(shards):
    with pytest.raises(ValueError):
        RolloutDataset(shards["val"], shards["val"].n_frames, dtype=torch.float64)


def test_channel_conversion_round_trips():
    field = torch.randn(3, 16, dtype=torch.complex128)

    assert torch.equal(as_complex(as_channels(field)), field)
    assert as_channels(field).shape == (3, 2, 16)


def test_trajectories_conserve_mass_along_the_stored_frames(shards):
    """The stored data must itself satisfy the invariant models are judged against."""

    shard = shards["train"]
    masses = l2_mass(shard.trajectories, SMALL.domain)

    drift = torch.abs(masses / masses[:, :1] - 1)
    assert float(drift.max()) < 1e-13
