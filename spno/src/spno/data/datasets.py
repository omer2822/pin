"""Trajectory shards, splits, and the two training views over them.

Split hygiene is enforced structurally rather than by convention: each split is
generated from its own seed and carries its own trajectory ids, so a frame can never
appear in two splits.  Training windows are drawn *within* a trajectory, and
trajectories never cross the split boundary.

Storage is complex128.  Training casts to complex64 at load time; keeping the shards
in float64 means the reference data itself is never the limiting precision, and the
invariant measurements in later phases stay meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset

from ..config import DataConfig
from ..domain import PeriodicDomain, l2_mass
from ..equations.nls import wrap_wavenumber
from ..seeding import seed_everything
from ..solvers.split_step import SubsteppedReference
from .generate import (
    energy_fraction_above,
    sample_initial_conditions,
    sample_parameters,
    sample_potentials,
)

Tensor = torch.Tensor

SPLIT_SEED_OFFSET = {"train": 0, "val": 10_000, "test": 20_000}


@dataclass
class TrajectoryShard:
    """One split's worth of trajectories plus the parameters that produced them."""

    trajectories: Tensor  # (n, frames, *shape) complex128
    potential: Tensor  # (n, *shape) float64
    alpha: Tensor  # (n,) float64
    beta: Tensor  # (n,) float64
    trajectory_ids: Tensor  # (n,) int64 -- globally unique across splits
    dt: float
    split: str
    metadata: dict = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        n, frames = self.trajectories.shape[0], self.trajectories.shape[1]
        if self.potential.shape[0] != n or self.alpha.shape != (n,):
            raise ValueError("shard components must agree on the trajectory count")
        if not self.trajectories.is_complex():
            raise ValueError("trajectories must be complex")
        if frames < 2:
            raise ValueError("a trajectory needs at least two frames")

    @property
    def n_trajectories(self) -> int:
        return self.trajectories.shape[0]

    @property
    def n_frames(self) -> int:
        return self.trajectories.shape[1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "trajectories": self.trajectories,
                "potential": self.potential,
                "alpha": self.alpha,
                "beta": self.beta,
                "trajectory_ids": self.trajectory_ids,
                "dt": self.dt,
                "split": self.split,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "TrajectoryShard":
        payload = torch.load(path, weights_only=False)
        return cls(**payload)


def generate_shard(
    config: DataConfig,
    split: str,
    *,
    potential_family: str = "random",
    steps: int | None = None,
    reference: "torch.nn.Module | None" = None,
) -> TrajectoryShard:
    """Generate one split.  Deterministic in ``(config.seed, split)``.

    ``reference`` injects the trajectory generator, which is how Phase 9 produces
    misspecified data without touching :class:`DataConfig` (whose hash names 206 MB of
    shards on disk).  Passing ``None`` builds the standard substepped reference, so
    every existing caller is unaffected and a dial-zero generator reproduces the
    production shards **bitwise**.
    """

    if split not in SPLIT_SEED_OFFSET:
        raise ValueError(f"split must be one of {sorted(SPLIT_SEED_OFFSET)}")
    counts = {"train": config.n_train, "val": config.n_val, "test": config.n_test}
    n = counts[split]
    steps = config.steps if steps is None else steps

    generator = seed_everything(config.seed + SPLIT_SEED_OFFSET[split])
    domain = config.domain

    field = sample_initial_conditions(
        domain, n, config.initial_bandwidth, config.mass_range, generator
    )
    potential = sample_potentials(
        domain,
        n,
        config.potential_amplitude_range,
        config.potential_correlation_length,
        generator,
        family=potential_family,
    )
    alpha, beta = sample_parameters(
        n, config.alpha_range, config.beta_range, generator
    )

    reference = (
        SubsteppedReference(domain, config.substeps)
        if reference is None
        else reference
    )
    k_wrap = wrap_wavenumber(config.alpha_range[1], config.dt)
    cascade_steps, above_train, above_wrap = [], [], []

    frames = [field]
    evolved = field
    with torch.no_grad():
        for step in range(steps):
            evolved = reference(evolved, potential, alpha, beta, config.dt)
            frames.append(evolved)
            if step + 1 in (1, 50, 100, steps):
                cascade_steps.append(step + 1)
                above_train.append(
                    float(
                        energy_fraction_above(
                            evolved, domain, config.initial_bandwidth
                        ).max()
                    )
                )
                above_wrap.append(
                    float(energy_fraction_above(evolved, domain, k_wrap).max())
                )

    trajectories = torch.stack(frames, dim=1)
    masses = l2_mass(field, domain)
    # Globally unique ids so a leakage check is a set intersection, not a convention.
    ids = torch.arange(n, dtype=torch.int64) + SPLIT_SEED_OFFSET[split]

    return TrajectoryShard(
        trajectories=trajectories,
        potential=potential,
        alpha=alpha,
        beta=beta,
        trajectory_ids=ids,
        dt=config.dt,
        split=split,
        metadata={
            "grid_size": config.grid_size,
            "substeps": config.substeps,
            "steps": steps,
            "potential_family": potential_family,
            "initial_bandwidth": config.initial_bandwidth,
            "seed": config.seed + SPLIT_SEED_OFFSET[split],
            "mass_min": float(masses.min()),
            "mass_max": float(masses.max()),
            "mass_ratio": float(masses.max() / masses.min()),
            "alpha_min": float(alpha.min()),
            "alpha_max": float(alpha.max()),
            "alpha_max_gap": float(torch.diff(alpha.sort().values).max()),
            "beta_min": float(beta.min()),
            "beta_max": float(beta.max()),
            "k_wrap": k_wrap,
            "cascade_steps": cascade_steps,
            "cascade_above_k_train": above_train,
            "cascade_above_k_wrap": above_wrap,
        },
    )


class OneStepDataset(Dataset):
    """Every consecutive frame pair: ``(psi_n, V, alpha, beta) -> psi_{n+1}``.

    Fields are returned as real ``(2, N)`` channel stacks, the layout every model in
    the project consumes, and cast to ``dtype`` (complex64/float32 for training).
    """

    def __init__(self, shard: TrajectoryShard, *, dtype: torch.dtype = torch.float32):
        self.shard = shard
        self.dtype = dtype
        self.pairs_per_trajectory = shard.n_frames - 1

    def __len__(self) -> int:
        return self.shard.n_trajectories * self.pairs_per_trajectory

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        trajectory, frame = divmod(index, self.pairs_per_trajectory)
        current = self.shard.trajectories[trajectory, frame]
        target = self.shard.trajectories[trajectory, frame + 1]
        return {
            "psi": as_channels(current).to(self.dtype),
            "target": as_channels(target).to(self.dtype),
            "potential": self.shard.potential[trajectory].to(self.dtype),
            "alpha": self.shard.alpha[trajectory].to(self.dtype),
            "beta": self.shard.beta[trajectory].to(self.dtype),
            "trajectory_id": self.shard.trajectory_ids[trajectory],
        }


class RolloutDataset(Dataset):
    """Windows of ``horizon`` consecutive targets, for rollout training and evaluation."""

    def __init__(
        self,
        shard: TrajectoryShard,
        horizon: int,
        *,
        dtype: torch.dtype = torch.float32,
        stride: int = 1,
    ):
        if horizon < 1 or horizon >= shard.n_frames:
            raise ValueError("horizon must satisfy 1 <= horizon < n_frames")
        self.shard = shard
        self.horizon = horizon
        self.dtype = dtype
        self.stride = stride
        self.starts_per_trajectory = (shard.n_frames - horizon - 1) // stride + 1

    def __len__(self) -> int:
        return self.shard.n_trajectories * self.starts_per_trajectory

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        trajectory, offset = divmod(index, self.starts_per_trajectory)
        start = offset * self.stride
        window = self.shard.trajectories[trajectory, start : start + self.horizon + 1]
        return {
            "psi": as_channels(window[0]).to(self.dtype),
            "targets": torch.stack([as_channels(f) for f in window[1:]]).to(self.dtype),
            "potential": self.shard.potential[trajectory].to(self.dtype),
            "alpha": self.shard.alpha[trajectory].to(self.dtype),
            "beta": self.shard.beta[trajectory].to(self.dtype),
            "trajectory_id": self.shard.trajectory_ids[trajectory],
        }


def as_channels(field: Tensor) -> Tensor:
    """Complex ``(..., N)`` -> real ``(..., 2, N)`` with real and imaginary parts."""

    return torch.stack((field.real, field.imag), dim=-2)


def as_complex(channels: Tensor) -> Tensor:
    """Inverse of :func:`as_channels`."""

    if channels.shape[-2] != 2:
        raise ValueError("expected a channel axis of size 2 before the spatial axes")
    return torch.complex(channels[..., 0, :], channels[..., 1, :])


def shard_paths(root: Path, config_identifier: str) -> dict[str, Path]:
    return {
        split: root / f"nls1d-{config_identifier}" / f"{split}.pt"
        for split in SPLIT_SEED_OFFSET
    }


def assert_no_leakage(shards: dict[str, TrajectoryShard]) -> None:
    """Trajectory ids must be pairwise disjoint across splits."""

    seen: dict[int, str] = {}
    for split, shard in shards.items():
        for identifier in shard.trajectory_ids.tolist():
            if identifier in seen:
                raise AssertionError(
                    f"trajectory {identifier} appears in both {seen[identifier]} and {split}"
                )
            seen[identifier] = split


def iter_batches(
    dataset: Dataset, batch_size: int, generator: torch.Generator, *, shuffle: bool = True
) -> Iterator[dict[str, Tensor]]:
    """Minimal collating iterator; avoids DataLoader worker overhead for in-memory shards.

    Correct but per-sample, so it is used in tests and small runs.  Training uses
    :class:`OneStepBatches`, which is device-resident and vectorized.
    """

    order = (
        torch.randperm(len(dataset), generator=generator)
        if shuffle
        else torch.arange(len(dataset))
    )
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        samples = [dataset[int(i)] for i in indices]
        yield {key: torch.stack([s[key] for s in samples]) for key in samples[0]}


class RolloutBatches:
    """Device-resident windows of ``horizon`` consecutive targets.

    The rollout-training counterpart of :class:`OneStepBatches`.  Windows never cross a
    trajectory boundary: ``start + horizon`` is capped per trajectory.
    """

    def __init__(
        self,
        shard: TrajectoryShard,
        horizon: int,
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.complex64,
        stride: int = 1,
    ) -> None:
        if horizon < 1 or horizon >= shard.n_frames:
            raise ValueError("horizon must satisfy 1 <= horizon < n_frames")
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        self.trajectories = shard.trajectories.to(device=device, dtype=dtype)
        self.potential = shard.potential.to(device=device, dtype=real_dtype)
        self.alpha = shard.alpha.to(device=device, dtype=real_dtype)
        self.beta = shard.beta.to(device=device, dtype=real_dtype)
        self.horizon = horizon
        self.device = device
        n = shard.n_trajectories
        starts = torch.arange(0, shard.n_frames - horizon, stride, device=device)
        self.trajectory_index = torch.arange(n, device=device).repeat_interleave(
            starts.numel()
        )
        self.start_index = starts.repeat(n)

    def __len__(self) -> int:
        return self.trajectory_index.numel()

    def subset(self, indices: Tensor) -> "RolloutBatches":
        """Restrict to a subset of windows, for sample-efficiency curves.

        Mirrors :meth:`OneStepBatches.subset`.  Without it ``max_train_pairs`` would be
        silently ignored in rollout mode and every point on the sample-efficiency curve
        would secretly use the full split.
        """

        clone = object.__new__(RolloutBatches)
        clone.__dict__.update(self.__dict__)
        clone.trajectory_index = self.trajectory_index[indices.to(self.device)]
        clone.start_index = self.start_index[indices.to(self.device)]
        return clone

    def batches(
        self, batch_size: int, generator: torch.Generator, *, shuffle: bool = True
    ) -> Iterator[dict[str, Tensor]]:
        count = len(self)
        order = (
            torch.randperm(count, generator=generator).to(self.device)
            if shuffle
            else torch.arange(count, device=self.device)
        )
        offsets = torch.arange(1, self.horizon + 1, device=self.device)
        for start in range(0, count, batch_size):
            selected = order[start : start + batch_size]
            trajectory = self.trajectory_index[selected]
            frame = self.start_index[selected]
            target_frames = frame.unsqueeze(1) + offsets.unsqueeze(0)
            yield {
                "psi": self.trajectories[trajectory, frame],
                "targets": self.trajectories[trajectory.unsqueeze(1), target_frames],
                "potential": self.potential[trajectory],
                "alpha": self.alpha[trajectory],
                "beta": self.beta[trajectory],
            }


class OneStepBatches:
    """Device-resident, vectorized view of all consecutive frame pairs.

    A shard holds 160k pairs; assembling those one dict at a time dominates the step
    time on this hardware.  Here the whole split lives on the device as a single
    ``(n_trajectories, frames, N)`` complex tensor and a batch is one gather, so the
    measured cost is the model rather than the data pipeline.
    """

    def __init__(
        self,
        shard: TrajectoryShard,
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        self.shard_ids = shard.trajectory_ids
        self.trajectories = shard.trajectories.to(device=device, dtype=dtype)
        self.potential = shard.potential.to(device=device, dtype=real_dtype)
        self.alpha = shard.alpha.to(device=device, dtype=real_dtype)
        self.beta = shard.beta.to(device=device, dtype=real_dtype)
        self.device = device
        n, frames = shard.n_trajectories, shard.n_frames
        pairs = frames - 1
        self.trajectory_index = (
            torch.arange(n, device=device).repeat_interleave(pairs)
        )
        self.frame_index = torch.arange(pairs, device=device).repeat(n)

    def __len__(self) -> int:
        return self.trajectory_index.numel()

    def subset(self, indices: Tensor) -> "OneStepBatches":
        """Restrict to a subset of pairs, for sample-efficiency curves."""

        clone = object.__new__(OneStepBatches)
        clone.__dict__.update(self.__dict__)
        clone.trajectory_index = self.trajectory_index[indices.to(self.device)]
        clone.frame_index = self.frame_index[indices.to(self.device)]
        return clone

    def batches(
        self, batch_size: int, generator: torch.Generator, *, shuffle: bool = True
    ) -> Iterator[dict[str, Tensor]]:
        count = len(self)
        order = (
            torch.randperm(count, generator=generator).to(self.device)
            if shuffle
            else torch.arange(count, device=self.device)
        )
        for start in range(0, count, batch_size):
            selected = order[start : start + batch_size]
            trajectory = self.trajectory_index[selected]
            frame = self.frame_index[selected]
            yield {
                "psi": self.trajectories[trajectory, frame],
                "target": self.trajectories[trajectory, frame + 1],
                "potential": self.potential[trajectory],
                "alpha": self.alpha[trajectory],
                "beta": self.beta[trajectory],
            }


class MultiDtBatches:
    """One-step pairs drawn from several shards, each generated at its own ``dt``.

    The G6a arm: multi-dt supervision as "a second route past ``k_wrap``".  At fixed
    ``alpha`` the one-step multiplier determines ``omega`` only modulo ``2 pi / dt``,
    so observing the *same* physical mode at several ``dt`` breaks the degeneracy the
    same way varying ``alpha`` does.

    Every batch stays **single-dt** and carries its step size as ``batch["dt"]``.  That
    is what keeps the model interface unchanged: ``step`` still takes one scalar ``dt``,
    and no architecture has to learn a batched step size.

    *Architectural note, not an optimization detail.*  The FNO ignores its ``dt``
    argument entirely (``fno.py`` never reads it), so it cannot represent the
    dependence this arm supervises.  Its G6a result is therefore a statement about the
    hypothesis class, not about optimization -- report it that way.
    """

    def __init__(
        self,
        shards: dict[float, TrajectoryShard],
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        if not shards:
            raise ValueError("MultiDtBatches needs at least one (dt, shard) pair")
        self.device = device
        self.views = {
            float(dt): OneStepBatches(shard, device=device, dtype=dtype)
            for dt, shard in shards.items()
        }

    def __len__(self) -> int:
        return sum(len(view) for view in self.views.values())

    def batches(
        self, batch_size: int, generator: torch.Generator, *, shuffle: bool = True
    ) -> Iterator[dict[str, Tensor]]:
        """Interleave the per-dt streams in a generator-determined order.

        The schedule is permuted rather than the pooled pairs, so each yielded batch
        stays single-dt while the *order* of dt values is still randomized.  Every pair
        appears exactly once per epoch: a schedule that dropped or repeated a sub-shard
        would silently reweight the dt distribution, and the G6a result would then be
        about sampling rather than architecture.
        """

        iterators = {
            dt: view.batches(batch_size, generator, shuffle=shuffle)
            for dt, view in self.views.items()
        }
        schedule: list[float] = []
        for dt, view in self.views.items():
            count = len(view)
            schedule.extend([dt] * ((count + batch_size - 1) // batch_size))
        if shuffle:
            order = torch.randperm(len(schedule), generator=generator).tolist()
            schedule = [schedule[index] for index in order]
        for dt in schedule:
            batch = next(iterators[dt])
            batch["dt"] = dt
            yield batch
