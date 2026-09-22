"""Training loop shared by every model in the comparison.

Deliberately plain: one objective, one schedule, one early-stopping rule, applied
identically to every architecture.  Any performance difference between models must be
attributable to the models, not to per-model tuning.

Normalization statistics come from the training split only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field as dataclass_field
import time

import torch

from .config import DataConfig
from .data.datasets import (
    MultiDtBatches,
    OneStepBatches,
    RolloutBatches,
    TrajectoryShard,
)
from .domain import PeriodicDomain, l2_mass
from .losses.pde_residual import residual_loss
from .losses.relative_l2 import relative_l2_loss
from .seeding import seed_everything
from .training_progress import TrainingProgress

Tensor = torch.Tensor


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 5
    seed: int = 0
    device: str = "cpu"
    max_train_pairs: int | None = None
    rollout_horizon: int = 1
    log_every: int = 1


@dataclass
class TrainHistory:
    train_loss: list[float] = dataclass_field(default_factory=list)
    val_loss: list[float] = dataclass_field(default_factory=list)
    best_val: float = float("inf")
    best_epoch: int = -1
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "best_val": self.best_val,
            "best_epoch": self.best_epoch,
            "seconds": self.seconds,
        }


def field_scale(shard: TrajectoryShard, domain: PeriodicDomain) -> float:
    """A single global amplitude scale, computed on the training split only.

    Not per-sample: the nonlinearity is amplitude-dependent and mass varies by design,
    so per-sample normalization would erase the very signal the model needs.
    """

    initial = shard.trajectories[:, 0]
    return float(torch.sqrt(l2_mass(initial, domain).mean() / domain.cell_volume
                            / initial.shape[-1]))


def _rollout_loss(model, batch, domain, dt, horizon: int) -> Tensor:
    """Mean relative-L2 over ``horizon`` autoregressive steps.

    The second training mode.  It must be reported separately from one-step training
    and never pooled with it: rollout training can substantially repair the
    unconstrained baseline's drift, and averaging the two modes together would erase
    exactly the contrast the study is measuring.
    """

    state = batch["psi"]
    total = 0.0
    for step in range(horizon):
        state = model(
            state, batch["potential"], batch["alpha"], batch["beta"], dt
        )
        total = total + relative_l2_loss(state, batch["targets"][:, step], domain)
    return total / horizon


def _one_step_loss(model, batch, domain, dt) -> Tensor:
    """Relative-L2 on one step, at the batch's own step size when it carries one.

    ``batch.get("dt", dt)`` is backward compatible by construction: neither
    ``OneStepBatches`` nor ``RolloutBatches`` emits a ``"dt"`` key (verified), so every
    existing caller keeps using the config's ``dt``.  Only ``MultiDtBatches`` sets it.
    """

    step = float(batch.get("dt", dt))
    prediction = model(
        batch["psi"], batch["potential"], batch["alpha"], batch["beta"], step
    )
    return relative_l2_loss(prediction, batch["target"], domain)


@torch.no_grad()
def evaluate_one_step(
    model, batches: OneStepBatches, domain, dt, config: TrainConfig
) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(0)
    total, count = 0.0, 0
    for batch in batches.batches(config.batch_size, generator, shuffle=False):
        loss = _one_step_loss(model, batch, domain, dt)
        weight = batch["psi"].shape[0]
        total += float(loss.detach()) * weight
        count += weight
    return total / max(count, 1)


def train_one_step(
    model,
    train_shard: TrajectoryShard,
    val_shard: TrajectoryShard,
    data_config: DataConfig,
    config: TrainConfig,
    *,
    verbose: bool = True,
    progress: TrainingProgress | None = None,
) -> TrainHistory:
    """Train on consecutive frame pairs with relative-L2, early stopping on val."""

    domain = data_config.domain
    dt = data_config.dt
    generator = seed_everything(config.seed)
    model.to(config.device)

    train_batches = OneStepBatches(train_shard, device=config.device)
    val_batches = OneStepBatches(val_shard, device=config.device)
    if config.max_train_pairs is not None:
        chosen = torch.randperm(len(train_batches), generator=generator)[
            : config.max_train_pairs
        ]
        train_batches = train_batches.subset(chosen)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.epochs, 1)
    )

    history = TrainHistory()
    best_state = None
    started = time.time()
    elapsed_before = 0.0
    start_epoch = 0
    context = {"data": asdict(data_config), "train": asdict(config), "mode": 'one-step'}
    context["train"].pop("device")
    context["train"].pop("log_every")
    if progress is not None:
        saved = progress.restore(model, optimizer, scheduler, generator, context)
        if saved is not None:
            history = TrainHistory(**saved["history"])
            best_state = saved["best_state"]
            elapsed_before = history.seconds
            start_epoch = config.epochs if saved["complete"] else saved["epoch"] + 1

    for epoch in range(start_epoch, config.epochs):
        model.train()
        running, seen = 0.0, 0
        for batch in train_batches.batches(config.batch_size, generator):
            optimizer.zero_grad()
            loss = _one_step_loss(model, batch, domain, dt)
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            weight = batch["psi"].shape[0]
            running += float(loss.detach()) * weight
            seen += weight
        scheduler.step()

        train_loss = running / max(seen, 1)
        val_loss = evaluate_one_step(model, val_batches, domain, dt, config)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)

        if val_loss < history.best_val:
            history.best_val = val_loss
            history.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % config.log_every == 0 or epoch == config.epochs - 1):
            print(
                f"  epoch {epoch:3d}  train {train_loss:.4e}  val {val_loss:.4e}"
                + ("  *" if history.best_epoch == epoch else "")
            )

        if progress is not None:
            history.seconds = elapsed_before + time.time() - started
            progress.save(
                model=model, optimizer=optimizer, scheduler=scheduler, generator=generator,
                context=context, history=history, best_state=best_state, epoch=epoch,
                complete=epoch + 1 >= config.epochs or epoch - history.best_epoch >= config.patience,
                selected_indices=locals().get("chosen"),
            )
        if epoch - history.best_epoch >= config.patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (patience {config.patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history.seconds = elapsed_before + time.time() - started
    return history


def train_rollout(
    model,
    train_shard: TrajectoryShard,
    val_shard: TrajectoryShard,
    data_config: DataConfig,
    config: TrainConfig,
    *,
    horizon: int = 4,
    verbose: bool = True,
) -> TrainHistory:
    """Short-horizon rollout training: the second training mode.

    Identical optimizer, schedule, and budget to :func:`train_one_step` so the two
    modes stay comparable.  Validation is still measured as one-step error, so the two
    modes report on the same scale; the rollout metrics come from the evaluation suite.

    Results from this mode must be reported separately from one-step training.  Rollout
    training can repair much of an unconstrained model's drift, and pooling the modes
    would hide the very contrast the study exists to measure.
    """

    domain = data_config.domain
    dt = data_config.dt
    generator = seed_everything(config.seed)
    model.to(config.device)

    train_batches = RolloutBatches(train_shard, horizon, device=config.device)
    val_batches = OneStepBatches(val_shard, device=config.device)
    # Kept in sync with train_one_step deliberately: without this, max_train_pairs was
    # honoured in one-step mode and silently ignored here, so every point on a rollout
    # sample-efficiency curve would have used the full split.
    if config.max_train_pairs is not None:
        chosen = torch.randperm(len(train_batches), generator=generator)[
            : config.max_train_pairs
        ]
        train_batches = train_batches.subset(chosen)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.epochs, 1)
    )

    history = TrainHistory()
    best_state = None
    started = time.time()

    for epoch in range(config.epochs):
        model.train()
        running, seen = 0.0, 0
        for batch in train_batches.batches(config.batch_size, generator):
            optimizer.zero_grad()
            loss = _rollout_loss(model, batch, domain, dt, horizon)
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            weight = batch["psi"].shape[0]
            running += float(loss.detach()) * weight
            seen += weight
        scheduler.step()

        train_loss = running / max(seen, 1)
        val_loss = evaluate_one_step(model, val_batches, domain, dt, config)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)

        if val_loss < history.best_val:
            history.best_val = val_loss
            history.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % config.log_every == 0 or epoch == config.epochs - 1):
            print(
                f"  epoch {epoch:3d}  rollout {train_loss:.4e}  val(1-step) {val_loss:.4e}"
                + ("  *" if history.best_epoch == epoch else "")
            )

        if epoch - history.best_epoch >= config.patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (patience {config.patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history.seconds = time.time() - started
    return history


def train_multi_dt(
    model,
    train_batches: MultiDtBatches,
    val_batches: OneStepBatches,
    data_config: DataConfig,
    config: TrainConfig,
    *,
    verbose: bool = True,
    progress: TrainingProgress | None = None,
) -> TrainHistory:
    """One-step training over several ``dt`` at once: the G6a arm.

    Takes pre-built batches rather than shards, because the caller owns the mapping from
    ``dt`` to shard and that mapping is part of the arm's identity.  Optimizer, schedule,
    early stopping and budget are identical to :func:`train_one_step` so the two remain
    comparable.

    Validation is still one-step at the config's ``dt``, so the reported scale matches
    every other arm's.

    **Read the result as a statement about the hypothesis class.**  The FNO ignores its
    ``dt`` argument entirely, so it cannot represent dt-dependence no matter how it is
    optimized; a flat multi-dt result for Model A is the architecture speaking, not the
    optimizer.  Only the split-step family, where ``dt`` multiplies a learned rate, can
    use this supervision.

    **The model must be built with ``trained_dt=None``.**  ``StepOperator._check_dt``
    refuses any ``dt`` other than ``trained_dt``, which is exactly the guard that makes
    the G6b transfer measurement meaningful -- so it also, correctly, blocks multi-dt
    *training* on a model that claims a single trained ``dt``.  A model supervised
    across several step sizes has no single one, and ``trained_dt=None`` is the honest
    declaration of that.  Do **not** work around this with
    :func:`~spno.models.base.allow_dt_transfer`: that context manager licenses
    *evaluation* outside the trained ``dt``, and using it here would erase the
    distinction G6a and G6b exist to separate.
    """

    if getattr(model, "trained_dt", None) is not None:
        raise RuntimeError(
            f"{type(model).__name__} declares trained_dt={model.trained_dt}, but "
            "multi-dt supervision trains across several step sizes and no single one "
            "is correct. Construct the model with trained_dt=None for the G6a arm. "
            "Do not use allow_dt_transfer here -- that licenses evaluation outside the "
            "trained dt and would conflate G6a with G6b."
        )

    domain = data_config.domain
    dt = data_config.dt
    generator = seed_everything(config.seed)
    model.to(config.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.epochs, 1)
    )

    history = TrainHistory()
    best_state = None
    started = time.time()
    elapsed_before = 0.0
    start_epoch = 0
    context = {"data": asdict(data_config), "train": asdict(config), "mode": 'multi-dt'}
    context["train"].pop("device")
    context["train"].pop("log_every")
    if progress is not None:
        saved = progress.restore(model, optimizer, scheduler, generator, context)
        if saved is not None:
            history = TrainHistory(**saved["history"])
            best_state = saved["best_state"]
            elapsed_before = history.seconds
            start_epoch = config.epochs if saved["complete"] else saved["epoch"] + 1

    for epoch in range(start_epoch, config.epochs):
        model.train()
        running, seen = 0.0, 0
        for batch in train_batches.batches(config.batch_size, generator):
            optimizer.zero_grad()
            loss = _one_step_loss(model, batch, domain, dt)
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            weight = batch["psi"].shape[0]
            running += float(loss.detach()) * weight
            seen += weight
        scheduler.step()

        train_loss = running / max(seen, 1)
        val_loss = evaluate_one_step(model, val_batches, domain, dt, config)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)

        if val_loss < history.best_val:
            history.best_val = val_loss
            history.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % config.log_every == 0 or epoch == config.epochs - 1):
            print(
                f"  epoch {epoch:3d}  multi-dt {train_loss:.4e}  val(1-step) {val_loss:.4e}"
                + ("  *" if history.best_epoch == epoch else "")
            )

        if progress is not None:
            history.seconds = elapsed_before + time.time() - started
            progress.save(
                model=model, optimizer=optimizer, scheduler=scheduler, generator=generator,
                context=context, history=history, best_state=best_state, epoch=epoch,
                complete=epoch + 1 >= config.epochs or epoch - history.best_epoch >= config.patience,
                selected_indices=locals().get("chosen"),
            )
        if epoch - history.best_epoch >= config.patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (patience {config.patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history.seconds = elapsed_before + time.time() - started
    return history


def _pino_loss(model, batch, domain, dt, physics_weight: float) -> Tensor:
    """Data loss plus ``lambda`` times the Crank--Nicolson residual.

    At ``physics_weight == 0`` this returns the data term by the *same expression*
    :func:`_one_step_loss` uses, so the sweep's low end is bit-identical to
    :func:`train_one_step` and is a genuine control rather than a near-miss.
    """

    step = float(batch.get("dt", dt))
    prediction = model(
        batch["psi"], batch["potential"], batch["alpha"], batch["beta"], step
    )
    data = relative_l2_loss(prediction, batch["target"], domain)
    if physics_weight == 0.0:
        return data
    physics = residual_loss(
        batch["psi"],
        prediction,
        batch["potential"],
        domain,
        batch["alpha"],
        batch["beta"],
        step,
    )
    return data + physics_weight * physics


def train_pino(
    model,
    train_shard: TrajectoryShard,
    val_shard: TrajectoryShard,
    data_config: DataConfig,
    config: TrainConfig,
    *,
    physics_weight: float,
    verbose: bool = True,
    progress: TrainingProgress | None = None,
) -> TrainHistory:
    """One-step training with a soft PDE-residual penalty: Phase 7a.

    **The confound, restated where it is applied.**  The midpoint residual uses the
    Delfour--Fortin--Payre averaging, so its own exact plane-wave solution advances by
    the Cayley transform, whose modulus is *exactly* one.  This physics loss therefore
    smuggles in the very invariant the study compares methods on, and every 7a number
    must be reported with that said.  See :mod:`spno.losses.pde_residual`.

    **Validation is the data loss only.**  ``evaluate_one_step`` ignores the physics
    term, so every ``lambda`` in the sweep reports on the same scale and early stopping
    is not confounded by a penalty whose magnitude changes with the weight.  Reporting
    the composite objective as "validation loss" would make the sweep's y-axis a
    different quantity at every point.

    Report the **whole sweep**, never a tuned value: which ``lambda`` wins is itself the
    result, and a single tuned number hides whether the physics term helped at all.
    """

    if physics_weight < 0:
        raise RuntimeError("physics_weight must be non-negative")

    domain = data_config.domain
    dt = data_config.dt
    generator = seed_everything(config.seed)
    model.to(config.device)

    train_batches = OneStepBatches(train_shard, device=config.device)
    val_batches = OneStepBatches(val_shard, device=config.device)
    if config.max_train_pairs is not None:
        chosen = torch.randperm(len(train_batches), generator=generator)[
            : config.max_train_pairs
        ]
        train_batches = train_batches.subset(chosen)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.epochs, 1)
    )

    history = TrainHistory()
    best_state = None
    started = time.time()
    elapsed_before = 0.0
    start_epoch = 0
    context = {"data": asdict(data_config), "train": asdict(config), "mode": 'pino'}
    context["train"].pop("device")
    context["train"].pop("log_every")
    context["physics_weight"] = physics_weight
    if progress is not None:
        saved = progress.restore(model, optimizer, scheduler, generator, context)
        if saved is not None:
            history = TrainHistory(**saved["history"])
            best_state = saved["best_state"]
            elapsed_before = history.seconds
            start_epoch = config.epochs if saved["complete"] else saved["epoch"] + 1

    for epoch in range(start_epoch, config.epochs):
        model.train()
        running, seen = 0.0, 0
        for batch in train_batches.batches(config.batch_size, generator):
            optimizer.zero_grad()
            loss = _pino_loss(model, batch, domain, dt, physics_weight)
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            weight = batch["psi"].shape[0]
            running += float(loss.detach()) * weight
            seen += weight
        scheduler.step()

        train_loss = running / max(seen, 1)
        val_loss = evaluate_one_step(model, val_batches, domain, dt, config)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)

        if val_loss < history.best_val:
            history.best_val = val_loss
            history.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % config.log_every == 0 or epoch == config.epochs - 1):
            print(
                f"  epoch {epoch:3d}  pino(l={physics_weight:g}) {train_loss:.4e}  "
                f"val(data) {val_loss:.4e}"
                + ("  *" if history.best_epoch == epoch else "")
            )

        if progress is not None:
            history.seconds = elapsed_before + time.time() - started
            progress.save(
                model=model, optimizer=optimizer, scheduler=scheduler, generator=generator,
                context=context, history=history, best_state=best_state, epoch=epoch,
                complete=epoch + 1 >= config.epochs or epoch - history.best_epoch >= config.patience,
                selected_indices=locals().get("chosen"),
            )
        if epoch - history.best_epoch >= config.patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (patience {config.patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history.seconds = elapsed_before + time.time() - started
    return history
