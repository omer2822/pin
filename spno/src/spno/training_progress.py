"""Durable, epoch-boundary training recovery for trusted local artifacts."""
from __future__ import annotations

from dataclasses import asdict
import os
import json
from pathlib import Path
import random
import shutil
import uuid

import numpy as np
import torch


def atomic_torch_save(payload, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Validate the completed publication before replacing anything.
        torch.load(temporary, map_location='cpu', weights_only=False)
        if path.exists():
            try:
                torch.load(path, map_location='cpu', weights_only=False)
            except Exception:
                pass  # Preserve the last known-good backup if current is corrupt.
            else:
                backup_tmp = temporary.with_suffix('.backup')
                shutil.copyfile(path, backup_tmp)
                os.replace(backup_tmp, path.with_name(path.name + '.previous'))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def recover_torch_load(path: Path):
    path = Path(path)
    failures = []
    for candidate in (path, path.with_name(path.name + '.previous')):
        if not candidate.exists():
            continue
        try:
            return torch.load(candidate, map_location='cpu', weights_only=False)
        except Exception as error:
            failures.append(f'{candidate}: {error}')
    raise RuntimeError(f'No readable training artifact: {path}; ' + '; '.join(failures))


class TrainingProgress:
    def __init__(self, path: Path, experiment_id: str):
        self.path = Path(path)
        self.experiment_id = experiment_id

    def restore(self, model, optimizer, scheduler, generator, context):
        if not self.path.exists() and not self.path.with_name(self.path.name + '.previous').exists():
            return None
        state = recover_torch_load(self.path)
        if (state.get('schema_version') != 1 or state.get('experiment_id') != self.experiment_id
                or state.get('context') != json.loads(json.dumps(context))):
            raise RuntimeError(f'Training progress is incompatible: {self.path}')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        generator.set_state(state['generator'])
        torch.set_rng_state(state['torch_rng'])
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        if state['cuda_rng'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        return state

    def save(self, *, model, optimizer, scheduler, generator, context, history,
             best_state, epoch, complete, selected_indices=None):
        atomic_torch_save({
            'schema_version': 1, 'experiment_id': self.experiment_id,
            'context': json.loads(json.dumps(context)), 'epoch': epoch, 'complete': complete,
            'model': model.state_dict(), 'best_state': best_state,
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'history': asdict(history), 'selected_indices': selected_indices,
            'epochs_without_improvement': epoch - history.best_epoch,
            'generator': generator.get_state(), 'torch_rng': torch.get_rng_state(),
            'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, self.path)
