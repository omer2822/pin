"""A shared experiment catalog: explicit preparation, training, and evaluation.

Completed legacy checkpoints are immutable inputs. New experiment identities include
training data bytes and training settings, never evaluation settings.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path

import torch

from .artifacts import atomic_json, file_digest, inspect_standalone, verified_copy
from .checkpoints import CheckpointMetadata, CheckpointPayload, load_checkpoint_payload
from .config import DataConfig, config_hash
from .data.datasets import TrajectoryShard, assert_no_leakage, shard_paths
from .experiments import converged, evaluate_model
from .misspecification import MisspecificationConfig
from .seeding import seed_everything
from .train import TrainConfig, field_scale, train_one_step, train_pino
from .training_progress import TrainingProgress, atomic_torch_save


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:20]


def protocol(config: TrainConfig) -> dict:
    value = asdict(config)
    for key in ('device', 'log_every'):
        value.pop(key)
    return value


#: Pointwise-local split steps: C1, and C1g (C1 with kappa_theta(0) = 0).
POINTWISE_C_FAMILY = ('C1', 'C1g')


def architecture(name: str, data: DataConfig, stored=None, *, kinetic='K0') -> dict:
    stored = stored or {}
    if name in ('A', 'B-loop', 'A-wide'):
        result = {'modes': int(stored.get('modes', 32 if name == 'A-wide' else 16)),
                  'width': int(stored.get('width', 64)), 'n_layers': int(stored.get('n_layers', 4)),
                  'use_coordinate_channel': bool(stored.get('use_coordinate_channel', False)),
                  'alpha_range': list(stored.get('alpha_range', data.alpha_range)),
                  'beta_range': list(stored.get('beta_range', data.beta_range))}
        if name == 'B-loop':
            result['projection'] = 'mass'
        return result
    pointwise = name in POINTWISE_C_FAMILY
    result = {'kinetic_mode': kinetic, 'local_mode': None if name == 'C3' else stored.get('local_mode', 'L0'),
              'width': int(stored.get('width', 32 if pointwise else 64))}
    if not pointwise:
        result.update(modes=int(stored.get('modes', 16)), n_layers=int(stored.get('n_layers', 4)))
    if name == 'C1g':
        # Only C1g carries the key, so every existing C1 identity is unchanged.
        result['kinetic_gauge'] = 'zero_mode'
    return result


@dataclass(frozen=True)
class Experiment:
    identifier: str
    name: str
    seed: int
    spec: dict
    paths: dict[str, Path]
    data_config: DataConfig
    train_config: TrainConfig
    physics_weight: float = 0.0

    def metadata(self, *, did_converge=False, best_epoch=-1):
        return CheckpointMetadata(1, self.name, self.spec['data_hash'], self.seed,
                                  self.spec['objective'], self.spec['field_scale'],
                                  self.data_config.dt, self.spec['architecture'],
                                  did_converge, best_epoch)


class Workflow:
    def __init__(self, source_root, output_root, *, data_config=None,
                 train_config=None, cache_root=None):
        self.source_root, self.output_root = Path(source_root), Path(output_root)
        if self.output_root.resolve() == self.source_root.resolve() or self.output_root.resolve().is_relative_to(self.source_root.resolve()):
            raise ValueError('Use a separate output_root; source artifacts are read-only')
        self.data_config, self.train_config, self._inventory = inspect_standalone(
            self.source_root, data_config, train_config)
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self._imports = {}
        self._digest_cache = {}
        # Evaluation-only transfers omit train.pt; fingerprints stay unknown (and
        # unrecorded) until both splits exist, so imports are never silently reused.
        self._source_fingerprints = None
        source_paths = shard_paths(self.source_root / 'data', config_hash(self.data_config))
        if all(source_paths[s].is_file() for s in ('train', 'val')):
            source_paths = self._paths(self.source_root / 'data', config_hash(self.data_config))
            self._source_fingerprints = {s: self._digest(source_paths[s]) for s in ('train', 'val')}
        # Freeze provenance across runtime restarts, not merely within this object.
        import_id = stable_hash(sorted(r['sha256'] for r in self._inventory))
        manifest_path = self.output_root / 'imports' / f'{import_id}.json'
        current_config = asdict(self.train_config) if self.train_config else None
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            self._source_fingerprints = manifest['data_fingerprints']
            declared = manifest.get('source_train_config')
            if declared is not None:
                original = TrainConfig(**declared)
                if self.train_config and protocol(original) != protocol(self.train_config):
                    raise ValueError('Source training protocol conflicts with the saved import; use prepare(train_config=...) for a new experiment')
                self.train_config = original
                for row in self._inventory:
                    row['training_config'] = asdict(replace(original, seed=row['seed']))
                    row['protocol_source'] = row['protocol_source'] or 'saved import declaration'
            elif current_config is not None:
                manifest['source_train_config'] = current_config
                atomic_json(manifest_path, manifest)
        elif self._source_fingerprints is not None:
            atomic_json(manifest_path, {'data_fingerprints': self._source_fingerprints,
                                       'source_train_config': current_config})
        self.seeds = sorted({r['seed'] for r in self._inventory if r['name'] == 'A'})
        self.quick = any(r['quick'] for r in self._inventory if r['name'] == 'A')

    def inventory(self):
        return list(self._inventory)

    def _paths(self, root, identifier, splits=('train', 'val', 'test')):
        paths = {s: p for s, p in shard_paths(Path(root), identifier).items() if s in splits}
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(f'Dataset missing: {path}; prepare this experiment first')
        if self.cache_root:
            paths = {split: verified_copy(path, self.cache_root / identifier / path.name)
                     for split, path in paths.items()}
        return paths

    def _digest(self, path):
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self._digest_cache:
            self._digest_cache[key] = file_digest(path)
        return self._digest_cache[key]

    def shards(self, job=None):
        paths = job.paths if job else self._paths(self.source_root / 'data', config_hash(self.data_config))
        shards = {s: TrajectoryShard.load(p) for s, p in paths.items()}
        assert_no_leakage(shards)
        return shards

    def _base_row(self, name, seed):
        candidates = [r for r in self._inventory if r['name'] == name and r['seed'] == seed
                      and r['metadata']['data_hash'] == config_hash(self.data_config)]
        if len(candidates) > 1:
            raise ValueError(f'Ambiguous source checkpoints for {name}, seed {seed}; select one source run')
        return candidates[0] if candidates else None

    def _job(self, name, seed, paths, dataset_id, train_config, weight=0.0, *, kinetic='K0'):
        config = replace(train_config, seed=seed)
        source = self._base_row(name, seed)
        stored = source['metadata']['architecture'] if source else None
        if name == 'C1g' and stored is None:
            # No Phase 6 C1g exists; copy C1's shape so the pair differs only in gauge.
            c1 = self._base_row('C1', seed)
            stored = c1['metadata']['architecture'] if c1 else None
        arch = architecture(name, self.data_config, stored, kinetic=kinetic)
        train_shard = TrajectoryShard.load(paths['train'])
        scale = field_scale(train_shard, self.data_config.domain)
        pairs = train_shard.n_trajectories * (train_shard.n_frames - 1)
        spec = {'schema_version': 1, 'name': name, 'seed': seed, 'data_hash': dataset_id,
                'data_config': asdict(self.data_config),
                'data_fingerprints': {s: self._digest(paths[s]) for s in ('train', 'val')},
                'architecture': arch, 'field_scale': scale, 'trained_dt': self.data_config.dt,
                'objective': 'pino' if weight else 'one-step', 'physics_weight': weight,
                'train': protocol(config), 'quick': self.quick}
        job = Experiment(stable_hash(spec), name, seed, spec, paths, self.data_config, config, weight)
        if source and dataset_id == config_hash(self.data_config) and not weight:
            meta = source['metadata']
            original_config = replace(self.train_config, seed=seed) if self.train_config else None
            original_arch = architecture(name, self.data_config, meta['architecture'],
                                         kinetic=meta['architecture'].get('kinetic_mode', 'K0'))
            if (spec['data_fingerprints'] == self._source_fingerprints
                    and original_config and protocol(original_config) == protocol(config)
                    and original_arch == arch and meta['train_mode'] == 'one-step'
                    and meta['field_scale'] == scale and meta['trained_dt'] == self.data_config.dt):
                self._imports[job.identifier] = {**source, 'source': 'phase6', 'spec': spec}
        return job

    def prepare(self, phase, *, seeds=None, lambdas=(0., .01, .1, 1., 10.),
                fractions=(.05, .1, .25, .5, 1.), sigmas=(0., .1), gammas=(0., .001),
                train_config=None, generate=True, c1g_lambdas=()):
        config = train_config or self.train_config
        if config is None:
            raise ValueError('Original training protocol is unknown; supply train_config explicitly, including epochs')
        seeds = self.seeds if seeds is None else list(seeds)
        if not seeds or any(not isinstance(s, int) or isinstance(s, bool) or s < 0 for s in seeds):
            raise ValueError('Provide at least one nonnegative integer seed')
        if config.epochs < 1 or config.batch_size < 1 or config.patience < 1:
            raise ValueError('epochs, batch_size and patience must be positive')
        base = self._paths(self.source_root / 'data', config_hash(self.data_config))
        jobs = []
        if phase == 7:
            weights = list(dict.fromkeys(float(weight) for weight in lambdas))
            if not weights or any(not math.isfinite(w) or w < 0 for w in weights):
                raise ValueError('Select at least one finite, nonnegative PINO lambda')
            # Every sweep includes its matched no-residual control, even when the
            # caller selects only positive weights. B is one projection baseline.
            if 0.0 not in weights:
                weights.append(0.0)
            # C1g (opt-in) has its own, usually shorter, sweep; empty keeps the
            # historical five-arm catalog unchanged.
            c1g_weights = list(dict.fromkeys(float(w) for w in c1g_lambdas))
            if any(not math.isfinite(w) or w < 0 for w in c1g_weights):
                raise ValueError('C1g lambda values must be finite and nonnegative')
            if c1g_weights and 0.0 not in c1g_weights:
                c1g_weights.append(0.0)
            sweeps = {'A': weights, 'C1': weights, 'B-loop': [0.0], 'C1g': c1g_weights}
            for name, selected in sweeps.items():
                for weight in selected:
                    for seed in seeds:
                        # C1g inherits the kinetic rung of the C1 it is paired with.
                        source = self._base_row('C1' if name == 'C1g' else name, seed)
                        kinetic = source['metadata']['architecture'].get('kinetic_mode', 'K0') if source else 'K0'
                        jobs.append(self._job(name, seed, base, config_hash(self.data_config),
                                              config, weight, kinetic=kinetic))
        elif phase == 8:
            shard = TrajectoryShard.load(base['train'])
            pairs = shard.n_trajectories * (shard.n_frames - 1)
            for fraction in fractions:
                if not math.isfinite(fraction) or not 0 < fraction <= 1 or int(pairs * fraction) < 1:
                    raise ValueError('Training fraction must select at least one pair and be <= 1')
                for seed in seeds:
                    for name in ('A', 'C1'):
                        selected = replace(config, max_train_pairs=None if fraction == 1 else int(pairs * fraction))
                        source = self._base_row(name, seed)
                        kinetic = source['metadata']['architecture'].get('kinetic_mode', 'K0') if source else 'K0'
                        jobs.append(self._job(name, seed, base, config_hash(self.data_config), selected, kinetic=kinetic))
        elif phase == 9:
            for dial, values in (('sigma', sigmas), ('gamma', gammas)):
                for value in values:
                    if not math.isfinite(value):
                        raise ValueError('Perturbations must be finite')
                    spec = MisspecificationConfig(**{'nonlocal_sigma' if dial == 'sigma' else 'gain_loss_gamma': float(value)})
                    dataset_id = spec.identifier(self.data_config)
                    if spec.is_exact:
                        paths = base
                    else:
                        if generate:
                            from scripts.run_phase9 import prepare_shards
                            prepare_shards(self.data_config, spec, quick=False, data_root=self.output_root / 'data')
                        paths = self._paths(self.output_root / 'data', dataset_id)
                    for seed in seeds:
                        for name in ('A', 'B-loop', 'C1', 'C2', 'C3'):
                            jobs.append(self._job(name, seed, paths, dataset_id, config))
        else:
            raise ValueError('Training catalog supports phases 7, 8, 9; Phase 6 uses imported artifacts')
        return list({j.identifier: j for j in jobs}.values())

    def _record_path(self, job):
        return self.output_root / 'experiments' / job.identifier / 'record.json'

    def _record(self, job):
        path = self._record_path(job)
        if path.exists():
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError) as error:
                raise RuntimeError(f'Corrupt experiment record: {path}') from error
            if stable_hash(record['spec']) != job.identifier:
                raise RuntimeError(f'Incompatible experiment record: {path}')
            record['path'] = str(path.parent / 'model.pt')
            return record
        return self._imports.get(job.identifier)

    def status(self, jobs):
        rows = []
        for job in jobs:
            try:
                record = self._record(job)
                if record:
                    self._validate_record(job, record)
                    status = 'ready'
                else:
                    progress = self._record_path(job).parent / 'latest.pt'
                    status = 'interrupted' if progress.exists() or progress.with_name('latest.pt.previous').exists() else 'missing'
                rows.append({'id': job.identifier, 'name': job.name, 'seed': job.seed,
                             'status': status, 'converged': record['converged'] if record else None,
                             'source': record['source'] if record else None,
                             'objective': job.spec['objective'], 'physics_weight': job.physics_weight,
                             'train': job.spec['train']})
            except (RuntimeError, OSError, ValueError, KeyError) as error:
                rows.append({'id': job.identifier, 'name': job.name, 'seed': job.seed,
                             'status': 'incompatible', 'error': str(error)})
        return rows

    def _validate_record(self, job, record):
        if any(self._digest(job.paths[s]) != digest for s, digest in job.spec['data_fingerprints'].items()):
            raise RuntimeError('Training data changed since preparation; prepare the experiment again')
        path = Path(record['path'])
        if not path.is_file() or file_digest(path) != record['sha256']:
            raise RuntimeError(f'Checkpoint missing or corrupt: {path}')
        payload = load_checkpoint_payload(path)
        meta = payload.metadata
        if (meta.model_name != job.name or meta.seed != job.seed
                or meta.data_hash != job.spec['data_hash'] or meta.trained_dt != job.data_config.dt
                or meta.field_scale != job.spec['field_scale']
                or meta.train_mode != job.spec['objective']
                or architecture(job.name, job.data_config, meta.architecture, kinetic=meta.architecture.get('kinetic_mode', 'K0')) != job.spec['architecture']):
            raise RuntimeError(f'Incompatible checkpoint: {path}')
        return payload

    def restore(self, job, *, allow_budget_bound=True):
        record = self._record(job)
        if not record:
            raise RuntimeError(f'Checkpoint missing for {job.name}, seed {job.seed}, experiment {job.identifier}; select it in the training notebook')
        payload = self._validate_record(job, record)
        if not allow_budget_bound and not record['converged']:
            raise RuntimeError(f'Checkpoint is budget-bound: {record["path"]}')
        from scripts.run_phase6 import _model_from_checkpoint
        model = _model_from_checkpoint(payload, job.data_config, expected_name=job.name)
        model.load_state_dict(payload.state_dict, strict=True)
        return model.eval(), record

    def base_models(self, *, seeds=None, device='cpu', allow_budget_bound=False):
        """Restore base weights, including B-post as an alias of A's core."""
        from .models.projected import MassProjectedOperator
        jobs = self.prepare(9, seeds=seeds, sigmas=(0.,), gammas=(), generate=False)
        models = {}
        for job in jobs:
            model, _ = self.restore(job, allow_budget_bound=allow_budget_bound)
            models.setdefault(job.seed, {})[job.name] = model.to(device)
        for by_name in models.values():
            by_name['B-post'] = MassProjectedOperator(by_name['A']).eval()
        return models

    def train(self, jobs, *, device='cpu'):
        results = []
        for job in jobs:
            record = self._record(job)
            if record:
                self._validate_record(job, record)
                print(f'Reusing {job.name}, seed {job.seed}: {job.identifier}', flush=True)
                results.append(record)
                continue
            from scripts.run_phase6 import _model_from_checkpoint
            seed_everything(job.seed)
            model = _model_from_checkpoint(CheckpointPayload(job.metadata(), {}), job.data_config, expected_name=job.name)
            shards = self.shards(job)
            output = self._record_path(job).parent
            progress = TrainingProgress(output / 'latest.pt', job.identifier)
            config = replace(job.train_config, device=device)
            print(f'Training/resuming {job.name}, seed {job.seed}: {job.identifier}', flush=True)
            kwargs = {'physics_weight': job.physics_weight} if job.physics_weight else {}
            trainer = train_pino if job.physics_weight else train_one_step
            history = trainer(model, shards['train'], shards['val'], job.data_config, config,
                              progress=progress, **kwargs)
            did_converge = not self.quick and converged(history)
            metadata = job.metadata(did_converge=did_converge, best_epoch=history.best_epoch)
            path = output / 'model.pt'
            atomic_torch_save({'metadata': asdict(metadata), 'state_dict': model.state_dict()}, path)
            record = {'source': 'catalog', 'spec': job.spec, 'path': str(path),
                      'sha256': file_digest(path), 'history': history.as_dict(),
                      'converged': did_converge, 'metadata': asdict(metadata)}
            atomic_json(self._record_path(job), record)
            results.append(record)
        return results

    def evaluate(self, job, *, device='cpu', checkpoints=None, n_rollout=100,
                 allow_budget_bound=False):
        model, record = self.restore(job, allow_budget_bound=allow_budget_bound)
        model.to(device)
        shards = self.shards(job)
        steps = checkpoints or tuple(s for s in (1, 10, 20, 50, 100, 200) if s <= job.data_config.steps)
        if not steps or any(s < 1 or s > job.data_config.steps for s in steps):
            raise ValueError('Evaluation checkpoints must fit the available trajectory horizon')
        result = evaluate_model(model, shards, job.data_config, replace(job.train_config, device=device),
                                checkpoints=tuple(steps), n_rollout=n_rollout)
        result.update(experiment_id=job.identifier, checkpoint=record['path'], checkpoint_sha256=record['sha256'],
                      history=record.get('history'), converged=record['converged'],
                      exploratory=not record['converged'] or self.quick)
        identity = stable_hash({'experiment': job.identifier, 'test': self._digest(job.paths['test']),
                                'checkpoints': steps, 'n_rollout': n_rollout, 'device': device})
        atomic_json(self.output_root / 'evaluations' / identity / 'metrics.json', result)
        return result
