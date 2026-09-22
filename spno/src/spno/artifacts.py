"""Read-only discovery and verified copying of legacy Phase 6 artifacts."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

from .checkpoints import load_checkpoint_payload
from .config import DataConfig, config_hash
from .train import TrainConfig


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    try:
        with temp.open('w') as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def verified_copy(source: Path, destination: Path) -> Path:
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        return source
    expected = file_digest(source)
    if destination.exists():
        if file_digest(destination) != expected:
            raise RuntimeError(f'Refusing to overwrite different artifact: {destination}')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + f'.{uuid.uuid4().hex}.tmp')
    try:
        shutil.copyfile(source, temp)
        if file_digest(temp) != expected or file_digest(source) != expected:
            raise RuntimeError(f'Artifact changed during copy: {source}')
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)
    return destination


def copy_standalone(source: Path, destination: Path) -> Path:
    """Copy completed files only, never move/delete the source run."""
    source, destination = Path(source), Path(destination)
    if not (source / 'checkpoints' / 'phase6').is_dir():
        raise FileNotFoundError(f'No Phase 6 checkpoints under {source}')
    if destination.resolve().is_relative_to(source.resolve()) and destination.resolve() != source.resolve():
        raise ValueError('Destination must not be nested inside source')
    for path in sorted(source.rglob('*')):
        if path.is_file() and path.suffix in {'.pt', '.json'}:
            verified_copy(path, destination / path.relative_to(source))
    # Standalone training histories live beside the artifact directory.
    for pattern in ('phase6-training-*/metrics.json', 'phase6-*/metrics.json'):
        for path in source.parent.glob(pattern):
            verified_copy(path, destination / 'reports' / path.parent.name / path.name)
    return destination


def inspect_standalone(root: Path, data_config=None, train_config=None):
    root = Path(root)
    paths = sorted((root / 'checkpoints' / 'phase6').rglob('*.pt'))
    if not paths:
        raise FileNotFoundError(f'No Phase 6 checkpoints in {root}; set SOURCE_ROOT to the standalone artifact directory')
    reports = []
    report_paths = set(root.glob('reports/*/metrics.json'))
    for pattern in ('phase6-training-*/metrics.json', 'phase6-*/metrics.json'):
        report_paths.update(root.parent.glob(pattern))
    for path in sorted(report_paths):
        try:
            reports.append((path, json.loads(path.read_text())))
        except (OSError, ValueError):
            continue  # A running writer may not have published its report yet.
    payloads = []
    for path in paths:
        try:
            payloads.append((path, load_checkpoint_payload(path)))
        except Exception as error:
            raise RuntimeError(f'Cannot read source checkpoint: {path}; restore a verified copy before continuing') from error
    base_hashes = {p.metadata.data_hash for _, p in payloads if p.metadata.model_name == 'A'}
    if data_config is None:
        candidates = [DataConfig(), replace(DataConfig(), n_train=2, n_val=2, n_test=2, steps=2)]
        matches = [c for c in candidates if base_hashes == {config_hash(c)}]
        if len(matches) != 1:
            raise ValueError('Provide data_config matching the original Phase 6 data hash')
        data_config = matches[0]
    if config_hash(data_config) not in base_hashes:
        raise ValueError('Source data_config does not match Phase 6 base checkpoints')
    matching_reports = [(p, r) for p, r in reports if r.get('data_hash') == config_hash(data_config)
                        and r.get('standalone') is True]
    epochs = {r['epochs'] for _, r in matching_reports if 'epochs' in r}
    protocol_source = 'explicit user declaration' if train_config is not None else None
    if train_config is None and len(epochs) == 1:
        quick = any(r.get('quick', False) for _, r in matching_reports)
        train_config = TrainConfig(epochs=epochs.pop(), batch_size=256, learning_rate=1e-3,
                                   patience=8, max_train_pairs=2 if quick else None)
        protocol_source = 'standalone report + original train_phase6_arms protocol'
    histories = {}
    def walk(value):
        if isinstance(value, dict):
            if 'checkpoint' in value and 'history' in value:
                key = str(value['checkpoint']).split('/checkpoints/')[-1]
                histories[key] = value['history']
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    for _, report in matching_reports:
        walk(report)
    rows = []
    for path, payload in payloads:
        meta = payload.metadata
        rows.append({
            'name': meta.model_name, 'seed': meta.seed, 'path': str(path),
            'metadata': asdict(meta), 'sha256': file_digest(path),
            'history': histories.get(str(path.relative_to(root / 'checkpoints'))),
            'training_config': asdict(replace(train_config, seed=meta.seed)) if train_config else None,
            'protocol_source': protocol_source, 'converged': meta.converged,
            'quick': path.parent.name.endswith('-quick') or '-quick' in str(path.relative_to(root)),
        })
    return data_config, train_config, rows
