from dataclasses import asdict, replace
import json

import pytest
import torch

from spno.checkpoints import CheckpointMetadata, checkpoint_path, save_checkpoint
from spno.config import DataConfig, config_hash
from spno.data.datasets import generate_shard, shard_paths
from spno.train import TrainConfig, field_scale


@pytest.fixture
def standalone(tmp_path):
    from scripts.run_phase6 import _model_from_checkpoint
    from spno.checkpoints import CheckpointPayload
    data = replace(DataConfig(), grid_size=8, initial_bandwidth=2, substeps=2,
                   steps=2, n_train=2, n_val=2, n_test=2)
    config = TrainConfig(epochs=2, batch_size=2, patience=2)
    root = tmp_path / 'phase6-standalone-artifacts'
    shards = {s: generate_shard(data, s) for s in ('train', 'val', 'test')}
    for split, path in shard_paths(root / 'data', config_hash(data)).items():
        shards[split].save(path)
    scale = field_scale(shards['train'], data.domain)
    originals = {}
    for seed in (3, 5):
        for name in ('A', 'B-loop', 'C1', 'C2', 'C3'):
            arch = {'modes': 3, 'width': 4, 'n_layers': 1, 'kinetic_mode': 'K0',
                    'local_mode': None if name == 'C3' else 'L0'}
            meta = CheckpointMetadata(1, name, config_hash(data), seed, 'one-step',
                                      scale, data.dt, arch, True, 0)
            torch.manual_seed(seed)
            model = _model_from_checkpoint(CheckpointPayload(meta, {}), data, expected_name=name)
            path = checkpoint_path(root, 'phase6', config_hash(data) + '-K0', name, seed)
            save_checkpoint(path, model, meta)
            originals[name, seed] = model
    return root, data, config, originals


def test_import_reuses_all_seeds_and_zero_controls_without_training(standalone, tmp_path, monkeypatch):
    from spno.workflow import Workflow
    root, data, config, originals = standalone
    before = {p: p.read_bytes() for p in root.rglob('*.pt')}
    workflow = Workflow(root, tmp_path / 'new', data_config=data, train_config=config)
    jobs = workflow.prepare(7, seeds=[3, 5], lambdas=[0.0])
    assert {j.seed for j in jobs} == {3, 5}
    assert all(row['status'] == 'ready' for row in workflow.status(jobs))
    def forbidden(*args, **kwargs):
        raise AssertionError('retrained an imported model')
    monkeypatch.setattr(torch.optim.AdamW, 'step', forbidden)
    workflow.train(jobs)
    for job in jobs:
        model, record = workflow.restore(job)
        assert all(torch.equal(v, model.state_dict()[k])
                   for k, v in originals['A', job.seed].state_dict().items())
        assert record['source'] == 'phase6'
    full = workflow.prepare(8, seeds=[3], fractions=[1.0])
    zero9 = workflow.prepare(9, seeds=[3], sigmas=[0.0], gammas=[0.0])
    a_id = next(j.identifier for j in jobs if j.seed == 3)
    assert next(j.identifier for j in full if j.name == 'A') == a_id
    assert next(j.identifier for j in zero9 if j.name == 'A') == a_id
    assert len(zero9) == 5
    assert all(p.read_bytes() == b for p, b in before.items())


def test_new_training_is_reused_and_configuration_changes_are_distinct(standalone, tmp_path, monkeypatch):
    from spno.workflow import Workflow
    root, data, config, _ = standalone
    workflow = Workflow(root, tmp_path / 'new', data_config=data, train_config=config)
    jobs = workflow.prepare(7, seeds=[3], lambdas=[0.01])
    assert workflow.status(jobs)[0]['status'] == 'missing'
    workflow.train(jobs)
    assert workflow.status(jobs)[0]['status'] == 'ready'
    def forbidden(*args, **kwargs):
        raise AssertionError('optimizer should not run')
    monkeypatch.setattr(torch.optim.AdamW, 'step', forbidden)
    workflow.train(jobs)
    with pytest.raises(RuntimeError, match='missing'):
        workflow.restore(workflow.prepare(7, seeds=[5], lambdas=[0.01])[0])
    assert workflow.prepare(7, seeds=[3], lambdas=[0.1])[0].identifier != jobs[0].identifier
    other = Workflow(root, tmp_path / 'new', data_config=data, train_config=config)
    assert other.prepare(7, seeds=[3], lambdas=[0.01], train_config=replace(config, batch_size=1))[0].identifier != jobs[0].identifier


def test_unknown_legacy_budget_never_becomes_matched_control(standalone, tmp_path):
    from spno.workflow import Workflow
    root, data, _, _ = standalone
    workflow = Workflow(root, tmp_path / 'new', data_config=data)
    assert workflow.inventory()[0]['training_config'] is None
    with pytest.raises(ValueError, match='training protocol'):
        workflow.prepare(7, seeds=[3], lambdas=[0.0])


@pytest.mark.parametrize('phase', [7, 8, 9])
def test_phase_evaluation_never_trains_and_preserves_sources(standalone, tmp_path, monkeypatch, phase):
    from spno.workflow import Workflow
    from spno.phase_workflow import evaluate_phase
    root, data, config, _ = standalone
    workflow = Workflow(root, tmp_path / 'new', data_config=data, train_config=config)
    def forbidden(*args, **kwargs):
        raise AssertionError('evaluation attempted training')
    monkeypatch.setattr(torch.optim.AdamW, 'step', forbidden)
    result = evaluate_phase(workflow, phase, seeds=[3], lambdas=[0.0], fractions=[1.0],
                            sigmas=[0.0], gammas=[0.0], grid=16, noise=[0.0])
    assert result['phase'] == phase
    assert result['checkpoint_sources']
    assert (tmp_path / 'new' / 'reports' / result['identifier'] / 'metrics.json').exists()


def test_phase7_missing_checkpoint_fails_without_starting_training(standalone, tmp_path, monkeypatch):
    from spno.workflow import Workflow
    from spno.phase_workflow import evaluate_phase
    root, data, config, _ = standalone
    workflow = Workflow(root, tmp_path / 'new', data_config=data, train_config=config)
    monkeypatch.setattr(torch.optim.AdamW, 'step', lambda *_: pytest.fail('training from evaluation'))
    with pytest.raises(RuntimeError, match='missing'):
        evaluate_phase(workflow, 7, seeds=[3], lambdas=[0.01])


def test_training_data_change_does_not_relabel_old_weights(standalone, tmp_path):
    from spno.workflow import Workflow
    from spno.data.datasets import TrajectoryShard
    root, data, config, _ = standalone
    workflow = Workflow(root, tmp_path / 'new', data_config=data, train_config=config)
    original = workflow.prepare(7, seeds=[3], lambdas=[0.0])[0]
    train_path = shard_paths(root / 'data', config_hash(data))['val']
    shard = TrajectoryShard.load(train_path)
    shard.trajectories.mul_(1.01)
    shard.save(train_path)
    changed = workflow.prepare(7, seeds=[3], lambdas=[0.0])[0]
    assert changed.identifier != original.identifier
    assert workflow.status([changed])[0]['status'] != 'ready'


def test_explicit_full_pair_sampling_is_not_aliased_to_unsampled_training(standalone, tmp_path):
    from spno.workflow import Workflow
    root, data, config, _ = standalone
    workflow = Workflow(root, tmp_path / 'new', data_config=data, train_config=config)
    original = workflow.prepare(7, seeds=[3], lambdas=[0.0])[0]
    changed = workflow.prepare(7, seeds=[3], lambdas=[0.0],
                               train_config=replace(config, max_train_pairs=4))[0]
    assert changed.identifier != original.identifier
    assert workflow.status([changed])[0]['status'] == 'missing'


def test_b_post_is_projection_of_the_imported_a(standalone, tmp_path):
    from spno.workflow import Workflow
    from spno.models.projected import MassProjectedOperator
    root, data, config, originals = standalone
    workflow = Workflow(root, tmp_path / 'new', data_config=data, train_config=config)
    models = workflow.base_models(seeds=[3])
    assert isinstance(models[3]['B-post'], MassProjectedOperator)
    assert models[3]['B-post'].core is models[3]['A']


def test_corrupt_checkpoint_and_changed_inputs_never_start_training(standalone, tmp_path, monkeypatch):
    from spno.workflow import Workflow
    root, data, config, _ = standalone
    workflow = Workflow(root, tmp_path / 'new', data_config=data, train_config=config)
    job = workflow.prepare(7, seeds=[3], lambdas=[0.0])[0]
    record = workflow.restore(job)[1]
    from pathlib import Path
    Path(record['path']).write_bytes(b'broken checkpoint')
    monkeypatch.setattr(torch.optim.AdamW, 'step', lambda *_: pytest.fail('implicit repair training'))
    assert workflow.status([job])[0]['status'] == 'incompatible'
    with pytest.raises(RuntimeError, match='corrupt'):
        workflow.train([job])


def test_phase6_evaluates_existing_standalone_without_trainer(tmp_path, monkeypatch):
    from scripts.train_phase6_arms import train_phase6_arms
    from spno.data.shift import ShiftSpec
    from spno.workflow import Workflow
    from spno.phase_workflow import evaluate_phase
    data = replace(DataConfig(), n_train=2, n_val=2, n_test=2, steps=2)
    fixed = ShiftSpec('G7-alpha-fixed', replace(DataConfig(), alpha_range=(.9, .9), seed=120))
    source = tmp_path / 'phase6-standalone-artifacts-quick'
    train_phase6_arms(data, seeds=[0], epochs=1, device='cpu', kinetic='K0', quick=True,
                      standalone=True, data_root=source / 'data', checkpoint_root=source,
                      artifact_root=source / 'data', shift_specs={'G7-alpha-fixed': fixed})
    workflow = Workflow(source, tmp_path / 'new', data_config=data,
                        train_config=TrainConfig(epochs=1, batch_size=256, patience=8, max_train_pairs=2))
    monkeypatch.setattr(torch.optim.AdamW, 'step', lambda *_: pytest.fail('phase6 retrained'))
    result = evaluate_phase(workflow, 6, arms=['G5a'])
    assert 'G5a' in result['experiments']
    assert result['quick']


def test_legacy_source_copy_verifies_bytes_and_keeps_history_reports(standalone, tmp_path):
    from spno.artifacts import copy_standalone, file_digest
    root, data, config, _ = standalone
    report = root.parent / 'phase6-training-demo' / 'metrics.json'
    report.parent.mkdir()
    report.write_text(json.dumps({'standalone': True, 'data_hash': config_hash(data)}))
    destination = tmp_path / 'persistent' / 'imported'
    copy_standalone(root, destination)
    assert (destination / 'reports' / report.parent.name / 'metrics.json').read_bytes() == report.read_bytes()
    for source in root.rglob('*.pt'):
        assert file_digest(source) == file_digest(destination / source.relative_to(root))


def test_new_protocol_can_be_evaluated_without_redeclaring_source(standalone, tmp_path):
    from spno.workflow import Workflow
    from spno.phase_workflow import evaluate_phase
    root, data, config, _ = standalone
    workflow = Workflow(root, tmp_path / 'new', data_config=data, train_config=config)
    requested = replace(config, epochs=1, batch_size=1)
    jobs = workflow.prepare(7, seeds=[3], lambdas=[0.0], train_config=requested)
    assert workflow.status(jobs)[0]['status'] == 'missing'
    workflow.train(jobs)
    result = evaluate_phase(workflow, 7, seeds=[3], lambdas=[0.0], train_config=requested,
                            allow_budget_bound=True)
    assert jobs[0].identifier in result['checkpoint_sources']
    assert result['checkpoint_sources'][jobs[0].identifier]['source'] == 'catalog'
    assert workflow.status(workflow.prepare(7, seeds=[3], lambdas=[0.0]))[0]['source'] == 'phase6'


def test_phase8_uses_source_kinetics_while_phase9_requests_k0(standalone, tmp_path):
    from spno.workflow import Workflow
    from spno.checkpoints import load_checkpoint_payload
    from scripts.run_phase6 import _model_from_checkpoint
    root, data, config, _ = standalone
    path = checkpoint_path(root, 'phase6', config_hash(data) + '-K0', 'C1', 3)
    payload = load_checkpoint_payload(path)
    meta = replace(payload.metadata, architecture={**payload.metadata.architecture, 'kinetic_mode': 'K1'})
    from spno.checkpoints import CheckpointPayload
    model = _model_from_checkpoint(CheckpointPayload(meta, {}), data, expected_name='C1')
    save_checkpoint(path, model, meta)
    workflow = Workflow(root, tmp_path / 'new', data_config=data, train_config=config)
    jobs = workflow.prepare(8, seeds=[3], fractions=[1.0])
    assert all(row['status'] == 'ready' for row in workflow.status(jobs))
    jobs9 = workflow.prepare(9, seeds=[3], sigmas=[0.0], gammas=[])
    c1 = next(j for j in jobs9 if j.name == 'C1')
    assert c1.spec['architecture']['kinetic_mode'] == 'K0'
    assert workflow.status([c1])[0]['status'] == 'missing'


def test_explicit_cli_stage_inherits_source_seeds(standalone, tmp_path):
    from scripts.run_phase7 import main
    root, data, config, _ = standalone
    source_config = tmp_path / 'source.json'
    source_config.write_text(json.dumps({'data': asdict(data), 'train': asdict(config)}))
    result = main(['--stage', 'prepare', '--source-root', str(root), '--output-root', str(tmp_path / 'new'),
                   '--source-config', str(source_config), '--lambdas', '0.0'])
    assert {j['seed'] for j in result['jobs']} == {3, 5}
    assert all(j['source'] == 'phase6' for j in result['jobs'])


def test_phase9_report_identity_includes_perturbed_test_data(standalone, tmp_path):
    from spno.workflow import Workflow
    from spno.phase_workflow import evaluate_phase
    from spno.data.datasets import TrajectoryShard
    root, data, config, _ = standalone
    workflow = Workflow(root, tmp_path / 'new', data_config=data, train_config=config)
    requested = replace(config, epochs=1)
    jobs = workflow.prepare(9, seeds=[3], sigmas=[0.1], gammas=[], train_config=requested)
    workflow.train(jobs)
    first = evaluate_phase(workflow, 9, seeds=[3], sigmas=[0.1], gammas=[],
                           train_config=requested, allow_budget_bound=True)
    path = jobs[0].paths['test']
    shard = TrajectoryShard.load(path)
    shard.trajectories[:, 1:].mul_(1.01)
    shard.save(path)
    second = evaluate_phase(workflow, 9, seeds=[3], sigmas=[0.1], gammas=[],
                            train_config=requested, allow_budget_bound=True)
    assert first['identifier'] != second['identifier']
    assert first['evaluation_datasets'] != second['evaluation_datasets']


def test_reopening_catalog_does_not_relabel_changed_validation_data(standalone, tmp_path):
    from spno.workflow import Workflow
    from spno.data.datasets import TrajectoryShard
    root, data, config, _ = standalone
    first = Workflow(root, tmp_path / 'new', data_config=data, train_config=config)
    first.prepare(7, seeds=[3], lambdas=[0.0])
    path = shard_paths(root / 'data', config_hash(data))['val']
    shard = TrajectoryShard.load(path)
    shard.trajectories[:, 1:].mul_(1.01)
    shard.save(path)
    reopened = Workflow(root, tmp_path / 'new', data_config=data, train_config=config)
    assert reopened.status(reopened.prepare(7, seeds=[3], lambdas=[0.0]))[0]['status'] == 'missing'
