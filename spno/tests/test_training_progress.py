from dataclasses import replace

import pytest
import torch

from spno.config import DataConfig
from spno.data.datasets import generate_shard, MultiDtBatches, OneStepBatches
from spno.models.fno import FNOStepOperator
from spno.models.split_learned import DensityPhaseSplitStep
from spno.train import TrainConfig, train_one_step, train_pino, train_multi_dt


@pytest.mark.parametrize('mode', ['one-step', 'pino', 'multi-dt'])
def test_interrupted_training_resumes_exactly(tmp_path, monkeypatch, mode):
    from spno.training_progress import TrainingProgress
    data = replace(DataConfig(), grid_size=8, initial_bandwidth=2, n_train=2,
                   n_val=2, n_test=2, steps=2, substeps=2)
    shards = {s: generate_shard(data, s) for s in ('train', 'val')}
    config = TrainConfig(epochs=4, batch_size=2, seed=3, patience=2,
                         max_train_pairs=3)
    def build():
        torch.manual_seed(3)
        if mode == 'multi-dt':
            return DensityPhaseSplitStep(data.domain, width=4, trained_dt=None)
        return FNOStepOperator(data.domain, modes=3, width=4, n_layers=1,
                               trained_dt=data.dt)
    def train(model, progress=None):
        if mode == 'multi-dt':
            return train_multi_dt(model, MultiDtBatches({data.dt: shards['train']}, device='cpu'),
                                  OneStepBatches(shards['val'], device='cpu'), data, config,
                                  verbose=False, progress=progress)
        fn = train_pino if mode == 'pino' else train_one_step
        kwargs = {'physics_weight': 0.01} if mode == 'pino' else {}
        return fn(model, shards['train'], shards['val'], data, config,
                  verbose=False, progress=progress, **kwargs)
    uninterrupted = build()
    expected = train(uninterrupted)
    progress = TrainingProgress(tmp_path / 'latest.pt', 'test-experiment')
    save = progress.save
    def interrupt(*args, **kwargs):
        save(*args, **kwargs)
        raise InterruptedError('disconnect after durable epoch')
    monkeypatch.setattr(progress, 'save', interrupt)
    with pytest.raises(InterruptedError):
        train(build(), progress)
    monkeypatch.setattr(progress, 'save', save)
    restored = build()
    actual = train(restored, progress)
    assert actual.train_loss == expected.train_loss
    assert actual.val_loss == expected.val_loss
    assert actual.best_epoch == expected.best_epoch
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in uninterrupted.state_dict().items())
    # Completed progress must not run the optimizer again, even if the final model
    # checkpoint publication was interrupted.
    def forbidden(*args, **kwargs):
        raise AssertionError('completed training restarted')
    monkeypatch.setattr(torch.optim.AdamW, 'step', forbidden)
    train(build(), progress)


def test_progress_recovers_previous_publication_and_rejects_wrong_identity(tmp_path):
    from spno.training_progress import atomic_torch_save, recover_torch_load
    path = tmp_path / 'state.pt'
    atomic_torch_save({'epoch': 1}, path)
    atomic_torch_save({'epoch': 2}, path)
    path.write_bytes(b'interrupted transfer')
    assert recover_torch_load(path)['epoch'] == 1


def test_resume_after_source_config_json_roundtrip(tmp_path, monkeypatch):
    import json
    from dataclasses import asdict
    from spno.training_progress import TrainingProgress
    data = replace(DataConfig(), grid_size=8, initial_bandwidth=2, n_train=2,
                   n_val=2, steps=2, substeps=2)
    shards = {s: generate_shard(data, s) for s in ('train', 'val')}
    config = TrainConfig(epochs=3, batch_size=2, patience=2, seed=3)
    def model():
        torch.manual_seed(3)
        return FNOStepOperator(data.domain, modes=3, width=4, n_layers=1, trained_dt=data.dt)
    progress = TrainingProgress(tmp_path / 'latest.pt', 'same-experiment')
    save = progress.save
    def interrupt(**kwargs):
        save(**kwargs)
        raise InterruptedError
    monkeypatch.setattr(progress, 'save', interrupt)
    with pytest.raises(InterruptedError):
        train_one_step(model(), shards['train'], shards['val'], data, config, progress=progress)
    monkeypatch.setattr(progress, 'save', save)
    restored_data = DataConfig(**json.loads(json.dumps(asdict(data))))
    history = train_one_step(model(), shards['train'], shards['val'], restored_data, config, progress=progress)
    assert len(history.train_loss) == 3
