"""Evaluate saved weights on the existing test set. No training or source edits."""
import os, sys, json, time, hashlib, statistics
from pathlib import Path
from datetime import datetime
ROOT = Path('C:/Users/omerm/PycharmProjects/pin/spno')
OUT = Path('C:/Users/omerm/Documents/Codex/2026-09-18/he/outputs/phase6-diagnosis-2026-09-22/frozen-test')
OUT.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(OUT.parent.parent.parent / 'work/mplconfig')
sys.path[:0] = [str(ROOT/'.venv/Lib/site-packages'), str(ROOT/'src'), str(ROOT)]
import torch
import matplotlib.pyplot as plt
from spno.config import DataConfig
from spno.checkpoints import CheckpointMetadata, CheckpointPayload
from spno.data.datasets import TrajectoryShard, OneStepBatches
from spno.experiments import evaluate_model, rollout_inputs
from spno.evaluation.rollout import evaluate_rollout
from spno.train import TrainConfig, evaluate_one_step
from spno.solvers.split_step import SplitStepNLSOperator
from scripts.run_phase6 import _model_from_checkpoint
torch.set_num_threads(2)
cfg = DataConfig()
train_cfg = TrainConfig(batch_size=256, device='cpu')
test_path = ROOT/'results/phase6-standalone-artifacts/data/nls1d-bd4e108527/test.pt'
shard = TrajectoryShard(**torch.load(test_path, map_location='cpu', weights_only=True, mmap=True))
test_hash = hashlib.sha256(test_path.read_bytes()).hexdigest()
audit = json.loads((OUT.parent/'audit.json').read_text())
records = []
started = time.time()
for row in [r for r in audit['records'] if r['arm'] in ('base', 'G6a')]:
    tag = row['arm']+'/'+row['model']
    path = Path(row['checkpoint'])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    output_path = OUT/(tag.replace('/', '-')+f"-seed{row['seed']}.json")
    if output_path.exists():
        record = json.loads(output_path.read_text())
        assert record['checkpoint_sha256'] == digest and record['test_sha256'] == test_hash
        records.append(record)
        print('REUSED '+str(output_path.name), flush=True)
        continue
    raw = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    meta = CheckpointMetadata(**raw['metadata'])
    model = _model_from_checkpoint(CheckpointPayload(meta, raw['state_dict']), cfg, expected_name=meta.model_name)
    model.load_state_dict(raw['state_dict'], strict=True)
    model.eval()
    model_started = time.time()
    metrics = evaluate_model(model, {'test': shard}, cfg, train_cfg, n_rollout=shard.n_trajectories)
    record = dict(tag=tag, seed=row['seed'], checkpoint=str(path), checkpoint_sha256=digest,
                  test_path=str(test_path), test_sha256=test_hash, saved_converged=meta.converged,
                  policy='frozen-budget diagnostic; original convergence metadata unchanged',
                  metrics=metrics, evaluation_seconds=time.time()-model_started)
    output_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    records.append(record)
    print(json.dumps({'tag':tag,'seed':row['seed'],'test':metrics['one_step_test'],
                      'rollout_200':metrics['rollout']['relative_error'][-1],
                      'mass_drift_200':metrics['rollout']['mass_drift'][-1],
                      'seconds':record['evaluation_seconds']}), flush=True)
reference = SplitStepNLSOperator(cfg.domain)
ref_one = evaluate_one_step(reference, OneStepBatches(shard,device='cpu'), cfg.domain, cfg.dt, train_cfg)
inp = rollout_inputs(shard, cfg, 'cpu', n=shard.n_trajectories, dtype=torch.complex128)
ref_roll = evaluate_rollout(reference, cfg.domain, inp['initial'],inp['trajectories'],inp['potential'],inp['alpha'],inp['beta'],cfg.dt)
summary = {}
for tag in dict.fromkeys(r['tag'] for r in records):
    rs = sorted([r for r in records if r['tag']==tag], key=lambda r:r['seed'])
    summary[tag] = {}
    for key in ('one_step_test','rollout_200','mass_drift_200','energy_drift_200'):
        vals = [r['metrics']['one_step_test'] if key=='one_step_test' else r['metrics']['rollout'][{'rollout_200':'relative_error','mass_drift_200':'mass_drift','energy_drift_200':'energy_drift'}[key]][-1] for r in rs]
        summary[tag][key] = dict(mean=statistics.mean(vals), std=statistics.stdev(vals), by_seed=vals)
result = dict(measured_at=datetime.now().astimezone().isoformat(),
              n_test_trajectories=shard.n_trajectories, n_test_pairs=shard.n_trajectories*(shard.n_frames-1),
              test_sha256=test_hash, dt=cfg.dt, steps=200, physical_horizon=200*cfg.dt,
              reference=dict(one_step_test=ref_one, rollout=ref_roll.as_dict()),
              summary=summary, records=records, seconds=time.time()-started,
              limitations=['IID test; not a replacement for G1-G4 distribution shifts or a multi-dt test sweep.',
                           'All 100 test trajectories and 20000 one-step pairs; paired test set across models.',
                           'One-step float32 and rollout float64 CPU widened weights, matching repository evaluator.',
                           '200-step horizon does not establish indefinitely bounded energy drift.',
                           'Three training seeds; no hypothesis test or population confidence claim.'])
(OUT/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
fig, axs = plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
colors = {'A':'#7b6bb0','B-loop':'#c37d2d','C1':'#27824e','C2':'#247fa6','C3':'#be5454','A-wide':'#777777'}
for tag in summary:
    arm,name = tag.split('/')
    rs = [r for r in records if r['tag']==tag]
    steps = rs[0]['metrics']['rollout']['steps']
    means = [statistics.mean(r['metrics']['rollout']['relative_error'][i] for r in rs) for i in range(len(steps))]
    axs[0 if arm=='base' else 1].semilogy(steps,means,'-o',label=name,color=colors[name],markersize=4)
for ax, title in zip(axs,('Base checkpoints','Multi-dt checkpoints evaluated at dt=0.01')):
    ax.semilogy(ref_roll.steps,ref_roll.relative_error,'--',label='Known-physics Strang',color='black')
    ax.set(xlabel='Autoregressive steps',ylabel='Mean relative L2 error',title=title)
    ax.grid(alpha=.2); ax.legend(fontsize=8)
fig.suptitle('Frozen weights: all 100 held-out test trajectories, 3 training seeds',fontsize=12)
fig.savefig(OUT/'rollout-test.png',dpi=160)
plt.close(fig)
print('DONE '+json.dumps({'seconds':time.time()-started,'models':len(records),'output':str(OUT)}),flush=True)
