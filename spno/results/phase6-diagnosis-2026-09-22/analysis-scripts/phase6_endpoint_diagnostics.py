"""Resolve large rollout means into trajectory tails and phase/density errors."""
import os, sys, json, math
from pathlib import Path
ROOT=Path('C:/Users/omerm/PycharmProjects/pin/spno')
OUT=Path('C:/Users/omerm/Documents/Codex/2026-09-18/he/outputs/phase6-diagnosis-2026-09-22/frozen-test')
sys.path[:0]=[str(ROOT/'.venv/Lib/site-packages'),str(ROOT/'src'),str(ROOT)]
import torch
from spno.config import DataConfig
from spno.checkpoints import CheckpointMetadata, CheckpointPayload
from spno.data.datasets import TrajectoryShard
from spno.experiments import rollout_inputs
from spno.precision import widen_to_double
from spno.losses.relative_l2 import relative_l2_per_sample
from spno.domain import l2_mass
from scripts.run_phase6 import _model_from_checkpoint
torch.set_num_threads(2)
cfg=DataConfig()
prior=json.loads((OUT/'summary.json').read_text())
shard=TrajectoryShard(**torch.load(prior['records'][0]['test_path'],weights_only=True,map_location='cpu',mmap=True))
inp=rollout_inputs(shard,cfg,'cpu',n=shard.n_trajectories,dtype=torch.complex128)
def stats(x):
    return dict(mean=float(x.mean()),median=float(x.median()),p90=float(torch.quantile(x,.9)),
                p95=float(torch.quantile(x,.95)),max=float(x.max()),n_gt_0p1=int((x>.1).sum()),
                n_gt_1=int((x>1).sum()),values=x.tolist())
rows=[]
selected={'base/A','base/B-loop','base/C1','base/A-wide','G6a/C1','G6a/C2'}
with torch.no_grad():
    for r in prior['records']:
        if r['tag'] not in selected: continue
        raw=torch.load(r['checkpoint'],weights_only=True,map_location='cpu',mmap=True)
        meta=CheckpointMetadata(**raw['metadata'])
        model=_model_from_checkpoint(CheckpointPayload(meta,raw['state_dict']),cfg,expected_name=meta.model_name)
        model.load_state_dict(raw['state_dict'],strict=True)
        model=widen_to_double(model,device='cpu').eval()
        state=inp['initial']
        for _ in range(200): state=model(state,inp['potential'],inp['alpha'],inp['beta'],cfg.dt)
        truth=inp['trajectories'][:,200]
        err=relative_l2_per_sample(state,truth,cfg.domain)
        assert abs(float(err.mean())-r['metrics']['rollout']['relative_error'][-1])<1e-10
        inner=(state.conj()*truth).sum(-1)
        aligned=state*torch.exp(1j*torch.angle(inner))[:,None]
        aligned_err=relative_l2_per_sample(aligned,truth,cfg.domain)
        density_error=torch.linalg.vector_norm(state.abs().square()-truth.abs().square(),dim=-1)/torch.linalg.vector_norm(truth.abs().square(),dim=-1)
        mass_ratio=l2_mass(state,cfg.domain)/l2_mass(inp['initial'],cfg.domain)
        row=dict(tag=r['tag'],seed=r['seed'],checkpoint_sha256=r['checkpoint_sha256'],
                 relative_error=stats(err),phase_aligned_error=stats(aligned_err),
                 density_error=stats(density_error),mass_ratio=stats(mass_ratio),
                 optimal_global_phase_radians=torch.angle(inner).tolist())
        rows.append(row)
        print(json.dumps({k:row[k] for k in ('tag','seed')}|{'error':{k:v for k,v in row['relative_error'].items() if k!='values'},'aligned_mean':float(aligned_err.mean()),'density_mean':float(density_error.mean())}),flush=True)
    k=cfg.domain.wave_number_squared().sqrt()
    power=torch.fft.fft(inp['trajectories'],dim=-1,norm='ortho').abs().square()
    exposure={}
    for cutoff in (8,15,19):
        frac=power[...,k>cutoff].sum(-1)/power.sum(-1)
        exposure[str(cutoff)]=dict(mean_all_test_frames=float(frac.mean()),max_over_test_frames=float(frac.max()),
                                  mean_initial=float(frac[:,0].mean()),mean_final=float(frac[:,-1].mean()))
result=dict(steps=200,n_test_trajectories=100,records=rows,test_spectral_energy_above_k=exposure,
            definition='Phase alignment minimizes over one global phase per trajectory and may conceal physical phase error; it is an auxiliary diagnostic.',
            inference='Tail counts are over 100 test trajectories within each seed, not counts of independent training runs.')
(OUT/'endpoint-diagnostics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('DONE',flush=True)
