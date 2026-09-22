"""Analyze saved histories and compute cheap reference baselines; no training."""
import os, sys, json, math, statistics as st
from pathlib import Path
from datetime import datetime

ROOT = Path('C:/Users/omerm/PycharmProjects/pin/spno')
OUT = Path('C:/Users/omerm/Documents/Codex/2026-09-18/he/outputs/phase6-diagnosis-2026-09-22')
os.environ['MPLCONFIGDIR'] = 'C:/Users/omerm/Documents/Codex/2026-09-18/he/work/mplconfig'
sys.path[:0] = [str(ROOT/'.venv/Lib/site-packages'), str(ROOT/'src'), str(ROOT)]
import torch
import matplotlib.pyplot as plt
from spno.config import DataConfig
from spno.checkpoints import CheckpointMetadata, CheckpointPayload
from spno.solvers.split_step import SplitStepNLSOperator
from spno.losses.relative_l2 import relative_l2_per_sample
from scripts.run_phase6 import _model_from_checkpoint
torch.set_num_threads(2)

audit = json.loads((OUT/'audit.json').read_text())
source = Path(audit['source'])
saved = json.loads(source.read_text())
rows = audit['records']
assert all(saved['base'][str(r['seed'])][r['model']]['history']['best_val'] == r['best_val']
           for r in rows if r['arm']=='base' and r['model']!='A-wide')
group_order = [('base', n) for n in ('A','B-loop','C1','C2','C3','A-wide')]
group_order += [('G6a',n) for n in ('C1','C2','C3')]
group_order += [('G7-alpha-fixed',n) for n in ('A','C1','C2')]
groups = {}
for arm, name in group_order:
    group = sorted([r for r in rows if (r['arm'],r['model'])==(arm,name)],key=lambda r:r['seed'])
    vals = [r['best_val'] for r in group]
    payload_raw = torch.load(group[0]['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    payload = CheckpointPayload(CheckpointMetadata(**payload_raw['metadata']),payload_raw['state_dict'])
    model = _model_from_checkpoint(payload,DataConfig(),expected_name=payload.metadata.model_name)
    counts = dict(project_parameter_count=model.parameter_count(),
                  real_scalar_parameter_count=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters()))
    groups[arm+'/'+name] = dict(
        values=vals, mean=st.mean(vals),median=st.median(vals),std=st.stdev(vals),
        minimum=min(vals),maximum=max(vals),cv_pct=100*st.stdev(vals)/st.mean(vals),
        best_epochs_1based=[r['best_epoch_index']+1 for r in group],
        epochs_run=[r['epochs'] for r in group],
        passes_gate=[r['passes_gate'] for r in group],
        final_val_over_online_train=[r['final_val']/r['train_loss'][-1] for r in group],
        hours=[r['elapsed_hours'] for r in group],
        last5_improvement_pct=[r['last_five_transitions_improvement_pct'] for r in group],
        **counts)

comparisons = {}
for target, reference in [('base/C1','base/A'),('base/C1','base/B-loop'),('base/C1','base/C2'),
                          ('base/C1','base/C3'),('base/C1','base/A-wide'),('base/B-loop','base/A'),
                          ('base/A-wide','base/A'),('G6a/C1','base/C1'),('G6a/C2','base/C2'),
                          ('G6a/C3','base/C3'),('G6a/C2','G6a/C1')]:
    x,y=groups[target],groups[reference]
    comparisons[target+' vs '+reference] = dict(
        lower_error_pct_by_seed=[100*(b-a)/b for a,b in zip(x['values'],y['values'])],
        lower_mean_error_pct=100*(y['mean']-x['mean'])/y['mean'],
        lower_median_error_pct=100*(y['median']-x['median'])/y['median'])

data_root=ROOT/'results/phase6-standalone-artifacts/data'
val=torch.load(data_root/'nls1d-bd4e108527/val.pt',map_location='cpu',weights_only=True,mmap=True)
val_multi=torch.load(data_root/'nls1d-g6a-bd4e108527/val.pt',map_location='cpu',weights_only=True,mmap=True)
same_val={key:torch.equal(val[key],val_multi[key]) for key in ('trajectories','potential','alpha','beta','trajectory_ids')}
domain=DataConfig().domain
solver=SplitStepNLSOperator(domain)

@torch.no_grad()
def baselines(raw,dtype):
    rdtype=torch.float32 if dtype==torch.complex64 else torch.float64
    traj=raw['trajectories'].to(dtype)
    n,frames,width=traj.shape
    current=traj[:,:-1,:].reshape(-1,width)
    target=traj[:,1:,:].reshape(-1,width)
    potential=raw['potential'].to(rdtype).repeat_interleave(frames-1,dim=0)
    alpha=raw['alpha'].to(rdtype).repeat_interleave(frames-1)
    beta=raw['beta'].to(rdtype).repeat_interleave(frames-1)
    identity_errors=[];strang_errors=[]
    for start in range(0,len(current),256):
        sl=slice(start,start+256)
        identity_errors.append(relative_l2_per_sample(current[sl],target[sl],domain).double())
        pred=solver(current[sl],potential[sl],alpha[sl],beta[sl],0.01)
        strang_errors.append(relative_l2_per_sample(pred,target[sl],domain).double())
    return dict(n_pairs=len(current),n_trajectories=n,
                persistence_mean=float(torch.cat(identity_errors).mean()),
                one_strang_step_mean=float(torch.cat(strang_errors).mean()),
                dtype=str(dtype))

refs={str(dtype):baselines(val,dtype) for dtype in (torch.complex64,torch.complex128)}
ref32=refs['torch.complex64']
for tag,entry in groups.items():
    if not tag.startswith('G7'):
        entry['mean_vs_persistence_ratio']=entry['mean']/ref32['persistence_mean']
        entry['mean_vs_one_strang_step_ratio']=entry['mean']/ref32['one_strang_step_mean']

# Descriptive matched-step view: multi-dt processes 3 x 160,000 pairs per epoch.
# It is not a controlled compute-matched experiment: the schedules also differ.
matched_updates={}
for name in ('C1','C2','C3'):
    rs=sorted([r for r in rows if r['arm']=='G6a' and r['model']==name],key=lambda r:r['seed'])
    at13=[min(r['val_loss'][:13]) for r in rs]
    baseline=groups['base/'+name]['values']
    matched_updates[name]=dict(
        multi_dt_first13_best=at13,
        base_up_to40_best=baseline,
        lower_error_pct_by_seed=[100*(b-a)/b for a,b in zip(at13,baseline)],
        multi_updates=13*1875,base_updates=40*625,
        caveat='Approximate update count only; different LR schedule position, data distribution, and checkpoint selection opportunities.')

result=dict(created_at=datetime.now().astimezone().isoformat(),groups=groups,
            comparisons=comparisons,reference_baselines=refs,
            base_and_multi_dt_validation_identical=same_val,
            approximate_matched_updates=matched_updates,
            saved_training_data=dict(n_train=800,n_val=100,frames=201,grid=64,dt=0.01,
                                     pairs_per_epoch_base=160000,pairs_per_epoch_multi_dt=480000,
                                     updates_per_epoch_base=625,updates_per_epoch_multi_dt=1875),
            notes=['Only saved validation histories were used for learned-model comparisons.',
                   'New computations are persistence and one-Strang-step baselines, not trained-model test evaluations.',
                   'No test, rollout, spectral-identification, conservation or distribution-shift experiment was run.',
                   'Means and sample standard deviations are descriptive across 3 training seeds, not confidence intervals.',
                   'All seeds share a validation split: 20,000 frame pairs from 100 trajectories are not 20,000 independent trajectories.'])
(OUT/'advanced-statistics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

fig,axes=plt.subplots(1,3,figsize=(16,5.5),layout='constrained')
colors=['#176b9b','#d66c21','#33844b']
names=['A','B-loop','C1','C2','C3','A-wide']
for i,name in enumerate(names):
    g=groups['base/'+name]
    axes[0].plot([i-.17,i+.17],[g['median']*1e4]*2,color='black',lw=2)
    for seed,v in enumerate(g['values']):
        axes[0].scatter(i+(seed-1)*.09,v*1e4,color=colors[seed],s=46,label=f'seed {seed}' if i==0 else None,zorder=3)
axes[0].set(xticks=range(len(names)),xticklabels=names,yscale='log',ylabel='Best validation relative L2 (x 1e-4)',title='Base models: all seeds, black bar = median')
axes[0].legend(fontsize=8)
axes[0].axhline(ref32['one_strang_step_mean']*1e4,color='#888888',ls=':',lw=1)
axes[0].text(.02,.025,'Dotted line: one Strang step (known equation)',transform=axes[0].transAxes,fontsize=8)
for i,name in enumerate(('C1','C2','C3')):
    for seed in range(3):
        before=groups['base/'+name]['values'][seed]*1e4
        after=groups['G6a/'+name]['values'][seed]*1e4
        axes[1].plot([2*i,2*i+1],[before,after],'-o',color=colors[seed],lw=1.3,alpha=.9)
axes[1].set(xticks=range(6),xticklabels=['C1\nbase','C1\nmulti','C2\nbase','C2\nmulti','C3\nbase','C3\nmulti'],ylabel='Best validation relative L2 (x 1e-4)',title='Multi-dt: same validation set, 3x updates/epoch')
ratios=[groups['base/'+n]['real_scalar_parameter_count']/groups['base/C1']['real_scalar_parameter_count'] for n in names]
axes[2].barh(names,ratios,color=['#7b91a1','#7b91a1','#33844b','#7b91a1','#7b91a1','#7b91a1'])
axes[2].invert_yaxis()
axes[2].set(xscale='log',xlabel='Real scalar parameter count / C1',title='C1 is much smaller (complex parameter = 2 reals)')
for i,v in enumerate(ratios):
    axes[2].text(v*1.08,i,f'{v:.0f}x' if v>2 else f'{v:.1f}x',va='center',fontsize=9)
axes[2].set_xlim(.8,max(ratios)*2)
for ax in axes:ax.grid(alpha=.17)
fig.suptitle('Phase 6 training evidence: promising one-step fits, central generalization tests still missing',fontsize=14)
fig.savefig(OUT/'analysis-overview.png',dpi=170)
plt.close(fig)
print(json.dumps(result,indent=2))
