"""Diagnostic spectral probes of saved weights; no optimizer, training, or source edits."""
import os,sys,json,time,hashlib,statistics
from pathlib import Path
from datetime import datetime
ROOT=Path('C:/Users/omerm/PycharmProjects/pin/spno')
OUT=Path('C:/Users/omerm/Documents/Codex/2026-09-18/he/outputs/phase6-diagnosis-2026-09-22/frozen-probes')
OUT.mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']='C:/Users/omerm/Documents/Codex/2026-09-18/he/work/mplconfig'
sys.path[:0]=[str(ROOT/'.venv/Lib/site-packages'),str(ROOT/'src'),str(ROOT)]
import torch
import matplotlib.pyplot as plt
from spno.config import DataConfig
from spno.checkpoints import CheckpointMetadata,CheckpointPayload
from scripts.run_phase6 import _model_from_checkpoint,run_gates,dispersion_arms
torch.set_num_threads(2)
audit=json.loads((OUT.parent/'audit.json').read_text())
cfg=DataConfig()
started=time.time()
gates=run_gates(cfg.domain,cfg)
print('Reference gates passed',flush=True)
records=[]
selected=[r for r in audit['records'] if r['arm'] in ('base','G6a')]
for r in selected:
    path=Path(r['checkpoint'])
    raw=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    meta=CheckpointMetadata(**raw['metadata'])
    model=_model_from_checkpoint(CheckpointPayload(meta,raw['state_dict']),cfg,expected_name=meta.model_name)
    model.load_state_dict(raw['state_dict'],strict=True)
    model.eval()
    tag=r['arm']+'/'+r['model']
    row=dict(tag=tag,seed=r['seed'],checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             saved_converged=meta.converged,policy='frozen-budget diagnostic; original convergence metadata unchanged',
             experiments=dispersion_arms({tag:model},cfg.domain,cfg))
    records.append(row)
    (OUT/(tag.replace('/','-')+f"-seed{r['seed']}.json")).write_text(json.dumps(row,indent=2),encoding='utf-8')
    derivative=row['experiments']['G5b']['alpha_derivative']['by_model'][tag]
    print(json.dumps({'tag':tag,'seed':r['seed'],'max_relative_alpha_derivative_error':{k:v['max_relative_error'] for k,v in derivative.items()},'elapsed_seconds':round(time.time()-started,1)}),flush=True)
summary={}
for tag in dict.fromkeys(r['tag'] for r in records):
    rs=sorted([r for r in records if r['tag']==tag],key=lambda r:r['seed'])
    summary[tag]={}
    for k in ('8','16','20','24','30','32'):
        vals=[r['experiments']['G5b']['alpha_derivative']['by_model'][tag][k]['max_relative_error'] for r in rs]
        summary[tag][k]={'mean':statistics.mean(vals),'min':min(vals),'max':max(vals),'by_seed':vals}
payload=dict(measured_at=datetime.now().astimezone().isoformat(),gates=gates,summary=summary,
             records=records,seconds=time.time()-started,
             limitations=['Plane-wave inputs are outside the random-field training distribution.',
                          'Metric is max relative alpha-derivative error over adjacent in-range alpha intervals, then mean across seeds.',
                          'These are frozen budget-bound weights, not a new convergence claim.',
                          'Alpha-continuation results query alpha=0 outside training and must not be interpreted as in-range derivative identification.',
                          'No G1-G4, G6a test sweep, G6b, G7, G9 or retraining was run by this script.'])
(OUT/'summary.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
fig,axs=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
colors={'A':'#7b6bb0','B-loop':'#c37d2d','C1':'#27824e','C2':'#247fa6','C3':'#be5454','A-wide':'#777777'}
for tag,entries in summary.items():
    arm,name=tag.split('/')
    xs=[int(k) for k in entries]
    ys=[entries[k]['mean'] for k in entries]
    axs[0 if arm=='base' else 1].semilogy(xs,ys,'-o',label=name,color=colors[name],markersize=4)
for ax,arm in zip(axs,('Base checkpoints','Multi-dt checkpoints')):
    ax.axvspan(8,32,color='#ddd',alpha=.2)
    ax.axhline(1,color='black',ls=':',lw=.8)
    ax.set(xlabel='Wave number k',ylabel='Relative error in alpha phase derivative',title=arm,xlim=(7,33))
    ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Frozen-weight G5b diagnostic: mean across 3 seeds\n1 = 100% relative error; plane waves are outside the training distribution',fontsize=12)
fig.savefig(OUT/'g5b-diagnostic.png',dpi=160)
plt.close(fig)
print('DONE '+json.dumps({'seconds':time.time()-started,'models':len(records),'output':str(OUT)}),flush=True)
