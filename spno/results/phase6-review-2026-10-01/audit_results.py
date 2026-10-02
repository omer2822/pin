"""Read-only secondary audit of the supplied Phase 6 artifacts; no training."""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from spno.config import DataConfig
from spno.checkpoints import load_checkpoint_payload
from spno.precision import widen_to_double
from spno.models.fno import FNOStepOperator
from spno.models.split_learned import DensityPhaseSplitStep, FieldDensityPhaseSplitStep, FullFieldPhaseSplitStep
from spno.evaluation.dispersion import dispersion_curve, probe_amplitude
from spno.models.base import allow_dt_transfer

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'results/phase6-diagnosis-2026-09-22'
torch.set_num_threads(2)

def fit(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    slope,intercept=np.polyfit(x,y,1)
    return dict(correlation=float(np.corrcoef(x,y)[0,1]),slope=float(slope),intercept=float(intercept),
                relative_rmse=float(np.linalg.norm(y-x)/np.linalg.norm(x)),
                relative_fit_residual=float(np.linalg.norm(y-(slope*x+intercept))/np.linalg.norm(y)))

output={'source_note':'Fresh calculations from stored JSON and frozen checkpoints. Notebook arm summaries are rounded to four significant figures.'}
records=json.loads((OLD/'frozen-probes/summary.json').read_text())['records']
fits=[]
for r in records:
    curve=next(iter(r['experiments']['G5a']['curves'].values()))
    x=np.asarray(curve['truth']); y=np.asarray(curve['principal']); k=np.asarray(curve['wave_numbers'])
    derivatives=next(iter(r['experiments']['G5b']['alpha_derivative']['by_model'].values()))
    dx=[v['truth'] for v in derivatives.values()]; dy=[v['estimate'] for v in derivatives.values()]
    fits.append({'tag':r['tag'],'seed':r['seed'],'G5a_low_0_8':fit(x[k<=8],y[k<=8]),
                 'G5a_0_18':fit(x[k<=18],y[k<=18]),'G5a_truth_assisted_all':fit(x,curve['unwrapped']),
                 'G5b_all':fit(dx,dy),
                 'G5b_endpoints':{str(q):derivatives[str(q)] for q in (8,16,32)}})
output['dispersion_fits']=fits
rates=json.loads((OLD/'frozen-probes/kinetic-rate-bounds.json').read_text())
output['kinetic_fits']=[{'tag':r['tag'],'seed':r['seed'],
   'low_0_8':fit(-.9*np.arange(9)**2,r['curve'][:9]),
   'all_0_32':fit(-.9*np.arange(33)**2,r['curve']),
   'relative_endpoint_errors':{k:abs(v-r['truth'][k])/abs(r['truth'][k]) for k,v in r['gauge_free_kappa'].items()}}
   for r in rates]
cfg=DataConfig()
models={'A':FNOStepOperator(cfg.domain),'A-wide':FNOStepOperator(cfg.domain,modes=32),
        'C1':DensityPhaseSplitStep(cfg.domain),'C2':FieldDensityPhaseSplitStep(cfg.domain),
        'C3':FullFieldPhaseSplitStep(cfg.domain)}
output['parameter_counts']={n:{'tensor_elements':sum(p.numel() for p in m.parameters()),
                             'real_scalars':sum(p.numel()*(2 if p.is_complex() else 1) for p in m.parameters())}
                           for n,m in models.items()}
training=json.loads((ROOT/'results/phase6-training-bd4e108527-K0-standalone/metrics.json').read_text())
histories=[]
groups=[('base/'+n,int(s),m) for s,ms in training['base'].items() for n,m in ms.items()]
groups += [('base/A-wide',int(s),m) for s,m in training['A-wide']['by_seed'].items()]
groups += [(g+'/'+n,int(s),m) for g in ('G6a','G7-alpha-fixed') for n,ss in training[g]['by_model'].items() for s,m in ss.items()]
for tag,s,m in groups:
 h=m['history']; val=h['val_loss']; train=h['train_loss']; j=h['best_epoch']
 histories.append({'tag':tag,'seed':s,'epochs':len(val),'best_epoch':j,'best_val':h['best_val'],
  'train_at_selected_epoch':train[j], 'last_train':train[-1],'last_val':val[-1],
  'val_change_last5':(val[max(0,len(val)-6)]-val[-1])/val[max(0,len(val)-6)],'seconds':h['seconds']})
output['histories']=histories

checkpoints=ROOT/'results/phase6-standalone-artifacts/checkpoints/phase6/bd4e108527-K0'
curves=[]; cp_metadata=[]
for name in ('C1','C2','C3'):
 for seed in (0,1,2):
  p=checkpoints/f'{name}-seed{seed}.pt'
  if not p.exists(): continue
  payload=load_checkpoint_payload(p)
  m=models[name]; m.load_state_dict(payload.state_dict); m=widen_to_double(m,device='cpu').eval()
  cp_metadata.append({'name':name,'seed':seed,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                      'metadata':payload.metadata.__dict__})
  for dt in (.005,.01,.02):
   with allow_dt_transfer(m):
    c=dispersion_curve(m,cfg.domain,range(33),alpha=.9,beta=.3,
                       amplitude=probe_amplitude(cfg.domain,cfg.mass_range),potential_constant=0.,dt=dt)
   curves.append({'name':name,'seed':seed,'dt':dt,'curve':c.as_dict()})
output['G6b_fresh_curves']=curves; output['checkpoint_metadata']=cp_metadata
output['missing_artifacts_note']='Full production checkpoints/data are absent on this Mac. No new trained-model inference claimed. Quick checkpoints excluded.' if not curves else ''

spectral=[]
for split in ('train','test'):
 p=ROOT/f'results/phase6-standalone-artifacts/data/nls1d-bd4e108527/{split}.pt'
 if not p.exists():
  possible=list((ROOT/'data').rglob(f'{split}.pt')); possible=[p for p in possible if 'bd4e108527' in str(p)]
  if not possible: continue
  p=possible[0]
 shard=torch.load(p,map_location='cpu',weights_only=False)
 print('SHARD',split,list(shard),flush=True)
 x=shard['trajectories']; k=torch.fft.fftfreq(64,d=1/64).abs(); sums={q:[] for q in (8,15,18)}
 for start in range(0,len(x),40):
  power=torch.fft.fft(x[start:start+40]).abs().square(); total=power.sum(-1)
  for q in sums: sums[q].append((power[...,k>q].sum(-1)/total).reshape(-1))
 spectral.append({'split':split,'source':str(p),'trajectories':len(x),
                  'fractions':{str(q):{'mean':float(torch.cat(v).mean()),'max':float(torch.cat(v).max()),
                                      'initial_mean':float((torch.fft.fft(x[:,0]).abs().square()[...,k>q].sum(-1)/torch.fft.fft(x[:,0]).abs().square().sum(-1)).mean())}
                               for q,v in sums.items()}})
output['spectral_occupancy']=spectral
(OUT/'fresh-audit.json').write_text(json.dumps(output,indent=2,default=str))
print('Wrote fresh-audit.json',flush=True)
