"""Inspect C1 kinetic rates without phase wrapping; retain the gauge-free difference."""
import sys,json,os
from pathlib import Path
root=Path('C:/Users/omerm/PycharmProjects/pin/spno')
out=Path('C:/Users/omerm/Documents/Codex/2026-09-18/he/outputs/phase6-diagnosis-2026-09-22')
os.environ['MPLCONFIGDIR']='C:/Users/omerm/Documents/Codex/2026-09-18/he/work/mplconfig'
sys.path[:0]=[str(root/'.venv/Lib/site-packages'),str(root/'src'),str(root)]
import torch
import matplotlib.pyplot as plt
from spno.config import DataConfig
from spno.checkpoints import CheckpointMetadata,CheckpointPayload
from scripts.run_phase6 import _model_from_checkpoint
from spno.precision import widen_to_double
s=json.loads((out/'frozen-test/summary.json').read_text())
rows=[]
fig,ax=plt.subplots(figsize=(8.8,5.0),layout='constrained')
ks=list(range(33))
ax.plot(ks,[-.9*k*k for k in ks],color='black',label='Physical rate difference: -0.9 k^2',lw=2)
with torch.no_grad():
 for r in s['records']:
  if not r['tag'].endswith('/C1'):continue
  raw=torch.load(r['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
  meta=CheckpointMetadata(**raw['metadata'])
  m=_model_from_checkpoint(CheckpointPayload(meta,raw['state_dict']),DataConfig(),expected_name=meta.model_name)
  m.load_state_dict(raw['state_dict'],strict=True)
  m=widen_to_double(m,device='cpu').eval()
  rate=m.kinetic(torch.tensor([[.9,.3]],dtype=torch.float64))[0]
  last=m.kinetic.network.network[-1]
  diff_bound=2*float(last.weight.abs().sum())
  curve=[float(rate[k]-rate[0]) for k in ks]
  row=dict(tag=r['tag'],seed=r['seed'],checkpoint_sha256=r['checkpoint_sha256'],
    alpha=.9,beta=.3,rate_difference_bound_for_any_two_inputs=diff_bound,
    gauge_free_kappa={str(k):curve[k] for k in (8,16,32)},
    truth={str(k):-.9*k*k for k in (8,16,32)},ks=ks,curve=curve)
  rows.append(row)
  base=r['tag'].startswith('base')
  ax.plot(ks,curve,color='#27824e' if base else '#247fa6',ls='-' if base else '--',alpha=.7,
          label=('C1 base (3 seeds)' if base else 'C1 multi-dt (3 seeds)') if r['seed']==0 else None)
ax.axvspan(0,8,color='#87c99b',alpha=.15,label='Initial training bandwidth')
ax.set(xlabel='Wave number k',ylabel='Kappa(k) - kappa(0)',title='C1 learns the low-frequency rate and saturates outside it\nDirect rate readout, alpha=0.9, beta=0.3; no phase unwrapping')
ax.grid(alpha=.2);ax.legend(fontsize=9)
fig.savefig(out/'frozen-probes/kinetic-saturation.png',dpi=170)
plt.close(fig)
(out/'frozen-probes/kinetic-rate-bounds.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print('Saved six kinetic curves and analytic bounds.')
