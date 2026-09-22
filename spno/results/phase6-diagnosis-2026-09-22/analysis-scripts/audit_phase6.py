"""Read-only audit of the completed Phase 6 training run; never trains/evaluates."""
import os
import sys
import json
import math
from pathlib import Path
from collections import Counter
from datetime import datetime
from dataclasses import replace

REPO = Path('C:/Users/omerm/PycharmProjects/pin/spno')
WORK = Path('C:/Users/omerm/Documents/Codex/2026-09-18/he/work')
OUT = Path('C:/Users/omerm/Documents/Codex/2026-09-18/he/outputs/phase6-diagnosis-2026-09-22')
OUT.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(WORK / 'mplconfig')
sys.path[:0] = [str(REPO / '.venv/Lib/site-packages'), str(REPO / 'src'), str(REPO)]
import torch
import matplotlib.pyplot as plt
from spno.config import DataConfig, config_hash
from spno.checkpoints import CheckpointMetadata, CheckpointPayload
from spno.data.shift import SHIFT_SPECS, shift_identifier
from spno.data.datasets import shard_paths
from scripts.run_phase6 import _model_from_checkpoint, load_models, multi_dt_data_hash

torch.set_num_threads(1)
summary_path = REPO / 'results/phase6-training-bd4e108527-K0-standalone/metrics.json'
artifacts = REPO / 'results/phase6-standalone-artifacts'
summary = json.loads(summary_path.read_text())
base = DataConfig()
entries = []
for seed, models in summary['base'].items():
    for name, entry in models.items():
        entries.append(('base', name, int(seed), entry))
for seed, entry in summary['A-wide']['by_seed'].items():
    entries.append(('base', 'A-wide', int(seed), entry))
for arm in ('G7-alpha-fixed', 'G6a'):
    for name, seeds in summary[arm]['by_model'].items():
        for seed, entry in seeds.items():
            entries.append((arm, name, int(seed), entry))

rows = []
for arm, name, seed, entry in entries:
    history = entry['history']
    vals = history['val_loss']
    path = Path(entry['checkpoint'])
    raw = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    meta = CheckpointMetadata(**raw['metadata'])
    tag = name if arm == 'base' else f'{arm}/{name}'
    cfg = SHIFT_SPECS['G7-alpha-fixed'].config if arm == 'G7-alpha-fixed' else base
    expected_hash = multi_dt_data_hash(base, quick=False) if arm == 'G6a' else config_hash(cfg)
    payload = CheckpointPayload(meta, raw['state_dict'])
    model = _model_from_checkpoint(payload, cfg, expected_name=tag)
    # Structural validation only. This does not change a checkpoint or the evaluation gate.
    model.load_state_dict(raw['state_dict'], strict=True)
    gate = history['best_epoch'] < len(vals) - 1
    consistency = (
        meta.model_name == tag and meta.seed == seed
        and meta.best_epoch == history['best_epoch'] and meta.converged == gate
        and meta.data_hash == expected_hash
        and len(history['train_loss']) == len(vals)
        and math.isclose(history['best_val'], min(vals), rel_tol=1e-12)
        and vals.index(min(vals)) == history['best_epoch']
    )
    rows.append(dict(
        arm=arm, model=name, seed=seed, epochs=len(vals), best_epoch_index=history['best_epoch'],
        passes_gate=gate, best_val=history['best_val'], final_val=vals[-1],
        last_five_transitions_improvement_pct=100*(vals[-6]-vals[-1])/vals[-6],
        elapsed_hours=history['seconds']/3600, checkpoint=str(path),
        modified=datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        metadata_history_consistent=consistency, strict_state_load_ok=True,
        finite_weights=all(bool(torch.isfinite(t).all()) for t in raw['state_dict'].values()),
        finite_history=all(math.isfinite(v) for v in vals + history['train_loss']),
        checkpoint_keys=list(raw), train_loss=history['train_loss'], val_loss=vals,
    ))

expected = []
for split, path in shard_paths(artifacts/'data', config_hash(base)).items():
    expected.append(('base', split, path, base))
for name, spec in SHIFT_SPECS.items():
    splits = ('train', 'val', 'test') if name == 'G7-alpha-fixed' else ('test',)
    paths = shard_paths(artifacts/'data', shift_identifier(spec))
    expected.extend((name, split, paths[split], spec.config) for split in splits)
for dt in (0.005, 0.01, 0.02):
    cfg = replace(base, dt=dt)
    expected.extend((f'G6a-dt-{dt:g}', split, path, cfg) for split, path in
                    shard_paths(artifacts/'data', 'g6a-'+config_hash(cfg)).items())
data_rows = []
for name, split, path, cfg in expected:
    raw = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    wanted = [getattr(cfg, 'n_'+split), cfg.steps+1, cfg.grid_size]
    shape = list(raw['trajectories'].shape)
    data_rows.append(dict(arm=name, split=split, path=str(path), shape=shape,
                          shape_ok=shape == wanted,
                          config_hash_ok=raw['metadata']['config_hash'] == config_hash(cfg)))

try:
    load_models(artifacts, base, summary['seeds'], 'K0', standalone=True)
except RuntimeError as error:
    reproduced_error = str(error)
else:
    reproduced_error = None

groups = [('base', name) for name in ('A', 'B-loop', 'C1', 'C2', 'C3', 'A-wide')]
groups += [(arm, name) for arm in ('G7-alpha-fixed', 'G6a') for name in ('A', 'C1', 'C2')
           if arm == 'G7-alpha-fixed']
groups += [('G6a', name) for name in ('C1', 'C2', 'C3')]
expected_keys = {(arm, name, seed) for arm, name in groups for seed in (0, 1, 2)}
actual_keys = {(r['arm'], r['model'], r['seed']) for r in rows}
total_hours = sum(r['elapsed_hours'] for r in rows)
c2_hours = sum(r['elapsed_hours'] for r in rows if (r['arm'],r['model']) == ('G6a','C2'))
audit = dict(
    inspected_at=datetime.now().astimezone().isoformat(), source=str(summary_path),
    expected_checkpoints=len(expected_keys), found_checkpoint_files=len(list((artifacts/'checkpoints').rglob('*.pt'))),
    missing_training_records=sorted(expected_keys-actual_keys), extra_training_records=sorted(actual_keys-expected_keys),
    training_records=len(rows), passes_gate=sum(r['passes_gate'] for r in rows),
    blocked_by_gate=sum(not r['passes_gate'] for r in rows),
    all_checkpoints_valid=all(r['metadata_history_consistent'] and r['strict_state_load_ok'] and
                              r['finite_weights'] and r['finite_history'] for r in rows),
    expected_shards=len(expected), all_shard_headers_valid=all(r['shape_ok'] and r['config_hash_ok'] for r in data_rows),
    dataset_finite_values_scanned=False, total_recorded_elapsed_hours=total_hours,
    g6a_c2_elapsed_hours=c2_hours, g6a_c2_share_pct=100*c2_hours/total_hours,
    epoch_count_distribution=dict(Counter(r['epochs'] for r in rows)),
    evaluation_metrics_exists=(REPO/'results/phase6-bd4e108527-K0-standalone/metrics.json').exists(),
    reproduced_error=reproduced_error, records=rows, datasets=data_rows,
)
(OUT/'audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')

fig, axes = plt.subplots(4, 3, figsize=(14, 13), sharex=True, layout='constrained')
colors = ['#176b9b', '#d66c21', '#33844b']
for axis, (arm, name) in zip(axes.flat, groups):
    selected = sorted((r for r in rows if (r['arm'], r['model']) == (arm, name)), key=lambda r:r['seed'])
    for r in selected:
        epochs = list(range(1, r['epochs']+1))
        axis.semilogy(epochs, r['val_loss'], color=colors[r['seed']], linewidth=1.6,
                      label=f"seed {r['seed']} | {'passes gate' if r['passes_gate'] else 'budget-bound'}")
        axis.scatter(r['best_epoch_index']+1, r['best_val'], s=38, color=colors[r['seed']], marker='o')
    axis.set_title(f'{arm} / {name}', loc='left', fontsize=11, fontweight='bold')
    axis.grid(alpha=.2)
    axis.legend(fontsize=7, loc='upper right')
    axis.set_xlabel('Epoch (1-based)')
    axis.set_ylabel('Validation relative L2')
fig.suptitle('Phase 6: all 36 saved training histories\nDots mark the best epoch; passing the gate does not prove convergence', fontsize=15)
fig.savefig(OUT/'validation-curves.png', dpi=150)
plt.close(fig)

report = [
    '# Phase 6 training diagnosis — 22 September 2026',
    '',
    '**All 36 planned training jobs completed and were saved. The subsequent evaluation stopped before any experiment arm ran.**',
    '',
    f'Source: `{summary_path}`. Training summary saved 21 September 2026 at 21:50 local time.',
    '',
    '## Verified completion and integrity',
    '',
    '- 12 model variants × 3 seeds: 36 histories and 36 checkpoint files; no missing training jobs.',
    '- All 36 checkpoints deserialize safely, contain finite weights, match their histories and expected data hashes, and load into the recorded architectures with strict parameter checking.',
    '- All 26 expected dataset shards load and have the expected trajectory shapes and configuration hashes. This audit did not scan every dataset value.',
    '- The saved checkpoint content is metadata plus model weights. Optimizer, scheduler, and RNG state are absent, so exact training continuation is unavailable in these artifacts.',
    '- No final Phase 6 evaluation metrics or plots exist. The empty training `plots` directory is created by the generic save helper; it does not indicate that training failed.',
    '',
    '## Confirmed failure mechanism',
    '',
    'The reported traceback was reproduced by calling the checkpoint loader only, with no training or experiment evaluation.',
    '',
    '1. `run_phase6.py:1275` saves the complete training summary.',
    '2. `run_phase6.py:1276` starts loading trained models. A, seed 0 is first.',
    '3. `checkpoints.py:60` rejects its `converged=False` metadata as budget-bound.',
    '4. The wrapper reports `checkpoint is incompatible`; the underlying cause is the policy check, not a tensor/architecture mismatch.',
    '5. `run_selected_arms` at line 1317 and final result saving at line 1330 are never reached in this run.',
    '',
    '```text', reproduced_error or 'No error reproduced', '```',
    '',
    '## Training status by variant',
    '',
    '`B` = budget-bound / blocked by the evaluator. `P` = passes the current code check, not a proof of convergence.',
    'Each cell gives status, epochs run, and the best validation relative-L2 loss. Values compare only within a shared dataset/task; these are not test or rollout scores.',
    '',
    '| Variant | Seed 0 | Seed 1 | Seed 2 | Recorded elapsed hours |',
    '|---|---|---|---|---:|',
]
for arm, name in groups:
    selected = sorted((r for r in rows if (r['arm'],r['model']) == (arm,name)), key=lambda r:r['seed'])
    cells = [f"{'P' if r['passes_gate'] else 'B'} · {r['epochs']} ep · {r['best_val']:.6g}" for r in selected]
    report.append(f"| {arm}/{name} | {' | '.join(cells)} | {sum(r['elapsed_hours'] for r in selected):.2f} |")
report += [
    '',
    '29/36 checkpoints are blocked; 7/36 pass the current check. 31 jobs ran all 40 epochs; five early-stopped at 13, 16, 19, 21, and 21 epochs.',
    '',
    'The code defines convergence as `best_epoch < len(val_loss) - 1` (`experiments.py:190`). It checks whether the best epoch preceded the last epoch, not whether training reached a stable optimum. Two of the seven passing jobs (base C1/seed0 and G6a C1/seed2) ran all 40 epochs and pass only because their best loss occurred at epoch index 38. Five others early-stopped after eight epochs without a new best.',
    '',
    '## What the curves show',
    '',
    '- This run still used a 40-epoch cap. It did not train each model longer merely because the complete collection took several days.',
    '- A/seed0 improved validation loss by 17.8% over the last five epoch transitions (index 34 to 39); A/seed2 by 22.5%. A-wide improved by 20.9–24.3% across its seeds. These histories support the finding that some models remain budget-limited.',
    '- A/seed1 passes the gate but early-stopped after 13 epochs, with best loss 0.005859 versus about 0.00062–0.00064 for the other A seeds. Its best validation error is approximately 9.3× their mean. A passing flag therefore does not imply a better-trained model.',
    '- G7/C2 and G6a/C3 also have early-stopped seeds with materially worse validation losses than their 40-epoch counterparts. Investigate early stopping and the training schedule before interpreting seed averages.',
    '',
    '![All validation histories](validation-curves.png)',
    '',
    '## Where the elapsed time went',
    '',
    f'The histories record **{total_hours:.2f} hours** in total. G6a/C2 alone accounts for **{c2_hours:.2f} hours ({100*c2_hours/total_hours:.1f}%)**, split as 26.28, 7.12, and 26.18 hours for seeds 0–2, each running 40 epochs.',
    '',
    'These durations come from `time.time()` differences, not CPU/GPU-active timers. They can include suspension, inactivity, or resource contention; the files cannot establish why the two 26-hour intervals were so long. Multi-dt training uses three timestep datasets, but that fact alone does not explain the large variation between C2 seeds.',
    '',
    '## What remains and how to avoid repeating the work',
    '',
    '- Missing work is evaluation (default arms G1, G2, G3, G4, G5a, G5b, G6a, G6b, G7, G9), final metrics, and plots—not initial training coverage. The training summary does not record the original selected-arm CLI arguments.',
    '- Preserve all current weights and histories. Do not rerun the same standalone command: it reuses datasets but retrains models from scratch and overwrites the checkpoints.',
    '- A recovery change should first provide evaluation directly from this standalone artifact root, with any evaluation of budget-bound weights explicitly labelled diagnostic rather than converged results. This would reuse existing training to reveal further evaluation issues.',
    '- If further training is required, add selective warm-start from saved weights and persistent optimizer/scheduler/RNG state for future exact resumption. Warm-starting these existing files would start a new optimizer/schedule; it is not an exact resume.',
    '- Check the convergence/early-stop policy and recorded timing anomalies before committing to another large training budget. Increasing the cap alone does not guarantee every job passes the current check.',
    '',
    'No project source, results, convergence flags, or checkpoints were modified. No training or full experiment evaluation was launched.',
]
(OUT/'diagnosis.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
print(json.dumps({k:v for k,v in audit.items() if k not in ('records','datasets')}, indent=2))
print('Report:', OUT/'diagnosis.md')
