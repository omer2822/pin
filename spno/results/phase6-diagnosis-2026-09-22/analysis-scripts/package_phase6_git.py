"""Prepare the analysis archive for a Git branch without touching the repository."""
import ast, hashlib, json, os, re, shutil
from pathlib import Path
BASE=Path('C:/Users/omerm/Documents/Codex/2026-09-18/he')
REPO=Path('C:/Users/omerm/PycharmProjects/pin')
SOURCE=BASE/'outputs/phase6-diagnosis-2026-09-22'
STAGE=BASE/'work/phase6-git-bundle'
DEST=STAGE/'spno/results/phase6-diagnosis-2026-09-22'
TRAINING='spno/results/phase6-training-bd4e108527-K0-standalone/metrics.json'
SCRIPTS=['audit_phase6.py','phase6_advanced_analysis.py','phase6_frozen_probes.py',
         'phase6_frozen_test.py','phase6_endpoint_diagnostics.py','phase6_kinetic_bounds.py']
DEST.mkdir(parents=True,exist_ok=True)
for src in SOURCE.rglob('*'):
    if src.is_file():
        dst=DEST/src.relative_to(SOURCE)
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
(DEST/'analysis-scripts').mkdir(exist_ok=True)
for name in SCRIPTS:
    src=BASE/'work'/name
    ast.parse(src.read_text(encoding='utf-8'),filename=name)
    shutil.copy2(src,DEST/'analysis-scripts'/name)
(STAGE/TRAINING).parent.mkdir(parents=True,exist_ok=True)
shutil.copy2(REPO/TRAINING,STAGE/TRAINING)

def mapped_link(match,current):
    old=Path(match.group(1))
    if old.is_relative_to(SOURCE): new=DEST/old.relative_to(SOURCE)
    elif old.is_relative_to(BASE/'work') and old.name in SCRIPTS: new=DEST/'analysis-scripts'/old.name
    elif old.is_relative_to(REPO): new=STAGE/old.relative_to(REPO)
    else: return match.group(0)
    return ']('+Path(os.path.relpath(new,current.parent)).as_posix()+')'
for md in DEST.rglob('*.md'):
    body=md.read_text(encoding='utf-8')
    body=re.sub(r'\]\(/?(C:/[^)]+)\)',lambda m:mapped_link(m,md),body)
    md.write_text(body,encoding='utf-8')

artifacts=REPO/'spno/results/phase6-standalone-artifacts'
items=[]
for path in sorted(artifacts.rglob('*.pt')):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024),b''): digest.update(chunk)
    items.append(dict(path=path.relative_to(REPO).as_posix(),bytes=path.stat().st_size,sha256=digest.hexdigest(),
                      kind='checkpoint' if 'checkpoints' in path.parts else 'dataset'))
assert len(items)==62
manifest=dict(version=1,policy='Original .pt files remain local, as required by the existing repository .gitignore. This manifest is not a backup of their bytes.',
              n_checkpoints=sum(x['kind']=='checkpoint' for x in items),n_datasets=sum(x['kind']=='dataset' for x in items),
              total_bytes=sum(x['bytes'] for x in items),artifacts=items)
(DEST/'local-artifacts-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(DEST/'README.md').write_text('''# Phase 6 frozen-weight analysis — 2026-09-22

Start with [the updated Hebrew research synthesis](research-synthesis-he.md).
This archive includes the completed training audit, new frozen-weight evaluations,
figures, per-seed metrics, and the exact analysis-script snapshots used locally.
It is an analysis archive, not a fix to the Phase 6 resume CLI.

## Contents

- [Original training metrics](../phase6-training-bd4e108527-K0-standalone/metrics.json): all 36 training jobs.
- [Training/checkpoint audit](audit.json) and [training-history statistics](advanced-statistics.json).
- [G5a/G5b spectral probes](frozen-probes/summary.json): 27 checkpoints.
- [Held-out test and 200-step rollouts](frozen-test/summary.json): 27 checkpoints, all 100 test trajectories.
- [Trajectory-tail, phase, and density diagnostics](frozen-test/endpoint-diagnostics.json): 18 checkpoints.
- [Direct C1 kinetic rates and analytic bounds](frozen-probes/kinetic-rate-bounds.json).
- [Evaluation code provenance](verification-manifest.json).
- [Local dataset/checkpoint inventory and SHA-256 hashes](local-artifacts-manifest.json).
- `analysis-scripts/`: six original script snapshots; model-training code is unchanged.

## Reproduction and scope

The Python scripts preserve their original Windows paths and environment setup for
provenance. Before running on another machine or Colab, adjust their project/output
paths and Python imports and restore the required artifact files. The raw JSON also
retains original source paths. The Markdown links are relative for GitHub browsing.

Run audit_phase6.py before phase6_advanced_analysis.py, phase6_frozen_probes.py, and
phase6_frozen_test.py. Endpoint and kinetic diagnostics consume the frozen-test
summary. These scripts do not train models. They are not a portable Colab notebook
or a complete evaluation-only Phase 6 runner.

The existing repository .gitignore excludes .pt datasets and checkpoints. The 36
checkpoints and 26 data shards (approximately 1.33 GB in total) therefore remain
local. Their hashes are recorded here; this Git branch alone cannot restore their
contents or perform inference without those files.

The original run finished training and then stopped at the budget-bound checkpoint
gate. New evaluations explicitly use the frozen training budget and leave that
metadata unchanged. G1-G4, the full G6a sweep, G6b, G7, and G9 remain outstanding.
Earlier diagnosis reports retain their original conclusions with an update notice;
the research synthesis incorporates the new measurements and takes precedence.
''',encoding='utf-8')
files=sorted(p for p in STAGE.rglob('*') if p.is_file())
for path in files:
    if path.suffix=='.json': json.loads(path.read_text(encoding='utf-8'))
    if path.suffix=='.md':
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' not in target:
                candidate=(path.parent/target).resolve()
                fallback=REPO/candidate.relative_to(STAGE.resolve()) if candidate.is_relative_to(STAGE.resolve()) else candidate
                assert candidate.exists() or fallback.exists(),(path,target)
assert len(files)==78,(len(files),[p.name for p in files])
assert not any(p.suffix=='.pt' for p in files)
print(json.dumps(dict(files=len(files),bytes=sum(p.stat().st_size for p in files),checkpoint_inventory=manifest['n_checkpoints'],dataset_inventory=manifest['n_datasets'],stage=str(STAGE)),indent=2))
