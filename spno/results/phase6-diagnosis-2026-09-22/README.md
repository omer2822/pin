# Phase 6 frozen-weight analysis — 2026-09-22

Start with [the updated Hebrew research synthesis](research-synthesis-he.md).
For the laptop switch, follow [the Mac and Colab handoff](HANDOFF_MAC.md).
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
