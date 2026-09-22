# Colab checkpoint workflow implementation ledger

The approved specification is the user's implementation request in this task.

Goal: five Hebrew Colab notebooks, reusable Phase 6 artifacts, isolated training and
evaluation for Phases 7–9, exact epoch-boundary recovery for new training, and a
Dirichlet FEM demonstration on rectangles and disks.

Tasks:
- [x] Recoverable epoch persistence and legacy checkpoint compatibility.
- [x] Artifact import, experiment identity, preparation and execution catalog.
- [x] Phase 6–9 adapters and explicit CLI stages.
- [x] Dirichlet mesh domain and FEM solver.
- [x] Five notebooks, documentation and execution tests.
- [x] Full suite and independent final review.

Ruling: work on xom/colab-checkpoint-workflow in the current checkout, preserving
the pre-existing domain.py changes and pca.ipynb. A new checkout would omit the
uncommitted domain implementation the approved extension depends on.
Ruling: preserve existing CLI defaults; the new notebooks always select a stage
explicitly. Legacy artifacts are read-only inputs, never implicitly upgraded.
Pre-flight: training progress consumes the same experiment identity as the catalog;
phase adapters and notebooks consume the same catalog, without separate trainers.


Completion evidence (2026-09-22):
- Baseline: 312 passed, 5 skipped before optional dependencies were installed.
- Final full suite: `.venv/bin/python -m pytest tests -q` — 350 passed in 25.41s.
- All five notebooks executed end-to-end under real local Jupyter kernels with
  optimizer steps prohibited; Colab's external authentication/upload UI was not
  exercised locally. Only tiny synthetic training fixtures were used.
- Exact CPU epoch recovery verified for one-step, PINO and multi-dt training,
  including recovery after JSON source-config serialization.
- Independent read-only review identified four issues; all were fixed and covered:
  canonical JSON recovery context, distinct requested-vs-source protocols,
  source kinetics for Phase 8, and all Phase 9 test fingerprints in report identity.
- Additional regression coverage prevents relabeling modified source data across
  runtime restarts, silently changing imported seeds in CLI stages, implicit
  evaluation training, and retraining after checkpoint corruption.
- `git diff --check` passed. Portable source ZIP verified byte-for-byte against
  all 59 source entries, including five notebooks; ZIP integrity check passed.
- The earlier final-suite attempt was stopped by the approval service's usage
  limit. The same authorized test operation was retried successfully on resume.

Ruling: Phase 9 intentionally retains its K0 controlled comparison; incompatible
source kinetics require separate explicitly selected controls. Phase 8 preserves
source kinetics. Source protocol declarations are immutable once recorded;
requested experiment overrides are separate in notebooks, API and CLI.
Ruling: implementation remains on xom/colab-checkpoint-workflow in the shared
checkout for desktop review. No merge, push, publication, or original-run mutation
was performed. Pre-existing domain.py and pca.ipynb changes remain untouched.
