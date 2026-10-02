# Paper & Phase 6 analysis (2026-10-01)

| File | What it is |
| --- | --- |
| `paper-draft-v1.md` / `.docx` | Paper draft v1 — "Small by Structure": C1 vs FNO, generalization, dispersion readout, component swaps |
| `phase6-skeptical-review.md` | Skeptical reviewer-style analysis of Phase 6: fairness audit, arm-by-arm verdicts, failure modes, decisive next experiments |
| `phase6-results-draft-v0.md` | Earlier short results draft (superseded by v1, kept for history) |
| `figures/` | Figures used by the drafts (exported from the live docs) |
| `analysis/phase6_paper_stats.py` | Recomputes the per-trajectory, paired and kinetic-fit numbers quoted in the drafts from `results/phase6-diagnosis-2026-09-22/` |
| `analysis/phase6_paper_stats.json` | Output of that script |

Live (editable) versions:
- Paper draft v1: https://claude.ai/code/artifact/bb6e8143-7826-4fbb-b842-9a614e3c4d83
- Skeptical review: https://claude.ai/code/artifact/5ee3675f-afd5-4003-a562-27a0fc986c1b
- Results draft v0: https://claude.ai/code/artifact/9578e976-b896-4f72-b765-7f30c90b5fa3

Numbers for G1–G9 come from `Phase6_all.ipynb` (Downloads, printed summary JSON), the component swaps from
`12_gauge_identifiable_c1.ipynb` and `learned_kinetic+exact_local.ipynb`, Phase 7 from `comparison.csv`.
Open items marked `[confirm]` / `[cite]` in the paper draft.
