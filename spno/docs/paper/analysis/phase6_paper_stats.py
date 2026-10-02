"""Recompute the Phase 6 numbers quoted in docs/paper/*.md.

Reads only frozen-weight outputs already in the repo (no model loading, no training):
  results/phase6-diagnosis-2026-09-22/frozen-test/endpoint-diagnostics.json  (per-trajectory step-200 errors)
  results/phase6-diagnosis-2026-09-22/frozen-test/summary.json               (one-step, rollout, drift by seed)
  results/phase6-diagnosis-2026-09-22/frozen-probes/kinetic-rate-bounds.json (learned kappa(k) readout)
  results/phase6-diagnosis-2026-09-22/advanced-statistics.json               (validation curves / budgets)
Optional: --phase6-all <Phase6_all.ipynb> to also extract G1-G9 seed medians from its printed summary JSON.

Usage (from spno/):  python docs/paper/analysis/phase6_paper_stats.py [--phase6-all ~/Downloads/Phase6_all.ipynb]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # spno/
DIAG = ROOT / "results/phase6-diagnosis-2026-09-22"
OUT = Path(__file__).with_suffix(".json")


def per_trajectory():
    d = json.loads((DIAG / "frozen-test/endpoint-diagnostics.json").read_text())
    R, out = {}, {}
    for r in d["records"]:
        v = np.array(r["relative_error"]["values"])
        R[(r["tag"], r["seed"])] = v
        out.setdefault(r["tag"], {})[r["seed"]] = {
            "mean": float(v.mean()), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "max": float(v.max()), "n_gt_10pct": int((v > 0.1).sum()), "n_gt_100pct": int((v > 1).sum()),
            "phase_aligned_mean": r["phase_aligned_error"]["mean"],
            "density_error_mean": r["density_error"]["mean"],
        }
    paired = {}
    for other in ("base/A", "base/B-loop", "base/A-wide"):
        for s in range(3):
            a, b = R[("base/C1", s)], R[(other, s)]
            paired[f"C1 vs {other} seed{s}"] = {
                "c1_win_fraction": float(np.mean(a < b)), "median_ratio_c1_over_other": float(np.median(a / b))}
    for s in range(3):
        a, b = R[("G6a/C1", s)], R[("G6a/C2", s)]
        paired[f"C1-multidt vs C2-multidt seed{s}"] = {
            "c1_win_fraction": float(np.mean(a < b)), "median_ratio_c1_over_other": float(np.median(a / b))}
    return out, paired


def kinetic_fits():
    res = []
    for x in json.loads((DIAG / "frozen-probes/kinetic-rate-bounds.json").read_text()):
        k = np.array(x["ks"], float); c = np.array(x["curve"]); t = -x["alpha"] * k**2
        row = {"tag": x["tag"], "seed": x["seed"], "output_bound": x["rate_difference_bound_for_any_two_inputs"],
               "plateau": float(c[-1])}
        for lo, hi in ((0, 8), (0, 32)):
            m = (k >= lo) & (k <= hi)
            slope, icpt = np.linalg.lstsq(np.vstack([t[m], np.ones(m.sum())]).T, c[m], rcond=None)[0]
            rel = np.abs(c[m][1:] - t[m][1:]) / np.abs(t[m][1:])
            row[f"k{lo}-{hi}"] = {"slope": float(slope), "intercept": float(icpt),
                                  "corr": float(np.corrcoef(t[m], c[m])[0, 1]), "max_rel_err": float(rel.max())}
        row["rel_err_at_k"] = {int(kk): float(abs(c[kk] - t[kk]) / abs(t[kk])) for kk in (9, 10, 11, 12, 16, 32)}
        res.append(row)
    return res


def budgets():
    d = json.loads((DIAG / "advanced-statistics.json").read_text())["groups"]
    return {g: {k: v[k] for k in ("values", "epochs_run", "best_epochs_1based", "last5_improvement_pct",
                                  "final_val_over_online_train", "hours", "project_parameter_count",
                                  "real_scalar_parameter_count")} for g, v in d.items()}


def g_arms(nb_path: Path):
    nb = json.loads(nb_path.read_text())
    text = "".join("".join(o.get("text", "")) for o in nb["cells"][7]["outputs"])
    d = json.loads(text.strip().splitlines()[0])
    out = {}
    for arm, v in d.items():
        if not isinstance(v, dict):
            continue
        for m, r in v.items():
            if isinstance(r, dict) and "1step_seeds" in r:
                ro = [s["relative_error"] for s in r["roll_seeds"]]
                out.setdefault(arm, {})[m] = {
                    "one_step_median": st.median(r["1step_seeds"]), "rollout_by_seed": ro,
                    "energy_drift_by_seed": [s["energy_drift"] for s in r["roll_seeds"]]}
        if arm == "G9":
            ref = v["reference"][-1]
            out["G9"] = {m: v[m][-1] / ref for m in v if m != "reference"}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase6-all", type=Path, default=None)
    a = p.parse_args()
    traj, paired = per_trajectory()
    payload = {"per_trajectory_step200": traj, "paired": paired, "kinetic_fits": kinetic_fits(),
               "training_budgets": budgets()}
    if a.phase6_all and a.phase6_all.exists():
        payload["phase6_arms"] = g_arms(a.phase6_all)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
