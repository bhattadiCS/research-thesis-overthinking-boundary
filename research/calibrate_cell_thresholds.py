"""Calibrated per-cell drift-threshold table (productization of protocol P3d).

Background (ThesisDocs/rigor_audit/05_offline_experiment_results.md §P3d): a
single scalar offset delta per cell on the hazard-drift stop rule
(stop at first step>=2 with mu_hat <= delta) was the only policy lever that
survived its pre-registration, worth +1,545.85 net utility out-of-fold on the
75,965-run corpus.

Leakage discipline (read before citing numbers):
  * The OOF columns (oof_dU, oof_dissolved, oof_win2loss, fold delta range)
    come from GroupKFold(5)-by-task_id fits evaluated on held-out folds —
    these are the HONEST performance estimates.
  * delta_full is fit on ALL of the cell's runs and is the DEPLOYMENT value
    (standard practice: validate the procedure out-of-fold, ship the full-data
    fit). Its cal_full_* companion columns are in-sample by construction and
    must never be quoted as expected gains — quote oof_dU.
  * Never tune delta on the same runs used to claim a gain.

Method: replicates the production scoring exactly (same sanitizer, temporal
features, GroupKFold(5)-by-run_id fold structure and per-fold model fits as
trace_analysis.out_of_sample_eval), recovers each run's out-of-fold per-step
mu_hat series, and VALIDATES that the recomputed first crossing equals the
recorded hazard_drift stop in detector_comparison_by_run.csv (mismatch counts
are reported per cell; the 2026-07-10 run matched 75,965/75,965).

Usage:
    python research/calibrate_cell_thresholds.py \
        [--matrix-root research/outputs/experiment_matrix] \
        [--out-csv research/outputs/experiment_matrix/_aggregate/calibrated_cell_thresholds.csv]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
STEP_COST = 0.05
DELTAS = np.round(np.arange(-0.15, 0.1501, 0.005), 3)

spec = importlib.util.spec_from_file_location("ta", HERE / "trace_analysis.py")
ta = importlib.util.module_from_spec(spec)
sys.modules["ta"] = ta
spec.loader.exec_module(ta)


def util(c: float, s: float) -> float:
    return float(c) - STEP_COST * (s - 1.0)


def stop_idx(steps, mu, delta):
    for i in range(len(steps)):
        if steps[i] >= 2 and mu[i] <= delta:
            return i
    return len(steps) - 1


def recover_series(cell_dir: Path):
    """Rebuild each run's out-of-fold mu series exactly as production scored it."""
    dcbr = pd.read_csv(cell_dir / "detector_comparison_by_run.csv")
    hd = dcbr[dcbr.detector == "hazard_drift"].set_index("run_id")
    ns = dcbr[dcbr.detector == "never_stop"].set_index("run_id")
    sf = pd.read_csv(cell_dir / "trace_steps.csv")
    sf, _ = ta._sanitize_step_frame(sf)
    sf = ta.add_temporal_features(sf)
    sf["repair"] = ((sf["correct"] == 0) & (sf["next_correct"] == 1)).astype(int)
    sf["corruption"] = ((sf["correct"] == 1) & (sf["next_correct"] == 0)).astype(int)
    sf = sf.reset_index(drop=True)

    runs, mismatches = {}, 0
    gkf = GroupKFold(n_splits=max(2, min(5, sf["run_id"].nunique())))
    for tr_idx, te_idx in gkf.split(sf, groups=sf["run_id"]):
        probe, repair, corruption = ta._fit_fold_models(sf.iloc[tr_idx])
        for rid, g in sf.iloc[te_idx].groupby("run_id"):
            if rid not in hd.index or rid not in ns.index:
                continue
            g = g.sort_values("step")
            q = ta.predict_probabilities(probe, g)[:, 1]
            a = ta.predict_probabilities(repair, g)[:, 1]
            b = ta.predict_probabilities(corruption, g)[:, 1]
            mu = (1.0 - q) * a - q * b - STEP_COST
            steps = g.step.astype(int).to_numpy()
            corr = g.correct.fillna(0).astype(float).to_numpy()
            i0 = stop_idx(steps, mu, 0.0)
            if int(steps[i0]) != int(hd.loc[rid, "stop_step"]):
                mismatches += 1
                continue
            runs[rid] = {"steps": steps, "corr": corr, "mu": mu,
                         "task": str(g.task_id.iloc[0]),
                         "u_hd": float(hd.loc[rid, "stop_utility"]),
                         "u_ns": float(ns.loc[rid, "stop_utility"])}
    return runs, mismatches


def best_delta(runs: dict, rids) -> float:
    best_d, best_u = 0.0, -1e18
    for d in DELTAS:
        tot = 0.0
        for rid in rids:
            r = runs[rid]
            i = stop_idx(r["steps"], r["mu"], d)
            tot += util(r["corr"][i], r["steps"][i])
        if tot > best_u or (tot == best_u and abs(d) < abs(best_d)):
            best_u, best_d = tot, float(d)
    return best_d


def wlt_du(runs: dict, rids, delta: float):
    w = t = l = 0
    du = 0.0
    for rid in rids:
        r = runs[rid]
        i = stop_idx(r["steps"], r["mu"], delta)
        u = util(r["corr"][i], r["steps"][i])
        du += u - r["u_hd"]
        if u > r["u_ns"]:
            w += 1
        elif u == r["u_ns"]:
            t += 1
        else:
            l += 1
    return w, t, l, du


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix-root", default=str(HERE / "outputs" / "experiment_matrix"))
    ap.add_argument("--out-csv", default=str(HERE / "outputs" / "experiment_matrix" / "_aggregate" / "calibrated_cell_thresholds.csv"))
    args = ap.parse_args()
    root = Path(args.matrix_root)

    rows = []
    cells = sorted(p.parent.name for p in root.glob("*/detector_comparison_by_run.csv"))
    for k, cell in enumerate(cells):
        runs, mism = recover_series(root / cell)
        rids = list(runs.keys())
        if not rids:
            continue
        base_w = sum(1 for r in runs.values() if r["u_hd"] > r["u_ns"])
        base_t = sum(1 for r in runs.values() if r["u_hd"] == r["u_ns"])
        base_l = len(rids) - base_w - base_t

        # honest OOF estimate: fold deltas fit on train tasks, applied to test tasks
        tasks = np.array([runs[r]["task"] for r in rids])
        gkf = GroupKFold(n_splits=max(2, min(5, len(set(tasks)))))
        fold_deltas, oof_du = [], 0.0
        oof_w = oof_t = oof_l = 0
        for tr_idx, te_idx in gkf.split(rids, groups=tasks):
            d = best_delta(runs, [rids[i] for i in tr_idx])
            fold_deltas.append(d)
            w, t, l, du = wlt_du(runs, [rids[i] for i in te_idx], d)
            oof_w += w; oof_t += t; oof_l += l; oof_du += du

        # deployment value: full-cell fit (in-sample companions labeled as such)
        d_full = best_delta(runs, rids)
        cw, ct, cl, cdu = wlt_du(runs, rids, d_full)

        model, dataset = cell.rsplit("__", 1)
        rows.append({
            "cell": cell, "model": model, "dataset": dataset,
            "n_runs": len(rids), "stop_mismatches": mism,
            "delta_full_DEPLOY": d_full,
            "fold_delta_min": min(fold_deltas), "fold_delta_max": max(fold_deltas),
            "oof_dU": round(oof_du, 2),
            "oof_win": oof_w, "oof_tie": oof_t, "oof_loss": oof_l,
            "base_win": base_w, "base_tie": base_t, "base_loss": base_l,
            "base_loss_pct": round(100 * base_l / len(rids), 2),
            "oof_loss_pct": round(100 * oof_l / len(rids), 2),
            "cal_full_dU_INSAMPLE": round(cdu, 2),
            "cal_full_loss_INSAMPLE": cl,
        })
        print(f"[{k+1}/{len(cells)}] {cell} mism={mism} delta={d_full:+.3f} oof_dU={oof_du:+.1f}", flush=True)

    df = pd.DataFrame(rows)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {out} ({len(df)} cells)")
    print(f"total stop mismatches: {int(df.stop_mismatches.sum())}")
    print(f"TOTAL oof_dU: {df.oof_dU.sum():+.2f}  (honest estimate; compare P3d's +1,545.85)")
    print(f"baseline loss {int(df.base_loss.sum())} ({100*df.base_loss.sum()/df.n_runs.sum():.2f}%) -> "
          f"OOF calibrated loss {int(df.oof_loss.sum())} ({100*df.oof_loss.sum()/df.n_runs.sum():.2f}%)")
    print("Reminder: quote oof_dU, deploy delta_full_DEPLOY; never quote cal_full_* as gains.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
