"""Algorithm-v2 Tier-1 experiments N1 (meta-calibration) + N4 (two-parameter rule).

Pre-registrations: ThesisDocs/rigor_audit/07_algorithm_v2_protocols.md.

N1 — Can the per-cell threshold be PREDICTED instead of looked up?
  * LOCO: ridge regression from observable cell statistics (step-1/2 accuracy,
    churn rate, mean entropy, mean trace length, dataset one-hots) to the
    fitted deployment delta, leave-one-cell-out across 52 cells; the held-out
    cell is scored on ALL its runs at the predicted delta (fully out-of-sample).
    Success: >= 60% of the P3d lookup gain (>= +927.5 of +1,545.85).
  * LOMO: leave-one-MODEL-out (all 4 cells of a model held out) — the
    deployment-realistic variant. Reported, no separate bar.
  * Learning curve: delta fit on n in {50,100,200,400} runs, scored on the
    cell's remaining runs — how much labeled warm-up a new cell needs.

N4 — stop at first step>=2 with mu <= delta + gamma*chg2 (chg2 = answer
  changed at step 2, observable at decision time), vs the one-parameter rule,
  PAIRED on identical GroupKFold(5)-by-task folds.
  Success: two-parameter OOF net dU exceeds one-parameter by > +150.

Method notes: per-run out-of-fold mu series are recovered exactly as
production scored them (same chain as research/calibrate_cell_thresholds.py,
which matched 75,965/75,965 recorded stops); runs whose recomputed stop
mismatches the recorded stop are excluded and counted (expect 0). All rule
evaluations use a per-run utility profile over a tau grid, so every
(delta, gamma) search is exact profile arithmetic, not re-simulation.

Usage:
    python research/algorithm_v2_experiments.py [--matrix-root DIR]
        [--cache-dir DIR] [--cells a,b]      # --cells for smoke tests
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
STEP_COST = 0.05
TAUS = np.round(np.arange(-0.21, 0.2101, 0.005), 3)          # profile grid
DELTAS = np.round(np.arange(-0.15, 0.1501, 0.005), 3)        # delta search
GAMMAS = np.round(np.arange(-0.06, 0.0601, 0.01), 3)         # gamma search
LC_SIZES = (50, 100, 200, 400)

spec = importlib.util.spec_from_file_location("ta", HERE / "trace_analysis.py")
ta = importlib.util.module_from_spec(spec)
sys.modules["ta"] = ta
spec.loader.exec_module(ta)


def tau_index(v: float) -> int:
    return int(round((v - TAUS[0]) / 0.005))


def recover_cell(cell_dir: Path):
    """Per run: utility profile over TAUS, baseline u_hd/u_ns, chg2, task.
    Returns (U, uns, uhd, chg2, tasks, run_ids, mismatches, feats)."""
    dcbr = pd.read_csv(cell_dir / "detector_comparison_by_run.csv")
    hd = dcbr[dcbr.detector == "hazard_drift"].set_index("run_id")
    ns = dcbr[dcbr.detector == "never_stop"].set_index("run_id")
    sf = pd.read_csv(cell_dir / "trace_steps.csv")
    sf, _ = ta._sanitize_step_frame(sf)
    sf = ta.add_temporal_features(sf)
    sf["repair"] = ((sf["correct"] == 0) & (sf["next_correct"] == 1)).astype(int)
    sf["corruption"] = ((sf["correct"] == 1) & (sf["next_correct"] == 0)).astype(int)
    sf = sf.reset_index(drop=True)

    # observable cell features for N1 (computable without any policy labels)
    s1 = sf[sf.step == 1]
    s2 = sf[sf.step == 2]
    feats = {
        "step1_acc": float(s1.correct.fillna(0).mean()) if len(s1) else 0.0,
        "step2_acc": float(s2.correct.fillna(0).mean()) if len(s2) else 0.0,
        "churn_rate": float(sf[sf.step >= 2].answer_changed.fillna(0).mean()),
        "mean_entropy": float(sf.entropy_mean.mean()),
        "mean_len": float(sf.groupby("run_id").step.max().mean()),
    }

    rows_U, rows_uns, rows_uhd, rows_chg2, rows_task, rids = [], [], [], [], [], []
    mism = 0
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
            # validation vs recorded stop at tau=0
            i0 = next((i for i in range(len(steps)) if steps[i] >= 2 and mu[i] <= 0.0), len(steps) - 1)
            if int(steps[i0]) != int(hd.loc[rid, "stop_step"]):
                mism += 1
                continue
            # utility profile: stop at first step>=2 with mu <= tau, else last
            elig = steps >= 2
            prof = np.empty(len(TAUS))
            for j, tau in enumerate(TAUS):
                k = np.argmax(elig & (mu <= tau)) if np.any(elig & (mu <= tau)) else len(steps) - 1
                prof[j] = corr[k] - STEP_COST * (steps[k] - 1.0)
            rows_U.append(prof)
            rows_uns.append(float(ns.loc[rid, "stop_utility"]))
            rows_uhd.append(float(hd.loc[rid, "stop_utility"]))
            g2 = g[g.step == 2]
            rows_chg2.append(int(g2.answer_changed.fillna(0).iloc[0]) if len(g2) else 0)
            rows_task.append(str(g.task_id.iloc[0]))
            rids.append(rid)
    return (np.array(rows_U), np.array(rows_uns), np.array(rows_uhd),
            np.array(rows_chg2), np.array(rows_task), rids, mism, feats)


def cell_pass(cell: str, root: Path, cache: Path):
    cf = cache / f"{cell}.json"
    if cf.exists():
        return json.loads(cf.read_text())
    U, uns, uhd, chg2, tasks, rids, mism, feats = recover_cell(root / cell)
    n = len(rids)
    if n == 0:
        return None
    dU = U - uhd[:, None]                                   # per-run gain profile
    didx = np.array([tau_index(d) for d in DELTAS])

    def best_delta_rows(rows) -> int:
        return int(didx[np.argmax(dU[np.ix_(rows, didx)].sum(axis=0))])

    # learning curve (N1 secondary)
    rng = np.random.RandomState(abs(hash(cell)) % (2**31))
    lc = {}
    full_j = best_delta_rows(np.arange(n))
    for sz in LC_SIZES:
        if sz >= n:
            continue
        pick = rng.choice(n, size=sz, replace=False)
        rest = np.setdiff1d(np.arange(n), pick)
        j_n = best_delta_rows(pick)
        lc[str(sz)] = {
            "dU_rest_at_fit": float(dU[rest, j_n].sum()),
            "dU_rest_at_full": float(dU[rest, full_j].sum()),
        }

    # N4: paired 1-param vs 2-param on identical folds (GroupKFold by task)
    gkf = GroupKFold(n_splits=max(2, min(5, len(set(tasks)))))
    du1 = du2 = 0.0
    g0, g1 = np.where(chg2 == 0)[0], np.where(chg2 == 1)[0]
    for tr, te in gkf.split(np.arange(n), groups=tasks):
        tr0, tr1 = np.intersect1d(tr, g0), np.intersect1d(tr, g1)
        te0, te1 = np.intersect1d(te, g0), np.intersect1d(te, g1)
        P0 = dU[tr0].sum(axis=0) if len(tr0) else np.zeros(len(TAUS))
        P1 = dU[tr1].sum(axis=0) if len(tr1) else np.zeros(len(TAUS))
        # 1-param on train
        j1 = int(didx[np.argmax((P0 + P1)[didx])])
        du1 += (dU[te0, j1].sum() if len(te0) else 0.0) + (dU[te1, j1].sum() if len(te1) else 0.0)
        # 2-param on train
        best, jd_b, jg_b = -1e18, didx[0], 0.0
        for jd in didx:
            for gam in GAMMAS:
                jg = jd + int(round(gam / 0.005))
                if not (0 <= jg < len(TAUS)):
                    continue
                v = P0[jd] + P1[jg]
                if v > best:
                    best, jd_b, jg_b = v, jd, jg
        du2 += (dU[te0, jd_b].sum() if len(te0) else 0.0) + (dU[te1, jg_b].sum() if len(te1) else 0.0)

    out = {"cell": cell, "n": n, "mismatch": mism, "features": feats,
           "delta_full": float(TAUS[full_j]),
           "dU_profile": dU.sum(axis=0).tolist(),
           "loss_profile": (U < uns[:, None]).sum(axis=0).tolist(),
           "lc": lc, "n4": {"du_1param": float(du1), "du_2param": float(du2)}}
    cf.write_text(json.dumps(out))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix-root", default=str(HERE / "outputs" / "experiment_matrix"))
    ap.add_argument("--cache-dir", default=".algov2_cache")
    ap.add_argument("--cells", default=None, help="comma-separated subset (smoke test)")
    args = ap.parse_args()
    root = Path(args.matrix_root)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    cells = sorted(p.parent.name for p in root.glob("*/detector_comparison_by_run.csv"))
    if args.cells:
        keep = set(args.cells.split(","))
        cells = [c for c in cells if c in keep]

    results = []
    for i, cell in enumerate(cells):
        r = cell_pass(cell, root, cache)
        if r is None:
            continue
        results.append(r)
        print(f"[{i+1}/{len(cells)}] {cell} n={r['n']} mism={r['mismatch']} "
              f"delta={r['delta_full']:+.3f} n4(1p/2p)={r['n4']['du_1param']:+.1f}/{r['n4']['du_2param']:+.1f}",
              flush=True)

    smoke = len(results) < 40
    total_mism = sum(r["mismatch"] for r in results)
    print(f"\n=== {len(results)} cells, {sum(r['n'] for r in results)} runs, {total_mism} stop mismatches")

    # ---- N4 verdict (paired) ----
    d1 = sum(r["n4"]["du_1param"] for r in results)
    d2 = sum(r["n4"]["du_2param"] for r in results)
    print("\n--- N4 two-parameter rule (paired, identical folds) ---")
    print(f"1-param OOF dU: {d1:+.2f}   2-param OOF dU: {d2:+.2f}   paired gain: {d2-d1:+.2f}")
    print("CRITERION (2-param beats 1-param by > +150):", (d2 - d1) > 150)

    # ---- N1 LOCO / LOMO ----
    if not smoke:
        feat_names = ["step1_acc", "step2_acc", "churn_rate", "mean_entropy", "mean_len"]
        ds_names = sorted({r["cell"].rsplit("__", 1)[1] for r in results})
        X = np.array([[r["features"][f] for f in feat_names] +
                      [1.0 if r["cell"].rsplit("__", 1)[1] == d else 0.0 for d in ds_names]
                      for r in results])
        y = np.array([r["delta_full"] for r in results])
        prof = np.array([r["dU_profile"] for r in results])
        models = [r["cell"].rsplit("__", 1)[0] for r in results]

        def held_out_dU(groups) -> float:
            tot = 0.0
            for g in sorted(set(groups)):
                te = [i for i, gg in enumerate(groups) if gg == g]
                tr = [i for i in range(len(results)) if i not in te]
                sc = StandardScaler().fit(X[tr])
                m = Ridge(alpha=1.0).fit(sc.transform(X[tr]), y[tr])
                pred = np.clip(m.predict(sc.transform(X[te])), DELTAS[0], DELTAS[-1])
                for k, i in enumerate(te):
                    tot += prof[i][tau_index(round(round(pred[k] / 0.005) * 0.005, 3))]
            return tot

        lookup = float(sum(r["dU_profile"][tau_index(r["delta_full"])] for r in results))
        loco = held_out_dU(list(range(len(results))))           # each cell its own group
        lomo = held_out_dU(models)
        print("\n--- N1 meta-calibration ---")
        print(f"P3d full-fit lookup ceiling (in-sample deploy deltas): {lookup:+.2f}")
        print(f"LOCO predicted-delta dU: {loco:+.2f}  ({100*loco/1545.85:.1f}% of the P3d OOF gain)")
        print(f"LOMO predicted-delta dU: {lomo:+.2f}  ({100*lomo/1545.85:.1f}%)")
        print("CRITERION (LOCO >= 60% of +1,545.85 => >= +927.5):", loco >= 927.5)

        rec = {str(sz): [] for sz in LC_SIZES}
        for r in results:
            for sz, v in r["lc"].items():
                if v["dU_rest_at_full"] > 0:
                    rec[sz].append(v["dU_rest_at_fit"] / v["dU_rest_at_full"])
        print("\n--- N1 learning curve (median fraction of full-delta gain recovered) ---")
        for sz in LC_SIZES:
            vals = rec[str(sz)]
            if vals:
                print(f"  n={sz:>3}: median {np.median(vals):.2f}  (cells: {len(vals)})")
    else:
        print("\n[smoke mode: N1 LOCO/LOMO skipped — needs the full 52-cell pass]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
