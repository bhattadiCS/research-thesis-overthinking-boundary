"""Algorithm-v2 Tier-1 experiments N2 (probe suite) + N3 (hierarchical hazards).

Pre-registrations: ThesisDocs/rigor_audit/07_algorithm_v2_protocols.md.

N2 — Probe upgrade suite (belief-state quality on existing features)
  * N2a: HistGradientBoostingClassifier instead of LogisticRegression.
  * N2b: Isotonic-calibrated probe outputs (nested cv in fold).
  * N2c: Trace-history window (lag-1 and lag-2 features included).

N3 — Hazard estimation: partial pooling/shrinkage across cells
  * Step-specific empirical hazard rates shrunk toward global step-specific averages.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
STEP_COST = 0.05
TAUS = np.round(np.arange(-0.21, 0.2101, 0.005), 3)          # profile grid
DELTAS = np.round(np.arange(-0.15, 0.1501, 0.005), 3)        # delta search
GAMMAS = np.round(np.arange(-0.06, 0.0601, 0.01), 3)         # gamma search
T_MIN = 2

spec = importlib.util.spec_from_file_location("ta", HERE / "trace_analysis.py")
ta = importlib.util.module_from_spec(spec)
sys.modules["ta"] = ta
spec.loader.exec_module(ta)

FEATURE_COLUMNS = ta.FEATURE_COLUMNS


def tau_index(v: float) -> int:
    return int(round((v - TAUS[0]) / 0.005))


class ConstantProbabilityModel:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = np.full(len(X), self.probability, dtype=float)
        return np.column_stack([1.0 - probabilities, probabilities])


class EBTransitionModel:
    def __init__(self, step_probs: dict[int, float]):
        self.step_probs = step_probs

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        steps = X["step"].astype(int).to_numpy()
        probs = np.array([self.step_probs.get(s, self.step_probs.get(10, 0.0)) for s in steps])
        return np.column_stack([1.0 - probs, probs])


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = [c for c in FEATURE_COLUMNS if c != "step"]
    
    # Shift per run
    lag1 = df.groupby("run_id")[cols].shift(1)
    lag2 = df.groupby("run_id")[cols].shift(2)
    
    # Fill shifting NaNs with the current step features (backfill logic for t < 3)
    for c in cols:
        lag1[c] = lag1[c].fillna(df[c])
        lag2[c] = lag2[c].fillna(lag1[c])
        
    lag1.columns = [f"{c}_lag1" for c in cols]
    lag2.columns = [f"{c}_lag2" for c in cols]
    
    # Reset indices to ensure concatenation aligns correctly
    return pd.concat([df.reset_index(drop=True), lag1.reset_index(drop=True), lag2.reset_index(drop=True)], axis=1)


def fit_calibrated_probe(train_frame: pd.DataFrame, features: list[str]) -> Any:
    if train_frame.empty:
        return ConstantProbabilityModel(0.0)
    target = train_frame["correct"].astype(int)
    if target.nunique() < 2:
        return ConstantProbabilityModel(float(target.mean()))
        
    base = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    min_class = target.value_counts().min()
    
    # Adjust calibration folds based on sample size to prevent crash
    if min_class >= 3:
        cv = 3
    elif min_class >= 2:
        cv = 2
    else:
        # Fallback to standard scaled logistic model
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", base)
        ]).fit(train_frame[features], target)
        
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", CalibratedClassifierCV(estimator=base, method="isotonic", cv=cv))
    ]).fit(train_frame[features], target)


def fit_eb_hazards(train_frame: pd.DataFrame, other_counts: dict[int, dict], target_column: str, k: float = 10.0) -> EBTransitionModel:
    step_probs = {}
    transitions = train_frame[train_frame["has_next"] == 1]
    
    filter_val = 0 if target_column == "repair" else 1
    haz = transitions[transitions["correct"] == filter_val]
    
    cell_counts = {}
    for step, group in haz.groupby("step"):
        step = int(step)
        cell_counts[step] = {"n": len(group), "sum": int(group[target_column].sum())}
        
    for step in range(2, 11):
        oth = other_counts.get(step, {"repair_n": 0, "repair_sum": 0, "corruption_n": 0, "corruption_sum": 0})
        if target_column == "repair":
            oth_n = oth["repair_n"]
            oth_sum = oth["repair_sum"]
        else:
            oth_n = oth["corruption_n"]
            oth_sum = oth["corruption_sum"]
            
        cell_n = cell_counts.get(step, {}).get("n", 0)
        cell_sum = cell_counts.get(step, {}).get("sum", 0)
        
        tot_n = oth_n + cell_n
        tot_sum = oth_sum + cell_sum
        global_rate = tot_sum / tot_n if tot_n > 0 else 0.0
        
        shrunk_rate = (cell_sum + k * global_rate) / (cell_n + k) if (cell_n + k) > 0 else 0.0
        step_probs[step] = shrunk_rate
        
    return EBTransitionModel(step_probs)


def fit_binary_model(train_frame: pd.DataFrame, target_column: str, features: list[str], model_type: str = "logistic") -> Any:
    if train_frame.empty:
        return ConstantProbabilityModel(0.0)
    target = train_frame[target_column].astype(int)
    if target.nunique() < 2:
        return ConstantProbabilityModel(float(target.mean()))
        
    if model_type == "gbt":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", HistGradientBoostingClassifier(random_state=42, max_depth=4, max_iter=100))
        ]).fit(train_frame[features], target)
    else:
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"))
        ]).fit(train_frame[features], target)


def recover_cell_for_arm(cell_dir: Path, arm: str, global_counts: dict | None = None):
    """Computes OOF drift utility profiles for a cell under the specified arm."""
    dcbr = pd.read_csv(cell_dir / "detector_comparison_by_run.csv")
    hd = dcbr[dcbr.detector == "hazard_drift"].set_index("run_id")
    ns = dcbr[dcbr.detector == "never_stop"].set_index("run_id")
    sf = pd.read_csv(cell_dir / "trace_steps.csv")
    sf, _ = ta._sanitize_step_frame(sf)
    sf = ta.add_temporal_features(sf)
    sf["repair"] = ((sf["correct"] == 0) & (sf["next_correct"] == 1)).astype(int)
    sf["corruption"] = ((sf["correct"] == 1) & (sf["next_correct"] == 0)).astype(int)
    sf = sf.reset_index(drop=True)
    
    # Feature set selection
    features = FEATURE_COLUMNS
    if arm == "n2c":
        sf = add_lag_features(sf)
        lag_cols = [c for c in FEATURE_COLUMNS if c != "step"]
        features = FEATURE_COLUMNS + [f"{c}_lag1" for c in lag_cols] + [f"{c}_lag2" for c in lag_cols]
        
    # Count totals for other cells in N3 EB hazards
    other_counts = {}
    if arm == "n3" and global_counts is not None:
        cell_total_counts = {}
        transitions = sf[sf["has_next"] == 1]
        for step, group in transitions.groupby("step"):
            step = int(step)
            inc = group[group["correct"] == 0]
            corr = group[group["correct"] == 1]
            cell_total_counts[step] = {
                "repair_n": len(inc),
                "repair_sum": int(inc["repair"].sum()),
                "corruption_n": len(corr),
                "corruption_sum": int(corr["corruption"].sum())
            }
        for step in range(2, 11):
            g = global_counts.get(step, {"repair_n": 0, "repair_sum": 0, "corruption_n": 0, "corruption_sum": 0})
            c_tot = cell_total_counts.get(step, {"repair_n": 0, "repair_sum": 0, "corruption_n": 0, "corruption_sum": 0})
            other_counts[step] = {
                "repair_n": max(0, g["repair_n"] - c_tot["repair_n"]),
                "repair_sum": max(0, g["repair_sum"] - c_tot["repair_sum"]),
                "corruption_n": max(0, g["corruption_n"] - c_tot["corruption_n"]),
                "corruption_sum": max(0, g["corruption_sum"] - c_tot["corruption_sum"]),
            }

    rows_U, rows_uns, rows_uhd, rows_chg2, rows_task, rids = [], [], [], [], [], []
    gkf = GroupKFold(n_splits=max(2, min(5, sf["run_id"].nunique())))
    
    for tr_idx, te_idx in gkf.split(sf, groups=sf["run_id"]):
        tr_df = sf.iloc[tr_idx]
        
        # Prepare transitions for hazard modeling (exactly matching ta._fit_fold_models)
        transitions = tr_df[tr_df["has_next"] == 1].copy()
        haz = transitions[transitions["step"] >= T_MIN]
        
        # Baseline model for mismatch check
        base_probe = fit_binary_model(tr_df, "correct", FEATURE_COLUMNS, "logistic")
        base_repair = fit_binary_model(haz[haz["correct"] == 0], "repair", FEATURE_COLUMNS, "logistic")
        base_corruption = fit_binary_model(haz[haz["correct"] == 1], "corruption", FEATURE_COLUMNS, "logistic")
        
        # Fit models for the active arm
        if arm == "n2a":  # GBT Probe
            probe = fit_binary_model(tr_df, "correct", features, "gbt")
            repair = fit_binary_model(haz[haz["correct"] == 0], "repair", features, "logistic")
            corruption = fit_binary_model(haz[haz["correct"] == 1], "corruption", features, "logistic")
        elif arm == "n2b":  # Isotonic Calibrated Probe
            probe = fit_calibrated_probe(tr_df, features)
            repair = fit_binary_model(haz[haz["correct"] == 0], "repair", features, "logistic")
            corruption = fit_binary_model(haz[haz["correct"] == 1], "corruption", features, "logistic")
        elif arm == "n3":  # EB Shrunk Hazards
            probe = fit_binary_model(tr_df, "correct", features, "logistic")
            repair = fit_eb_hazards(haz, other_counts, "repair", k=10.0)
            corruption = fit_eb_hazards(haz, other_counts, "corruption", k=10.0)
        else:  # Baseline (Logistic) and N2c (Lagged features on logistic)
            probe = fit_binary_model(tr_df, "correct", features, "logistic")
            repair = fit_binary_model(haz[haz["correct"] == 0], "repair", features, "logistic")
            corruption = fit_binary_model(haz[haz["correct"] == 1], "corruption", features, "logistic")
            
        for rid, g in sf.iloc[te_idx].groupby("run_id"):
            if rid not in hd.index or rid not in ns.index:
                continue
            g = g.sort_values("step")
            steps = g.step.astype(int).to_numpy()
            corr = g.correct.fillna(0).astype(float).to_numpy()
            
            # Baseline mismatch check (matching algorithm_v2_experiments.py recovery)
            base_q = base_probe.predict_proba(g[FEATURE_COLUMNS])[:, 1]
            base_a = base_repair.predict_proba(g[FEATURE_COLUMNS])[:, 1]
            base_b = base_corruption.predict_proba(g[FEATURE_COLUMNS])[:, 1]
            base_mu = (1.0 - base_q) * base_a - base_q * base_b - STEP_COST
            
            i0 = next((i for i in range(len(steps)) if steps[i] >= 2 and base_mu[i] <= 0.0), len(steps) - 1)
            if int(steps[i0]) != int(hd.loc[rid, "stop_step"]):
                continue
            
            # Predict probability scores for the arm
            if arm == "n2c":
                q = probe.predict_proba(g[features])[:, 1]
                a = repair.predict_proba(g[features])[:, 1]
                b = corruption.predict_proba(g[features])[:, 1]
            else:
                q = probe.predict_proba(g[FEATURE_COLUMNS])[:, 1]
                a = repair.predict_proba(g[FEATURE_COLUMNS])[:, 1]
                b = corruption.predict_proba(g[FEATURE_COLUMNS])[:, 1]
                
            mu = (1.0 - q) * a - q * b - STEP_COST
            
            # Utility profile evaluation
            elig = steps >= 2
            prof = np.empty(len(TAUS))
            for j, tau in enumerate(TAUS):
                k_idx = np.argmax(elig & (mu <= tau)) if np.any(elig & (mu <= tau)) else len(steps) - 1
                prof[j] = corr[k_idx] - STEP_COST * (steps[k_idx] - 1.0)
                
            rows_U.append(prof)
            rows_uns.append(float(ns.loc[rid, "stop_utility"]))
            rows_uhd.append(float(hd.loc[rid, "stop_utility"]))
            
            g2 = g[g.step == 2]
            rows_chg2.append(int(g2.answer_changed.fillna(0).iloc[0]) if len(g2) else 0)
            rows_task.append(str(g.task_id.iloc[0]))
            rids.append(rid)
            
    return np.array(rows_U), np.array(rows_uns), np.array(rows_uhd), np.array(rows_chg2), np.array(rows_task)


def compute_global_transition_counts(root: Path, cells: list[str]) -> dict:
    print("Computing global transition counts for N3 EB hazards...", flush=True)
    counts = {}
    for cell in cells:
        csv_path = root / cell / "trace_steps.csv"
        if not csv_path.exists():
            continue
        sf = pd.read_csv(csv_path)
        sf, _ = ta._sanitize_step_frame(sf)
        sf = ta.add_temporal_features(sf)
        sf["repair"] = ((sf["correct"] == 0) & (sf["next_correct"] == 1)).astype(int)
        sf["corruption"] = ((sf["correct"] == 1) & (sf["next_correct"] == 0)).astype(int)
        
        transitions = sf[sf["has_next"] == 1]
        for step, group in transitions.groupby("step"):
            step = int(step)
            if step < 2:
                continue
            if step not in counts:
                counts[step] = {"repair_n": 0, "repair_sum": 0, "corruption_n": 0, "corruption_sum": 0}
            
            # Repair stats
            inc = group[group["correct"] == 0]
            counts[step]["repair_n"] += len(inc)
            counts[step]["repair_sum"] += int(inc["repair"].sum())
            
            # Corruption stats
            corr = group[group["correct"] == 1]
            counts[step]["corruption_n"] += len(corr)
            counts[step]["corruption_sum"] += int(corr["corruption"].sum())
            
    return counts


def evaluate_arm(cell: str, root: Path, arm: str, global_counts: dict | None = None) -> dict:
    U, uns, uhd, chg2, tasks = recover_cell_for_arm(root / cell, arm, global_counts)
    n = len(U)
    if n == 0:
        return {}
        
    dU = U - uhd[:, None]
    didx = np.array([tau_index(d) for d in DELTAS])
    
    # 1-param / 2-param paired evaluation on identical folds by task_id
    gkf = GroupKFold(n_splits=max(2, min(5, len(set(tasks)))))
    du1 = du2 = 0.0
    g0, g1 = np.where(chg2 == 0)[0], np.where(chg2 == 1)[0]
    
    for tr, te in gkf.split(np.arange(n), groups=tasks):
        tr0, tr1 = np.intersect1d(tr, g0), np.intersect1d(tr, g1)
        te0, te1 = np.intersect1d(te, g0), np.intersect1d(te, g1)
        
        P0 = dU[tr0].sum(axis=0) if len(tr0) else np.zeros(len(TAUS))
        P1 = dU[tr1].sum(axis=0) if len(tr1) else np.zeros(len(TAUS))
        
        # 1-parameter rule calibration
        j1 = int(didx[np.argmax((P0 + P1)[didx])])
        du1 += (dU[te0, j1].sum() if len(te0) else 0.0) + (dU[te1, j1].sum() if len(te1) else 0.0)
        
        # 2-parameter rule calibration
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
        
    return {
        "n": n,
        "du_1param": float(du1),
        "du_2param": float(du2)
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix-root", default=str(HERE / "outputs" / "experiment_matrix"))
    ap.add_argument("--cache-dir", default=str(HERE / "outputs" / "experiments_v2" / "algov2_cache_n2_n3"))
    ap.add_argument("--cells", default=None)
    args = ap.parse_args()
    
    root = Path(args.matrix_root)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    
    cells = sorted(p.parent.name for p in root.glob("*/detector_comparison_by_run.csv"))
    if args.cells:
        keep = set(args.cells.split(","))
        cells = [c for c in cells if c in keep]
        
    # Global transition counts for N3
    global_counts = compute_global_transition_counts(root, cells)
    
    arms = ["baseline", "n2a", "n2b", "n2c", "n3"]
    results = {arm: [] for arm in arms}
    
    for i, cell in enumerate(cells):
        cf = cache / f"{cell}.json"
        if cf.exists():
            cell_res = json.loads(cf.read_text())
            for arm in arms:
                if arm in cell_res:
                    results[arm].append(cell_res[arm])
            print(f"[{i+1}/{len(cells)}] {cell} (cached)")
            continue
            
        cell_res = {"cell": cell}
        valid = True
        for arm in arms:
            res = evaluate_arm(cell, root, arm, global_counts)
            if not res:
                valid = False
                break
            cell_res[arm] = res
            results[arm].append(res)
            
        if not valid:
            continue
            
        cf.write_text(json.dumps(cell_res))
        b1, b2 = cell_res["baseline"]["du_1param"], cell_res["baseline"]["du_2param"]
        print(f"[{i+1}/{len(cells)}] {cell} baseline(1p/2p)={b1:+.1f}/{b2:+.1f} "
              f"n2a={cell_res['n2a']['du_1param']:+.1f} "
              f"n2b={cell_res['n2b']['du_1param']:+.1f} "
              f"n2c={cell_res['n2c']['du_1param']:+.1f} "
              f"n3={cell_res['n3']['du_1param']:+.1f}", flush=True)
              
    print("\n=== Tier 1 (N2/N3) Aggregated Out-of-Fold dU Gains vs Baseline (Hazard Drift at delta=0) ===")
    for arm in arms:
        tot_1p = sum(r["du_1param"] for r in results[arm])
        tot_2p = sum(r["du_2param"] for r in results[arm])
        print(f"Arm {arm:<10} | 1-param calibrated OOF dU: {tot_1p:+.2f} | 2-param calibrated OOF dU: {tot_2p:+.2f}")
        
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
