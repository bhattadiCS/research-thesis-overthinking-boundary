"""P9 lambda-sensitivity sweep, exact analytic form (rigor_audit/04 P9).

The hazard_drift rule stops at the first step>=2 with
mu_lambda = (1-q)alpha - q*beta - lambda <= 0, i.e. base <= lambda where
base = (1-q)alpha - q*beta is lambda-INDEPENDENT (models are fit on
features/labels only). So recovering each run's validated out-of-fold base
series once (same fold replication as research/offline_policy_arms_p3.py,
which matched all 75,965 recorded stops exactly) gives the EXACT sweep:
stop(lambda) = first step>=2 with base <= lambda; utility(lambda) uses
lambda in the step cost for both hazard_drift and never_stop.
Validation anchor: lambda=0.05 must reproduce 68,095/2,135/5,735 exactly.
"""
import json, sys
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

REPO = Path("C:/Aditya_Data/Personal/ResearchThesis")
ROOT = REPO / "research/outputs/experiment_matrix"
CACHE = Path(__file__).parent / "p9fast_cache"
CACHE.mkdir(exist_ok=True)
LAMBDAS = [0.01, 0.02, 0.05, 0.10, 0.20]

sys.path.insert(0, str(REPO / "research"))
spec = importlib.util.spec_from_file_location("ta", REPO / "research/trace_analysis.py")
ta = importlib.util.module_from_spec(spec); sys.modules["ta"] = ta
spec.loader.exec_module(ta)


def cell_result(cell):
    cf = CACHE / f"{cell}.json"
    if cf.exists():
        return json.loads(cf.read_text())
    d = ROOT / cell
    dcbr = pd.read_csv(d / "detector_comparison_by_run.csv")
    hd = dcbr[dcbr.detector == "hazard_drift"].set_index("run_id")

    sf = pd.read_csv(d / "trace_steps.csv")
    sf, _ = ta._sanitize_step_frame(sf)
    sf = ta.add_temporal_features(sf)
    sf["repair"] = ((sf["correct"] == 0) & (sf["next_correct"] == 1)).astype(int)
    sf["corruption"] = ((sf["correct"] == 1) & (sf["next_correct"] == 0)).astype(int)
    sf = sf.reset_index(drop=True)

    out = {"cell": cell, "mismatch05": 0, "runs": 0,
           "wlt": {str(l): [0, 0, 0] for l in LAMBDAS}}
    gkf = GroupKFold(n_splits=max(2, min(5, sf["run_id"].nunique())))
    for tr_idx, te_idx in gkf.split(sf, groups=sf["run_id"]):
        probe, repair, corruption = ta._fit_fold_models(sf.iloc[tr_idx])
        for rid, g in sf.iloc[te_idx].groupby("run_id"):
            if rid not in hd.index:
                continue
            g = g.sort_values("step")
            q = ta.predict_probabilities(probe, g)[:, 1]
            a = ta.predict_probabilities(repair, g)[:, 1]
            b = ta.predict_probabilities(corruption, g)[:, 1]
            base = (1.0 - q) * a - q * b
            steps = g.step.astype(int).to_numpy()
            corr = g.correct.fillna(0).astype(float).to_numpy()
            out["runs"] += 1
            for lam in LAMBDAS:
                idx = len(steps) - 1
                for i in range(len(steps)):
                    if steps[i] >= 2 and base[i] <= lam:
                        idx = i
                        break
                u_hd = corr[idx] - lam * (steps[idx] - 1.0)
                u_ns = corr[-1] - lam * (steps[-1] - 1.0)
                w = out["wlt"][str(lam)]
                if u_hd > u_ns:
                    w[0] += 1
                elif u_hd == u_ns:
                    w[1] += 1
                else:
                    w[2] += 1
                if lam == 0.05 and steps[idx] != int(hd.loc[rid, "stop_step"]):
                    out["mismatch05"] += 1
    cf.write_text(json.dumps(out))
    return out


cells = sorted(p.parent.name for p in ROOT.glob("*/detector_comparison_by_run.csv"))
tot = {str(l): np.zeros(3, dtype=int) for l in LAMBDAS}
runs = mism = 0
for i, cell in enumerate(cells):
    r = cell_result(cell)
    runs += r["runs"]; mism += r["mismatch05"]
    for l in LAMBDAS:
        tot[str(l)] += np.array(r["wlt"][str(l)])
    print(f"[{i+1}/{len(cells)}] {cell} runs={r['runs']} mism05={r['mismatch05']}", flush=True)

print(f"\n=== corpus {runs} runs; lambda=0.05 stop mismatches vs recorded: {mism}")
print(f"{'lambda':>7} {'win':>7} {'tie':>6} {'loss':>6} {'win%':>7} {'loss%':>7}")
ok = True
for l in LAMBDAS:
    w, t, x = tot[str(l)]
    n = w + t + x
    print(f"{l:>7} {w:>7} {t:>6} {x:>6} {100*w/n:>7.2f} {100*x/n:>7.2f}")
    if 0.02 <= l <= 0.10 and abs(100 * w / n - 89.6) > 5:
        ok = False
print("VALIDATION (lam=0.05 == 68095/2135/5735):", list(tot["0.05"]) == [68095, 2135, 5735])
print("SUCCESS CRITERION (win within +-5pp of 89.6% for lam in [0.02,0.10]):", ok)
