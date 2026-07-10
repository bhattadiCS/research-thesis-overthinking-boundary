"""Pre-registered offline policy arms P3b / P3c / P3d (rigor_audit/04 P3).

Faithful replication: per cell, rebuild the production scoring exactly as
trace_analysis.main() + out_of_sample_eval do (same read order, sanitizer,
temporal features, GroupKFold(5) by run_id, per-fold _fit_fold_models), then
recover each run's OUT-OF-FOLD per-step mu_hat / q_hat series by replicating
hazard_stop_for_group's arithmetic. VALIDATION GATE: the recomputed first
crossing must equal the recorded hazard_drift stop_step in
detector_comparison_by_run.csv for every run; cells with mismatches are
excluded and reported.

Arms (single IV each, vs the frozen baseline):
  P3b churn-gated hysteresis: stop at first step>=2 with mu<=0 AND
      (answer_changed==0 at that row OR the previous row also has mu<=0);
      else last step.
  P3c best-so-far selection: stop step unchanged; returned answer = argmax
      q_hat over eligible steps [2, stop] (sub-arm c2: [1, stop] — crosses the
      T_MIN line, reported as answer-selection only). u = correct(sel) - 0.05*(stop-1).
  P3d per-cell threshold offset: stop rule mu<=delta, delta fit per cell on
      GroupKFold(5)-by-task_id TRAIN folds (grid -0.15..0.15 step 0.005,
      objective = total utility), evaluated out-of-fold.

Criteria (fixed in advance): P3b net dU>0 AND >=150 E-losses dissolved;
P3c >=180 C-losses dissolved AND <=90 win->loss; P3d OOF net dU > each of
P3a/P3b/P3c's net dU. Per-cell cache; restartable.
"""
import json, sys
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

REPO = Path("C:/Aditya_Data/Personal/ResearchThesis")
ROOT = REPO / "research/outputs/experiment_matrix"
CACHE = Path(__file__).parent / "p3_cache"
CACHE.mkdir(exist_ok=True)
STEP_COST = 0.05
DELTAS = np.round(np.arange(-0.15, 0.1501, 0.005), 3)

sys.path.insert(0, str(REPO / "research"))
spec = importlib.util.spec_from_file_location("ta", REPO / "research/trace_analysis.py")
ta = importlib.util.module_from_spec(spec); sys.modules["ta"] = ta
spec.loader.exec_module(ta)


def util(c, s):
    return float(c) - STEP_COST * (s - 1.0)


def stop_from_mu(steps, mu, delta=0.0):
    for i in range(len(steps)):
        if steps[i] >= 2 and mu[i] <= delta:
            return i
    return len(steps) - 1


def cell_result(cell):
    cf = CACHE / f"{cell}.json"
    if cf.exists():
        return json.loads(cf.read_text())
    d = ROOT / cell
    if not (d / "detector_comparison_by_run.csv").exists():
        return None
    dcbr = pd.read_csv(d / "detector_comparison_by_run.csv")
    hd = dcbr[dcbr.detector == "hazard_drift"].set_index("run_id")
    ns = dcbr[dcbr.detector == "never_stop"].set_index("run_id")

    sf = pd.read_csv(d / "trace_steps.csv")
    sf, _ = ta._sanitize_step_frame(sf)
    sf = ta.add_temporal_features(sf)
    sf["repair"] = ((sf["correct"] == 0) & (sf["next_correct"] == 1)).astype(int)
    sf["corruption"] = ((sf["correct"] == 1) & (sf["next_correct"] == 0)).astype(int)
    sf = sf.reset_index(drop=True)

    # Recover per-run OOF mu/q series with the production fold structure.
    runs = {}
    gkf = GroupKFold(n_splits=max(2, min(5, sf["run_id"].nunique())))
    for tr_idx, te_idx in gkf.split(sf, groups=sf["run_id"]):
        train, test = sf.iloc[tr_idx], sf.iloc[te_idx]
        probe, repair, corruption = ta._fit_fold_models(train)
        for rid, g in test.groupby("run_id"):
            g = g.sort_values("step")
            q = ta.predict_probabilities(probe, g)[:, 1]
            a = ta.predict_probabilities(repair, g)[:, 1]
            b = ta.predict_probabilities(corruption, g)[:, 1]
            mu = (1.0 - q) * a - q * b - STEP_COST
            runs[rid] = {
                "steps": g.step.astype(int).to_numpy(),
                "corr": g.correct.fillna(0).astype(float).to_numpy(),
                "chg": g.answer_changed.fillna(0).astype(float).to_numpy() >= 1,
                "mu": mu, "q": q,
                "task": str(g.task_id.iloc[0]),
            }

    out = {"cell": cell, "runs": 0, "stop_mismatch": 0,
           "p3b": {"dU": 0.0, "dissolved": 0, "e_dissolved": 0, "win2loss": 0, "moved": 0},
           "p3c1": {"dU": 0.0, "dissolved": 0, "c_dissolved": 0, "win2loss": 0, "changed": 0},
           "p3c2": {"dU": 0.0, "dissolved": 0, "win2loss": 0, "changed": 0},
           "p3d": {"dU": 0.0, "dissolved": 0, "win2loss": 0, "deltas": []}}

    valid = {}
    for rid, r in runs.items():
        if rid not in hd.index or rid not in ns.index:
            continue
        i_prod = stop_from_mu(r["steps"], r["mu"], 0.0)
        s_prod = int(r["steps"][i_prod])
        if s_prod != int(hd.loc[rid, "stop_step"]):
            out["stop_mismatch"] += 1
            continue
        r["i_prod"], r["s_prod"] = i_prod, s_prod
        r["u_hd"] = float(hd.loc[rid, "stop_utility"])
        r["u_ns"] = float(ns.loc[rid, "stop_utility"])
        r["verdict"] = "win" if r["u_hd"] > r["u_ns"] else ("tie" if r["u_hd"] == r["u_ns"] else "loss")
        elig = [(int(s), float(c)) for s, c in zip(r["steps"], r["corr"]) if s >= 2 and c == 1]
        r["first_ok"] = elig[0][0] if elig else None
        valid[rid] = r
        out["runs"] += 1

    if out["stop_mismatch"] > 0.001 * max(len(valid), 1):
        out["INVALID"] = f"{out['stop_mismatch']} stop mismatches"
        cf.write_text(json.dumps(out))
        return out

    def score(bucket, r, u_new, changed):
        bucket["dU"] += u_new - r["u_hd"]
        v = "win" if u_new > r["u_ns"] else ("tie" if u_new == r["u_ns"] else "loss")
        if changed:
            bucket["moved"] = bucket.get("moved", 0) + 1
        if r["verdict"] == "loss" and v != "loss":
            bucket["dissolved"] += 1
        if r["verdict"] == "win" and v == "loss":
            bucket["win2loss"] += 1
        return v

    for rid, r in valid.items():
        steps, mu, chg, corr = r["steps"], r["mu"], r["chg"], r["corr"]
        # P3b
        i_new = None
        for i in range(len(steps)):
            if steps[i] >= 2 and mu[i] <= 0 and ((not chg[i]) or (i >= 1 and mu[i - 1] <= 0)):
                i_new = i
                break
        if i_new is None:
            i_new = len(steps) - 1
        u_new = util(corr[i_new], steps[i_new])
        v = score(out["p3b"], r, u_new, i_new != r["i_prod"])
        if r["verdict"] == "loss" and v != "loss" and r["first_ok"] == r["s_prod"] + 1:
            out["p3b"]["e_dissolved"] += 1
        # P3c
        for lo, key in ((2, "p3c1"), (1, "p3c2")):
            idx = [i for i in range(len(steps)) if lo <= steps[i] <= r["s_prod"]]
            sel = max(idx, key=lambda i: r["q"][i]) if idx else r["i_prod"]
            u_sel = float(corr[sel]) - STEP_COST * (r["s_prod"] - 1.0)
            v = score(out[key], r, u_sel, sel != r["i_prod"])
            out[key]["changed"] = out[key].get("changed", 0) + int(sel != r["i_prod"])
            if key == "p3c1" and r["verdict"] == "loss" and v != "loss" \
               and r["first_ok"] is not None and r["first_ok"] < r["s_prod"]:
                out[key]["c_dissolved"] += 1

    # P3d: per-cell delta, GroupKFold(5) by task_id, OOF
    rids = list(valid.keys())
    tasks = np.array([valid[r]["task"] for r in rids])
    gkf2 = GroupKFold(n_splits=max(2, min(5, len(set(tasks)))))
    for tr_idx, te_idx in gkf2.split(rids, groups=tasks):
        tr = [rids[i] for i in tr_idx]
        best_delta, best_u = 0.0, -1e18
        for delta in DELTAS:
            tot = 0.0
            for rid in tr:
                r = valid[rid]
                i = stop_from_mu(r["steps"], r["mu"], delta)
                tot += util(r["corr"][i], r["steps"][i])
            if tot > best_u or (tot == best_u and abs(delta) < abs(best_delta)):
                best_u, best_delta = tot, float(delta)
        out["p3d"]["deltas"].append(best_delta)
        for i_ in te_idx:
            r = valid[rids[i_]]
            i = stop_from_mu(r["steps"], r["mu"], best_delta)
            u_new = util(r["corr"][i], r["steps"][i])
            score(out["p3d"], r, u_new, i != r["i_prod"])

    cf.write_text(json.dumps(out))
    return out


def main():
    cells = sorted(p.parent.name for p in ROOT.glob("*/detector_comparison_by_run.csv"))
    tot = {k: {} for k in ("p3b", "p3c1", "p3c2", "p3d")}
    runs = mism = invalid = 0
    for i, cell in enumerate(cells):
        r = cell_result(cell)
        if r is None:
            continue
        flag = " INVALID" if "INVALID" in r else ""
        print(f"[{i+1}/{len(cells)}] {cell} runs={r['runs']} mism={r['stop_mismatch']}{flag}", flush=True)
        if "INVALID" in r:
            invalid += 1
            continue
        runs += r["runs"]; mism += r["stop_mismatch"]
        for k in tot:
            for kk, vv in r[k].items():
                if kk == "deltas":
                    tot[k].setdefault("deltas", []).extend(vv)
                else:
                    tot[k][kk] = tot[k].get(kk, 0) + vv

    print(f"\n=== corpus {runs} validated runs; {mism} stop mismatches; {invalid} invalid cells")
    b = tot["p3b"]
    print("\n--- P3b churn-gated hysteresis ---"); print(json.dumps(b, indent=1))
    print("CRITERIA: net dU>0:", b["dU"] > 0, f"({b['dU']:.2f})",
          "| E-losses dissolved>=150:", b["e_dissolved"] >= 150, f"({b['e_dissolved']})")
    c = tot["p3c1"]
    print("\n--- P3c best-so-far selection (eligible >=2) ---"); print(json.dumps(c, indent=1))
    print("CRITERIA: C-losses dissolved>=180:", c["c_dissolved"] >= 180, f"({c['c_dissolved']})",
          "| win->loss<=90:", c["win2loss"] <= 90, f"({c['win2loss']})",
          f"| net dU={c['dU']:.2f}")
    c2 = tot["p3c2"]
    print("\n--- P3c2 selection incl. step 1 (answer-selection framing ONLY) ---")
    print(json.dumps(c2, indent=1))
    d = tot["p3d"]
    ds = d.pop("deltas", [])
    print("\n--- P3d per-cell threshold calibration (OOF) ---"); print(json.dumps(d, indent=1))
    print("fitted deltas: mean", float(np.mean(ds)) if ds else None,
          "range", (min(ds), max(ds)) if ds else None)
    print("CRITERION: P3d OOF net dU > max(P3a=-8033.65, P3b, P3c):",
          d["dU"] > max(-8033.65, b["dU"], c["dU"]), f"({d['dU']:.2f})")


if __name__ == "__main__":
    main()
