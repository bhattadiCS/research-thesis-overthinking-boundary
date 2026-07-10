"""Pre-registered offline policy experiments P1 / P2 / P3a / P13 on frozen traces.

Census counterfactual harness for the protocols in
ThesisDocs/rigor_audit/04_next_experiment_protocols.md; results recorded in
ThesisDocs/rigor_audit/05_offline_experiment_results.md (2026-07-10 run:
P1 FAILED net dU=-116.55; P2 FLOOR DEFENDED dU(T_MIN=3)=-1,912; P3a FAILED
net dU=-8,034; P13 monotone temperature trend confirmed 4/4 datasets).

Counterfactual semantics: stop steps come from the frozen hazard_drift
decisions in detector_comparison_by_run.csv; a guard veto defers the stop to
the first later step where the predicate clears (drift-persistence
assumption), else the run's last step; utility = correct - 0.05*(step-1);
never_stop rows are never modified. Validation: the recomputed stop-step
utility must equal the pipeline's stop_utility (mismatches are counted and
reported; 0/75,965 on the 2026-07-10 corpus).

Usage:
    python research/offline_policy_experiments.py [--matrix-root DIR] [--cache-dir DIR]

Per-cell results are cached as JSON in --cache-dir, so an interrupted census
resumes where it left off. Delete the cache after the matrix changes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STEP_COST = 0.05

spec = importlib.util.spec_from_file_location("cl", HERE / "classify_losses.py")
cl = importlib.util.module_from_spec(spec)
sys.modules["cl"] = cl
spec.loader.exec_module(cl)
ta = cl.trace_analysis  # provides the production _sanitize_step_frame

USECOLS = {"run_id", "task_id", "step", "correct", "utility", "answer", "answer_normalized",
           "parse_success", "answer_extraction_source", "model_stop_flag", "hit_max_new_tokens",
           "temperature", "answer_changed", "confidence", "entropy_mean"}


def util(correct: float, step: float) -> float:
    return float(correct) - STEP_COST * (step - 1.0)


def defer_stop(steps, corrects, mask_bad, s0):
    """First step > s0 where mask_bad is False; else the run's last step."""
    for i in range(len(steps)):
        if steps[i] > s0 and not mask_bad[i]:
            return steps[i], corrects[i]
    return steps[-1], corrects[-1]


def cell_result(root: Path, cache: Path, cell: str):
    cache_f = cache / f"{cell}.json"
    if cache_f.exists():
        return json.loads(cache_f.read_text())
    d = root / cell
    if not (d / "detector_comparison_by_run.csv").exists():
        return None
    dcbr = pd.read_csv(d / "detector_comparison_by_run.csv")
    hd = dcbr[dcbr.detector == "hazard_drift"].set_index("run_id")
    ns = dcbr[dcbr.detector == "never_stop"].set_index("run_id")
    ts = pd.read_csv(d / "trace_steps.csv", usecols=lambda c: c in USECOLS)
    ts, _ = ta._sanitize_step_frame(ts)
    ts = ts[ts.run_id.isin(hd.index)].sort_values(["run_id", "step"])

    out = {"cell": cell, "mismatch": 0, "runs": 0,
           "p1": {"aff_loss": 0, "aff_win": 0, "aff_tie": 0, "dissolved": 0, "win2loss": 0, "dU": 0.0},
           "p2": {"aff": 0, "aff_win": 0, "aff_loss": 0, "aff_tie": 0, "dissolved": 0, "win2loss": 0, "dU": 0.0},
           "p3a": {"aff_loss": 0, "aff_win": 0, "aff_tie": 0, "dissolved": 0, "win2loss": 0, "dU": 0.0},
           "p13": {}}

    for rid, g in ts.groupby("run_id", sort=False):
        if rid not in hd.index or rid not in ns.index:
            continue
        steps = g.step.astype(int).to_numpy()
        corr = g.correct.fillna(0).astype(float).to_numpy()
        ans = g.answer_normalized
        empty = (ans.isna() | (ans.astype(str).str.strip() == "")).to_numpy()
        capped = g.hit_max_new_tokens.fillna(0).astype(float).to_numpy() >= 1
        s0 = int(hd.loc[rid, "stop_step"])
        u_ns = float(ns.loc[rid, "stop_utility"])
        u_hd = float(hd.loc[rid, "stop_utility"])
        idx0 = np.where(steps == s0)[0]
        if len(idx0) == 0:
            out["mismatch"] += 1
            continue
        i0 = int(idx0[0])
        if abs(util(corr[i0], s0) - u_hd) > 1e-9:
            out["mismatch"] += 1
        out["runs"] += 1
        verdict = "win" if u_hd > u_ns else ("tie" if u_hd == u_ns else "loss")

        temp = float(g.temperature.iloc[0])
        dct = out["p13"].setdefault(f"{temp:.2f}", [0, 0])
        dct[0] += 1
        dct[1] += int(verdict == "loss")

        def apply_guard(mask_bad, bucket):
            if not mask_bad[i0]:
                return
            bucket[f"aff_{verdict}"] = bucket.get(f"aff_{verdict}", 0) + 1
            s1, c1 = defer_stop(steps, corr, mask_bad, s0)
            u1 = util(c1, s1)
            bucket["dU"] += u1 - u_hd
            v1 = "win" if u1 > u_ns else ("tie" if u1 == u_ns else "loss")
            if verdict == "loss" and v1 != "loss":
                bucket["dissolved"] += 1
            if verdict == "win" and v1 == "loss":
                bucket["win2loss"] += 1

        apply_guard(empty, out["p1"])
        apply_guard(capped, out["p3a"])

        if s0 == 2:
            b = out["p2"]
            b["aff"] += 1
            b[f"aff_{verdict}"] = b.get(f"aff_{verdict}", 0) + 1
            later = steps[steps >= 3]
            if len(later):
                s1 = int(later[0])
                c1 = corr[np.where(steps == s1)[0][0]]
            else:
                s1, c1 = int(steps[-1]), corr[-1]
            u1 = util(c1, s1)
            b["dU"] += u1 - u_hd
            v1 = "win" if u1 > u_ns else ("tie" if u1 == u_ns else "loss")
            if verdict == "loss" and v1 != "loss":
                b["dissolved"] += 1
            if verdict == "win" and v1 == "loss":
                b["win2loss"] += 1

    cache_f.write_text(json.dumps(out))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix-root", default=str(HERE / "outputs" / "experiment_matrix"))
    ap.add_argument("--cache-dir", default=".p123_cache")
    args = ap.parse_args()
    root = Path(args.matrix_root)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    cells = cl.cell_names(root, None)
    agg: dict = {}
    total: dict = {"p1": {}, "p2": {}, "p3a": {}}
    p13: dict = {}
    mism = runs = 0
    for i, cell in enumerate(cells):
        r = cell_result(root, cache, cell)
        if r is None:
            print(f"[{i+1}/{len(cells)}] {cell} SKIPPED (no dcbr)", flush=True)
            continue
        print(f"[{i+1}/{len(cells)}] {cell} runs={r['runs']} mismatch={r['mismatch']}", flush=True)
        mism += r["mismatch"]
        runs += r["runs"]
        ds = cell.split("__")[1]
        for p in ("p1", "p2", "p3a"):
            for k, v in r[p].items():
                total[p][k] = total[p].get(k, 0) + v
                agg.setdefault(ds, {}).setdefault(p, {})
                agg[ds][p][k] = agg[ds][p].get(k, 0) + v
        for t, (n, l) in r["p13"].items():
            d = p13.setdefault(ds, {}).setdefault(t, [0, 0])
            d[0] += n
            d[1] += l

    print("\n=== corpus:", runs, "runs; stop-utility mismatches:", mism)
    p1 = total["p1"]
    print("\n--- P1 empty-answer guard ---")
    print(json.dumps(p1, indent=1))
    print("CRITERIA: net dU>0:", p1["dU"] > 0, f"({p1['dU']:.2f})",
          "| dissolved>=100:", p1.get("dissolved", 0) >= 100, f"({p1.get('dissolved', 0)})",
          "| win->loss<=50:", p1.get("win2loss", 0) <= 50, f"({p1.get('win2loss', 0)})")
    p2 = total["p2"]
    print("\n--- P2 T_MIN 2 vs 3 ---")
    print(json.dumps(p2, indent=1))
    print("CRITERION (floor defended = moving to 3 does NOT gain): dU(T_MIN=3) =",
          f"{p2['dU']:.2f}", "->", "FLOOR DEFENDED" if p2["dU"] <= 0 else
          ("FLOOR CHALLENGED (>200)" if p2["dU"] > 200 else "small positive - floor stands per criterion"))
    p3 = total["p3a"]
    print("\n--- P3a token-cap guard ---")
    print(json.dumps(p3, indent=1))
    print("CRITERIA: net dU>0:", p3["dU"] > 0, f"({p3['dU']:.2f})",
          "| dissolved>=200:", p3.get("dissolved", 0) >= 200, f"({p3.get('dissolved', 0)})")
    print("\n--- per-dataset dU ---")
    for ds in sorted(agg):
        print(ds, {p: round(agg[ds][p].get("dU", 0), 2) for p in ("p1", "p2", "p3a")},
              {p: agg[ds][p].get("dissolved", 0) for p in ("p1", "p2", "p3a")})

    print("\n--- P13 Cochran-Armitage (scores 0/1/2 for temps asc) ---")
    sig_up = 0
    for ds in sorted(p13):
        temps = sorted(p13[ds], key=float)
        N = [p13[ds][t][0] for t in temps]
        L = [p13[ds][t][1] for t in temps]
        x = list(range(len(temps)))
        n = sum(N)
        pbar = sum(L) / n
        T = sum(xi * li for xi, li in zip(x, L))
        ET = pbar * sum(xi * ni for xi, ni in zip(x, N))
        var = pbar * (1 - pbar) * (sum(xi * xi * ni for xi, ni in zip(x, N)) -
                                   (sum(xi * ni for xi, ni in zip(x, N)) ** 2) / n)
        Z = (T - ET) / math.sqrt(var) if var > 0 else float("nan")
        rates = [f"{t}:{li / ni * 100:.2f}%" for t, ni, li in zip(temps, N, L)]
        up = Z > 1.96
        sig_up += int(up)
        print(f"{ds:8s} Z={Z:+.2f} {'SIG-UP' if up else 'ns/other'}  " + " ".join(rates))
    print("CRITERION: increasing trend Z>1.96 in >=3/4 datasets:", sig_up >= 3, f"({sig_up}/4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
