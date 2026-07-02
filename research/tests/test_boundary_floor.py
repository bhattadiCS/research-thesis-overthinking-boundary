#!/usr/bin/env python
"""Regression test: boundary-detection code must respect T_MIN (the
step-1 forced-commit floor audit fix, commit ef45dda).

This guards against a THIRD script reimplementing "first step where
predictable drift mu_t <= 0" without the T_MIN floor -- exactly the bug
independently reintroduced by run_stakes_sweep.py's find_boundary() and
compare_prompted_vs_distilled.py's boundary lookup before this fix. Step 1
is the forced-commit init state (the model hasn't had a chance to revise
anything yet), not a real decision point, and must never be returned as a
stopping boundary.

Run:  python research/tests/test_boundary_floor.py
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_DIR))

import pandas as pd  # noqa: E402

import trace_analysis as ta  # noqa: E402
import run_stakes_sweep as rss  # noqa: E402
import compare_prompted_vs_distilled as cpd  # noqa: E402


def _synthetic_haz() -> pd.DataFrame:
    """A hazard-like frame where the step-1 drift is already <= 0 (mimicking
    the forced-commit init state), genuine steps 2-4 are positive, and step 5
    onward crosses back to <= 0. A T_MIN-floored detector must return 5;
    an unfloored one would wrongly return 1."""
    rows = []
    for step in range(1, 8):
        if step == 1:
            mu = -0.5
        elif step < 5:
            mu = 0.1
        else:
            mu = -0.1
        rows.append({"step": step, "mu_t": mu, "q_t": 0.5, "alpha_t": 0.2, "beta_t": 0.1})
    return pd.DataFrame(rows)


def main() -> int:
    failures = 0
    synthetic = _synthetic_haz()

    # 1. trace_analysis.first_zero_crossing is the audited reference impl.
    got = ta.first_zero_crossing(synthetic, "mu_t")
    ok = got == 5 and got >= ta.T_MIN
    failures += not ok
    print(f"  {'OK' if ok else 'XX'} trace_analysis.first_zero_crossing floors at T_MIN: got {got} (want 5)")

    # 2. run_stakes_sweep.find_boundary must produce a T_MIN-floored result
    #    (it derives its own mu_t column internally from q_t/alpha_t/beta_t,
    #    so drive it with a penalty of 0 against columns equivalent to the
    #    synthetic mu_t case above).
    haz_in = synthetic[["step", "q_t", "alpha_t", "beta_t"]].copy()
    # alpha/beta/q_t alone don't reproduce an arbitrary mu_t curve, so instead
    # assert the structural property directly: find_boundary never returns a
    # step below T_MIN, on real or degenerate input.
    got = rss.find_boundary(haz_in, penalty=0.0)
    ok = got >= ta.T_MIN
    failures += not ok
    print(f"  {'OK' if ok else 'XX'} run_stakes_sweep.find_boundary never returns step < T_MIN: got {got} (T_MIN={ta.T_MIN})")

    got_empty = rss.find_boundary(haz_in.iloc[0:0], penalty=0.0)
    ok = got_empty >= ta.T_MIN
    failures += not ok
    print(f"  {'OK' if ok else 'XX'} run_stakes_sweep.find_boundary floors the empty-input fallback: got {got_empty}")

    # 3. compare_prompted_vs_distilled's boundary lookup (now first_zero_crossing).
    got = cpd.first_zero_crossing(synthetic, "mu_t")
    ok = got == 5 and got >= ta.T_MIN
    failures += not ok
    print(f"  {'OK' if ok else 'XX'} compare_prompted_vs_distilled boundary path respects T_MIN: got {got} (want 5)")

    # 4. Structural guard: neither script may reimplement first-crossing scan
    #    logic locally -- both must delegate to trace_analysis.first_zero_crossing.
    rss_src = inspect.getsource(rss.find_boundary)
    ok = "first_zero_crossing" in rss_src and "for _, row in" not in rss_src
    failures += not ok
    print(f"  {'OK' if ok else 'XX'} run_stakes_sweep.find_boundary delegates to trace_analysis.first_zero_crossing (no local scan loop)")

    cpd_main_src = inspect.getsource(cpd.main)
    ok = "first_zero_crossing" in cpd_main_src and '<= 0]["step"].values' not in cpd_main_src
    failures += not ok
    print(f"  {'OK' if ok else 'XX'} compare_prompted_vs_distilled.main delegates to first_zero_crossing (no local unfloored filter)")

    # 5. Both scripts must import T_MIN from trace_analysis (not redefine it).
    ok = rss.T_MIN is ta.T_MIN and cpd.T_MIN is ta.T_MIN
    failures += not ok
    print(f"  {'OK' if ok else 'XX'} both scripts import (not redefine) trace_analysis.T_MIN")

    total = 6
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
