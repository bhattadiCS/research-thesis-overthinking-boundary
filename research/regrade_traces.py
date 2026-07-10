#!/usr/bin/env python
"""Re-grade persisted trace_steps.csv with the fixed answer graders (audit remediation).

Recomputes `correct`, `answer_normalized`, and `utility` IN PLACE from the stored
`answer` and `expected_answer` columns, using the corrected graders in
real_trace_experiments.py. Generation features (entropy, logprobs, hidden shifts)
are untouched — only label-derived fields change — so this needs no GPU and no
re-collection. After re-grading, re-run trace_analysis.py per cell and
cross_family_analysis.py to refresh all downstream artifacts.

Usage:
    python research/regrade_traces.py --root research/outputs/experiment_matrix
    python research/regrade_traces.py --root <dir> --dry-run   # report only
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rte", HERE / "real_trace_experiments.py")
rte = importlib.util.module_from_spec(spec)
sys.modules["rte"] = rte
spec.loader.exec_module(rte)

STEP_COST = 0.05


def _answer_type(domain: str, task_source: str) -> str:
    # Must mirror collection-time answer_type per dataset: ARC/GPQA are
    # multiple-choice; grading them as "number" rewrites ~12,284 correct
    # labels (rigor_audit/01 §4b).
    src = (task_source or domain or "").strip().lower()
    if src == "math":
        return "math"
    if src in ("arc", "gpqa"):
        return "mcq"
    return "number"


# Well-formed run_id suffix; SIGKILL-corruption fragments fail this and must
# not be re-graded as if they were rows (rigor_audit/01 Axis 5a).
_RUN_ID_OK = re.compile(r"__temp\d+(?:\.\d+)?__seed\d+$")


def regrade_file(path: Path, dry_run: bool) -> tuple[int, int]:
    # keep_default_na=False keeps literal 'N/A'-style answers as strings
    # (pandas would otherwise coerce them to NaN); na_values=[""] keeps
    # genuinely empty fields detectable via pd.isna.
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    if not {"answer", "expected_answer", "correct", "step"}.issubset(df.columns):
        return (0, 0)
    new_correct, new_norm, new_util = [], [], []
    flipped = 0
    skipped_fragments = 0
    for _, row in df.iterrows():
        rid = row.get("run_id")
        if not isinstance(rid, str) or not _RUN_ID_OK.search(rid):
            skipped_fragments += 1
            new_correct.append(row.get("correct"))
            new_norm.append(row.get("answer_normalized"))
            new_util.append(row.get("utility"))
            continue
        atype = _answer_type(str(row.get("domain", "")), str(row.get("task_source", "")))
        ans = "" if pd.isna(row.get("answer")) else str(row.get("answer"))
        exp = "" if pd.isna(row.get("expected_answer")) else str(row.get("expected_answer"))
        task = rte.TaskSpec(task_id="", domain=str(row.get("domain", "")), difficulty="",
                            prompt="", answer_type=atype, expected_answer=exp,
                            notes="", source=str(row.get("task_source", "")), source_index=-1)
        correct = int(rte.verify_answer(task, ans))
        step_raw = row.get("step", 1)
        try:
            step = 1 if pd.isna(step_raw) else int(float(step_raw))
        except (TypeError, ValueError):
            step = 1
        new_correct.append(correct)
        new_norm.append(rte.normalize_answer(ans, atype))
        new_util.append(float(correct) - STEP_COST * (step - 1))
        prev_correct_raw = row.get("correct", 0)
        try:
            prev_correct = 0 if pd.isna(prev_correct_raw) else int(float(prev_correct_raw))
        except (TypeError, ValueError):
            prev_correct = 0
        if correct != prev_correct:
            flipped += 1
    if not dry_run:
        df["correct"] = new_correct
        df["answer_normalized"] = new_norm
        df["utility"] = new_util
        df.to_csv(path, index=False)
    return (flipped, len(df))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="Directory containing <cell>/trace_steps.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = sorted(Path(args.root).glob("*/trace_steps.csv"))
    if not files:
        print(f"No trace_steps.csv under {args.root}")
        return 1
    total_flips = total_rows = 0
    print(f"{'cell':44} {'flipped':>9} {'rows':>7} {'%':>6}")
    for f in files:
        flips, rows = regrade_file(f, args.dry_run)
        total_flips += flips
        total_rows += rows
        pct = (100.0 * flips / rows) if rows else 0.0
        print(f"{f.parent.name:44} {flips:>9} {rows:>7} {pct:>5.2f}%")
    pct = (100.0 * total_flips / total_rows) if total_rows else 0.0
    print(f"{'TOTAL':44} {total_flips:>9} {total_rows:>7} {pct:>5.2f}%")
    print(("[dry-run] no files written." if args.dry_run else "Re-graded in place. "
           "Now re-run trace_analysis.py per cell + cross_family_analysis.py."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
