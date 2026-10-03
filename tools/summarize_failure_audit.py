"""Validate and summarize the completed, frozen-label failure classification.

Run classify_losses.py first. Historical category tags in that script contain
old probe scores and interpretations; this summary deliberately makes no claim
that those probes or a label regrade ran in the present audit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "research/reports/thesis_failure_audit_v1"
LABELS = {
    "B_stopped_on_empty_answer": "No extracted candidate at the stop",
    "C_passed_earlier_correct": "An earlier decision-eligible candidate was correct",
    "D_step1_only_correct": "Step one was correct; no eligible pre-stop answer was correct",
    "E_next_step_repair": "First eligible correct answer arrived one step after the stop",
    "F_late_repair": "First eligible correct answer arrived at least two steps after the stop",
}


def main() -> None:
    runs = pd.read_csv(FOLDER / "per_run_verdicts.csv")
    losses = pd.read_csv(FOLDER / "loss_classification.csv")
    taxonomy = pd.read_csv(FOLDER / "taxonomy_summary.csv")
    if runs.run_id.duplicated().any() or losses.run_id.duplicated().any():
        raise ValueError("Duplicate trajectories in paired audit")
    expected_loss_ids = set(runs.loc[runs.verdict == "loss", "run_id"])
    if set(losses.run_id) != expected_loss_ids or set(losses.category) != set(LABELS):
        raise ValueError("Loss taxonomy does not partition the paired losses")
    if not np.allclose(runs.util_diff, runs.hd_util - runs.ns_util, atol=1e-12, rtol=0):
        raise ValueError("Stored paired utility differences are inconsistent")
    verdicts = np.where(runs.util_diff > 0, "win", np.where(runs.util_diff < 0, "loss", "tie"))
    if not np.array_equal(verdicts, runs.verdict.to_numpy()):
        raise ValueError("Stored verdicts do not match the signs of paired utility differences")
    for prefix in ("hd", "ns"):
        correct = runs[f"{prefix}_util"] + .05 * (runs[f"{prefix}_step"] - 1)
        if not np.all(np.isclose(correct, 0, atol=1e-10) | np.isclose(correct, 1, atol=1e-10)):
            raise ValueError("Utilities do not reconstruct binary endpoint labels")
        if prefix == "hd" and not np.allclose(correct[runs.verdict == "loss"], 0, atol=1e-10):
            raise ValueError("A loss stopped on a correct candidate")
        if prefix == "ns" and not np.allclose(correct[runs.verdict == "loss"], 1, atol=1e-10):
            raise ValueError("A loss did not have a correct terminal candidate")
    counts = {key: int(value) for key, value in runs.verdict.value_counts().items()}
    categories = []
    for key, label in LABELS.items():
        count = int((losses.category == key).sum())
        stored = taxonomy.loc[taxonomy.category == key, "count"]
        if len(stored) != 1 or int(stored.iloc[0]) != count:
            raise ValueError("Taxonomy summary and individual classifications differ")
        categories.append({"category": key, "description": label, "count": count,
                           "share_of_losses": count / len(losses)})
    sources = [*sorted(FOLDER.glob("*.csv")), FOLDER / "join_audit.txt", FOLDER / "slice_tables.txt",
               ROOT / "research/classify_losses.py", ROOT / "research/analyze_runs.py",
               ROOT / "research/trace_analysis.py", Path(__file__)]
    result = {
        "command": ".venv/Scripts/python.exe research/classify_losses.py --matrix-root research/outputs/experiment_matrix --out research/reports/thesis_failure_audit_v1",
        "followup_command": "python tools/summarize_failure_audit.py",
        "audit": "passed: unique paired runs, complete loss partition, utility arithmetic, and binary endpoint reconstruction",
        "trajectory_count": len(runs), "verdict_counts": counts,
        "verdict_rates": {key: value / len(runs) for key, value in counts.items()},
        "categories": categories,
        "scope": "Archived fitted hazard policy versus full recorded horizon, using frozen correctness labels; descriptive development evidence",
        "regrade_executed": False, "new_probe_training_executed": False,
        "historical_category_tags": "Ignore embedded old probe AUCs and claims about prediction limits; they were not recomputed by this command",
        "source_sha256": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
    }
    (FOLDER / "audit_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"audit": result["audit"], "trajectories": len(runs), "verdicts": counts}))


if __name__ == "__main__":
    main()
