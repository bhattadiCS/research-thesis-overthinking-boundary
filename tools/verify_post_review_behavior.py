"""Verify revised controllers on saved prefixes without changing frozen outputs.

This performs saved-prefix replay and structural audits, not new model inference.
Use --output-dir to choose a new directory; an existing directory is never replaced.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from analyze_online_stopping_results import audit_ledgers
from learned_online_stopping_controller import LearnedOnlineStoppingController, LearnedPolicy
from online_stopping_controller import Observation, OnlineStoppingController, PolicyConfig
from prefix_stopping_model import PrefixStoppingModel
from run_online_stopping_evaluation import file_sha256, read_jsonl, write_json


def verify(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    predictor_path = ROOT / "research/outputs/semester2/prefix_model_v1/prefix_model.json"
    predictor = PrefixStoppingModel.load(predictor_path)
    base = ROOT / "research/outputs/semester2/online_stopping_20261002"
    reports = {}
    for name in ("main", "adversarial_live", "learned_main", "learned_adversarial"):
        source = base if name == "main" else base / name
        fixture = output / "ledger_audits" / name
        fixture.mkdir(parents=True)
        filenames = ("live_manifest.json", "live_public_tasks.jsonl", "live_sealed_gold.jsonl",
                     "live_generation_results.jsonl", "live_batch_metrics.csv")
        inputs = {}
        for filename in filenames:
            shutil.copyfile(source / filename, fixture / filename)
            inputs[(source / filename).relative_to(ROOT).as_posix()] = file_sha256(fixture / filename)
        shutil.copytree(source / "locked_code", fixture / "locked_code")
        audit = audit_ledgers(fixture)
        manifest = json.loads((fixture / "live_manifest.json").read_text(encoding="utf-8"))
        tasks = {row["task_id"]: row for row in read_jsonl(fixture / "live_public_tasks.jsonl")}
        results = read_jsonl(fixture / "live_generation_results.jsonl")
        mismatches = []
        for row in results:
            if row["policy"] == manifest["baseline_policy"]["name"]:
                controller = OnlineStoppingController(PolicyConfig(**manifest["baseline_policy"]))
            elif manifest["schema_version"] == "paired-learned-prefix-live-v1":
                controller = LearnedOnlineStoppingController(predictor, tasks[row["task_id"]]["domain"],
                                                             LearnedPolicy(**manifest["active_policy"]))
            else:
                controller = OnlineStoppingController(PolicyConfig(**manifest["active_policy"]))
            actual = []
            for raw in row["observations"]:
                # Original monotonic timestamps belong to their original process.
                observation = dataclasses.replace(Observation(**raw), observed_ns=0)
                decision = controller.observe(observation)
                actual.append((decision.step, decision.stop, decision.reason,
                               decision.selected_answer, decision.selected_step))
                if decision.stop:
                    break
            recorded = [(d["step"], d["stop"], d["reason"], d["selected_answer"], d["selected_step"])
                        for d in row["decisions"]]
            if actual != recorded:
                mismatches.append({"task_id": row["task_id"], "policy": row["policy"]})
        reports[name] = {"recorded_rows": len(results), "matching_decision_histories": len(results) - len(mismatches),
                         "mismatches": mismatches, "ledger_checks": audit["checks"], "input_sha256": inputs}
    sources = ("research/online_stopping_controller.py", "research/learned_online_stopping_controller.py",
               "research/prefix_stopping_model.py", "research/analyze_online_stopping_results.py",
               "research/run_online_stopping_evaluation.py", "tools/verify_post_review_behavior.py")
    report = {"kind": "post_review_saved_prefix_behavior_verification_v1", "panels": reports,
              "source_sha256": {path: file_sha256(ROOT / path) for path in sources},
              "predictor_sha256": file_sha256(predictor_path),
              "all_passed": all(not r["mismatches"] and all(r["ledger_checks"].values()) for r in reports.values()),
              "claim_limit": "Saved-prefix replay and isolated ledger audits only; no new model generation, accuracy grading, training, or timing benchmark. Learned ledgers include reused baseline rows."}
    write_json(output / "behavior_verification.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    report = verify(parser.parse_args().output_dir.resolve())
    print(json.dumps({"all_passed": report["all_passed"],
                      "recorded_rows": sum(p["recorded_rows"] for p in report["panels"].values()),
                      "matching_decision_histories": sum(p["matching_decision_histories"] for p in report["panels"].values())}))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
