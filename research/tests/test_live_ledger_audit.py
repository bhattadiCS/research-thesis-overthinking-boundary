"""Reject incomplete or duplicated paired ledgers that otherwise balance costs."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_online_stopping_results import audit_ledgers
from run_online_stopping_evaluation import file_sha256, write_csv, write_json, write_jsonl


def paired_ledger(directory):
    tasks = [{"task_id": name, "prompt": "12+13"} for name in ("one", "two")]
    write_jsonl(directory / "live_public_tasks.jsonl", tasks)
    write_jsonl(directory / "live_sealed_gold.jsonl", [{"task_id": name, "expected_answer": "25"} for name in ("one", "two")])
    locked = directory / "locked_code"
    locked.mkdir()
    (locked / "fixture.py").write_text("# fixture only\n", encoding="utf-8")
    write_json(directory / "live_manifest.json", {
        "public_task_count": 2,
        "public_tasks_sha256": file_sha256(directory / "live_public_tasks.jsonl"),
        "sealed_gold_sha256": file_sha256(directory / "live_sealed_gold.jsonl"),
        "code_sha256": {"fixture.py": file_sha256(locked / "fixture.py")},
        "active_policy": {"name": "active"}, "baseline_policy": {"name": "never"},
    })
    results = []
    for policy in ("active", "never"):
        for task in tasks:
            observations = [{"step": step, "answer": "25", "observed_ns": step * 20,
                             "generated_tokens": 7, "prompt_tokens": 11} for step in (1, 2)]
            decisions = [{"step": step, "stop": step == 2, "decision_ns": step * 20 + 10,
                          "selected_step": step, "selected_answer": "25"} for step in (1, 2)]
            results.append({"task_id": task["task_id"], "policy": policy, "stopped_at_step": 2,
                            "selected_step": 2, "answer": "25", "observations": observations,
                            "decisions": decisions, "generated_tokens": 14, "prompt_tokens": 22})
    write_jsonl(directory / "live_generation_results.jsonl", results)
    write_csv(directory / "live_batch_metrics.csv", [{"policy": policy, "generated_tokens": 28,
               "prompt_tokens": 44, "padded_prefill_token_slots": 44,
               "decode_token_slots": 28} for policy in ("active", "never")])
    return results


def test_complete_paired_ledger_passes(tmp_path):
    paired_ledger(tmp_path)
    assert audit_ledgers(tmp_path)["all_passed"]


@pytest.mark.parametrize("corruption", ("duplicate_task", "wrong_policy", "missing_decision", "duplicate_public_task", "auxiliary_cost", "peer_cost", "future_step_before_decision", "batch_input_cost", "missing_prefill_cost", "terminal_selection"))
def test_pairing_and_decision_coverage_are_required(tmp_path, corruption):
    results = paired_ledger(tmp_path)
    if corruption == "duplicate_task":
        results[1] = copy.deepcopy(results[0])
    elif corruption == "wrong_policy":
        results[0]["policy"] = "unregistered"
        # Keep accounting consistent so the policy contract detects the change.
        write_csv(tmp_path / "live_batch_metrics.csv", [{"policy": policy, "generated_tokens": count,
                   "prompt_tokens": 44 if policy == "never" else 22,
                   "padded_prefill_token_slots": 44 if policy == "never" else 22,
                   "decode_token_slots": count} for policy, count in (("unregistered", 14), ("active", 14), ("never", 28))])
    elif corruption == "missing_decision":
        results[0]["decisions"] = results[0]["decisions"][1:]
    elif corruption in ("auxiliary_cost", "peer_cost"):
        results[0]["auxiliary_tokens" if corruption == "auxiliary_cost" else "peer_tokens"] = 999
    elif corruption == "future_step_before_decision":
        results[0]["observations"][1]["observed_ns"] = 25
    elif corruption in ("batch_input_cost", "missing_prefill_cost"):
        write_csv(tmp_path / "live_batch_metrics.csv", [{"policy": policy, "generated_tokens": 28,
                   "decode_token_slots": 28, "prompt_tokens": 0 if corruption == "batch_input_cost" else 44,
                   "padded_prefill_token_slots": 0} for policy in ("active", "never")])
    elif corruption == "terminal_selection":
        results[0]["decisions"][-1].update(selected_step=1, selected_answer="999")
    else:
        tasks = [{"task_id": "one", "prompt": "12+13"}] * 2
        write_jsonl(tmp_path / "live_public_tasks.jsonl", tasks)
        manifest = json.loads((tmp_path / "live_manifest.json").read_text())
        manifest["public_tasks_sha256"] = file_sha256(tmp_path / "live_public_tasks.jsonl")
        write_json(tmp_path / "live_manifest.json", manifest)
    write_jsonl(tmp_path / "live_generation_results.jsonl", results)
    with pytest.raises(AssertionError, match="live audit failed"):
        audit_ledgers(tmp_path)


@pytest.mark.parametrize("tokens", (-7, .5, True))
def test_balanced_invalid_token_counts_are_rejected(tmp_path, tokens):
    results = paired_ledger(tmp_path)
    for row in results:
        for observation in row["observations"]:
            observation["generated_tokens"] = tokens
        row["generated_tokens"] = 2 * tokens
    write_jsonl(tmp_path / "live_generation_results.jsonl", results)
    write_csv(tmp_path / "live_batch_metrics.csv", [{"policy": policy, "generated_tokens": 4 * tokens,
               "prompt_tokens": 44, "padded_prefill_token_slots": 44,
               "decode_token_slots": max(0, 4 * tokens)} for policy in ("active", "never")])
    with pytest.raises(AssertionError, match="live audit failed"):
        audit_ledgers(tmp_path)
