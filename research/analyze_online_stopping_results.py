"""Offline uncertainty, ledger audit, and Pareto figures after live collection.

This module never calls a model. Shadow policy points are explicitly replay of
the completed live baseline, not additional actual online-generation trials.
The live active point uses its independently generated, actually stopped run.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from online_stopping_controller import Observation, OnlineStoppingController, PolicyConfig
from run_online_stopping_evaluation import (DEFAULT_OUTPUT, file_sha256, read_jsonl, write_csv, write_json)


def paired_bootstrap(rows: list[dict[str, str]], *, seed: int = 20261002, samples: int = 10000) -> dict[str, Any]:
    n = len(rows)
    delta = np.asarray([float(r["active_correct"] == "True") - float(r["baseline_correct"] == "True") for r in rows])
    baseline = np.asarray([int(r["baseline_generated_tokens"]) for r in rows])
    active = np.asarray([int(r["active_generated_tokens"]) for r in rows])
    indices = np.random.default_rng(seed).integers(0, n, size=(samples, n))
    differences = delta[indices].mean(axis=1)
    savings = 1 - active[indices].sum(axis=1) / baseline[indices].sum(axis=1)
    return {"resampling_unit": "paired problem", "seed": seed, "samples": samples,
            "accuracy_delta_95ci": np.quantile(differences, [.025, .975]).tolist(),
            "completion_token_savings_95ci": np.quantile(savings, [.025, .975]).tolist(),
            "noninferiority_margin_registered": False,
            "qualification": "descriptive paired bootstrap; no proof of zero loss or population safety"}


def audit_ledgers(output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "live_manifest.json").read_text(encoding="utf-8"))
    results = read_jsonl(output / "live_generation_results.jsonl")
    public_tasks = read_jsonl(output / "live_public_tasks.jsonl")
    task_ids = [task["task_id"] for task in public_tasks]
    gold_ids = [row["task_id"] for row in read_jsonl(output / "live_sealed_gold.jsonl")]
    policy_names = (manifest["active_policy"]["name"], manifest["baseline_policy"]["name"])
    checks: dict[str, bool] = {}
    checks["public_tasks_match_frozen_hash"] = file_sha256(output / "live_public_tasks.jsonl") == manifest["public_tasks_sha256"]
    checks["gold_ledger_matches_frozen_hash"] = file_sha256(output / "live_sealed_gold.jsonl") == manifest["sealed_gold_sha256"]
    locked = output / "locked_code"
    checks["executed_source_copies_match_frozen_hashes"] = all(
        (locked / name).is_file() and file_sha256(locked / name) == digest for name, digest in manifest["code_sha256"].items()
    )
    checks["paired_result_count"] = len(results) == 2 * manifest["public_task_count"]
    checks["public_task_ids_unique"] = bool(task_ids) and len(task_ids) == len(set(task_ids)) == manifest["public_task_count"]
    checks["gold_task_coverage"] = Counter(gold_ids) == Counter(task_ids)
    checks["paired_task_policy_coverage"] = len(set(policy_names)) == 2 and Counter(
        (row["task_id"], row["policy"]) for row in results
    ) == Counter((task_id, policy) for task_id in task_ids for policy in policy_names)
    checks["minimum_two_steps"] = all(r["stopped_at_step"] >= 2 for r in results)
    checks["no_future_steps_after_stop"] = all([o["step"] for o in r["observations"]] == list(range(1, r["stopped_at_step"] + 1)) for r in results)
    checks["one_decision_per_observation"] = all([d["step"] for d in r["decisions"]] == [o["step"] for o in r["observations"]] for r in results)
    checks["exactly_one_terminal_decision"] = all(sum(d["stop"] for d in r["decisions"]) == 1 and r["decisions"][-1]["stop"] for r in results)
    checks["terminal_decision_matches_result"] = all(r["decisions"] and
        r["decisions"][-1]["step"] == r["stopped_at_step"] and
        r["decisions"][-1]["selected_step"] == r["selected_step"] and
        r["decisions"][-1]["selected_answer"] == r["answer"] and
        r["decisions"][-1].get("reason") == r.get("reason") for r in results)
    checks["selected_answer_already_observed"] = all(1 <= r["selected_step"] <= min(r["stopped_at_step"], len(r["observations"])) and r["answer"] == r["observations"][r["selected_step"] - 1]["answer"] for r in results)
    checks["decision_selections_already_observed"] = all(
        1 <= d["selected_step"] <= min(d["step"], len(r["observations"])) and
        d["selected_answer"] == r["observations"][d["selected_step"] - 1]["answer"]
        for r in results for d in r["decisions"])
    checks["observations_precede_decisions"] = all(o["observed_ns"] <= d["decision_ns"] for r in results for o, d in zip(r["observations"], r["decisions"]))
    checks["decision_precedes_next_observation"] = all(d["decision_ns"] <= o["observed_ns"]
        for r in results for d, o in zip(r["decisions"][:-1], r["observations"][1:]))
    costs = [row.get(name, 0) for row in results for name in ("generated_tokens", "prompt_tokens", "auxiliary_tokens", "peer_tokens")]
    costs.extend(o.get(name, 0) for row in results for o in row["observations"] for name in ("generated_tokens", "prompt_tokens", "auxiliary_tokens"))
    costs.extend(d.get("peer_tokens", 0) for row in results for d in row["decisions"])
    checks["nonnegative_integer_costs"] = all(type(value) is int and value >= 0 for value in costs)
    checks["completion_cost_sum"] = all(r["generated_tokens"] == sum(o["generated_tokens"] for o in r["observations"]) for r in results)
    checks["input_cost_sum"] = all(r["prompt_tokens"] == sum(o["prompt_tokens"] for o in r["observations"]) for r in results)
    checks["auxiliary_cost_sum"] = all(r.get("auxiliary_tokens", 0) == sum(o.get("auxiliary_tokens", 0) for o in r["observations"]) for r in results)
    checks["peer_cost_sum"] = all(r.get("peer_tokens", 0) == sum(d.get("peer_tokens", 0) for d in r["decisions"]) for r in results)
    forbidden = {"gold", "gold_answer", "expected_answer", "correct", "selected_correct", "oracle_stop", "utility"}
    checks["no_label_keys_in_runtime_task_or_event"] = not any(forbidden.intersection(o) for r in results for o in r["observations"]) and not any(forbidden.intersection(t) for t in public_tasks)
    with (output / "live_batch_metrics.csv").open(encoding="utf-8", newline="") as handle:
        batches = list(csv.DictReader(handle))
    checks["batch_policies_registered"] = all(batch["policy"] in policy_names for batch in batches)
    batch_costs = [batch.get(name, "0") for batch in batches for name in ("generated_tokens", "prompt_tokens", "padded_prefill_token_slots", "decode_token_slots")]
    checks["batch_nonnegative_integer_costs"] = all(isinstance(value, str) and value.isascii() and value.isdecimal() for value in batch_costs)
    checks["batch_completion_cost_sum"] = checks["batch_nonnegative_integer_costs"] and all(sum(int(b["generated_tokens"]) for b in batches if b["policy"] == policy) == sum(r["generated_tokens"] for r in results if r["policy"] == policy) for policy in {r["policy"] for r in results})
    checks["batch_input_cost_sum"] = checks["batch_nonnegative_integer_costs"] and all(
        sum(int(b.get("prompt_tokens", "0")) for b in batches if b["policy"] == policy) ==
        sum(r["prompt_tokens"] for r in results if r["policy"] == policy) for policy in policy_names)
    checks["prefill_padding_accounted"] = checks["batch_nonnegative_integer_costs"] and all(
        int(b.get("padded_prefill_token_slots", "0")) >= int(b.get("prompt_tokens", "0")) for b in batches)
    checks["decoder_padding_accounted"] = checks["batch_nonnegative_integer_costs"] and all(int(b["decode_token_slots"]) >= int(b["generated_tokens"]) for b in batches)
    report = {"kind": "live_prefix_and_accounting_ledger_audit", "checks": checks, "all_passed": all(checks.values()),
              "qualification": "structural audit and preserved execution sources; these checks establish runtime provenance, not model accuracy"}
    write_json(output / "live_ledger_audit.json", report)
    if not report["all_passed"]:
        raise AssertionError(f"live audit failed: {[name for name, value in checks.items() if not value]}")
    return report


def baseline_replay_pareto(output: Path) -> list[dict[str, Any]]:
    from real_trace_experiments import math_answers_equivalent, normalize_answer
    manifest = json.loads((output / "live_manifest.json").read_text(encoding="utf-8"))
    labels = {row["task_id"]: row for row in read_jsonl(output / "live_sealed_gold.jsonl")}
    results = read_jsonl(output / "live_generation_results.jsonl")
    baseline = {r["task_id"]: r for r in results if r["policy"] == "never"}
    total_baseline_tokens = sum(r["generated_tokens"] for r in baseline.values())
    policies = [PolicyConfig(mode="fixed", name=f"fixed_{step}", fixed_step=step) for step in (2, 3, 4)] + [
        PolicyConfig(name=f"confidence_{threshold}", confidence_threshold=threshold) for threshold in (80, 90, 95)]
    # These exact policies were chosen by the source before either live dataset
    # was graded, matching the locked replay-policy variants in the main run.
    write_json(output / "baseline_replay_policies.json", {"policies": [asdict(p) | {"sha256": p.sha256} for p in policies],
               "kind": "replay_of_live_baseline_prefixes", "threshold_tuning": False})
    points = []
    for policy in policies:
        tokens, corrects, steps = 0, 0, []
        for task_id, run in baseline.items():
            controller = OnlineStoppingController(policy)
            for row in run["observations"]:
                observation = Observation(**row)
                decision = controller.observe(observation)
                tokens += observation.generated_tokens
                if decision.stop:
                    break
            label = labels[task_id]
            if label["answer_type"] == "math":
                correct = math_answers_equivalent(decision.selected_answer, label["expected_answer"])
            else:
                correct = normalize_answer(decision.selected_answer, label["answer_type"]) == normalize_answer(label["expected_answer"], label["answer_type"])
            corrects += correct
            steps.append(decision.step)
        points.append({"policy": policy.name, "kind": "replay_of_live_baseline_prefixes", "problems": len(baseline),
                       "generated_tokens": tokens, "accuracy": corrects / len(baseline), "correct": corrects,
                       "completion_savings": 1 - tokens / total_baseline_tokens, "mean_stop_step": float(np.mean(steps))})
    metrics = json.loads((output / "live_metrics.json").read_text(encoding="utf-8"))
    points.extend([{"policy": "never", "kind": "actual_live_generation", "problems": len(baseline),
                    "generated_tokens": metrics["baseline_generated_tokens"], "accuracy": metrics["baseline_accuracy"],
                    "correct": metrics["baseline_correct"], "completion_savings": 0.0, "mean_stop_step": 5.0},
                   {"policy": manifest["active_policy"]["name"], "kind": "actual_live_generation", "problems": len(baseline),
                    "generated_tokens": metrics["active_generated_tokens"], "accuracy": metrics["active_accuracy"],
                    "correct": metrics["active_correct"], "completion_savings": metrics["measured_completion_token_savings"],
                    "mean_stop_step": metrics["mean_active_stop_step"]}])
    for point in points:
        point["pareto_nondominated_in_displayed_set"] = not any(
            other["generated_tokens"] <= point["generated_tokens"] and other["accuracy"] >= point["accuracy"] and
            (other["generated_tokens"] < point["generated_tokens"] or other["accuracy"] > point["accuracy"]) for other in points)
    write_csv(output / "live_and_baseline_replay_pareto.csv", points)
    return points


def plot_results(output: Path, points: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.2, 4.9), constrained_layout=True)
    replay = [p for p in points if p["kind"] == "replay_of_live_baseline_prefixes"]
    actual = [p for p in points if p["kind"] == "actual_live_generation"]
    ax.scatter([p["generated_tokens"] / p["problems"] for p in replay], [100 * p["accuracy"] for p in replay], c="#7b8492", s=55, marker="o", label="Replay of the live baseline")
    colors = ["#26354b", "#006f9c"]
    for index, (point, color) in enumerate(zip(actual, colors)):
        x, y = point["generated_tokens"] / point["problems"], point["accuracy"] * 100
        ax.scatter([x], [y], c=color, s=115, marker="D", zorder=4, label=f"Actual live: {point['policy']}")
        ax.annotate(f"{point['correct']}/{point['problems']}", (x, y), xytext=(8, -16 - 16 * index),
                    textcoords="offset points", color=color)
    for point in replay:
        xy = (point["generated_tokens"] / point["problems"], 100 * point["accuracy"])
        if point["policy"].startswith("confidence_"):
            threshold = int(point["policy"].split("_")[-1])
            offset = {80: (-110, 72), 90: (-110, 47), 95: (-110, 22)}[threshold]
            ax.annotate(point["policy"], xy, xytext=offset, textcoords="offset points", fontsize=8,
                        color="#545d6a", arrowprops={"arrowstyle": "-", "color": "#9ba2ac", "lw": .6})
        else:
            ax.annotate(point["policy"], xy, xytext=(4, 7), textcoords="offset points", fontsize=8, color="#545d6a")
    ax.set_xlabel("Measured completion tokens per problem (fewer is better)")
    ax.set_ylabel("Graded final-answer accuracy (%)")
    ax.set_title("Online stopping: actual paired run and baseline-prefix replay")
    ax.grid(alpha=.2)
    ax.set_ylim(0, max(10, max(100 * p["accuracy"] for p in points) + 3))
    ax.margins(x=.12)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(output / "online_stopping_pareto.png", dpi=200)
    fig.savefig(output / "online_stopping_pareto.svg")
    plt.close(fig)


def analyze(output: Path) -> dict[str, Any]:
    with (output / "live_paired_results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    bootstrap = paired_bootstrap(rows)
    audit = audit_ledgers(output)
    points = baseline_replay_pareto(output)
    plot_results(output, points)
    summary = {"paired_bootstrap": bootstrap, "ledger_audit_passed": audit["all_passed"], "pareto_points": points,
               "source_sha256": file_sha256(Path(__file__))}
    write_json(output / "live_posthoc_analysis.json", summary)
    print(json.dumps({"output": str(output), "paired_bootstrap": bootstrap, "audit_passed": audit["all_passed"]}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    analyze(args.output_dir)
