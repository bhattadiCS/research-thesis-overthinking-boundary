"""Run a frozen, prefix-trained drift policy against actually generated baselines.

    python research/run_learned_online_stopping.py --baseline-dir research/outputs/semester2/online_stopping_20261002 --predictor-artifact MODEL.json --output-dir research/outputs/semester2/online_stopping_20261002/learned_main

Only learned-policy trajectories are generated here. The never-stop baseline
was actually executed by the linked original manifest, with the same public
prompts, local model bytes, generation adapter, horizon and initial batch order.
Shared-prefix identity is measured rather than assumed. Reusing that baseline
does not turn saved-prefix replay into an actual learned-policy generation run.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from learned_online_stopping_controller import LearnedOnlineStoppingController, LearnedPolicy
from online_stopping_controller import CancellationToken, Observation, PublicTask
from run_online_stopping_evaluation import (DEFAULT_OUTPUT, file_sha256, grade_live, latency_summary, read_jsonl, runtime_provenance,
                                            write_csv, write_json, write_jsonl)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    original = json.loads((args.baseline_dir / "live_manifest.json").read_text(encoding="utf-8"))
    if not (args.baseline_dir / "live_metrics.json").is_file():
        raise ValueError("complete, already graded actual baseline collection is required")
    if (args.output_dir / "live_manifest.json").exists():
        raise FileExistsError("learned manifest already frozen; choose a new output directory")
    if original["generation"]["do_sample"] or original["baseline_policy"]["max_steps"] != 5:
        raise ValueError("this learned-policy comparison requires greedy five-step baselines")
    for name, digest in original["code_sha256"].items():
        if name in {"online_generation.py", "online_stopping_controller.py"} and file_sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError("original generator/controller bytes changed; paired generation protocol cannot be reused")
    model_path = Path(original["model_path"])
    for name, digest in original["model_files_sha256"].items():
        if file_sha256(model_path / name) != digest:
            raise ValueError("model/tokenizer bytes changed since baseline generation")
    for name, key in (("live_public_tasks.jsonl", "public_tasks_sha256"), ("live_sealed_gold.jsonl", "sealed_gold_sha256")):
        if file_sha256(args.baseline_dir / name) != original[key]:
            raise ValueError("original frozen task/gold ledger changed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Copy labels as opaque bytes. Actual policy collection never parses them.
    for name in ("live_public_tasks.jsonl", "live_sealed_gold.jsonl"):
        shutil.copyfile(args.baseline_dir / name, args.output_dir / name)
    shutil.copyfile(args.predictor_artifact, args.output_dir / "frozen_prefix_predictor.json")
    artifact_hash = file_sha256(args.output_dir / "frozen_prefix_predictor.json")
    if args.expected_predictor_sha256 and artifact_hash != args.expected_predictor_sha256:
        raise ValueError("predictor artifact does not match the explicit final handoff SHA256")
    policy = LearnedPolicy(artifact_hash)
    names = ("online_stopping_controller.py", "online_generation.py", "run_online_stopping_evaluation.py",
             "learned_online_stopping_controller.py", "prefix_stopping_model.py", Path(__file__).name)
    locked = args.output_dir / "locked_code"
    locked.mkdir()
    sources = {}
    for name in names:
        source = Path(__file__).with_name(name)
        shutil.copyfile(source, locked / name)
        sources[name] = file_sha256(locked / name)
    write_json(locked / "code_bindings.json", sources)
    manifest = {**original, "schema_version": "paired-learned-prefix-live-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "phase": "development", "confirmation_eligible": False,
        "active_policy": asdict(policy), "active_policy_sha256": policy.sha256,
        "trained_predictor": {"path": "frozen_prefix_predictor.json", "sha256": artifact_hash,
                             "training_source": "archived leader-only GSM8K-train and MATH; task-disjoint fit/calibration/test split",
                             "runtime_inputs": "observed current/prior observations and public domain; no gold, labels or future observations"},
        "baseline_reused": True, "baseline_manifest_path": str(args.baseline_dir / "live_manifest.json"),
        "baseline_manifest_sha256": file_sha256(args.baseline_dir / "live_manifest.json"),
        "baseline_results_sha256": file_sha256(args.baseline_dir / "live_generation_results.jsonl"),
        "baseline_batch_metrics_sha256": file_sha256(args.baseline_dir / "live_batch_metrics.csv"),
        "code_sha256": sources, "runtime": runtime_provenance(),
        "comparison": "actual newly generated learned trajectories paired with actual previously generated never-stop trajectories; shared-prefix identity audited",
        "stopping_rule": "mu=(p_next-q_current)*(reward_value+wrong_penalty)-step_cost; stop if mu<=0 after>=2 steps; terminal5 enforced before predictor inference"}
    write_json(args.output_dir / "live_manifest.json", manifest)
    return manifest


def collect_batch(tasks: list[PublicTask], predictor: Any, policy: LearnedPolicy, generator: Any,
                  chunk: int, batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    controllers = [LearnedOnlineStoppingController(predictor, task.domain, policy) for task in tasks]
    tokens = [CancellationToken() for _ in tasks]
    decisions: list[list[dict[str, Any]]] = [[] for _ in tasks]
    active = list(range(len(tasks)))
    started = time.perf_counter()
    try:
        for step in range(1, policy.max_steps + 1):
            before = len(generator.metrics)
            observations = generator.generate_batch([tasks[i] for i in active], [controllers[i].prefix for i in active], step, [tokens[i] for i in active])
            if len(observations) != len(active):
                raise ValueError("generator must return one observation per active task")
            batches.extend(asdict(metric) | {"policy": policy.name, "chunk": chunk, "step": step} for metric in generator.metrics[before:])
            survivors = []
            for i, observation in zip(active, observations):
                decision = controllers[i].observe(observation)
                decisions[i].append(asdict(decision))
                if decision.stop:
                    tokens[i].cancel()
                else:
                    survivors.append(i)
            active = survivors
            if not active:
                break
    finally:
        for token in tokens:
            token.cancel()
        generator.cancel()
    if active:
        raise AssertionError("learned terminal policy failed to close all tasks")
    elapsed = time.perf_counter() - started
    results = []
    for i, task in enumerate(tasks):
        prefix, final = controllers[i].prefix, decisions[i][-1]
        results.append({"task_id": task.task_id, "policy": policy.name, "answer": final["selected_answer"], "selected_step": final["selected_step"],
            "stopped_at_step": final["step"], "reason": final["reason"], "observations": [asdict(o) for o in prefix], "decisions": decisions[i],
            "generated_tokens": sum(o.generated_tokens for o in prefix), "prompt_tokens": sum(o.prompt_tokens for o in prefix),
            "auxiliary_tokens": sum(o.auxiliary_tokens for o in prefix), "peer_tokens": 0, "chunk_elapsed_seconds": elapsed})
    return results


def benchmark_predictor(output: Path, predictor: Any, policy: LearnedPolicy,
                        tasks: list[PublicTask], actual_baselines: list[dict[str, Any]]) -> None:
    baseline = {row["task_id"]: row for row in actual_baselines}
    rows = []
    for repeat in range(20):
        for task in tasks:
            controller = LearnedOnlineStoppingController(predictor, task.domain, policy)
            for raw in baseline[task.task_id]["observations"]:
                obs = Observation(**raw)
                started = time.perf_counter_ns()
                decision = controller.observe(obs)
                rows.append({"repeat": repeat, "task_id": task.task_id, "step": obs.step,
                             "latency_ms": (time.perf_counter_ns() - started) / 1e6})
                if decision.stop:
                    break
    write_csv(output / "learned_decision_latency_repeated_prefixes.csv", rows)
    summary = latency_summary(rows) | {"distinct_problems": len(tasks), "repeats": 20,
        "kind": "learned_controller_only_on_actual_baseline_prefixes", "includes": "feature extraction, current/next probability heads, calibration, runtime drift and validation",
        "excludes": "model generation/loading, tokenization, peer waits"}
    write_json(output / "learned_latency_summary.json", summary)
    print(json.dumps({"learned_latency": summary}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--predictor-artifact", type=Path, required=True)
    parser.add_argument("--expected-predictor-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args)
    from prefix_stopping_model import PrefixStoppingModel
    from online_generation import HuggingFaceStepGenerator
    predictor = PrefixStoppingModel.load(args.output_dir / "frozen_prefix_predictor.json")
    policy = LearnedPolicy(**manifest["active_policy"])
    tasks = [PublicTask(**row) for row in read_jsonl(args.output_dir / "live_public_tasks.jsonl")]
    original_results = read_jsonl(args.baseline_dir / "live_generation_results.jsonl")
    all_results = [row for row in original_results if row["policy"] == "never"]
    if len(all_results) != len(tasks) or {r["task_id"] for r in all_results} != {t.task_id for t in tasks}:
        raise ValueError("actual baseline roster mismatch")
    with (args.baseline_dir / "live_batch_metrics.csv").open(encoding="utf-8", newline="") as handle:
        batches = [row for row in csv.DictReader(handle) if row["policy"] == "never"]
    benchmark_predictor(args.output_dir, predictor, policy, tasks, all_results)
    generation = manifest["generation"]
    started = time.perf_counter()
    generator = HuggingFaceStepGenerator(manifest["model_path"], device=generation["device"], max_steps=policy.max_steps, max_new_tokens=generation["max_new_tokens"])
    load_seconds = time.perf_counter() - started
    for chunk, offset in enumerate(range(0, len(tasks), generation["batch_size"])):
        task_batch = tasks[offset:offset + generation["batch_size"]]
        results = collect_batch(task_batch, predictor, policy, generator, chunk, batches)
        all_results.extend(results)
        write_jsonl(args.output_dir / "live_generation_results.jsonl", all_results)
        write_csv(args.output_dir / "live_batch_metrics.csv", batches)
        print(json.dumps({"completed_policy": policy.name, "chunk": chunk, "problems": len(task_batch), "generated_tokens": sum(r["generated_tokens"] for r in results),
                          "elapsed_since_start_seconds": time.perf_counter() - started}), flush=True)
    write_json(args.output_dir / "live_collection_runtime.json", {"model_load_seconds": load_seconds,
        "new_learned_collection_elapsed_seconds_including_load": time.perf_counter() - started,
        "baseline_execution_reused_from": str(args.baseline_dir), "torch": generator.torch.__version__,
        "gpu": generator.torch.cuda.get_device_name(0) if generation["device"] == "cuda" else None,
        "peak_cuda_allocated_bytes": generator.torch.cuda.max_memory_allocated() if generation["device"] == "cuda" else 0})
    # Only after all actual learned generation, read sealed labels for grading.
    grade_live(args.output_dir)
    metrics_path = args.output_dir / "live_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.update({"kind": "actual_local_llm_learned_prefix_policy_paired_with_actual_prior_baseline", "baseline_execution_reused": True,
                    "baseline_manifest_sha256": manifest["baseline_manifest_sha256"],
                    "predictor_sha256": manifest["trained_predictor"]["sha256"],
                    "risk_note": "prefix-trained calibrated development predictor; covariate shift from archived instrument and no registered noninferiority margin",
                    "learned_stop_reasons": dict(Counter(row["reason"] for row in all_results if row["policy"] == policy.name))})
    write_json(metrics_path, metrics)
    print(json.dumps({"learned_live_final_metrics": metrics}), flush=True)
    from analyze_online_stopping_results import analyze
    analyze(args.output_dir)


if __name__ == "__main__":
    main()
