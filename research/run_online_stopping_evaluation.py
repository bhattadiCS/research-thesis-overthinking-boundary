"""Reproducible online-stopper latency, locked replay, and true local live runs.

Examples (PowerShell, repository root)::

    python research/run_online_stopping_evaluation.py --benchmark --replay
    python research/run_online_stopping_evaluation.py --live --max-tasks 100
    python research/run_online_stopping_evaluation.py --live --task-file tasks.jsonl --gold-file gold.jsonl --output-dir research/outputs/semester2/adversarial_live

Replay is retrospective evaluation on saved, measured completion tokens. It
does not save real generation work. ``--live`` loads an existing local snapshot
and actually schedules only the surviving problems at each next reasoning
step. Public tasks, policies, labels, model, and code are hashed before model
generation. Gold is read by the grader only after both policies finish.
These are development experiments, not a prospective confirmation of the
retrospective 0.955 ensemble, and no learned model is silently substituted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import shutil
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from online_stopping_controller import CancellationToken, Observation, OnlineStoppingController, PolicyConfig, PublicTask


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "research/outputs/semester2/online_stopping_20261002"
DEFAULT_MODEL = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DEFAULT_TRACES = [ROOT / "research/outputs/experiments_v2/global_qwen2p5_0p5b_math/trace_steps.csv",
                  ROOT / "research/outputs/experiments_v2/global_qwen2p5_0p5b_gsm8k/trace_steps.csv"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_provenance() -> dict[str, Any]:
    """Record the actual interpreter without importing or modifying packages."""
    packages = {}
    for name in ("torch", "transformers", "numpy", "pandas", "scikit-learn", "datasets", "safetensors", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "python_executable": sys.executable,
            "platform": platform.platform(), "packages": packages}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in rows))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def boolean(value: Any) -> bool:
    if str(value).lower().strip() in {"1", "1.0", "true"}:
        return True
    if str(value).lower().strip() in {"0", "0.0", "false", ""}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def measured_token(value: Any, name: str) -> int:
    number = float(value)
    if not math.isfinite(number) or number < 0 or int(number) != number:
        raise ValueError(f"invalid or unmeasured {name}: {value!r}")
    return int(number)


def sanitized_observation(row: dict[str, Any]) -> Observation:
    confidence = float(row["confidence"]) if str(row.get("confidence", "")).strip() else None
    if confidence is not None and not math.isfinite(confidence):
        confidence = None
    # Only these explicit observable columns cross into the runtime API.
    return Observation(
        step=int(row["step"]), answer=str(row.get("answer_normalized") or row.get("answer") or ""),
        confidence=confidence, parse_success=boolean(row.get("parse_success", 0)),
        generated_tokens=measured_token(row["raw_generation_tokens"], "raw_generation_tokens"),
        auxiliary_tokens=measured_token(row.get("k2_raw_generation_tokens", 0) or 0, "k2_raw_generation_tokens"),
    )


def load_replay(paths: list[Path]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    runs: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if boolean(row.get("is_baseline", 0)):
                    continue
                source_key = f"{path.parent.name}::{row['run_id']}"
                if source_key not in runs:
                    runs[source_key] = (str(row["task_id"]), [])
                runs[source_key][1].append(row)
    result = []
    for key, (task_id, rows) in sorted(runs.items()):
        rows.sort(key=lambda row: int(row["step"]))
        if [int(row["step"]) for row in rows] != list(range(1, 6)):
            raise ValueError(f"replay requires exactly five consecutive steps: {key}")
        # Validate token measurement even for suffixes never exposed to policy.
        for row in rows:
            sanitized_observation(row)
        result.append((key, task_id, rows))
    if not result:
        raise ValueError("no complete replay trajectories")
    return result


def latency_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(float(row["latency_ms"]) for row in rows)
    def percentile(probability: float) -> float:
        index = (len(values) - 1) * probability
        low, high = math.floor(index), math.ceil(index)
        return values[low] + (values[high] - values[low]) * (index - low)
    return {"decisions": len(values), "mean_ms": statistics.mean(values), "median_ms": percentile(.5),
            "p95_ms": percentile(.95), "p99_ms": percentile(.99), "max_ms": max(values),
            "over_10ms": sum(value >= 10 for value in values), "all_under_10ms": max(values) < 10}


def benchmark(runs: list[tuple[str, str, list[dict[str, Any]]]], output: Path, repeats: int) -> None:
    tasks: dict[str, tuple[str, str, list[dict[str, Any]]]] = {}
    math_runs = [run for run in runs if run[2][0].get("domain") == "math"]
    for run in math_runs or runs:
        tasks.setdefault(run[1], run)
    selected = list(tasks.values())[:100]
    if len(selected) != 100:
        raise ValueError("100 distinct math problem IDs are required for the latency milestone")
    rows = []
    policies = (PolicyConfig(), PolicyConfig(mode="never", name="never"))
    for repeat in range(repeats):
        for key, task_id, trace in selected:
            for policy in policies:
                controller = OnlineStoppingController(policy)
                for row in trace:
                    observation = sanitized_observation(row)  # prepared outside timed decision
                    started = time.perf_counter_ns()
                    decision = controller.observe(observation)
                    elapsed = time.perf_counter_ns() - started
                    rows.append({"repeat": repeat, "policy": policy.name, "task_id": task_id, "trajectory": key,
                                 "step": observation.step, "latency_ms": elapsed / 1e6})
                    if decision.stop:
                        break
    write_csv(output / "latency_100_math_problems.csv", rows)
    summary = {"kind": "controller_only_on_saved_math_prefixes", "distinct_problems": 100, "repeats": repeats,
               "all": latency_summary(rows),
               "policies": {p.name: latency_summary([r for r in rows if r["policy"] == p.name]) for p in policies},
               "timer": "perf_counter_ns", "cold_first_repeat": latency_summary([r for r in rows if r["repeat"] == 0]),
               "includes": "validation, prefix-state update, actual confidence/stability policy, answer selection and terminal decisions; excludes model generation, loading, telemetry extraction or peer wait",
               "runtime": {"python": platform.python_version(), "platform": platform.platform()}}
    write_json(output / "latency_summary.json", summary)
    print(json.dumps({"benchmark": summary}), flush=True)


def paired_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        raise ValueError("paired metrics require at least one complete pair")
    base = sum(int(row["baseline_correct"]) for row in rows)
    active = sum(int(row["active_correct"]) for row in rows)
    improved = sum(not row["baseline_correct"] and row["active_correct"] for row in rows)
    worsened = sum(row["baseline_correct"] and not row["active_correct"] for row in rows)
    baseline_tokens = sum(row["baseline_generated_tokens"] for row in rows)
    active_tokens = sum(row["active_generated_tokens"] for row in rows)
    baseline_aux = sum(row.get("baseline_auxiliary_tokens", 0) for row in rows)
    active_aux = sum(row.get("active_auxiliary_tokens", 0) for row in rows)
    baseline_prompt = sum(row.get("baseline_prompt_tokens", 0) for row in rows)
    active_prompt = sum(row.get("active_prompt_tokens", 0) for row in rows)
    prompts_available = all("baseline_prompt_tokens" in row and "active_prompt_tokens" in row for row in rows)
    changes = improved + worsened
    exact_p = min(1.0, 2 * sum(math.comb(changes, k) for k in range(min(improved, worsened) + 1)) / 2**changes) if changes else 1.0
    def wilson(correct: int) -> list[float]:
        z, p = 1.959963984540054, correct / n
        scale = 1 + z*z/n
        center = (p + z*z/(2*n)) / scale
        half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / scale
        return [center - half, center + half]
    return {
        "problems_or_trajectories": n, "baseline_correct": base, "active_correct": active,
        "baseline_accuracy": base / n, "active_accuracy": active / n, "accuracy_delta": (active - base) / n,
        "baseline_accuracy_wilson_95ci": wilson(base), "active_accuracy_wilson_95ci": wilson(active),
        "paired_improved": improved, "paired_worsened": worsened, "paired_unchanged": n - changes,
        "mcnemar_exact_two_sided_p": exact_p, "baseline_generated_tokens": baseline_tokens,
        "active_generated_tokens": active_tokens,
        "measured_completion_token_savings": 1 - active_tokens / baseline_tokens if baseline_tokens else None,
        "baseline_auxiliary_tokens": baseline_aux, "active_auxiliary_tokens": active_aux,
        "completion_plus_auxiliary_token_savings": 1 - (active_tokens + active_aux) / (baseline_tokens + baseline_aux) if baseline_tokens + baseline_aux else None,
        "baseline_prompt_tokens": baseline_prompt if prompts_available else None,
        "active_prompt_tokens": active_prompt if prompts_available else None,
        "total_prompt_completion_auxiliary_token_savings": 1 - (active_tokens + active_aux + active_prompt) / (baseline_tokens + baseline_aux + baseline_prompt) if prompts_available and baseline_tokens + baseline_aux + baseline_prompt else None,
        "mean_active_stop_step": statistics.mean(row["active_stop_step"] for row in rows),
    }


def replay(runs: list[tuple[str, str, list[dict[str, Any]]]], paths: list[Path], output: Path) -> None:
    policies = [PolicyConfig(mode="never", name="never"),
                *(PolicyConfig(mode="fixed", name=f"fixed_{step}", fixed_step=step) for step in (2, 3, 4)),
                *(PolicyConfig(name=f"confidence_{threshold}", confidence_threshold=threshold) for threshold in (80, 90, 95))]
    # Serialize all variants before labels/outcomes are evaluated.
    write_json(output / "locked_replay_policies.json", {"kind": "development_locked_replay", "policies": [asdict(p) | {"sha256": p.sha256} for p in policies],
               "sources": [{"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)} for path in paths],
               "trained_predictor": None, "selection": "fixed variants; no label-dependent tuning"})
    results = []
    for policy in policies:
        paired_rows = []
        for key, task_id, trace in runs:
            # Use the SAME latest-nonempty terminal selector in both arms.
            # Historical final-row correctness treats an empty last response
            # as wrong even when this runtime retains an earlier candidate.
            baseline_controller = OnlineStoppingController(PolicyConfig(mode="never", name="never"))
            for baseline_row in trace:
                baseline_decision = baseline_controller.observe(sanitized_observation(baseline_row))
            controller = OnlineStoppingController(policy)
            consumed = []
            for row in trace:
                observation = sanitized_observation(row)
                decision = controller.observe(observation)
                consumed.append(observation)
                if decision.stop:
                    break
            paired_rows.append({"policy": policy.name, "trajectory": key, "task_id": task_id,
                "baseline_correct": boolean(trace[baseline_decision.selected_step - 1]["correct"]),
                "active_correct": boolean(trace[decision.selected_step - 1]["correct"]),
                "baseline_generated_tokens": sum(measured_token(r["raw_generation_tokens"], "raw_generation_tokens") for r in trace),
                "active_generated_tokens": sum(o.generated_tokens for o in consumed),
                "baseline_auxiliary_tokens": sum(measured_token(r.get("k2_raw_generation_tokens", 0) or 0, "k2_raw_generation_tokens") for r in trace),
                "active_auxiliary_tokens": sum(o.auxiliary_tokens for o in consumed),
                "active_stop_step": decision.step, "selected_step": decision.selected_step, "reason": decision.reason})
        write_csv(output / f"replay_{policy.name}_paired.csv", paired_rows)
        results.append({"policy": policy.name, "policy_sha256": policy.sha256, **paired_metrics(paired_rows)})
    for result in results:
        result["pareto_nondominated"] = not any(
            other["active_generated_tokens"] <= result["active_generated_tokens"] and other["active_accuracy"] >= result["active_accuracy"]
            and (other["active_generated_tokens"] < result["active_generated_tokens"] or other["active_accuracy"] > result["active_accuracy"])
            for other in results
        )
    write_json(output / "replay_metrics.json", {"kind": "retrospective_locked_policy_replay", "live_compute_savings_measured": False,
        "real_completion_token_lengths_available": True, "prompt_tokens_available": False,
        "peer_tokens": "no peer features requested; recorded second-path tokens included when present",
        "qualification": "saved legacy-instrument trajectories; identical latest-nonempty terminal selection in both arms; correctness evaluated outside controller; not prospective and no threshold confirmation; missing prompt costs are not treated as zero",
        "policies": results})
    write_csv(output / "replay_pareto.csv", results)
    print(json.dumps({"replay_trajectories": len(runs), "policies": results}), flush=True)


def prepare_live(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir
    manifest_path = output / "live_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("live manifest already frozen: use a new directory, or --grade-live for finished collection")
    if args.task_file:
        public_rows = read_jsonl(args.task_file)
        if not args.gold_file:
            raise ValueError("a separate gold-file is required for offline grading of custom tasks")
        label_rows = read_jsonl(args.gold_file)
        source = {"kind": "custom_public_task_bank", "public_file": str(args.task_file), "public_sha256": file_sha256(args.task_file),
                  "gold_file": str(args.gold_file), "gold_sha256": file_sha256(args.gold_file)}
    else:
        from datasets import Dataset
        cache = Path.home() / ".cache/huggingface/datasets/openai___gsm8k"
        matches = sorted(cache.rglob("gsm8k-test.arrow"))
        if not matches:
            raise FileNotFoundError("cached GSM8K test Arrow file not found; provide --task-file and --gold-file (no download attempted)")
        data = Dataset.from_file(str(matches[0]))
        indices = list(range(len(data)))
        random.Random(args.seed).shuffle(indices)
        public_rows, label_rows = [], []
        for index in indices[:args.max_tasks]:
            example = data[index]
            prompt = str(example["question"])
            task_id = f"gsm8k_test_{index:05d}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}"
            public_rows.append(asdict(PublicTask(task_id, prompt, "number", "gsm8k", "grade_school_math")))
            label_rows.append({"task_id": task_id, "expected_answer": str(example["answer"]).split("####")[-1].strip(), "answer_type": "number", "source_index": index})
        source = {"kind": "cached_GSM8K_test", "path": str(matches[0]), "sha256": file_sha256(matches[0]),
                  "allocation": "shuffle seed fixed before collection; first requested count", "source_indices": indices[:args.max_tasks]}
    tasks = [PublicTask(**row) for row in public_rows]
    if not tasks or len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError("task IDs must be unique and task set nonempty")
    if {row["task_id"] for row in label_rows} != {t.task_id for t in tasks} or len(label_rows) != len(tasks):
        raise ValueError("sealed label ledger must match public task IDs exactly")
    for row in label_rows:
        if not isinstance(row.get("expected_answer"), str):
            raise ValueError("all gold values must be strings")
    policy = PolicyConfig(max_steps=args.max_steps)
    baseline = PolicyConfig(name="never", mode="never", max_steps=args.max_steps)
    write_jsonl(output / "live_public_tasks.jsonl", public_rows)
    write_jsonl(output / "live_sealed_gold.jsonl", label_rows)
    sources = {name: file_sha256(Path(__file__).with_name(name)) for name in
               ("online_stopping_controller.py", "online_generation.py", Path(__file__).name)}
    locked = output / "locked_code"
    locked.mkdir(exist_ok=True)
    for name in sources:
        shutil.copyfile(Path(__file__).with_name(name), locked / name)
    write_json(locked / "code_bindings.json", sources)
    model_path = args.model_path.resolve()
    model_files = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
    model_hashes = {name: file_sha256(model_path / name) for name in model_files if (model_path / name).is_file()}
    if "model.safetensors" not in model_hashes:
        raise FileNotFoundError("this runner requires complete local single-file safetensors model weights")
    manifest = {"schema_version": "paired-local-live-stopping-v1", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "confirmation_eligible": False, "phase": "development", "public_task_count": len(tasks), "source": source,
        "public_tasks_sha256": file_sha256(output / "live_public_tasks.jsonl"), "sealed_gold_sha256": file_sha256(output / "live_sealed_gold.jsonl"),
        "active_policy": asdict(policy), "active_policy_sha256": policy.sha256, "baseline_policy": asdict(baseline),
        "baseline_policy_sha256": baseline.sha256, "trained_predictor": None, "code_sha256": sources,
        "model_path": str(model_path), "model_snapshot": model_path.name, "model_files_sha256": model_hashes,
        "generation": {"device": args.device, "do_sample": False, "temperature": 0.0, "max_new_tokens": args.max_new_tokens,
                       "batch_size": args.batch_size, "seed": args.seed, "boundary": "first strictly valid complete JSON object or EOS or token cap"},
        "runtime": runtime_provenance(),
        "peer_protocol": {"mode": "single_model", "roster": [], "peer_tokens": 0, "peer_features_used": False},
        "accounting": "all main generated tokens, repeated input tokens, padded compute slots and wall time; no diagnostic/verifier/peer calls",
        "novelty": "development; source novelty versus historic corpus not assumed; not confirmation of full-sequence detector"}
    write_json(manifest_path, manifest)
    return manifest


def collect_policy_batch(tasks: list[PublicTask], config: PolicyConfig, generator: Any, chunk: int, batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    controllers = [OnlineStoppingController(config) for _ in tasks]
    tokens = [CancellationToken() for _ in tasks]
    decision_rows: list[list[dict[str, Any]]] = [[] for _ in tasks]
    active = list(range(len(tasks)))
    start = time.perf_counter()
    try:
        for step in range(1, config.max_steps + 1):
            before = len(generator.metrics)
            observations = generator.generate_batch([tasks[i] for i in active], [controllers[i].prefix for i in active], step, [tokens[i] for i in active])
            if len(observations) != len(active):
                raise ValueError("generator must return one observation per active task")
            batches.extend(asdict(metric) | {"policy": config.name, "chunk": chunk, "step": step} for metric in generator.metrics[before:])
            survivors = []
            for i, observation in zip(active, observations):
                decision = controllers[i].observe(observation)
                decision_rows[i].append(asdict(decision))
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
        raise AssertionError("terminal policy failed to close all tasks")
    elapsed = time.perf_counter() - start
    result = []
    for i, task in enumerate(tasks):
        prefix = controllers[i].prefix
        final = decision_rows[i][-1]
        result.append({"task_id": task.task_id, "policy": config.name, "answer": final["selected_answer"], "selected_step": final["selected_step"],
            "stopped_at_step": final["step"], "reason": final["reason"], "observations": [asdict(o) for o in prefix], "decisions": decision_rows[i],
            "generated_tokens": sum(o.generated_tokens for o in prefix), "prompt_tokens": sum(o.prompt_tokens for o in prefix),
            "auxiliary_tokens": sum(o.auxiliary_tokens for o in prefix), "peer_tokens": 0, "chunk_elapsed_seconds": elapsed})
    return result


def live(args: argparse.Namespace) -> None:
    from online_generation import HuggingFaceStepGenerator
    manifest = prepare_live(args)  # freeze before loading/generation
    # Runtime input has no correctness/gold fields and no label ledger reads.
    tasks = [PublicTask(**row) for row in read_jsonl(args.output_dir / "live_public_tasks.jsonl")]
    active_policy, baseline_policy = PolicyConfig(**manifest["active_policy"]), PolicyConfig(**manifest["baseline_policy"])
    started = time.perf_counter()
    generator = HuggingFaceStepGenerator(args.model_path, device=args.device, max_steps=args.max_steps, max_new_tokens=args.max_new_tokens)
    load_seconds = time.perf_counter() - started
    all_results, batches = [], []
    for chunk, offset in enumerate(range(0, len(tasks), args.batch_size)):
        task_batch = tasks[offset:offset + args.batch_size]
        for policy in (baseline_policy, active_policy):
            result = collect_policy_batch(task_batch, policy, generator, chunk, batches)
            all_results.extend(result)
            write_jsonl(args.output_dir / "live_generation_results.jsonl", all_results)
            write_csv(args.output_dir / "live_batch_metrics.csv", batches)
            print(json.dumps({"completed_policy": policy.name, "chunk": chunk, "problems": len(task_batch),
                              "completed_results": len(all_results), "generated_tokens": sum(r["generated_tokens"] for r in result),
                              "elapsed_since_start_seconds": time.perf_counter() - started}), flush=True)
    runtime = {"model_load_seconds": load_seconds, "collection_elapsed_seconds_including_load": time.perf_counter() - started,
               "torch": generator.torch.__version__, "device": args.device,
               "gpu": generator.torch.cuda.get_device_name(0) if args.device == "cuda" else None,
               "peak_cuda_allocated_bytes": generator.torch.cuda.max_memory_allocated() if args.device == "cuda" else 0}
    write_json(args.output_dir / "live_collection_runtime.json", runtime)
    # First and only label-read phase, after all actual generation is complete.
    grade_live(args.output_dir)


def grade_live(output: Path) -> None:
    from real_trace_experiments import normalize_answer, math_answers_equivalent
    manifest = json.loads((output / "live_manifest.json").read_text(encoding="utf-8"))
    if file_sha256(output / "live_public_tasks.jsonl") != manifest["public_tasks_sha256"] or file_sha256(output / "live_sealed_gold.jsonl") != manifest["sealed_gold_sha256"]:
        raise ValueError("frozen task/label hash mismatch")
    labels = {row["task_id"]: row for row in read_jsonl(output / "live_sealed_gold.jsonl")}
    generated: dict[str, dict[str, dict[str, Any]]] = {}
    latency = []
    for result in read_jsonl(output / "live_generation_results.jsonl"):
        if result["policy"] in generated.setdefault(result["task_id"], {}):
            raise ValueError("duplicate live result")
        generated[result["task_id"]][result["policy"]] = result
        latency.extend({"task_id": result["task_id"], "policy": result["policy"], "step": d["step"], "latency_ms": d["latency_ns"] / 1e6} for d in result["decisions"])
    if set(generated) != set(labels):
        raise ValueError("live collection incomplete")
    rows = []
    for task_id, label in labels.items():
        baseline = generated[task_id]["never"]
        active = generated[task_id][manifest["active_policy"]["name"]]
        def correct(answer: str) -> bool:
            if label["answer_type"] == "math":
                return math_answers_equivalent(answer, label["expected_answer"])
            return normalize_answer(answer, label["answer_type"]) == normalize_answer(label["expected_answer"], label["answer_type"])
        prefix_match = all(a["raw_text"] == b["raw_text"] for a, b in zip(active["observations"], baseline["observations"]))
        rows.append({"task_id": task_id, "baseline_correct": correct(baseline["answer"]), "active_correct": correct(active["answer"]),
            "baseline_answer": baseline["answer"], "active_answer": active["answer"], "expected_answer": label["expected_answer"],
            "baseline_generated_tokens": baseline["generated_tokens"], "active_generated_tokens": active["generated_tokens"],
            "baseline_prompt_tokens": baseline["prompt_tokens"], "active_prompt_tokens": active["prompt_tokens"],
            "baseline_auxiliary_tokens": baseline["auxiliary_tokens"], "active_auxiliary_tokens": active["auxiliary_tokens"],
            "active_stop_step": active["stopped_at_step"], "selected_step": active["selected_step"], "reason": active["reason"],
            "shared_prefix_identical": prefix_match, "trap_category": label.get("trap_category", "")})
    write_csv(output / "live_paired_results.csv", rows)
    write_csv(output / "live_controller_latency.csv", latency)
    metrics = paired_metrics(rows)
    metrics.update({"kind": "actual_local_llm_paired_generation", "development_only": True,
                    "model": manifest["model_snapshot"], "policy_sha256": manifest["active_policy_sha256"],
                    "shared_prefix_identical_problems": sum(row["shared_prefix_identical"] for row in rows),
                    "stop_reasons": dict(Counter(row["reason"] for row in rows)), "decision_latency": latency_summary(latency),
                    "runtime_api_used_gold_or_future": False, "peer_generation_tokens": 0, "verifier_generation_tokens": 0,
                    "baseline_parse_success_steps": sum(o["parse_success"] for value in generated.values() for o in value["never"]["observations"]),
                    "active_parse_success_steps": sum(o["parse_success"] for value in generated.values() for o in value[manifest["active_policy"]["name"]]["observations"]),
                    "token_note": "actual generated completion tokens including EOS, excludes padded decoder slots; all repeated prompt tokens reported separately; no proxy or unit-step savings",
                    "risk_note": "model-reported confidence is uncalibrated; this is a heuristic development test, not deployment of the full-sequence 0.955 detector"})
    with (output / "live_batch_metrics.csv").open(encoding="utf-8", newline="") as handle:
        batches = list(csv.DictReader(handle))
    metrics["batch_compute"] = {policy: {
        "model_seconds": sum(float(b["model_seconds"]) for b in batches if b["policy"] == policy),
        "tokenize_seconds": sum(float(b["tokenize_seconds"]) for b in batches if b["policy"] == policy),
        "padded_prefill_token_slots": sum(int(b["padded_prefill_token_slots"]) for b in batches if b["policy"] == policy),
        "decode_token_slots": sum(int(b["decode_token_slots"]) for b in batches if b["policy"] == policy),
        "calls": sum(b["policy"] == policy for b in batches),
    } for policy in ("never", manifest["active_policy"]["name"])}
    write_json(output / "live_metrics.json", metrics)
    print(json.dumps({"live_metrics": metrics}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--grade-live", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trace", type=Path, action="append")
    parser.add_argument("--benchmark-repeats", type=int, default=20)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--task-file", type=Path)
    parser.add_argument("--gold-file", type=Path)
    parser.add_argument("--max-tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if not any((args.benchmark, args.replay, args.live, args.grade_live)):
        parser.error("select at least one evaluation action")
    if args.benchmark_repeats < 1 or args.batch_size < 1 or args.max_tasks < 1:
        parser.error("repeat, batch size and task count must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.benchmark or args.replay:
        paths = args.trace or DEFAULT_TRACES
        runs = load_replay(paths)
        if args.benchmark:
            benchmark(runs, args.output_dir, args.benchmark_repeats)
        if args.replay:
            replay(runs, paths, args.output_dir)
    if args.live:
        live(args)
    elif args.grade_live:
        grade_live(args.output_dir)


if __name__ == "__main__":
    main()
