from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import set_seed


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = REPO_ROOT / "research"
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from real_trace_experiments import (  # noqa: E402
    MODEL_CATALOG,
    append_records,
    build_pilot_summary,
    chunked,
    first_pending_run,
    infer_existing_runtime_context,
    load_model,
    load_tasks,
    output_paths,
    reconcile_existing_outputs,
    run_batch_traces,
    run_id_for,
    summarize_transitions,
    write_runtime_metadata,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def append_text_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def git_command(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=check,
    )


def ensure_git_author(log_path: Path) -> dict[str, str]:
    name = git_command(["config", "user.name"], check=False).stdout.strip()
    email = git_command(["config", "user.email"], check=False).stdout.strip()
    if name and email:
        append_text_log(log_path, f"Using configured git author: {name} <{email}>")
        return {"action": "existing", "name": name, "email": email}

    author_lines = git_command(["log", "-1", "--format=%an%n%ae"]).stdout.splitlines()
    if len(author_lines) < 2:
        raise RuntimeError("Unable to recover git author identity from recent history.")
    name = author_lines[0].strip()
    email = author_lines[1].strip()
    git_command(["config", "user.name", name])
    git_command(["config", "user.email", email])
    append_text_log(log_path, f"Configured repo-local git author from recent history: {name} <{email}>")
    return {"action": "configured", "name": name, "email": email}


def relative_repo_path(path: Path) -> str:
    resolved = path if path.is_absolute() else (REPO_ROOT / path)
    return str(resolved.relative_to(REPO_ROOT))


def stage_paths(paths: list[Path]) -> None:
    repo_paths = [relative_repo_path(path) for path in paths]
    git_command(["add", "--", *repo_paths])


def staged_changes_exist() -> bool:
    completed = git_command(["diff", "--cached", "--quiet"], check=False)
    return completed.returncode != 0


def count_completed_tasks_for_block(all_runs: list[dict[str, Any]], temperature: float, seed: int) -> int:
    completed_indexes = {
        int(run["task_source_index"])
        for run in all_runs
        if float(run["temperature"]) == float(temperature) and int(run["seed"]) == int(seed)
    }
    return len(completed_indexes)


def write_current_summaries(
    *,
    all_rows: list[dict[str, Any]],
    all_runs: list[dict[str, Any]],
    paths: dict[str, Path],
    model_spec: Any,
    backend: str,
    actual_device: str,
) -> None:
    if not all_rows or not all_runs:
        return
    step_frame = pd.DataFrame(all_rows)
    run_frame = pd.DataFrame(all_runs)
    transition_frame = summarize_transitions(step_frame)
    pilot_summary = build_pilot_summary(step_frame, run_frame, transition_frame, model_spec, backend, actual_device)
    transition_frame.to_csv(paths["hazard"], index=False)
    pilot_summary.to_csv(paths["pilot"], index=False)


def write_metadata_snapshot(
    *,
    paths: dict[str, Path],
    model_spec: Any,
    backend: str,
    actual_device: str,
    quantization: str,
    device_map: str | None,
    attn_implementation: str,
    temperatures: list[float],
    seeds: list[int],
    max_steps: int,
    max_new_tokens: int,
    max_tasks: int,
    task_source: str,
    dataset_split: str,
    dataset_shuffle_seed: int,
    batch_size: int,
    prompt_mode: str,
    system_prompt_mode: str,
    resume_enabled: bool,
    completed_run_ids: set[str],
    total_requested_runs: int,
    reconciliation_report: dict[str, Any],
    tasks: list[Any],
) -> None:
    next_pending = first_pending_run(
        model_spec=model_spec,
        tasks=tasks,
        temperatures=temperatures,
        seeds=seeds,
        completed_run_ids=completed_run_ids if resume_enabled else set(),
    )
    write_runtime_metadata(
        metadata_path=paths["metadata"],
        model_spec=model_spec,
        backend=backend,
        actual_device=actual_device,
        quantization=quantization,
        device_map=device_map,
        attn_implementation=attn_implementation,
        temperatures=temperatures,
        seeds=seeds,
        max_steps=max_steps,
        max_new_tokens=max_new_tokens,
        max_tasks=max_tasks,
        task_source=task_source,
        dataset_split=dataset_split,
        dataset_shuffle_seed=dataset_shuffle_seed,
        batch_size=batch_size,
        prompt_mode=prompt_mode,
        system_prompt_mode=system_prompt_mode,
        resume_enabled=resume_enabled,
        completed_run_ids=completed_run_ids,
        pending_run_count=max(total_requested_runs - len(completed_run_ids), 0),
        next_pending_run=next_pending,
        reconciliation_report=reconciliation_report,
        tasks=tasks,
    )


def checkpoint_commit(
    *,
    output_dir: Path,
    checkpoint_log_path: Path,
    checkpoint_history_path: Path,
    log_path: Path,
    checkpoint_label: str,
    temperature: float,
    seed: int,
    completed_tasks_in_block: int,
    completed_runs_total: int,
    total_requested_runs: int,
) -> None:
    commit_message = f"exp: mistral checkpoint {checkpoint_label}"
    append_jsonl(
        checkpoint_history_path,
        {
            "timestamp_utc": utc_now(),
            "event": "checkpoint_precommit",
            "checkpoint_label": checkpoint_label,
            "temperature": temperature,
            "seed": seed,
            "completed_tasks_in_block": completed_tasks_in_block,
            "completed_runs_total": completed_runs_total,
            "pending_runs_total": max(total_requested_runs - completed_runs_total, 0),
            "commit_message": commit_message,
        },
    )
    stage_paths([output_dir, checkpoint_log_path, checkpoint_history_path])
    if not staged_changes_exist():
        append_text_log(log_path, f"Checkpoint {checkpoint_label} skipped because no staged changes were detected.")
        return

    git_command(["commit", "-m", commit_message])
    commit_hash = git_command(["rev-parse", "HEAD"]).stdout.strip()
    push_result = git_command(["push", "origin", "main"], check=False)
    push_status = "pushed" if push_result.returncode == 0 else "failed"
    if push_status == "pushed":
        append_text_log(
            log_path,
            f"Checkpoint {checkpoint_label} committed as {commit_hash} and pushed successfully.",
        )
    else:
        append_text_log(
            log_path,
            f"Checkpoint {checkpoint_label} committed as {commit_hash} but push failed: {push_result.stderr.strip() or push_result.stdout.strip() or 'unknown error'}",
        )
    append_jsonl(
        checkpoint_history_path,
        {
            "timestamp_utc": utc_now(),
            "event": "checkpoint_result",
            "checkpoint_label": checkpoint_label,
            "temperature": temperature,
            "seed": seed,
            "completed_tasks_in_block": completed_tasks_in_block,
            "completed_runs_total": completed_runs_total,
            "pending_runs_total": max(total_requested_runs - completed_runs_total, 0),
            "commit_message": commit_message,
            "commit_hash": commit_hash,
            "push_status": push_status,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real-trace experiment with aggressive git checkpoints and in-process resume.")
    parser.add_argument("--model", default="mistral_7b_instruct_v0p3", choices=sorted(MODEL_CATALOG.keys()))
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--quantization", default="none", choices=["none", "8bit", "4bit"])
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--attn-implementation", default="sdpa", choices=["auto", "sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.1, 0.6, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-tasks", type=int, default=300)
    parser.add_argument("--task-source", default="gsm8k", choices=["builtin", "gsm8k"])
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-shuffle-seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-mode", default="minimal_json", choices=["structured_four_line", "minimal_json", "answer_only"])
    parser.add_argument("--system-prompt-mode", default="default", choices=["default", "short", "none"])
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "research" / "outputs" / "real_traces_l4_mistral_7b"))
    parser.add_argument("--checkpoint-log", default=str(REPO_ROOT / "research" / "outputs" / "real_traces_l4_mistral_7b_checkpointed.log"))
    parser.add_argument("--checkpoint-history", default=str(REPO_ROOT / "research" / "outputs" / "real_traces_l4_mistral_7b_checkpoint_history.jsonl"))
    parser.add_argument("--checkpoint-every-tasks", type=int, default=25)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_log_path = Path(args.checkpoint_log)
    checkpoint_history_path = Path(args.checkpoint_history)
    ensure_git_author(checkpoint_log_path)
    append_text_log(
        checkpoint_log_path,
        f"Starting checkpointed run for model={args.model}, quantization={args.quantization}, attn={args.attn_implementation}, batch_size={args.batch_size}, max_tasks={args.max_tasks}.",
    )

    hidden_dir = output_dir / "hidden_states"
    paths = output_paths(output_dir, False)
    model_spec = MODEL_CATALOG[args.model]
    tasks = load_tasks(
        task_source=args.task_source,
        max_tasks=args.max_tasks,
        dataset_split=args.dataset_split,
        shuffle_seed=args.dataset_shuffle_seed,
    )
    total_requested_runs = len(tasks) * len(args.temperatures) * len(args.seeds)

    existing_steps, existing_runs, completed_run_ids, reconciliation_report = reconcile_existing_outputs(
        output_dir=output_dir,
        hidden_dir=hidden_dir,
        is_baseline=False,
        max_steps=args.max_steps,
    )
    if reconciliation_report["anomalies_detected"]:
        append_text_log(checkpoint_log_path, f"Reconciled existing checkpoint artifacts: {json.dumps(reconciliation_report, sort_keys=True)}")

    backend, actual_device = infer_existing_runtime_context(paths, args.device)
    pending_requested_runs = total_requested_runs - len(completed_run_ids)
    append_text_log(
        checkpoint_log_path,
        f"Existing completed runs={len(completed_run_ids)} pending_runs={pending_requested_runs} output_dir={relative_repo_path(output_dir)}",
    )

    all_rows: list[dict[str, Any]] = list(existing_steps)
    all_runs: list[dict[str, Any]] = list(existing_runs)
    model = None
    tokenizer = None
    if pending_requested_runs > 0:
        model, tokenizer, actual_device, backend = load_model(
            model_spec=model_spec,
            device=args.device,
            quantization=args.quantization,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
        )
        append_text_log(checkpoint_log_path, f"Model loaded on device={actual_device} backend={backend}")

    write_metadata_snapshot(
        paths=paths,
        model_spec=model_spec,
        backend=backend,
        actual_device=actual_device,
        quantization=args.quantization,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        temperatures=args.temperatures,
        seeds=args.seeds,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        max_tasks=args.max_tasks,
        task_source=args.task_source,
        dataset_split=args.dataset_split,
        dataset_shuffle_seed=args.dataset_shuffle_seed,
        batch_size=args.batch_size,
        prompt_mode=args.prompt_mode,
        system_prompt_mode=args.system_prompt_mode,
        resume_enabled=args.resume,
        completed_run_ids=completed_run_ids,
        total_requested_runs=total_requested_runs,
        reconciliation_report=reconciliation_report,
        tasks=tasks,
    )

    if model is not None and tokenizer is not None:
        for temperature in args.temperatures:
            for seed in args.seeds:
                pending_tasks = [
                    task
                    for task in tasks
                    if not args.resume or run_id_for(model_spec.alias, task, temperature, seed) not in completed_run_ids
                ]
                if not pending_tasks:
                    append_text_log(
                        checkpoint_log_path,
                        f"Skipping temp={temperature:.2f} seed={seed} because all requested runs already exist.",
                    )
                    continue

                set_seed(seed)
                completed_before_block = count_completed_tasks_for_block(all_runs, temperature, seed)
                last_checkpoint_bucket = completed_before_block // max(args.checkpoint_every_tasks, 1)
                append_text_log(
                    checkpoint_log_path,
                    f"Running temp={temperature:.2f} seed={seed} with {len(pending_tasks)} pending tasks and {completed_before_block} completed tasks already on disk.",
                )

                for task_batch in chunked(pending_tasks, max(1, args.batch_size)):
                    batch_started_at = time.perf_counter()
                    batch_rows, batch_runs, batch_metrics = run_batch_traces(
                        model=model,
                        tokenizer=tokenizer,
                        model_spec=model_spec,
                        tasks=task_batch,
                        actual_device=actual_device,
                        temperature=temperature,
                        seed=seed,
                        max_steps=args.max_steps,
                        max_new_tokens=args.max_new_tokens,
                        hidden_dir=hidden_dir,
                        prompt_mode=args.prompt_mode,
                        system_prompt_mode=args.system_prompt_mode,
                        is_baseline=False,
                    )
                    append_records(paths["steps"], batch_rows)
                    append_records(paths["runs"], batch_runs)
                    append_records(paths["batch_metrics"], batch_metrics)
                    all_rows.extend(batch_rows)
                    all_runs.extend(batch_runs)
                    completed_run_ids.update(run["run_id"] for run in batch_runs)

                    write_metadata_snapshot(
                        paths=paths,
                        model_spec=model_spec,
                        backend=backend,
                        actual_device=actual_device,
                        quantization=args.quantization,
                        device_map=args.device_map,
                        attn_implementation=args.attn_implementation,
                        temperatures=args.temperatures,
                        seeds=args.seeds,
                        max_steps=args.max_steps,
                        max_new_tokens=args.max_new_tokens,
                        max_tasks=args.max_tasks,
                        task_source=args.task_source,
                        dataset_split=args.dataset_split,
                        dataset_shuffle_seed=args.dataset_shuffle_seed,
                        batch_size=args.batch_size,
                        prompt_mode=args.prompt_mode,
                        system_prompt_mode=args.system_prompt_mode,
                        resume_enabled=args.resume,
                        completed_run_ids=completed_run_ids,
                        total_requested_runs=total_requested_runs,
                        reconciliation_report=reconciliation_report,
                        tasks=tasks,
                    )

                    completed_in_block = count_completed_tasks_for_block(all_runs, temperature, seed)
                    batch_elapsed_seconds = time.perf_counter() - batch_started_at
                    append_text_log(
                        checkpoint_log_path,
                        f"Completed batch temp={temperature:.2f} seed={seed} batch_runs={len(batch_runs)} completed_tasks_in_block={completed_in_block} completed_runs_total={len(completed_run_ids)} elapsed={batch_elapsed_seconds:.1f}s",
                    )

                    checkpoint_bucket = completed_in_block // max(args.checkpoint_every_tasks, 1)
                    if checkpoint_bucket > last_checkpoint_bucket:
                        write_current_summaries(
                            all_rows=all_rows,
                            all_runs=all_runs,
                            paths=paths,
                            model_spec=model_spec,
                            backend=backend,
                            actual_device=actual_device,
                        )
                        write_metadata_snapshot(
                            paths=paths,
                            model_spec=model_spec,
                            backend=backend,
                            actual_device=actual_device,
                            quantization=args.quantization,
                            device_map=args.device_map,
                            attn_implementation=args.attn_implementation,
                            temperatures=args.temperatures,
                            seeds=args.seeds,
                            max_steps=args.max_steps,
                            max_new_tokens=args.max_new_tokens,
                            max_tasks=args.max_tasks,
                            task_source=args.task_source,
                            dataset_split=args.dataset_split,
                            dataset_shuffle_seed=args.dataset_shuffle_seed,
                            batch_size=args.batch_size,
                            prompt_mode=args.prompt_mode,
                            system_prompt_mode=args.system_prompt_mode,
                            resume_enabled=args.resume,
                            completed_run_ids=completed_run_ids,
                            total_requested_runs=total_requested_runs,
                            reconciliation_report=reconciliation_report,
                            tasks=tasks,
                        )
                        checkpoint_label = f"temp{temperature:.2f}_t{completed_in_block:03d}"
                        checkpoint_commit(
                            output_dir=output_dir,
                            checkpoint_log_path=checkpoint_log_path,
                            checkpoint_history_path=checkpoint_history_path,
                            log_path=checkpoint_log_path,
                            checkpoint_label=checkpoint_label,
                            temperature=temperature,
                            seed=seed,
                            completed_tasks_in_block=completed_in_block,
                            completed_runs_total=len(completed_run_ids),
                            total_requested_runs=total_requested_runs,
                        )
                        last_checkpoint_bucket = checkpoint_bucket

    write_current_summaries(
        all_rows=all_rows,
        all_runs=all_runs,
        paths=paths,
        model_spec=model_spec,
        backend=backend,
        actual_device=actual_device,
    )
    write_metadata_snapshot(
        paths=paths,
        model_spec=model_spec,
        backend=backend,
        actual_device=actual_device,
        quantization=args.quantization,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        temperatures=args.temperatures,
        seeds=args.seeds,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        max_tasks=args.max_tasks,
        task_source=args.task_source,
        dataset_split=args.dataset_split,
        dataset_shuffle_seed=args.dataset_shuffle_seed,
        batch_size=args.batch_size,
        prompt_mode=args.prompt_mode,
        system_prompt_mode=args.system_prompt_mode,
        resume_enabled=args.resume,
        completed_run_ids=completed_run_ids,
        total_requested_runs=total_requested_runs,
        reconciliation_report=reconciliation_report,
        tasks=tasks,
    )

    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    append_text_log(
        checkpoint_log_path,
        f"Checkpointed collection finished with completed_runs={len(completed_run_ids)} pending_runs={max(total_requested_runs - len(completed_run_ids), 0)}.",
    )


if __name__ == "__main__":
    main()