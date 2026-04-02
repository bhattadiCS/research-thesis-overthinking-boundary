from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "research" / "outputs" / "benchmark_mistral7b_l4"


def config_name(quantization: str, attn_implementation: str, batch_size: int) -> str:
    quant_label = "fp" if quantization == "none" else quantization
    return f"q{quant_label}_attn_{attn_implementation}_bs{batch_size}"


def query_gpu_snapshot() -> dict[str, Any] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) != 6:
        return None
    return {
        "name": parts[0],
        "memory_total_mib": float(parts[1]),
        "memory_free_mib": float(parts[2]),
        "memory_used_mib": float(parts[3]),
        "utilization_gpu": float(parts[4]),
        "utilization_memory": float(parts[5]),
    }


def monitor_gpu(samples: list[dict[str, Any]], stop_event: threading.Event, interval_seconds: float) -> None:
    while not stop_event.is_set():
        snapshot = query_gpu_snapshot()
        if snapshot is not None:
            snapshot["timestamp"] = time.time()
            samples.append(snapshot)
        stop_event.wait(interval_seconds)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def tail_text(path: Path, max_lines: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def summarize_run(
    *,
    config_label: str,
    command: list[str],
    output_dir: Path,
    log_path: Path,
    elapsed_seconds: float,
    return_code: int,
    gpu_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    batch_metrics_path = output_dir / "batch_metrics.csv"
    trace_steps_path = output_dir / "trace_steps.csv"

    batch_metrics = pd.read_csv(batch_metrics_path) if batch_metrics_path.exists() else pd.DataFrame()
    trace_steps = pd.read_csv(trace_steps_path) if trace_steps_path.exists() else pd.DataFrame()
    generate_batches = batch_metrics[batch_metrics["phase"] == "generate"].copy() if not batch_metrics.empty else pd.DataFrame()
    hidden_write_batches = batch_metrics[batch_metrics["phase"] == "hidden_state_write"].copy() if not batch_metrics.empty else pd.DataFrame()

    generation_time = float(generate_batches["generation_seconds"].sum()) if not generate_batches.empty else 0.0
    forward_time = float(generate_batches["forward_seconds"].sum()) if not generate_batches.empty else 0.0
    tokenize_time = float(generate_batches["tokenize_seconds"].sum()) if not generate_batches.empty else 0.0
    postprocess_time = float(generate_batches["postprocess_seconds"].sum()) if not generate_batches.empty else 0.0
    hidden_write_time = float(hidden_write_batches["hidden_state_write_seconds"].sum()) if not hidden_write_batches.empty else 0.0
    generate_wall = float(generate_batches["wall_clock_seconds"].sum()) if not generate_batches.empty else 0.0
    residual_cpu_time = max(generate_wall - generation_time - forward_time - tokenize_time - postprocess_time, 0.0)
    cpu_preprocessing_time = tokenize_time + postprocess_time + residual_cpu_time

    bottleneck_times = {
        "model_compute": generation_time,
        "hidden_state_extraction": forward_time,
        "cpu_preprocessing": cpu_preprocessing_time,
        "disk_writes": hidden_write_time,
    }
    dominant_bottleneck = max(bottleneck_times, key=bottleneck_times.get) if any(bottleneck_times.values()) else "unknown"

    gpu_frame = pd.DataFrame(gpu_samples)
    mean_gpu_utilization = safe_float(gpu_frame["utilization_gpu"].mean()) if not gpu_frame.empty else float("nan")
    max_gpu_utilization = safe_float(gpu_frame["utilization_gpu"].max()) if not gpu_frame.empty else float("nan")
    mean_gpu_memory_utilization = safe_float(gpu_frame["utilization_memory"].mean()) if not gpu_frame.empty else float("nan")
    mean_gpu_memory_used_gb = safe_float(gpu_frame["memory_used_mib"].mean() / 1024.0) if not gpu_frame.empty else float("nan")
    max_gpu_memory_used_gb = safe_float(gpu_frame["memory_used_mib"].max() / 1024.0) if not gpu_frame.empty else float("nan")

    oom_retry_count = int(batch_metrics["oom_retry_count"].sum()) if not batch_metrics.empty else 0
    split_events = int((batch_metrics["split_count"] > 1).sum()) if not batch_metrics.empty else 0
    stable_memory = int(return_code == 0 and oom_retry_count == 0 and split_events == 0)

    return {
        "config_name": config_label,
        "status": "ok" if return_code == 0 else "failed",
        "stable_memory": stable_memory,
        "return_code": return_code,
        "output_dir": str(output_dir),
        "command": " ".join(command),
        "elapsed_seconds": elapsed_seconds,
        "n_batches": int(len(generate_batches)),
        "n_runs": int(trace_steps["run_id"].nunique()) if not trace_steps.empty else 0,
        "n_step_rows": int(len(trace_steps)),
        "total_generated_tokens": int(generate_batches["generated_tokens"].sum()) if not generate_batches.empty else 0,
        "mean_examples_per_second": safe_float(generate_batches["examples_per_second"].mean()) if not generate_batches.empty else float("nan"),
        "mean_tokens_per_second": safe_float(generate_batches["tokens_per_second"].mean()) if not generate_batches.empty else float("nan"),
        "mean_wall_clock_per_batch": safe_float(generate_batches["wall_clock_seconds"].mean()) if not generate_batches.empty else float("nan"),
        "max_wall_clock_per_batch": safe_float(generate_batches["wall_clock_seconds"].max()) if not generate_batches.empty else float("nan"),
        "peak_gpu_allocated_gb": safe_float(batch_metrics["gpu_max_memory_allocated_gb"].max()) if not batch_metrics.empty else float("nan"),
        "peak_gpu_reserved_gb": safe_float(batch_metrics["gpu_max_memory_reserved_gb"].max()) if not batch_metrics.empty else float("nan"),
        "mean_gpu_utilization": mean_gpu_utilization,
        "max_gpu_utilization": max_gpu_utilization,
        "mean_gpu_memory_utilization": mean_gpu_memory_utilization,
        "mean_gpu_memory_used_gb": mean_gpu_memory_used_gb,
        "max_gpu_memory_used_gb": max_gpu_memory_used_gb,
        "generation_seconds": generation_time,
        "forward_seconds": forward_time,
        "cpu_preprocessing_seconds": cpu_preprocessing_time,
        "hidden_state_write_seconds": hidden_write_time,
        "dominant_bottleneck": dominant_bottleneck,
        "oom_retry_count": oom_retry_count,
        "split_events": split_events,
        "log_path": str(log_path),
        "failure_tail": tail_text(log_path) if return_code != 0 else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark L4 throughput and memory stability across batch, quantization, and attention settings.")
    parser.add_argument("--model", default="mistral_7b_instruct_v0p3")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--quantizations", nargs="+", default=["4bit", "8bit", "none"])
    parser.add_argument("--attn-implementations", nargs="+", default=["sdpa", "auto", "flash_attention_2"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 6, 8])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--task-source", default="gsm8k")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-shuffle-seed", type=int, default=17)
    parser.add_argument("--max-tasks", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--prompt-mode", default="minimal_json")
    parser.add_argument("--system-prompt-mode", default="default")
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    hardware_snapshot = query_gpu_snapshot()
    if hardware_snapshot is not None:
        with (output_root / "hardware_snapshot.json").open("w", encoding="utf-8") as handle:
            json.dump(hardware_snapshot, handle, indent=2)

    summary_rows: list[dict[str, Any]] = []
    for quantization in args.quantizations:
        for attn_implementation in args.attn_implementations:
            for batch_size in args.batch_sizes:
                label = config_name(quantization, attn_implementation, batch_size)
                output_dir = output_root / label
                log_path = output_root / f"{label}.log"
                if args.fresh and output_dir.exists():
                    shutil.rmtree(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                command = [
                    PYTHON,
                    str(REPO_ROOT / "research" / "real_trace_experiments.py"),
                    "--model",
                    args.model,
                    "--device",
                    "cuda",
                    "--quantization",
                    quantization,
                    "--attn-implementation",
                    attn_implementation,
                    "--max-tasks",
                    str(args.max_tasks),
                    "--max-steps",
                    str(args.max_steps),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--task-source",
                    args.task_source,
                    "--dataset-split",
                    args.dataset_split,
                    "--dataset-shuffle-seed",
                    str(args.dataset_shuffle_seed),
                    "--batch-size",
                    str(batch_size),
                    "--prompt-mode",
                    args.prompt_mode,
                    "--system-prompt-mode",
                    args.system_prompt_mode,
                    "--output-dir",
                    str(output_dir),
                    "--temperatures",
                    *[str(value) for value in args.temperatures],
                    "--seeds",
                    *[str(value) for value in args.seeds],
                ]
                if quantization in {"4bit", "8bit"} and args.device_map:
                    command.extend(["--device-map", args.device_map])

                gpu_samples: list[dict[str, Any]] = []
                stop_event = threading.Event()
                monitor = threading.Thread(target=monitor_gpu, args=(gpu_samples, stop_event, args.sample_interval), daemon=True)

                started_at = time.perf_counter()
                env = {**os.environ, "PYTHONUNBUFFERED": "1"}
                with log_path.open("w", encoding="utf-8") as log_handle:
                    process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
                    monitor.start()
                    return_code = process.wait()
                stop_event.set()
                monitor.join(timeout=5.0)
                elapsed_seconds = time.perf_counter() - started_at

                summary_rows.append(
                    summarize_run(
                        config_label=label,
                        command=command,
                        output_dir=output_dir,
                        log_path=log_path,
                        elapsed_seconds=elapsed_seconds,
                        return_code=return_code,
                        gpu_samples=gpu_samples,
                    )
                )
                pd.DataFrame(summary_rows).to_csv(output_root / "benchmark_summary.csv", index=False)

    summary_frame = pd.DataFrame(summary_rows)
    stable_frame = summary_frame[(summary_frame["status"] == "ok") & (summary_frame["stable_memory"] == 1)].copy()
    selection: dict[str, Any] = {
        "selected_config": None,
        "selection_metric": "mean_examples_per_second",
        "selection_reason": "No stable configuration completed.",
    }
    if not stable_frame.empty:
        best_row = stable_frame.sort_values(
            ["mean_examples_per_second", "mean_tokens_per_second", "peak_gpu_reserved_gb"],
            ascending=[False, False, True],
        ).iloc[0]
        selection = {
            "selected_config": str(best_row["config_name"]),
            "selection_metric": "mean_examples_per_second",
            "selection_reason": "Highest stable throughput among configurations without OOM retries or emergency batch splitting.",
            "summary": best_row.to_dict(),
        }
    with (output_root / "benchmark_selection.json").open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2, default=str)

    print(summary_frame.to_string(index=False))
    print(json.dumps(selection, indent=2, default=str))


if __name__ == "__main__":
    main()