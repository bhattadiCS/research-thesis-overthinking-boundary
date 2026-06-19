#!/usr/bin/env python
"""Full-precision (bf16) Qwen2.5 capability-ladder launcher.

Runs ``real_trace_experiments.py`` sequentially across the Qwen2.5 size ladder
in bf16 (``--quantization none``) on GSM8K, reusing the exact canonical protocol
from the original 4-bit Qwen-7B run so that task IDs line up. Because the dataset
shuffle seed and generation seed are held fixed, the bf16 7B run lands on the
*same* GSM8K items as ``real_traces_l4_qwen_7b_4bit`` -- giving a clean paired
4-bit-vs-bf16 comparison that directly tests whether the late boundary was a
quantization artifact, while the 14B/32B rungs test the capability-linkage claim
within one family at fixed precision and benchmark.

Why bf16 + sdpa: ``real_trace_experiments.load_model`` loads ``--quantization
none`` in bfloat16 on a single GPU; the bf16 path has no FlashAttention-2
fallback, so we pin ``--attn-implementation sdpa`` (also what the canonical run
used). 32B in bf16 is ~64 GB and fits on a 96 GB card with room for KV cache.

Examples
--------
    # Full ladder (7B -> 14B -> 32B), 500 tasks x 3 temps:
    python research/run_qwen_bf16_ladder.py

    # Just re-run the 7B witness in bf16 (the paired-comparison job):
    python research/run_qwen_bf16_ladder.py --models qwen2p5_7b

    # Print the commands without running them:
    python research/run_qwen_bf16_ladder.py --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENT_SCRIPT = HERE / "real_trace_experiments.py"

# Canonical protocol from research/outputs/real_traces_l4_qwen_7b_4bit/metadata.json.
# Held fixed so bf16 runs align task-for-task with the existing 4-bit run.
SHARED_PROTOCOL = [
    "--device", "cuda",
    "--quantization", "none",          # -> bf16 single-GPU load
    "--attn-implementation", "sdpa",   # bf16 path has no FA2 fallback; sdpa matches canonical
    "--temperatures", "0.1", "0.6", "1.0",
    "--seeds", "7",
    "--max-steps", "10",
    "--max-new-tokens", "256",
    "--task-source", "gsm8k",
    "--dataset-split", "train",
    "--dataset-shuffle-seed", "17",
    "--prompt-mode", "minimal_json",
    "--system-prompt-mode", "default",
]

# Ascending size order. Batch sizes are conservative defaults for a 96 GB card
# in bf16 (weights: 3B~6GB, 7B~15GB, 14B~28GB, 32B~64GB); raise if VRAM allows.
BATCH_SIZES = {
    "qwen2p5_0p5b": 64,
    "qwen2p5_3b": 48,
    "qwen2p5_7b": 32,
    "qwen2p5_14b": 24,
    "qwen2p5_32b": 8,
}

DEFAULT_LADDER = ["qwen2p5_7b", "qwen2p5_14b", "qwen2p5_32b"]


def build_command(
    model: str,
    output_dir: Path,
    max_tasks: int,
    batch_size: int,
    resume: bool,
) -> list[str]:
    cmd = [sys.executable, str(EXPERIMENT_SCRIPT), "--model", model]
    cmd += SHARED_PROTOCOL
    cmd += [
        "--max-tasks", str(max_tasks),
        "--batch-size", str(batch_size),
        "--output-dir", str(output_dir),
    ]
    if not resume:
        cmd += ["--no-resume"]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_LADDER,
        help=f"Ladder rungs to run, in order. Default: {' '.join(DEFAULT_LADDER)}",
    )
    parser.add_argument("--max-tasks", type=int, default=500,
                        help="GSM8K tasks per model (canonical 4-bit run used 300; 500 for more power).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override the per-model batch size for every rung.")
    parser.add_argument("--output-root", default=str(HERE / "outputs" / "real_traces_bf16_ladder"),
                        help="Root dir; each model writes to <root>/<alias>.")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Disable checkpoint resume (default: resume enabled).")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    results: list[tuple[str, str, float]] = []

    for model in args.models:
        if model not in BATCH_SIZES:
            print(f"[warn] No default batch size for '{model}'; using 8. "
                  f"Pass --batch-size to override.", flush=True)
        batch_size = args.batch_size if args.batch_size is not None else BATCH_SIZES.get(model, 8)
        output_dir = output_root / model
        cmd = build_command(model, output_dir, args.max_tasks, batch_size, args.resume)

        print("\n" + "=" * 80, flush=True)
        print(f"[ladder] model={model}  batch_size={batch_size}  max_tasks={args.max_tasks}", flush=True)
        print(f"[ladder] output_dir={output_dir}", flush=True)
        print("[ladder] " + " ".join(cmd), flush=True)
        print("=" * 80, flush=True)

        if args.dry_run:
            results.append((model, "dry-run", 0.0))
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        proc = subprocess.run(cmd)
        elapsed = time.monotonic() - start
        status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        results.append((model, status, elapsed))
        # Continue to the next rung even on failure so an OOM at 32B does not
        # discard the already-collected 7B/14B traces.

    print("\n" + "#" * 80, flush=True)
    print("[ladder] summary", flush=True)
    any_failed = False
    for model, status, elapsed in results:
        print(f"  {model:<16} {status:<22} {elapsed/60:6.1f} min", flush=True)
        any_failed = any_failed or status.startswith("FAILED")
    print("#" * 80, flush=True)
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
