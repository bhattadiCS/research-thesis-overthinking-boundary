#!/usr/bin/env python3
"""Re-verification & Replication Orchestrator Suite

This script runs the entire capability-ladder re-verification pipeline on the GPU
sequentially. To ensure robustness, prevent data loss from preemption or crashes, 
and maintain a clean audit trail, this orchestrator:
1. Sequentially runs trace collection for each model in bf16 (full precision)
2. Automatically runs trace analysis & report generation for that model
3. Performs a Git commit and push of the model's output directory immediately
4. Aggregates all completed runs into a final cross-family analysis and pushes it.

Usage:
------
    python run_reverification_suite.py --max-tasks 500
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Setup paths
HERE = Path(__file__).resolve().parent
EXPERIMENT_SCRIPT = HERE / "research" / "real_trace_experiments.py"
ANALYZE_SCRIPT = HERE / "research" / "trace_analysis.py"
REPORT_SCRIPT = HERE / "research" / "generate_thesis_artifacts.py"
AGGREGATE_SCRIPT = HERE / "research" / "cross_family_analysis.py"

DEFAULT_OUTPUT_ROOT = HERE / "research" / "outputs" / "real_traces_bf16_ladder"
DEFAULT_LADDER = ["qwen2p5_7b", "qwen2p5_14b", "qwen2p5_32b"]

# Map models to their safe default batch sizes on a 96 GB card (bf16 path)
BATCH_SIZES = {
    "qwen2p5_0p5b": 64,
    "qwen2p5_3b": 48,
    "qwen2p5_7b": 32,
    "qwen2p5_14b": 24,
    "qwen2p5_32b": 8,
}


def run_command(cmd: list[str], description: str) -> bool:
    """Run a shell command, printing output in real-time."""
    print(f"\n>>> Running: {description}")
    print(f"Command: {' '.join(cmd)}\n", flush=True)
    
    # Run process and let output stream to stdout/stderr
    proc = subprocess.run(cmd)
    
    if proc.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {proc.returncode}: {description}\n", flush=True)
        return False
    
    print(f"[SUCCESS] Finished: {description}\n", flush=True)
    return True


def get_current_branch() -> str:
    """Retrieve the current active git branch name."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return proc.stdout.strip()
    except subprocess.SubprocessError:
        return "main"  # Fallback


def git_commit_and_push(target_dir: Path, commit_msg: str) -> bool:
    """Add a directory, commit, and push to the remote repository."""
    print(f"\n>>> Backing up outputs: {commit_msg}")
    
    # 1. Add target directory
    add_proc = subprocess.run(["git", "add", str(target_dir)])
    if add_proc.returncode != 0:
        print("[WARN] Git add failed. Skipping backup.")
        return False
        
    # Check if there are changes to commit
    status_proc = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if not status_proc.stdout.strip():
        print("[INFO] No new changes to commit.")
        return True

    # 2. Commit
    commit_proc = subprocess.run(["git", "commit", "-m", commit_msg])
    if commit_proc.returncode != 0:
        print("[WARN] Git commit failed. Skipping backup.")
        return False
        
    # 3. Push to current branch
    branch = get_current_branch()
    print(f"Pushing changes to origin/{branch}...")
    push_proc = subprocess.run(["git", "push", "origin", branch])
    if push_proc.returncode != 0:
        print("[WARN] Git push failed. Changes are committed locally but not pushed.")
        return False
        
    print("[SUCCESS] Output successfully pushed to git remote.\n", flush=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_LADDER,
        help=f"Ladder rungs to run. Default: {' '.join(DEFAULT_LADDER)}",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=500,
        help="Number of tasks per model (canonical matrix uses 500).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the batch size for all models in the sweep.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"Root output directory. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint resume (default: resume enabled).",
    )
    parser.add_argument(
        "--skip-git",
        action="store_true",
        help="Disable automated git commit and push backup steps.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("STARTING RE-VERIFICATION SUITE ORCHESTRATION")
    print(f"Target Models: {', '.join(args.models)}")
    print(f"Tasks Per Model: {args.max_tasks}")
    print(f"Output Root: {output_root}")
    print(f"Git Backup Enabled: {not args.skip_git}")
    print("=" * 80, flush=True)

    completed_dirs = []
    failed_models = []

    for model in args.models:
        model_start = time.monotonic()
        output_dir = output_root / model
        output_dir.mkdir(parents=True, exist_ok=True)
        
        batch_size = args.batch_size if args.batch_size is not None else BATCH_SIZES.get(model, 8)
        
        print("\n" + "#" * 80)
        print(f" PROCESSING rung: {model} (Batch Size: {batch_size})")
        print("#" * 80, flush=True)
        
        # 1. Run Trace Collection
        # Canonical bf16 protocol configuration
        collect_cmd = [
            sys.executable, str(EXPERIMENT_SCRIPT),
            "--model", model,
            "--device", "cuda",
            "--quantization", "none",
            "--attn-implementation", "sdpa",
            "--temperatures", "0.1", "0.6", "1.0",
            "--seeds", "7",
            "--max-steps", "10",
            "--max-new-tokens", "256",
            "--task-source", "gsm8k",
            "--dataset-split", "train",
            "--dataset-shuffle-seed", "17",
            "--prompt-mode", "minimal_json",
            "--system-prompt-mode", "default",
            "--max-tasks", str(args.max_tasks),
            "--batch-size", str(batch_size),
            "--output-dir", str(output_dir),
        ]
        if args.no_resume:
            collect_cmd.append("--no-resume")
            
        success = run_command(collect_cmd, f"Inference Trace Collection: {model}")
        if not success:
            print(f"[FAIL] Inference collection failed for {model}. Moving to next rung.")
            failed_models.append(model)
            continue

        # 2. Run Trace Analysis
        analyze_cmd = [
            sys.executable, str(ANALYZE_SCRIPT),
            "--input-dir", str(output_dir)
        ]
        success = run_command(analyze_cmd, f"Trace Analysis: {model}")
        if not success:
            print(f"[FAIL] Trace analysis failed for {model}. Moving to next rung.")
            failed_models.append(model)
            continue

        # 3. Generate Report Artifacts
        report_cmd = [
            sys.executable, str(REPORT_SCRIPT),
            "--input-dir", str(output_dir),
            "--answers-output", str(output_dir / "answers.md"),
            "--open-questions-output", str(output_dir / "open_questions.md"),
            "--research-report-output", str(output_dir / "final_results.md"),
            "--root-report-output", str(output_dir / "summary.md")
        ]
        success = run_command(report_cmd, f"Report Generation: {model}")
        if not success:
            print(f"[FAIL] Report generation failed for {model}. Moving to next rung.")
            failed_models.append(model)
            continue
            
        completed_dirs.append(output_dir)
        elapsed_min = (time.monotonic() - model_start) / 60
        print(f"[SUCCESS] Fully completed capability rung: {model} in {elapsed_min:.1f} mins.")

        # 4. Periodic Backup
        if not args.skip_git:
            git_commit_and_push(
                output_dir, 
                f"Re-verification: Completed {model} collection and analysis ({elapsed_min:.1f}m)"
            )

    # 5. Cross-Family Aggregation
    if completed_dirs:
        print("\n" + "=" * 80)
        print("RUNNING FINAL CROSS-FAMILY AGGREGATION")
        print("=" * 80, flush=True)
        
        run_dirs_args = [str(d) for d in completed_dirs]
        aggregate_dir = output_root / "cross_family"
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        
        agg_cmd = [
            sys.executable, str(AGGREGATE_SCRIPT),
            "--run-dirs"
        ] + run_dirs_args + [
            "--output-dir", str(aggregate_dir),
            "--report-output", str(aggregate_dir / "report.md"),
            "--open-questions-output", str(aggregate_dir / "open_questions.md")
        ]
        
        success = run_command(agg_cmd, "Cross-Family Report Aggregation")
        if success and not args.skip_git:
            git_commit_and_push(
                aggregate_dir, 
                "Re-verification: Completed final cross-family aggregation report"
            )
    else:
        print("\n[WARN] No runs completed successfully; skipping cross-family aggregation.")

    # Print summary
    print("\n" + "#" * 80)
    print(" RE-VERIFICATION SUITE SUMMARY")
    print("#" * 80)
    print(f"Successful Rungs: {', '.join([d.name for d in completed_dirs]) or 'None'}")
    print(f"Failed Rungs:     {', '.join(failed_models) or 'None'}")
    print("#" * 80, flush=True)

    return 1 if failed_models else 0


if __name__ == "__main__":
    raise SystemExit(main())
