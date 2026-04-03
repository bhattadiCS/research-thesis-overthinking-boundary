from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_command(command: list[str]) -> None:
    print(f"[run] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def git_command(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=check)


def relative_repo_path(path: Path) -> str:
    resolved = path if path.is_absolute() else (REPO_ROOT / path)
    return str(resolved.relative_to(REPO_ROOT))


def read_pending_run_count(metadata_path: Path) -> int | None:
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    value = payload.get("pending_run_count")
    return int(value) if value is not None else None


def read_completed_runs(runs_path: Path) -> int:
    if not runs_path.exists():
        return 0
    runs = pd.read_csv(runs_path)
    return int(runs["run_id"].nunique()) if not runs.empty else 0


def log_contains_finished(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return "Checkpointed collection finished" in text


def wait_for_completion(
    *,
    runs_path: Path,
    metadata_path: Path,
    checkpoint_log: Path,
    expected_runs: int,
    poll_seconds: int,
) -> None:
    last_reported_runs = -1
    while True:
        completed_runs = read_completed_runs(runs_path)
        pending_run_count = read_pending_run_count(metadata_path)
        finished = log_contains_finished(checkpoint_log)
        if completed_runs != last_reported_runs:
            print(
                f"[wait] completed_runs={completed_runs} expected_runs={expected_runs} pending_run_count={pending_run_count} finished={finished}",
                flush=True,
            )
            last_reported_runs = completed_runs
        if completed_runs >= expected_runs and pending_run_count == 0 and finished:
            return
        time.sleep(poll_seconds)


def stage_paths(paths: list[Path]) -> None:
    repo_paths = [relative_repo_path(path) for path in paths]
    git_command(["add", "--", *repo_paths])


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for the Mistral run to finish, then finalize analysis artifacts and commit them.")
    parser.add_argument("--input-dir", default=str(REPO_ROOT / "research" / "outputs" / "real_traces_l4_mistral_7b"))
    parser.add_argument("--checkpoint-log", default=str(REPO_ROOT / "research" / "outputs" / "real_traces_l4_mistral_7b_checkpointed.log"))
    parser.add_argument("--checkpoint-history", default=str(REPO_ROOT / "research" / "outputs" / "real_traces_l4_mistral_7b_checkpoint_history.jsonl"))
    parser.add_argument("--expected-runs", type=int, default=900)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--answers-output", default=str(REPO_ROOT / "research" / "ANSWERS_MISTRAL_L4.md"))
    parser.add_argument("--open-questions-output", default=str(REPO_ROOT / "research" / "open_questions_mistral_l4.md"))
    parser.add_argument("--research-report-output", default=str(REPO_ROOT / "research" / "FINAL_MISTRAL_L4_RESULTS.md"))
    parser.add_argument("--root-report-output", default=str(REPO_ROOT / "MISTRAL_L4_OVERTHINKING_RESULTS.md"))
    parser.add_argument("--report-title", default="Mistral 7B L4 Overthinking Results")
    parser.add_argument(
        "--cross-run-dirs",
        nargs="+",
        default=[
            str(REPO_ROOT / "research" / "outputs" / "real_traces_l4_deepseek_1p5b"),
            str(REPO_ROOT / "research" / "outputs" / "real_traces_l4_qwen_0p5b"),
            str(REPO_ROOT / "research" / "outputs" / "real_traces_l4_qwen_7b_4bit"),
        ],
    )
    parser.add_argument("--cross-output-dir", default=str(REPO_ROOT / "research" / "outputs" / "cross_family"))
    parser.add_argument("--cross-report-output", default=str(REPO_ROOT / "research" / "CROSS_FAMILY_REPORT.md"))
    parser.add_argument("--cross-open-questions-output", default=str(REPO_ROOT / "research" / "CROSS_FAMILY_OPEN_QUESTIONS.md"))
    parser.add_argument("--commit-message", default="analysis: finalize mistral l4 cycle")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    checkpoint_log = Path(args.checkpoint_log)
    checkpoint_history = Path(args.checkpoint_history)
    runs_path = input_dir / "trace_runs.csv"
    metadata_path = input_dir / "metadata.json"

    wait_for_completion(
        runs_path=runs_path,
        metadata_path=metadata_path,
        checkpoint_log=checkpoint_log,
        expected_runs=args.expected_runs,
        poll_seconds=args.poll_seconds,
    )

    run_command([PYTHON, str(REPO_ROOT / "research" / "trace_analysis.py"), "--input-dir", str(input_dir)])
    run_command(
        [
            PYTHON,
            str(REPO_ROOT / "research" / "generate_thesis_artifacts.py"),
            "--input-dir",
            str(input_dir),
            "--answers-output",
            args.answers_output,
            "--open-questions-output",
            args.open_questions_output,
            "--research-report-output",
            args.research_report_output,
            "--root-report-output",
            args.root_report_output,
            "--report-title",
            args.report_title,
        ]
    )
    run_command(
        [
            PYTHON,
            str(REPO_ROOT / "research" / "cross_family_analysis.py"),
            "--run-dirs",
            *args.cross_run_dirs,
            str(input_dir),
            "--output-dir",
            args.cross_output_dir,
            "--report-output",
            args.cross_report_output,
            "--open-questions-output",
            args.cross_open_questions_output,
        ]
    )

    stage_paths(
        [
            input_dir,
            checkpoint_log,
            checkpoint_history,
            REPO_ROOT / "research" / "generate_thesis_artifacts.py",
            REPO_ROOT / "tools" / "watch_and_finalize_mistral_cycle.py",
            Path(args.answers_output),
            Path(args.open_questions_output),
            Path(args.research_report_output),
            Path(args.root_report_output),
            Path(args.cross_output_dir),
            Path(args.cross_report_output),
            Path(args.cross_open_questions_output),
        ]
    )
    git_command(["commit", "-m", args.commit_message])
    commit_hash = git_command(["rev-parse", "HEAD"]).stdout.strip()
    print(f"[done] final analysis commit={commit_hash}", flush=True)
    if args.push:
        git_command(["push", "origin", "main"])
        print("[done] final analysis push succeeded", flush=True)


if __name__ == "__main__":
    main()