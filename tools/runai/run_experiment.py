from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_CHECK = REPO_ROOT / "tools" / "runai" / "check_env.py"
COLAB_RUNNER = REPO_ROOT / "tools" / "run_colab_experiment.py"


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"$ {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd=REPO_ROOT, check=check, text=True)


def ensure_clean_before_pull() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if status.returncode != 0:
        raise RuntimeError("Unable to inspect git status before pull.")
    if status.stdout.strip():
        raise RuntimeError("Refusing --pull because the worktree has local changes.")


def main() -> None:
    parser = argparse.ArgumentParser(description="RUN:AI wrapper around the repo's Colab-safe experiment runner.")
    parser.add_argument("--model", default="deepseek_r1_distill_1p5b")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--pull", action="store_true", help="Run git pull --ff-only before launching the experiment.")
    parser.add_argument(
        "--run-simulator",
        action="store_true",
        help="Include the synthetic simulator phase before the smoke/full run.",
    )
    args, extra_args = parser.parse_known_args()

    if args.pull:
        ensure_clean_before_pull()
        run(["git", "pull", "--ff-only"])

    run([sys.executable, str(ENV_CHECK)], check=False)

    command = [sys.executable, str(COLAB_RUNNER), "--model", args.model]
    if args.mode == "smoke":
        command.append("--smoke-only")
    if not args.run_simulator:
        command.append("--skip-simulator")
    command.extend(extra_args)
    run(command)


if __name__ == "__main__":
    main()
