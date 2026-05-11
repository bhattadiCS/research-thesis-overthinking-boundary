from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def print_output(label: str, value: str) -> None:
    print(f"{label}: {value}")


def main() -> None:
    print("=== Repo ===")
    print_output("root", str(REPO_ROOT))

    git_head = run_capture(["git", "rev-parse", "--short", "HEAD"])
    if git_head.returncode == 0:
        print_output("git_head", git_head.stdout.strip())

    git_status = run_capture(["git", "status", "--short"])
    if git_status.returncode == 0:
        dirty = "yes" if git_status.stdout.strip() else "no"
        print_output("git_dirty", dirty)

    print("\n=== GPU ===")
    nvidia_smi = run_capture(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    if nvidia_smi.returncode == 0 and nvidia_smi.stdout.strip():
        for index, line in enumerate(nvidia_smi.stdout.strip().splitlines(), start=1):
            print_output(f"gpu_{index}", line)
    else:
        print("nvidia-smi unavailable or no GPU detected")

    print("\n=== Python ===")
    print_output("python_executable", sys.executable)
    print_output("python_version", sys.version.split()[0])

    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        print_output("torch_import", f"failed: {exc}")
        return

    print_output("torch_version", torch.__version__)
    print_output("cuda_available", str(torch.cuda.is_available()).lower())
    if torch.cuda.is_available():
        print_output("cuda_device", torch.cuda.get_device_name(0))
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print_output("cuda_memory_gb", f"{total_memory_gb:.2f}")


if __name__ == "__main__":
    main()
