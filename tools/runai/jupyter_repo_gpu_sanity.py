from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

REPO_URL = os.environ.get(
    "REPO_URL",
    "https://github.com/bhattadiCS/research-thesis-overthinking-boundary.git",
)
REPO_NAME = os.environ.get("REPO_NAME", "research-thesis-overthinking-boundary")
WORKDIR = Path(
    os.environ.get(
        "WORKDIR",
        "/workspace" if Path("/workspace").exists() else str(Path.cwd()),
    )
).resolve()


def run(command: list[str], *, cwd: Path | None = None, check: bool = True, capture_output: bool = False):
    printable = " ".join(shlex.quote(str(part)) for part in command)
    print(f"$ {printable}")
    return subprocess.run(
        [str(part) for part in command],
        cwd=str(cwd) if cwd else None,
        text=True,
        check=check,
        capture_output=capture_output,
    )


def find_git_root(start: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(start),
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception:
        return None
    return Path(result.stdout.strip())


def resolve_repo_dir() -> Path:
    existing_root = find_git_root(Path.cwd())
    if existing_root and (existing_root / "tools" / "run_colab_experiment.py").exists():
        return existing_root
    return WORKDIR / REPO_NAME


def clone_or_update_repo() -> Path:
    repo_dir = resolve_repo_dir()

    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)])
    elif not (repo_dir / ".git").exists():
        raise RuntimeError(f"{repo_dir} exists but is not a git repository.")

    remote = run(["git", "remote", "get-url", "origin"], cwd=repo_dir, check=False, capture_output=True)
    remote_url = (remote.stdout or "").strip()
    if remote.returncode == 0 and remote_url and "research-thesis-overthinking-boundary" not in remote_url:
        raise RuntimeError(f"Unexpected git remote for {repo_dir}: {remote_url}")

    run(["git", "fetch", "--all", "--prune"], cwd=repo_dir)

    status = run(["git", "status", "--porcelain"], cwd=repo_dir, check=False, capture_output=True)
    if (status.stdout or "").strip():
        print("Skipping git pull because the worktree has local changes.")
    else:
        run(["git", "pull", "--ff-only"], cwd=repo_dir)

    head = run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir, check=False, capture_output=True)
    if head.returncode == 0:
        print(f"Current HEAD: {head.stdout.strip()}")

    os.chdir(repo_dir)
    return repo_dir


def gpu_sanity_check() -> None:
    print("\n=== GPU visibility via nvidia-smi ===")
    nvidia = run(["nvidia-smi"], check=False)
    if nvidia.returncode != 0:
        print("nvidia-smi failed or no NVIDIA GPU is available to this notebook session.")

    print("\n=== PyTorch CUDA sanity check ===")
    try:
        import torch
    except Exception as exc:
        print(f"Torch import failed: {exc}")
        return

    print(f"torch_version: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return

    print(f"device_count: {torch.cuda.device_count()}")
    print(f"device_name: {torch.cuda.get_device_name(0)}")
    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"device_0_total_memory_gb: {total_memory_gb:.2f}")

    tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    print(f"cuda_tensor_device: {tensor.device}")
    print(f"cuda_tensor_sum: {tensor.sum().item()}")


repo_dir = clone_or_update_repo()
print(f"\nRepository ready at: {repo_dir}")
gpu_sanity_check()
