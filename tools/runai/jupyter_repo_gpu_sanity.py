from __future__ import annotations

from datetime import datetime
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
DEFAULT_TORCH_PACKAGES = "torch>=2.10.0 torchvision torchaudio"
RECOMMENDED_BLACKWELL_TORCH_INDEX_URL = DEFAULT_TORCH_INDEX_URL
RECOMMENDED_BLACKWELL_TORCH_PACKAGES = DEFAULT_TORCH_PACKAGES
LEGACY_BLACKWELL_TORCH_INDEX_URLS = {"https://download.pytorch.org/whl/cu124"}
VALID_GPU_FAILURE_MODES = {"stop", "skip-experiment", "cpu"}
RUNTIME_OPTIONAL_NUMPY_DEPENDENCIES = {
    "numexpr": "numexpr>=2.14.1",
    "bottleneck": "bottleneck>=1.6.0",
}

REPO_URL = os.environ.get(
    "REPO_URL",
    "https://github.com/bhattadiCS/research-thesis-overthinking-boundary.git",
)
REPO_NAME = os.environ.get("REPO_NAME", "research-thesis-overthinking-boundary")
WORKDIR = Path(
    os.environ.get(
        "WORKDIR",
        "/workspaces" if Path("/workspaces").exists() else str(Path.cwd()),
    )
).resolve()
MODEL = os.environ.get("MODEL", "gemma_4_e4b_it")
EXPERIMENT_MODE = os.environ.get("EXPERIMENT_MODE", "full")
START_EXPERIMENT = os.environ.get("START_EXPERIMENT", "1") == "1"
RUN_SIMULATOR = os.environ.get("RUN_SIMULATOR", "0") == "1"
SKIP_INSTALL = os.environ.get("SKIP_INSTALL", "0") == "1"
AUTO_INSTALL_TORCH = os.environ.get("AUTO_INSTALL_TORCH", "1") == "1"
TORCH_INDEX_URL = os.environ.get("TORCH_INDEX_URL", DEFAULT_TORCH_INDEX_URL)
TORCH_PACKAGES = shlex.split(os.environ.get("TORCH_PACKAGES", DEFAULT_TORCH_PACKAGES))
GPU_FAILURE_MODE = os.environ.get("GPU_FAILURE_MODE", "stop").strip().lower()

# Default EXPERIMENT_ARGS tuned for Gemma-4-E4B-It on a ~10 GB Blackwell GPU.
# 4-bit quantization + batch size 1 keeps peak VRAM under 9.78 GB.
_DEFAULT_EXPERIMENT_ARGS = (
    "--quantization 4bit "
    "--smoke-batch-size 1 "
    "--full-batch-size 1 "
    "--io-threads 4 "
    "--attn-implementation sdpa"
)
EXPERIMENT_ARGS = shlex.split(os.environ.get("EXPERIMENT_ARGS", _DEFAULT_EXPERIMENT_ARGS))


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture_output: bool = False,
    env: dict[str, str] | None = None,
):
    working_dir = cwd or Path.cwd()
    printable = " ".join(shlex.quote(str(part)) for part in command)
    log(f"Running in {working_dir}: {printable}")
    result = subprocess.run(
        [str(part) for part in command],
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=capture_output,
        check=False,
        env={**os.environ, **env} if env else None,
    )
    if capture_output and result.stdout.strip():
        log(f"stdout:\n{result.stdout.strip()}")
    if capture_output and result.stderr.strip():
        log(f"stderr:\n{result.stderr.strip()}")
    log(f"Command exit code: {result.returncode}")
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            [str(part) for part in command],
            output=result.stdout,
            stderr=result.stderr,
        )
    return result


def pip_install(args: list[str]) -> None:
    run([sys.executable, "-m", "pip", "install", *args])


def version_key(raw_version: str | None) -> tuple[int, ...]:
    if not raw_version:
        return (0,)
    numbers = [int(value) for value in re.findall(r"\d+", raw_version)]
    return tuple(numbers) if numbers else (0,)


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
        log(f"Found existing repository checkout at {existing_root}")
        return existing_root
    repo_dir = WORKDIR / REPO_NAME
    log(f"Using repository path {repo_dir}")
    return repo_dir


def git_output(repo_dir: Path, *args: str) -> str:
    result = run(["git", *args], cwd=repo_dir, check=False, capture_output=True)
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip()


def git_status_lines(repo_dir: Path) -> list[str]:
    return [line for line in git_output(repo_dir, "status", "--porcelain").splitlines() if line.strip()]


def is_ignorable_status_line(line: str) -> bool:
    if not line.startswith("?? "):
        return False
    relative_path = line[3:].strip()
    return (
        relative_path == "research/outputs"
        or relative_path.startswith("research/outputs/")
        or relative_path == "run_logs"
        or relative_path.startswith("run_logs/")
    )


def non_ignorable_status_lines(repo_dir: Path) -> list[str]:
    return [line for line in git_status_lines(repo_dir) if not is_ignorable_status_line(line)]


def print_repo_state(repo_dir: Path) -> None:
    remote_url = git_output(repo_dir, "remote", "get-url", "origin")
    branch_name = git_output(repo_dir, "branch", "--show-current")
    head_commit = git_output(repo_dir, "rev-parse", "--short", "HEAD")

    log(f"Repository root: {repo_dir}")
    if remote_url:
        log(f"origin: {remote_url}")
    if branch_name:
        log(f"branch: {branch_name}")
    if head_commit:
        log(f"HEAD: {head_commit}")

    status_lines = git_status_lines(repo_dir)
    if status_lines:
        log(f"Worktree is dirty with {len(status_lines)} change(s); showing first {min(20, len(status_lines))}.")
        for line in status_lines[:20]:
            print(f"    {line}", flush=True)
    else:
        log("Worktree is clean.")


def clone_or_update_repo() -> Path:
    repo_dir = resolve_repo_dir()

    if not repo_dir.exists():
        log("Repository is missing locally; cloning a fresh checkout.")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)])
    elif not (repo_dir / ".git").exists():
        raise RuntimeError(f"{repo_dir} exists but is not a git repository.")
    else:
        log("Repository already exists locally; validating and updating it.")

    remote = run(["git", "remote", "get-url", "origin"], cwd=repo_dir, check=False, capture_output=True)
    remote_url = (remote.stdout or "").strip()
    if remote.returncode == 0 and remote_url and "research-thesis-overthinking-boundary" not in remote_url:
        raise RuntimeError(f"Unexpected git remote for {repo_dir}: {remote_url}")

    before_head = git_output(repo_dir, "rev-parse", "--short", "HEAD")
    run(["git", "fetch", "--all", "--prune"], cwd=repo_dir)

    dirty_lines = non_ignorable_status_lines(repo_dir)
    if dirty_lines:
        log("Skipping git pull because the worktree has non-generated local changes.")
    else:
        all_status_lines = git_status_lines(repo_dir)
        if all_status_lines:
            log("Ignoring generated research/outputs changes while pulling the latest code.")
        run(["git", "pull", "--ff-only"], cwd=repo_dir)

    head = run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir, check=False, capture_output=True)
    if head.returncode == 0:
        after_head = head.stdout.strip()
        if before_head and before_head != after_head:
            log(f"Repository advanced from {before_head} to {after_head}")
        else:
            log(f"Repository already at latest visible HEAD: {after_head}")

    os.chdir(repo_dir)
    log(f"Changed working directory to {repo_dir}")
    print_repo_state(repo_dir)
    return repo_dir


def query_nvidia_smi_summary() -> list[dict[str, object]]:
    result = run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return []

    records: list[dict[str, object]] = []
    for line in (result.stdout or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            memory_mib = float(parts[1])
        except ValueError:
            continue
        records.append(
            {
                "name": parts[0],
                "memory_mib": int(memory_mib),
                "memory_gb": round(memory_mib / 1024, 2),
                "driver_version": parts[2],
            }
        )
    return records


def probe_torch() -> dict[str, object]:
    probe_code = """
import json
import sys
import warnings

payload = {
    'import_ok': False,
    'cuda_probe_ok': False,
    'python_executable': sys.executable,
    'warnings': [],
}

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    try:
        import torch
        payload.update({
            'import_ok': True,
            'version': torch.__version__,
            'cuda_version': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(),
        })
        if torch.cuda.device_count() > 0:
            payload['device_count'] = torch.cuda.device_count()
            payload['device_name'] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            payload['device_memory_gb'] = round(props.total_memory / (1024 ** 3), 2)
            capability = torch.cuda.get_device_capability(0)
            payload['device_capability'] = f"{capability[0]}.{capability[1]}"
        if payload.get('cuda_available'):
            try:
                tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
                tensor_sum = float(tensor.sum().item())
                torch.cuda.synchronize()
                payload['tensor_device'] = str(tensor.device)
                payload['tensor_sum'] = tensor_sum
                payload['cuda_probe_ok'] = True
            except Exception as cuda_exc:
                payload['cuda_error'] = repr(cuda_exc)
    except Exception as exc:
        payload['error'] = repr(exc)
    payload['warnings'] = [str(item.message) for item in caught]

print(json.dumps(payload))
"""
    result = run([sys.executable, "-c", probe_code], check=False, capture_output=True)
    lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {
        "import_ok": False,
        "cuda_probe_ok": False,
        "python_executable": sys.executable,
        "warnings": [],
        "error": (result.stderr or result.stdout or "Unable to parse torch probe output").strip(),
    }


def torch_ready_for_gpu(info: dict[str, object]) -> bool:
    return bool(
        info.get("import_ok")
        and info.get("cuda_available")
        and info.get("cuda_probe_ok")
        and not is_blackwell_incompatibility(info)
    )


def probe_text(info: dict[str, object]) -> str:
    parts = [
        str(info.get("error") or ""),
        str(info.get("cuda_error") or ""),
        "\n".join(str(item) for item in info.get("warnings", [])),
    ]
    return "\n".join(part for part in parts if part)


def is_blackwell_gpu(info: dict[str, object]) -> bool:
    device_name = str(info.get("device_name") or "")
    device_capability = str(info.get("device_capability") or "")
    text = probe_text(info).lower()
    return (
        "blackwell" in device_name.lower()
        or device_capability.startswith("12.")
        or "sm_120" in text
    )


def is_blackwell_incompatibility(info: dict[str, object]) -> bool:
    text = probe_text(info).lower()
    return is_blackwell_gpu(info) and (
        "sm_120" in text
        or "no kernel image is available" in text
        or "not compatible with the current pytorch installation" in text
    )


def recommended_torch_command() -> str:
    return (
        f"{sys.executable} -m pip install --upgrade {RECOMMENDED_BLACKWELL_TORCH_PACKAGES} "
        f"--index-url {RECOMMENDED_BLACKWELL_TORCH_INDEX_URL}"
    )


def resolve_torch_install_target(info: dict[str, object]) -> tuple[str, list[str]]:
    index_url = TORCH_INDEX_URL
    packages = list(TORCH_PACKAGES)

    if not is_blackwell_gpu(info):
        return index_url, packages

    stale_index = index_url.rstrip("/") in LEGACY_BLACKWELL_TORCH_INDEX_URLS
    stale_packages = packages == ["torch", "torchvision", "torchaudio"]
    installed_version = str(info.get("version") or "")
    needs_blackwell_override = (
        stale_index
        or stale_packages
        or version_key(installed_version) < version_key("2.10.0")
        or is_blackwell_incompatibility(info)
    )
    if not needs_blackwell_override:
        return index_url, packages

    return (
        RECOMMENDED_BLACKWELL_TORCH_INDEX_URL,
        shlex.split(RECOMMENDED_BLACKWELL_TORCH_PACKAGES),
    )


def log_torch_diagnosis(info: dict[str, object]) -> None:
    for warning in info.get("warnings", []):
        log(f"torch warning: {warning}")

    if info.get("import_ok"):
        log(
            "Torch probe summary: "
            f"version={info.get('version')}, "
            f"cuda_version={info.get('cuda_version')}, "
            f"cuda_available={info.get('cuda_available')}, "
            f"cuda_probe_ok={info.get('cuda_probe_ok')}"
        )
    else:
        log(f"Torch probe failed: {info.get('error')}")

    if is_blackwell_incompatibility(info):
        log("Detected a Blackwell GPU with a PyTorch wheel that does not include sm_120 CUDA kernels.")
        log(f"Recommended TORCH_INDEX_URL={RECOMMENDED_BLACKWELL_TORCH_INDEX_URL}")
        log(f"Recommended TORCH_PACKAGES={RECOMMENDED_BLACKWELL_TORCH_PACKAGES}")
        log(f"Recommended install command: {recommended_torch_command()}")
        if TORCH_INDEX_URL != RECOMMENDED_BLACKWELL_TORCH_INDEX_URL:
            log(
                f"Current TORCH_INDEX_URL={TORCH_INDEX_URL}. "
                f"For Blackwell on RUN:AI, use {RECOMMENDED_BLACKWELL_TORCH_INDEX_URL} or newer."
            )


def ensure_torch() -> dict[str, object]:
    info = probe_torch()
    if torch_ready_for_gpu(info):
        log("CUDA-enabled PyTorch is already available in this notebook kernel environment.")
        return info

    log_torch_diagnosis(info)

    if not AUTO_INSTALL_TORCH:
        log("AUTO_INSTALL_TORCH=0, so the notebook will not modify the kernel environment.")
        return info

    log("Attempting to install or repair PyTorch for this notebook kernel.")
    pip_install(["--upgrade", "pip"])
    install_index_url, install_packages = resolve_torch_install_target(info)
    if install_index_url != TORCH_INDEX_URL or install_packages != list(TORCH_PACKAGES):
        log(
            "Overriding the requested PyTorch install target for Blackwell compatibility: "
            f"TORCH_INDEX_URL={install_index_url}, TORCH_PACKAGES={' '.join(install_packages)}"
        )
    install_args = ["--upgrade"]
    if info.get("version"):
        install_args.extend(["--force-reinstall", "--no-cache-dir"])
    install_args.extend(install_packages)
    if install_index_url:
        install_args.extend(["--index-url", install_index_url])
    pip_install(install_args)

    info = probe_torch()
    if torch_ready_for_gpu(info):
        log("PyTorch install completed and CUDA is now available.")
    else:
        log_torch_diagnosis(info)
        log("PyTorch install did not produce a CUDA-ready environment. Check TORCH_INDEX_URL and the RUN:AI base image.")
    return info


def probe_module_import(module_name: str) -> dict[str, object]:
    probe_code = (
        "import importlib, json\n"
        f"module_name = {module_name!r}\n"
        "payload = {'module': module_name, 'import_ok': False}\n"
        "try:\n"
        "    importlib.import_module(module_name)\n"
        "    payload['import_ok'] = True\n"
        "except Exception as exc:\n"
        "    payload['error'] = repr(exc)\n"
        "print(json.dumps(payload))\n"
    )
    result = run([sys.executable, "-c", probe_code], check=False, capture_output=True)
    lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {
        "module": module_name,
        "import_ok": False,
        "error": (result.stderr or result.stdout or "Unable to parse module import probe output").strip(),
    }


def collect_numpy_optional_dependency_issues() -> list[tuple[str, str, str]]:
    issues: list[tuple[str, str, str]] = []
    for module_name, requirement in RUNTIME_OPTIONAL_NUMPY_DEPENDENCIES.items():
        info = probe_module_import(module_name)
        if not info.get("import_ok"):
            issues.append((module_name, requirement, str(info.get("error") or "import failed")))
    return issues


def ensure_numpy_optional_dependencies() -> None:
    issues = collect_numpy_optional_dependency_issues()
    if not issues:
        return

    log("Detected NumPy optional dependencies that need reinstall for NumPy 2.x compatibility.")
    for module_name, _, error in issues:
        log(f"{module_name} import failed: {error}")

    install_args = ["--upgrade", "--force-reinstall", "--no-cache-dir"]
    install_args.extend(requirement for _, requirement, _ in issues)
    pip_install(install_args)

    remaining = collect_numpy_optional_dependency_issues()
    if remaining:
        details = "; ".join(f"{module_name}: {error}" for module_name, _, error in remaining)
        raise RuntimeError(f"NumPy optional dependency repair failed: {details}")

    log("NumPy optional dependencies are importable.")


def gpu_sanity_check() -> dict[str, object]:
    log("Starting GPU sanity checks.")
    print("\n=== GPU visibility via nvidia-smi ===")
    nvidia = run(["nvidia-smi"], check=False)
    if nvidia.returncode != 0:
        log("nvidia-smi failed or no NVIDIA GPU is available to this notebook session.")
        gpu_inventory: list[dict[str, object]] = []
    else:
        gpu_inventory = query_nvidia_smi_summary()
        for index, record in enumerate(gpu_inventory):
            print(f"nvidia_gpu_{index}_name: {record['name']}")
            print(f"nvidia_gpu_{index}_total_memory_gb: {record['memory_gb']}")
            print(f"nvidia_gpu_{index}_driver_version: {record['driver_version']}")

    print("\n=== PyTorch CUDA sanity check ===")
    info = ensure_torch()
    print(f"python_executable: {info.get('python_executable', sys.executable)}")
    print(f"torch_import_ok: {info.get('import_ok')}")
    if info.get("version"):
        print(f"torch_version: {info.get('version')}")
    if info.get("cuda_version") is not None or info.get("import_ok"):
        print(f"torch_cuda_version: {info.get('cuda_version')}")
    if info.get("cuda_available") is not None:
        print(f"cuda_available: {info.get('cuda_available')}")
    if info.get("device_count") is not None:
        print(f"device_count: {info.get('device_count')}")
    if info.get("device_name"):
        print(f"device_name: {info.get('device_name')}")
    if info.get("device_capability"):
        print(f"device_capability: {info.get('device_capability')}")
    if info.get("device_memory_gb") is not None:
        print(f"device_0_total_memory_gb: {info.get('device_memory_gb')}")
    print(f"torch_cuda_probe_ok: {info.get('cuda_probe_ok')}")
    if info.get("tensor_device"):
        print(f"cuda_tensor_device: {info.get('tensor_device')}")
    if info.get("tensor_sum") is not None:
        print(f"cuda_tensor_sum: {info.get('tensor_sum')}")

    if not info.get("import_ok"):
        log(f"Torch import still failed after setup: {info.get('error')}")
    elif not info.get("cuda_available"):
        if "+cpu" in str(info.get("version") or "") or info.get("cuda_version") is None:
            log("Detected a CPU-only PyTorch build in this notebook kernel.")
        log("CUDA is not available to PyTorch.")
    elif not info.get("cuda_probe_ok"):
        log(f"CUDA tensor probe failed: {info.get('cuda_error')}")
        log("PyTorch detected the GPU, but the installed wheel could not execute a CUDA kernel.")

    gpu_ok = torch_ready_for_gpu(info)
    if gpu_ok:
        log("GPU sanity checks passed.")

    return {
        "gpu_ok": gpu_ok,
        "torch_info": info,
        "gpu_inventory": gpu_inventory,
    }


def resolve_launch_device(sanity: dict[str, object]) -> str | None:
    if sanity.get("gpu_ok"):
        return "cuda"

    info = sanity.get("torch_info", {})
    if GPU_FAILURE_MODE == "cpu":
        if isinstance(info, dict) and info.get("import_ok"):
            log("GPU validation did not pass. GPU_FAILURE_MODE=cpu, so the experiment will run with CUDA hidden from the child process.")
            log("CPU fallback is best suited to smoke runs or small models; full experiments may be slow.")
            return "cpu"
        log("GPU_FAILURE_MODE=cpu was requested, but PyTorch still does not import cleanly in this kernel. Skipping the experiment step.")
        return None

    if GPU_FAILURE_MODE == "skip-experiment":
        log("GPU validation did not pass. GPU_FAILURE_MODE=skip-experiment, so the experiment step will be skipped.")
        return None

    log("Stopping after sanity checks because GPU validation did not pass.")
    return None


def expected_output_dir(repo_dir: Path) -> Path:
    prefix = "real_traces_colab_smoke_" if EXPERIMENT_MODE == "smoke" else "real_traces_colab_"
    return repo_dir / "research" / "outputs" / f"{prefix}{MODEL}"


def launch_experiment(repo_dir: Path, launch_device: str) -> None:
    if EXPERIMENT_MODE not in {"smoke", "full"}:
        raise ValueError(f"EXPERIMENT_MODE must be 'smoke' or 'full', got: {EXPERIMENT_MODE}")
    if launch_device not in {"cuda", "cpu"}:
        raise ValueError(f"launch_device must be 'cuda' or 'cpu', got: {launch_device}")

    runner = repo_dir / "tools" / "run_colab_experiment.py"
    if not runner.exists():
        raise RuntimeError(f"Missing experiment runner: {runner}")

    output_dir = expected_output_dir(repo_dir)
    log(f"Preparing experiment launch for model={MODEL}, mode={EXPERIMENT_MODE}, launch_device={launch_device}")
    log(f"Expected output directory: {output_dir}")
    if os.environ.get("HF_TOKEN"):
        log("HF_TOKEN detected in environment.")
    else:
        log("HF_TOKEN not set. Public models should still work; gated models may fail to download.")

    command = [sys.executable, str(runner), "--model", MODEL]
    if EXPERIMENT_MODE == "smoke":
        command.append("--smoke-only")
    if not RUN_SIMULATOR:
        command.append("--skip-simulator")
    if SKIP_INSTALL:
        command.append("--skip-install")
    command.extend(EXPERIMENT_ARGS)

    command_env = {"CUDA_VISIBLE_DEVICES": ""} if launch_device == "cpu" else None
    run(command, cwd=repo_dir, env=command_env)
    log("Experiment command completed.")

    artifact_candidates = [
        output_dir / "metadata.json",
        output_dir / "pilot_summary.csv",
        output_dir / "detector_comparison.csv",
    ]
    for artifact in artifact_candidates:
        if artifact.exists():
            log(f"Artifact ready: {artifact}")
        else:
            log(f"Artifact not found yet: {artifact}")


def main() -> None:
    if GPU_FAILURE_MODE not in VALID_GPU_FAILURE_MODES:
        valid_modes = ", ".join(sorted(VALID_GPU_FAILURE_MODES))
        raise ValueError(f"GPU_FAILURE_MODE must be one of {valid_modes}; got: {GPU_FAILURE_MODE}")

    # Enable expandable CUDA memory segments for tight-VRAM Blackwell cards.
    if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    log("Notebook bootstrap starting.")
    log(f"WORKDIR={WORKDIR}")
    log(f"REPO_URL={REPO_URL}")
    log(
        "Configuration: "
        f"MODEL={MODEL}, "
        f"EXPERIMENT_MODE={EXPERIMENT_MODE}, "
        f"START_EXPERIMENT={START_EXPERIMENT}, "
        f"RUN_SIMULATOR={RUN_SIMULATOR}, "
        f"SKIP_INSTALL={SKIP_INSTALL}, "
        f"AUTO_INSTALL_TORCH={AUTO_INSTALL_TORCH}, "
        f"GPU_FAILURE_MODE={GPU_FAILURE_MODE}, "
        f"TORCH_INDEX_URL={TORCH_INDEX_URL}"
    )

    repo_dir = clone_or_update_repo()
    log(f"Repository ready at: {repo_dir}")

    sanity = gpu_sanity_check()
    launch_device = resolve_launch_device(sanity)
    if launch_device is None:
        return

    if not START_EXPERIMENT:
        log("START_EXPERIMENT=0, stopping after repo update and GPU sanity checks.")
        return

    ensure_numpy_optional_dependencies()
    launch_experiment(repo_dir, launch_device)


if __name__ == "__main__":
    main()
