from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
COLAB_REQUIREMENTS = REPO_ROOT / "requirements-colab.txt"
DEFAULT_SMOKE_DIR = REPO_ROOT / "research" / "outputs" / "real_traces_colab_smoke"
DEFAULT_FULL_DIR = REPO_ROOT / "research" / "outputs" / "real_traces_colab"


def run_command(command: list[str], cwd: Path = REPO_ROOT) -> None:
    printable = " ".join(command)
    print(f"\n[run] {printable}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def ensure_packages(skip_install: bool) -> None:
    if skip_install:
        print("[setup] Skipping dependency installation by request.", flush=True)
        return

    required_modules = {
        "transformers": "transformers",
        "accelerate": "accelerate",
        "sentencepiece": "sentencepiece",
        "safetensors": "safetensors",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "numpy": "numpy",
    }
    missing_packages = [package for module, package in required_modules.items() if importlib.util.find_spec(module) is None]
    if not missing_packages:
        print("[setup] Required Python packages are already present.", flush=True)
        return

    print(f"[setup] Installing missing packages: {', '.join(missing_packages)}", flush=True)
    run_command([PYTHON, "-m", "pip", "install", "-q", "-r", str(COLAB_REQUIREMENTS)])


def print_environment() -> str:
    import torch

    print(f"[env] Python: {sys.version.split()[0]}", flush=True)
    print(f"[env] Torch: {torch.__version__}", flush=True)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[env] CUDA available: yes ({gpu_name}, {gpu_mem_gb:.1f} GB)", flush=True)
        return "cuda"

    print("[env] CUDA available: no", flush=True)
    return "cpu"


def prepare_output_dir(path: Path) -> None:
    if path.exists():
        for child in path.iterdir():
            if child.is_file():
                child.unlink()
            else:
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_file():
                        nested.unlink()
                    else:
                        nested.rmdir()
                child.rmdir()
    path.mkdir(parents=True, exist_ok=True)


def run_simulator() -> None:
    run_command([PYTHON, str(REPO_ROOT / "research" / "simulate_overthinking_boundary.py")])


def run_real_trace_experiment(
    *,
    model: str,
    device: str,
    max_tasks: int,
    max_steps: int,
    max_new_tokens: int,
    temperatures: list[float],
    seeds: list[int],
    output_dir: Path,
) -> None:
    command = [
        PYTHON,
        str(REPO_ROOT / "research" / "real_trace_experiments.py"),
        "--model",
        model,
        "--device",
        device,
        "--max-tasks",
        str(max_tasks),
        "--max-steps",
        str(max_steps),
        "--max-new-tokens",
        str(max_new_tokens),
        "--output-dir",
        str(output_dir),
        "--temperatures",
    ]
    command.extend(str(value) for value in temperatures)
    command.append("--seeds")
    command.extend(str(value) for value in seeds)
    run_command(command)


def run_analysis(output_dir: Path) -> None:
    run_command([PYTHON, str(REPO_ROOT / "research" / "trace_analysis.py"), "--input-dir", str(output_dir)])


def print_csv(path: Path, title: str, max_rows: int = 10) -> None:
    import pandas as pd

    if not path.exists():
        print(f"[summary] Missing expected file: {path}", flush=True)
        return

    frame = pd.read_csv(path)
    print(f"\n=== {title} ===", flush=True)
    if frame.empty:
        print("<empty>", flush=True)
        return
    print(frame.head(max_rows).to_string(index=False), flush=True)


def print_json(path: Path, title: str) -> None:
    if not path.exists():
        print(f"[summary] Missing expected file: {path}", flush=True)
        return
    print(f"\n=== {title} ===", flush=True)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    print(json.dumps(payload, indent=2), flush=True)


def zip_results(output_dir: Path) -> Path:
    archive_path = output_dir.with_suffix(".zip")
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.relative_to(output_dir.parent))
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the overthinking-boundary experiment safely on Google Colab.")
    parser.add_argument("--model", default="deepseek_r1_distill_1p5b", choices=["deepseek_r1_distill_1p5b", "qwen2p5_0p5b"])
    parser.add_argument("--smoke-model", default="qwen2p5_0p5b", choices=["deepseek_r1_distill_1p5b", "qwen2p5_0p5b"])
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-simulator", action="store_true")
    parser.add_argument("--full-max-tasks", type=int, default=4)
    parser.add_argument("--full-max-steps", type=int, default=5)
    parser.add_argument("--full-max-new-tokens", type=int, default=56)
    parser.add_argument("--full-temperatures", nargs="+", type=float, default=[0.2, 0.6])
    parser.add_argument("--full-seeds", nargs="+", type=int, default=[7, 13])
    parser.add_argument("--smoke-max-tasks", type=int, default=2)
    parser.add_argument("--smoke-max-steps", type=int, default=2)
    parser.add_argument("--smoke-max-new-tokens", type=int, default=12)
    parser.add_argument("--smoke-temperatures", nargs="+", type=float, default=[0.2])
    parser.add_argument("--smoke-seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--output-dir", default=str(DEFAULT_FULL_DIR))
    parser.add_argument("--smoke-output-dir", default=str(DEFAULT_SMOKE_DIR))
    args = parser.parse_args()

    started_at = time.time()
    ensure_packages(skip_install=args.skip_install)
    device = print_environment()

    full_output_dir = Path(args.output_dir)
    smoke_output_dir = Path(args.smoke_output_dir)

    if not args.skip_simulator:
        run_simulator()
        print_csv(REPO_ROOT / "research" / "outputs" / "summary.csv", "Synthetic Detector Summary")

    if not args.skip_smoke:
        prepare_output_dir(smoke_output_dir)
        print("\n[phase] Running smoke test before the full experiment.", flush=True)
        run_real_trace_experiment(
            model=args.smoke_model,
            device=device,
            max_tasks=args.smoke_max_tasks,
            max_steps=args.smoke_max_steps,
            max_new_tokens=args.smoke_max_new_tokens,
            temperatures=args.smoke_temperatures,
            seeds=args.smoke_seeds,
            output_dir=smoke_output_dir,
        )
        run_analysis(smoke_output_dir)
        print_csv(smoke_output_dir / "pilot_summary.csv", "Smoke Pilot Summary")
        print_csv(smoke_output_dir / "detector_comparison.csv", "Smoke Detector Comparison")
        if args.smoke_only:
            archive = zip_results(smoke_output_dir)
            print(f"\n[done] Smoke test archive: {archive}", flush=True)
            return

    prepare_output_dir(full_output_dir)
    print("\n[phase] Running the full real-trace experiment.", flush=True)
    run_real_trace_experiment(
        model=args.model,
        device=device,
        max_tasks=args.full_max_tasks,
        max_steps=args.full_max_steps,
        max_new_tokens=args.full_max_new_tokens,
        temperatures=args.full_temperatures,
        seeds=args.full_seeds,
        output_dir=full_output_dir,
    )
    run_analysis(full_output_dir)

    print_json(full_output_dir / "metadata.json", "Run Metadata")
    print_csv(full_output_dir / "pilot_summary.csv", "Pilot Summary")
    print_csv(full_output_dir / "detector_comparison.csv", "Detector Comparison")
    print_csv(full_output_dir / "hazard_drift_summary.csv", "Hazard Drift Summary")
    print_csv(full_output_dir / "correctness_probe_metrics.csv", "Correctness Probe Metrics")

    archive = zip_results(full_output_dir)
    elapsed_minutes = (time.time() - started_at) / 60.0
    print(f"\n[done] Full results directory: {full_output_dir}", flush=True)
    print(f"[done] Zipped archive: {archive}", flush=True)
    print(f"[done] Total runtime: {elapsed_minutes:.2f} minutes", flush=True)


if __name__ == "__main__":
    main()
