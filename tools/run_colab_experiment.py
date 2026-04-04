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
MODEL_CHOICES = [
    "deepseek_r1_distill_1p5b",
    "deepseek_r1_distill_7b",
    "qwen2p5_0p5b",
    "qwen2p5_7b",
    "mistral_7b_instruct_v0p3",
    "gemma_4_31b_it",
    "gemma_4_26b_moe_it",
    "gemma_4_e4b_it",
    "qwen_3p5_7b_instruct",
    "qwen_3p5_35b_moe_it",
    "llama_4_8b_it",
    "llama_3p1_8b_instruct",
]


def run_command(command: list[str], cwd: Path = REPO_ROOT) -> None:
    printable = " ".join(command)
    print(f"\n[run] {printable}", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    subprocess.run(command, cwd=cwd, check=True, env=env)


def ensure_packages(skip_install: bool) -> None:
    if skip_install:
        print("[setup] Skipping dependency installation by request.", flush=True)
        return

    required_modules = {
        "transformers": "transformers",
        "accelerate": "accelerate",
        "datasets": "datasets",
        "evaluate": "evaluate",
        "bitsandbytes": "bitsandbytes",
        "sentencepiece": "sentencepiece",
        "safetensors": "safetensors",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "numpy": "numpy",
        "tqdm": "tqdm",
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


def prepare_output_dir(path: Path, clear: bool = False) -> None:
    if clear and path.exists():
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


def resolve_model_output_dir(requested: str, default_path: Path, model: str, label: str) -> Path:
    requested_path = Path(requested)
    if requested_path != default_path:
        return requested_path

    resolved_path = default_path.parent / f"{default_path.name}_{model}"
    print(
        f"[output] No explicit {label} output directory supplied. Using model-scoped path: {resolved_path}",
        flush=True,
    )
    return resolved_path


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
    task_source: str,
    dataset_split: str,
    dataset_shuffle_seed: int,
    batch_size: int,
    prompt_mode: str,
    system_prompt_mode: str,
    quantization: str,
    device_map: str | None,
    attn_implementation: str,
    resume: bool,
    io_threads: int = 4,
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
        "--task-source",
        task_source,
        "--dataset-split",
        dataset_split,
        "--dataset-shuffle-seed",
        str(dataset_shuffle_seed),
        "--batch-size",
        str(batch_size),
        "--prompt-mode",
        prompt_mode,
        "--system-prompt-mode",
        system_prompt_mode,
        "--attn-implementation",
        attn_implementation,
        "--quantization",
        quantization,
        "--output-dir",
        str(output_dir),
        "--io-threads",
        str(io_threads),
    ]
    if device_map:
        command.extend(["--device-map", device_map])
    command.append("--temperatures")
    command.extend(str(value) for value in temperatures)
    command.append("--seeds")
    command.extend(str(value) for value in seeds)
    if not resume:
        command.append("--no-resume")
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
    parser.add_argument(
        "--model",
        default="deepseek_r1_distill_1p5b",
        choices=MODEL_CHOICES,
    )
    parser.add_argument(
        "--smoke-model",
        default=None,
        choices=MODEL_CHOICES,
    )
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-simulator", action="store_true")
    parser.add_argument("--prompt-mode", default="minimal_json", choices=["structured_four_line", "minimal_json", "answer_only"])
    parser.add_argument("--system-prompt-mode", default="default", choices=["none", "short", "default"])
    parser.add_argument("--task-source", default="gsm8k", choices=["builtin", "gsm8k"])
    parser.add_argument("--smoke-task-source", default="builtin", choices=["builtin", "gsm8k"])
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-shuffle-seed", type=int, default=17)
    parser.add_argument("--quantization", default="none", choices=["none", "8bit", "4bit"])
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--attn-implementation", default="sdpa", choices=["auto", "sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--full-max-tasks", type=int, default=300)
    parser.add_argument("--full-max-steps", type=int, default=10)
    parser.add_argument("--full-max-new-tokens", type=int, default=256)
    parser.add_argument("--full-batch-size", type=int, default=4)
    parser.add_argument("--full-temperatures", nargs="+", type=float, default=[0.1, 0.6, 1.0])
    parser.add_argument("--full-seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--smoke-max-tasks", type=int, default=2)
    parser.add_argument("--smoke-max-steps", type=int, default=2)
    parser.add_argument("--smoke-max-new-tokens", type=int, default=128)
    parser.add_argument("--smoke-batch-size", type=int, default=1)
    parser.add_argument("--smoke-temperatures", nargs="+", type=float, default=[0.1])
    parser.add_argument("--smoke-seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--output-dir", default=str(DEFAULT_FULL_DIR))
    parser.add_argument("--smoke-output-dir", default=str(DEFAULT_SMOKE_DIR))
    parser.add_argument("--io-threads", type=int, default=4, help="Number of background threads for IO operations")
    parser.add_argument("--fresh-output", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    started_at = time.time()
    ensure_packages(skip_install=args.skip_install)
    device = print_environment()
    smoke_model = args.smoke_model or args.model

    full_output_dir = resolve_model_output_dir(args.output_dir, DEFAULT_FULL_DIR, args.model, "full")
    smoke_output_dir = resolve_model_output_dir(args.smoke_output_dir, DEFAULT_SMOKE_DIR, smoke_model, "smoke")

    if not args.skip_simulator:
        run_simulator()
        print_csv(REPO_ROOT / "research" / "outputs" / "summary.csv", "Synthetic Detector Summary")

    if not args.skip_smoke:
        prepare_output_dir(smoke_output_dir, clear=True)
        print("\n[phase] Running smoke test before the full experiment.", flush=True)
        smoke_quantization = args.quantization if smoke_model == args.model else "none"
        smoke_device_map = args.device_map if smoke_model == args.model else None
        run_real_trace_experiment(
            model=smoke_model,
            device=device,
            max_tasks=args.smoke_max_tasks,
            max_steps=args.smoke_max_steps,
            max_new_tokens=args.smoke_max_new_tokens,
            temperatures=args.smoke_temperatures,
            seeds=args.smoke_seeds,
            output_dir=smoke_output_dir,
            task_source=args.smoke_task_source,
            dataset_split=args.dataset_split,
            dataset_shuffle_seed=args.dataset_shuffle_seed,
            batch_size=args.smoke_batch_size,
            prompt_mode=args.prompt_mode,
            system_prompt_mode=args.system_prompt_mode,
            quantization=smoke_quantization,
            device_map=smoke_device_map,
            attn_implementation=args.attn_implementation,
            resume=False,
            io_threads=args.io_threads,
        )
        run_analysis(smoke_output_dir)
        print_csv(smoke_output_dir / "pilot_summary.csv", "Smoke Pilot Summary")
        print_csv(smoke_output_dir / "detector_comparison.csv", "Smoke Detector Comparison")
        if args.smoke_only:
            archive = zip_results(smoke_output_dir)
            print(f"\n[done] Smoke test archive: {archive}", flush=True)
            return

    prepare_output_dir(full_output_dir, clear=args.fresh_output)
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
        task_source=args.task_source,
        dataset_split=args.dataset_split,
        dataset_shuffle_seed=args.dataset_shuffle_seed,
        batch_size=args.full_batch_size,
        prompt_mode=args.prompt_mode,
        system_prompt_mode=args.system_prompt_mode,
        quantization=args.quantization,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        resume=args.resume,
        io_threads=args.io_threads,
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

    # Trigger Colab automated shutdown to save GPU credits
    if os.path.exists("/content"):
        print("[done] Triggering Colab autonomous shutdown...", flush=True)
        os.system("touch /content/SHUTDOWN_COLAB.txt")


if __name__ == "__main__":
    main()
