#!/usr/bin/env python
"""Experiment-matrix orchestrator: every model x every dataset, end to end.

One command runs the whole grid and chains all four pipeline stages per cell:

    collect    real_trace_experiments.py   -> <root>/<model>__<dataset>/
    analyze    trace_analysis.py           -> figures + detector tables in the cell dir
    report     generate_thesis_artifacts.py-> per-cell markdown (routed into the cell dir)
    aggregate  cross_family_analysis.py    -> one cross-family report per dataset

Engine
------
* VRAM-budget scheduler. collect (GPU) jobs run concurrently up to a memory
  budget: small models pack onto the card together, big models run solo, all
  bounded so the card never overcommits. analyze/report (CPU) run pipelined on a
  thread pool while other models are still collecting on the GPU. Per-cell OOM is
  already handled inside real_trace_experiments.py (automatic microbatch split),
  so the budget is the coarse guard and the harness is the fine one.
* Progress. A tqdm bar tracks collect across the grid; every stage prints a
  "> model | dataset" start line and a check/cross finish line, so you always
  know which model and dataset are running.
* Git checkpoints. After each cell finishes, the light artifacts (CSVs, figures,
  reports, manifest) are committed (and optionally pushed); large .npz hidden
  states are excluded via .gitignore so the repo stays small.
* Resumable. A manifest + on-disk markers skip finished cells; re-running only
  does outstanding work. real_trace_experiments.py also resumes mid-cell.
* Import-light control. Models are discovered by AST-parsing MODEL_CATALOG and
  CSVs are read with the stdlib, so --status/--dry-run work without torch.

Datasets: only `gsm8k`/`builtin` are wired in real_trace_experiments.py today.
Add the parser there + one line to DATASETS below and the matrix picks it up.

Examples
--------
    python research/run_experiment_matrix.py --status
    python research/run_experiment_matrix.py --dry-run
    python research/run_experiment_matrix.py                       # all outstanding work
    python research/run_experiment_matrix.py --max-parallel 1      # strictly sequential
    python research/run_experiment_matrix.py --git-checkpoint --git-push
    python research/run_experiment_matrix.py --smoke --models qwen2p5_0p5b
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
PYTHON = sys.executable

COLLECT_SCRIPT = HERE / "real_trace_experiments.py"
ANALYZE_SCRIPT = HERE / "trace_analysis.py"
REPORT_SCRIPT = HERE / "generate_thesis_artifacts.py"
AGGREGATE_SCRIPT = HERE / "cross_family_analysis.py"

DEFAULT_OUTPUT_ROOT = HERE / "outputs" / "experiment_matrix"
MANIFEST_NAME = "matrix_manifest.json"
PHASES = ("collect", "analyze", "report", "aggregate")

# Canonical protocol -- matches research/outputs/real_traces_l4_qwen_7b_4bit/metadata.json
# so task IDs line up across models and with the legacy single-model runs.
CANONICAL = {
    "temperatures": ["0.1", "0.6", "1.0"],
    "seeds": ["7"],
    "max_steps": 10,
    "max_new_tokens": 256,
    "dataset_shuffle_seed": 17,
    "prompt_mode": "minimal_json",
    "system_prompt_mode": "default",
    "attn_implementation": "sdpa",  # bf16 load path has no FlashAttention-2 fallback
}

# Dataset registry. Only task-sources wired into real_trace_experiments.py work.
DATASETS: dict[str, dict[str, Any]] = {
    "gsm8k": {"split": "train", "max_tasks": 500, "max_steps": 10},
    "builtin": {"split": "train", "max_tasks": None, "max_steps": 5},
    "math": {"split": "test", "max_tasks": 500, "max_steps": 14},
    "arc": {"split": "test", "max_tasks": 500, "max_steps": 8},      # ARC-Challenge (public, MCQ)
    "gpqa": {"split": "train", "max_tasks": 448, "max_steps": 10},   # GPQA main (gated, MCQ) -> needs HF_TOKEN
    # "humaneval": {"split": "test", "max_tasks": 164, "max_steps": 12},  # needs parser (code execution)
}
DEFAULT_DATASETS = ["gsm8k"]

# Excluded from "all models" by default: unreleased / unverified-download placeholders.
SPECULATIVE_PREFIXES = ("gemma_4", "qwen_3p5")

_MANIFEST_LOCK = threading.Lock()
_GIT_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Discovery & profiling (import-light)
# --------------------------------------------------------------------------- #
def parse_model_catalog(path: Path) -> dict[str, dict[str, Any]]:
    """Extract MODEL_CATALOG (alias -> ModelSpec kwargs) without importing torch."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "MODEL_CATALOG" for t in node.targets
        ) and isinstance(node.value, ast.Dict):
            catalog: dict[str, dict[str, Any]] = {}
            for key_node, val_node in zip(node.value.keys, node.value.values):
                alias = ast.literal_eval(key_node)
                kwargs = {kw.arg: ast.literal_eval(kw.value)
                          for kw in getattr(val_node, "keywords", []) if kw.arg}
                catalog[alias] = kwargs
            return catalog
    raise RuntimeError(f"MODEL_CATALOG not found in {path}")


def params_billions(parameter_count: str) -> float:
    found = re.findall(r"[\d.]+", parameter_count or "")
    return float(found[0]) if found else float("nan")


def model_profile(spec: dict[str, Any]) -> dict[str, Any]:
    """Auto precision + batch size for a ~96 GB card. Override via --config."""
    p = params_billions(spec.get("parameter_count", ""))
    if p != p:  # NaN -> be conservative
        return {"quantization": "4bit", "batch_size": 4, "params_b": float("nan")}
    quant = "none" if p <= 34 else "4bit"  # bf16 up to ~34B (32B bf16 ~64 GB)
    # THROUGHPUT FIX: decode is memory-bandwidth-bound, so bigger batches raise
    # tok/s (measured ~70 tok/s at the old sizes -- far below the card's ceiling).
    # ~2x the old batches for small/mid models (ample VRAM headroom); modest for
    # 32B (VRAM-limited). The harness OOM-microbatch-splits if a cell overshoots.
    bs = (128 if p <= 1 else 96 if p <= 4 else 64 if p <= 8
          else 40 if p <= 15 else 16 if p <= 34 else 8)
    return {"quantization": quant, "batch_size": bs, "params_b": p}


def estimate_footprint_gb(profile: dict[str, Any]) -> float:
    """Rough peak VRAM for one collect process: weights + activation/KV headroom."""
    p = profile.get("params_b")
    if not p or p != p:
        return 10.0
    bytes_per_param = 2.0 if profile["quantization"] == "none" else 0.6  # bf16 vs 4-bit
    weights = p * bytes_per_param
    # SPEEDUP FIX: the old reserve (0.30*weights) badly overestimated peak VRAM
    # (it put the 32B at ~85 GB when it actually uses ~70 GB), so the scheduler
    # ran big models solo and left the GPU under-packed. Measured peak is close
    # to weights + a modest KV/activation margin; the harness OOM-splits if a
    # cell overshoots, so a tighter estimate is safe and lets models co-run.
    # Batch-aware: bigger batches grow the KV cache, so packing must account for
    # it (calibrated so 32B@bs12 ~= 71 GB and 7B@bs32 ~= 18 GB, matching peaks).
    batch = profile.get("batch_size") or 16
    kv_activations = 0.018 * p * batch  # KV cache + activations at a few-thousand-token context
    return weights + kv_activations + 2.5


def query_total_vram_gb() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout
        first = next(line for line in out.splitlines() if line.strip())
        return float(first.strip()) / 1024.0
    except Exception:  # noqa: BLE001 -- nvidia-smi absent or unparsable
        return None


# --------------------------------------------------------------------------- #
# Cell helpers
# --------------------------------------------------------------------------- #
def cell_key(model: str, dataset: str) -> str:
    return f"{model}__{dataset}"


def cell_dir(root: Path, model: str, dataset: str) -> Path:
    return root / cell_key(model, dataset)


def collection_complete(cdir: Path, scope: dict[str, Any] | None = None, dataset: str | None = None) -> bool:
    meta, steps = cdir / "metadata.json", cdir / "trace_steps.csv"
    if not meta.exists() or not steps.exists():
        return False
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    pending = data.get("pending_run_count")
    if pending is not None:
        if pending != 0:
            return False
    elif data.get("next_pending_run") is not None:
        return False
    # AUDIT FIX: a cell counts as complete only for the REQUESTED scope. A
    # leftover smoke run (e.g. 6 tasks, 1 temp) has pending==0 but must not
    # satisfy a full 500-task / 3-temp request, or its tiny, statistically
    # meaningless trace set silently flows into per-cell + cross-family results.
    if scope is not None:
        ds = DATASETS.get(dataset or "", {})
        req_tasks = scope.get("max_tasks") if scope.get("max_tasks") is not None else ds.get("max_tasks")
        on_disk_tasks = data.get("max_tasks")
        if req_tasks is not None and (on_disk_tasks is None or int(on_disk_tasks) < int(req_tasks)):
            return False
        req_temps = {float(t) for t in scope.get("temperatures", [])}
        on_disk_temps = {float(t) for t in data.get("temperatures", [])}
        if req_temps and not req_temps.issubset(on_disk_temps):
            return False
    return True


def analysis_complete(cdir: Path) -> bool:
    # Existence alone let a cell whose collection later GREW keep passing as
    # analyzed (qwen2p5_7b__math sat at 75/1500 detector-stage runs for two
    # weeks -- rigor_audit/01 §5c). Analysis is complete only if its outputs
    # are no older than the trace data they were computed from.
    marker = cdir / "real_trace_feature_weights.png"
    dcbr = cdir / "detector_comparison_by_run.csv"
    steps = cdir / "trace_steps.csv"
    if not (marker.exists() and dcbr.exists()):
        return False
    if steps.exists() and dcbr.stat().st_mtime < steps.stat().st_mtime:
        return False
    return True


def report_complete(cdir: Path) -> bool:
    return (cdir / "final_results.md").exists()


def trace_counts(cdir: Path) -> tuple[int, int]:
    steps = cdir / "trace_steps.csv"
    if not steps.exists():
        return 0, 0
    run_ids: set[str] = set()
    rows = 0
    # Corruption fragments (field-shifted rows from interrupted writes) carry
    # malformed run_ids and previously inflated collected-run counts by 24
    # phantom ids (rigor_audit/01 Axis 5a). Count well-formed run_ids only.
    run_id_ok = re.compile(r"__temp\d+(?:\.\d+)?__seed\d+$")
    try:
        with steps.open(newline="", encoding="utf-8", errors="replace") as fh:
            for row in csv.DictReader(fh):
                rows += 1
                rid = row.get("run_id")
                if rid and run_id_ok.search(rid):
                    run_ids.add(rid)
    except OSError:
        return 0, 0
    return len(run_ids), rows


# --------------------------------------------------------------------------- #
# Command builders
# --------------------------------------------------------------------------- #
def collect_cmd(model: str, dataset: str, cdir: Path, profile: dict[str, Any],
                scope: dict[str, Any]) -> list[str]:
    ds = DATASETS.get(dataset, {})
    max_tasks = scope["max_tasks"] if scope["max_tasks"] is not None else ds.get("max_tasks")
    max_steps = scope["max_steps"] or ds.get("max_steps") or CANONICAL["max_steps"]
    split = ds.get("split", "train")
    cmd = [
        PYTHON, str(COLLECT_SCRIPT),
        "--model", model,
        "--device", "cuda",
        "--quantization", profile["quantization"],
        "--attn-implementation", CANONICAL["attn_implementation"],
        "--task-source", dataset,
        "--dataset-split", split,
        "--dataset-shuffle-seed", str(CANONICAL["dataset_shuffle_seed"]),
        "--max-steps", str(max_steps),
        "--max-new-tokens", str(scope["max_new_tokens"]),
        "--batch-size", str(profile["batch_size"]),
        "--prompt-mode", CANONICAL["prompt_mode"],
        "--system-prompt-mode", CANONICAL["system_prompt_mode"],
        "--temperatures", *scope["temperatures"],
        "--seeds", *scope["seeds"],
        "--output-dir", str(cdir),
    ]
    if max_tasks is not None:
        cmd += ["--max-tasks", str(max_tasks)]
    # Only large quantized models need an explicit device map; passing it for
    # small 4-bit models triggers accelerate's meta-tensor offload bug.
    if profile["quantization"] in {"4bit", "8bit"} and not (profile.get("params_b", 99) <= 8):
        cmd += ["--device-map", "auto"]
    return cmd


def analyze_cmd(cdir: Path) -> list[str]:
    return [PYTHON, str(ANALYZE_SCRIPT), "--input-dir", str(cdir)]


def report_cmd(cdir: Path) -> list[str]:
    return [
        PYTHON, str(REPORT_SCRIPT), "--input-dir", str(cdir),
        "--answers-output", str(cdir / "answers.md"),
        "--open-questions-output", str(cdir / "open_questions.md"),
        "--research-report-output", str(cdir / "final_results.md"),
        "--root-report-output", str(cdir / "summary.md"),
    ]


def aggregate_cmd(dataset: str, run_dirs: list[Path], agg_dir: Path) -> list[str]:
    return [
        PYTHON, str(AGGREGATE_SCRIPT), "--run-dirs", *[str(d) for d in run_dirs],
        "--output-dir", str(agg_dir / "cross_family"),
        "--report-output", str(agg_dir / "CROSS_FAMILY_REPORT.md"),
        "--open-questions-output", str(agg_dir / "CROSS_FAMILY_OPEN_QUESTIONS.md"),
    ]


# --------------------------------------------------------------------------- #
# Execution helpers
# --------------------------------------------------------------------------- #
def run_blocking(cmd: list[str], log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8", errors="replace") as fh:
        code = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=fh, stderr=subprocess.STDOUT).returncode
    return code, (time.monotonic() - start) / 60.0


def record(manifest: dict[str, Any], root: Path, model: str, dataset: str,
           stage: str, code: int, elapsed_min: float, cmd: list[str]) -> None:
    with _MANIFEST_LOCK:
        cell = manifest["cells"].setdefault(cell_key(model, dataset), {})
        cell[stage] = {"status": "ok" if code == 0 else "failed", "return_code": code,
                       "elapsed_min": round(elapsed_min, 2), "command": " ".join(cmd)}
        cell["n_runs"], _ = trace_counts(cell_dir(root, model, dataset))
        save_manifest(root, manifest)


def git_checkpoint(paths: list[Path], message: str, push: bool) -> str:
    """Commit light artifacts (npz excluded via .gitignore). Never fatal."""
    def _git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True)
    with _GIT_LOCK:
        try:
            _git("add", "--", *[str(p) for p in paths])
            if _git("diff", "--cached", "--quiet").returncode == 0:
                return "nothing to commit"
            commit = _git("commit", "-m", message)
            if commit.returncode != 0:
                return f"commit failed: {commit.stderr.strip()[:160]}"
            if push:
                pushed = _git("push")
                if pushed.returncode != 0:
                    return f"committed; push FAILED: {pushed.stderr.strip()[:160]}"
                return "committed + pushed"
            return "committed"
        except Exception as exc:  # noqa: BLE001
            return f"git error: {exc}"


def make_bar(total: int, enabled: bool):
    """tqdm bar with a stdlib fallback; .write() always prints even if disabled."""
    try:
        from tqdm.auto import tqdm
        bar = tqdm(total=total, desc="collect", unit="cell",
                   disable=(not enabled) or (not sys.stderr.isatty()))

        class _Wrap:
            def update(self, n: int = 1) -> None: bar.update(n)
            def set_postfix_str(self, s: str) -> None: bar.set_postfix_str(s)
            def write(self, s: str) -> None: bar.write(s)
            def close(self) -> None: bar.close()
        return _Wrap()
    except Exception:  # noqa: BLE001 -- tqdm missing
        class _Plain:
            def update(self, n: int = 1) -> None: ...
            def set_postfix_str(self, s: str) -> None: ...
            def write(self, s: str) -> None: print(s, flush=True)
            def close(self) -> None: ...
        return _Plain()


# --------------------------------------------------------------------------- #
# Manifest / config
# --------------------------------------------------------------------------- #
def load_manifest(root: Path) -> dict[str, Any]:
    path = root / MANIFEST_NAME
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"cells": {}, "aggregate": {}}


def save_manifest(root: Path, manifest: dict[str, Any]) -> None:
    # AUDIT FIX: write atomically so a concurrent reader (or a crash mid-write)
    # never sees a torn manifest.
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, root / MANIFEST_NAME)


def apply_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    for name, cfg in (data.get("datasets_config") or {}).items():
        DATASETS.setdefault(name, {}).update(cfg)
    return data


def resolve_models(args: argparse.Namespace, catalog: dict[str, Any], config: dict[str, Any]) -> list[str]:
    if args.models:
        requested = args.models
    elif config.get("models"):
        requested = config["models"]
    else:
        requested = [m for m in catalog if not m.startswith(SPECULATIVE_PREFIXES)]
    unknown = [m for m in requested if m not in catalog]
    if unknown:
        raise SystemExit(f"Unknown model alias(es): {', '.join(unknown)}\n"
                         f"Available: {', '.join(sorted(catalog))}")
    return requested


def resolve_datasets(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    datasets = args.datasets or config.get("datasets") or DEFAULT_DATASETS
    for ds in datasets:
        if ds not in DATASETS:
            print(f"[warn] dataset '{ds}' is not in the registry; the cell will fail if unsupported.")
    return datasets


def build_scope(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoke:
        return {"temperatures": ["0.1"], "seeds": ["7"], "max_tasks": 6,
                "max_steps": 3, "max_new_tokens": 128}
    return {
        "temperatures": args.temperatures or CANONICAL["temperatures"],
        "seeds": args.seeds or CANONICAL["seeds"],
        "max_tasks": args.max_tasks,
        "max_steps": args.max_steps,
        "max_new_tokens": args.max_new_tokens or CANONICAL["max_new_tokens"],
    }


def resolve_profile(spec: dict[str, Any], args: argparse.Namespace,
                    config: dict[str, Any], model: str) -> dict[str, Any]:
    profile = model_profile(spec)
    override = (config.get("model_overrides") or {}).get(model, {})
    profile.update({k: v for k, v in override.items() if k in {"quantization", "batch_size"}})
    if args.quantization:
        profile["quantization"] = args.quantization
    if args.batch_size:
        profile["batch_size"] = args.batch_size
    return profile


def resolve_max_parallel(value: str, n_jobs: int) -> int:
    if value == "auto":
        return max(1, min(n_jobs, 6))
    try:
        return max(1, int(value))
    except ValueError:
        raise SystemExit(f"--max-parallel must be an int or 'auto' (got {value!r})")


# --------------------------------------------------------------------------- #
# Status display
# --------------------------------------------------------------------------- #
def status_symbol(done: bool, failed: bool) -> str:
    return "x" if failed else ("OK" if done else "..")


def print_status(root: Path, models: list[str], datasets: list[str], manifest: dict[str, Any]) -> None:
    print(f"\nMatrix root: {root}")
    print(f"{len(models)} models x {len(datasets)} datasets = {len(models) * len(datasets)} cells")
    print("Legend: C=collect A=analyze R=report   OK=done  ..=pending  x=failed\n")
    width = max((len(m) for m in models), default=12) + 2
    header = "model".ljust(width) + "  " + "  ".join(f"{d:^14}" for d in datasets)
    print(header)
    print("-" * len(header))
    for model in models:
        row = model.ljust(width)
        for dataset in datasets:
            cdir = cell_dir(root, model, dataset)
            cell = manifest.get("cells", {}).get(cell_key(model, dataset), {})
            c = status_symbol(collection_complete(cdir), cell.get("collect", {}).get("status") == "failed")
            a = status_symbol(analysis_complete(cdir), cell.get("analyze", {}).get("status") == "failed")
            r = status_symbol(report_complete(cdir), cell.get("report", {}).get("status") == "failed")
            n_runs, _ = trace_counts(cdir)
            cellstr = f"C{c} A{a} R{r}" + (f" {n_runs}" if n_runs else "")
            row += "  " + f"{cellstr:^14}"
        print(row)
    print()


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #
def run_matrix(args: argparse.Namespace, catalog: dict[str, Any], models: list[str],
               datasets: list[str], config: dict[str, Any], scope: dict[str, Any],
               root: Path, manifest: dict[str, Any]) -> int:
    cells = [(m, d) for d in datasets for m in models]
    profiles = {m: resolve_profile(catalog[m], args, config, m) for m in models}
    force = args.force

    # Plan: which cells need collecting now.
    gpu_jobs = [(m, d) for (m, d) in cells
                if "collect" in args.phases and not (collection_complete(cell_dir(root, m, d), scope, d) and not force)]

    total_vram = query_total_vram_gb()
    budget = args.vram_budget_gb or (total_vram or 96.0) * args.vram_budget_frac
    max_parallel = resolve_max_parallel(args.max_parallel, len(gpu_jobs))
    cpu_workers = args.cpu_workers or min(4, (os.cpu_count() or 4))

    print(f"\n[matrix] {len(models)} models x {len(datasets)} datasets | phases: {', '.join(args.phases)}")
    print(f"[matrix] GPU VRAM total={total_vram:.0f}GB | budget={budget:.0f}GB | "
          f"max-parallel={max_parallel} | cpu-workers={cpu_workers}"
          if total_vram else
          f"[matrix] GPU VRAM unknown | budget={budget:.0f}GB | max-parallel={max_parallel} | cpu-workers={cpu_workers}")
    print(f"[matrix] git-checkpoint={args.git_checkpoint} push={args.git_push} | output: {root}")

    if args.dry_run:
        print("\n[matrix] DRY RUN -- planned collect schedule (footprints):")
        for (m, d) in gpu_jobs:
            fp = estimate_footprint_gb(profiles[m])
            print(f"  collect {m:<32} {d:<8} ~{fp:5.1f}GB  bs={profiles[m]['batch_size']:<3} {profiles[m]['quantization']}")
        if not gpu_jobs:
            print("  (no collect work outstanding)")
        print("  then: analyze+report per cell (CPU pool), aggregate per dataset (>=2 models).")
        return 0

    bar = make_bar(len(gpu_jobs), enabled=not args.no_progress)
    cpu_pool = ThreadPoolExecutor(max_workers=cpu_workers)
    futures = []
    running: dict[subprocess.Popen, tuple] = {}
    committed = 0.0

    def postfix() -> None:
        labels = [f"{m}/{d}" for (_, (m, d, *_)) in [(p, running[p]) for p in running]]
        bar.set_postfix_str(",".join(labels) if labels else "idle")

    def cpu_followup(model: str, dataset: str) -> None:
        cdir = cell_dir(root, model, dataset)
        if "analyze" in args.phases and not (analysis_complete(cdir) and not force):
            bar.write(f"  > analyze | {model} | {dataset}")
            code, el = run_blocking(analyze_cmd(cdir), cdir / "analyze.log")
            record(manifest, root, model, dataset, "analyze", code, el, analyze_cmd(cdir))
            bar.write(f"  {'OK' if code == 0 else 'XX'} analyze | {model} | {dataset} [{el:.1f}m]")
        if "report" in args.phases and not (report_complete(cdir) and not force):
            bar.write(f"  > report  | {model} | {dataset}")
            code, el = run_blocking(report_cmd(cdir), cdir / "report.log")
            record(manifest, root, model, dataset, "report", code, el, report_cmd(cdir))
            bar.write(f"  {'OK' if code == 0 else 'XX'} report  | {model} | {dataset} [{el:.1f}m]")
        if args.git_checkpoint:
            st = git_checkpoint([cdir, root / MANIFEST_NAME],
                                f"matrix checkpoint: {model} x {dataset}", args.git_push)
            bar.write(f"  git [{model} x {dataset}]: {st}")

    # Cells already collected (skip collect) but with CPU work outstanding -> start now.
    for (m, d) in cells:
        if (m, d) not in gpu_jobs and collection_complete(cell_dir(root, m, d), scope, d):
            futures.append(cpu_pool.submit(cpu_followup, m, d))

    # GPU admission loop.
    queue = deque(gpu_jobs)
    while queue or running:
        while queue and len(running) < max_parallel:
            m, d = queue[0]
            fp = estimate_footprint_gb(profiles[m])
            if running and committed + fp > budget:
                break  # wait for VRAM to free up
            queue.popleft()
            cdir = cell_dir(root, m, d)
            cdir.mkdir(parents=True, exist_ok=True)
            log = cdir / "collect.log"
            fh = log.open("w", encoding="utf-8", errors="replace")
            proc = subprocess.Popen(collect_cmd(m, d, cdir, profiles[m], scope),
                                    cwd=str(REPO_ROOT), stdout=fh, stderr=subprocess.STDOUT)
            running[proc] = (m, d, fp, time.monotonic(), log, fh)
            committed += fp
            bar.write(f"> collect | {m} | {d}  (bs={profiles[m]['batch_size']}, ~{fp:.0f}GB, "
                      f"in-flight={len(running)}, committed={committed:.0f}/{budget:.0f}GB)")
            postfix()
        done = [p for p in running if p.poll() is not None]
        for proc in done:
            m, d, fp, start, log, fh = running.pop(proc)
            fh.close()
            committed -= fp
            code = proc.returncode
            el = (time.monotonic() - start) / 60.0
            record(manifest, root, m, d, "collect", code, el, collect_cmd(m, d, cell_dir(root, m, d), profiles[m], scope))
            bar.update(1)
            if code == 0:
                bar.write(f"OK collect | {m} | {d} [{el:.1f}m]")
                futures.append(cpu_pool.submit(cpu_followup, m, d))
            else:
                bar.write(f"XX collect | {m} | {d} [{el:.1f}m] exit={code} -> {log}")
            postfix()
        if not done:
            time.sleep(1.0)

    for fut in futures:  # surface any exception raised inside a CPU task
        fut.result()
    cpu_pool.shutdown(wait=True)
    bar.close()

    # Aggregate per dataset (CPU, cheap) once all its collects are resolved.
    if "aggregate" in args.phases:
        for d in datasets:
            dirs = [cell_dir(root, m, d) for m in models if collection_complete(cell_dir(root, m, d), scope, d)]
            if len(dirs) < 2:
                print(f"[aggregate] {d} -> skip (need >=2 completed models, have {len(dirs)})")
                continue
            agg = root / "_aggregate" / d
            print(f"[aggregate] {d} <- {len(dirs)} models")
            code, el = run_blocking(aggregate_cmd(d, dirs, agg), agg / "aggregate.log")
            with _MANIFEST_LOCK:
                manifest["aggregate"][d] = {"status": "ok" if code == 0 else "failed",
                                            "return_code": code, "n_models": len(dirs)}
                save_manifest(root, manifest)
            print(f"  {'OK' if code == 0 else 'XX'} aggregate | {d} [{el:.1f}m]")
            if args.git_checkpoint:
                print(f"  git [aggregate {d}]: "
                      + git_checkpoint([agg, root / MANIFEST_NAME], f"matrix aggregate: {d}", args.git_push))

    print("\n[matrix] done.")
    print_status(root, models, datasets, manifest)
    failed = sum(1 for c in manifest["cells"].values()
                 for s in ("collect", "analyze", "report") if c.get(s, {}).get("status") == "failed")
    failed += sum(1 for a in manifest["aggregate"].values() if a.get("status") == "failed")
    return 1 if failed else 0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", help="Model aliases (default: all non-speculative).")
    parser.add_argument("--datasets", nargs="+", help=f"Datasets (default: {DEFAULT_DATASETS}).")
    parser.add_argument("--phases", nargs="+", choices=PHASES, default=list(PHASES))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--config", help="JSON: models / datasets / model_overrides / datasets_config.")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperatures", nargs="+")
    parser.add_argument("--seeds", nargs="+")
    parser.add_argument("--quantization", choices=["none", "8bit", "4bit"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-parallel", default="auto", help="Concurrent collect jobs: int or 'auto' (default).")
    parser.add_argument("--vram-budget-gb", type=float, help="Hard VRAM budget for concurrent collects.")
    parser.add_argument("--vram-budget-frac", type=float, default=0.88, help="Budget as fraction of total VRAM.")
    parser.add_argument("--cpu-workers", type=int, help="Parallel analyze/report workers (default min(4,cpus)).")
    parser.add_argument("--git-checkpoint", action="store_true", help="Commit light artifacts after each cell.")
    parser.add_argument("--git-push", action="store_true", help="Also push after each commit (implies checkpoint).")
    parser.add_argument("--no-progress", action="store_true", help="Disable the tqdm progress bar.")
    parser.add_argument("--smoke", action="store_true", help="Tiny scope for a fast end-to-end test.")
    parser.add_argument("--force", action="store_true", help="Redo cells even if already complete.")
    parser.add_argument("--status", action="store_true", help="Print the status grid and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without executing.")
    args = parser.parse_args()
    if args.git_push:
        args.git_checkpoint = True

    config = apply_config(args.config)
    catalog = parse_model_catalog(COLLECT_SCRIPT)
    models = resolve_models(args, catalog, config)
    datasets = resolve_datasets(args, config)
    root = Path(args.output_root)
    manifest = load_manifest(root)

    if args.status:
        print_status(root, models, datasets, manifest)
        return 0

    scope = build_scope(args)
    return run_matrix(args, catalog, models, datasets, config, scope, root, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
