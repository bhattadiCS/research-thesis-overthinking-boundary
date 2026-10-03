#!/usr/bin/env python3
"""Freeze and audit thesis data without loading models or changing source files.

The archived tournament manifest selects the corpus. A freeze refuses to silently
accept changed data, duplicate trajectory steps, missing cells, or a different
corpus. Raw byte hashes detect local changes; canonical LF hashes permit an
explicitly requested audit of a Git checkout with different text line endings.

Examples (from the repository root):
    python tools/freeze_research_data.py freeze
    python tools/freeze_research_data.py verify
    python tools/freeze_research_data.py verify --allow-line-ending-changes
    python tools/freeze_research_data.py environment

Only the named output is written by freeze. Existing outputs require --replace.
Environment capture describes the executing machine, never the original training
environment. All commands use Python's standard library and require no GPU.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import fnmatch
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import platform
import re
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
AUTHORITY = "research/outputs/experiments_v2/ultimate_tournament_manifest.json"
DATA_ROOT = "research/outputs/experiments_v2"
SCHEMA = "research-data-freeze-v1"
TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".py", ".sh", ".txt", ".log", ".yml", ".yaml", ".toml"}

# These are actual dependencies of the retained progress-report computations.
# Globs and fixed paths are stored in the manifest, so a later audit also detects
# added/removed evidence files. Rejected, experimental, or mutable checkpoints
# are not implicitly swept into this evidence set.
THESIS_FIXED = [
    ".gitattributes",
    "requirements-colab.txt",
    "research/outputs/experiment_matrix/matrix_manifest.json",
    "research/outputs/experiments_v2/ultimate_oof_predictions.npz",
    "research/outputs/experiments_v2/ultimate_tournament_results.csv",
    "research/outputs/experiments_v2/ultimate_tournament_results.log",
    "research/outputs/experiments_v2/ultimate_research_graph.json",
    "research/real_trace_experiments.py",
    "research/trace_analysis.py",
    "research/run_active_stopping.py",
    "research/run_ultimate_multi_day_tournament.py",
    "tools/run_global_52cell_sweep.sh",
    "tools/compute_progress_report_review_metrics.py",
    "tools/recompute_thesis_evidence.py",
    "tools/freeze_research_data.py",
    "tools/capture_repository_venv_environment.py",
    "requirements.repository-venv.lock.txt",
    "software_repository_venv_observed_v1.json",
    "research/tests/test_data_freeze.py",
    "research/adversarial_tasks_v1.jsonl",
    "research/adversarial_gold_v1.jsonl",
    "research/online_stopping_controller.py",
    "research/online_generation.py",
    "research/run_online_stopping_evaluation.py",
    "research/learned_online_stopping_controller.py",
    "research/run_learned_online_stopping.py",
    "research/analyze_online_stopping_results.py",
    "research/prefix_stopping_model.py",
    "research/train_prefix_stopping_model.py",
    "research/tests/test_learned_online_controller.py",
    "research/tests/test_prefix_stopping_model.py",
    "research/tests/test_live_uncertainty.py",
    "tools/analyze_live_stopping_uncertainty.py",
    "research/tests/test_online_controller.py",
    "research/tests/test_online_replay_accounting.py",
    "research/tests/test_graders.py",
    "research/tests/test_boundary_floor.py",
    "research/mathematical_foundations.md",
    "research/tests/test_mathematical_foundations.py",
    "research/classify_losses.py",
    "research/analyze_runs.py",
    "tools/summarize_failure_audit.py",
    "research/reports/data_freeze_verification_2026-10-02.md",
    "research/run_ultimate_blackwell_5day_tournament.py",
    "research/outputs/experiments_v2/blackwell_5day_tournament_v1/blackwell_tournament_report.json",
    "research/outputs/experiments_v2/blackwell_5day_tournament_v1/blackwell_tournament_report.md",
]
THESIS_GLOBS = [
    "research/outputs/experiments_v2/global_*/metadata.json",
    "research/outputs/experiments_v2/global_*/trace_runs.csv",
    "research/outputs/experiment_matrix/*/trace_steps.csv",
    "research/outputs/experiment_matrix/*/trace_runs.csv",
    "research/outputs/experiment_matrix/*/metadata.json",
    "research/outputs/experiment_matrix/*/detector_comparison_by_run.csv",
    "research/outputs/experiments_v2/algov2_cache/*.json",
    "research/outputs/experiments_v2/algov2_cache_n2_n3/*.json",
    "research/outputs/experiments_v2/p4b_*/*.csv",
    "research/outputs/experiments_v2/p4b_*/metadata.json",
    "research/outputs/experiments_v2/p8_*/*.csv",
    "research/outputs/experiments_v2/p8_*/metadata.json",
    "research/outputs/real_traces_bf16_ladder/qwen2p5_7b/*.csv",
    "research/outputs/real_traces_bf16_ladder/qwen2p5_7b/metadata.json",
    "research/outputs/progress_report_review_metrics/*.csv",
    "research/outputs/progress_report_review_metrics/*.json",
    "research/outputs/thesis_v1/evidence/*.csv",
    "research/outputs/thesis_v1/evidence/*.json",
    "research/reports/thesis_failure_audit_v1/*",
    "research/outputs/semester2/prefix_model_v1/**/*",
    "research/outputs/semester2/online_stopping_20261002/**/*",
    "research/outputs/semester2/adversarial_live_20261002/**/*",
]
RELEVANT_PACKAGES = [
    "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "torch",
    "lightgbm", "transformers", "accelerate", "datasets", "evaluate",
    "bitsandbytes", "sentencepiece", "safetensors", "optuna", "tqdm", "pytest",
]


class IntegrityError(ValueError):
    """A frozen source or declared corpus is inconsistent."""


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def checked_path(root: Path, relative: str) -> Path:
    """Resolve a portable relative path and reject escapes or ambiguous aliases."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise IntegrityError(f"Expected a nonempty POSIX relative path: {relative!r}")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or any(part in {"..", "."} for part in relative.split("/")) or ":" in relative:
        raise IntegrityError(f"Path escapes or is not portable: {relative!r}")
    if str(parsed) != relative:
        raise IntegrityError(f"Noncanonical relative path: {relative!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise IntegrityError(f"Path escapes repository: {relative!r}")
    return path


def file_record(root: Path, relative: str) -> dict[str, Any]:
    path = checked_path(root, relative)
    if not path.is_file():
        raise IntegrityError(f"Missing file: {relative}")
    raw_digest = hashlib.sha256()
    lf_digest = hashlib.sha256()
    raw_bytes = lf_bytes = 0
    text_file = path.suffix.lower() in TEXT_SUFFIXES or path.name == ".gitattributes"
    trailing_cr = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            raw_digest.update(block)
            raw_bytes += len(block)
            if text_file:
                block = trailing_cr + block
                trailing_cr = b"\r" if block.endswith(b"\r") else b""
                if trailing_cr:
                    block = block[:-1]
                normalized = block.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                lf_digest.update(normalized)
                lf_bytes += len(normalized)
    record = {"path": relative, "bytes": raw_bytes, "sha256": raw_digest.hexdigest()}
    if text_file:
        if trailing_cr:
            lf_digest.update(b"\n")
            lf_bytes += 1
        record.update({"canonical_lf_bytes": lf_bytes, "canonical_lf_sha256": lf_digest.hexdigest()})
    return record


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IntegrityError(f"Unable to read JSON {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise IntegrityError(f"Expected a JSON object in {path.name}")
    return value


def inspect_trace(path: Path) -> tuple[dict[str, Any], set[str], dict[str, str]]:
    """Validate every tournament row and every complete source-qualified path."""
    trajectories: dict[str, dict[str, Any]] = {}
    task_ids: set[str] = set()
    models: set[str] = set()
    benchmarks: set[str] = set()
    steps = Counter()
    labels = Counter()
    temperatures = Counter()
    seeds = Counter()
    row_count = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise IntegrityError(f"Duplicate CSV column names in {path.name}")
        required = {"run_id", "task_id", "step", "correct", "model_alias", "task_source"}
        if not required.issubset(fields):
            raise IntegrityError(f"Missing CSV fields {sorted(required - set(fields))} in {path}")
        for row in reader:
            row_count += 1
            if None in row or any(row.get(key) is None or not row[key].strip() for key in required):
                raise IntegrityError(f"Malformed required fields at record {row_count} in {path}")
            try:
                step = int(row["step"])
                label = int(row["correct"])
            except ValueError as error:
                raise IntegrityError(f"Invalid step/label at record {row_count} in {path}") from error
            if step <= 0 or label not in {0, 1}:
                raise IntegrityError(f"Nonpositive step or nonbinary label at record {row_count} in {path}")
            run_id, task_id = row["run_id"], row["task_id"]
            run = trajectories.setdefault(run_id, {"task_id": task_id, "steps": set()})
            if run["task_id"] != task_id:
                raise IntegrityError(f"Trajectory maps to multiple task IDs in {path}: {run_id}")
            if step in run["steps"]:
                raise IntegrityError(f"Duplicate trajectory step in {path}: {run_id}, {step}")
            run["steps"].add(step)
            models.add(row["model_alias"])
            benchmarks.add(row["task_source"])
            task_ids.add(task_id)
            steps[str(step)] += 1
            labels[str(label)] += 1
            temperatures[row.get("temperature", "")] += 1
            seeds[row.get("seed", "")] += 1
    if not row_count or len(models) != 1 or len(benchmarks) != 1:
        raise IntegrityError(f"Expected one nonempty model-benchmark cell: {path}")
    for run_id, run in trajectories.items():
        if run["steps"] != set(range(1, len(run["steps"]) + 1)):
            raise IntegrityError(f"Noncontiguous trajectory steps in {path}: {run_id}")
    summary = {
        "rows": row_count,
        "source_qualified_trajectories": len(trajectories),
        "task_ids": len(task_ids),
        "model_alias": next(iter(models)),
        "benchmark": next(iter(benchmarks)),
        "csv_columns": fields,
        "rows_by_step": dict(sorted(steps.items(), key=lambda pair: int(pair[0]))),
        "rows_by_correctness_label": dict(sorted(labels.items())),
        "sequence_length_distribution": dict(sorted(Counter(str(len(run["steps"])) for run in trajectories.values()).items())),
        "rows_by_temperature": dict(sorted(temperatures.items())),
        "rows_by_seed": dict(sorted(seeds.items())),
    }
    return summary, task_ids, {run_id: run["task_id"] for run_id, run in trajectories.items()}


def evidence_paths(root: Path, selection: dict[str, Any]) -> list[str]:
    selected = set()
    for relative in selection["fixed"]:
        checked_path(root, relative)
        selected.add(relative)
    for pattern in selection["globs"]:
        # Globs are also repository-relative and must not traverse out of scope.
        checked_path(root, pattern)
        selected.update(path.relative_to(root).as_posix() for path in root.glob(pattern) if path.is_file())
    for pattern in selection.get("exclude_globs", []):
        checked_path(root, pattern)
        selected = {relative for relative in selected if not fnmatch.fnmatchcase(relative, pattern)}
    return sorted(selected)


def glob_matches(relative: str, pattern: str) -> bool:
    """Match a file path as pathlib does, including zero-directory ``**``."""
    parts = PurePosixPath(relative).parts
    tokens = PurePosixPath(pattern).parts
    # A terminal ``**`` selects directories, not files, in Path.glob.
    if tokens and tokens[-1] == "**":
        return False

    @lru_cache(maxsize=None)
    def match(part: int, token: int) -> bool:
        if token == len(tokens):
            return part == len(parts)
        if tokens[token] == "**":
            return match(part, token + 1) or (part < len(parts) and match(part + 1, token))
        return part < len(parts) and fnmatch.fnmatch(parts[part], tokens[token]) and match(part + 1, token + 1)

    return match(0, 0)


def checked_freeze_output(root: Path, relative: str, manifest: dict[str, Any]) -> Path:
    """Refuse source overwrites and outputs that would change frozen membership."""
    output = checked_path(root, relative)
    records = [manifest["authority"], *manifest["files"], *manifest["auxiliary_evidence"]["files"]]
    if any(output == checked_path(root, record["path"]) for record in records):
        raise IntegrityError(f"Freeze output would overwrite a selected input: {relative}")
    if glob_matches(relative, f"{manifest['scope']['data_root']}/global_*/trace_steps.csv"):
        raise IntegrityError(f"Freeze output would change tournament file membership: {relative}")
    selection = manifest["auxiliary_evidence"]["selection"]
    included = relative in selection["fixed"] or any(glob_matches(relative, pattern) for pattern in selection["globs"])
    excluded = any(fnmatch.fnmatchcase(relative, pattern) for pattern in selection.get("exclude_globs", []))
    if included and not excluded:
        raise IntegrityError(f"Freeze output would enter its own evidence membership: {relative}")
    return output


def build_manifest(
    root: Path,
    *,
    authority_relative: str = AUTHORITY,
    data_relative: str = DATA_ROOT,
    selection: dict[str, Any] | None = None,
    created_utc: str | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    authority = read_object(checked_path(root, authority_relative))
    declared_files = authority.get("files")
    if not isinstance(declared_files, list) or not declared_files:
        raise IntegrityError("Authoritative manifest has no nonempty file list")
    declared_paths = [entry["path"] for entry in declared_files]
    if len(declared_paths) != len(set(declared_paths)):
        raise IntegrityError("Authoritative manifest contains duplicate paths")
    if declared_paths != sorted(declared_paths):
        raise IntegrityError("Authoritative file order is not canonical sorted order")
    data_path = checked_path(root, data_relative)
    discovered = sorted(path.relative_to(data_path).as_posix() for path in data_path.glob("global_*/trace_steps.csv"))
    if discovered != declared_paths:
        raise IntegrityError("Authoritative file membership differs from discovered global_* trace cells")

    records = []
    all_tasks: set[str] = set()
    run_cells: dict[str, set[str]] = defaultdict(set)
    sequence_lengths = Counter()
    step_rows = Counter()
    label_rows = Counter()
    benchmark_tasks: dict[str, set[str]] = defaultdict(set)
    benchmark_rows = Counter()
    benchmark_runs = Counter()
    model_rows = Counter()
    model_runs = Counter()
    cell_configs = []
    split_annotations = []
    archived_files = []
    for original in declared_files:
        relative = f"{data_relative}/{original['path']}"
        record = file_record(root, relative)
        archived_hash = original.get("canonical_lf_sha256", original.get("sha256"))
        if record["canonical_lf_sha256"] != archived_hash:
            raise IntegrityError(f"Content differs from archived tournament SHA256: {relative}")
        if "canonical_lf_sha256" not in original and record["canonical_lf_bytes"] != original.get("bytes"):
            raise IntegrityError(f"LF byte count differs from archived tournament: {relative}")
        summary, task_ids, run_tasks = inspect_trace(checked_path(root, relative))
        source_cell = PurePosixPath(original["path"]).parent.as_posix()
        if source_cell != f"global_{summary['model_alias']}_{summary['benchmark']}":
            raise IntegrityError(f"CSV model/benchmark disagrees with source cell name: {source_cell}")
        metadata_relative = f"{data_relative}/{source_cell}/metadata.json"
        metadata = read_object(checked_path(root, metadata_relative))
        if metadata.get("model", {}).get("alias") != summary["model_alias"] or metadata.get("task_source") != summary["benchmark"]:
            raise IntegrityError(f"CSV identity disagrees with metadata: {source_cell}")
        config_keys = [
            "model", "model_source", "backend", "device", "quantization", "attn_implementation",
            "max_steps", "max_new_tokens", "max_tasks", "task_source", "dataset_split",
            "dataset_shuffle_seed", "batch_size", "step_cost", "prompt_mode", "system_prompt_mode",
            "seeds", "temperatures", "completed_run_count", "pending_run_count",
        ]
        cell_configs.append({"source_cell": source_cell, "metadata_path": metadata_relative,
                             "recorded_configuration": {key: metadata[key] for key in config_keys if key in metadata}})
        notes = sorted({task.get("notes", "") for task in metadata.get("tasks", [])})
        configured_split = metadata.get("dataset_split")
        if summary["benchmark"] == "gpqa":
            split_annotations.append({"source_cell": source_cell, "configured_metadata_split": configured_split,
                                      "effective_split_interpretation": "train", "metadata_task_notes": notes,
                                      "evidence": "Retained real_trace_experiments.py load_gpqa_tasks unconditionally requests split='train', ignoring dataset_split. The original generation source revision is not recorded.",
                                      "inconsistency": configured_split != "train"})
        else:
            split_annotations.append({"source_cell": source_cell, "configured_metadata_split": configured_split,
                                      "effective_split_interpretation": configured_split, "metadata_task_notes": notes,
                                      "evidence": "Recorded dataset_split corroborated by the retained per-task notes; remote dataset revision is not recorded.",
                                      "inconsistency": False})
        record.update({"source_cell": source_cell, "archived_sha256": archived_hash,
                       "archived_bytes": original.get("bytes"),
                       "archived_match": "raw" if record["sha256"] == original.get("sha256") else "canonical_lf", **summary})
        records.append(record)
        archived_files.append({"path": original["path"], "bytes": record["canonical_lf_bytes"], "sha256": record["canonical_lf_sha256"]})
        all_tasks.update(task_ids)
        for run_id in run_tasks:
            run_cells[run_id].add(source_cell)
        sequence_lengths.update(summary["sequence_length_distribution"])
        step_rows.update(summary["rows_by_step"])
        label_rows.update(summary["rows_by_correctness_label"])
        benchmark_tasks[summary["benchmark"]].update(task_ids)
        benchmark_rows[summary["benchmark"]] += summary["rows"]
        benchmark_runs[summary["benchmark"]] += summary["source_qualified_trajectories"]
        model_rows[summary["model_alias"]] += summary["rows"]
        model_runs[summary["model_alias"]] += summary["source_qualified_trajectories"]

    corpus = {
        "selected_cell_count": len(records), "rows": sum(record["rows"] for record in records),
        "source_qualified_trajectories": sum(record["source_qualified_trajectories"] for record in records),
        "raw_run_ids": len(run_cells), "raw_run_id_cross_cell_collisions": sum(len(cells) > 1 for cells in run_cells.values()),
        "task_ids": len(all_tasks), "model_count": len(model_rows), "benchmark_count": len(benchmark_rows),
        "rows_by_step": dict(sorted(step_rows.items(), key=lambda pair: int(pair[0]))),
        "rows_by_correctness_label": dict(sorted(label_rows.items())),
        "sequence_length_distribution": dict(sorted(sequence_lengths.items())),
        "benchmarks": {name: {"rows": benchmark_rows[name], "trajectories": benchmark_runs[name], "task_ids": len(benchmark_tasks[name])} for name in sorted(benchmark_rows)},
        "models": {name: {"rows": model_rows[name], "trajectories": model_runs[name]} for name in sorted(model_rows)},
    }
    for key in ["selected_cell_count", "rows", "source_qualified_trajectories", "raw_run_ids", "raw_run_id_cross_cell_collisions", "task_ids", "sequence_length_distribution"]:
        if authority.get(key) != corpus[key]:
            raise IntegrityError(f"Observed corpus {key}={corpus[key]!r} disagrees with authoritative {authority.get(key)!r}")
    # Historical manifests hashed {path,bytes,sha256}; newer runners hash
    # {path,canonical_lf_sha256}. Store both formats and state their provenance.
    legacy_fingerprint = stable_hash(archived_files)
    canonical_runner_fingerprint = stable_hash([
        {"path": entry["path"], "canonical_lf_sha256": record["canonical_lf_sha256"]}
        for entry, record in zip(declared_files, records)
    ])
    canonical_authority = any("canonical_lf_sha256" in entry for entry in declared_files)
    expected_fingerprint = canonical_runner_fingerprint if canonical_authority else legacy_fingerprint
    if expected_fingerprint != authority.get("dataset_fingerprint"):
        kind = "canonical" if canonical_authority else "legacy"
        raise IntegrityError(f"Reconstructed {kind} dataset fingerprint differs from the authority")
    if "raw_dataset_fingerprint" in authority:
        raw_files = [{"path": entry["path"], "raw_sha256": entry.get("raw_sha256", entry.get("sha256"))}
                     for entry in declared_files]
        if stable_hash(raw_files) != authority["raw_dataset_fingerprint"]:
            raise IntegrityError("Reconstructed raw dataset fingerprint differs from the authority")
    selected = selection or {"profile": "tournament-only", "fixed": [config["metadata_path"] for config in cell_configs], "globs": []}
    auxiliary = [file_record(root, relative) for relative in evidence_paths(root, selected)]
    manifest = {
        "schema_version": SCHEMA,
        "created_utc": created_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": {"data_root": data_relative, "authority_manifest": authority_relative,
                  "selection": "exact archived manifest paths; no rejected/smoke or matrix cells in tournament totals",
                  "trajectory_key": "source_cell::run_id"},
        "hash_contract": {"strict": "SHA256 of exact local bytes plus byte length",
                          "portable_text": "replace CRLF and bare CR with LF; no other normalization",
                          "binary": "exact raw bytes only",
                          "claim_limit": "File identity and corpus structure; not model retraining, labels' semantic validity, or prospective effectiveness."},
        "authority": {**file_record(root, authority_relative), "recorded_dataset_fingerprint": authority.get("dataset_fingerprint"),
                      "reconstructed_legacy_dataset_fingerprint": legacy_fingerprint,
                      "canonical_runner_dataset_fingerprint": canonical_runner_fingerprint,
                      "recorded_feature_count": authority.get("feature_count"), "recorded_feature_fingerprint": authority.get("feature_fingerprint"),
                      "recorded_protocol": authority.get("protocol"), "recorded_preflight": authority.get("preflight")},
        "corpus": corpus, "files": records, "generation_configurations": cell_configs,
        "dataset_split_annotations": split_annotations,
        "auxiliary_evidence": {"selection": selected, "files": auxiliary,
                               "claim_limit": "Raw source/evidence fingerprints only; matrix analysis uses its separately documented sanitizer and inclusion rules."},
    }
    manifest["content_fingerprint"] = content_fingerprint(manifest)
    return manifest


def content_fingerprint(manifest: dict[str, Any]) -> str:
    """Protect all scope, provenance, structure, and hash declarations in the freeze."""
    return stable_hash({key: value for key, value in manifest.items() if key not in {"content_fingerprint", "created_utc"}})


def verify_manifest(root: Path, manifest: dict[str, Any], *, allow_line_ending_changes: bool = False) -> dict[str, Any]:
    if manifest.get("schema_version") != SCHEMA:
        raise IntegrityError("Unsupported data freeze schema")
    if manifest.get("content_fingerprint") != content_fingerprint(manifest):
        raise IntegrityError("Manifest content fingerprint is inconsistent")
    scope = manifest["scope"]
    actual = build_manifest(root, authority_relative=scope["authority_manifest"], data_relative=scope["data_root"],
                            selection=manifest["auxiliary_evidence"]["selection"], created_utc=manifest["created_utc"])
    if actual["corpus"] != manifest["corpus"]:
        raise IntegrityError("Observed corpus structure differs from freeze")
    mismatches = []
    newline_changes = []
    groups = [("authority", [manifest["authority"]], [actual["authority"]]),
              ("data", manifest["files"], actual["files"]),
              ("auxiliary", manifest["auxiliary_evidence"]["files"], actual["auxiliary_evidence"]["files"])]
    for label, old_records, new_records in groups:
        old = {item["path"]: item for item in old_records}
        new = {item["path"]: item for item in new_records}
        if len(old) != len(old_records) or old.keys() != new.keys():
            raise IntegrityError(f"{label} file membership differs from freeze")
        for relative, previous in old.items():
            current = new[relative]
            if (current["sha256"], current["bytes"]) != (previous["sha256"], previous["bytes"]):
                same_lf = previous.get("canonical_lf_sha256") is not None and (current.get("canonical_lf_sha256"), current.get("canonical_lf_bytes")) == (previous.get("canonical_lf_sha256"), previous.get("canonical_lf_bytes"))
                if allow_line_ending_changes and same_lf:
                    newline_changes.append(relative)
                else:
                    mismatches.append(relative)
    if mismatches:
        raise IntegrityError(f"Changed file bytes: {', '.join(mismatches[:8])}")
    # Byte matches alone do not justify silently accepting a forged metadata
    # summary or hash declaration. Compare all rebuilt declarations after removing
    # only the raw fields an explicitly portable audit is permitted to differ on.
    def comparable(value: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(value)
        # Only the freeze's own timestamp/fingerprint are comparison metadata.
        # Identically named fields nested in source provenance remain protected.
        result.pop("content_fingerprint", None)
        result.pop("created_utc", None)
        if allow_line_ending_changes:
            records = [result["authority"], *result["files"], *result["auxiliary_evidence"]["files"]]
            for record in records:
                if "canonical_lf_sha256" in record:
                    for key in ("bytes", "sha256", "archived_match"):
                        record.pop(key, None)
        return result
    if comparable(actual) != comparable(manifest):
        raise IntegrityError("Rebuilt scope, provenance, hashes, or structure differs from freeze")
    return {"status": "verified", "audit_mode": "canonical_lf" if allow_line_ending_changes else "strict_bytes",
            "data_files": len(manifest["files"]), "auxiliary_files": len(manifest["auxiliary_evidence"]["files"]),
            "rows": manifest["corpus"]["rows"], "trajectories": manifest["corpus"]["source_qualified_trajectories"],
            "line_ending_only_changes": newline_changes, "content_fingerprint": manifest["content_fingerprint"]}


def write_output(path: Path, content: str, *, replace: bool) -> None:
    if path.exists() and not replace:
        raise IntegrityError(f"Output exists; choose a new path or explicitly use --replace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing user files by default.
    with path.open("w" if replace else "x", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def capture_environment(root: Path, *, replace: bool) -> dict[str, Any]:
    """Record package versions and their actual evidence contexts separately."""
    installed = {}
    unknown_sources = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise IntegrityError(f"Distribution name cannot be safely pinned: {name!r}")
        key = re.sub(r"[-_.]+", "-", name).lower()
        previous = installed.get(key)
        if previous is not None and previous != distribution.version:
            raise IntegrityError(f"Conflicting installed distribution versions: {name}")
        installed[key] = distribution.version
        if distribution.read_text("direct_url.json") is not None:
            unknown_sources.append(key)
    relevant = {name: installed.get(re.sub(r"[-_.]+", "-", name).lower()) for name in RELEVANT_PACKAGES}
    authority = read_object(checked_path(root, AUTHORITY))
    archived_contexts = []
    for path in sorted(checked_path(root, DATA_ROOT).glob("**/*manifest.json")):
        archived = read_object(path)
        if "runtime_contract" in archived:
            relative = path.relative_to(root).as_posix()
            archived_contexts.append({"manifest": file_record(root, relative),
                                      "schema_version": archived.get("schema_version"),
                                      "recorded_versions": archived["runtime_contract"]})
    provenance = {
        "schema_version": "research-software-provenance-v1",
        "captured_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "current_local_environment": {"purpose": "Current audit/reanalysis environment only; not evidence of original training versions.",
                                      "python": platform.python_version(), "python_implementation": platform.python_implementation(),
                                      "executable": sys.executable, "platform": platform.platform(),
                                      "relevant_packages": relevant, "installed_distribution_count": len(installed),
                                      "lock_file": "requirements.local-environment.lock.txt",
                                      "roadmap_lock_alias": "requirements.lock.txt",
                                      "distributions_with_direct_install_metadata": sorted(set(unknown_sources)),
                                      "limitations": "Version inventory; package indexes, wheel hashes, external system libraries and direct-install source URLs are not reconstructed."},
        "archived_tournament_environment": {"source": file_record(root, AUTHORITY), "recorded_preflight": authority.get("preflight"),
                                            "partial_lock_file": "requirements.historical-tournament.partial.txt",
                                            "unknown": ["Python version", "NumPy", "pandas", "SciPy", "scikit-learn", "LightGBM", "Transformers", "generation model and dataset revision hashes", "exact package wheel sources/hashes"],
                                            "limitations": "Only recorded values are historical evidence; a current pip/package inventory cannot reconstruct this environment."},
        "archived_later_analysis_environments": archived_contexts,
        "repository_generation_requirements": {"source": file_record(root, "requirements-colab.txt"),
                                                "declared_constraints": checked_path(root, "requirements-colab.txt").read_text(encoding="utf-8").splitlines(),
                                                "interpretation": "Declared repository constraints, not an observed original experiment lock; most versions are lower bounds."},
    }
    local_lock = "# Observed CURRENT local environment, not the original experiment environment.\n"
    provenance["separately_observed_current_environments"] = []
    if checked_path(root, "software_repository_venv_observed_v1.json").is_file():
        provenance["separately_observed_current_environments"].append({
            "purpose": "Repository .venv used for the new failure-classification command; base audit, live generation, and prefix training use the primary observed environment.",
            "provenance": file_record(root, "software_repository_venv_observed_v1.json"),
            "lock": file_record(root, "requirements.repository-venv.lock.txt"),
        })
    local_lock += f"# Python {platform.python_version()}; {platform.platform()}\n"
    local_lock += "# Inventory only: exact wheel indexes/hashes and direct-install sources remain unspecified.\n"
    local_lock += "# This is the observed workstation inventory, not a clean minimal or historical regeneration environment.\n"
    local_lock += "# See software_provenance_v1.json before using CUDA or inference packages.\n"
    local_lock += "".join(f"{name}=={version}\n" for name, version in sorted(installed.items()))
    preflight = authority.get("preflight", {})
    historical = "# PARTIAL historical tournament version evidence; not a complete installable training lock.\n"
    historical += f"# Source: {AUTHORITY}\n# Recorded CUDA runtime: {preflight.get('cuda_runtime', 'unknown')}\n"
    historical += "# Original Python and all unrecorded package versions/wheel sources are unknown.\n"
    if preflight.get("torch_version"):
        historical += f"torch=={preflight['torch_version']}\n"
    outputs = {"requirements.local-environment.lock.txt": local_lock,
               "requirements.lock.txt": local_lock,
               "requirements.historical-tournament.partial.txt": historical,
               "software_provenance_v1.json": json.dumps(provenance, indent=2, sort_keys=True) + "\n"}
    # Refuse the whole operation before writing any outputs when one exists.
    for relative in outputs:
        if checked_path(root, relative).exists() and not replace:
            raise IntegrityError(f"Environment output exists; use --replace explicitly: {relative}")
    for relative, content in outputs.items():
        write_output(checked_path(root, relative), content, replace=replace)
    return {"status": "captured", "python": platform.python_version(), "installed_distribution_count": len(installed),
            "outputs": list(outputs), "historical_lock_complete": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPOSITORY)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="Validate the archived corpus and write one immutable-by-default fingerprint")
    freeze.add_argument("--output", default="data_manifest_v1.json")
    freeze.add_argument("--profile", choices=["thesis", "tournament"], default="thesis")
    freeze.add_argument("--extra-evidence", action="append", default=[], help="Additional repository-relative file or glob to freeze")
    freeze.add_argument("--replace", action="store_true", help="Explicitly refresh the named freeze output")
    verify = subparsers.add_parser("verify", help="Read-only exhaustive hash, file-membership, and row-structure audit")
    verify.add_argument("--manifest", default="data_manifest_v1.json")
    verify.add_argument("--allow-line-ending-changes", action="store_true", help="Accept only exact CRLF/CR-to-LF equivalent text changes")
    environment = subparsers.add_parser("environment", help="Capture current local versions and separately cite historical evidence")
    environment.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command == "freeze":
            selection = None
            if args.profile == "thesis":
                selection = {"profile": "thesis-evidence-v1", "fixed": THESIS_FIXED.copy(), "globs": THESIS_GLOBS.copy(),
                             "exclude_globs": ["**/__pycache__/**", "**/*.pyc"]}
                for relative in ["requirements.lock.txt", "requirements.local-environment.lock.txt", "requirements.historical-tournament.partial.txt", "software_provenance_v1.json"]:
                    if checked_path(root, relative).is_file():
                        selection["fixed"].append(relative)
            if args.extra_evidence:
                if selection is None:
                    raise IntegrityError("Additional evidence requires the thesis profile")
                for relative in args.extra_evidence:
                    selection["globs" if any(token in relative for token in "*?[") else "fixed"].append(relative)
            manifest = build_manifest(root, selection=selection)
            write_output(checked_freeze_output(root, args.output, manifest), json.dumps(manifest, indent=2, sort_keys=True) + "\n", replace=args.replace)
            result = {"status": "frozen", "output": args.output, "data_files": len(manifest["files"]),
                      "auxiliary_files": len(manifest["auxiliary_evidence"]["files"]), "corpus": manifest["corpus"],
                      "content_fingerprint": manifest["content_fingerprint"]}
        elif args.command == "verify":
            manifest = read_object(checked_path(root, args.manifest))
            result = verify_manifest(root, manifest, allow_line_ending_changes=args.allow_line_ending_changes)
        else:
            result = capture_environment(root, replace=args.replace)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (IntegrityError, KeyError, TypeError, OSError, csv.Error) as error:
        print(json.dumps({"status": "invalid", "error": str(error)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
