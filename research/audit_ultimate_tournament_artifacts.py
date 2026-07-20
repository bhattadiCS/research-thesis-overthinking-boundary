#!/usr/bin/env python3
"""Read-only integrity audit for ultimate tournament artifacts.

The tournament writes several durable artifacts into one directory.  A new run can
legitimately replace its manifest and status before it replaces results, telemetry,
or a checkpoint from an older smoke run.  Reading the result table alone in that
state is unsafe: it may describe a different corpus and run fingerprint.

This utility never writes to the artifact directory.  It cross-checks the current
manifest, status, fold summaries, telemetry SQLite ledger, result corpus metadata,
OOF archive shapes, and research graph.  Checkpoints are intentionally not
deserialized because ``.pth`` is pickle-backed; their timestamps are still audited.

Exit statuses:
  0  COMPLETE: a terminal, internally coherent tournament is present.
 10  INCOMPLETE: the current run is coherent but has not completed all folds.
 20  MIXED_OR_STALE: artifacts point at different runs or a stale smoke artifact.
 30  INVALID: required manifest/status metadata is missing or malformed.

Use ``--json`` for a machine-readable report on stdout.  In either mode the tool is
strictly read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXIT_COMPLETE = 0
EXIT_INCOMPLETE = 10
EXIT_MIXED_OR_STALE = 20
EXIT_INVALID = 30

AUDIT_VERSION = 2
STALE_TOLERANCE_SECONDS = 2.0
MAX_LOG_READ_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class Finding:
    """One compact, serializable observation made by the auditor."""

    severity: str
    category: str
    code: str
    message: str
    evidence: dict[str, Any]


class Findings:
    """Collect findings while preserving the category used for exit status."""

    def __init__(self) -> None:
        self.items: list[Finding] = []

    def add(self, category: str, code: str, message: str, **evidence: Any) -> None:
        severity = {
            "invalid": "error",
            "mixed": "error",
            "incomplete": "warning",
            "warning": "warning",
            "info": "info",
        }.get(category, "warning")
        self.items.append(Finding(severity, category, code, message, evidence))

    def has_category(self, category: str) -> bool:
        return any(item.category == category for item in self.items)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self.items]


def utc_timestamp(value: Any) -> str | None:
    """Format an epoch timestamp without letting malformed metadata crash an audit."""

    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not (0.0 < number < 32_503_680_000.0):  # Before year 3000, after epoch.
        return None
    return datetime.fromtimestamp(number, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def valid_epoch(value: Any) -> float | None:
    """Return a plausible epoch timestamp, or None for metadata that is not one."""

    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if utc_timestamp(number) is None:
        return None
    return number


def as_nonempty_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def as_positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_hash(value: Any) -> str:
    """Match the runner's stable JSON hashing convention for JSON-compatible values."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256_bytes(encoded)


def normalize_newlines(data: bytes) -> bytes:
    """Make CRLF, LF, and legacy CR manifest encodings comparable."""

    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def line_ending_counts(data: bytes) -> dict[str, int]:
    crlf = data.count(b"\r\n")
    without_crlf = data.replace(b"\r\n", b"")
    return {
        "crlf": crlf,
        "lf": without_crlf.count(b"\n"),
        "bare_cr": without_crlf.count(b"\r"),
    }


def relative_display(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def file_info(path: Path, root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"path": relative_display(path, root), "exists": path.is_file()}
    if not path.is_file():
        return info
    stat = path.stat()
    info.update(
        {
            "bytes": stat.st_size,
            "mtime_epoch": stat.st_mtime,
            "mtime_utc": utc_timestamp(stat.st_mtime),
        }
    )
    return info


def read_json_object(path: Path, root: Path, findings: Findings, *, required: bool, label: str) -> dict[str, Any] | None:
    if not path.is_file():
        category = "invalid" if required else "warning"
        findings.add(category, f"{label.upper()}_MISSING", f"{label} is not present.", path=relative_display(path, root))
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        category = "invalid" if required else "mixed"
        findings.add(
            category,
            f"{label.upper()}_UNREADABLE",
            f"{label} could not be parsed as UTF-8 JSON.",
            path=relative_display(path, root),
            error=str(error),
        )
        return None
    if not isinstance(value, dict):
        category = "invalid" if required else "mixed"
        findings.add(
            category,
            f"{label.upper()}_NOT_OBJECT",
            f"{label} must be a JSON object.",
            path=relative_display(path, root),
            actual_type=type(value).__name__,
        )
        return None
    return value


def read_limited_text(path: Path, limit: int = MAX_LOG_READ_BYTES) -> tuple[str | None, bool, str | None]:
    """Read enough of a log to identify its run without loading a multi-day log wholesale."""

    try:
        with path.open("rb") as handle:
            data = handle.read(limit + 1)
    except OSError as error:
        return None, False, str(error)
    truncated = len(data) > limit
    if truncated:
        data = data[:limit]
    return data.decode("utf-8", errors="replace"), truncated, None


def parse_corpus_from_results_log(text: str) -> dict[str, int] | None:
    match = re.search(
        r"Corpus:\s*(?P<cells>\d+)\s+cells,\s*(?P<trajectories>\d+)\s+source-qualified\s+trajectories,\s*"
        r"(?P<tasks>\d+)\s+task\s+groups\.",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return {key: int(value) for key, value in match.groupdict().items()}


def parse_corpus_from_runtime_log(text: str) -> dict[str, int] | None:
    match = re.search(
        r"Loaded\s+(?P<cells>\d+)\s+cells,\s+(?P<trajectories>\d+)\s+source-qualified\s+trajectories,\s*"
        r"(?P<rows>\d+)\s+rows,\s+(?P<tasks>\d+)\s+task\s+groups",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return {key: int(value) for key, value in match.groupdict().items()}


def parse_runtime_fold_count(text: str) -> int | None:
    match = re.search(r"Starting\s+(?P<folds>\d+)-fold\s+task-grouped\s+tournament", text, flags=re.IGNORECASE)
    return int(match.group("folds")) if match else None


def corpus_from_manifest(manifest: dict[str, Any], findings: Findings) -> dict[str, int | None]:
    fields = {
        "cells": "selected_cell_count",
        "trajectories": "source_qualified_trajectories",
        "rows": "rows",
        "tasks": "task_ids",
    }
    corpus: dict[str, int | None] = {}
    for short_name, manifest_name in fields.items():
        value = as_positive_int(manifest.get(manifest_name))
        corpus[short_name] = value
        if value is None:
            findings.add(
                "invalid",
                "MANIFEST_CORPUS_FIELD_INVALID",
                "Manifest corpus metadata must contain positive integers.",
                field=manifest_name,
                value=manifest.get(manifest_name),
            )
    return corpus


def compare_corpus(
    actual: dict[str, Any] | None,
    expected: dict[str, int | None],
    findings: Findings,
    *,
    source: str,
) -> bool:
    """Return True only when every supplied corpus dimension matches the manifest."""

    if actual is None:
        return False
    comparable = {key: value for key, value in actual.items() if key in expected and expected[key] is not None}
    mismatches = {
        key: {"artifact": value, "manifest": expected[key]}
        for key, value in comparable.items()
        if value != expected[key]
    }
    if mismatches:
        findings.add(
            "mixed",
            "CORPUS_MISMATCH",
            f"{source} describes a corpus different from the current manifest.",
            source=source,
            mismatches=mismatches,
        )
        return False
    return bool(comparable)


def inspect_sqlite(path: Path, root: Path, findings: Findings) -> dict[str, Any]:
    """Read telemetry with SQLite immutable/read-only semantics and never create sidecars."""

    report = file_info(path, root)
    report["read_only"] = True
    if not path.is_file():
        return report
    wal_path = path.with_name(path.name + "-wal")
    shm_path = path.with_name(path.name + "-shm")
    report["sidecars"] = {"wal": file_info(wal_path, root), "shm": file_info(shm_path, root)}
    if wal_path.is_file() and wal_path.stat().st_size > 0:
        findings.add(
            "warning",
            "TELEMETRY_WAL_NOT_CONSUMED",
            "Telemetry has a non-empty WAL sidecar; immutable read-only mode may not include uncheckpointed records.",
            path=relative_display(wal_path, root),
            bytes=wal_path.stat().st_size,
        )
    try:
        # ``as_uri`` correctly quotes spaces on both Windows and POSIX.  ``mode=ro``
        # prevents DB creation; ``immutable=1`` additionally prevents SQLite from
        # creating a lock, -journal, or -shm sidecar during the audit.  That makes
        # the audit safe even when pointed at a user-owned results directory.
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
        try:
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            report["tables"] = sorted(tables)
            run_meta: dict[str, Any] = {}
            if "run_meta" in tables:
                for key, raw_value in connection.execute("SELECT key, value FROM run_meta"):
                    try:
                        run_meta[str(key)] = json.loads(raw_value)
                    except (TypeError, json.JSONDecodeError):
                        run_meta[str(key)] = raw_value
            else:
                findings.add(
                    "mixed",
                    "TELEMETRY_SCHEMA_MISSING_RUN_META",
                    "Telemetry database lacks the run_meta table needed to bind it to a run fingerprint.",
                    path=relative_display(path, root),
                )
            report["run_fingerprint"] = run_meta.get("run_fingerprint")
            telemetry_manifest = run_meta.get("manifest")
            if isinstance(telemetry_manifest, dict):
                report["manifest_corpus"] = {
                    key: telemetry_manifest.get(source)
                    for key, source in {
                        "cells": "selected_cell_count",
                        "trajectories": "source_qualified_trajectories",
                        "rows": "rows",
                        "tasks": "task_ids",
                    }.items()
                }
                report["manifest_dataset_fingerprint"] = telemetry_manifest.get("dataset_fingerprint")
            if "folds" in tables:
                rows = list(connection.execute("SELECT fold, status, payload, ts FROM folds ORDER BY fold"))
                report["folds"] = [
                    {"fold": int(row[0]), "status": str(row[1]), "timestamp": float(row[3])}
                    for row in rows
                ]
            if "events" in tables:
                completion_events = int(
                    connection.execute("SELECT COUNT(*) FROM events WHERE kind = 'tournament_complete'").fetchone()[0]
                )
                report["tournament_complete_events"] = completion_events
            report["readable"] = True
        finally:
            connection.close()
    except (OSError, sqlite3.Error, ValueError) as error:
        report["readable"] = False
        report["error"] = str(error)
        findings.add(
            "mixed",
            "TELEMETRY_UNREADABLE",
            "Telemetry SQLite artifact could not be opened in read-only mode.",
            path=relative_display(path, root),
            error=str(error),
        )
    return report


def inspect_npz(path: Path, root: Path, findings: Findings) -> dict[str, Any]:
    """Inspect OOF array names/shapes without enabling pickle loading."""

    report = file_info(path, root)
    report["pickle_loading"] = False
    if not path.is_file():
        return report
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
        report["members"] = members
    except (OSError, zipfile.BadZipFile) as error:
        report["readable"] = False
        report["error"] = str(error)
        findings.add(
            "mixed",
            "OOF_ARCHIVE_UNREADABLE",
            "OOF prediction archive is not a readable NPZ/ZIP file.",
            path=relative_display(path, root),
            error=str(error),
        )
        return report
    try:
        import numpy as np

        shapes: dict[str, list[int]] = {}
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                shapes[name] = [int(dimension) for dimension in archive[name].shape]
        report["arrays"] = shapes
        report["readable"] = True
    except ImportError:
        # The rest of the auditor has no third-party dependency.  The zip member
        # list is still useful when numpy is unavailable, but shapes cannot be
        # validated in that environment.
        report["readable"] = True
        report["shape_validation"] = "skipped: numpy unavailable"
        findings.add(
            "warning",
            "OOF_SHAPE_VALIDATION_SKIPPED",
            "numpy is unavailable, so OOF array dimensions could not be validated.",
            path=relative_display(path, root),
        )
    except Exception as error:  # numpy can raise several format-specific errors.
        report["readable"] = False
        report["error"] = str(error)
        findings.add(
            "mixed",
            "OOF_ARRAY_UNREADABLE",
            "OOF prediction arrays could not be safely read with allow_pickle=False.",
            path=relative_display(path, root),
            error=str(error),
        )
    return report


def max_sequence_length(manifest: dict[str, Any]) -> int | None:
    distribution = manifest.get("sequence_length_distribution")
    if not isinstance(distribution, dict) or not distribution:
        return None
    lengths: list[int] = []
    for key in distribution:
        try:
            length = int(key)
        except (TypeError, ValueError):
            return None
        if length <= 0:
            return None
        lengths.append(length)
    return max(lengths) if lengths else None


def inspect_research_graph(path: Path, root: Path, findings: Findings) -> dict[str, Any]:
    report = file_info(path, root)
    if not path.is_file():
        return report
    graph = read_json_object(path, root, findings, required=False, label="research_graph")
    if graph is None:
        return report
    entities = graph.get("entities")
    if not isinstance(entities, list):
        findings.add(
            "mixed",
            "RESEARCH_GRAPH_ENTITIES_INVALID",
            "Research graph has no entities list.",
            path=relative_display(path, root),
        )
        return report
    corpora = [entity for entity in entities if isinstance(entity, dict) and entity.get("type") == "TrajectoryCorpus"]
    report["trajectory_corpus_count"] = len(corpora)
    if len(corpora) != 1:
        findings.add(
            "mixed",
            "RESEARCH_GRAPH_CORPUS_AMBIGUOUS",
            "Research graph must contain exactly one TrajectoryCorpus entity.",
            path=relative_display(path, root),
            count=len(corpora),
        )
        return report
    corpus = corpora[0]
    observations = corpus.get("observations")
    report["dataset_entity_id"] = corpus.get("id")
    if isinstance(observations, dict):
        report["corpus"] = {
            "cells": observations.get("cells"),
            "trajectories": observations.get("trajectories"),
            "tasks": observations.get("task_ids"),
        }
    return report


def artifact_predates_run(path: Path, started_at: float | None) -> bool:
    if started_at is None or not path.is_file():
        return False
    return path.stat().st_mtime < started_at - STALE_TOLERANCE_SECONDS


def audit_artifacts(output_dir: Path) -> dict[str, Any]:
    """Perform a full read-only cross-artifact audit and return JSON-safe data."""

    root = output_dir.resolve()
    findings = Findings()
    report: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "read_only": True,
        "output_dir": str(root),
    }
    if not root.is_dir():
        findings.add("invalid", "OUTPUT_DIRECTORY_MISSING", "Artifact directory does not exist.", output_dir=str(root))
        report["findings"] = findings.as_dicts()
        report["assessment"] = "invalid"
        report["exit_code"] = EXIT_INVALID
        return report

    manifest_path = root / "ultimate_tournament_manifest.json"
    status_path = root / "ultimate_tournament_status.json"
    manifest = read_json_object(manifest_path, root, findings, required=True, label="manifest")
    status = read_json_object(status_path, root, findings, required=True, label="status")
    report["manifest_file"] = file_info(manifest_path, root)
    report["status_file"] = file_info(status_path, root)

    manifest_fingerprint: str | None = None
    expected_corpus: dict[str, int | None] = {"cells": None, "trajectories": None, "rows": None, "tasks": None}
    expected_folds: int | None = None
    expected_length: int | None = None
    if manifest is not None:
        raw_manifest = manifest_path.read_bytes()
        files_value = manifest.get("files")
        manifest_hashes: dict[str, Any] = {
            "raw_sha256": sha256_bytes(raw_manifest),
            "line_ending_normalized_sha256": sha256_bytes(normalize_newlines(raw_manifest)),
            "canonical_json_sha256": canonical_json_hash(manifest),
            "line_endings": line_ending_counts(raw_manifest),
        }
        if isinstance(files_value, list):
            # v2 runners fingerprint path + canonical-LF file hash so a Windows
            # CRLF checkout can resume a Linux-created run.  Historical manifests
            # fingerprinted the full raw file records instead; retain that audit
            # path for backward compatibility.
            canonical_inventory: list[dict[str, Any]] = []
            uses_canonical_lf = all(
                isinstance(item, dict) and isinstance(item.get("canonical_lf_sha256"), str)
                for item in files_value
            )
            if uses_canonical_lf:
                canonical_inventory = [
                    {"path": item.get("path"), "canonical_lf_sha256": item.get("canonical_lf_sha256")}
                    for item in files_value
                ]
            else:
                canonical_inventory = files_value
            manifest_hashes["files_canonical_sha256"] = canonical_json_hash(canonical_inventory)
            manifest_hashes["uses_canonical_lf_inventory"] = uses_canonical_lf
            manifest_hashes["dataset_fingerprint_matches_files"] = (
                manifest.get("dataset_fingerprint") == manifest_hashes["files_canonical_sha256"]
            )
            if not manifest_hashes["dataset_fingerprint_matches_files"]:
                findings.add(
                    "mixed",
                    "MANIFEST_DATASET_FINGERPRINT_MISMATCH",
                    "Manifest dataset_fingerprint does not match the canonical hash of its file inventory.",
                    manifest_dataset_fingerprint=manifest.get("dataset_fingerprint"),
                    files_canonical_sha256=manifest_hashes["files_canonical_sha256"],
                )
        else:
            findings.add(
                "invalid",
                "MANIFEST_FILES_INVALID",
                "Manifest files inventory must be a list.",
                actual_type=type(files_value).__name__,
            )
        manifest_fingerprint = as_nonempty_string(manifest.get("run_fingerprint"))
        if manifest_fingerprint is None:
            findings.add("invalid", "MANIFEST_RUN_FINGERPRINT_MISSING", "Manifest has no non-empty run_fingerprint.")
        expected_corpus = corpus_from_manifest(manifest, findings)
        protocol = manifest.get("protocol")
        if not isinstance(protocol, dict):
            findings.add("invalid", "MANIFEST_PROTOCOL_INVALID", "Manifest protocol must be a JSON object.")
        else:
            expected_folds = as_positive_int(protocol.get("outer_folds"))
            if expected_folds is None:
                findings.add(
                    "invalid",
                    "MANIFEST_OUTER_FOLDS_INVALID",
                    "Manifest protocol.outer_folds must be a positive integer.",
                    value=protocol.get("outer_folds"),
                )
        expected_length = max_sequence_length(manifest)
        report["manifest"] = {
            "run_fingerprint": manifest_fingerprint,
            "dataset_fingerprint": manifest.get("dataset_fingerprint"),
            "corpus": expected_corpus,
            "expected_outer_folds": expected_folds,
            "max_sequence_length": expected_length,
            "hashes": manifest_hashes,
        }

    status_fingerprint: str | None = None
    status_complete: bool | None = None
    status_name: str | None = None
    started_at: float | None = None
    status_fold_numbers: list[int] | None = None
    if status is not None:
        status_fingerprint = as_nonempty_string(status.get("fingerprint"))
        if status_fingerprint is None:
            findings.add("invalid", "STATUS_FINGERPRINT_MISSING", "Status has no non-empty fingerprint.")
        candidate_complete = status.get("complete")
        if not isinstance(candidate_complete, bool):
            findings.add(
                "invalid",
                "STATUS_COMPLETE_INVALID",
                "Status complete field must be a boolean.",
                value=candidate_complete,
            )
        else:
            status_complete = candidate_complete
        status_name = as_nonempty_string(status.get("status"))
        if status_name is None:
            findings.add("invalid", "STATUS_NAME_MISSING", "Status has no non-empty status string.")
        started_at = valid_epoch(status.get("started_at"))
        if status.get("started_at") is not None and started_at is None:
            findings.add(
                "invalid",
                "STATUS_STARTED_AT_INVALID",
                "Status started_at is not a plausible epoch timestamp.",
                value=status.get("started_at"),
            )
        if status_complete is True and status_name != "complete":
            findings.add(
                "mixed",
                "STATUS_TERMINAL_CONTRADICTION",
                "Status complete=true but textual status is not 'complete'.",
                status=status_name,
            )
        if status_name == "complete" and status_complete is not True:
            findings.add(
                "mixed",
                "STATUS_TERMINAL_CONTRADICTION",
                "Textual status is 'complete' but complete is not true.",
                complete=status_complete,
            )
        raw_completed = status.get("completed_folds")
        if raw_completed is not None:
            if not isinstance(raw_completed, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in raw_completed):
                findings.add(
                    "invalid",
                    "STATUS_COMPLETED_FOLDS_INVALID",
                    "Status completed_folds must be a list of zero-based integer fold indexes.",
                    value=raw_completed,
                )
            else:
                status_fold_numbers = [item + 1 for item in raw_completed]
                if len(set(status_fold_numbers)) != len(status_fold_numbers):
                    findings.add(
                        "mixed",
                        "STATUS_COMPLETED_FOLDS_DUPLICATE",
                        "Status completed_folds contains duplicate entries.",
                        completed_folds=raw_completed,
                    )
                if expected_folds is not None and any(number < 1 or number > expected_folds for number in status_fold_numbers):
                    findings.add(
                        "mixed",
                        "STATUS_COMPLETED_FOLDS_OUT_OF_RANGE",
                        "Status completed_folds contains an index outside the manifest fold count.",
                        completed_folds=raw_completed,
                        expected_folds=expected_folds,
                    )
        report["status"] = {
            "status": status_name,
            "complete": status_complete,
            "fingerprint": status_fingerprint,
            "started_at": started_at,
            "started_at_utc": utc_timestamp(started_at),
            "completed_fold_numbers": status_fold_numbers,
        }

    fingerprint_sources: dict[str, str] = {
        key: value
        for key, value in {"manifest": manifest_fingerprint, "status": status_fingerprint}.items()
        if value is not None
    }
    if manifest_fingerprint and status_fingerprint and manifest_fingerprint != status_fingerprint:
        findings.add(
            "mixed",
            "MANIFEST_STATUS_FINGERPRINT_MISMATCH",
            "Manifest and status reference different run fingerprints.",
            manifest=manifest_fingerprint,
            status=status_fingerprint,
        )

    # Historical fold summaries lacked a fingerprint.  v2 summaries embed one and
    # are checked directly against the manifest.
    fold_reports: list[dict[str, Any]] = []
    discovered_folds: dict[int, dict[str, Any]] = {}
    for path in sorted(root.glob("ultimate_fold_*_summary.json")):
        match = re.fullmatch(r"ultimate_fold_(\d+)_summary\.json", path.name)
        if not match:
            continue
        filename_fold = int(match.group(1))
        summary = read_json_object(path, root, findings, required=False, label="fold_summary")
        item = file_info(path, root)
        item["filename_fold"] = filename_fold
        if summary is None:
            fold_reports.append(item)
            continue
        declared_fold = summary.get("fold")
        item["declared_fold"] = declared_fold
        item["status"] = summary.get("status")
        completed_at = valid_epoch(summary.get("completed_at"))
        item["completed_at"] = completed_at
        item["completed_at_utc"] = utc_timestamp(completed_at)
        item["has_embedded_fingerprint"] = "fingerprint" in summary
        summary_fingerprint = as_nonempty_string(summary.get("fingerprint"))
        item["fingerprint"] = summary_fingerprint
        if summary_fingerprint is not None and manifest_fingerprint is not None and summary_fingerprint != manifest_fingerprint:
            findings.add(
                "mixed",
                "FOLD_FINGERPRINT_MISMATCH",
                "Fold summary belongs to a different run fingerprint than the manifest.",
                path=relative_display(path, root),
                manifest=manifest_fingerprint,
                fold=summary_fingerprint,
            )
        if declared_fold != filename_fold:
            findings.add(
                "mixed",
                "FOLD_FILENAME_DECLARATION_MISMATCH",
                "Fold summary filename and JSON fold value disagree.",
                path=relative_display(path, root),
                filename_fold=filename_fold,
                declared_fold=declared_fold,
            )
        if filename_fold in discovered_folds:
            findings.add(
                "mixed",
                "DUPLICATE_FOLD_SUMMARY",
                "Multiple fold summaries resolve to the same fold number.",
                fold=filename_fold,
            )
        discovered_folds[filename_fold] = item
        if expected_folds is not None and not 1 <= filename_fold <= expected_folds:
            findings.add(
                "mixed",
                "FOLD_OUT_OF_RANGE",
                "Fold summary is outside the manifest's expected fold range.",
                path=relative_display(path, root),
                fold=filename_fold,
                expected_folds=expected_folds,
            )
        if summary.get("status") != "complete":
            findings.add(
                "incomplete",
                "FOLD_NOT_COMPLETE",
                "A discovered fold summary is not terminally complete.",
                path=relative_display(path, root),
                fold=filename_fold,
                status=summary.get("status"),
            )
        if started_at is not None and completed_at is not None and completed_at < started_at - STALE_TOLERANCE_SECONDS:
            findings.add(
                "mixed",
                "FOLD_PRECEDES_CURRENT_RUN",
                "Fold summary completion time predates the current status started_at.",
                path=relative_display(path, root),
                fold=filename_fold,
                completed_at=completed_at,
                started_at=started_at,
            )
        fold_reports.append(item)
    complete_fold_numbers = sorted(
        number for number, item in discovered_folds.items() if item.get("status") == "complete"
    )
    expected_fold_numbers = list(range(1, expected_folds + 1)) if expected_folds is not None else []
    missing_fold_numbers = sorted(set(expected_fold_numbers).difference(discovered_folds))
    if expected_folds is not None and missing_fold_numbers:
        findings.add(
            "incomplete",
            "FOLD_SUMMARIES_MISSING",
            "Not all expected outer-fold summaries are present.",
            present=sorted(discovered_folds),
            missing=missing_fold_numbers,
            expected=expected_fold_numbers,
        )
    if status_fold_numbers is not None:
        status_set = set(status_fold_numbers)
        summary_set = set(complete_fold_numbers)
        missing_summaries = sorted(status_set.difference(summary_set))
        if missing_summaries:
            findings.add(
                "mixed",
                "STATUS_FOLD_SUMMARY_MISMATCH",
                "Status claims completed folds that have no matching complete fold summary.",
                status_completed=status_fold_numbers,
                summary_complete=complete_fold_numbers,
                missing_summaries=missing_summaries,
            )
        if status_complete is True and status_set != set(expected_fold_numbers):
            findings.add(
                "mixed",
                "TERMINAL_STATUS_FOLD_COUNT_MISMATCH",
                "Terminal status does not enumerate every expected completed fold.",
                status_completed=status_fold_numbers,
                expected=expected_fold_numbers,
            )
    report["folds"] = {
        "expected": expected_fold_numbers,
        "complete": complete_fold_numbers,
        "missing": missing_fold_numbers,
        "summaries_embed_fingerprint": bool(fold_reports) and all(
            bool(item.get("has_embedded_fingerprint")) for item in fold_reports
        ),
        "items": fold_reports,
    }

    artifacts: dict[str, Any] = {}
    telemetry_path = root / "ultimate_tournament_telemetry.sqlite3"
    telemetry = inspect_sqlite(telemetry_path, root, findings)
    artifacts["telemetry"] = telemetry
    telemetry_fingerprint = as_nonempty_string(telemetry.get("run_fingerprint"))
    if telemetry_fingerprint is not None:
        fingerprint_sources["telemetry"] = telemetry_fingerprint
        if manifest_fingerprint is not None and telemetry_fingerprint != manifest_fingerprint:
            findings.add(
                "mixed",
                "TELEMETRY_FINGERPRINT_MISMATCH",
                "Telemetry SQLite ledger belongs to a different run fingerprint than the manifest.",
                manifest=manifest_fingerprint,
                telemetry=telemetry_fingerprint,
            )
    telemetry_corpus = telemetry.get("manifest_corpus")
    if isinstance(telemetry_corpus, dict):
        compare_corpus(telemetry_corpus, expected_corpus, findings, source="telemetry manifest")

    results_log_path = root / "ultimate_tournament_results.log"
    results_log = file_info(results_log_path, root)
    if results_log_path.is_file():
        text, truncated, error = read_limited_text(results_log_path)
        results_log["truncated"] = truncated
        if error is not None:
            results_log["readable"] = False
            results_log["error"] = error
            findings.add(
                "mixed",
                "RESULTS_LOG_UNREADABLE",
                "Result log could not be read.",
                path=relative_display(results_log_path, root),
                error=error,
            )
        elif text is not None:
            results_log["readable"] = True
            results_log["corpus"] = parse_corpus_from_results_log(text)
            if results_log["corpus"] is None:
                findings.add(
                    "mixed",
                    "RESULTS_LOG_CORPUS_MISSING",
                    "Result log does not contain the required corpus declaration.",
                    path=relative_display(results_log_path, root),
                )
            else:
                compare_corpus(results_log["corpus"], expected_corpus, findings, source="results log")
    artifacts["results_log"] = results_log

    runtime_path = root / "ultimate_tournament_runtime.log"
    runtime_log = file_info(runtime_path, root)
    if runtime_path.is_file():
        text, truncated, error = read_limited_text(runtime_path)
        runtime_log["truncated"] = truncated
        if error is not None:
            runtime_log["readable"] = False
            runtime_log["error"] = error
            findings.add(
                "mixed",
                "RUNTIME_LOG_UNREADABLE",
                "Runtime log could not be read.",
                path=relative_display(runtime_path, root),
                error=error,
            )
        elif text is not None:
            runtime_log["readable"] = True
            runtime_log["corpus"] = parse_corpus_from_runtime_log(text)
            runtime_log["outer_folds"] = parse_runtime_fold_count(text)
            if runtime_log["corpus"] is not None:
                compare_corpus(runtime_log["corpus"], expected_corpus, findings, source="runtime log")
            if expected_folds is not None and runtime_log["outer_folds"] is not None and runtime_log["outer_folds"] != expected_folds:
                findings.add(
                    "mixed",
                    "RUNTIME_FOLD_COUNT_MISMATCH",
                    "Runtime log's requested fold count differs from the current manifest.",
                    runtime_folds=runtime_log["outer_folds"],
                    manifest_folds=expected_folds,
                )
    artifacts["runtime_log"] = runtime_log

    results_csv_path = root / "ultimate_tournament_results.csv"
    results_csv = file_info(results_csv_path, root)
    if results_csv_path.is_file():
        try:
            with results_csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            results_csv["readable"] = True
            results_csv["row_count"] = len(rows)
            results_csv["configurations"] = [row.get("configuration") for row in rows]
            if not rows:
                findings.add(
                    "mixed",
                    "RESULTS_CSV_EMPTY",
                    "Results CSV is present but has no result rows.",
                    path=relative_display(results_csv_path, root),
                )
        except (OSError, UnicodeDecodeError, csv.Error) as error:
            results_csv["readable"] = False
            results_csv["error"] = str(error)
            findings.add(
                "mixed",
                "RESULTS_CSV_UNREADABLE",
                "Results CSV could not be parsed.",
                path=relative_display(results_csv_path, root),
                error=str(error),
            )
    artifacts["results_csv"] = results_csv

    results_manifest_path = root / "ultimate_tournament_results_manifest.json"
    results_manifest = read_json_object(
        results_manifest_path, root, findings, required=False, label="results_manifest"
    )
    results_manifest_report = file_info(results_manifest_path, root)
    if results_manifest is not None:
        results_fingerprint = as_nonempty_string(results_manifest.get("fingerprint"))
        results_manifest_report["fingerprint"] = results_fingerprint
        if results_fingerprint is None:
            findings.add(
                "mixed",
                "RESULTS_MANIFEST_FINGERPRINT_MISSING",
                "Final-results manifest does not identify its source run.",
                path=relative_display(results_manifest_path, root),
            )
        elif manifest_fingerprint is not None and results_fingerprint != manifest_fingerprint:
            findings.add(
                "mixed",
                "RESULTS_MANIFEST_FINGERPRINT_MISMATCH",
                "Final-results manifest belongs to a different run fingerprint than the tournament manifest.",
                manifest=manifest_fingerprint,
                results=results_fingerprint,
            )
        declared_hashes = results_manifest.get("files")
        if isinstance(declared_hashes, dict):
            hash_mismatches: dict[str, dict[str, str | None]] = {}
            for filename, expected_hash in declared_hashes.items():
                if not isinstance(filename, str) or not isinstance(expected_hash, str):
                    continue
                candidate = root / filename
                actual_hash = sha256_bytes(candidate.read_bytes()) if candidate.is_file() else None
                if actual_hash != expected_hash:
                    hash_mismatches[filename] = {"expected": expected_hash, "actual": actual_hash}
            results_manifest_report["verified_file_hashes"] = not bool(hash_mismatches)
            if hash_mismatches:
                findings.add(
                    "mixed",
                    "RESULTS_MANIFEST_FILE_HASH_MISMATCH",
                    "A final artifact differs from the hash recorded in the results manifest.",
                    mismatches=hash_mismatches,
                )
        else:
            findings.add(
                "mixed",
                "RESULTS_MANIFEST_FILES_INVALID",
                "Final-results manifest must map filenames to SHA-256 hashes.",
                path=relative_display(results_manifest_path, root),
            )
    artifacts["results_manifest"] = results_manifest_report

    oof_path = root / "ultimate_oof_predictions.npz"
    oof = inspect_npz(oof_path, root, findings)
    arrays = oof.get("arrays")
    if isinstance(arrays, dict) and expected_corpus["trajectories"] is not None:
        row_mismatches: dict[str, list[int]] = {}
        length_mismatches: dict[str, list[int]] = {}
        for name, shape in arrays.items():
            if not isinstance(shape, list) or not shape:
                continue
            if shape[0] != expected_corpus["trajectories"]:
                row_mismatches[name] = shape
            if expected_length is not None and len(shape) >= 2 and shape[1] != expected_length:
                length_mismatches[name] = shape
        if row_mismatches:
            findings.add(
                "mixed",
                "OOF_TRAJECTORY_DIMENSION_MISMATCH",
                "OOF arrays have a trajectory dimension different from the current manifest.",
                expected_trajectories=expected_corpus["trajectories"],
                arrays=row_mismatches,
            )
        if length_mismatches:
            findings.add(
                "mixed",
                "OOF_SEQUENCE_DIMENSION_MISMATCH",
                "OOF arrays have a sequence length different from the current manifest.",
                expected_sequence_length=expected_length,
                arrays=length_mismatches,
            )
    artifacts["oof_predictions"] = oof

    graph_path = root / "ultimate_research_graph.json"
    graph = inspect_research_graph(graph_path, root, findings)
    graph_corpus = graph.get("corpus")
    if isinstance(graph_corpus, dict):
        compare_corpus(graph_corpus, expected_corpus, findings, source="research graph")
    graph_entity_id = as_nonempty_string(graph.get("dataset_entity_id"))
    expected_dataset = manifest.get("dataset_fingerprint") if manifest else None
    if graph_entity_id is not None and isinstance(expected_dataset, str):
        expected_prefix = f"dataset:{expected_dataset[:16]}"
        if graph_entity_id != expected_prefix:
            findings.add(
                "mixed",
                "RESEARCH_GRAPH_DATASET_MISMATCH",
                "Research graph corpus entity references a different dataset fingerprint prefix.",
                graph_dataset_entity=graph_entity_id,
                expected_dataset_entity=expected_prefix,
            )
    artifacts["research_graph"] = graph

    checkpoint_path = root / "ultimate_tournament_checkpoint.pth"
    checkpoint = file_info(checkpoint_path, root)
    checkpoint["deserialized"] = False
    checkpoint["reason"] = "pickle-backed checkpoint intentionally not deserialized by read-only audit"
    artifacts["checkpoint"] = checkpoint

    # These artifacts are created during a training run.  An older mtime is not the
    # sole source of truth (files may be copied), but paired with a current status it
    # is strong evidence of a mixed output directory and should block interpretation.
    predating = [
        relative_display(path, root)
        for path in [telemetry_path, results_log_path, runtime_path, results_csv_path, oof_path, graph_path, checkpoint_path]
        if artifact_predates_run(path, started_at)
    ]
    if predating:
        findings.add(
            "mixed",
            "ARTIFACTS_PREDATE_CURRENT_RUN",
            "Artifacts expected to be produced by the current run are older than status.started_at.",
            started_at=started_at,
            started_at_utc=utc_timestamp(started_at),
            artifacts=predating,
        )

    # A two-cell/two-fold run is the runner's standard smoke-test footprint.  It is
    # not automatically bad when the manifest itself requests two cells, but it is
    # decisive stale-artifact evidence when a full corpus manifest is current.
    smoke_evidence: list[dict[str, Any]] = []
    expected_cells = expected_corpus.get("cells")
    if isinstance(expected_cells, int) and expected_cells > 2:
        for source, corpus in {
            "telemetry": telemetry_corpus,
            "results_log": results_log.get("corpus"),
            "runtime_log": runtime_log.get("corpus"),
            "research_graph": graph_corpus,
        }.items():
            if isinstance(corpus, dict) and isinstance(corpus.get("cells"), int) and corpus["cells"] <= 2:
                smoke_evidence.append({"artifact": source, "cells": corpus["cells"]})
        runtime_folds = runtime_log.get("outer_folds")
        if isinstance(runtime_folds, int) and runtime_folds <= 2 and expected_folds is not None and expected_folds > 2:
            smoke_evidence.append({"artifact": "runtime_log", "outer_folds": runtime_folds})
    if smoke_evidence:
        findings.add(
            "mixed",
            "STALE_SMOKE_ARTIFACTS",
            "One or more artifacts have the standard small smoke-test footprint rather than the current full manifest.",
            expected_cells=expected_cells,
            expected_outer_folds=expected_folds,
            evidence=smoke_evidence,
        )

    if status_complete is True:
        terminal_required = {
            "results_log": results_log_path,
            "results_csv": results_csv_path,
            "results_manifest": results_manifest_path,
            "oof_predictions": oof_path,
            "research_graph": graph_path,
            "telemetry": telemetry_path,
        }
        missing_terminal = [name for name, path in terminal_required.items() if not path.is_file()]
        if missing_terminal:
            findings.add(
                "mixed",
                "TERMINAL_ARTIFACTS_MISSING",
                "Status is terminal but required final artifacts are absent.",
                missing=missing_terminal,
            )
        if expected_folds is not None and set(complete_fold_numbers) != set(expected_fold_numbers):
            findings.add(
                "mixed",
                "TERMINAL_FOLDS_INCOMPLETE",
                "Status is terminal but complete fold summaries do not cover every expected outer fold.",
                complete=complete_fold_numbers,
                expected=expected_fold_numbers,
            )
    elif status_complete is False:
        findings.add(
            "incomplete",
            "STATUS_NOT_TERMINAL",
            "Current tournament status is not complete.",
            status=status_name,
            completed_folds=status_fold_numbers,
        )

    report["fingerprint_sources"] = fingerprint_sources
    report["artifacts"] = artifacts
    report["findings"] = findings.as_dicts()
    if findings.has_category("invalid"):
        report["assessment"] = "invalid"
        report["exit_code"] = EXIT_INVALID
    elif findings.has_category("mixed"):
        report["assessment"] = "mixed_or_stale"
        report["exit_code"] = EXIT_MIXED_OR_STALE
    elif status_complete is True and expected_folds is not None and set(complete_fold_numbers) == set(expected_fold_numbers):
        report["assessment"] = "complete"
        report["exit_code"] = EXIT_COMPLETE
    else:
        report["assessment"] = "incomplete_or_running"
        report["exit_code"] = EXIT_INCOMPLETE
    return report


def human_report(report: dict[str, Any]) -> str:
    """Render a compact human-readable summary without hiding evidence in JSON."""

    lines = [
        "ULTIMATE TOURNAMENT ARTIFACT AUDIT (read-only)",
        f"Directory: {report.get('output_dir')}",
        f"Assessment: {report.get('assessment')} (exit {report.get('exit_code')})",
    ]
    manifest = report.get("manifest")
    if isinstance(manifest, dict):
        lines.append(f"Manifest run fingerprint: {manifest.get('run_fingerprint')}")
        hashes = manifest.get("hashes")
        if isinstance(hashes, dict):
            lines.append(f"Manifest SHA-256 (LF-normalized): {hashes.get('line_ending_normalized_sha256')}")
        corpus = manifest.get("corpus")
        if isinstance(corpus, dict):
            lines.append(
                "Manifest corpus: "
                f"{corpus.get('cells')} cells, {corpus.get('trajectories')} trajectories, "
                f"{corpus.get('tasks')} task groups"
            )
    status = report.get("status")
    if isinstance(status, dict):
        lines.append(
            f"Status: {status.get('status')} (complete={status.get('complete')}, "
            f"fingerprint={status.get('fingerprint')})"
        )
    folds = report.get("folds")
    if isinstance(folds, dict):
        lines.append(
            f"Fold summaries: complete={folds.get('complete')}, missing={folds.get('missing')}, "
            f"expected={folds.get('expected')}"
        )
    lines.append("Findings:")
    findings = report.get("findings")
    if not findings:
        lines.append("  [INFO] No findings.")
    elif isinstance(findings, list):
        for item in findings:
            if not isinstance(item, dict):
                continue
            lines.append(f"  [{str(item.get('severity', 'warning')).upper()}] {item.get('code')}: {item.get('message')}")
            evidence = item.get("evidence")
            if evidence:
                lines.append(f"    Evidence: {json.dumps(evidence, sort_keys=True, default=str)}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/outputs/experiments_v2"),
        help="Directory containing ultimate_tournament artifacts.",
    )
    parser.add_argument("--json", action="store_true", help="Emit the full report as JSON on stdout.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_artifacts(args.output_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        print(human_report(report))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
