#!/usr/bin/env python3
"""Task-held-out causal text-output screen for stopping correctness.

The model sees only the final ``--tail-chars`` characters of the *current*
model output.  It never receives prompt/task identifiers, gold answers,
correctness, timing, model aliases, source cells, or text from a future step.
Vocabulary/IDF, classifier, and Platt calibration are fit inside each outer
task GroupKFold training partition.

This is a historical content-signal screen, not fresh deployment proof: a
model's natural response can quote or paraphrase its prompt.  The strict
task split prevents exact task memorization, while a fresh corpus is still
required before a deployment claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold


SCHEMA_VERSION = "strict-text-stopping-oof-v1"
DEFAULT_INPUT = Path("research/outputs/experiments_v2")
DEFAULT_OUTPUT = Path("research/outputs/experiments_v2/strict_text_stopping_oof_v1")
PREPARED_FRAME = "prepared_text_rows.pkl"
PREPARED_MANIFEST = "prepared_manifest.json"


@dataclass(frozen=True)
class TextConfig:
    name: str = "tail512_word12_sgdlog_v1"
    tail_chars: int = 512
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 4
    max_df: float = 0.995
    max_features: int = 80_000
    alpha: float = 1.0e-5
    max_iter: int = 30
    tol: float = 1.0e-3


CONFIG = TextConfig()


def load_module(filename: str, name: str) -> Any:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(value) + b"\n")
    os.replace(temporary, path)


def atomic_pickle(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".pkl") as handle:
        temporary = Path(handle.name)
    try:
        frame.to_pickle(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".npz") as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".csv") as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        Path(__file__).name: file_hash(Path(__file__).resolve()),
        "run_ultimate_multi_day_tournament.py": file_hash(root / "run_ultimate_multi_day_tournament.py"),
    }


def runtime_contract() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def frame_key_hash(frame: pd.DataFrame) -> str:
    values = [
        (str(task), str(trajectory), int(step))
        for task, trajectory, step in frame[["task_id", "trajectory_id", "step"]].itertuples(index=False, name=None)
    ]
    return hashlib.sha256(canonical_json(values)).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--fold", type=int, help="One-based outer task GroupKFold fold.")
    action.add_argument("--aggregate", action="store_true")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--tail-chars", type=int, default=CONFIG.tail_chars)
    parser.add_argument("--status-file", type=Path, default=None)
    return parser.parse_args()


def build_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    if args.tail_chars != CONFIG.tail_chars:
        raise ValueError(f"The frozen v1 protocol requires --tail-chars {CONFIG.tail_chars}.")
    runtime = load_module("run_ultimate_multi_day_tournament.py", "strict_text_runtime")
    loader_args = SimpleNamespace(input_dir=str(args.input_dir), include_all_cells=False, max_cells=None)
    raw, input_manifest = runtime.load_trace_frame(loader_args)
    if "raw_text" not in raw.columns:
        raise ValueError("Trace corpus has no raw_text field for the causal text screen.")
    result = raw[["task_id", "trajectory_id", "source_cell", "step", "correct"]].copy()
    # The tail avoids directly modeling the beginning of responses, where a
    # model often copies the question verbatim.  It still preserves the final
    # answer/reasoning state that is available at the stop barrier.
    result["text_tail"] = raw["raw_text"].fillna("").astype(str).str[-CONFIG.tail_chars :]
    result["task_id"] = result["task_id"].astype(str)
    result["trajectory_id"] = result["trajectory_id"].astype(str)
    result["source_cell"] = result["source_cell"].astype(str)
    result["step"] = pd.to_numeric(result["step"], errors="raise").astype(np.int16)
    result["correct"] = pd.to_numeric(result["correct"], errors="raise").astype(np.int8)
    if (
        result.duplicated(["trajectory_id", "step"]).any()
        or not result["correct"].isin([0, 1]).all()
        or result["text_tail"].str.len().gt(CONFIG.tail_chars).any()
    ):
        raise RuntimeError("Text frame violates the causal row/label/tail contract.")
    return result, input_manifest


def prepare(args: argparse.Namespace) -> None:
    frame, input_manifest = build_frame(args)
    if args.n_splits < 2 or frame["task_id"].nunique() < args.n_splits:
        raise ValueError("Insufficient task groups for requested GroupKFold protocol.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = args.output_dir / PREPARED_FRAME
    manifest_path = args.output_dir / PREPARED_MANIFEST
    if manifest_path.exists():
        raise FileExistsError("Prepared manifest already exists; use a new output directory.")
    atomic_pickle(frame_path, frame)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "source_hashes": source_hashes(),
        "runtime_contract": runtime_contract(),
        "input_dir": str(args.input_dir),
        "input_manifest": input_manifest,
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "config": asdict(CONFIG),
        "frame_sha256": file_hash(frame_path),
        "frame_key_sha256": frame_key_hash(frame),
        "rows": int(len(frame)),
        "task_groups": int(frame["task_id"].nunique()),
        "strict_contract": {
            "outer_group": "task_id",
            "text": "only final 512 characters of current raw model output",
            "excluded": "prompt/task/source/model identifiers, expected answer, gold label, timing, raw answer fields, future output",
            "vocabulary": "fit inside model-fit task groups only",
            "calibration": "separate task-disjoint partition within each outer train fold",
        },
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps({"prepared": str(args.output_dir), "rows": len(frame), "tasks": manifest["task_groups"]}, sort_keys=True), flush=True)


def load_prepared(args: argparse.Namespace) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    frame_path = args.output_dir / PREPARED_FRAME
    manifest_path = args.output_dir / PREPARED_MANIFEST
    if not frame_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Prepared text frame is missing; run --prepare first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("source_hashes") != source_hashes()
        or manifest.get("runtime_contract") != runtime_contract()
        or manifest.get("input_dir") != str(args.input_dir)
        or int(manifest.get("n_splits", -1)) != args.n_splits
        or int(manifest.get("seed", -1)) != args.seed
        or manifest.get("config") != asdict(CONFIG)
        or file_hash(frame_path) != manifest.get("frame_sha256")
    ):
        raise RuntimeError("Prepared text artifact has a different source, split, text, or runtime contract.")
    frame = pd.read_pickle(frame_path)
    required = {"task_id", "trajectory_id", "source_cell", "step", "correct", "text_tail"}
    if (
        required - set(frame.columns)
        or len(frame) != int(manifest.get("rows", -1))
        or frame["task_id"].nunique() != int(manifest.get("task_groups", -1))
        or frame_key_hash(frame) != manifest.get("frame_key_sha256")
        or frame.duplicated(["trajectory_id", "step"]).any()
        or frame["text_tail"].fillna("").astype(str).str.len().gt(CONFIG.tail_chars).any()
    ):
        raise RuntimeError("Prepared text frame is malformed or no longer matches its manifest.")
    runtime = load_module("run_ultimate_multi_day_tournament.py", "strict_text_runtime")
    return runtime, frame, manifest


def inner_fit_calibration(outer_train: np.ndarray, groups: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    local = np.arange(len(outer_train), dtype=np.int64)
    split = GroupKFold(n_splits=5)
    fit_local, calibration_local = list(split.split(local, groups=groups[outer_train]))[fold % 5]
    fit, calibration = outer_train[fit_local], outer_train[calibration_local]
    if set(groups[fit]) & set(groups[calibration]):
        raise AssertionError("Task leakage in text calibration partition.")
    return fit, calibration


def vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(CONFIG.ngram_min, CONFIG.ngram_max),
        token_pattern=r"(?u)\b\w+\b",
        min_df=CONFIG.min_df,
        max_df=CONFIG.max_df,
        max_features=CONFIG.max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )


def fold_path(output_dir: Path, fold: int) -> Path:
    return output_dir / "fold_checkpoints" / f"tail512_word12_fold_{fold:02d}.npz"


def train_fold(args: argparse.Namespace) -> None:
    runtime, frame, _manifest = load_prepared(args)
    if args.fold is None or not 1 <= args.fold <= args.n_splits:
        raise ValueError(f"--fold must be in [1, {args.n_splits}].")
    groups = frame["task_id"].to_numpy(dtype=str)
    labels = frame["correct"].to_numpy(dtype=np.int8)
    indices = np.arange(len(frame), dtype=np.int64)
    outer_train, outer_test = (
        np.asarray(item, dtype=np.int64)
        for item in list(GroupKFold(n_splits=args.n_splits).split(indices, labels, groups))[args.fold - 1]
    )
    if set(groups[outer_train]) & set(groups[outer_test]):
        raise AssertionError("Task leakage in text outer fold.")
    model_fit, calibration = inner_fit_calibration(outer_train, groups, args.fold - 1)
    checkpoint = fold_path(args.output_dir, args.fold)
    if checkpoint.exists():
        raise FileExistsError(f"Immutable fold checkpoint already exists: {checkpoint}")
    text = frame["text_tail"].fillna("").astype(str).to_numpy(dtype=object)
    started = time.time()
    extractor = vectorizer()
    fit_x = extractor.fit_transform(text[model_fit])
    calibration_x = extractor.transform(text[calibration])
    test_x = extractor.transform(text[outer_test])
    model = SGDClassifier(
        loss="log_loss",
        alpha=CONFIG.alpha,
        max_iter=CONFIG.max_iter,
        tol=CONFIG.tol,
        random_state=args.seed + args.fold,
        average=True,
    )
    model.fit(fit_x, labels[model_fit])
    calibration_logit = np.asarray(model.decision_function(calibration_x), dtype=np.float64)
    scaler = runtime.PlattScaler().fit(calibration_logit, labels[calibration])
    test_logit = np.asarray(model.decision_function(test_x), dtype=np.float64)
    probability = scaler.transform(test_logit).astype(np.float64)
    binding = {
        "schema_version": SCHEMA_VERSION,
        "prepared_manifest_sha256": file_hash(args.output_dir / PREPARED_MANIFEST),
        "config": asdict(CONFIG),
        "fold": int(args.fold),
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "runtime_contract": runtime_contract(),
        "test_indices_sha256": hashlib.sha256(outer_test.tobytes()).hexdigest(),
        "vocabulary_size": int(len(extractor.vocabulary_)),
        "elapsed_seconds": time.time() - started,
        "fold_auc": float(roc_auc_score(labels[outer_test], probability)),
    }
    atomic_npz(
        checkpoint,
        test_indices=outer_test,
        probabilities=probability,
        binding_json=np.asarray(json.dumps(binding, sort_keys=True)),
    )
    print(json.dumps({"fold": args.fold, "auc": binding["fold_auc"], "vocabulary": binding["vocabulary_size"]}, sort_keys=True), flush=True)


def task_bootstrap_auc(frame: pd.DataFrame, probability: np.ndarray, replicates: int, seed: int) -> list[float]:
    labels = frame["correct"].to_numpy(dtype=np.int8)
    groups = frame["task_id"].to_numpy(dtype=str)
    unique, inverse = np.unique(groups, return_inverse=True)
    buckets = [np.flatnonzero(inverse == index) for index in range(len(unique))]
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        selected = rng.integers(0, len(unique), size=len(unique))
        sample = np.concatenate([buckets[index] for index in selected])
        if np.unique(labels[sample]).size == 2:
            values.append(float(roc_auc_score(labels[sample], probability[sample])))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] if values else [float("nan"), float("nan")]


def aggregate(args: argparse.Namespace) -> None:
    _runtime, frame, manifest = load_prepared(args)
    labels = frame["correct"].to_numpy(dtype=np.int8)
    groups = frame["task_id"].to_numpy(dtype=str)
    indices = np.arange(len(frame), dtype=np.int64)
    splits = list(GroupKFold(n_splits=args.n_splits).split(indices, labels, groups))
    probability = np.full(len(frame), np.nan, dtype=np.float64)
    fold_id = np.full(len(frame), -1, dtype=np.int8)
    manifest_hash = file_hash(args.output_dir / PREPARED_MANIFEST)
    fold_auc: list[float] = []
    for fold in range(1, args.n_splits + 1):
        path = fold_path(args.output_dir, fold)
        if not path.is_file():
            raise FileNotFoundError(f"Missing strict text fold checkpoint: {path}")
        with np.load(path, allow_pickle=False) as payload:
            test_indices = np.asarray(payload["test_indices"], dtype=np.int64)
            values = np.asarray(payload["probabilities"], dtype=np.float64)
            binding = json.loads(str(payload["binding_json"].item()))
        expected = np.asarray(splits[fold - 1][1], dtype=np.int64)
        expected_hash = hashlib.sha256(expected.tobytes()).hexdigest()
        if (
            binding.get("schema_version") != SCHEMA_VERSION
            or binding.get("prepared_manifest_sha256") != manifest_hash
            or binding.get("config") != asdict(CONFIG)
            or binding.get("fold") != fold
            or binding.get("n_splits") != args.n_splits
            or binding.get("seed") != args.seed
            or binding.get("runtime_contract") != runtime_contract()
            or binding.get("test_indices_sha256") != expected_hash
            or not np.array_equal(test_indices, expected)
            or len(np.unique(test_indices)) != len(test_indices)
            or len(values) != len(test_indices)
            or not np.isfinite(values).all()
        ):
            raise RuntimeError(f"Invalid or foreign text fold checkpoint: {path}")
        if np.isfinite(probability[test_indices]).any():
            raise RuntimeError(f"Overlapping text OOF rows in fold {fold}.")
        probability[test_indices] = values
        fold_id[test_indices] = fold
        fold_auc.append(float(binding["fold_auc"]))
    if not np.isfinite(probability).all() or (fold_id < 1).any():
        raise RuntimeError("Text fold checkpoints do not cover every row exactly once.")
    metrics_path = args.output_dir / "tail512_word12_metrics.json"
    predictions_path = args.output_dir / "tail512_word12_predictions.csv"
    if metrics_path.exists():
        raise FileExistsError("Final text metrics already exist; use a new output directory.")
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "configuration": asdict(CONFIG),
        "prepared_manifest_sha256": manifest_hash,
        "rows": int(len(frame)),
        "task_groups": int(frame["task_id"].nunique()),
        "fold_auc": fold_auc,
        "oof_auc": float(roc_auc_score(labels, probability)),
        "raw_brier": float(brier_score_loss(labels, probability)),
        "task_cluster_bootstrap_auc_95_ci": task_bootstrap_auc(frame, probability, replicates=1000, seed=args.seed),
        "strict_contract": manifest["strict_contract"],
        "interpretation": "strict task-held-out historical content-output screen; natural prompt echoes remain a fresh-corpus limitation.",
    }
    prediction = frame[["task_id", "trajectory_id", "source_cell", "step", "correct"]].copy()
    prediction["outer_fold"] = fold_id
    prediction["oof_probability"] = probability
    atomic_csv(predictions_path, prediction)
    atomic_npz(args.output_dir / "tail512_word12_oof.npz", probability=probability, outer_fold=fold_id)
    atomic_json(metrics_path, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


def main() -> int:
    args = parse_args()
    if args.n_splits < 2 or args.tail_chars <= 0:
        raise ValueError("--n-splits must be at least two and --tail-chars must be positive.")
    action = "prepare" if args.prepare else (f"fold_{args.fold:02d}" if args.fold is not None else "aggregate")
    status_path = args.status_file or args.output_dir / "strict_text_status.json"
    atomic_json(status_path, {"schema_version": SCHEMA_VERSION, "state": "running", "action": action, "started_at_unix": time.time()})
    try:
        if args.prepare:
            prepare(args)
        elif args.fold is not None:
            train_fold(args)
        else:
            aggregate(args)
    except Exception as error:
        atomic_json(
            status_path,
            {"schema_version": SCHEMA_VERSION, "state": "failed", "action": action, "finished_at_unix": time.time(), "error": f"{type(error).__name__}: {error}"},
        )
        raise
    atomic_json(status_path, {"schema_version": SCHEMA_VERSION, "state": "complete", "action": action, "finished_at_unix": time.time()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
