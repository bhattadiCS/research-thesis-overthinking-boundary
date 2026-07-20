#!/usr/bin/env python3
"""Task-held-out retrospective thought-content screen for selected-answer correctness.

Each decision barrier first chooses the same deterministic plurality answer as
``run_selected_answer_oof_experiments.py``.  The learner then sees only the
last 512 characters of the sanitized *current parsed thought* fields that
support that plurality.  Raw completions are deliberately excluded because
the legacy raw-text field sometimes contains task metadata and prior steps.
Supporter identities, answer fields, task identifiers, gold answers, timing,
and all future barriers are excluded from the text document.

This is intentionally a historical, retrospective screen rather than evidence
of a deployable stopping policy: the source traces lack peer-completion
timestamps, and natural responses can still quote their prompt.  Its metric is
decision-level selected-answer correctness, not per-trajectory stopping AUC.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold


SCHEMA_VERSION = "selected-answer-content-oof-v4-sanitized-thought"
DEFAULT_INPUT = Path("research/outputs/experiments_v2")
DEFAULT_OUTPUT = Path("research/outputs/experiments_v2/selected_answer_content_oof_v4_reasoning_only")
PREPARED_FRAME = "prepared_selected_answer_thought_content.pkl"
PREPARED_MANIFEST = "prepared_selected_answer_thought_content_manifest.json"
ECHO_MARKERS = (
    "task id:",
    "task:",
    "domain:",
    "difficulty band:",
    "previous steps:",
    "research protocol:",
)
TASK_ID_PATTERN = re.compile(r"\b(?:arc|gpqa|gsm8k|math)_[a-z]+_[0-9]+_[0-9a-f]{8}\b", flags=re.IGNORECASE)
STRUCTURED_FIELD_PATTERN = re.compile(
    r"\b(?:answer|confidence|stop)\s*[:=][^|]*",
    flags=re.IGNORECASE,
)
EMPTY_SENTINEL = "__EMPTY_THOUGHT__"
REDACTED_SENTINEL = "__REDACTED_PROMPT_OR_HISTORY_ECHO__"


@dataclass(frozen=True)
class ContentConfig:
    name: str = "plurality_supporter_reasoning_only_thought512_word12_sgdlog_v4"
    tail_chars_per_supporter: int = 512
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 4
    max_df: float = 0.995
    max_features: int = 80_000
    alpha: float = 1.0e-5
    max_iter: int = 30
    tol: float = 1.0e-3


CONFIG = ContentConfig()


def load_reference() -> Any:
    path = Path(__file__).with_name("run_selected_answer_oof_experiments.py")
    spec = importlib.util.spec_from_file_location("selected_answer_content_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_runtime() -> Any:
    path = Path(__file__).with_name("run_ultimate_multi_day_tournament.py")
    spec = importlib.util.spec_from_file_location("selected_answer_content_runtime", path)
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


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
        "run_selected_answer_oof_experiments.py": file_hash(root / "run_selected_answer_oof_experiments.py"),
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
        (str(task), int(step), str(document_hash))
        for task, step, document_hash in frame[["task_id", "step", "document_sha256"]].itertuples(index=False, name=None)
    ]
    return hashlib.sha256(canonical_json(values)).hexdigest()


def sanitize_current_thought(value: Any) -> tuple[str, bool, bool]:
    """Return a bounded reasoning-only thought without known metadata echoes."""
    text = re.sub(r"\s+", " ", "" if pd.isna(value) else str(value)).strip()
    if not text:
        return EMPTY_SENTINEL, False, False
    if any(marker in text.casefold() for marker in ECHO_MARKERS) or TASK_ID_PATTERN.search(text):
        return REDACTED_SENTINEL, True, False
    stripped = STRUCTURED_FIELD_PATTERN.sub("", text).strip()
    structured_redacted = stripped != text
    if not stripped:
        return EMPTY_SENTINEL, False, structured_redacted
    return stripped[-CONFIG.tail_chars_per_supporter :], False, structured_redacted


def stable_supporter_document(values: list[str]) -> str:
    """Anonymize supporter ordering without adding source/model identifiers."""
    tails = [str(value) for value in values]
    if not tails or any(not tail for tail in tails):
        raise ValueError("Every selected plurality supporter must have a nonempty sanitized current thought")
    # A content hash gives a fixed order while removing the model-alias order.
    tails = sorted(tails, key=hash_text)
    return "\n\n<ANONYMOUS_PLURALITY_SUPPORTER>\n".join(tails)


def build_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    reference = load_reference()
    raw, input_files = reference.load_panel(args.input_dir)
    if "thought" not in raw.columns:
        raise ValueError("Trace corpus has no parsed thought field for the selected-answer content screen.")
    decisions, _numeric, _categorical, aliases = reference.build_decision_frame(raw, exclude_batch_timing=True)
    keys = ["task_id", "step"]
    work = raw[["task_id", "step", "answer_normalized", "thought"]].copy()
    work["_answer_key"] = work["answer_normalized"].map(reference.normalized_answer_key)
    work = work.merge(
        decisions[keys + ["selected_answer_key"]],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    if work["selected_answer_key"].isna().any():
        raise AssertionError("Could not align every response with a selected decision")
    supporters = work.loc[
        (work["_answer_key"] != "") & (work["_answer_key"] == work["selected_answer_key"]),
        keys + ["thought"],
    ].copy()
    sanitized = supporters["thought"].map(sanitize_current_thought)
    supporters["_sanitized_thought"] = [value[0] for value in sanitized]
    supporters["_redacted_echo"] = [value[1] for value in sanitized]
    supporters["_stripped_structured_field"] = [value[2] for value in sanitized]
    documents: list[dict[str, Any]] = []
    for (task_id, step), group in supporters.groupby(keys, sort=False):
        document = stable_supporter_document(group["_sanitized_thought"].astype(str).tolist())
        documents.append(
            {
                "task_id": str(task_id),
                "step": int(step),
                "content_document": document,
                "document_sha256": hash_text(document),
                "plurality_supporter_count": int(len(group)),
                "document_chars": int(len(document)),
            }
        )
    documents_frame = pd.DataFrame(documents)
    result = decisions[["task_id", "step", "domain", "selected_correct"]].merge(
        documents_frame,
        on=keys,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(result) != len(decisions) or result["content_document"].isna().any():
        raise AssertionError("Every selected decision must have exactly one current-supporter content document")
    result["task_id"] = result["task_id"].astype(str)
    result["step"] = pd.to_numeric(result["step"], errors="raise").astype(np.int16)
    result["selected_correct"] = pd.to_numeric(result["selected_correct"], errors="raise").astype(np.int8)
    result["plurality_supporter_count"] = pd.to_numeric(result["plurality_supporter_count"], errors="raise").astype(np.int8)
    result["document_chars"] = pd.to_numeric(result["document_chars"], errors="raise").astype(np.int32)
    result = result.sort_values(["task_id", "step"], kind="stable").reset_index(drop=True)
    if (
        result.duplicated(keys).any()
        or not result["selected_correct"].isin([0, 1]).all()
        or result["content_document"].str.len().eq(0).any()
        or result["content_document"].str.contains("<ANONYMOUS_PLURALITY_SUPPORTER>", regex=False).eq(False).all()
        and len(result) > 1
    ):
        raise RuntimeError("Selected-answer content frame violates the current-barrier document contract.")
    diagnostics = {
        "candidate_rows": int(len(raw)),
        "decision_rows": int(len(result)),
        "task_groups": int(result["task_id"].nunique()),
        "redacted_supporter_thoughts": int(supporters["_redacted_echo"].sum()),
        "structured_field_stripped_supporter_thoughts": int(supporters["_stripped_structured_field"].sum()),
        "frozen_alias_order_used_only_for_selection_ties": list(aliases),
        "supporter_count_quantiles": {
            str(key): float(value)
            for key, value in result["plurality_supporter_count"].quantile([0.0, 0.25, 0.5, 0.75, 0.9, 1.0]).items()
        },
        "document_char_quantiles": {
            str(key): float(value)
            for key, value in result["document_chars"].quantile([0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]).items()
        },
    }
    return result, input_files, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--fold", type=int, help="One-based outer task GroupKFold fold.")
    action.add_argument("--aggregate", action="store_true")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--status-file", type=Path, default=None)
    return parser.parse_args()


def prepare(args: argparse.Namespace) -> None:
    frame, input_files, diagnostics = build_frame(args)
    if args.n_splits < 2 or frame["task_id"].nunique() < args.n_splits:
        raise ValueError("Insufficient task groups for requested GroupKFold protocol.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = args.output_dir / PREPARED_FRAME
    manifest_path = args.output_dir / PREPARED_MANIFEST
    if manifest_path.exists() or frame_path.exists():
        raise FileExistsError("Prepared selected-answer content artifact already exists; use a new output directory.")
    atomic_pickle(frame_path, frame)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "source_hashes": source_hashes(),
        "runtime_contract": runtime_contract(),
        "input_dir": str(args.input_dir),
        "input_files": input_files,
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "config": asdict(CONFIG),
        "frame_sha256": file_hash(frame_path),
        "frame_key_sha256": frame_key_hash(frame),
        **diagnostics,
        "strict_contract": {
            "target": "decision-level selected-answer correctness",
            "selection": "nonempty normalized-answer plurality; frozen-alias tie break",
            "text": "only final 512 characters of each reasoning-only sanitized current plurality-supporting parsed thought; known task/prompt/history echoes become a fixed sentinel, structured answer/confidence/stop fields are stripped, and supporter order is content-hash sorted",
            "excluded": "raw_text completions, task/source/model identifiers, expected answers, gold labels, answer fields, timing, other-panel text, and future output",
            "vocabulary": "fit only inside model-fit task groups",
            "calibration": "separate task-disjoint partition within each outer training fold",
            "historical_limit": "same-step peer completion lacks event timestamps; natural output may quote/paraphrase the prompt",
            "interpretation": "retrospective content signal screen, not raw stopping AUC or prospective deployment proof",
        },
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps({"prepared": str(args.output_dir), "rows": len(frame), "tasks": diagnostics["task_groups"]}, sort_keys=True), flush=True)


def load_prepared(args: argparse.Namespace) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    frame_path = args.output_dir / PREPARED_FRAME
    manifest_path = args.output_dir / PREPARED_MANIFEST
    if not frame_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Prepared selected-answer content frame is missing; run --prepare first.")
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
        raise RuntimeError("Prepared content artifact has a different source, split, text, or runtime contract.")
    frame = pd.read_pickle(frame_path)
    required = {
        "task_id",
        "step",
        "domain",
        "selected_correct",
        "content_document",
        "document_sha256",
        "plurality_supporter_count",
        "document_chars",
    }
    if (
        required - set(frame.columns)
        or len(frame) != int(manifest.get("decision_rows", -1))
        or frame["task_id"].nunique() != int(manifest.get("task_groups", -1))
        or frame_key_hash(frame) != manifest.get("frame_key_sha256")
        or frame.duplicated(["task_id", "step"]).any()
        or frame["content_document"].fillna("").astype(str).str.len().eq(0).any()
    ):
        raise RuntimeError("Prepared selected-answer content frame is malformed or no longer matches its manifest.")
    return load_runtime(), frame, manifest


def inner_fit_calibration(outer_train: np.ndarray, groups: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    local = np.arange(len(outer_train), dtype=np.int64)
    split = GroupKFold(n_splits=5)
    fit_local, calibration_local = list(split.split(local, groups=groups[outer_train]))[fold % 5]
    fit, calibration = outer_train[fit_local], outer_train[calibration_local]
    if set(groups[fit]) & set(groups[calibration]):
        raise AssertionError("Task leakage in selected-answer content calibration partition.")
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
    return output_dir / "fold_checkpoints" / f"plurality_supporter_reasoning_only_thought512_fold_{fold:02d}.npz"


def train_fold(args: argparse.Namespace) -> None:
    runtime, frame, _manifest = load_prepared(args)
    if args.fold is None or not 1 <= args.fold <= args.n_splits:
        raise ValueError(f"--fold must be in [1, {args.n_splits}].")
    groups = frame["task_id"].to_numpy(dtype=str)
    labels = frame["selected_correct"].to_numpy(dtype=np.int8)
    indices = np.arange(len(frame), dtype=np.int64)
    outer_train, outer_test = (
        np.asarray(item, dtype=np.int64)
        for item in list(GroupKFold(n_splits=args.n_splits).split(indices, labels, groups))[args.fold - 1]
    )
    if set(groups[outer_train]) & set(groups[outer_test]):
        raise AssertionError("Task leakage in selected-answer content outer fold.")
    model_fit, calibration = inner_fit_calibration(outer_train, groups, args.fold - 1)
    checkpoint = fold_path(args.output_dir, args.fold)
    if checkpoint.exists():
        raise FileExistsError(f"Immutable fold checkpoint already exists: {checkpoint}")
    text = frame["content_document"].fillna("").astype(str).to_numpy(dtype=object)
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
        "fit_rows": int(len(model_fit)),
        "calibration_rows": int(len(calibration)),
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


def expected_calibration_error(labels: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probability = np.clip(np.asarray(probability, dtype=np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, bins - 1)
    value = 0.0
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            value += float(mask.mean()) * abs(float(labels[mask].mean()) - float(probability[mask].mean()))
    return float(value)


def task_bootstrap_auc(frame: pd.DataFrame, probability: np.ndarray, replicates: int, seed: int) -> list[float]:
    labels = frame["selected_correct"].to_numpy(dtype=np.int8)
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


def task_macro_auc(frame: pd.DataFrame, probability: np.ndarray) -> tuple[float, int]:
    values: list[float] = []
    for indices in frame.groupby("task_id", sort=False).indices.values():
        labels = frame.iloc[indices]["selected_correct"].to_numpy(dtype=np.int8)
        if np.unique(labels).size == 2:
            values.append(float(roc_auc_score(labels, probability[np.asarray(indices, dtype=np.int64)])))
    return (float(np.mean(values)) if values else float("nan"), len(values))


def aggregate(args: argparse.Namespace) -> None:
    _runtime, frame, manifest = load_prepared(args)
    labels = frame["selected_correct"].to_numpy(dtype=np.int8)
    groups = frame["task_id"].to_numpy(dtype=str)
    indices = np.arange(len(frame), dtype=np.int64)
    splits = list(GroupKFold(n_splits=args.n_splits).split(indices, labels, groups))
    probability = np.full(len(frame), np.nan, dtype=np.float64)
    fold_id = np.full(len(frame), -1, dtype=np.int8)
    manifest_hash = file_hash(args.output_dir / PREPARED_MANIFEST)
    fold_auc: list[float] = []
    vocabulary_sizes: list[int] = []
    for fold in range(1, args.n_splits + 1):
        path = fold_path(args.output_dir, fold)
        if not path.is_file():
            raise FileNotFoundError(f"Missing selected-answer content fold checkpoint: {path}")
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
            raise RuntimeError(f"Invalid or foreign selected-answer content fold checkpoint: {path}")
        if np.isfinite(probability[test_indices]).any():
            raise RuntimeError(f"Overlapping selected-answer content OOF rows in fold {fold}.")
        probability[test_indices] = values
        fold_id[test_indices] = fold
        fold_auc.append(float(binding["fold_auc"]))
        vocabulary_sizes.append(int(binding["vocabulary_size"]))
    if not np.isfinite(probability).all() or (fold_id < 1).any():
        raise RuntimeError("Selected-answer content fold checkpoints do not cover every row exactly once.")
    metrics_path = args.output_dir / "plurality_supporter_reasoning_only_thought512_metrics.json"
    predictions_path = args.output_dir / "plurality_supporter_reasoning_only_thought512_predictions.csv"
    if metrics_path.exists() or predictions_path.exists():
        raise FileExistsError("Final selected-answer content metrics already exist; use a new output directory.")
    macro_auc, macro_tasks = task_macro_auc(frame, probability)
    per_domain: dict[str, dict[str, float | int]] = {}
    for domain, group in frame.groupby("domain", sort=True):
        position = group.index.to_numpy(dtype=np.int64)
        target = labels[position]
        if np.unique(target).size == 2:
            per_domain[str(domain)] = {
                "decisions": int(len(group)),
                "selected_accuracy": float(target.mean()),
                "oof_auc": float(roc_auc_score(target, probability[position])),
            }
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "configuration": asdict(CONFIG),
        "prepared_manifest_sha256": manifest_hash,
        "decision_rows": int(len(frame)),
        "task_groups": int(frame["task_id"].nunique()),
        "fold_auc": fold_auc,
        "vocabulary_sizes": vocabulary_sizes,
        "oof_auc": float(roc_auc_score(labels, probability)),
        "raw_brier": float(brier_score_loss(labels, probability)),
        "raw_ece_15": expected_calibration_error(labels, probability),
        "task_macro_auc": macro_auc,
        "tasks_with_both_labels": macro_tasks,
        "task_cluster_bootstrap_auc_95_ci": task_bootstrap_auc(frame, probability, replicates=1000, seed=args.seed),
        "per_domain": per_domain,
        "strict_contract": manifest["strict_contract"],
        "interpretation": "retrospective selected-answer content OOF screen; do not compare or pool with raw stopping OOF AUC.",
    }
    prediction = frame[["task_id", "step", "domain", "selected_correct", "document_sha256", "plurality_supporter_count", "document_chars"]].copy()
    prediction["outer_fold"] = fold_id
    prediction["oof_probability"] = probability
    atomic_csv(predictions_path, prediction)
    atomic_npz(args.output_dir / "plurality_supporter_reasoning_only_thought512_oof.npz", probability=probability, outer_fold=fold_id)
    atomic_json(metrics_path, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


def main() -> int:
    args = parse_args()
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least two.")
    action = "prepare" if args.prepare else (f"fold_{args.fold:02d}" if args.fold is not None else "aggregate")
    status_path = args.status_file or args.output_dir / "selected_answer_content_status.json"
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
