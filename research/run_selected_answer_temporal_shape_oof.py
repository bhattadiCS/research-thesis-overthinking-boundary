#!/usr/bin/env python3
"""Strict OOF ablation for anonymous, coordinate-safe temporal shape.

The stored hidden projections have model-specific coordinate systems.  This
experiment calculates prefix-only shape scalars inside each model's own
trajectory, robust-normalizes those scalars per model using only outer-training
tasks, and finally aggregates them anonymously at a completed decision barrier.
Model identities and raw coordinates are never learner features.

It is one predeclared historical ablation against the existing coordinate-safe
kinematics result.  It scores decision-level selected-answer correctness, not
raw trajectory stopping, and remains retrospective because peer-completion
timestamps are absent from the legacy traces.
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
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


SCHEMA_VERSION = "selected-answer-temporal-shape-oof-v3"
DEFAULT_INPUT = Path("research/outputs/experiments_v2")
DEFAULT_OUTPUT = Path("research/outputs/experiments_v2/selected_answer_temporal_shape_oof_v3")
PREPARED_DECISIONS = "prepared_base_decisions.pkl"
PREPARED_RESPONSES = "prepared_response_shape_rows.pkl"
PREPARED_MANIFEST = "prepared_manifest.json"
FEATURE_SET_BASE_SANITIZED = "base_sanitized"
FEATURE_SET_BASE_SANITIZED_SHAPE = "base_sanitized_plus_temporal_shape"
FEATURE_SETS = (FEATURE_SET_BASE_SANITIZED, FEATURE_SET_BASE_SANITIZED_SHAPE)
COORDINATE_DEPENDENT_BASE_TOKENS = ("hidden_norm", "hidden_l2_shift", "hidden_cosine_shift")


@dataclass(frozen=True)
class ModelConfig:
    name: str = "regularized_lgbm_v1"
    n_estimators: int = 800
    learning_rate: float = 0.030
    num_leaves: int = 31
    min_child_samples: int = 60
    subsample: float = 0.90
    colsample_bytree: float = 0.78
    reg_lambda: float = 8.0
    reg_alpha: float = 0.20


CONFIG = ModelConfig()


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
        "selected_answer_temporal_shape.py": file_hash(root / "selected_answer_temporal_shape.py"),
        "run_selected_answer_oof_experiments.py": file_hash(root / "run_selected_answer_oof_experiments.py"),
        "run_selected_answer_representation_oof_experiments.py": file_hash(root / "run_selected_answer_representation_oof_experiments.py"),
    }


def runtime_contract() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
    }


def decision_key_hash(frame: pd.DataFrame) -> str:
    values = [(str(task), int(step)) for task, step in frame[["task_id", "step"]].itertuples(index=False, name=None)]
    return hashlib.sha256(canonical_json(values)).hexdigest()


def response_key_hash(frame: pd.DataFrame) -> str:
    values = [
        (str(task), int(step), str(alias))
        for task, step, alias in frame[["task_id", "step", "model_alias"]].itertuples(index=False, name=None)
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
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default=FEATURE_SET_BASE_SANITIZED_SHAPE)
    parser.add_argument("--status-file", type=Path, default=None)
    return parser.parse_args()


def prepare(args: argparse.Namespace) -> None:
    reference = load_module("run_selected_answer_oof_experiments.py", "temporal_shape_reference")
    representation = load_module("run_selected_answer_representation_oof_experiments.py", "temporal_shape_representation")
    shape = load_module("selected_answer_temporal_shape.py", "temporal_shape_features")
    raw, input_files = reference.load_panel(args.input_dir)
    decisions, base_features, categories, aliases = reference.build_decision_frame(raw, exclude_batch_timing=True)
    if categories:
        raise AssertionError("Identity categorical fields are forbidden in the temporal-shape ablation.")
    responses, raw_sources, shape_diagnostics = shape.build_response_shape_rows(raw, decisions, reference, representation)
    if args.n_splits < 2 or decisions["task_id"].nunique() < args.n_splits:
        raise ValueError("Insufficient task groups for requested GroupKFold protocol.")
    base_sanitized_features = [
        feature for feature in base_features if not any(token in feature for token in COORDINATE_DEPENDENT_BASE_TOKENS)
    ]
    if not base_sanitized_features or len(base_sanitized_features) == len(base_features):
        raise AssertionError("Coordinate-dependent base feature exclusion did not take effect.")
    base_columns = list(dict.fromkeys(["task_id", "step", "domain", "selected_correct", *base_sanitized_features]))
    base = decisions[base_columns].copy()
    stored_response = responses[["task_id", "step", "model_alias", "_winner", *raw_sources]].copy()
    if (
        base.duplicated(["task_id", "step"]).any()
        or stored_response.duplicated(["task_id", "step", "model_alias"]).any()
        or len(stored_response) != len(raw)
    ):
        raise AssertionError("Prepared temporal-shape rows do not meet the barrier/trajectory contract.")
    reference.validate_feature_contract(base_sanitized_features, [])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = args.output_dir / PREPARED_DECISIONS
    responses_path = args.output_dir / PREPARED_RESPONSES
    manifest_path = args.output_dir / PREPARED_MANIFEST
    if any(path.exists() for path in (decisions_path, responses_path, manifest_path)):
        raise FileExistsError("Prepared temporal-shape artifacts already exist; use a new output directory.")
    atomic_pickle(decisions_path, base)
    atomic_pickle(responses_path, stored_response)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "source_hashes": source_hashes(),
        "runtime_contract": runtime_contract(),
        "input_dir": str(args.input_dir),
        "input_files": input_files,
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "model_config": asdict(CONFIG),
        "decision_rows": int(len(base)),
        "response_rows": int(len(stored_response)),
        "task_groups": int(base["task_id"].nunique()),
        "base_sanitized_feature_columns": list(base_sanitized_features),
        "excluded_coordinate_dependent_base_features": sorted(set(base_features) - set(base_sanitized_features)),
        "state_shape_sources": raw_sources,
        "decision_key_sha256": decision_key_hash(base),
        "response_key_sha256": response_key_hash(stored_response),
        "artifact_sha256": {"decisions": file_hash(decisions_path), "responses": file_hash(responses_path)},
        "shape_diagnostics": shape_diagnostics,
        "frozen_alias_order_used_only_for_selection_ties": list(aliases),
        "strict_contract": {
            "target": "decision-level selected-answer correctness",
            "selection": "deterministic nonempty normalized-answer plurality; frozen-alias tie break",
            "geometry": "same-model/layer current-and-prior difference geometry only; no raw coordinate, state-origin cosine, or cross-model coordinate feature",
            "normalization": "per-model/per-step robust median/IQR and tie-aware ECDF fit only on each outer-fold training task set, then anonymized barrier aggregation",
            "base_control": "all hidden_norm/hidden_l2_shift/hidden_cosine_shift aggregate fields removed from both control and temporal-shape feature sets",
            "learner_excluded": "model/task/run/source identifiers, raw coordinates, answer values, text, expected/gold labels, timing, K2, future barriers",
            "historical_limit": "same-step peer completion lacks event timestamps",
            "interpretation": "one predeclared retrospective ablation, not a deployable stopping claim",
        },
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps({"prepared": str(args.output_dir), "decisions": len(base), "responses": len(stored_response), "shape_sources": len(raw_sources)}, sort_keys=True), flush=True)


def load_prepared(args: argparse.Namespace) -> tuple[Any, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    decisions_path = args.output_dir / PREPARED_DECISIONS
    responses_path = args.output_dir / PREPARED_RESPONSES
    manifest_path = args.output_dir / PREPARED_MANIFEST
    if not all(path.is_file() for path in (decisions_path, responses_path, manifest_path)):
        raise FileNotFoundError("Prepared temporal-shape artifacts are missing; run --prepare first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("source_hashes") != source_hashes()
        or manifest.get("runtime_contract") != runtime_contract()
        or manifest.get("input_dir") != str(args.input_dir)
        or int(manifest.get("n_splits", -1)) != args.n_splits
        or int(manifest.get("seed", -1)) != args.seed
        or manifest.get("model_config") != asdict(CONFIG)
        or file_hash(decisions_path) != manifest.get("artifact_sha256", {}).get("decisions")
        or file_hash(responses_path) != manifest.get("artifact_sha256", {}).get("responses")
    ):
        raise RuntimeError("Prepared temporal-shape artifacts have a different source/split/runtime contract.")
    decisions = pd.read_pickle(decisions_path)
    responses = pd.read_pickle(responses_path)
    base_features = manifest.get("base_sanitized_feature_columns")
    sources = manifest.get("state_shape_sources")
    required_decisions = {"task_id", "step", "domain", "selected_correct"}
    required_responses = {"task_id", "step", "model_alias", "_winner"}
    if (
        not isinstance(base_features, list)
        or not isinstance(sources, list)
        or required_decisions - set(decisions.columns)
        or required_responses - set(responses.columns)
        or set(base_features) - set(decisions.columns)
        or set(sources) - set(responses.columns)
        or decisions.duplicated(["task_id", "step"]).any()
        or responses.duplicated(["task_id", "step", "model_alias"]).any()
        or len(decisions) != int(manifest.get("decision_rows", -1))
        or len(responses) != int(manifest.get("response_rows", -1))
        or decisions["task_id"].nunique() != int(manifest.get("task_groups", -1))
        or decision_key_hash(decisions) != manifest.get("decision_key_sha256")
        or response_key_hash(responses) != manifest.get("response_key_sha256")
    ):
        raise RuntimeError("Prepared temporal-shape frames are malformed or misaligned.")
    reference = load_module("run_selected_answer_oof_experiments.py", "temporal_shape_reference")
    reference.validate_feature_contract(base_features, [])
    return reference, decisions, responses, manifest


def robust_normalize_and_aggregate(
    decisions: pd.DataFrame,
    responses: pd.DataFrame,
    sources: list[str],
    train_task_ids: set[str],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Fit per-alias/per-step z+ECDF transforms only on outer-train tasks."""
    work = responses.copy()
    work["task_id"] = work["task_id"].astype(str)
    fit_mask = work["task_id"].isin(train_task_ids).to_numpy(dtype=bool)
    if not fit_mask.any():
        raise AssertionError("Outer training set contains no response rows for shape normalization.")
    normalized_sources: list[str] = []
    fallback_scales = 0
    normalizer_records: list[tuple[str, int, str, float, float, int]] = []
    for source in sources:
        z_output = source.replace("shape_state_", "shape_z_")
        ecdf_output = source.replace("shape_state_", "shape_ecdf_")
        normalized_sources.extend((z_output, ecdf_output))
        values = pd.to_numeric(work[source], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
        z_values = np.full(len(work), np.nan, dtype=np.float64)
        ecdf_values = np.full(len(work), np.nan, dtype=np.float64)
        for (alias, step), positions in work.groupby(["model_alias", "step"], sort=False).indices.items():
            indices = np.asarray(positions, dtype=np.int64)
            fit_values = values[indices[fit_mask[indices]]]
            fit_values = fit_values[np.isfinite(fit_values)]
            if not len(fit_values):
                continue
            median = float(np.median(fit_values))
            scale = float(np.quantile(fit_values, 0.75) - np.quantile(fit_values, 0.25))
            if not np.isfinite(scale) or scale < 1.0e-8:
                scale = 1.0
                fallback_scales += 1
            valid = indices[np.isfinite(values[indices])]
            z_values[valid] = np.clip((values[valid] - median) / scale, -8.0, 8.0)
            ordered = np.sort(fit_values)
            left = np.searchsorted(ordered, values[valid], side="left")
            right = np.searchsorted(ordered, values[valid], side="right")
            # Tie-aware empirical CDF using only the outer-training sample.
            ecdf_values[valid] = (left + right + 1.0) / (2.0 * (len(ordered) + 1.0))
            normalizer_records.append((str(alias), int(step), source, median, scale, int(len(ordered))))
        work[z_output] = z_values
        work[ecdf_output] = ecdf_values

    # Cross-layer comparisons are permitted only after each layer's scalar was
    # independently normalized; there is never a direct l1/l2 vector dot product.
    for primitive in ("speed_share", "path_efficiency", "mean_turn_prefix"):
        left = f"shape_ecdf_l1_{primitive}"
        right = f"shape_ecdf_l2_{primitive}"
        if left in work.columns and right in work.columns:
            output = f"shape_ecdf_l1_minus_l2_{primitive}"
            left_values = work[left].to_numpy(dtype=np.float64)
            right_values = work[right].to_numpy(dtype=np.float64)
            work[output] = np.where(
                np.isfinite(left_values) & np.isfinite(right_values), left_values - right_values, np.nan
            )
            normalized_sources.append(output)

    # Materialize the per-barrier statistics with vectorized groupby calls.
    # The earlier row-by-row implementation was correct but prohibitively
    # slow on 14,740 barriers × 43 transformed scalar channels.
    keys = ["task_id", "step"]
    barrier_index = pd.MultiIndex.from_frame(decisions[keys])
    feature_data: dict[str, np.ndarray] = {}
    feature_columns: list[str] = []
    stat_names = ("median", "iqr", "q10", "q90", "available_fraction")
    panel_mask = np.ones(len(work), dtype=bool)
    winner_mask = work["_winner"].to_numpy(dtype=bool)
    nonwinner_mask = ~winner_mask

    def population_statistics(mask: np.ndarray, column: str) -> dict[str, pd.Series]:
        subset = work.loc[mask, [*keys, column]]
        grouped = subset.groupby(keys, sort=False, observed=True)[column]
        total = subset.groupby(keys, sort=False, observed=True).size()
        quantiles = grouped.quantile([0.10, 0.25, 0.75, 0.90]).unstack(level=-1)
        return {
            "median": grouped.median(),
            "iqr": quantiles[0.75] - quantiles[0.25],
            "q10": quantiles[0.10],
            "q90": quantiles[0.90],
            "available_fraction": grouped.count() / total,
        }

    for normalized in normalized_sources:
        short = normalized.removeprefix("shape_")
        panel = population_statistics(panel_mask, normalized)
        winner = population_statistics(winner_mask, normalized)
        nonwinner = population_statistics(nonwinner_mask, normalized)
        for population, stats in (("panel", panel), ("winner", winner)):
            for stat in stat_names:
                name = f"shape_{population}_{short}_{stat}"
                feature_columns.append(name)
                values = stats[stat].reindex(barrier_index)
                feature_data[name] = values.fillna(0.0).to_numpy(dtype=np.float64) if stat == "available_fraction" else values.to_numpy(dtype=np.float64)
        delta_name = f"shape_winner_vs_nonwinner_{short}_median_delta"
        feature_columns.append(delta_name)
        delta = winner["median"].reindex(barrier_index) - nonwinner["median"].reindex(barrier_index)
        usable = (winner["available_fraction"].reindex(barrier_index).fillna(0.0) > 0.0) & (
            nonwinner["available_fraction"].reindex(barrier_index).fillna(0.0) > 0.0
        )
        feature_data[delta_name] = delta.where(usable, np.nan).to_numpy(dtype=np.float64)
    result = decisions[keys].copy()
    for name in feature_columns:
        result[name] = feature_data[name]
    if len(result) != len(decisions) or result.duplicated(["task_id", "step"]).any():
        raise AssertionError("Fold temporal-shape aggregation did not cover each decision barrier exactly once.")
    diagnostics = {
        "normalizer_fit_task_groups": int(len(train_task_ids)),
        "normalizer_aliases": int(work["model_alias"].nunique()),
        "normalizer_alias_step_source_groups": int(len(normalizer_records)),
        "normalizer_iqr_fallbacks": int(fallback_scales),
        "normalizer_parameters_sha256": hashlib.sha256(canonical_json(sorted(normalizer_records))).hexdigest(),
        "fold_shape_feature_count": int(len(feature_columns)),
    }
    return result, feature_columns, diagnostics


def fold_path(output_dir: Path, feature_set: str, fold: int) -> Path:
    return output_dir / "fold_checkpoints" / f"{feature_set}_regularized_lgbm_fold_{fold:02d}.npz"


def train_fold(args: argparse.Namespace) -> None:
    reference, decisions, responses, manifest = load_prepared(args)
    if args.fold is None or not 1 <= args.fold <= args.n_splits:
        raise ValueError(f"--fold must be in [1, {args.n_splits}].")
    labels = decisions["selected_correct"].to_numpy(dtype=np.int8)
    groups = decisions["task_id"].astype(str).to_numpy()
    indices = np.arange(len(decisions), dtype=np.int64)
    outer_train, outer_test = (
        np.asarray(value, dtype=np.int64)
        for value in list(GroupKFold(n_splits=args.n_splits).split(indices, labels, groups))[args.fold - 1]
    )
    if set(groups[outer_train]) & set(groups[outer_test]):
        raise AssertionError("Task leakage in temporal-shape outer fold.")
    checkpoint = fold_path(args.output_dir, args.feature_set, args.fold)
    if checkpoint.exists():
        raise FileExistsError(f"Immutable temporal-shape fold checkpoint already exists: {checkpoint}")
    base_features = list(manifest["base_sanitized_feature_columns"])
    if args.feature_set == FEATURE_SET_BASE_SANITIZED_SHAPE:
        shape_frame, shape_features, normalization = robust_normalize_and_aggregate(
            decisions,
            responses,
            list(manifest["state_shape_sources"]),
            set(groups[outer_train].tolist()),
        )
        frame = decisions.merge(shape_frame, on=["task_id", "step"], how="left", validate="one_to_one", sort=False)
        features = [*base_features, *shape_features]
    else:
        shape_features = []
        normalization = {"status": "not_used_base_sanitized_control"}
        frame = decisions.copy()
        features = base_features
    if len(features) != len(set(features)):
        raise AssertionError("Temporal-shape learner feature columns are not unique.")
    reference.validate_feature_contract(features, [])
    if len(frame) != len(decisions) or (shape_features and frame[shape_features].isna().all(axis=None)):
        raise AssertionError("Temporal-shape fold frame alignment failed.")
    train_x, test_x = reference.make_matrices(frame, outer_train, outer_test, features, [])
    model = LGBMClassifier(
        objective="binary",
        n_estimators=CONFIG.n_estimators,
        learning_rate=CONFIG.learning_rate,
        num_leaves=CONFIG.num_leaves,
        min_child_samples=CONFIG.min_child_samples,
        subsample=CONFIG.subsample,
        subsample_freq=1,
        colsample_bytree=CONFIG.colsample_bytree,
        reg_lambda=CONFIG.reg_lambda,
        reg_alpha=CONFIG.reg_alpha,
        n_jobs=args.jobs,
        random_state=args.seed + args.fold,
        verbosity=-1,
    )
    started = time.time()
    model.fit(train_x, labels[outer_train])
    probability = model.predict_proba(test_x)[:, 1].astype(np.float64)
    binding = {
        "schema_version": SCHEMA_VERSION,
        "prepared_manifest_sha256": file_hash(args.output_dir / PREPARED_MANIFEST),
        "model_config": asdict(CONFIG),
        "fold": int(args.fold),
        "feature_set": args.feature_set,
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "jobs": int(args.jobs),
        "runtime_contract": runtime_contract(),
        "test_indices_sha256": hashlib.sha256(outer_test.tobytes()).hexdigest(),
        "feature_sha256": hashlib.sha256(canonical_json(features)).hexdigest(),
        "feature_count": int(len(features)),
        "normalization": normalization,
        "elapsed_seconds": time.time() - started,
        "fold_auc": float(roc_auc_score(labels[outer_test], probability)),
    }
    atomic_npz(
        checkpoint,
        test_indices=outer_test,
        probabilities=probability,
        binding_json=np.asarray(json.dumps(binding, sort_keys=True)),
    )
    print(json.dumps({"fold": args.fold, "auc": binding["fold_auc"], "features": binding["feature_count"]}, sort_keys=True), flush=True)


def aggregate(args: argparse.Namespace) -> None:
    reference, decisions, _responses, manifest = load_prepared(args)
    labels = decisions["selected_correct"].to_numpy(dtype=np.int8)
    groups = decisions["task_id"].astype(str).to_numpy()
    indices = np.arange(len(decisions), dtype=np.int64)
    splits = list(GroupKFold(n_splits=args.n_splits).split(indices, labels, groups))
    probabilities = np.full(len(decisions), np.nan, dtype=np.float64)
    fold_id = np.full(len(decisions), -1, dtype=np.int8)
    manifest_hash = file_hash(args.output_dir / PREPARED_MANIFEST)
    fold_auc: list[float] = []
    normalization: list[dict[str, Any]] = []
    feature_hashes: list[str] = []
    for fold in range(1, args.n_splits + 1):
        path = fold_path(args.output_dir, args.feature_set, fold)
        if not path.is_file():
            raise FileNotFoundError(f"Missing temporal-shape fold checkpoint: {path}")
        with np.load(path, allow_pickle=False) as payload:
            test_indices = np.asarray(payload["test_indices"], dtype=np.int64)
            values = np.asarray(payload["probabilities"], dtype=np.float64)
            binding = json.loads(str(payload["binding_json"].item()))
        expected = np.asarray(splits[fold - 1][1], dtype=np.int64)
        if (
            binding.get("schema_version") != SCHEMA_VERSION
            or binding.get("prepared_manifest_sha256") != manifest_hash
            or binding.get("model_config") != asdict(CONFIG)
            or binding.get("fold") != fold
            or binding.get("feature_set") != args.feature_set
            or binding.get("n_splits") != args.n_splits
            or binding.get("seed") != args.seed
            or binding.get("jobs") != args.jobs
            or binding.get("runtime_contract") != runtime_contract()
            or binding.get("test_indices_sha256") != hashlib.sha256(expected.tobytes()).hexdigest()
            or not np.array_equal(test_indices, expected)
            or len(values) != len(test_indices)
            or len(np.unique(test_indices)) != len(test_indices)
            or not np.isfinite(values).all()
        ):
            raise RuntimeError(f"Invalid or foreign temporal-shape checkpoint: {path}")
        if np.isfinite(probabilities[test_indices]).any():
            raise RuntimeError(f"Temporal-shape OOF overlap in fold {fold}.")
        probabilities[test_indices] = values
        fold_id[test_indices] = fold
        fold_auc.append(float(binding["fold_auc"]))
        normalization.append(dict(binding["normalization"]))
        feature_hashes.append(str(binding["feature_sha256"]))
    if not np.isfinite(probabilities).all() or (fold_id < 1).any():
        raise RuntimeError("Temporal-shape checkpoints do not cover every decision exactly once.")
    metrics_path = args.output_dir / f"{args.feature_set}_regularized_lgbm_metrics.json"
    predictions_path = args.output_dir / f"{args.feature_set}_regularized_lgbm_predictions.csv"
    if metrics_path.exists() or predictions_path.exists():
        raise FileExistsError("Final temporal-shape metrics already exist; use a new output directory.")
    metrics = reference.score_summary(decisions, probabilities, bootstrap_replicates=1000, seed=args.seed)
    metrics.update(
        {
            "schema_version": SCHEMA_VERSION,
            "model_config": asdict(CONFIG),
            "prepared_manifest_sha256": manifest_hash,
            "fold_auc": fold_auc,
            "fold_normalization": normalization,
            "fold_feature_sha256": feature_hashes,
            "feature_set": args.feature_set,
            "interpretation": "strict coordinate-safe retrospective selected-answer ablation; not a raw stopping or prospective deployment result.",
        }
    )
    prediction = decisions[["task_id", "step", "domain", "selected_correct"]].copy()
    prediction["outer_fold"] = fold_id
    prediction["oof_probability"] = probabilities
    atomic_csv(predictions_path, prediction)
    atomic_npz(args.output_dir / f"{args.feature_set}_regularized_lgbm_oof.npz", probability=probabilities, outer_fold=fold_id)
    atomic_json(metrics_path, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


def main() -> int:
    args = parse_args()
    if args.n_splits < 2 or args.jobs < 1:
        raise ValueError("--n-splits must be at least two and --jobs must be positive.")
    action = "prepare" if args.prepare else (f"fold_{args.fold:02d}" if args.fold is not None else "aggregate")
    status_path = args.status_file or args.output_dir / "temporal_shape_status.json"
    atomic_json(status_path, {"schema_version": SCHEMA_VERSION, "state": "running", "action": action, "started_at_unix": time.time()})
    try:
        if args.prepare:
            prepare(args)
        elif args.fold is not None:
            train_fold(args)
        else:
            aggregate(args)
    except Exception as error:
        atomic_json(status_path, {"schema_version": SCHEMA_VERSION, "state": "failed", "action": action, "finished_at_unix": time.time(), "error": f"{type(error).__name__}: {error}"})
        raise
    atomic_json(status_path, {"schema_version": SCHEMA_VERSION, "state": "complete", "action": action, "finished_at_unix": time.time()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
