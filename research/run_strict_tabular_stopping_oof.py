#!/usr/bin/env python3
"""Resumable, source-qualified tabular baseline for causal stopping research.

This experiment deliberately tests a different hypothesis from the historical
all-cell ``0.8656`` result.  It uses source-qualified trajectories, holds out
raw ``task_id`` groups, and learns only from features observable after the
current reasoning step.  Batch timing fields and model-specific raw projection
coordinates are excluded.  Within-trajectory scalar geometry remains allowed.

The three heads are compatible with the published stopping rule:

* ``q_t = P(correct_t | causal telemetry)``;
* ``r_t = P(correct_{t+1} | incorrect_t, causal telemetry)``;
* ``c_t = P(incorrect_{t+1} | correct_t, causal telemetry)``.

Run bounded stages on constrained machines::

    python research/run_strict_tabular_stopping_oof.py --prepare
    python research/run_strict_tabular_stopping_oof.py --fold 1
    # ... folds 2--5 ...
    python research/run_strict_tabular_stopping_oof.py --aggregate

No partial set of folds can be aggregated.  Each fold checkpoint is bound to
the expected task-GroupKFold test indices and prepared artifact hash.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


SCHEMA_VERSION = "strict-tabular-stopping-oof-v1"
DEFAULT_INPUT = Path("research/outputs/experiments_v2")
DEFAULT_OUTPUT = Path("research/outputs/experiments_v2/strict_tabular_stopping_oof_v1")
PREPARED_STORE = "prepared_store.npz"
PREPARED_MANIFEST = "prepared_manifest.json"

# These fields are determined by execution environment/batching rather than the
# completed reasoning state.  They are intentionally unavailable to the model.
EXCLUDED_TIMING_FEATURES = {
    "elapsed_seconds",
    "elapsed_seconds__missing",
    "tokens_per_second",
    "tokens_per_second__missing",
}

# These depend on arbitrary absolute raw-coordinate bases rather than
# within-trajectory motion, so pooled multi-model learning must not use them in
# the coordinate-safe primary analysis.
EXCLUDED_COORDINATE_DERIVED_FEATURES = {
    "conformal_phi",
    "renyi_state_divergence",
}


@dataclass(frozen=True)
class TabularConfig:
    name: str = "safe_scalar_lgbm_v1"
    n_estimators: int = 700
    learning_rate: float = 0.030
    num_leaves: int = 31
    min_child_samples: int = 120
    subsample: float = 0.90
    colsample_bytree: float = 0.80
    reg_lambda: float = 8.0
    reg_alpha: float = 0.20


CONFIG = TabularConfig()


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
        "lightgbm": lightgbm.__version__,
    }


def is_raw_projection_coordinate(name: str) -> bool:
    return (
        (name.startswith("mid1_") or name.startswith("mid2_"))
        and len(name) == 8
        and name[-3:].isdigit()
    )


def safe_feature_columns(names: list[str]) -> list[str]:
    selected = [
        name
        for name in names
        if (
            name not in EXCLUDED_TIMING_FEATURES
            and name not in EXCLUDED_COORDINATE_DERIVED_FEATURES
            and not is_raw_projection_coordinate(name)
        )
    ]
    if not selected or len(selected) != len(set(selected)):
        raise RuntimeError("The strict scalar feature contract is empty or has duplicates.")
    if any(
        name in EXCLUDED_TIMING_FEATURES
        or name in EXCLUDED_COORDINATE_DERIVED_FEATURES
        or is_raw_projection_coordinate(name)
        for name in selected
    ):
        raise AssertionError("Unsafe timing or raw coordinate feature escaped the contract.")
    return selected


def store_key_hash(store: Any) -> str:
    pairs = [
        (str(task), str(trajectory), int(length))
        for task, trajectory, length in zip(store.task_ids.tolist(), store.trajectory_ids.tolist(), store.lengths.tolist())
    ]
    return hashlib.sha256(canonical_json(pairs)).hexdigest()


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
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--persistent-homology", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--topology-window", type=int, default=5)
    parser.add_argument("--status-file", type=Path, default=None)
    return parser.parse_args()


def build_runtime_store(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any], list[str]]:
    runtime = load_module("run_ultimate_multi_day_tournament.py", "strict_tabular_runtime")
    loader_args = SimpleNamespace(
        input_dir=str(args.input_dir),
        include_all_cells=False,
        max_cells=None,
    )
    frame, input_manifest = runtime.load_trace_frame(loader_args)
    all_features = runtime.build_feature_frame(
        frame,
        include_persistence=bool(args.persistent_homology),
        topology_window=int(args.topology_window),
    )
    feature_columns = safe_feature_columns(list(all_features.columns))
    store = runtime.build_sequence_store(frame, all_features[feature_columns])
    return runtime, store, input_manifest, feature_columns


def prepare(args: argparse.Namespace) -> None:
    runtime, store, input_manifest, feature_columns = build_runtime_store(args)
    if args.n_splits < 2 or len(np.unique(store.task_ids)) < args.n_splits:
        raise ValueError("Insufficient distinct task IDs for requested GroupKFold protocol.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store_path = args.output_dir / PREPARED_STORE
    manifest_path = args.output_dir / PREPARED_MANIFEST
    if manifest_path.exists():
        raise FileExistsError("Prepared manifest already exists; use a new output directory.")
    atomic_npz(
        store_path,
        x=store.x.astype(np.float32, copy=False),
        y=store.y.astype(np.int64, copy=False),
        next_y=store.next_y.astype(np.int64, copy=False),
        lengths=store.lengths.astype(np.int64, copy=False),
        row_ids=store.row_ids.astype(np.int64, copy=False),
        steps=store.steps.astype(np.int64, copy=False),
        thought_tokens=store.thought_tokens.astype(np.float32, copy=False),
        generation_tokens=store.generation_tokens.astype(np.float32, copy=False),
        k2_tokens=store.k2_tokens.astype(np.float32, copy=False),
        k2_agreement=store.k2_agreement.astype(np.int64, copy=False),
        k2_available=store.k2_available.astype(np.int64, copy=False),
        trajectory_ids=np.asarray(store.trajectory_ids, dtype=str),
        task_ids=np.asarray(store.task_ids, dtype=str),
        source_cells=np.asarray(store.source_cells, dtype=str),
        feature_names=np.asarray(store.feature_names, dtype=str),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "source_hashes": source_hashes(),
        "runtime_contract": runtime_contract(),
        "input_dir": str(args.input_dir),
        "input_manifest": input_manifest,
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "persistent_homology": bool(args.persistent_homology),
        "topology_window": int(args.topology_window),
        "feature_columns": feature_columns,
        "feature_sha256": hashlib.sha256(canonical_json(feature_columns)).hexdigest(),
        "store_sha256": file_hash(store_path),
        "store_key_sha256": store_key_hash(store),
        "shape": {"runs": int(store.n_runs), "max_len": int(store.max_len), "features": len(store.feature_names)},
        "task_groups": int(len(np.unique(store.task_ids))),
        "strict_contract": {
            "outer_group": "task_id",
            "trajectory_key": "source_cell::run_id",
            "targets": "current correctness plus causal next-step repair/corruption hazards",
            "excluded": "elapsed_seconds, tokens_per_second, raw mid-layer projection coordinates, and coordinate-basis-dependent geometry",
            "allowed_geometry": "same-trajectory scalar dynamics derived from current and prior steps only",
            "model_selection": "fixed regularized LightGBM configuration; no outer-test tuning",
        },
    }
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {"prepared": str(args.output_dir), "runs": store.n_runs, "features": len(feature_columns), "tasks": manifest["task_groups"]},
            sort_keys=True,
        ),
        flush=True,
    )


def load_prepared(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    manifest_path = args.output_dir / PREPARED_MANIFEST
    store_path = args.output_dir / PREPARED_STORE
    if not manifest_path.is_file() or not store_path.is_file():
        raise FileNotFoundError("Prepared store is missing; run --prepare first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("source_hashes") != source_hashes()
        or manifest.get("runtime_contract") != runtime_contract()
        or manifest.get("input_dir") != str(args.input_dir)
        or int(manifest.get("n_splits", -1)) != args.n_splits
        or int(manifest.get("seed", -1)) != args.seed
        or bool(manifest.get("persistent_homology")) != bool(args.persistent_homology)
        or int(manifest.get("topology_window", -1)) != args.topology_window
        or file_hash(store_path) != manifest.get("store_sha256")
    ):
        raise RuntimeError("Prepared store has a different source, feature, split, seed, or runtime contract.")
    runtime = load_module("run_ultimate_multi_day_tournament.py", "strict_tabular_runtime")
    with np.load(store_path, allow_pickle=False) as payload:
        required = {
            "x", "y", "next_y", "lengths", "row_ids", "steps", "thought_tokens", "generation_tokens",
            "k2_tokens", "k2_agreement", "k2_available", "trajectory_ids", "task_ids", "source_cells", "feature_names",
        }
        if required - set(payload.files):
            raise RuntimeError("Prepared store payload is incomplete.")
        store = runtime.SequenceStore(
            x=np.asarray(payload["x"], dtype=np.float32),
            y=np.asarray(payload["y"], dtype=np.int64),
            next_y=np.asarray(payload["next_y"], dtype=np.int64),
            lengths=np.asarray(payload["lengths"], dtype=np.int64),
            row_ids=np.asarray(payload["row_ids"], dtype=np.int64),
            steps=np.asarray(payload["steps"], dtype=np.int64),
            thought_tokens=np.asarray(payload["thought_tokens"], dtype=np.float32),
            generation_tokens=np.asarray(payload["generation_tokens"], dtype=np.float32),
            k2_tokens=np.asarray(payload["k2_tokens"], dtype=np.float32),
            k2_agreement=np.asarray(payload["k2_agreement"], dtype=np.int64),
            k2_available=np.asarray(payload["k2_available"], dtype=np.int64),
            trajectory_ids=np.asarray(payload["trajectory_ids"], dtype=str),
            task_ids=np.asarray(payload["task_ids"], dtype=str),
            source_cells=np.asarray(payload["source_cells"], dtype=str),
            feature_names=[str(item) for item in np.asarray(payload["feature_names"], dtype=str).tolist()],
        )
    expected_features = manifest.get("feature_columns")
    if (
        not isinstance(expected_features, list)
        or expected_features != store.feature_names
        or hashlib.sha256(canonical_json(expected_features)).hexdigest() != manifest.get("feature_sha256")
        or store_key_hash(store) != manifest.get("store_key_sha256")
        or any(
            name in EXCLUDED_TIMING_FEATURES
            or name in EXCLUDED_COORDINATE_DERIVED_FEATURES
            or is_raw_projection_coordinate(name)
            for name in store.feature_names
        )
    ):
        raise RuntimeError("Prepared feature or trajectory contract is malformed.")
    shape = manifest.get("shape", {})
    if (
        shape.get("runs") != store.n_runs
        or shape.get("max_len") != store.max_len
        or shape.get("features") != len(store.feature_names)
        or manifest.get("task_groups") != len(np.unique(store.task_ids))
    ):
        raise RuntimeError("Prepared store shape/task metadata is inconsistent.")
    return runtime, store, manifest


def flat_view(store: Any, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = store.valid_mask_np(indices)
    has_next = np.arange(store.max_len)[None, :] < (store.lengths[indices, None] - 1)
    x = store.x[indices]
    return x[valid], store.y[indices][valid], store.next_y[indices][valid], valid, has_next


def probability_to_logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1.0e-6, 1.0 - 1.0e-6)
    return np.log(clipped / (1.0 - clipped))


def constant_or_lgbm(
    x: np.ndarray,
    target: np.ndarray,
    config: TabularConfig,
    seed: int,
    jobs: int,
) -> tuple[Any | None, float | None]:
    target = np.asarray(target, dtype=np.int64)
    if len(target) == 0:
        return None, 0.5
    values = np.unique(target)
    if len(values) < 2:
        return None, float(target.mean())
    model = LGBMClassifier(
        objective="binary",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        subsample_freq=1,
        colsample_bytree=config.colsample_bytree,
        reg_lambda=config.reg_lambda,
        reg_alpha=config.reg_alpha,
        max_bin=255,
        class_weight=None,
        n_jobs=jobs,
        random_state=seed,
        verbosity=-1,
    )
    model.fit(x, target)
    return model, None


def predict_probability(model: Any | None, constant: float | None, x: np.ndarray) -> np.ndarray:
    if model is None:
        return np.full(len(x), float(constant if constant is not None else 0.5), dtype=np.float64)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)


def train_head(
    train_x: np.ndarray,
    train_target: np.ndarray,
    calibration_x: np.ndarray,
    calibration_target: np.ndarray,
    test_x: np.ndarray,
    config: TabularConfig,
    seed: int,
    jobs: int,
    runtime: Any,
) -> np.ndarray:
    model, constant = constant_or_lgbm(train_x, train_target, config, seed, jobs)
    calibration_probability = predict_probability(model, constant, calibration_x)
    scaler = runtime.PlattScaler().fit(probability_to_logit(calibration_probability), calibration_target)
    test_probability = predict_probability(model, constant, test_x)
    return scaler.transform(probability_to_logit(test_probability)).astype(np.float32)


def fold_path(output_dir: Path, fold: int) -> Path:
    return output_dir / "fold_checkpoints" / f"safe_scalar_lgbm_fold_{fold:02d}.npz"


def train_fold(args: argparse.Namespace) -> None:
    runtime, store, manifest = load_prepared(args)
    if args.fold is None or not 1 <= args.fold <= args.n_splits:
        raise ValueError(f"--fold must be in [1, {args.n_splits}].")
    run_indices = np.arange(store.n_runs, dtype=np.int64)
    splits = list(GroupKFold(n_splits=args.n_splits).split(run_indices, groups=store.task_ids))
    outer_train, outer_test = (np.asarray(value, dtype=np.int64) for value in splits[args.fold - 1])
    runtime.assert_disjoint_task_groups(store.task_ids, outer_train, outer_test)
    _tuning_train, _tuning_validation, model_fit, calibration = runtime.outer_and_inner_partitions(
        store, outer_train, outer_test, args.fold - 1
    )
    checkpoint = fold_path(args.output_dir, args.fold)
    if checkpoint.exists():
        raise FileExistsError(f"Immutable fold checkpoint already exists: {checkpoint}")

    fit_x, fit_y, fit_next, fit_valid, fit_has_next = flat_view(store, model_fit)
    calibration_x, calibration_y, calibration_next, calibration_valid, calibration_has_next = flat_view(store, calibration)
    test_x, _test_y, _test_next, test_valid, _test_has_next = flat_view(store, outer_test)
    fit_repair = fit_has_next & (store.y[model_fit] == 0)
    fit_corruption = fit_has_next & (store.y[model_fit] == 1)
    calibration_repair = calibration_has_next & (store.y[calibration] == 0)
    calibration_corruption = calibration_has_next & (store.y[calibration] == 1)
    # Masks are shaped like the selected run panels.  Flatten them against the
    # same C-order rows used by ``flat_view`` only after restricting to valid
    # positions, so padded examples cannot enter any head.
    fit_valid_positions = np.flatnonzero(fit_valid.ravel())
    calibration_valid_positions = np.flatnonzero(calibration_valid.ravel())
    fit_repair_flat = fit_repair.ravel()[fit_valid_positions]
    fit_corruption_flat = fit_corruption.ravel()[fit_valid_positions]
    calibration_repair_flat = calibration_repair.ravel()[calibration_valid_positions]
    calibration_corruption_flat = calibration_corruption.ravel()[calibration_valid_positions]

    started = time.time()
    q = train_head(
        fit_x, fit_y, calibration_x, calibration_y, test_x, CONFIG, args.seed + 100 * args.fold + 1, args.jobs, runtime
    )
    repair = train_head(
        fit_x[fit_repair_flat],
        fit_next[fit_repair_flat],
        calibration_x[calibration_repair_flat],
        calibration_next[calibration_repair_flat],
        test_x,
        CONFIG,
        args.seed + 100 * args.fold + 2,
        args.jobs,
        runtime,
    )
    corruption = train_head(
        fit_x[fit_corruption_flat],
        1 - fit_next[fit_corruption_flat],
        calibration_x[calibration_corruption_flat],
        1 - calibration_next[calibration_corruption_flat],
        test_x,
        CONFIG,
        args.seed + 100 * args.fold + 3,
        args.jobs,
        runtime,
    )
    shape = (len(outer_test), store.max_len)
    q_panel = np.full(shape, np.nan, dtype=np.float32)
    repair_panel = np.full(shape, np.nan, dtype=np.float32)
    corruption_panel = np.full(shape, np.nan, dtype=np.float32)
    q_panel[test_valid] = q
    repair_panel[test_valid] = repair
    corruption_panel[test_valid] = corruption
    labels = store.y[outer_test][test_valid]
    binding = {
        "schema_version": SCHEMA_VERSION,
        "prepared_manifest_sha256": file_hash(args.output_dir / PREPARED_MANIFEST),
        "config": asdict(CONFIG),
        "fold": int(args.fold),
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "jobs": int(args.jobs),
        "runtime_contract": runtime_contract(),
        "test_indices_sha256": hashlib.sha256(outer_test.tobytes()).hexdigest(),
        "elapsed_seconds": time.time() - started,
        "fold_auc": float(roc_auc_score(labels, q)),
    }
    atomic_npz(
        checkpoint,
        test_indices=outer_test,
        q=q_panel,
        repair=repair_panel,
        corruption=corruption_panel,
        binding_json=np.asarray(json.dumps(binding, sort_keys=True)),
    )
    print(json.dumps({"fold": args.fold, "auc": binding["fold_auc"], "checkpoint": str(checkpoint)}, sort_keys=True), flush=True)


def task_bootstrap_auc(store: Any, probability: np.ndarray, replicates: int, seed: int) -> list[float]:
    mask = store.valid_mask_np()
    labels = store.y[mask]
    scores = probability[mask]
    run_task = np.repeat(store.task_ids, store.lengths)
    unique_tasks, inverse = np.unique(run_task, return_inverse=True)
    buckets = [np.flatnonzero(inverse == index) for index in range(len(unique_tasks))]
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        selected = rng.integers(0, len(unique_tasks), size=len(unique_tasks))
        indices = np.concatenate([buckets[index] for index in selected])
        if np.unique(labels[indices]).size == 2:
            values.append(float(roc_auc_score(labels[indices], scores[indices])))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] if values else [float("nan"), float("nan")]


def aggregate(args: argparse.Namespace) -> None:
    runtime, store, manifest = load_prepared(args)
    run_indices = np.arange(store.n_runs, dtype=np.int64)
    expected_splits = list(GroupKFold(n_splits=args.n_splits).split(run_indices, groups=store.task_ids))
    q = np.full((store.n_runs, store.max_len), np.nan, dtype=np.float32)
    repair = np.full_like(q, np.nan)
    corruption = np.full_like(q, np.nan)
    manifest_hash = file_hash(args.output_dir / PREPARED_MANIFEST)
    fold_auc: list[float] = []
    for fold in range(1, args.n_splits + 1):
        checkpoint = fold_path(args.output_dir, fold)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing strict fold checkpoint: {checkpoint}")
        with np.load(checkpoint, allow_pickle=False) as payload:
            test_indices = np.asarray(payload["test_indices"], dtype=np.int64)
            local_q = np.asarray(payload["q"], dtype=np.float32)
            local_repair = np.asarray(payload["repair"], dtype=np.float32)
            local_corruption = np.asarray(payload["corruption"], dtype=np.float32)
            binding = json.loads(str(payload["binding_json"].item()))
        expected = np.asarray(expected_splits[fold - 1][1], dtype=np.int64)
        expected_hash = hashlib.sha256(expected.tobytes()).hexdigest()
        if (
            binding.get("schema_version") != SCHEMA_VERSION
            or binding.get("prepared_manifest_sha256") != manifest_hash
            or binding.get("config") != asdict(CONFIG)
            or binding.get("fold") != fold
            or binding.get("n_splits") != args.n_splits
            or binding.get("seed") != args.seed
            or binding.get("jobs") != args.jobs
            or binding.get("runtime_contract") != runtime_contract()
            or binding.get("test_indices_sha256") != expected_hash
            or not np.array_equal(test_indices, expected)
            or len(np.unique(test_indices)) != len(test_indices)
            or local_q.shape != (len(test_indices), store.max_len)
            or local_repair.shape != local_q.shape
            or local_corruption.shape != local_q.shape
            or not np.isfinite(local_q[store.valid_mask_np(test_indices)]).all()
            or not np.isfinite(local_repair[store.valid_mask_np(test_indices)]).all()
            or not np.isfinite(local_corruption[store.valid_mask_np(test_indices)]).all()
        ):
            raise RuntimeError(f"Invalid or foreign fold checkpoint: {checkpoint}")
        if np.isfinite(q[test_indices][store.valid_mask_np(test_indices)]).any():
            raise RuntimeError(f"Overlapping OOF rows in fold {fold}.")
        q[test_indices] = local_q
        repair[test_indices] = local_repair
        corruption[test_indices] = local_corruption
        fold_auc.append(float(binding["fold_auc"]))
    valid = store.valid_mask_np()
    if not (np.isfinite(q[valid]).all() and np.isfinite(repair[valid]).all() and np.isfinite(corruption[valid]).all()):
        raise RuntimeError("Fold checkpoints do not cover every valid source-qualified trajectory step exactly once.")
    metrics_path = args.output_dir / "safe_scalar_lgbm_metrics.json"
    prediction_path = args.output_dir / "safe_scalar_lgbm_predictions.csv"
    if metrics_path.exists():
        raise FileExistsError("Final metrics already exist; use a new output directory for another run.")
    policy = runtime.evaluate_stopping_policy(store, run_indices, q, repair, corruption, hysteresis=False)
    step_utility, token_utility, win_tie_loss = runtime.policy_summary(policy)
    labels = store.y[valid]
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "configuration": asdict(CONFIG),
        "prepared_manifest_sha256": manifest_hash,
        "runs": int(store.n_runs),
        "task_groups": int(len(np.unique(store.task_ids))),
        "valid_steps": int(valid.sum()),
        "feature_count": len(store.feature_names),
        "fold_auc": fold_auc,
        "oof_auc": float(runtime.safe_auc(labels, q[valid])),
        "ece_15": float(runtime.calculate_ece(q[valid], labels)),
        "task_cluster_bootstrap_auc_95_ci": task_bootstrap_auc(store, q, replicates=1000, seed=args.seed),
        "step_utility": float(step_utility),
        "token_utility": float(token_utility),
        "win_tie_loss": str(win_tie_loss),
        "strict_contract": manifest["strict_contract"],
        "interpretation": "fixed-configuration, strict source-qualified retrospective baseline; no claim of fresh deployment validation.",
    }
    rows: list[dict[str, Any]] = []
    for run_index in range(store.n_runs):
        length = int(store.lengths[run_index])
        for position in range(length):
            rows.append(
                {
                    "task_id": str(store.task_ids[run_index]),
                    "trajectory_id": str(store.trajectory_ids[run_index]),
                    "source_cell": str(store.source_cells[run_index]),
                    "step": int(store.steps[run_index, position]),
                    "correct": int(store.y[run_index, position]),
                    "oof_q": float(q[run_index, position]),
                    "oof_repair": float(repair[run_index, position]),
                    "oof_corruption": float(corruption[run_index, position]),
                }
            )
    atomic_csv(prediction_path, pd.DataFrame(rows))
    atomic_npz(args.output_dir / "safe_scalar_lgbm_oof.npz", q=q, repair=repair, corruption=corruption)
    atomic_json(metrics_path, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


def main() -> int:
    args = parse_args()
    if args.n_splits < 2 or args.jobs < 1 or args.topology_window < 2:
        raise ValueError("--n-splits >= 2, --jobs >= 1, and --topology-window >= 2 are required.")
    action = "prepare" if args.prepare else (f"fold_{args.fold:02d}" if args.fold is not None else "aggregate")
    status_path = args.status_file or args.output_dir / "strict_tabular_status.json"
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
            {
                "schema_version": SCHEMA_VERSION,
                "state": "failed",
                "action": action,
                "finished_at_unix": time.time(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    atomic_json(status_path, {"schema_version": SCHEMA_VERSION, "state": "complete", "action": action, "finished_at_unix": time.time()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
