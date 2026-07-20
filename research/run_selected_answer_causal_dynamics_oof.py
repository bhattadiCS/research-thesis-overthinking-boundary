#!/usr/bin/env python3
"""Resumable strict OOF evaluation of causal hidden-state dynamics.

Use this in bounded stages on a constrained host:

1. ``--prepare-dynamics`` materializes coordinate-safe base plus causal
   dynamics features.  The optional ``--prepare-representation`` stage is only
   for the explicitly exploratory raw-space combined analysis.
2. ``--assemble`` validates immutable stage manifests and materializes a final
   prepared artifact.
3. ``--fold N`` fits exactly one task-held-out outer fold and writes an atomic
   checkpoint.
4. ``--aggregate`` verifies the exact expected GroupKFold partition before
   scoring the combined OOF vector.

The staged design makes an interruption incapable of converting a partial OOF
screen into a result.  It is intentionally separate from the prospective
collector: historical traces lack live peer-completion timestamps, so any
positive result remains a retrospective candidate until fresh collection.
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

import numpy as np
import pandas as pd
import lightgbm
import sklearn
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


SCHEMA_VERSION = "selected-answer-causal-dynamics-oof-v1"
DEFAULT_INPUT = Path("research/outputs/experiments_v2")
DEFAULT_OUTPUT = Path("research/outputs/experiments_v2/selected_answer_causal_dynamics_oof_v1")
FEATURE_SET_BASE_DYNAMICS = "base_dynamics"
FEATURE_SET_COMBINED_EXPLORATORY = "combined_exploratory"
FEATURE_SETS = (FEATURE_SET_BASE_DYNAMICS, FEATURE_SET_COMBINED_EXPLORATORY)
PREPARED_FRAME = "prepared_decisions.pkl"
PREPARED_COORDINATE = "prepared_coordinates.npy"
PREPARED_MANIFEST = "prepared_manifest.json"
REPRESENTATION_STAGE_FRAME = "stage_representation_decisions.pkl"
REPRESENTATION_STAGE_COORDINATE = "stage_representation_coordinates.npy"
REPRESENTATION_STAGE_MANIFEST = "stage_representation_manifest.json"
DYNAMICS_STAGE_FRAME = "stage_dynamics_decisions.pkl"
DYNAMICS_STAGE_MANIFEST = "stage_dynamics_manifest.json"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    n_estimators: int
    learning_rate: float
    num_leaves: int
    min_child_samples: int
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    reg_alpha: float


MODEL_CONFIGS: tuple[ModelConfig, ...] = (
    ModelConfig("regularized", 800, 0.030, 31, 60, 0.90, 0.78, 8.0, 0.20),
    ModelConfig("medium_capacity", 800, 0.030, 63, 35, 0.90, 0.85, 4.0, 0.10),
)


def load_module(filename: str, name: str) -> Any:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(value) + b"\n")
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


def atomic_pickle(path: Path, frame: pd.DataFrame) -> None:
    """Write a dataframe atomically so interruption cannot create a usable partial stage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".pkl") as handle:
        temporary = Path(handle.name)
    try:
        frame.to_pickle(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    """Write an ndarray atomically, including a stable filename extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".npy") as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
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


def storage_columns(features: list[str]) -> list[str]:
    """Keep key/label metadata once while allowing ``step`` as a real model feature."""
    columns = ["task_id", "step", "domain", "selected_correct", *features]
    return list(dict.fromkeys(columns))


def key_order_hash(frame: pd.DataFrame) -> str:
    if set(("task_id", "step")) - set(frame.columns):
        raise ValueError("A staged frame is missing task_id/step keys.")
    pairs = [(str(task_id), int(step)) for task_id, step in frame[["task_id", "step"]].itertuples(index=False, name=None)]
    return hashlib.sha256(canonical_bytes(pairs)).hexdigest()


def runtime_contract() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
    }


def config_by_name(name: str) -> ModelConfig:
    for config in MODEL_CONFIGS:
        if config.name == name:
            return config
    raise ValueError(f"Unknown model config {name!r}; choose one of {[item.name for item in MODEL_CONFIGS]}.")


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "run_selected_answer_causal_dynamics_oof.py": file_hash(Path(__file__).resolve()),
        "selected_answer_causal_dynamics.py": file_hash(root / "selected_answer_causal_dynamics.py"),
        "run_selected_answer_representation_oof_experiments.py": file_hash(root / "run_selected_answer_representation_oof_experiments.py"),
        "run_selected_answer_oof_experiments.py": file_hash(root / "run_selected_answer_oof_experiments.py"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true", help="Run every preparation stage (for an unconstrained host).")
    action.add_argument(
        "--prepare-representation",
        action="store_true",
        help="Materialize base plus representation features as an immutable stage.",
    )
    action.add_argument(
        "--prepare-dynamics",
        action="store_true",
        help="Materialize causal dynamics features as an immutable stage.",
    )
    action.add_argument(
        "--assemble",
        action="store_true",
        help="Validate both immutable stages and assemble the final prepared artifact.",
    )
    action.add_argument("--fold", type=int, help="One-based strict outer fold to train.")
    action.add_argument("--aggregate", action="store_true")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", default="regularized", choices=[config.name for config in MODEL_CONFIGS])
    parser.add_argument(
        "--feature-set",
        default=FEATURE_SET_BASE_DYNAMICS,
        choices=FEATURE_SETS,
        help="base_dynamics is the primary coordinate-safe analysis; combined_exploratory retains raw-space representation blocks.",
    )
    parser.add_argument("--pca-components", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--status-file", type=Path, default=None, help="Optional atomic lifecycle status JSON path.")
    return parser.parse_args()


def stage_manifest(
    args: argparse.Namespace,
    stage: str,
    input_files: Any,
    decision_rows: int,
    task_groups: int,
    artifact_hashes: dict[str, str],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "created_at_unix": time.time(),
        "source_hashes": source_hashes(),
        "input_dir": str(args.input_dir),
        "input_files": input_files,
        "decision_rows": int(decision_rows),
        "task_groups": int(task_groups),
        "artifact_sha256": artifact_hashes,
        **extra,
    }


def read_valid_stage(args: argparse.Namespace, path: Path, expected_stage: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing immutable {expected_stage} stage: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("stage") != expected_stage
        or payload.get("source_hashes") != source_hashes()
        or payload.get("input_dir") != str(args.input_dir)
    ):
        raise RuntimeError(f"{expected_stage} stage has a different source/input contract.")
    hashes = payload.get("artifact_sha256")
    if not isinstance(hashes, dict) or not all(isinstance(value, str) for value in hashes.values()):
        raise RuntimeError(f"{expected_stage} stage has no valid artifact hashes.")
    return payload


def require_unfinalized_stage(path: Path, stage: str) -> None:
    if path.exists():
        raise FileExistsError(f"Immutable {stage} stage already exists: {path}")


def prepare_representation_stage(args: argparse.Namespace) -> None:
    reference = load_module("run_selected_answer_oof_experiments.py", "selected_answer_reference")
    representation = load_module("run_selected_answer_representation_oof_experiments.py", "selected_answer_representation")
    raw, input_files = reference.load_panel(args.input_dir)
    decisions, base_features, categories, aliases = reference.build_decision_frame(raw, exclude_batch_timing=True)
    if categories:
        raise AssertionError("Identity categorical fields are forbidden in the strict dynamics experiment.")
    representation_frame, coordinate, representation_diagnostics = representation.build_representation_frame(raw, decisions, reference)
    representation_features = sorted(
        column for column in representation_frame.columns if column.startswith(representation.REP_PREFIX)
    )
    features = list(base_features) + representation_features
    reference.validate_feature_contract(features, [])
    forbidden = set(features) & set(reference.FORBIDDEN_FEATURES)
    if forbidden:
        raise AssertionError(f"Forbidden feature leakage: {sorted(forbidden)}")
    if len(features) != len(set(features)):
        raise AssertionError("Representation-stage feature contract is not unique.")
    stored_columns = storage_columns(features)
    stored = representation_frame[stored_columns].copy()
    if stored.columns.duplicated().any():
        raise AssertionError("Representation-stage feature columns are not unique.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = args.output_dir / REPRESENTATION_STAGE_FRAME
    coordinate_path = args.output_dir / REPRESENTATION_STAGE_COORDINATE
    manifest_path = args.output_dir / REPRESENTATION_STAGE_MANIFEST
    require_unfinalized_stage(manifest_path, "representation")
    atomic_pickle(frame_path, stored)
    atomic_npy(coordinate_path, coordinate.astype(np.float32, copy=False))
    manifest = stage_manifest(
        args,
        "representation",
        input_files,
        len(stored),
        stored["task_id"].nunique(),
        {"frame": file_hash(frame_path), "coordinate": file_hash(coordinate_path)},
        base_feature_columns=list(base_features),
        representation_feature_columns=representation_features,
        coordinate_shape=list(coordinate.shape),
        key_order_sha256=key_order_hash(stored),
        frozen_alias_order=list(aliases),
        representation_diagnostics=representation_diagnostics,
    )
    atomic_json(manifest_path, manifest)
    print(json.dumps({"stage": "representation", "rows": len(stored), "features": len(features)}, sort_keys=True), flush=True)


def prepare_dynamics_stage(args: argparse.Namespace) -> None:
    reference = load_module("run_selected_answer_oof_experiments.py", "selected_answer_reference")
    representation = load_module("run_selected_answer_representation_oof_experiments.py", "selected_answer_representation")
    dynamics = load_module("selected_answer_causal_dynamics.py", "selected_answer_causal_dynamics")
    raw, input_files = reference.load_panel(args.input_dir)
    decisions, base_features, categories, aliases = reference.build_decision_frame(raw, exclude_batch_timing=True)
    if categories:
        raise AssertionError("Identity categorical fields are forbidden in the strict dynamics experiment.")
    dynamics_frame, dynamics_features, dynamics_diagnostics = dynamics.build_causal_dynamics(
        raw, decisions, reference, representation
    )
    features = [*base_features, *dynamics_features]
    reference.validate_feature_contract(features, [])
    forbidden = set(features) & set(reference.FORBIDDEN_FEATURES)
    if forbidden:
        raise AssertionError(f"Forbidden dynamics feature leakage: {sorted(forbidden)}")
    if len(features) != len(set(features)):
        raise AssertionError("Dynamics-stage feature contract is not unique.")
    keys = ["task_id", "step"]
    base = decisions[storage_columns(base_features)].copy()
    stored = base.merge(
        dynamics_frame[keys + dynamics_features], on=keys, how="left", validate="one_to_one", sort=False
    )
    if stored.columns.duplicated().any() or len(stored) != len(decisions):
        raise AssertionError("Dynamics-stage alignment failed.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = args.output_dir / DYNAMICS_STAGE_FRAME
    manifest_path = args.output_dir / DYNAMICS_STAGE_MANIFEST
    require_unfinalized_stage(manifest_path, "dynamics")
    atomic_pickle(frame_path, stored)
    manifest = stage_manifest(
        args,
        "dynamics",
        input_files,
        len(stored),
        stored["task_id"].nunique(),
        {"frame": file_hash(frame_path)},
        base_feature_columns=list(base_features),
        dynamics_feature_columns=dynamics_features,
        key_order_sha256=key_order_hash(stored),
        frozen_alias_order=list(aliases),
        dynamics_diagnostics=dynamics_diagnostics,
    )
    atomic_json(manifest_path, manifest)
    print(json.dumps({"stage": "dynamics", "rows": len(stored), "features": len(dynamics_features)}, sort_keys=True), flush=True)


def assemble_prepared(args: argparse.Namespace) -> None:
    if args.pca_components < 0 or args.n_splits < 2:
        raise ValueError("--pca-components must be nonnegative and --n-splits must be at least two.")
    if args.feature_set == FEATURE_SET_BASE_DYNAMICS and args.pca_components != 0:
        raise ValueError("base_dynamics is coordinate-safe only with --pca-components 0.")
    reference = load_module("run_selected_answer_oof_experiments.py", "selected_answer_reference")
    dynamics_manifest_path = args.output_dir / DYNAMICS_STAGE_MANIFEST
    dynamics_manifest = read_valid_stage(args, dynamics_manifest_path, "dynamics")
    dynamics_frame_path = args.output_dir / DYNAMICS_STAGE_FRAME
    expected_dyn_hashes = dynamics_manifest["artifact_sha256"]
    if file_hash(dynamics_frame_path) != expected_dyn_hashes.get("frame"):
        raise RuntimeError("Dynamics staged artifact hash does not match its immutable manifest.")
    dynamics_frame = pd.read_pickle(dynamics_frame_path)
    if key_order_hash(dynamics_frame) != dynamics_manifest.get("key_order_sha256"):
        raise RuntimeError("Dynamics staged decision keys do not match its immutable manifest.")
    base_features = dynamics_manifest.get("base_feature_columns")
    dynamics_features = dynamics_manifest.get("dynamics_feature_columns")
    if not all(
        isinstance(value, list) and all(isinstance(item, str) for item in value)
        for value in (base_features, dynamics_features)
    ):
        raise RuntimeError("Dynamics staged feature contract is malformed.")

    stage_manifest_hashes: dict[str, str] = {"dynamics": file_hash(dynamics_manifest_path)}
    if args.feature_set == FEATURE_SET_BASE_DYNAMICS:
        features = [*base_features, *dynamics_features]
        stored = dynamics_frame[storage_columns(features)].copy()
        coordinate = np.empty((len(stored), 0), dtype=np.float32)
        representation_diagnostics: Any = {"status": "not_used_coordinate_safe_primary"}
        frozen_alias_order = dynamics_manifest["frozen_alias_order"]
    else:
        representation_manifest_path = args.output_dir / REPRESENTATION_STAGE_MANIFEST
        representation_manifest = read_valid_stage(args, representation_manifest_path, "representation")
        if (
            representation_manifest.get("input_files") != dynamics_manifest.get("input_files")
            or representation_manifest.get("decision_rows") != dynamics_manifest.get("decision_rows")
            or representation_manifest.get("task_groups") != dynamics_manifest.get("task_groups")
        ):
            raise RuntimeError("Representation and dynamics stages are not based on the same decision panel.")
        representation_frame_path = args.output_dir / REPRESENTATION_STAGE_FRAME
        coordinate_path = args.output_dir / REPRESENTATION_STAGE_COORDINATE
        expected_rep_hashes = representation_manifest["artifact_sha256"]
        if (
            file_hash(representation_frame_path) != expected_rep_hashes.get("frame")
            or file_hash(coordinate_path) != expected_rep_hashes.get("coordinate")
        ):
            raise RuntimeError("Representation staged artifact hash does not match its immutable manifest.")
        representation_frame = pd.read_pickle(representation_frame_path)
        coordinate = np.load(coordinate_path, allow_pickle=False).astype(np.float32, copy=False)
        if (
            key_order_hash(representation_frame) != representation_manifest.get("key_order_sha256")
            or len(coordinate) != len(representation_frame)
            or list(coordinate.shape) != representation_manifest.get("coordinate_shape")
        ):
            raise RuntimeError("Representation staged coordinates are not aligned to its immutable decision keys.")
        if not representation_frame[["task_id", "step"]].equals(dynamics_frame[["task_id", "step"]]):
            raise RuntimeError("Representation and dynamics decision keys differ in content or order.")
        representation_features = representation_manifest.get("representation_feature_columns")
        representation_base = representation_manifest.get("base_feature_columns")
        if (
            not isinstance(representation_features, list)
            or not isinstance(representation_base, list)
            or representation_base != base_features
            or not all(isinstance(item, str) for item in representation_features)
        ):
            raise RuntimeError("Representation staged feature contract is incompatible with dynamics.")
        features = [*base_features, *representation_features, *dynamics_features]
        frame = representation_frame.merge(
            dynamics_frame[["task_id", "step", *dynamics_features]],
            on=["task_id", "step"],
            how="left",
            validate="one_to_one",
            sort=False,
        )
        stored = frame[storage_columns(features)].copy()
        representation_diagnostics = representation_manifest["representation_diagnostics"]
        frozen_alias_order = representation_manifest["frozen_alias_order"]
        stage_manifest_hashes["representation"] = file_hash(representation_manifest_path)

    if len(features) != len(set(features)):
        raise RuntimeError("Staged feature contracts overlap.")
    reference.validate_feature_contract(features, [])
    forbidden = set(features) & set(reference.FORBIDDEN_FEATURES)
    if forbidden:
        raise AssertionError(f"Forbidden feature leakage: {sorted(forbidden)}")
    if (
        stored.columns.duplicated().any()
        or len(stored) != len(coordinate)
        or key_order_hash(stored) != dynamics_manifest.get("key_order_sha256")
        or stored[features].isna().all(axis=None)
    ):
        raise AssertionError("Final prepared-stage alignment failed.")
    final_frame_path = args.output_dir / PREPARED_FRAME
    final_coordinate_path = args.output_dir / PREPARED_COORDINATE
    final_manifest_path = args.output_dir / PREPARED_MANIFEST
    require_unfinalized_stage(final_manifest_path, "prepared")
    atomic_pickle(final_frame_path, stored)
    atomic_npy(final_coordinate_path, coordinate)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "source_hashes": source_hashes(),
        "input_dir": str(args.input_dir),
        "input_files": dynamics_manifest["input_files"],
        "decision_rows": int(len(stored)),
        "task_groups": int(stored["task_id"].nunique()),
        "outer_group": "task_id",
        "feature_set": args.feature_set,
        "n_splits": int(args.n_splits),
        "pca_components": int(args.pca_components),
        "seed": int(args.seed),
        "runtime_contract": runtime_contract(),
        "feature_columns": features,
        "feature_sha256": hashlib.sha256(canonical_bytes(features)).hexdigest(),
        "coordinate_shape": list(coordinate.shape),
        "key_order_sha256": key_order_hash(stored),
        "frozen_alias_order": frozen_alias_order,
        "representation_diagnostics": representation_diagnostics,
        "dynamics_diagnostics": dynamics_manifest["dynamics_diagnostics"],
        "stage_manifest_sha256": stage_manifest_hashes,
        "artifact_sha256": {
            "frame": file_hash(final_frame_path),
            "coordinate": file_hash(final_coordinate_path),
        },
        "strict_contract": {
            "response_selection": "deterministic current-barrier plurality; lexical frozen-alias tie break",
            "timing": "batch-level elapsed_seconds/tokens_per_second excluded",
            "dynamics": "same-model t-1/t-2 hidden kinematics only; aggregate anonymous scalar invariants after barrier closure",
            "missing_history": "contiguous history only; no gap bridging/backfill/future normalization",
            "preprocessing": "outer-train-fold median imputation, StandardScaler, and PCA only",
            "excluded": "raw answers/text, task/model/run identities, gold/correctness, K2, future barriers",
            "historical_limit": "same-step peer-completion timestamps are absent",
        },
    }
    atomic_json(final_manifest_path, manifest)
    print(json.dumps({"assembled": str(args.output_dir), "rows": len(stored), "features": len(features)}, sort_keys=True), flush=True)


def prepare(args: argparse.Namespace) -> None:
    """Run every stage for unconstrained hosts; bounded hosts invoke the stages separately."""
    if args.feature_set == FEATURE_SET_COMBINED_EXPLORATORY:
        prepare_representation_stage(args)
    prepare_dynamics_stage(args)
    assemble_prepared(args)


def load_prepared(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any], Any, Any]:
    manifest_path = args.output_dir / PREPARED_MANIFEST
    frame_path = args.output_dir / PREPARED_FRAME
    coordinate_path = args.output_dir / PREPARED_COORDINATE
    if not all(path.is_file() for path in (manifest_path, frame_path, coordinate_path)):
        raise FileNotFoundError("Missing prepared artifacts; run --prepare first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("source_hashes") != source_hashes():
        raise RuntimeError("Prepared artifacts use a different source contract; prepare a new output directory.")
    if (
        int(manifest.get("n_splits", -1)) != args.n_splits
        or int(manifest.get("pca_components", -1)) != args.pca_components
        or manifest.get("feature_set") != args.feature_set
        or int(manifest.get("seed", -1)) != args.seed
        or manifest.get("runtime_contract") != runtime_contract()
    ):
        raise RuntimeError("Prepared split, feature-set, seed, or runtime contract differs from this invocation.")
    if args.feature_set == FEATURE_SET_BASE_DYNAMICS and args.pca_components != 0:
        raise RuntimeError("base_dynamics requires PCA-disabled coordinates.")
    expected_hashes = manifest.get("artifact_sha256")
    if (
        not isinstance(expected_hashes, dict)
        or file_hash(frame_path) != expected_hashes.get("frame")
        or file_hash(coordinate_path) != expected_hashes.get("coordinate")
    ):
        raise RuntimeError("Prepared artifact hashes do not match the immutable manifest.")
    frame = pd.read_pickle(frame_path)
    coordinate = np.load(coordinate_path, allow_pickle=False)
    features = manifest.get("feature_columns")
    if (
        not isinstance(features, list)
        or not all(isinstance(value, str) for value in features)
        or len(features) != len(set(features))
        or hashlib.sha256(canonical_bytes(features)).hexdigest() != manifest.get("feature_sha256")
    ):
        raise RuntimeError("Prepared feature contract is malformed.")
    if (
        len(frame) != len(coordinate)
        or set(storage_columns(features)) - set(frame.columns)
        or list(coordinate.shape) != manifest.get("coordinate_shape")
        or key_order_hash(frame) != manifest.get("key_order_sha256")
        or len(frame) != int(manifest.get("decision_rows", -1))
        or frame["task_id"].nunique() != int(manifest.get("task_groups", -1))
    ):
        raise RuntimeError("Prepared frame/coordinate artifacts are inconsistent.")
    reference = load_module("run_selected_answer_oof_experiments.py", "selected_answer_reference")
    representation = load_module("run_selected_answer_representation_oof_experiments.py", "selected_answer_representation")
    return frame, coordinate.astype(np.float32, copy=False), manifest, reference, representation


def fold_checkpoint_path(output_dir: Path, config: str, fold: int) -> Path:
    return output_dir / "fold_checkpoints" / f"{config}_fold_{fold:02d}.npz"


def train_one_fold(args: argparse.Namespace) -> None:
    frame, coordinate, manifest, reference, representation = load_prepared(args)
    if args.fold is None or not 1 <= args.fold <= args.n_splits:
        raise ValueError(f"--fold must be in [1, {args.n_splits}].")
    config = config_by_name(args.config)
    labels = frame["selected_correct"].to_numpy(dtype=np.int8)
    groups = frame["task_id"].to_numpy(dtype=object)
    splits = list(GroupKFold(n_splits=args.n_splits).split(frame, labels, groups))
    train, test = splits[args.fold - 1]
    if set(groups[train]) & set(groups[test]):
        raise AssertionError(f"Task leakage in outer fold {args.fold}.")
    checkpoint = fold_checkpoint_path(args.output_dir, config.name, args.fold)
    if checkpoint.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint}; immutable fold results are not overwritten.")
    train_x, test_x = representation.fold_matrix(
        frame,
        coordinate,
        train,
        test,
        list(manifest["feature_columns"]),
        args.pca_components,
    )
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
        n_jobs=args.jobs,
        random_state=args.seed + args.fold,
        verbosity=-1,
    )
    started = time.time()
    model.fit(train_x, labels[train])
    probabilities = model.predict_proba(test_x)[:, 1].astype(np.float64)
    auc = float(roc_auc_score(labels[test], probabilities))
    binding = {
        "schema_version": SCHEMA_VERSION,
        "prepared_manifest_sha256": file_hash(args.output_dir / PREPARED_MANIFEST),
        "config": asdict(config),
        "fold": int(args.fold),
        "n_splits": int(args.n_splits),
        "feature_set": args.feature_set,
        "pca_components": int(args.pca_components),
        "seed": int(args.seed),
        "jobs": int(args.jobs),
        "runtime_contract": runtime_contract(),
        "test_indices_sha256": hashlib.sha256(np.asarray(test, dtype=np.int64).tobytes()).hexdigest(),
        "elapsed_seconds": time.time() - started,
        "fold_auc": auc,
    }
    atomic_npz(
        checkpoint,
        test_indices=np.asarray(test, dtype=np.int64),
        probabilities=probabilities,
        binding_json=np.asarray(json.dumps(binding, sort_keys=True)),
    )
    print(json.dumps({"checkpoint": str(checkpoint), "fold": args.fold, "auc": auc}, sort_keys=True), flush=True)


def aggregate(args: argparse.Namespace) -> None:
    frame, _coordinate, manifest, reference, _representation = load_prepared(args)
    config = config_by_name(args.config)
    labels = frame["selected_correct"].to_numpy(dtype=np.int8)
    expected = set(range(len(frame)))
    oof = np.full(len(frame), np.nan, dtype=np.float64)
    fold_auc: list[float] = []
    manifest_hash = file_hash(args.output_dir / PREPARED_MANIFEST)
    groups = frame["task_id"].to_numpy(dtype=object)
    expected_splits = list(GroupKFold(n_splits=args.n_splits).split(frame, labels, groups))
    for fold in range(1, args.n_splits + 1):
        checkpoint = fold_checkpoint_path(args.output_dir, config.name, fold)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing strict fold checkpoint: {checkpoint}")
        with np.load(checkpoint, allow_pickle=False) as payload:
            indices = np.asarray(payload["test_indices"], dtype=np.int64)
            probabilities = np.asarray(payload["probabilities"], dtype=np.float64)
            binding = json.loads(str(payload["binding_json"].item()))
        expected_indices = np.asarray(expected_splits[fold - 1][1], dtype=np.int64)
        expected_indices_hash = hashlib.sha256(expected_indices.tobytes()).hexdigest()
        if (
            binding.get("schema_version") != SCHEMA_VERSION
            or binding.get("prepared_manifest_sha256") != manifest_hash
            or binding.get("config") != asdict(config)
            or binding.get("fold") != fold
            or binding.get("n_splits") != args.n_splits
            or binding.get("feature_set") != args.feature_set
            or binding.get("pca_components") != args.pca_components
            or binding.get("seed") != args.seed
            or binding.get("jobs") != args.jobs
            or binding.get("runtime_contract") != runtime_contract()
            or binding.get("test_indices_sha256") != expected_indices_hash
            or indices.ndim != 1
            or len(indices) != len(probabilities)
            or len(np.unique(indices)) != len(indices)
            or set(indices.tolist()) - expected
            or not np.array_equal(indices, expected_indices)
            or np.isfinite(probabilities).sum() != len(probabilities)
        ):
            raise RuntimeError(f"Invalid or foreign fold checkpoint: {checkpoint}")
        if np.isfinite(oof[indices]).any():
            raise RuntimeError(f"Overlapping OOF rows in {checkpoint}")
        oof[indices] = probabilities
        fold_auc.append(float(roc_auc_score(labels[indices], probabilities)))
    if not np.isfinite(oof).all():
        raise RuntimeError("Fold checkpoints do not cover every decision exactly once.")
    metrics_path = args.output_dir / f"{config.name}_metrics.json"
    predictions_path = args.output_dir / f"{config.name}_predictions.csv"
    if metrics_path.exists():
        raise FileExistsError("Final aggregate metrics already exist; use a new output directory to preserve provenance.")
    metrics = reference.score_summary(frame, oof, bootstrap_replicates=1000, seed=args.seed)
    metrics.update(
        {
            "schema_version": SCHEMA_VERSION,
            "configuration": asdict(config),
            "fold_auc": fold_auc,
            "prepared_manifest_sha256": manifest_hash,
            "feature_set": args.feature_set,
            "seed": int(args.seed),
            "runtime_contract": runtime_contract(),
            "interpretation": "strict retrospective candidate only; configuration selection still requires a separate nested/fresh confirmation.",
        }
    )
    prediction = frame[["task_id", "step", "domain", "selected_correct"]].copy()
    prediction["oof_probability"] = oof
    atomic_csv(predictions_path, prediction)
    atomic_json(metrics_path, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True, default=str), flush=True)


def main() -> int:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive.")
    if args.pca_components < 0 or args.n_splits < 2:
        raise ValueError("--pca-components must be nonnegative and --n-splits must be at least two.")
    status_path = args.status_file or (args.output_dir / "dynamics_status.json")
    if args.prepare:
        action = "prepare"
    elif args.prepare_representation:
        action = "prepare_representation"
    elif args.prepare_dynamics:
        action = "prepare_dynamics"
    elif args.assemble:
        action = "assemble"
    elif args.fold is not None:
        action = f"fold_{args.fold:02d}"
    else:
        action = "aggregate"
    atomic_json(
        status_path,
        {"schema_version": SCHEMA_VERSION, "state": "running", "action": action, "started_at_unix": time.time()},
    )
    try:
        if args.prepare:
            prepare(args)
        elif args.prepare_representation:
            prepare_representation_stage(args)
        elif args.prepare_dynamics:
            prepare_dynamics_stage(args)
        elif args.assemble:
            assemble_prepared(args)
        elif args.fold is not None:
            train_one_fold(args)
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
    atomic_json(
        status_path,
        {"schema_version": SCHEMA_VERSION, "state": "complete", "action": action, "finished_at_unix": time.time()},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
