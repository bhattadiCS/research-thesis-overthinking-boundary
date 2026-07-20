#!/usr/bin/env python3
"""Strict hidden-state OOF ablation for deterministic selected answers.

This evaluator extends ``run_selected_answer_oof_experiments.py`` with
current-barrier geometry from the two stored 64-dimensional hidden projections.
It deliberately keeps the conservative timing-free base contract and fits the
coordinate PCA separately in every task-held-out outer training fold.

Raw answer strings are used transiently only to form the deterministic
plurality and its supporting set.  They, task identifiers, labels, model IDs,
K2 fields, raw text, and future barriers never enter a feature matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


SCHEMA_VERSION = "selected-answer-representation-oof-v1"
DEFAULT_INPUT = Path("research/outputs/experiments_v2")
DEFAULT_OUTPUT = Path("research/outputs/experiments_v2/selected_answer_representation_oof_v1")
VECTOR_DIM = 64
EPS = 1.0e-12
REP_PREFIX = "rep_"


def load_reference() -> Any:
    path = Path(__file__).with_name("run_selected_answer_oof_experiments.py")
    spec = importlib.util.spec_from_file_location("selected_answer_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import selected-answer evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def raw_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def parse_projection(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Strictly parse finite fixed-width CSV projection fields.

    Missing, malformed, nonfinite, or wrong-width vectors receive a zero
    sentinel and an explicit availability value of zero; they cannot silently
    become a plausible numeric vector.
    """
    matrix = np.zeros((len(values), VECTOR_DIM), dtype=np.float32)
    available = np.zeros(len(values), dtype=np.float32)
    for index, raw in enumerate(values.to_numpy()):
        if not isinstance(raw, str) or not raw.strip():
            continue
        vector = np.fromstring(raw.strip().strip("[]"), sep=",", dtype=np.float32)
        if vector.size == VECTOR_DIM and np.isfinite(vector).all():
            matrix[index] = vector
            available[index] = 1.0
    return matrix, available


def scalar_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {f"{prefix}_{name}": np.nan for name in ("mean", "std", "min", "max")}
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
    }


def pairwise_cosine_stats(matrix: np.ndarray, prefix: str) -> dict[str, float]:
    result: dict[str, float] = {f"{prefix}_count": float(len(matrix))}
    if len(matrix) < 2:
        result.update({f"{prefix}_{name}": np.nan for name in ("mean", "std", "min", "max")})
        return result
    norms = np.linalg.norm(matrix, axis=1)
    matrix = matrix[norms > EPS]
    if len(matrix) < 2:
        result.update({f"{prefix}_{name}": np.nan for name in ("mean", "std", "min", "max")})
        return result
    unit = matrix / np.linalg.norm(matrix, axis=1)[:, None]
    values = (unit @ unit.T)[np.triu_indices(len(unit), k=1)]
    result.update(scalar_stats(values, prefix))
    return result


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    if not len(left) or not len(right):
        return float("nan")
    a = left.mean(axis=0)
    b = right.mean(axis=0)
    scale = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / scale) if scale > EPS else float("nan")


def build_representation_frame(
    frame: pd.DataFrame, decisions: pd.DataFrame, reference: Any
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Attach only current-barrier hidden geometry to selected decisions."""
    work = frame.copy().reset_index(drop=True)
    keys = ["task_id", "step"]
    work["_answer_key"] = work["answer_normalized"].map(reference.normalized_answer_key)
    work = work.merge(decisions[keys + ["selected_answer_key"]], on=keys, how="left", validate="many_to_one")
    if work["selected_answer_key"].isna().any():
        raise AssertionError("Could not align every response to a deterministic selected decision")
    work["_winner"] = (
        (work["_answer_key"] != "") & (work["_answer_key"] == work["selected_answer_key"])
    )
    l1, l1_available = parse_projection(work["mid_hidden_1_proj"])
    l2, l2_available = parse_projection(work["mid_hidden_2_proj"])
    both = (l1_available.astype(bool) & l2_available.astype(bool))
    cross = np.full(len(work), np.nan, dtype=np.float32)
    l1_norm = np.linalg.norm(l1, axis=1)
    l2_norm = np.linalg.norm(l2, axis=1)
    valid_cross = both & (l1_norm > EPS) & (l2_norm > EPS)
    cross[valid_cross] = np.sum(l1[valid_cross] * l2[valid_cross], axis=1) / (
        l1_norm[valid_cross] * l2_norm[valid_cross]
    )

    rows: list[dict[str, Any]] = []
    coordinate_blocks: list[np.ndarray] = []
    for (task_id, step), group in work.groupby(keys, sort=False):
        indices = group.index.to_numpy(dtype=np.int64)
        winner = group["_winner"].to_numpy(dtype=bool)
        loser = ~winner
        a1 = l1_available[indices].astype(bool)
        a2 = l2_available[indices].astype(bool)
        a12 = both[indices]
        vectors = {"l1": (l1[indices], a1), "l2": (l2[indices], a2)}
        row: dict[str, Any] = {"task_id": str(task_id), "step": int(step)}
        row.update(
            {
                "rep_panel_l1_available_fraction": float(a1.mean()),
                "rep_panel_l2_available_fraction": float(a2.mean()),
                "rep_panel_both_available_fraction": float(a12.mean()),
                "rep_winner_l1_available_fraction": float(a1[winner].mean()) if winner.any() else 0.0,
                "rep_winner_l2_available_fraction": float(a2[winner].mean()) if winner.any() else 0.0,
                "rep_winner_member_count": float(winner.sum()),
            }
        )
        for population, mask in (("panel", np.ones(len(group), dtype=bool)), ("winner", winner)):
            for layer, (matrix, available) in vectors.items():
                values = matrix[mask & available]
                row.update(scalar_stats(np.linalg.norm(values, axis=1) if len(values) else np.empty(0), f"rep_{population}_{layer}_norm"))
                row.update(pairwise_cosine_stats(values, f"rep_{population}_{layer}_paircos"))
                row[f"rep_{population}_{layer}_centroid_norm"] = (
                    float(np.linalg.norm(values.mean(axis=0))) if len(values) else np.nan
                )
            row.update(scalar_stats(cross[indices][mask & a12], f"rep_{population}_cross_layer_cos"))
        for layer, (matrix, available) in vectors.items():
            winner_values = matrix[winner & available]
            loser_values = matrix[loser & available]
            row[f"rep_{layer}_winner_vs_nonwinner_centroid_cos"] = cosine(winner_values, loser_values)
            row[f"rep_{layer}_winner_vs_nonwinner_centroid_l2"] = (
                float(np.linalg.norm(winner_values.mean(axis=0) - loser_values.mean(axis=0)))
                if len(winner_values) and len(loser_values)
                else np.nan
            )

        def mean_or_zero(matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
            values = matrix[mask]
            return values.mean(axis=0) if len(values) else np.zeros(VECTOR_DIM, dtype=np.float32)

        coordinate_blocks.append(
            np.concatenate(
                [
                    mean_or_zero(l1[indices], a1),
                    mean_or_zero(l2[indices], a2),
                    mean_or_zero(l1[indices], winner & a1),
                    mean_or_zero(l2[indices], winner & a2),
                ]
            ).astype(np.float32)
        )
        rows.append(row)

    representation = pd.DataFrame(rows)
    result = decisions.merge(representation, on=keys, how="left", validate="one_to_one", sort=False)
    coordinate = np.stack(coordinate_blocks).astype(np.float32)
    lookup = {tuple(row): index for index, row in enumerate(representation[keys].itertuples(index=False, name=None))}
    coordinate = np.stack(
        [coordinate[lookup[tuple(row)]] for row in decisions[keys].itertuples(index=False, name=None)]
    ).astype(np.float32)
    rep_columns = [column for column in result.columns if column.startswith(REP_PREFIX)]
    if len(result) != len(decisions) or not rep_columns:
        raise AssertionError("Representation feature merge failed")
    return result, coordinate, {
        "projection_dimension": VECTOR_DIM,
        "coordinate_block_dimension": int(coordinate.shape[1]),
        "representation_scalar_feature_count": len(rep_columns),
        "l1_available_fraction": float(l1_available.mean()),
        "l2_available_fraction": float(l2_available.mean()),
    }


def fold_matrix(
    frame: pd.DataFrame,
    coordinate: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    feature_columns: list[str],
    pca_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_data: dict[str, np.ndarray] = {}
    test_data: dict[str, np.ndarray] = {}
    for column in feature_columns:
        values_train = pd.to_numeric(frame.iloc[train][column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        values_test = pd.to_numeric(frame.iloc[test][column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = float(values_train.median()) if values_train.notna().any() else 0.0
        train_data[column] = values_train.fillna(median).to_numpy(dtype=np.float32)
        test_data[column] = values_test.fillna(median).to_numpy(dtype=np.float32)
    if pca_components:
        scaler = StandardScaler()
        train_block = scaler.fit_transform(coordinate[train])
        test_block = scaler.transform(coordinate[test])
        count = min(pca_components, train_block.shape[0], train_block.shape[1])
        pca = PCA(n_components=count, svd_solver="full", random_state=0)
        train_pc = pca.fit_transform(train_block).astype(np.float32)
        test_pc = pca.transform(test_block).astype(np.float32)
        for index in range(count):
            train_data[f"rep_fold_pca_{index:02d}"] = train_pc[:, index]
            test_data[f"rep_fold_pca_{index:02d}"] = test_pc[:, index]
    return pd.DataFrame(train_data), pd.DataFrame(test_data)


def evaluate(
    name: str,
    frame: pd.DataFrame,
    coordinate: np.ndarray,
    features: list[str],
    pca_components: int,
    splits: list[tuple[np.ndarray, np.ndarray]],
    reference: Any,
    seed: int,
    jobs: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    labels = frame["selected_correct"].to_numpy(dtype=np.int8)
    oof = np.full(len(frame), np.nan, dtype=np.float64)
    fold_ids = np.full(len(frame), -1, dtype=np.int16)
    fold_auc: list[float] = []
    config = reference.LightGBMConfig()
    for fold, (train, test) in enumerate(splits, start=1):
        train_x, test_x = fold_matrix(frame, coordinate, train, test, features, pca_components)
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
            n_jobs=jobs,
            random_state=seed + fold,
            verbosity=-1,
        )
        model.fit(train_x, labels[train])
        oof[test] = model.predict_proba(test_x)[:, 1]
        fold_ids[test] = fold
        fold_auc.append(float(roc_auc_score(labels[test], oof[test])))
        print(f"[{name}] fold {fold}/{len(splits)} rows={len(test)} auc={fold_auc[-1]:.6f}", flush=True)
    if not np.isfinite(oof).all() or (fold_ids < 1).any():
        raise AssertionError(f"{name} did not yield full OOF coverage")
    report = reference.score_summary(frame, oof, bootstrap_replicates=1000, seed=seed)
    report.update(
        {
            "name": name,
            "feature_count_without_fold_pca": len(features),
            "fold_local_pca_components": pca_components,
            "fold_auc": fold_auc,
        }
    )
    return report, oof, fold_ids


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing result bundle: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def self_test() -> None:
    values = pd.Series(["[1,2,3]", "[" + ",".join(["1"] * VECTOR_DIM) + "]", None])
    matrix, available = parse_projection(values)
    if available.tolist() != [0.0, 1.0, 0.0] or matrix.shape != (3, VECTOR_DIM):
        raise AssertionError("Projection parsing safety contract failed")
    pair = pairwise_cosine_stats(np.eye(2, VECTOR_DIM, dtype=np.float32), "synthetic")
    if pair["synthetic_count"] != 2.0 or pair["synthetic_mean"] != 0.0:
        raise AssertionError("Pairwise cosine feature self-test failed")
    print("Selected-answer representation self-test passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.n_splits < 2 or args.pca_components < 0 or args.jobs < 1:
        raise ValueError("--n-splits >=2, --pca-components >=0, and --jobs >0 are required")
    reference = load_reference()
    raw, input_files = reference.load_panel(args.input_dir)
    decisions, base_features, categories, aliases = reference.build_decision_frame(raw, exclude_batch_timing=True)
    if categories:
        raise AssertionError("Identity categorical features are intentionally excluded from this experiment")
    frame, coordinate, diagnostics = build_representation_frame(raw, decisions, reference)
    rep_features = sorted(column for column in frame.columns if column.startswith(REP_PREFIX))
    features = list(base_features) + rep_features
    reference.validate_feature_contract(features, [])
    if set(features) & set(reference.FORBIDDEN_FEATURES):
        raise AssertionError("Forbidden selected-answer field leaked into representation experiment")
    labels = frame["selected_correct"].to_numpy(dtype=np.int8)
    groups = frame["task_id"].to_numpy(dtype=object)
    splitter = GroupKFold(n_splits=args.n_splits)
    splits = list(splitter.split(frame, labels, groups))
    for number, (train, test) in enumerate(splits, start=1):
        if set(groups[train]) & set(groups[test]):
            raise AssertionError(f"Task leakage in fold {number}")
    prepare_output(args.output_dir, args.overwrite)
    started = time.time()
    invariant_metrics, invariant_oof, folds = evaluate(
        "hidden_invariants", frame, coordinate, features, 0, splits, reference, args.seed, args.jobs
    )
    pca_metrics, pca_oof, pca_folds = evaluate(
        f"hidden_invariants_plus_fold_pca{args.pca_components}",
        frame,
        coordinate,
        features,
        args.pca_components,
        splits,
        reference,
        args.seed,
        args.jobs,
    )
    if not np.array_equal(folds, pca_folds):
        raise AssertionError("Experiment folds unexpectedly differ")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "script_sha256": raw_hash(Path(__file__).resolve()),
        "reference_script_sha256": raw_hash(Path(__file__).with_name("run_selected_answer_oof_experiments.py")),
        "input_dir": str(args.input_dir),
        "input_files": input_files,
        "candidate_rows": len(raw),
        "decision_rows": len(frame),
        "task_groups": int(frame["task_id"].nunique()),
        "outer_group": "task_id",
        "pca_components": args.pca_components,
        "base_feature_count": len(base_features),
        "representation_feature_count": len(rep_features),
        "representation_features": rep_features,
        "representation_diagnostics": diagnostics,
        "frozen_alias_order": list(aliases),
        "strict_contract": {
            "selection": "nonempty normalized-answer plurality; lexical frozen-alias tie break",
            "timing": "batch-level elapsed_seconds/tokens_per_second excluded",
            "representation_scope": "current barrier panel and plurality-supporting subset only",
            "coordinate_pca": "StandardScaler/PCA fit only on each outer training fold",
            "excluded": sorted(reference.FORBIDDEN_FEATURES),
            "historical_limit": "same-step barrier completion lacks event timestamps",
            "architecture_caveat": "projection coordinates originate in model-specific hidden spaces",
        },
    }
    metrics = {"hidden_invariants": invariant_metrics, "hidden_invariants_plus_fold_pca": pca_metrics}
    atomic_json(args.output_dir / "selected_answer_representation_manifest.json", manifest)
    atomic_json(args.output_dir / "selected_answer_representation_metrics.json", metrics)
    predictions = frame[
        ["task_id", "step", "domain", "selected_model_alias", "selected_answer_hash", "selected_correct"]
    ].copy()
    predictions["fold"] = folds
    predictions["hidden_invariants_oof"] = invariant_oof
    predictions["hidden_invariants_plus_fold_pca_oof"] = pca_oof
    temporary = args.output_dir / "selected_answer_representation_predictions.csv.tmp"
    predictions.to_csv(temporary, index=False)
    os.replace(temporary, args.output_dir / "selected_answer_representation_predictions.csv")
    log = "\n".join(
        [
            "=" * 104,
            "SELECTED-ANSWER HIDDEN-REPRESENTATION OOF EXPERIMENT",
            "=" * 104,
            f"Candidate rows: {len(raw)} | decision barriers: {len(frame)} | task groups: {frame['task_id'].nunique()}",
            "Validation: GroupKFold(task_id); current-barrier geometry; timing-free base features.",
            "",
            "Configuration                         | OOF AUC | Task macro AUC | Task bootstrap 95% CI | ECE15",
            "-" * 104,
            (
                f"Hidden invariants                    | {invariant_metrics['oof_auc']:.6f} | "
                f"{invariant_metrics['task_macro_auc']:.6f} | "
                f"{invariant_metrics['task_cluster_bootstrap_auc_95_ci']} | {invariant_metrics['raw_ece_15']:.6f}"
            ),
            (
                f"Hidden invariants + fold PCA-{args.pca_components:02d} | {pca_metrics['oof_auc']:.6f} | "
                f"{pca_metrics['task_macro_auc']:.6f} | "
                f"{pca_metrics['task_cluster_bootstrap_auc_95_ci']} | {pca_metrics['raw_ece_15']:.6f}"
            ),
            "",
            "Retrospective only: cross-model projection geometry requires prospective fixed-roster confirmation.",
            "",
        ]
    )
    atomic_text(args.output_dir / "selected_answer_representation_results.log", log)
    print(log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
