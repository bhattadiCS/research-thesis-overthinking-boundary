#!/usr/bin/env python3
"""Strict outer-fold capacity screen for the selected-answer representation probe.

This is an explicitly exploratory screen, not a claim that its winning
configuration is a confirmed result: every configuration is evaluated on the
same five task-held-out folds so that the result can identify candidates for a
separate nested confirmation.  It preserves the established timing-free,
answer-text-free feature contract and fold-local PCA protocol.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


SCHEMA_VERSION = "selected-answer-capacity-sweep-v1"
DEFAULT_INPUT = Path("research/outputs/experiments_v2")
DEFAULT_OUTPUT = Path("research/outputs/experiments_v2/selected_answer_capacity_sweep_v1")


@dataclass(frozen=True)
class Configuration:
    name: str
    n_estimators: int
    learning_rate: float
    num_leaves: int
    min_child_samples: int
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    reg_alpha: float
    max_depth: int = -1


CONFIGURATIONS: tuple[Configuration, ...] = (
    Configuration("conservative_reference", 600, 0.035, 31, 45, 0.88, 0.82, 4.0, 0.12),
    Configuration("shallow_high_regularization", 900, 0.025, 15, 75, 0.90, 0.90, 8.0, 0.20),
    Configuration("medium_capacity", 800, 0.030, 63, 28, 0.90, 0.90, 3.0, 0.05),
    Configuration("deep_regularized", 1000, 0.020, 127, 55, 0.85, 0.75, 10.0, 0.25),
    Configuration("wide_low_shrinkage", 1200, 0.015, 95, 40, 0.95, 1.00, 6.0, 0.08),
    Configuration("small_leaf_balanced", 750, 0.030, 23, 30, 1.00, 1.00, 1.0, 0.0),
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


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}; use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--config", nargs="*", default=None, help="Optional configuration names to screen.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_splits < 2 or args.pca_components < 0 or args.jobs < 1:
        raise ValueError("--n-splits >= 2, --pca-components >= 0, and --jobs >= 1 are required.")
    selected = CONFIGURATIONS if args.config is None else tuple(
        config for config in CONFIGURATIONS if config.name in set(args.config)
    )
    unknown = set(args.config or []) - {config.name for config in CONFIGURATIONS}
    if unknown:
        raise ValueError(f"Unknown --config values: {sorted(unknown)}")
    if not selected:
        raise ValueError("Select at least one configuration.")

    reference = load_module("run_selected_answer_oof_experiments.py", "selected_answer_reference")
    representation = load_module("run_selected_answer_representation_oof_experiments.py", "selected_answer_representation")
    raw, input_files = reference.load_panel(args.input_dir)
    decisions, base_features, categories, aliases = reference.build_decision_frame(raw, exclude_batch_timing=True)
    if categories:
        raise AssertionError("Identity categorical features are forbidden in this strict screen.")
    frame, coordinate, diagnostics = representation.build_representation_frame(raw, decisions, reference)
    features = list(base_features) + sorted(column for column in frame.columns if column.startswith(representation.REP_PREFIX))
    reference.validate_feature_contract(features, [])
    labels = frame["selected_correct"].to_numpy(dtype=np.int8)
    groups = frame["task_id"].to_numpy(dtype=object)
    splits = list(GroupKFold(n_splits=args.n_splits).split(frame, labels, groups))
    for fold, (train, test) in enumerate(splits, start=1):
        if set(groups[train]) & set(groups[test]):
            raise AssertionError(f"Task leakage in outer fold {fold}")

    prepare_output(args.output_dir, args.overwrite)
    started = time.time()
    # Fold matrices are fitted exactly once from their outer training rows and
    # then reused across configurations; no configuration sees held-out PCA or
    # imputation statistics.
    matrices = []
    for fold, (train, test) in enumerate(splits, start=1):
        train_x, test_x = representation.fold_matrix(
            frame, coordinate, train, test, features, args.pca_components
        )
        matrices.append((train, test, train_x, test_x))
        print(f"Prepared strict outer-fold matrix {fold}/{len(splits)}", flush=True)

    metrics: dict[str, Any] = {}
    predictions = frame[["task_id", "step", "domain", "selected_correct"]].copy()
    for config_index, config in enumerate(selected, start=1):
        oof = np.full(len(frame), np.nan, dtype=np.float64)
        fold_ids = np.full(len(frame), -1, dtype=np.int16)
        fold_auc: list[float] = []
        for fold, (train, test, train_x, test_x) in enumerate(matrices, start=1):
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
                max_depth=config.max_depth,
                n_jobs=args.jobs,
                random_state=args.seed + config_index * 100 + fold,
                verbosity=-1,
            )
            model.fit(train_x, labels[train])
            oof[test] = model.predict_proba(test_x)[:, 1]
            fold_ids[test] = fold
            fold_auc.append(float(roc_auc_score(labels[test], oof[test])))
            print(
                f"[{config.name}] fold {fold}/{len(splits)} auc={fold_auc[-1]:.6f}",
                flush=True,
            )
        if not np.isfinite(oof).all() or (fold_ids < 1).any():
            raise AssertionError(f"{config.name} did not produce complete OOF coverage.")
        summary = reference.score_summary(frame, oof, bootstrap_replicates=1000, seed=args.seed + config_index)
        summary.update({"configuration": asdict(config), "fold_auc": fold_auc})
        metrics[config.name] = summary
        predictions[f"{config.name}_oof"] = oof
        predictions[f"{config.name}_fold"] = fold_ids
        print(
            f"[{config.name}] strict exploratory OOF AUC={summary['oof_auc']:.6f} "
            f"task_macro={summary['task_macro_auc']:.6f}",
            flush=True,
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "script_sha256": file_hash(Path(__file__).resolve()),
        "representation_script_sha256": file_hash(Path(__file__).with_name("run_selected_answer_representation_oof_experiments.py")),
        "reference_script_sha256": file_hash(Path(__file__).with_name("run_selected_answer_oof_experiments.py")),
        "input_dir": str(args.input_dir),
        "input_files": input_files,
        "candidate_rows": int(len(raw)),
        "decision_rows": int(len(frame)),
        "task_groups": int(frame["task_id"].nunique()),
        "outer_group": "task_id",
        "fold_local_pca_components": args.pca_components,
        "feature_count_without_pca": len(features),
        "frozen_alias_order": list(aliases),
        "representation_diagnostics": diagnostics,
        "configurations": [asdict(config) for config in selected],
        "strict_contract": {
            "selection": "deterministic current-barrier plurality; lexical frozen-alias tie break",
            "timing": "batch-level elapsed_seconds/tokens_per_second excluded",
            "representation": "current-barrier panel and selected-supporter geometry only",
            "preprocessing": "imputation, StandardScaler, and PCA fit on each outer training fold only",
            "scope": "exploratory outer-fold configuration screen; requires nested confirmation before model selection claim",
        },
    }
    atomic_json(args.output_dir / "capacity_sweep_manifest.json", manifest)
    atomic_json(args.output_dir / "capacity_sweep_metrics.json", metrics)
    predictions.to_csv(args.output_dir / "capacity_sweep_predictions.csv", index=False)
    lines = [
        "STRICT SELECTED-ANSWER CAPACITY SCREEN (EXPLORATORY)",
        f"Decisions={len(frame)} | task groups={frame['task_id'].nunique()} | feature count={len(features)}",
        "configuration | OOF AUC | task macro AUC | bootstrap 95% CI | ECE15",
    ]
    for config in selected:
        summary = metrics[config.name]
        lines.append(
            f"{config.name} | {summary['oof_auc']:.6f} | {summary['task_macro_auc']:.6f} | "
            f"{summary['task_cluster_bootstrap_auc_95_ci']} | {summary['raw_ece_15']:.6f}"
        )
    (args.output_dir / "capacity_sweep_results.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
