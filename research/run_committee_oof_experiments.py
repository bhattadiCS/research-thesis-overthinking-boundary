#!/usr/bin/env python3
"""Strict, task-held-out fleet-committee stopping experiments.

This is deliberately a *separate* experiment from the ultimate neural
tournament.  The latter is a single-trajectory stopping probe; this program
measures the additional information supplied by a synchronous, independent
model fleet.  At decision barrier ``t`` it can use only:

* the candidate's current/prefix telemetry;
* other model aliases' answers for the same task at the same barrier; and
* deployment-known model and collection metadata.

It never gives a learner an answer string, task identifier, expected answer,
grader output, K2 signal, future step, or source-cell identity.  In
particular, an answer string is used transiently only to compute a
leave-one-model-out agreement statistic.  This makes the result meaningful
only when all committee members have actually completed their barrier-t
generation before the score is consumed in production.

The default experiment has two fixed ablations:

``telemetry``
    Candidate telemetry only.
``strict_committee``
    Telemetry plus leave-one-model-out live agreement features.

All outer folds hold every trajectory for a raw ``task_id`` together.  The
script writes an immutable result bundle by default and refuses to overwrite
an existing bundle without ``--overwrite``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold


SCHEMA_VERSION = "committee-oof-v1"
DEFAULT_INPUT = Path("research/outputs/experiments_v2")
DEFAULT_OUTPUT = Path("research/outputs/experiments_v2/committee_oof_strict_v1")
EPS = 1.0e-12

# These fields are observed after the candidate's current generation completes.
# K2 is intentionally absent: it is a separate generation and must be modeled as
# a separately-costed post-query decision, not as a free pre-query feature.
BASE_NUMERIC_COLUMNS = [
    "step",
    "confidence",
    "model_stop_flag",
    "answer_changed",
    "thought_token_count",
    "raw_generation_tokens",
    "mean_token_logprob",
    "entropy_mean",
    "entropy_std",
    "hidden_norm",
    "hidden_l2_shift",
    "hidden_cosine_shift",
    "lexical_echo",
    "verbose_confidence_proxy",
    "elapsed_seconds",
    "tokens_per_second",
    "temperature",
    "parse_success",
    "hit_max_new_tokens",
    "truncated_output_suspected",
    "answer_span_mean_logprob",
    "answer_span_min_logprob",
    "answer_span_mean_entropy",
    "answer_span_std_entropy",
    "raw_text_length_chars",
    "raw_text_length_tokens",
]

PREFIX_DELTA_COLUMNS = [
    "confidence",
    "mean_token_logprob",
    "entropy_mean",
    "entropy_std",
    "hidden_norm",
    "hidden_l2_shift",
    "hidden_cosine_shift",
    "answer_span_mean_logprob",
    "answer_span_min_logprob",
    "answer_span_mean_entropy",
    "answer_span_std_entropy",
]

LOG1P_COLUMNS = [
    "thought_token_count",
    "raw_generation_tokens",
    "elapsed_seconds",
    "raw_text_length_chars",
    "raw_text_length_tokens",
    "answer_char_len",
]

# Each is a deployment-known property of the scored generation.  ``source_cell``
# is intentionally not included: using it turns historical storage layout into a
# model feature and makes a deployment contract needlessly fragile.
CATEGORICAL_COLUMNS = [
    "model_alias",
    "domain",
    "difficulty",
    "output_format_type",
    "answer_extraction_source",
    "stop_extraction_source",
    "confidence_extraction_source",
    "prompt_mode",
    "system_prompt_mode",
    "is_baseline",
    "device",
]

DERIVED_TELEMETRY_COLUMNS = [
    "answer_nonempty",
    "answer_char_len",
    "same_prev_answer",
    "n_answer_changes_prefix",
]

STRICT_COMMITTEE_COLUMNS = [
    "committee_panel_nonempty",
    "independent_vote_count",
    "independent_peer_count",
    "independent_vote_fraction",
    "committee_has_independent_match",
    "committee_nonempty_fraction",
]

FORBIDDEN_MODEL_COLUMNS = {
    "correct",
    "utility",
    "expected_answer",
    "answer",
    "answer_normalized",
    "answer_key",
    "task_id",
    "task_source_index",
    "run_id",
    "trajectory_id",
    "source_cell",
    "thought",
    "raw_text",
    "k2_agreement",
    "k2_raw_generation_tokens",
}


@dataclass(frozen=True)
class LightGBMConfig:
    """Fixed configuration frozen before the confirmatory run.

    The objective is deliberately unweighted binary log loss.  Unlike the
    class-balanced heads in the older tournament, it does not create a known
    posterior-intercept shift that a temperature-only calibrator cannot repair.
    """

    n_estimators: int = 450
    learning_rate: float = 0.045
    num_leaves: int = 63
    min_child_samples: int = 70
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_lambda: float = 3.0
    reg_alpha: float = 0.15


@dataclass
class CVResult:
    name: str
    oof: np.ndarray
    fold_ids: np.ndarray
    fold_auc: list[float]
    feature_columns: list[str]
    categorical_columns: list[str]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def canonical_file_hash(path: Path) -> str:
    """Hash CSV bytes after LF normalization for Windows/Linux portability."""

    digest = hashlib.sha256()
    trailing_cr = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            block = trailing_cr + block
            # Do not split a CRLF pair at the chunk boundary.  This matters for
            # manifest identity because the source files are large enough to span
            # many read blocks.
            if block.endswith(b"\r"):
                trailing_cr = b"\r"
                block = block[:-1]
            else:
                trailing_cr = b""
            digest.update(block.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
    if trailing_cr:
        digest.update(b"\n")
    return digest.hexdigest()


def raw_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def answer_key(value: Any) -> str:
    """Return a comparison key without ever exposing answer text to the model."""

    if pd.isna(value):
        return ""
    return str(value).strip().casefold()


def select_trace_paths(input_dir: Path) -> list[Path]:
    paths = sorted(input_dir.glob("global_*/trace_steps.csv"))
    if not paths:
        raise FileNotFoundError(f"No canonical global_*/trace_steps.csv files under {input_dir}")
    return paths


def load_canonical_panel(input_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Load the canonical fleet and validate independent-alias panel semantics."""

    paths = select_trace_paths(input_dir)
    frames: list[pd.DataFrame] = []
    file_manifest: list[dict[str, Any]] = []
    required = {"run_id", "model_alias", "task_id", "step", "correct", "answer_normalized"}
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        frame["source_cell"] = path.parent.name
        frames.append(frame)
        file_manifest.append(
            {
                "path": path.as_posix(),
                "bytes": int(path.stat().st_size),
                "raw_sha256": raw_file_hash(path),
                "canonical_lf_sha256": canonical_file_hash(path),
            }
        )

    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame["task_id"] = frame["task_id"].astype(str)
    frame["run_id"] = frame["run_id"].astype(str)
    frame["model_alias"] = frame["model_alias"].astype(str)
    frame["step"] = pd.to_numeric(frame["step"], errors="raise").astype(int)
    frame["correct"] = pd.to_numeric(frame["correct"], errors="raise").astype(int)
    if not frame["correct"].isin([0, 1]).all():
        raise ValueError("correct must be a binary evaluation label")
    frame["trajectory_id"] = frame["source_cell"].astype(str) + "::" + frame["run_id"]
    frame = frame.sort_values(["trajectory_id", "step"], kind="stable").reset_index(drop=True)

    duplicate = frame.duplicated(["task_id", "step", "model_alias"], keep=False)
    if duplicate.any():
        examples = frame.loc[duplicate, ["task_id", "step", "model_alias", "source_cell"]].head(10)
        raise ValueError(
            "A strict committee requires one candidate per task/step/model alias; "
            f"duplicates found: {examples.to_dict('records')}"
        )
    trajectory_steps = frame.groupby("trajectory_id", sort=False)["step"].nunique()
    if trajectory_steps.min() < 1:
        raise ValueError("Empty trajectory detected")
    return frame, file_manifest


def build_prefix_and_committee_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct causal candidate features and leave-one-alias-out votes.

    The aggregation is within (task_id, step), but only uses peer answer keys.
    Because outer folds put *all* candidates of a task on one side, this uses no
    response labels or training-fold information from an outer test task.
    """

    result = frame.copy()
    result["answer_key"] = result["answer_normalized"].map(answer_key)
    result["answer_nonempty"] = (result["answer_key"] != "").astype(np.int8)

    nonempty = result.loc[result["answer_nonempty"] == 1]
    panel_size = nonempty.groupby(["task_id", "step"], sort=False).size().rename("committee_panel_nonempty")
    matching_count = (
        nonempty.groupby(["task_id", "step", "answer_key"], sort=False)
        .size()
        .rename("same_answer_count")
    )
    result = result.join(panel_size, on=["task_id", "step"])
    result = result.join(matching_count, on=["task_id", "step", "answer_key"])
    result["committee_panel_nonempty"] = result["committee_panel_nonempty"].fillna(0.0).astype(np.float32)
    result["same_answer_count"] = result["same_answer_count"].fillna(0.0).astype(np.float32)

    # A candidate never votes for itself.  An empty peer response is absent rather
    # than counted as a disagreeing answer.
    result["independent_vote_count"] = (
        result["same_answer_count"] - result["answer_nonempty"]
    ).clip(lower=0.0)
    result["independent_peer_count"] = (
        result["committee_panel_nonempty"] - result["answer_nonempty"]
    ).clip(lower=0.0)
    result["independent_vote_fraction"] = (
        result["independent_vote_count"] / result["independent_peer_count"].clip(lower=1.0)
    )
    result["committee_has_independent_match"] = (result["independent_vote_count"] > 0).astype(np.int8)
    aliases_per_barrier = result.groupby(["task_id", "step"], sort=False)["model_alias"].transform("nunique")
    result["committee_nonempty_fraction"] = result["committee_panel_nonempty"] / aliases_per_barrier.clip(lower=1)

    previous_answer = result.groupby("trajectory_id", sort=False)["answer_key"].shift(1)
    result["answer_char_len"] = result["answer_key"].str.len().astype(np.float32)
    result["same_prev_answer"] = (
        (result["answer_nonempty"] == 1) & (result["answer_key"] == previous_answer)
    ).astype(np.int8)
    changed = (
        (result["answer_nonempty"] == 1)
        & previous_answer.notna()
        & (result["answer_key"] != previous_answer)
    ).astype(np.int8)
    result["n_answer_changes_prefix"] = (
        changed.groupby(result["trajectory_id"], sort=False).cumsum().astype(np.float32)
    )

    for column in PREFIX_DELTA_COLUMNS:
        values = pd.Series(pd.to_numeric(result.get(column), errors="coerce"), index=result.index)
        result[f"delta_{column}"] = values.groupby(result["trajectory_id"], sort=False).diff().fillna(0.0)
    for column in LOG1P_COLUMNS:
        values = pd.Series(pd.to_numeric(result.get(column), errors="coerce"), index=result.index)
        result[f"log1p_{column}"] = np.log1p(values.clip(lower=0.0))
    return result


def committee_feature_self_test() -> None:
    """Regression test for the no-self-vote and no-empty-disagreement rules."""

    fixture = pd.DataFrame(
        {
            "task_id": ["task"] * 8,
            "step": [1, 1, 1, 1, 2, 2, 2, 2],
            "model_alias": ["a", "b", "c", "d", "a", "b", "c", "d"],
            "trajectory_id": ["a", "b", "c", "d", "a", "b", "c", "d"],
            "answer_normalized": ["42", "42", "7", None, "42", "7", "7", ""],
        }
    )
    # The remaining columns are only needed by causal numeric transforms.
    for column in sorted(set(PREFIX_DELTA_COLUMNS) | set(LOG1P_COLUMNS)):
        fixture[column] = 0.0
    output = build_prefix_and_committee_features(fixture)
    first = output.loc[output["step"] == 1].set_index("model_alias")
    if float(first.loc["a", "independent_vote_count"]) != 1.0:
        raise AssertionError("Scored alias incorrectly included itself in its agreement count")
    if float(first.loc["c", "independent_vote_count"]) != 0.0:
        raise AssertionError("A disagreement incorrectly became a peer vote")
    if float(first.loc["d", "independent_peer_count"]) != 3.0:
        raise AssertionError("An empty scored answer should not remove a nonempty peer")
    second = output.loc[output["step"] == 2].set_index("model_alias")
    if float(second.loc["b", "independent_vote_count"]) != 1.0:
        raise AssertionError("Leave-one-model-out count failed after a peer answer changed")
    print("Committee feature self-test passed.")


def telemetry_feature_columns() -> list[str]:
    return BASE_NUMERIC_COLUMNS + DERIVED_TELEMETRY_COLUMNS + [
        f"delta_{column}" for column in PREFIX_DELTA_COLUMNS
    ] + [f"log1p_{column}" for column in LOG1P_COLUMNS]


def validate_feature_contract(numeric_columns: Iterable[str], categorical_columns: Iterable[str]) -> None:
    selected = set(numeric_columns) | set(categorical_columns)
    forbidden = sorted(selected & FORBIDDEN_MODEL_COLUMNS)
    if forbidden:
        raise AssertionError(f"Forbidden label/future/provenance features selected: {forbidden}")
    if any(column.startswith("k2_") for column in selected):
        raise AssertionError("K2 is a separately-costed generation and is forbidden in the strict pre-K2 model")


def make_fold_matrices(
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit imputation/category maps only on outer-train tasks."""

    train = pd.DataFrame(index=np.arange(len(train_indices)))
    test = pd.DataFrame(index=np.arange(len(test_indices)))
    train_frame = frame.iloc[train_indices]
    test_frame = frame.iloc[test_indices]
    for column in numeric_columns:
        train_values = pd.to_numeric(train_frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        test_values = pd.to_numeric(test_frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = float(train_values.median()) if train_values.notna().any() else 0.0
        train[column] = train_values.fillna(median).to_numpy(dtype=np.float32)
        test[column] = test_values.fillna(median).to_numpy(dtype=np.float32)
    for column in categorical_columns:
        train_values = train_frame[column].fillna("__MISSING__").astype(str)
        test_values = test_frame[column].fillna("__MISSING__").astype(str)
        category_map = {value: code for code, value in enumerate(pd.unique(train_values))}
        train[column] = train_values.map(category_map).fillna(-1).to_numpy(dtype=np.int32)
        test[column] = test_values.map(category_map).fillna(-1).to_numpy(dtype=np.int32)
    return train, test


def fit_outer_cv(
    frame: pd.DataFrame,
    *,
    name: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
    n_splits: int,
    seed: int,
    jobs: int,
    config: LightGBMConfig,
) -> CVResult:
    """Return task-held-out OOF scores under one frozen feature contract."""

    validate_feature_contract(numeric_columns, categorical_columns)
    labels = frame["correct"].to_numpy(dtype=np.int8)
    groups = frame["task_id"].to_numpy(dtype=object)
    splitter = GroupKFold(n_splits=n_splits)
    oof = np.full(len(frame), np.nan, dtype=np.float64)
    fold_ids = np.full(len(frame), -1, dtype=np.int16)
    fold_auc: list[float] = []

    for fold, (train_indices, test_indices) in enumerate(splitter.split(frame, labels, groups), start=1):
        train_groups = set(groups[train_indices].tolist())
        test_groups = set(groups[test_indices].tolist())
        overlap = train_groups & test_groups
        if overlap:
            raise AssertionError(f"Task leakage in fold {fold}: {sorted(overlap)[:5]}")
        train_x, test_x = make_fold_matrices(
            frame, train_indices, test_indices, numeric_columns, categorical_columns
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
            n_jobs=jobs,
            random_state=seed + fold,
            verbosity=-1,
        )
        model.fit(train_x, labels[train_indices], categorical_feature=categorical_columns)
        probabilities = model.predict_proba(test_x)[:, 1]
        oof[test_indices] = probabilities
        fold_ids[test_indices] = fold
        auc = float(roc_auc_score(labels[test_indices], probabilities))
        fold_auc.append(auc)
        print(f"[{name}] fold {fold}/{n_splits}: test_rows={len(test_indices)} auc={auc:.6f}", flush=True)

    if not np.isfinite(oof).all() or (fold_ids < 1).any():
        raise AssertionError(f"{name} does not provide complete OOF coverage")
    return CVResult(
        name=name,
        oof=oof,
        fold_ids=fold_ids,
        fold_auc=fold_auc,
        feature_columns=list(numeric_columns),
        categorical_columns=list(categorical_columns),
    )


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignment = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, bins - 1)
    ece = 0.0
    for bucket in range(bins):
        mask = assignment == bucket
        if mask.any():
            ece += float(mask.mean()) * abs(float(labels[mask].mean()) - float(probabilities[mask].mean()))
    return float(ece)


def cross_fold_platt(labels: np.ndarray, scores: np.ndarray, fold_ids: np.ndarray) -> np.ndarray:
    """Calibrate each outer test fold using only the other folds' OOF labels.

    This is a reporting calibration layer, not an input to training or AUC
    selection.  It prevents the older tournament's temperature-only / weighted
    posterior issue from being silently reproduced in ECE reporting.
    """

    result = np.empty_like(scores, dtype=np.float64)
    for fold in sorted(np.unique(fold_ids)):
        fit = fold_ids != fold
        apply = fold_ids == fold
        calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        calibrator.fit(scores[fit].reshape(-1, 1), labels[fit])
        result[apply] = calibrator.predict_proba(scores[apply].reshape(-1, 1))[:, 1]
    return result


def task_macro_auc(frame: pd.DataFrame, scores: np.ndarray) -> tuple[float, int]:
    values: list[float] = []
    for _, group in frame.groupby("task_id", sort=False):
        indices = group.index.to_numpy(dtype=np.int64)
        labels = group["correct"].to_numpy(dtype=np.int8)
        if np.unique(labels).size == 2:
            values.append(float(roc_auc_score(labels, scores[indices])))
    return (float(np.mean(values)) if values else float("nan"), len(values))


def task_cluster_bootstrap_auc(
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float] | None:
    """Two-sided 95% percentile CI resampling tasks, not correlated rows."""

    if replicates <= 0:
        return None
    groups = frame.groupby("task_id", sort=False).indices
    task_indices = list(groups.values())
    generator = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        chosen = generator.integers(0, len(task_indices), size=len(task_indices))
        indices = np.concatenate([task_indices[index] for index in chosen])
        labels = frame["correct"].to_numpy(dtype=np.int8)[indices]
        if np.unique(labels).size == 2:
            values.append(float(roc_auc_score(labels, scores[indices])))
    if not values:
        return None
    return (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))


def evaluate_result(
    frame: pd.DataFrame,
    result: CVResult,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    labels = frame["correct"].to_numpy(dtype=np.int8)
    calibrated = cross_fold_platt(labels, result.oof, result.fold_ids)
    eligible = frame["step"].to_numpy(dtype=int) >= 2
    macro_auc, macro_count = task_macro_auc(frame, result.oof)
    per_domain = {
        str(domain): {
            "rows": int(len(group)),
            "auc": float(roc_auc_score(group["correct"], result.oof[group.index])),
        }
        for domain, group in frame.groupby("domain", sort=True)
        if group["correct"].nunique() == 2
    }
    per_model = {
        str(model): {
            "rows": int(len(group)),
            "auc": float(roc_auc_score(group["correct"], result.oof[group.index])),
        }
        for model, group in frame.groupby("model_alias", sort=True)
        if group["correct"].nunique() == 2
    }
    return {
        "name": result.name,
        "rows": int(len(frame)),
        "task_groups": int(frame["task_id"].nunique()),
        "oof_auc": float(roc_auc_score(labels, result.oof)),
        "eligible_step_gte_2_auc": float(roc_auc_score(labels[eligible], result.oof[eligible])),
        "fold_auc": result.fold_auc,
        "fold_auc_mean": float(np.mean(result.fold_auc)),
        "fold_auc_std": float(np.std(result.fold_auc, ddof=1)),
        "raw_brier": float(brier_score_loss(labels, result.oof)),
        "raw_ece_15": expected_calibration_error(labels, result.oof, bins=15),
        "cross_fold_platt_brier": float(brier_score_loss(labels, calibrated)),
        "cross_fold_platt_ece_15": expected_calibration_error(labels, calibrated, bins=15),
        "task_macro_auc": macro_auc,
        "tasks_with_both_classes_for_macro_auc": int(macro_count),
        "task_cluster_bootstrap_auc_95_ci": task_cluster_bootstrap_auc(
            frame, result.oof, replicates=bootstrap_replicates, seed=seed
        ),
        "per_domain": per_domain,
        "per_model_alias": per_model,
        "numeric_features": result.feature_columns,
        "categorical_features": result.categorical_columns,
    }


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory already contains artifacts: {path}. Choose a new --output-dir or pass --overwrite."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_bundle(
    output_dir: Path,
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    telemetry: CVResult,
    committee: CVResult,
    metrics: dict[str, Any],
) -> None:
    atomic_write_json(output_dir / "committee_oof_manifest.json", manifest)
    atomic_write_json(output_dir / "committee_oof_metrics.json", metrics)
    rows = frame[
        ["task_id", "trajectory_id", "source_cell", "model_alias", "domain", "step", "correct"]
    ].copy()
    rows["fold"] = committee.fold_ids
    rows["telemetry_score"] = telemetry.oof
    rows["strict_committee_score"] = committee.oof
    rows["independent_vote_count"] = frame["independent_vote_count"].to_numpy()
    rows["independent_peer_count"] = frame["independent_peer_count"].to_numpy()
    rows["independent_vote_fraction"] = frame["independent_vote_fraction"].to_numpy()
    temporary = output_dir / "committee_oof_predictions.csv.tmp"
    rows.to_csv(temporary, index=False)
    os.replace(temporary, output_dir / "committee_oof_predictions.csv")

    telemetry_metrics = metrics["telemetry"]
    committee_metrics = metrics["strict_committee"]
    body = "\n".join(
        [
            "=" * 108,
            "STRICT TASK-HELD-OUT FLEET-COMMITTEE OOF EXPERIMENT",
            "=" * 108,
            f"Corpus: {manifest['rows']} rows, {manifest['trajectories']} source-qualified trajectories, "
            f"{manifest['task_groups']} task groups, {manifest['cells']} canonical cells.",
            "Protocol: GroupKFold(task_id), one candidate per task/step/model alias, leave-target-alias-out votes.",
            "Strict feature contract: no labels/gold answers/task IDs/raw answer text/source cell/future/K2 fields.",
            "Important deployment condition: peers must complete the same step barrier before a score is consumed.",
            "",
            "Configuration       | OOF AUC | Eligible t>=2 | Raw ECE15 | Platt ECE15 | Task macro AUC | Task bootstrap 95% CI",
            "-" * 108,
            (
                f"Telemetry only      | {telemetry_metrics['oof_auc']:.6f} | "
                f"{telemetry_metrics['eligible_step_gte_2_auc']:.6f} | "
                f"{telemetry_metrics['raw_ece_15']:.6f} | {telemetry_metrics['cross_fold_platt_ece_15']:.6f} | "
                f"{telemetry_metrics['task_macro_auc']:.6f} | {telemetry_metrics['task_cluster_bootstrap_auc_95_ci']}"
            ),
            (
                f"Strict committee    | {committee_metrics['oof_auc']:.6f} | "
                f"{committee_metrics['eligible_step_gte_2_auc']:.6f} | "
                f"{committee_metrics['raw_ece_15']:.6f} | {committee_metrics['cross_fold_platt_ece_15']:.6f} | "
                f"{committee_metrics['task_macro_auc']:.6f} | {committee_metrics['task_cluster_bootstrap_auc_95_ci']}"
            ),
            "",
            "Per-domain strict-committee AUC:",
            *[
                f"  {domain}: {values['auc']:.6f} ({values['rows']} rows)"
                for domain, values in committee_metrics["per_domain"].items()
            ],
            "",
            "This is a fleet-committee correctness result, not proof of net stopping utility. "
            "Prospective timestamped rollouts, fixed roster/seed replication, and a separate post-K2 policy "
            "remain required before claiming production stopping gains.",
            "",
        ]
    )
    atomic_write_text(output_dir / "committee_oof_results.log", body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate the panel and feature contract without fitting.")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic no-self-vote regression checks.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        committee_feature_self_test()
        return 0
    if args.n_splits < 2:
        raise ValueError("--n-splits must be at least two")
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    frame, files = load_canonical_panel(args.input_dir)
    frame = build_prefix_and_committee_features(frame)
    numeric_telemetry = telemetry_feature_columns()
    numeric_committee = numeric_telemetry + STRICT_COMMITTEE_COLUMNS
    validate_feature_contract(numeric_telemetry, CATEGORICAL_COLUMNS)
    validate_feature_contract(numeric_committee, CATEGORICAL_COLUMNS)
    missing = sorted((set(numeric_committee) | set(CATEGORICAL_COLUMNS)) - set(frame.columns))
    if missing:
        raise ValueError(f"Input corpus is missing deployment-safe feature columns: {missing}")
    task_groups = int(frame["task_id"].nunique())
    if task_groups < args.n_splits:
        raise ValueError(f"Need at least {args.n_splits} task groups; found {task_groups}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "started_at_unix": time.time(),
        "script_sha256": raw_file_hash(Path(__file__).resolve()),
        "input_dir": str(args.input_dir),
        "cells": len(files),
        "rows": int(len(frame)),
        "trajectories": int(frame["trajectory_id"].nunique()),
        "task_groups": task_groups,
        "model_aliases": int(frame["model_alias"].nunique()),
        "steps_per_trajectory": {
            str(length): int(count)
            for length, count in frame.groupby("trajectory_id", sort=False).size().value_counts().sort_index().items()
        },
        "files": files,
        "canonical_dataset_fingerprint": stable_hash(
            [{"path": item["path"], "canonical_lf_sha256": item["canonical_lf_sha256"]} for item in files]
        ),
        "lightgbm": asdict(LightGBMConfig()),
        "n_splits": args.n_splits,
        "seed": args.seed,
        "strict_contract": {
            "outer_group": "task_id",
            "committee_vote": "same task and same step; distinct model alias; scored alias excluded",
            "required_production_barrier": "all peer outputs timestamped complete at or before decision t",
            "excluded": sorted(FORBIDDEN_MODEL_COLUMNS),
            "telemetry_numeric": numeric_telemetry,
            "committee_numeric": numeric_committee,
            "categorical": CATEGORICAL_COLUMNS,
        },
    }
    manifest["run_id"] = stable_hash(
        {
            "schema": SCHEMA_VERSION,
            "dataset": manifest["canonical_dataset_fingerprint"],
            "script": manifest["script_sha256"],
            "seed": args.seed,
            "splits": args.n_splits,
            "lightgbm": manifest["lightgbm"],
        }
    )[:20]
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    prepare_output_dir(args.output_dir, args.overwrite)
    config = LightGBMConfig()
    telemetry = fit_outer_cv(
        frame,
        name="telemetry",
        numeric_columns=numeric_telemetry,
        categorical_columns=CATEGORICAL_COLUMNS,
        n_splits=args.n_splits,
        seed=args.seed,
        jobs=args.jobs,
        config=config,
    )
    committee = fit_outer_cv(
        frame,
        name="strict_committee",
        numeric_columns=numeric_committee,
        categorical_columns=CATEGORICAL_COLUMNS,
        n_splits=args.n_splits,
        seed=args.seed,
        jobs=args.jobs,
        config=config,
    )
    metrics = {
        "telemetry": evaluate_result(
            frame, telemetry, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed
        ),
        "strict_committee": evaluate_result(
            frame, committee, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed + 1
        ),
    }
    manifest["completed_at_unix"] = time.time()
    manifest["status"] = "complete"
    write_bundle(args.output_dir, frame, manifest, telemetry, committee, metrics)
    print((args.output_dir / "committee_oof_results.log").read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; no partial result bundle was published.", file=sys.stderr)
        raise SystemExit(130)
