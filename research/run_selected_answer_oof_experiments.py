#!/usr/bin/env python3
"""Task-held-out evaluation of an actual selected-answer committee policy.

The historical committee experiment scores every candidate model response.  That
is useful for measuring conditional correctness, but it is not itself a live
answer-selection-and-stop policy: a deployment must select one answer at each
task/step barrier.  This program closes that gap retrospectively and, crucially,
keeps the same strict causal feature boundary:

* one deterministic plurality-selected answer per ``(task_id, step)``;
* answer strings are used only transiently to choose the plurality and derive
  counts/margins; they are never passed to the learner;
* outer validation is ``GroupKFold(task_id)``;
* no gold answer, correctness, utility, task/run/source IDs, raw text, K2, or
  future barriers can become a model feature.

It produces a *decision-level* OOF probability of the selected answer being
correct.  Historical trace files lack barrier timestamps, so even a strong score
from this program remains conditional on a future synchronized-fleet rollout.
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


SCHEMA_VERSION = "selected-answer-oof-v4-winner-panel"
DEFAULT_INPUT = Path("research/outputs/experiments_v2")
DEFAULT_OUTPUT = Path("research/outputs/experiments_v2/selected_answer_oof_v4_winner_panel")
EPS = 1.0e-12

# Raw output text and identity keys may appear in the source frame or output
# artifact for auditing, but never in FEATURE columns below.
FORBIDDEN_FEATURES = {
    "correct",
    "selected_correct",
    "expected_answer",
    "utility",
    "task_id",
    "run_id",
    "trajectory_id",
    "source_cell",
    "task_source_index",
    "answer",
    "answer_normalized",
    "answer_key",
    "selected_answer_key",
    "selected_answer_hash",
    "thought",
    "raw_text",
    "k2_agreement",
    "k2_raw_generation_tokens",
}

PANEL_AGGREGATE_SOURCE = [
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

# Model, task/domain, and response-provenance identities remain excluded rather
# than being encoded indirectly as category values.
CATEGORICAL_CANDIDATES: list[str] = []


@dataclass(frozen=True)
class LightGBMConfig:
    n_estimators: int = 600
    learning_rate: float = 0.035
    num_leaves: int = 31
    min_child_samples: int = 45
    subsample: float = 0.88
    colsample_bytree: float = 0.82
    reg_lambda: float = 4.0
    reg_alpha: float = 0.12


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def raw_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    pending_cr = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            block = pending_cr + block
            if block.endswith(b"\r"):
                pending_cr = b"\r"
                block = block[:-1]
            else:
                pending_cr = b""
            digest.update(block.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
    if pending_cr:
        digest.update(b"\n")
    return digest.hexdigest()


def normalized_answer_key(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().casefold()


def load_panel(input_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    paths = sorted(input_dir.glob("global_*/trace_steps.csv"))
    if not paths:
        raise FileNotFoundError(f"No global_*/trace_steps.csv files found under {input_dir}")
    required = {"task_id", "step", "model_alias", "answer_normalized", "correct"}
    frames: list[pd.DataFrame] = []
    manifest: list[dict[str, Any]] = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        frame["source_cell"] = path.parent.name
        frames.append(frame)
        manifest.append(
            {
                "path": path.as_posix(),
                "bytes": int(path.stat().st_size),
                "raw_sha256": raw_file_hash(path),
                "canonical_lf_sha256": canonical_file_hash(path),
            }
        )
    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame["task_id"] = frame["task_id"].astype(str)
    frame["model_alias"] = frame["model_alias"].astype(str)
    frame["step"] = pd.to_numeric(frame["step"], errors="raise").astype(np.int16)
    frame["correct"] = pd.to_numeric(frame["correct"], errors="raise").astype(np.int8)
    if not frame["correct"].isin([0, 1]).all():
        raise ValueError("correct must be binary")
    duplicates = frame.duplicated(["task_id", "step", "model_alias"], keep=False)
    if duplicates.any():
        example = frame.loc[duplicates, ["task_id", "step", "model_alias", "source_cell"]].head(10)
        raise ValueError(
            "A selected-answer panel requires one response per task/step/model alias; "
            f"duplicates found: {example.to_dict('records')}"
        )
    return frame, manifest


def numeric(values: pd.Series | Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def entropy_from_counts(counts: pd.Series, totals: pd.Series) -> pd.Series:
    probabilities = counts / totals.clip(lower=1)
    return -(probabilities * np.log(probabilities.clip(lower=EPS)))


def build_decision_frame(
    frame: pd.DataFrame, *, exclude_batch_timing: bool = False
) -> tuple[pd.DataFrame, list[str], list[str], tuple[str, ...]]:
    """Choose one deterministic plurality answer and construct legal features.

    ``answer_key`` is deliberately short-lived.  It is necessary to implement
    a plurality policy but is removed before matrix construction and is guarded
    by ``validate_feature_contract``.
    """
    work = frame.copy()
    work["answer_key"] = work["answer_normalized"].map(normalized_answer_key)
    work["answer_nonempty"] = (work["answer_key"] != "").astype(np.int8)
    aliases = tuple(sorted(work["model_alias"].unique().tolist()))
    alias_rank = {alias: index for index, alias in enumerate(aliases)}
    work["_alias_rank"] = work["model_alias"].map(alias_rank).astype(np.int16)
    barrier_keys = ["task_id", "step"]

    valid = work.loc[work["answer_nonempty"] == 1].copy()
    if valid.empty:
        raise ValueError("No nonempty answers are available for selected-answer policy construction")
    # Vote table has one record per observed normalized answer.  The first
    # supporting alias is an explicit, frozen tie-break—not a learned feature.
    votes = (
        valid.groupby(barrier_keys + ["answer_key"], sort=False)
        .agg(vote_count=("model_alias", "size"), first_alias_rank=("_alias_rank", "min"))
        .reset_index()
    )
    valid_count = valid.groupby(barrier_keys, sort=False).size().rename("valid_answer_count")
    votes = votes.join(valid_count, on=barrier_keys)
    votes["vote_fraction"] = votes["vote_count"] / votes["valid_answer_count"].clip(lower=1)
    votes["vote_entropy_term"] = entropy_from_counts(votes["vote_count"], votes["valid_answer_count"])
    votes["_is_top_vote"] = votes["vote_count"].eq(
        votes.groupby(barrier_keys, sort=False)["vote_count"].transform("max")
    )
    votes = votes.sort_values(
        barrier_keys + ["vote_count", "first_alias_rank", "answer_key"],
        ascending=[True, True, False, True, True],
        kind="stable",
    )
    winners = votes.drop_duplicates(barrier_keys, keep="first").copy()
    winners = winners.rename(
        columns={
            "answer_key": "selected_answer_key",
            "vote_count": "selected_vote_count",
            "vote_fraction": "selected_support_fraction",
        }
    )
    # Second-largest answer vote and entropy are panel aggregates, not text.
    ranked = votes.copy()
    ranked["_vote_rank"] = ranked.groupby(barrier_keys, sort=False).cumcount()
    second = ranked.loc[ranked["_vote_rank"] == 1, barrier_keys + ["vote_count"]].rename(
        columns={"vote_count": "second_vote_count"}
    )
    vote_entropy = votes.groupby(barrier_keys, sort=False)["vote_entropy_term"].sum().rename("answer_vote_entropy")
    unique_answers = votes.groupby(barrier_keys, sort=False).size().rename("unique_nonempty_answers")
    top_tie_count = votes.groupby(barrier_keys, sort=False)["_is_top_vote"].sum().rename("top_tie_count")
    winner_summary = winners.join(second.set_index(barrier_keys), on=barrier_keys)
    winner_summary = (
        winner_summary.join(vote_entropy, on=barrier_keys)
        .join(unique_answers, on=barrier_keys)
        .join(top_tie_count, on=barrier_keys)
    )
    winner_summary["second_vote_count"] = winner_summary["second_vote_count"].fillna(0.0)
    winner_summary["selected_vote_margin_count"] = (
        winner_summary["selected_vote_count"] - winner_summary["second_vote_count"]
    )
    winner_summary["selected_vote_margin_fraction"] = (
        winner_summary["selected_vote_margin_count"]
        / winner_summary["valid_answer_count"].clip(lower=1)
    )

    # Select a concrete response that supports the selected answer using the
    # same frozen alias order.  Its correctness is the one decision label.
    selected = work.merge(
        winner_summary[barrier_keys + ["selected_answer_key"]],
        how="inner",
        left_on=barrier_keys + ["answer_key"],
        right_on=barrier_keys + ["selected_answer_key"],
        validate="many_to_one",
    )
    selected = selected.sort_values(barrier_keys + ["_alias_rank"], kind="stable")
    selected = selected.drop_duplicates(barrier_keys, keep="first").copy()
    selected = selected.merge(
        winner_summary.drop(columns=["first_alias_rank", "vote_entropy_term"], errors="ignore"),
        on=barrier_keys + ["selected_answer_key"],
        how="left",
        validate="one_to_one",
    )
    if selected.duplicated(barrier_keys).any():
        raise AssertionError("Plurality selection did not yield one decision per barrier")
    expected_barriers = int(work.groupby(barrier_keys, sort=False).ngroups)
    if len(selected) != expected_barriers:
        # A future deployment must specify an explicit abstain/escalate policy
        # for an all-empty panel.  Failing closed is preferable to silently
        # dropping those barriers from an OOF denominator.
        raise ValueError(
            "At least one barrier has no nonempty answer. Define and audit an explicit "
            "all-empty fallback before evaluating this selected-answer policy."
        )

    panel = work.groupby(barrier_keys, sort=False)
    panel_size = panel["model_alias"].nunique().rename("panel_model_count")
    panel_valid = panel["answer_nonempty"].sum().rename("panel_nonempty_count")
    parse_fraction = panel["parse_success"].mean().rename("panel_parse_success_fraction")
    selected = selected.join(panel_size, on=barrier_keys).join(panel_valid, on=barrier_keys).join(parse_fraction, on=barrier_keys)
    selected["panel_nonempty_fraction"] = selected["panel_nonempty_count"] / selected["panel_model_count"].clip(lower=1)

    # Aggregated telemetry is available only after all peer responses at the
    # current barrier have completed.  We use two anonymous populations: all
    # panel members and the members that support the deterministic winner.
    # Neither the answer strings nor model identities enter the learner.
    aggregate_columns = [column for column in PANEL_AGGREGATE_SOURCE if column in work.columns]
    if exclude_batch_timing:
        aggregate_columns = [
            column for column in aggregate_columns if column not in {"elapsed_seconds", "tokens_per_second"}
        ]
    winner_members = work.merge(
        winner_summary[barrier_keys + ["selected_answer_key"]],
        how="inner",
        left_on=barrier_keys + ["answer_key"],
        right_on=barrier_keys + ["selected_answer_key"],
        validate="many_to_one",
    )
    winner_members = winner_members.loc[
        (winner_members["answer_nonempty"] == 1)
        & (winner_members["answer_key"] == winner_members["selected_answer_key"])
    ].copy()
    inconsistent = winner_members.groupby(barrier_keys, sort=False)["correct"].nunique()
    if (inconsistent > 1).any():
        example = inconsistent.loc[inconsistent > 1].head(3).to_dict()
        raise ValueError(f"Selected normalized answers have inconsistent correctness labels: {example}")
    for prefix, population in (("panel", work), ("winner", winner_members)):
        for column in aggregate_columns:
            values = numeric(population[column])
            temp = pd.DataFrame(
                {"task_id": population["task_id"], "step": population["step"], "value": values}
            )
            stats = temp.groupby(barrier_keys, sort=False)["value"].agg(["mean", "std", "min", "max"])
            stats = stats.rename(columns={stat: f"{prefix}_{column}_{stat}" for stat in stats.columns})
            selected = selected.join(stats, on=barrier_keys)

    selected["selected_model_alias"] = selected["model_alias"].astype(str)
    selected = selected.sort_values(["task_id", "step"], kind="stable").reset_index(drop=True)
    previous_key = selected.groupby("task_id", sort=False)["selected_answer_key"].shift(1)
    selected["plurality_same_prev"] = (
        previous_key.notna() & (selected["selected_answer_key"] == previous_key)
    ).astype(np.int8)
    selected["winner_tied"] = (selected["top_tie_count"] > 1).astype(np.int8)
    for column in [
        "selected_support_fraction",
        "selected_vote_margin_fraction",
        "panel_nonempty_fraction",
        "unique_nonempty_answers",
        "selected_vote_count",
    ]:
        values = numeric(selected[column])
        name = {
            "selected_support_fraction": "delta_winner_vote_fraction",
            "selected_vote_margin_fraction": "delta_winner_margin_fraction",
            "panel_nonempty_fraction": "delta_panel_nonempty_fraction",
            "unique_nonempty_answers": "delta_unique_nonempty_answers",
            "selected_vote_count": "delta_winner_vote_count",
        }[column]
        selected[name] = values.groupby(selected["task_id"], sort=False).diff().fillna(0.0)

    # Keep the label under an explicit decision-level name to make accidental
    # row-level interpretation difficult.
    selected["selected_correct"] = selected["correct"].astype(np.int8)
    selected["selected_answer_hash"] = selected["selected_answer_key"].map(
        lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    )
    selected["no_nonempty_answer"] = 0
    numeric_columns = [
        "step",
        "panel_model_count",
        "panel_nonempty_count",
        "unique_nonempty_answers",
        "selected_vote_count",
        "second_vote_count",
        "top_tie_count",
        "no_nonempty_answer",
        "panel_nonempty_fraction",
        "selected_support_fraction",
        "selected_vote_margin_count",
        "selected_vote_margin_fraction",
        "winner_tied",
        "plurality_same_prev",
        "delta_winner_vote_fraction",
        "delta_winner_margin_fraction",
        "delta_panel_nonempty_fraction",
        "delta_unique_nonempty_answers",
        "delta_winner_vote_count",
    ]
    numeric_columns += [
        f"{prefix}_{column}_{stat}"
        for prefix in ("panel", "winner")
        for column in aggregate_columns
        for stat in ("mean", "std", "min", "max")
    ]
    numeric_columns = [column for column in numeric_columns if column in selected.columns]
    categorical_columns = [column for column in CATEGORICAL_CANDIDATES if column in selected.columns]
    validate_feature_contract(numeric_columns, categorical_columns)
    return selected, numeric_columns, categorical_columns, aliases


def validate_feature_contract(numeric_columns: Iterable[str], categorical_columns: Iterable[str]) -> None:
    selected = set(numeric_columns) | set(categorical_columns)
    forbidden = sorted(selected & FORBIDDEN_FEATURES)
    if forbidden:
        raise AssertionError(f"Forbidden label/raw/provenance fields selected: {forbidden}")
    k2 = sorted(column for column in selected if column.startswith("k2_"))
    if k2:
        raise AssertionError(f"K2 is a separately-costed post-query generation and is excluded here: {k2}")


def make_matrices(
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame = frame.iloc[train_indices]
    test_frame = frame.iloc[test_indices]
    train_data: dict[str, np.ndarray] = {}
    test_data: dict[str, np.ndarray] = {}
    for column in numeric_columns:
        train_values = numeric(train_frame[column])
        test_values = numeric(test_frame[column])
        median = float(train_values.median()) if train_values.notna().any() else 0.0
        train_data[column] = train_values.fillna(median).to_numpy(dtype=np.float32)
        test_data[column] = test_values.fillna(median).to_numpy(dtype=np.float32)
    for column in categorical_columns:
        train_values = train_frame[column].fillna("__MISSING__").astype(str)
        test_values = test_frame[column].fillna("__MISSING__").astype(str)
        mapping = {value: code for code, value in enumerate(pd.unique(train_values))}
        train_data[column] = train_values.map(mapping).fillna(-1).to_numpy(dtype=np.int32)
        test_data[column] = test_values.map(mapping).fillna(-1).to_numpy(dtype=np.int32)
    return pd.DataFrame(train_data), pd.DataFrame(test_data)


def fit_oof(
    frame: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    *,
    n_splits: int,
    seed: int,
    jobs: int,
    config: LightGBMConfig,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    labels = frame["selected_correct"].to_numpy(dtype=np.int8)
    groups = frame["task_id"].to_numpy(dtype=object)
    splitter = GroupKFold(n_splits=n_splits)
    oof = np.full(len(frame), np.nan, dtype=np.float64)
    fold_ids = np.full(len(frame), -1, dtype=np.int16)
    fold_auc: list[float] = []
    for fold, (train_indices, test_indices) in enumerate(splitter.split(frame, labels, groups), start=1):
        train_groups = set(groups[train_indices].tolist())
        test_groups = set(groups[test_indices].tolist())
        if train_groups & test_groups:
            raise AssertionError(f"Task leakage in fold {fold}")
        train_x, test_x = make_matrices(frame, train_indices, test_indices, numeric_columns, categorical_columns)
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
        probability = model.predict_proba(test_x)[:, 1]
        oof[test_indices] = probability
        fold_ids[test_indices] = fold
        auc = float(roc_auc_score(labels[test_indices], probability))
        fold_auc.append(auc)
        print(f"[selected_answer_lgbm] fold {fold}/{n_splits}: rows={len(test_indices)} auc={auc:.6f}", flush=True)
    if not np.isfinite(oof).all() or (fold_ids < 1).any():
        raise AssertionError("OOF coverage is incomplete")
    return oof, fold_ids, fold_auc


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    labels = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(values, edges[1:-1], right=False), 0, bins - 1)
    result = 0.0
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            result += float(mask.mean()) * abs(float(labels[mask].mean()) - float(values[mask].mean()))
    return float(result)


def cross_fold_platt(labels: np.ndarray, scores: np.ndarray, fold_ids: np.ndarray) -> np.ndarray:
    calibrated = np.empty_like(scores, dtype=np.float64)
    for fold in sorted(np.unique(fold_ids)):
        train = fold_ids != fold
        test = fold_ids == fold
        calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        calibrator.fit(scores[train].reshape(-1, 1), labels[train])
        calibrated[test] = calibrator.predict_proba(scores[test].reshape(-1, 1))[:, 1]
    return calibrated


def task_macro_auc(frame: pd.DataFrame, scores: np.ndarray) -> tuple[float, int]:
    values: list[float] = []
    for _, group in frame.groupby("task_id", sort=False):
        labels = group["selected_correct"].to_numpy(dtype=np.int8)
        indices = group.index.to_numpy(dtype=np.int64)
        if np.unique(labels).size == 2:
            values.append(float(roc_auc_score(labels, scores[indices])))
    return (float(np.mean(values)) if values else float("nan"), len(values))


def cluster_bootstrap_auc(frame: pd.DataFrame, scores: np.ndarray, *, seed: int, replicates: int) -> tuple[float, float] | None:
    if replicates <= 0:
        return None
    group_indices = list(frame.groupby("task_id", sort=False).indices.values())
    generator = np.random.default_rng(seed)
    values: list[float] = []
    labels_all = frame["selected_correct"].to_numpy(dtype=np.int8)
    for _ in range(replicates):
        selection = generator.integers(0, len(group_indices), size=len(group_indices))
        indices = np.concatenate([group_indices[value] for value in selection])
        labels = labels_all[indices]
        if np.unique(labels).size == 2:
            values.append(float(roc_auc_score(labels, scores[indices])))
    if not values:
        return None
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def score_summary(frame: pd.DataFrame, score: np.ndarray, *, bootstrap_replicates: int, seed: int) -> dict[str, Any]:
    labels = frame["selected_correct"].to_numpy(dtype=np.int8)
    macro, macro_tasks = task_macro_auc(frame, score)
    per_domain = {}
    for domain, group in frame.groupby("domain", sort=True):
        if group["selected_correct"].nunique() == 2:
            per_domain[str(domain)] = {
                "decisions": int(len(group)),
                "selected_accuracy": float(group["selected_correct"].mean()),
                "auc": float(roc_auc_score(group["selected_correct"], score[group.index])),
            }
    return {
        "decisions": int(len(frame)),
        "task_groups": int(frame["task_id"].nunique()),
        "selected_accuracy": float(labels.mean()),
        "oof_auc": float(roc_auc_score(labels, score)),
        "raw_brier": float(brier_score_loss(labels, score)),
        "raw_ece_15": expected_calibration_error(labels, score),
        "task_macro_auc": macro,
        "tasks_with_both_labels": macro_tasks,
        "task_cluster_bootstrap_auc_95_ci": cluster_bootstrap_auc(
            frame, score, seed=seed, replicates=bootstrap_replicates
        ),
        "per_domain": per_domain,
    }


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory already has artifacts: {path}; use --overwrite deliberately.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def self_test() -> None:
    fixture = pd.DataFrame(
        {
            "task_id": ["a"] * 6 + ["b"] * 6,
            "step": [1, 1, 1, 2, 2, 2] * 2,
            "model_alias": ["a", "b", "c"] * 4,
            "answer_normalized": ["x", "x", "y", "x", "z", "z", "q", "q", "r", "q", "r", "r"],
            "correct": [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
            "confidence": [80, 70, 20, 90, 30, 40, 60, 50, 10, 70, 40, 30],
            "parse_success": [1] * 12,
            "domain": ["synthetic"] * 12,
            "difficulty": ["synthetic"] * 12,
        }
    )
    selected, numeric_columns, categorical_columns, aliases = build_decision_frame(fixture)
    if len(selected) != 4 or aliases != ("a", "b", "c"):
        raise AssertionError("Synthetic plurality selection has the wrong barrier coverage or frozen alias order")
    first = selected.loc[(selected["task_id"] == "a") & (selected["step"] == 1)].iloc[0]
    if first["selected_model_alias"] != "a" or first["selected_correct"] != 1 or first["selected_vote_count"] != 2:
        raise AssertionError("Plurality did not choose the first frozen supporting alias")
    validate_feature_contract(numeric_columns, categorical_columns)
    all_empty = fixture.copy()
    all_empty.loc[(all_empty["task_id"] == "b") & (all_empty["step"] == 1), "answer_normalized"] = ""
    try:
        build_decision_frame(all_empty)
    except ValueError as error:
        if "all-empty" not in str(error):
            raise
    else:
        raise AssertionError("All-empty barrier must fail closed rather than disappear from the denominator")
    print("Selected-answer policy self-test passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--exclude-batch-timing",
        action="store_true",
        help="Exclude elapsed_seconds/tokens_per_second aggregates, which are historical batch timing rather than event timing.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.n_splits < 2 or args.jobs < 1:
        raise ValueError("--n-splits must be >=2 and --jobs must be positive")
    frame, input_files = load_panel(args.input_dir)
    decisions, numeric_columns, categorical_columns, aliases = build_decision_frame(
        frame, exclude_batch_timing=args.exclude_batch_timing
    )
    if decisions["task_id"].nunique() < args.n_splits:
        raise ValueError("Not enough task groups for requested GroupKFold splits")
    baseline_scores = decisions["selected_vote_margin_fraction"].to_numpy(dtype=np.float64)
    baseline = score_summary(
        decisions, baseline_scores, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "rows": int(len(frame)),
                    "decisions": int(len(decisions)),
                    "task_groups": int(decisions["task_id"].nunique()),
                    "frozen_alias_order": aliases,
                    "numeric_feature_count": len(numeric_columns),
                    "categorical_feature_count": len(categorical_columns),
                    "parameter_free_vote_margin": baseline,
                },
                indent=2,
            )
        )
        return 0
    prepare_output_dir(args.output_dir, args.overwrite)
    started = time.time()
    oof, fold_ids, fold_auc = fit_oof(
        decisions,
        numeric_columns,
        categorical_columns,
        n_splits=args.n_splits,
        seed=args.seed,
        jobs=args.jobs,
        config=LightGBMConfig(),
    )
    oof_metrics = score_summary(
        decisions, oof, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed
    )
    calibrated = cross_fold_platt(decisions["selected_correct"].to_numpy(dtype=np.int8), oof, fold_ids)
    oof_metrics["fold_auc"] = fold_auc
    oof_metrics["cross_fold_platt_brier"] = float(
        brier_score_loss(decisions["selected_correct"], calibrated)
    )
    oof_metrics["cross_fold_platt_ece_15"] = expected_calibration_error(
        decisions["selected_correct"].to_numpy(dtype=np.int8), calibrated
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "script_sha256": raw_file_hash(Path(__file__).resolve()),
        "input_dir": str(args.input_dir),
        "input_files": input_files,
        "candidate_rows": int(len(frame)),
        "decision_rows": int(len(decisions)),
        "task_groups": int(decisions["task_id"].nunique()),
        "frozen_alias_order": list(aliases),
        "outer_group": "task_id",
        "selection_rule": "nonempty normalized-answer plurality; tie=first frozen alias",
        "strict_contract": {
            "numeric_features": numeric_columns,
            "categorical_features": categorical_columns,
            "forbidden": sorted(FORBIDDEN_FEATURES),
            "answer_handling": "answer equality only for deterministic plurality/count/margin; no raw answer feature",
            "k2": "excluded; separately-costed post-query generation",
            "batch_timing": (
                "excluded by --exclude-batch-timing"
                if args.exclude_batch_timing
                else "included retrospectively; future confirmation must record per-event timing"
            ),
            "historical_limit": "same-step peer availability is not timestamped in historical traces",
        },
        "lightgbm": asdict(LightGBMConfig()),
        "n_splits": args.n_splits,
        "seed": args.seed,
    }
    atomic_write_json(args.output_dir / "selected_answer_oof_manifest.json", manifest)
    atomic_write_json(
        args.output_dir / "selected_answer_oof_metrics.json",
        {"parameter_free_vote_margin": baseline, "selected_answer_lgbm": oof_metrics},
    )
    predictions = decisions[
        [
            "task_id",
            "step",
            "domain",
            "selected_model_alias",
            "selected_answer_hash",
            "selected_correct",
            "selected_support_fraction",
            "selected_vote_margin_fraction",
        ]
    ].copy()
    predictions["fold"] = fold_ids
    predictions["selected_answer_lgbm_oof"] = oof
    predictions["selected_answer_lgbm_platt"] = calibrated
    temporary = args.output_dir / "selected_answer_oof_predictions.csv.tmp"
    predictions.to_csv(temporary, index=False)
    os.replace(temporary, args.output_dir / "selected_answer_oof_predictions.csv")
    report = "\n".join(
        [
            "=" * 104,
            "DECISION-LEVEL SELECTED-ANSWER OOF EXPERIMENT",
            "=" * 104,
            f"Candidate rows: {len(frame)} | decision barriers: {len(decisions)} | task groups: {decisions['task_id'].nunique()}",
            "Policy: deterministic nonempty-answer plurality; ties by frozen alias order.",
            "Validation: GroupKFold(task_id). Answer text is used only transiently to form plurality/count features.",
            "",
            "Configuration                 | OOF AUC | Selected accuracy | Task macro AUC | Task bootstrap 95% CI | ECE15",
            "-" * 104,
            (
                f"Parameter-free vote margin    | {baseline['oof_auc']:.6f} | {baseline['selected_accuracy']:.6f} | "
                f"{baseline['task_macro_auc']:.6f} | {baseline['task_cluster_bootstrap_auc_95_ci']} | {baseline['raw_ece_15']:.6f}"
            ),
            (
                f"Selected-answer LGBM OOF      | {oof_metrics['oof_auc']:.6f} | {oof_metrics['selected_accuracy']:.6f} | "
                f"{oof_metrics['task_macro_auc']:.6f} | {oof_metrics['task_cluster_bootstrap_auc_95_ci']} | {oof_metrics['raw_ece_15']:.6f}"
            ),
            "",
            "This is a retrospective decision-level result. Historical artifacts do not prove synchronized peer availability.",
            "",
        ]
    )
    atomic_write_text(args.output_dir / "selected_answer_oof_results.log", report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
