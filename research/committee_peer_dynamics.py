"""Causal, anonymous peer-panel features for candidate-level stopping models.

The input has one current response per ``(task_id, step, model_alias)``.  Raw
normalized answers are used only transiently to form equality/count aggregates;
no answer string, model identity, task identity, label, or text becomes a
returned feature.  Every peer statistic excludes the scored response itself.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1.0e-12
KEYS = ["task_id", "step"]

# Public names deliberately avoid source-field names such as ``answer_*`` and
# ``raw_*``. The values are current-barrier response telemetry only.
PEER_SOURCE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("confidence", "confidence"),
    ("mean_token_logprob", "token_logprob"),
    ("entropy_mean", "entropy"),
    ("answer_span_mean_logprob", "span_logprob"),
    ("answer_span_min_logprob", "span_min_logprob"),
    ("answer_span_mean_entropy", "span_entropy"),
    ("thought_token_count", "reasoning_tokens"),
    ("raw_generation_tokens", "generation_tokens"),
    ("parse_success", "parse_success"),
    ("model_stop_flag", "self_stop_flag"),
)

# These 20 fields use only anonymous equality/count topology at the current
# closed panel plus prior-closed-panel equality topology. They intentionally do
# not consume peer confidence, token probability, length, or model identity.
TOPOLOGY_FEATURE_COLUMNS: tuple[str, ...] = (
    "peer_valid_response_count",
    "peer_valid_response_fraction",
    "peer_unique_response_count",
    "peer_support_count",
    "peer_support_fraction",
    "peer_top_support_fraction",
    "peer_support_margin_count",
    "peer_matches_top_response",
    "peer_response_entropy",
    "panel_top_support_fraction",
    "panel_top_margin_fraction",
    "panel_response_entropy",
    "panel_top_tied",
    "has_previous_closed_panel",
    "current_support_in_previous_panel_fraction",
    "current_matches_previous_panel_top",
    "delta_panel_top_support_fraction",
    "delta_panel_response_entropy",
    "delta_peer_support_fraction",
    "peer_support_fraction_increased",
)

_FORBIDDEN_FEATURE_TOKENS = (
    "answer",
    "correct",
    "label",
    "task",
    "alias",
    "model",
    "text",
    "raw",
    "key",
    "expected",
    "gold",
)


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _normalized_key(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip().str.casefold()


def _join_group_stat(
    frame: pd.DataFrame,
    values: pd.Series,
    *,
    prefix: str,
) -> pd.DataFrame:
    """Leave the target row out of mean/std for an all-peer population."""

    valid = values.notna().astype(np.int8)
    filled = values.fillna(0.0)
    group_count = valid.groupby([frame["task_id"], frame["step"]], sort=False).transform("sum").astype(np.float64)
    group_sum = filled.groupby([frame["task_id"], frame["step"]], sort=False).transform("sum")
    group_sum_sq = (filled * filled).groupby([frame["task_id"], frame["step"]], sort=False).transform("sum")
    peer_count = group_count - valid
    peer_sum = group_sum - filled * valid
    peer_mean = peer_sum / peer_count.where(peer_count > 0)
    peer_second_moment = (group_sum_sq - filled * filled * valid) / peer_count.where(peer_count > 0)
    peer_variance = (peer_second_moment - peer_mean * peer_mean).clip(lower=0.0)
    return pd.DataFrame(
        {
            f"peer_all_{prefix}_mean": peer_mean.astype(np.float32),
            f"peer_all_{prefix}_std": np.sqrt(peer_variance).astype(np.float32),
            f"peer_all_{prefix}_available": (peer_count > 0).astype(np.int8),
        },
        index=frame.index,
    )


def _support_telemetry(
    work: pd.DataFrame,
    values: pd.Series,
    *,
    prefix: str,
) -> pd.DataFrame:
    """Leave-target-out support/opposition telemetry conditional on equality."""

    valid_numeric = values.notna()
    support_population = work.loc[work["_peer_nonempty"].eq(1) & valid_numeric].copy()
    support_population["_peer_value"] = values.loc[support_population.index].astype(np.float64)
    total = support_population.groupby(KEYS, sort=False)["_peer_value"].agg(["sum", "count"])
    support = support_population.groupby(KEYS + ["_peer_key"], sort=False)["_peer_value"].agg(["sum", "count"])

    result = work[KEYS + ["_peer_key", "_peer_nonempty"]].copy()
    result = result.join(total.rename(columns={"sum": "_total_sum", "count": "_total_count"}), on=KEYS)
    result = result.join(
        support.rename(columns={"sum": "_support_sum", "count": "_support_count"}),
        on=KEYS + ["_peer_key"],
    )
    current = values.fillna(0.0).astype(np.float64)
    self_support = (work["_peer_nonempty"].eq(1) & valid_numeric).astype(np.int8)
    support_sum = result["_support_sum"].fillna(0.0) - current * self_support
    support_count = result["_support_count"].fillna(0.0) - self_support
    total_sum = result["_total_sum"].fillna(0.0) - current * self_support
    total_count = result["_total_count"].fillna(0.0) - self_support
    opposition_sum = total_sum - support_sum
    opposition_count = total_count - support_count
    support_mean = support_sum / support_count.where(support_count > 0)
    opposition_mean = opposition_sum / opposition_count.where(opposition_count > 0)
    return pd.DataFrame(
        {
            f"peer_support_{prefix}_sum": support_sum.astype(np.float32),
            f"peer_support_{prefix}_mean": support_mean.astype(np.float32),
            f"peer_opposition_{prefix}_mean": opposition_mean.astype(np.float32),
            f"peer_support_minus_opposition_{prefix}": (support_mean - opposition_mean).astype(np.float32),
            f"peer_support_{prefix}_available": (support_count > 0).astype(np.int8),
            f"peer_opposition_{prefix}_available": (opposition_count > 0).astype(np.int8),
        },
        index=work.index,
    )


def _distribution_features(work: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build equality-only current and prior-panel aggregates.

    ``summary`` retains transient normalized keys for the strictly internal
    previous-barrier lookup. The returned feature table never exposes them.
    """

    valid = work.loc[work["_peer_nonempty"].eq(1), KEYS + ["_peer_key"]].copy()
    counts = valid.groupby(KEYS + ["_peer_key"], sort=False).size().rename("_count").reset_index()
    if counts.empty:
        raise ValueError("At least one nonempty normalized response is required per panel.")
    counts["_top_count"] = counts.groupby(KEYS, sort=False)["_count"].transform("max")
    counts["_is_top"] = counts["_count"].eq(counts["_top_count"])
    counts["_count_log_count"] = counts["_count"].astype(np.float64) * np.log(counts["_count"].astype(np.float64))
    ranked = counts.sort_values(KEYS + ["_count", "_peer_key"], ascending=[True, True, False, True], kind="stable").copy()
    ranked["_rank"] = ranked.groupby(KEYS, sort=False).cumcount()
    summary = counts.groupby(KEYS, sort=False).agg(
        _valid_count=("_count", "sum"),
        _unique_count=("_count", "size"),
        _top_count=("_top_count", "first"),
        _top_tie_count=("_is_top", "sum"),
        _count_log_count=("_count_log_count", "sum"),
    )
    second = ranked.loc[ranked["_rank"].eq(1), KEYS + ["_count"]].rename(columns={"_count": "_second_count"})
    summary = summary.join(second.set_index(KEYS), on=KEYS)
    summary["_second_count"] = summary["_second_count"].fillna(0.0)
    summary["_top_fraction"] = summary["_top_count"] / summary["_valid_count"].clip(lower=1)
    summary["_entropy"] = np.log(summary["_valid_count"].clip(lower=1)) - summary["_count_log_count"] / summary["_valid_count"].clip(lower=1)

    result = work[KEYS + ["_peer_key", "_peer_nonempty"]].copy()
    result = result.join(summary, on=KEYS)
    own_count = counts.set_index(KEYS + ["_peer_key"])["_count"]
    result = result.join(own_count.rename("_own_count"), on=KEYS + ["_peer_key"])
    result["_own_count"] = result["_own_count"].fillna(0.0)
    candidate_nonempty = result["_peer_nonempty"].astype(np.float64)
    peer_valid = result["_valid_count"].fillna(0.0) - candidate_nonempty
    peer_support = result["_own_count"] - candidate_nonempty
    unique_top = result["_own_count"].eq(result["_top_count"]) & result["_top_tie_count"].eq(1)
    other_top = np.where(unique_top, result["_second_count"], result["_top_count"])
    other_top = pd.Series(other_top, index=result.index, dtype=np.float64).fillna(0.0)
    peer_top = np.maximum(peer_support.to_numpy(dtype=np.float64), other_top.to_numpy(dtype=np.float64))
    removed_term = np.where(
        candidate_nonempty.to_numpy(dtype=bool),
        result["_own_count"].to_numpy(dtype=np.float64) * np.log(result["_own_count"].clip(lower=1).to_numpy(dtype=np.float64)),
        0.0,
    )
    remaining_entropy = np.where(
        peer_valid.to_numpy(dtype=np.float64) > 0,
        np.log(np.maximum(peer_valid.to_numpy(dtype=np.float64), 1.0))
        - (result["_count_log_count"].fillna(0.0).to_numpy(dtype=np.float64) - removed_term)
        / np.maximum(peer_valid.to_numpy(dtype=np.float64), 1.0),
        0.0,
    )
    feature = pd.DataFrame(
        {
            "peer_valid_response_count": peer_valid.astype(np.float32),
            "peer_valid_response_fraction": (peer_valid / (work.groupby(KEYS, sort=False)["model_alias"].transform("size").astype(np.float64) - 1.0).clip(lower=1.0)).astype(np.float32),
            "peer_unique_response_count": (result["_unique_count"].fillna(0.0) - ((candidate_nonempty.eq(1)) & result["_own_count"].eq(1)).astype(np.float64)).astype(np.float32),
            "peer_support_count": peer_support.astype(np.float32),
            "peer_support_fraction": (peer_support / peer_valid.clip(lower=1.0)).astype(np.float32),
            "peer_top_support_fraction": (peer_top / np.maximum(peer_valid.to_numpy(dtype=np.float64), 1.0)).astype(np.float32),
            "peer_support_margin_count": (peer_support - other_top).astype(np.float32),
            "peer_matches_top_response": ((candidate_nonempty.eq(1)) & peer_support.eq(pd.Series(peer_top, index=result.index))).astype(np.int8),
            "peer_response_entropy": remaining_entropy.astype(np.float32),
            "panel_top_support_fraction": result["_top_fraction"].fillna(0.0).astype(np.float32),
            "panel_top_margin_fraction": ((result["_top_count"].fillna(0.0) - result["_second_count"].fillna(0.0)) / result["_valid_count"].fillna(0.0).clip(lower=1.0)).astype(np.float32),
            "panel_response_entropy": result["_entropy"].fillna(0.0).astype(np.float32),
            "panel_top_tied": (result["_top_tie_count"].fillna(0.0) > 1.0).astype(np.int8),
        },
        index=work.index,
    )

    # The prior answer support lookup is equality-only and uses only the
    # previously closed barrier. A candidate is top if it belongs to *any*
    # prior tie, avoiding a lexical/semantic answer tie-break.
    prior_summary = summary.reset_index()[["task_id", "step", "_valid_count", "_top_count", "_top_fraction", "_entropy"]].copy()
    prior_summary["step"] = prior_summary["step"].astype(int) + 1
    prior_summary = prior_summary.rename(
        columns={
            "_valid_count": "_prior_valid_count",
            "_top_count": "_prior_top_count",
            "_top_fraction": "_prior_top_fraction",
            "_entropy": "_prior_entropy",
        }
    )
    prior_counts = counts[KEYS + ["_peer_key", "_count"]].copy()
    prior_counts["step"] = prior_counts["step"].astype(int) + 1
    prior_counts = prior_counts.rename(columns={"_count": "_prior_current_count"})
    temporal = work[KEYS + ["_peer_key", "_peer_nonempty"]].copy()
    temporal = temporal.join(prior_summary.set_index(KEYS), on=KEYS)
    temporal = temporal.join(prior_counts.set_index(KEYS + ["_peer_key"]), on=KEYS + ["_peer_key"])
    has_prior = temporal["_prior_valid_count"].notna()
    prior_valid = temporal["_prior_valid_count"].fillna(0.0)
    prior_current = temporal["_prior_current_count"].fillna(0.0)
    feature["has_previous_closed_panel"] = has_prior.astype(np.int8)
    feature["current_support_in_previous_panel_fraction"] = (prior_current / prior_valid.clip(lower=1.0)).where(has_prior).astype(np.float32)
    feature["current_matches_previous_panel_top"] = (
        has_prior & temporal["_peer_nonempty"].eq(1) & prior_current.eq(temporal["_prior_top_count"].fillna(-1.0))
    ).astype(np.int8)
    feature["delta_panel_top_support_fraction"] = (
        feature["panel_top_support_fraction"] - temporal["_prior_top_fraction"].fillna(feature["panel_top_support_fraction"])
    ).astype(np.float32)
    feature["delta_panel_response_entropy"] = (
        feature["panel_response_entropy"] - temporal["_prior_entropy"].fillna(feature["panel_response_entropy"])
    ).astype(np.float32)
    return feature, summary


def _assert_feature_contract(columns: Iterable[str]) -> None:
    invalid = [
        column
        for column in columns
        if any(token in column.casefold() for token in _FORBIDDEN_FEATURE_TOKENS)
    ]
    if invalid:
        raise AssertionError(f"Peer feature names expose a forbidden runtime field: {invalid}")


def build_peer_dynamics_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return the frame plus anonymous same-barrier/prefix peer features."""

    required = {"task_id", "step", "model_alias", "trajectory_id", "answer_normalized"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Peer dynamics input is missing: {missing}")
    if frame.duplicated(["task_id", "step", "model_alias"], keep=False).any():
        raise ValueError("Peer dynamics requires one current response per task/step/model alias.")
    work = frame.copy()
    work["_peer_key"] = _normalized_key(work["answer_normalized"])
    work["_peer_nonempty"] = (work["_peer_key"] != "").astype(np.int8)

    feature_blocks: list[pd.DataFrame] = []
    distribution, _ = _distribution_features(work)
    feature_blocks.append(distribution)
    for source, safe_name in PEER_SOURCE_COLUMNS:
        if source not in work.columns:
            continue
        values = _numeric(work[source])
        feature_blocks.append(_join_group_stat(work, values, prefix=safe_name))
        feature_blocks.append(_support_telemetry(work, values, prefix=safe_name))
    result = work.drop(columns=["_peer_key", "_peer_nonempty"])
    features = pd.concat(feature_blocks, axis=1)
    if features.columns.duplicated().any():
        duplicated = features.columns[features.columns.duplicated()].tolist()
        raise AssertionError(f"Peer feature construction emitted duplicates: {duplicated}")
    feature_columns = features.columns.tolist()
    _assert_feature_contract(feature_columns)
    result = result.join(features)

    # Per-trajectory support dynamics are causal because each current support
    # fraction is computed from the barrier that has just closed.
    result = result.sort_values(["trajectory_id", "step"], kind="stable")
    previous_step = result.groupby("trajectory_id", sort=False)["step"].shift(1)
    contiguous = previous_step.eq(result["step"] - 1)
    prior_support = result.groupby("trajectory_id", sort=False)["peer_support_fraction"].shift(1)
    result["delta_peer_support_fraction"] = (
        result["peer_support_fraction"] - prior_support.where(contiguous)
    ).fillna(0.0).astype(np.float32)
    result["peer_support_fraction_increased"] = (
        contiguous & result["peer_support_fraction"].gt(prior_support)
    ).astype(np.int8)
    feature_columns += ["delta_peer_support_fraction", "peer_support_fraction_increased"]
    missing_topology = sorted(set(TOPOLOGY_FEATURE_COLUMNS) - set(feature_columns))
    if missing_topology:
        raise AssertionError(f"Peer topology contract is incomplete: {missing_topology}")
    _assert_feature_contract(feature_columns)
    return result.sort_index(kind="stable"), feature_columns


def peer_dynamics_self_test() -> None:
    fixture = pd.DataFrame(
        {
            "task_id": ["t"] * 6,
            "step": [1, 1, 1, 2, 2, 2],
            "model_alias": ["a", "b", "c", "a", "b", "c"],
            "trajectory_id": ["a", "b", "c", "a", "b", "c"],
            "answer_normalized": ["x", "x", "y", "x", "z", "x"],
            "confidence": [90.0, 80.0, 10.0, 90.0, 20.0, 70.0],
            "mean_token_logprob": [-0.2, -0.3, -1.1, -0.2, -0.9, -0.4],
            "entropy_mean": [0.1, 0.2, 0.9, 0.2, 0.8, 0.3],
        }
    )
    output, columns = build_peer_dynamics_features(fixture)
    first = output.loc[(output["step"] == 1) & (output["model_alias"] == "a")].iloc[0]
    if not np.isclose(float(first["peer_support_count"]), 1.0):
        raise AssertionError("Scored response was not excluded from its peer support count.")
    if not np.isclose(float(first["peer_support_fraction"]), 0.5):
        raise AssertionError("Peer support fraction is incorrect for the synthetic panel.")
    if not np.isclose(float(first["peer_support_confidence_mean"]), 80.0):
        raise AssertionError("Support telemetry must exclude the scored response and retain the peer value.")
    second = output.loc[(output["step"] == 2) & (output["model_alias"] == "a")].iloc[0]
    if int(second["current_matches_previous_panel_top"]) != 1:
        raise AssertionError("Current response should match the previous panel's tied-or-top response.")
    _assert_feature_contract(columns)
    print("Peer dynamics self-test passed.")
