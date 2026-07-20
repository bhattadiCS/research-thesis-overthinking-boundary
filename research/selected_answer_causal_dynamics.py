"""Leakage-safe hidden-state kinematics for selected-answer barriers.

The two stored hidden projections are model-specific coordinate systems.  This
module therefore computes motion only *within* each `(task_id, model_alias)`
trajectory, where differences and angles are meaningful, and subsequently
aggregates scalar invariants anonymously at a current completed panel barrier.
It never emits aliases, answer values, task IDs, labels, text, or future-step
values as learner features.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


EPS = 1.0e-12
DYNAMICS_PREFIX = "dyn_"


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > EPS else float("nan")


def _finite_mean_std(values: np.ndarray) -> tuple[float, float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan"), 0.0
    return float(np.mean(finite)), float(np.std(finite)), float(len(finite) / len(values))


def _scalar_sources() -> tuple[str, ...]:
    layers = ("l1", "l2")
    values: list[str] = []
    for layer in layers:
        values.extend(
            [
                f"{layer}_velocity_norm",
                f"{layer}_relative_velocity",
                f"{layer}_previous_cosine",
                f"{layer}_log_norm_delta",
                f"{layer}_acceleration_norm",
                f"{layer}_relative_acceleration",
                f"{layer}_turn_cosine",
            ]
        )
    values.extend(
        [
            "cross_layer_velocity_cosine",
            "cross_layer_log_velocity_ratio",
            "answer_same_previous",
            "supports_previous_selected",
        ]
    )
    return tuple(values)


def build_causal_dynamics(
    raw: pd.DataFrame,
    decisions: pd.DataFrame,
    reference: Any,
    representation: Any,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Return current-barrier scalar kinematics plus diagnostics.

    Previous states are used only if their step number is exactly `t-1` (and
    `t-2` for acceleration).  Missing history remains missing, rather than
    being backfilled or bridged across a gap.  Answer values exist only as
    temporary equality relations needed for selected-supporter aggregation.
    """

    required = {"task_id", "step", "model_alias", "answer_normalized", "mid_hidden_1_proj", "mid_hidden_2_proj"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Raw trace panel lacks required dynamics columns: {missing}")
    keys = ["task_id", "step"]
    work = raw.copy().reset_index(drop=True)
    if work.duplicated(["task_id", "step", "model_alias"]).any():
        raise ValueError("Dynamics require exactly one response per task/step/model trajectory.")
    work["_answer_key"] = work["answer_normalized"].map(reference.normalized_answer_key)
    selection = decisions[keys + ["selected_answer_key"]].copy()
    work = work.merge(selection, on=keys, how="left", validate="many_to_one")
    if work["selected_answer_key"].isna().any():
        raise AssertionError("Could not align raw responses with deterministic selected barriers.")
    work["_winner"] = (work["_answer_key"] != "") & (work["_answer_key"] == work["selected_answer_key"])
    previous_selection = selection.rename(columns={"selected_answer_key": "_previous_selected_answer"}).copy()
    previous_selection["step"] = previous_selection["step"].astype(int) + 1
    work = work.merge(previous_selection, on=keys, how="left", validate="many_to_one")

    layer_one, layer_one_available = representation.parse_projection(work["mid_hidden_1_proj"])
    layer_two, layer_two_available = representation.parse_projection(work["mid_hidden_2_proj"])
    count = len(work)
    sources = _scalar_sources()
    candidate = {source: np.full(count, np.nan, dtype=np.float64) for source in sources}
    prior_available = np.zeros(count, dtype=np.float64)
    acceleration_available = np.zeros(count, dtype=np.float64)
    answer_key = work["_answer_key"].to_numpy(dtype=object)
    previous_selected = work["_previous_selected_answer"].to_numpy(dtype=object)
    steps = work["step"].to_numpy(dtype=np.int64)

    def fill_layer(
        prefix: str,
        current_index: int,
        previous_index: int,
        earlier_index: int | None,
        values: np.ndarray,
        available: np.ndarray,
    ) -> np.ndarray | None:
        if not (available[current_index] and available[previous_index]):
            return None
        current = values[current_index]
        previous = values[previous_index]
        velocity = current - previous
        velocity_norm = float(np.linalg.norm(velocity))
        previous_norm = float(np.linalg.norm(previous))
        current_norm = float(np.linalg.norm(current))
        candidate[f"{prefix}_velocity_norm"][current_index] = velocity_norm
        candidate[f"{prefix}_relative_velocity"][current_index] = velocity_norm / max(previous_norm, EPS)
        candidate[f"{prefix}_previous_cosine"][current_index] = _cosine(current, previous)
        candidate[f"{prefix}_log_norm_delta"][current_index] = math.log(max(current_norm, EPS)) - math.log(max(previous_norm, EPS))
        if earlier_index is not None and available[earlier_index]:
            prior_velocity = previous - values[earlier_index]
            acceleration = velocity - prior_velocity
            acceleration_norm = float(np.linalg.norm(acceleration))
            candidate[f"{prefix}_acceleration_norm"][current_index] = acceleration_norm
            candidate[f"{prefix}_relative_acceleration"][current_index] = acceleration_norm / max(velocity_norm, EPS)
            candidate[f"{prefix}_turn_cosine"][current_index] = _cosine(velocity, prior_velocity)
        return velocity

    contiguous_links = 0
    contiguous_accelerations = 0
    for _trajectory, group in work.groupby(["task_id", "model_alias"], sort=False):
        positions = group.sort_values("step", kind="stable").index.to_numpy(dtype=np.int64)
        local_steps = steps[positions]
        for position in range(1, len(positions)):
            current_index = int(positions[position])
            previous_index = int(positions[position - 1])
            if local_steps[position] != local_steps[position - 1] + 1:
                continue
            prior_available[current_index] = 1.0
            contiguous_links += 1
            earlier_index: int | None = None
            if position >= 2 and local_steps[position - 1] == local_steps[position - 2] + 1:
                earlier_index = int(positions[position - 2])
                acceleration_available[current_index] = 1.0
                contiguous_accelerations += 1
            velocity_one = fill_layer(
                "l1", current_index, previous_index, earlier_index, layer_one, layer_one_available
            )
            velocity_two = fill_layer(
                "l2", current_index, previous_index, earlier_index, layer_two, layer_two_available
            )
            if velocity_one is not None and velocity_two is not None:
                candidate["cross_layer_velocity_cosine"][current_index] = _cosine(velocity_one, velocity_two)
                candidate["cross_layer_log_velocity_ratio"][current_index] = math.log(
                    max(float(np.linalg.norm(velocity_two)), EPS) / max(float(np.linalg.norm(velocity_one)), EPS)
                )
            if answer_key[current_index] != "":
                candidate["answer_same_previous"][current_index] = float(answer_key[current_index] == answer_key[previous_index])
                prior_selected = previous_selected[current_index]
                candidate["supports_previous_selected"][current_index] = float(
                    isinstance(prior_selected, str) and prior_selected != "" and answer_key[current_index] == prior_selected
                )

    candidate_frame = pd.DataFrame(candidate)
    work = pd.concat([work, candidate_frame], axis=1)
    rows: list[dict[str, Any]] = []
    for (task_id, step), group in work.groupby(keys, sort=False):
        indices = group.index.to_numpy(dtype=np.int64)
        winner = group["_winner"].to_numpy(dtype=bool)
        nonwinner = ~winner
        row: dict[str, Any] = {
            "task_id": str(task_id),
            "step": int(step),
            "dyn_panel_contiguous_previous_fraction": float(prior_available[indices].mean()),
            "dyn_panel_contiguous_acceleration_fraction": float(acceleration_available[indices].mean()),
            "dyn_winner_contiguous_previous_fraction": float(prior_available[indices][winner].mean()) if winner.any() else 0.0,
            "dyn_winner_contiguous_acceleration_fraction": float(acceleration_available[indices][winner].mean()) if winner.any() else 0.0,
        }
        for source in sources:
            values = candidate[source][indices]
            for population, mask in (("panel", np.ones(len(indices), dtype=bool)), ("winner", winner)):
                mean, std, fraction = _finite_mean_std(values[mask])
                prefix = f"dyn_{population}_{source}"
                row[f"{prefix}_mean"] = mean
                row[f"{prefix}_std"] = std
                row[f"{prefix}_available_fraction"] = fraction
            winner_mean, _winner_std, winner_fraction = _finite_mean_std(values[winner])
            nonwinner_mean, _nonwinner_std, nonwinner_fraction = _finite_mean_std(values[nonwinner])
            row[f"dyn_winner_vs_nonwinner_{source}_mean_delta"] = (
                winner_mean - nonwinner_mean if winner_fraction and nonwinner_fraction else float("nan")
            )
        rows.append(row)
    dynamics = pd.DataFrame(rows)
    result = decisions.merge(dynamics, on=keys, how="left", validate="one_to_one", sort=False)
    feature_columns = sorted(column for column in result.columns if column.startswith(DYNAMICS_PREFIX))
    if len(result) != len(decisions) or not feature_columns:
        raise AssertionError("Could not build decision-level causal dynamics features.")
    # Features must be anonymous scalars.  This guards against a future edit
    # accidentally returning one of the temporary answer/identity fields.
    forbidden_tokens = ("answer_key", "model_alias", "task_id", "correct", "expected", "raw", "text", "run_id")
    unsafe = [column for column in feature_columns if any(token in column for token in forbidden_tokens)]
    if unsafe:
        raise AssertionError(f"Unsafe dynamics feature names: {unsafe}")
    diagnostics = {
        "candidate_rows": count,
        "task_model_trajectories": int(work.groupby(["task_id", "model_alias"], sort=False).ngroups),
        "contiguous_link_rows": int(contiguous_links),
        "contiguous_acceleration_rows": int(contiguous_accelerations),
        "link_coverage": float(prior_available.mean()),
        "acceleration_coverage": float(acceleration_available.mean()),
        "layer_one_available_fraction": float(layer_one_available.mean()),
        "layer_two_available_fraction": float(layer_two_available.mean()),
        "feature_count": len(feature_columns),
        "scalar_sources": list(sources),
    }
    return result, feature_columns, diagnostics
