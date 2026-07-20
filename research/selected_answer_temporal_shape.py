"""Coordinate-safe, prefix-only temporal-shape features for committee barriers.

The stored projections have model-specific coordinate systems.  This module
therefore never compares raw coordinates across models.  It derives scalar
trajectory shape values only inside one ``(task_id, model_alias, layer)``
history, using states through the current step, then leaves anonymous
panel/winner aggregation to the outer-fold evaluator.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


EPS = 1.0e-12
# ``state`` distinguishes private per-response intermediate quantities from
# the final anonymous ``shape_*`` learner features without implying that raw
# coordinates are ever emitted.
SHAPE_PREFIX = "shape_state_"


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > EPS else float("nan")


def shape_state_sources() -> tuple[str, ...]:
    """Names of model-local scalar quantities, before fold-local scaling."""
    values: list[str] = []
    for layer in ("l1", "l2"):
        values.extend(
            [
                f"{SHAPE_PREFIX}{layer}_log_speed",
                f"{SHAPE_PREFIX}{layer}_log_acceleration",
                f"{SHAPE_PREFIX}{layer}_speed_share",
                f"{SHAPE_PREFIX}{layer}_path_efficiency",
                f"{SHAPE_PREFIX}{layer}_turn_cosine",
                f"{SHAPE_PREFIX}{layer}_mean_turn_prefix",
                f"{SHAPE_PREFIX}{layer}_relative_jerk",
                f"{SHAPE_PREFIX}{layer}_nonadjacent_recurrence",
                f"{SHAPE_PREFIX}{layer}_velocity_log_slope",
                f"{SHAPE_PREFIX}{layer}_net_alignment",
            ]
        )
    return tuple(values)


def build_response_shape_rows(
    raw: pd.DataFrame,
    decisions: pd.DataFrame,
    reference: Any,
    representation: Any,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Build per-response prefix-only shape values without raw coordinates.

    ``model_alias`` exists only in this intermediate table so an outer fold can
    fit a separate robust normalizer for each model.  It is never emitted into
    an anonymous barrier-level learner matrix.
    """
    required = {
        "task_id",
        "step",
        "model_alias",
        "answer_normalized",
        "mid_hidden_1_proj",
        "mid_hidden_2_proj",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Raw trace panel lacks temporal-shape columns: {missing}")
    keys = ["task_id", "step"]
    work = raw[list(required)].copy().reset_index(drop=True)
    if work.duplicated(["task_id", "step", "model_alias"]).any():
        raise ValueError("Temporal shape requires exactly one response per task/step/model trajectory.")
    work["task_id"] = work["task_id"].astype(str)
    work["model_alias"] = work["model_alias"].astype(str)
    work["step"] = pd.to_numeric(work["step"], errors="raise").astype(np.int16)
    work["_answer_key"] = work["answer_normalized"].map(reference.normalized_answer_key)
    selection = decisions[keys + ["selected_answer_key"]].copy()
    work = work.merge(selection, on=keys, how="left", validate="many_to_one")
    if work["selected_answer_key"].isna().any():
        raise AssertionError("Could not align every temporal-shape row to a deterministic decision barrier.")
    work["_winner"] = (work["_answer_key"] != "") & (work["_answer_key"] == work["selected_answer_key"])

    layer_one, one_available = representation.parse_projection(work["mid_hidden_1_proj"])
    layer_two, two_available = representation.parse_projection(work["mid_hidden_2_proj"])
    sources = shape_state_sources()
    values = {source: np.full(len(work), np.nan, dtype=np.float64) for source in sources}
    history_links = 0
    history_turns = 0
    prior_pairs = 0

    def fill_layer(
        layer: str,
        indices: np.ndarray,
        matrix: np.ndarray,
        available: np.ndarray,
    ) -> None:
        nonlocal history_links, history_turns, prior_pairs
        prefix = f"{SHAPE_PREFIX}{layer}_"
        cumulative = 0.0
        previous_velocity: np.ndarray | None = None
        previous_state: np.ndarray | None = None
        start_state: np.ndarray | None = None
        prior_states: list[np.ndarray] = []
        prior_steps: list[int] = []
        speed_history: list[float] = []
        turn_history: list[float] = []
        for index in indices:
            position = int(index)
            step = int(work.at[position, "step"])
            if not available[position]:
                # Do not bridge a missing projected state when deriving prefix
                # geometry.  A later state starts a new observable segment.
                previous_state = None
                previous_velocity = None
                start_state = None
                cumulative = 0.0
                prior_states = []
                prior_steps = []
                speed_history = []
                turn_history = []
                continue
            current = matrix[position]
            if start_state is None:
                start_state = current
            contiguous_previous = (
                previous_state is not None and prior_steps and step == prior_steps[-1] + 1
            )
            if contiguous_previous:
                velocity = current - previous_state
                velocity_norm = float(np.linalg.norm(velocity))
                cumulative += velocity_norm
                speed_history.append(math.log(max(velocity_norm, EPS)))
                values[f"{prefix}log_speed"][position] = speed_history[-1]
                if len(speed_history) >= 2 and cumulative > EPS:
                    values[f"{prefix}speed_share"][position] = velocity_norm / cumulative
                    values[f"{prefix}path_efficiency"][position] = float(np.linalg.norm(current - start_state)) / cumulative
                    values[f"{prefix}net_alignment"][position] = _cosine(velocity, current - start_state)
                if len(prior_states) >= 3 and cumulative > EPS:
                    nonadjacent = np.asarray(
                        [np.linalg.norm(current - prior) for prior in prior_states[:-1]], dtype=np.float64
                    )
                    values[f"{prefix}nonadjacent_recurrence"][position] = float(nonadjacent.min()) / cumulative
                    prior_pairs += len(nonadjacent)
                history_links += 1
                if previous_velocity is not None:
                    jerk = velocity - previous_velocity
                    turn = _cosine(velocity, previous_velocity)
                    values[f"{prefix}turn_cosine"][position] = turn
                    if math.isfinite(turn):
                        turn_history.append(turn)
                        values[f"{prefix}mean_turn_prefix"][position] = float(np.mean(turn_history))
                    jerk_norm = float(np.linalg.norm(jerk))
                    values[f"{prefix}log_acceleration"][position] = math.log(max(jerk_norm, EPS))
                    values[f"{prefix}relative_jerk"][position] = jerk_norm / max(velocity_norm, EPS)
                    history_turns += 1
                if len(speed_history) >= 2:
                    x = np.arange(len(speed_history), dtype=np.float64)
                    centered = x - x.mean()
                    values[f"{prefix}velocity_log_slope"][position] = float(
                        np.dot(centered, np.asarray(speed_history) - np.mean(speed_history)) / np.dot(centered, centered)
                    )
                previous_velocity = velocity
            else:
                previous_velocity = None
            previous_state = current
            prior_states.append(current)
            prior_steps.append(step)

    # Every trajectory is isolated before any temporal comparison.  Sorting
    # means a state can only see its own earlier sequence positions.
    for _trajectory, group in work.groupby(["task_id", "model_alias"], sort=False):
        indices = group.sort_values("step", kind="stable").index.to_numpy(dtype=np.int64)
        fill_layer("l1", indices, layer_one, one_available)
        fill_layer("l2", indices, layer_two, two_available)

    scalar = pd.DataFrame(values)
    response = work[["task_id", "step", "model_alias", "_winner"]].copy()
    response = pd.concat([response, scalar], axis=1)
    if response.columns.duplicated().any() or len(response) != len(raw):
        raise AssertionError("Temporal-shape response table alignment failed.")
    if response.duplicated(["task_id", "step", "model_alias"]).any():
        raise AssertionError("Temporal-shape response rows are not unique.")
    forbidden = ("answer", "correct", "raw", "text", "expected", "run_id", "trajectory_id")
    unsafe = [source for source in sources if any(token in source for token in forbidden)]
    if unsafe:
        raise AssertionError(f"Unsafe temporal-shape source names: {unsafe}")
    diagnostics = {
        "response_rows": int(len(response)),
        "task_model_trajectories": int(response.groupby(["task_id", "model_alias"], sort=False).ngroups),
        "state_scalar_source_count": len(sources),
        "layer_one_available_fraction": float(one_available.mean()),
        "layer_two_available_fraction": float(two_available.mean()),
        "contiguous_velocity_links": int(history_links),
        "contiguous_turns": int(history_turns),
        "current_to_prior_pairs": int(prior_pairs),
        "scope": "same-model/layer prefix geometry only; no raw coordinates emitted",
    }
    return response, list(sources), diagnostics
