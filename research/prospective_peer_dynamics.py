#!/usr/bin/env python3
"""Reconstruct the frozen anonymous peer-dynamics contract from closed panels.

The historical winner combines 43 candidate/self/committee observables with
110 anonymous leave-target-out peer observables.  This module is deliberately
feature-only: it does not fit a model, choose a stopping threshold, inspect a
gold label, or decide whether to stop.  A future policy can call it only after
``prepare_closed_main_barrier`` has closed the current full roster.

Raw answers, thoughts, task IDs, replica IDs, and model aliases are used only
inside this module to construct equality and causal-prefix aggregates.  They
are never returned in the feature matrix or its serializable payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from committee_peer_dynamics import build_peer_dynamics_features
from real_trace_experiments import lexical_overlap
from run_committee_oof_experiments import (
    STRICT_COMMITTEE_COLUMNS,
    build_prefix_and_committee_features,
    validate_feature_contract,
)
from run_committee_oof_peer_dynamics import minimal_columns


SCHEMA_VERSION = "prospective-anonymous-peer-dynamics-v1"
FEATURE_CONTRACT_ID = "anonymous_minimal_peer_dynamics_full_v1"
FULL_PEER_FEATURE_COUNT = 110
FULL_NUMERIC_FEATURE_COUNT = 153
FORBIDDEN_RUNTIME_KEYS = {
    "correct",
    "is_correct",
    "expected_answer",
    "gold",
    "gold_answer",
    "gold_label",
    "label",
    "target",
    "target_label",
    "utility",
}
FORBIDDEN_OUTPUT_COLUMNS = {
    "task_id",
    "trajectory_id",
    "model_alias",
    "replica_id",
    "answer",
    "answer_normalized",
    "answer_key",
    "thought",
    "raw_text",
    "domain",
    "difficulty",
    "elapsed_seconds",
    "tokens_per_second",
    "batch_id",
    "batch_index",
    "device",
}
ANSWER_SPAN_COLUMNS = (
    "answer_span_mean_logprob",
    "answer_span_min_logprob",
    "answer_span_mean_entropy",
)


class ProspectivePeerFeatureError(ValueError):
    """Raised when a caller supplies an unclosed, mixed, or leaky panel."""


@dataclass(frozen=True)
class ClosedBarrierFeatureBatch:
    """Anonymous model inputs for the last (currently closed) barrier.

    ``event_ids`` preserve the roster-order mapping needed by a later policy to
    attach a score to a candidate event.  They are metadata, not model inputs.
    ``values`` contains exactly the frozen numeric columns and no identifiers
    or raw response material.
    """

    event_ids: tuple[str, ...]
    feature_columns: tuple[str, ...]
    feature_contract_id: str
    feature_contract_sha256: str
    values: pd.DataFrame

    def payload_for_row(self, row_index: int) -> dict[str, float | int | None]:
        """Return one JSON-safe, identifier-free feature payload."""

        if row_index < 0 or row_index >= len(self.values):
            raise IndexError(f"Feature row {row_index} is outside [0, {len(self.values)}).")
        row = self.values.iloc[row_index]
        payload: dict[str, float | int | None] = {}
        for column in self.feature_columns:
            value = row[column]
            if pd.isna(value):
                payload[column] = None
            elif isinstance(value, (np.integer, int)):
                payload[column] = int(value)
            else:
                payload[column] = float(value)
        _assert_safe_output_columns(payload)
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def canonical_contract_sha256(columns: Sequence[str]) -> str:
    encoded = json.dumps(list(columns), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProspectivePeerFeatureError(f"{label} must be a non-empty string.")
    return value


def _as_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ProspectivePeerFeatureError(f"{label} must be an integer, not boolean.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ProspectivePeerFeatureError(f"{label} must be an integer.") from exc
    if parsed < 0:
        raise ProspectivePeerFeatureError(f"{label} must be non-negative.")
    return parsed


def _float_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if np.isfinite(parsed) else float("nan")


def _assert_no_runtime_labels(event: Mapping[str, Any]) -> None:
    forbidden = sorted(FORBIDDEN_RUNTIME_KEYS & set(event))
    if forbidden:
        raise ProspectivePeerFeatureError(
            f"A main event exposes forbidden label/oracle fields to the feature adapter: {forbidden}"
        )


def _assert_safe_output_columns(values: Mapping[str, Any] | pd.DataFrame) -> None:
    columns = set(values) if isinstance(values, Mapping) else set(values.columns)
    forbidden = sorted(columns & FORBIDDEN_OUTPUT_COLUMNS)
    if forbidden:
        raise AssertionError(f"Feature payload exposes forbidden identifiers or raw fields: {forbidden}")
    if any(column.startswith("peer_") and "answer" in column for column in columns):
        raise AssertionError("Peer feature payload exposes a raw-answer-named column.")


def _validate_closed_panels(
    panels: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]],
    *,
    roster: Sequence[str],
) -> list[tuple[Mapping[str, Any], list[Mapping[str, Any]]]]:
    if not panels:
        raise ProspectivePeerFeatureError("At least one closed main barrier is required.")
    frozen_roster = tuple(_as_nonempty_string(alias, label="roster alias") for alias in roster)
    if len(set(frozen_roster)) != len(frozen_roster):
        raise ProspectivePeerFeatureError("Frozen roster contains duplicate aliases.")

    canonical: list[tuple[Mapping[str, Any], list[Mapping[str, Any]]]] = []
    scope: tuple[str, int] | None = None
    observed_steps: list[int] = []
    for panel_index, (barrier, raw_events) in enumerate(panels, start=1):
        if not isinstance(barrier, Mapping):
            raise ProspectivePeerFeatureError(f"Panel {panel_index} barrier must be an object.")
        if barrier.get("barrier_complete") is not True or barrier.get("complete_roster") is not True:
            raise ProspectivePeerFeatureError(f"Panel {panel_index} is not a closed complete barrier.")
        if tuple(barrier.get("expected_aliases", ())) != frozen_roster:
            raise ProspectivePeerFeatureError(f"Panel {panel_index} expected aliases differ from the frozen roster.")
        if tuple(barrier.get("completed_aliases", ())) != frozen_roster:
            raise ProspectivePeerFeatureError(f"Panel {panel_index} completed aliases differ from the frozen roster.")
        barrier_id = _as_nonempty_string(barrier.get("barrier_id"), label=f"panel {panel_index}.barrier_id")
        task_id = _as_nonempty_string(barrier.get("task_id"), label=f"panel {panel_index}.task_id")
        replica_id = _as_nonnegative_int(barrier.get("replica_id"), label=f"panel {panel_index}.replica_id")
        step = _as_nonnegative_int(barrier.get("step"), label=f"panel {panel_index}.step")
        if step < 1:
            raise ProspectivePeerFeatureError(f"Panel {panel_index}.step must start at one.")
        panel_scope = (task_id, replica_id)
        if scope is None:
            scope = panel_scope
        elif panel_scope != scope:
            raise ProspectivePeerFeatureError("A feature call may contain only one task/replica scope.")
        if step in observed_steps:
            raise ProspectivePeerFeatureError(f"Duplicate closed barrier step {step}.")
        observed_steps.append(step)

        events = list(raw_events)
        if len(events) != len(frozen_roster):
            raise ProspectivePeerFeatureError(
                f"Panel {panel_index} has {len(events)} main events, expected {len(frozen_roster)}."
            )
        by_alias: dict[str, Mapping[str, Any]] = {}
        for event in events:
            if not isinstance(event, Mapping):
                raise ProspectivePeerFeatureError(f"Panel {panel_index} contains a non-object event.")
            _assert_no_runtime_labels(event)
            if event.get("generation_kind") != "main":
                raise ProspectivePeerFeatureError("Only main-generation events may participate in the peer panel.")
            if event.get("barrier_id") != barrier_id:
                raise ProspectivePeerFeatureError("An event is attached to a different barrier.")
            if event.get("task_id") != task_id or _as_nonnegative_int(event.get("replica_id"), label="event.replica_id") != replica_id:
                raise ProspectivePeerFeatureError("An event scope differs from its barrier scope.")
            if _as_nonnegative_int(event.get("step"), label="event.step") != step:
                raise ProspectivePeerFeatureError("An event step differs from its barrier step.")
            alias = _as_nonempty_string(event.get("model_alias"), label="event.model_alias")
            if alias in by_alias:
                raise ProspectivePeerFeatureError(f"Duplicate model alias {alias!r} within one closed panel.")
            by_alias[alias] = event
        if tuple(by_alias) != frozen_roster:
            raise ProspectivePeerFeatureError("Panel aliases differ from the frozen roster or roster order.")
        ordered = [by_alias[alias] for alias in frozen_roster]
        event_ids = [_as_nonempty_string(event.get("event_id"), label="event.event_id") for event in ordered]
        if list(barrier.get("main_event_ids", ())) != event_ids:
            raise ProspectivePeerFeatureError("Barrier main_event_ids do not exactly bind the ordered main events.")
        canonical.append((barrier, ordered))

    canonical.sort(key=lambda pair: int(pair[0]["step"]))
    expected_steps = list(range(1, len(canonical) + 1))
    actual_steps = [int(barrier["step"]) for barrier, _ in canonical]
    if actual_steps != expected_steps:
        raise ProspectivePeerFeatureError(
            f"Closed panels must be a contiguous causal prefix {expected_steps}, received {actual_steps}."
        )
    return canonical


def _candidate_rows(
    panels: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]],
    *,
    roster: Sequence[str],
    max_new_tokens: int,
) -> pd.DataFrame:
    if max_new_tokens < 1:
        raise ProspectivePeerFeatureError("max_new_tokens must be positive.")
    alias_to_internal = {alias: f"candidate_{index:03d}" for index, alias in enumerate(roster, start=1)}
    histories: dict[str, list[dict[str, str]]] = {alias: [] for alias in roster}
    rows: list[dict[str, Any]] = []
    span_observed = False
    for barrier, events in panels:
        step = int(barrier["step"])
        for alias, event in zip(roster, events, strict=True):
            if event.get("model_alias") != alias:
                raise AssertionError("Validated panel lost frozen roster order.")
            thought = str(event.get("thought", ""))
            answer_normalized = str(event.get("answer_normalized", ""))
            history = histories[alias]
            prior_answer = history[-1]["answer_normalized"] if history else ""
            prior_thought = " ".join(item["thought"] for item in history[-2:])
            completion_tokens = _as_nonnegative_int(event.get("completion_tokens"), label="event.completion_tokens")
            confidence = _as_nonnegative_int(event.get("confidence"), label="event.confidence")
            if confidence > 100:
                raise ProspectivePeerFeatureError("event.confidence must be in [0, 100].")
            parse_success = _as_nonnegative_int(event.get("parse_success"), label="event.parse_success")
            model_stop_flag = _as_nonnegative_int(event.get("model_stop_flag"), label="event.model_stop_flag")
            if parse_success not in {0, 1} or model_stop_flag not in {0, 1}:
                raise ProspectivePeerFeatureError("parse_success and model_stop_flag must be binary.")
            spans = {column: _float_or_nan(event.get(column)) for column in ANSWER_SPAN_COLUMNS}
            span_observed = span_observed or any(np.isfinite(value) for value in spans.values())
            rows.append(
                {
                    "event_id": _as_nonempty_string(event.get("event_id"), label="event.event_id"),
                    # These values are internal grouping keys and are removed
                    # before the feature matrix leaves this module.
                    "task_id": "scope_000",
                    "trajectory_id": alias_to_internal[alias],
                    "model_alias": alias_to_internal[alias],
                    "_roster_order": int(tuple(roster).index(alias)),
                    "step": step,
                    "answer_normalized": answer_normalized,
                    "confidence": confidence,
                    "model_stop_flag": model_stop_flag,
                    "answer_changed": int(bool(history) and answer_normalized != prior_answer and answer_normalized != ""),
                    "thought_token_count": int(len(re.findall(r"\w+", thought))),
                    "raw_generation_tokens": completion_tokens,
                    "mean_token_logprob": _float_or_nan(event.get("mean_token_logprob")),
                    "entropy_mean": _float_or_nan(event.get("entropy_mean")),
                    "entropy_std": _float_or_nan(event.get("entropy_std")),
                    "lexical_echo": float(lexical_overlap(thought, prior_thought)),
                    "verbose_confidence_proxy": float(confidence) / 100.0 + 0.01 * completion_tokens,
                    "parse_success": parse_success,
                    "hit_max_new_tokens": int(completion_tokens == max_new_tokens),
                    "truncated_output_suspected": int(completion_tokens == max_new_tokens and not bool(parse_success)),
                    "raw_text_length_chars": _as_nonnegative_int(
                        event.get("raw_text_length_chars"), label="event.raw_text_length_chars"
                    ),
                    "raw_text_length_tokens": completion_tokens,
                    **spans,
                }
            )
            history.append({"thought": thought, "answer_normalized": answer_normalized})
    if not span_observed:
        raise ProspectivePeerFeatureError(
            "No answer-span telemetry was observed. Collect the frozen profile with --extended-observables."
        )
    return pd.DataFrame(rows)


def build_closed_barrier_features(
    panels: Sequence[tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]],
    *,
    roster: Sequence[str],
    max_new_tokens: int,
    extended_observables: bool,
) -> ClosedBarrierFeatureBatch:
    """Return the 153 anonymous numeric inputs for the final closed panel.

    ``panels`` must be the complete step-1 through current-step commit prefix
    for exactly one `(task_id, replica_id)` scope.  Caller code should obtain
    each panel from immutable main barrier commits; it must not pass verifier
    events, partial model outputs, or records from another replica.
    """

    if extended_observables is not True:
        raise ProspectivePeerFeatureError(
            "The full historical profile requires --extended-observables for answer-span telemetry."
        )
    canonical_panels = _validate_closed_panels(panels, roster=roster)
    frame = _candidate_rows(canonical_panels, roster=roster, max_new_tokens=max_new_tokens)
    frame = build_prefix_and_committee_features(frame)
    frame, peer_columns = build_peer_dynamics_features(frame)
    if len(peer_columns) != FULL_PEER_FEATURE_COUNT:
        raise AssertionError(
            f"Expected {FULL_PEER_FEATURE_COUNT} anonymous peer features, emitted {len(peer_columns)}."
        )
    feature_columns = [*minimal_columns(), *STRICT_COMMITTEE_COLUMNS, *peer_columns]
    if len(feature_columns) != FULL_NUMERIC_FEATURE_COUNT or len(feature_columns) != len(set(feature_columns)):
        raise AssertionError("Prospective full anonymous feature contract drifted from the frozen 153-column layout.")
    validate_feature_contract(feature_columns, [])
    missing = sorted(set(feature_columns) - set(frame.columns))
    if missing:
        raise AssertionError(f"Feature construction is missing frozen columns: {missing}")

    final_step = int(canonical_panels[-1][0]["step"])
    current = frame.loc[frame["step"].eq(final_step)].sort_values("_roster_order", kind="stable")
    if len(current) != len(roster):
        raise AssertionError("Final closed panel did not yield one feature row per roster member.")
    values = current.loc[:, feature_columns].copy().reset_index(drop=True)
    _assert_safe_output_columns(values)
    event_ids = tuple(str(value) for value in current["event_id"].tolist())
    if len(set(event_ids)) != len(event_ids):
        raise AssertionError("Final feature batch contains duplicate event IDs.")
    return ClosedBarrierFeatureBatch(
        event_ids=event_ids,
        feature_columns=tuple(feature_columns),
        feature_contract_id=FEATURE_CONTRACT_ID,
        feature_contract_sha256=canonical_contract_sha256(feature_columns),
        values=values,
    )


def _fixture_event(*, barrier_id: str, step: int, alias: str, answer: str, thought: str, confidence: int) -> dict[str, Any]:
    return {
        "event_id": f"{barrier_id}:{alias}",
        "barrier_id": barrier_id,
        "generation_kind": "main",
        "task_id": "opaque_task",
        "replica_id": 0,
        "step": step,
        "model_alias": alias,
        "answer_normalized": answer,
        "thought": thought,
        "confidence": confidence,
        "model_stop_flag": 0,
        "parse_success": 1,
        "completion_tokens": 12 + step,
        "raw_text_length_chars": 40 + step,
        "mean_token_logprob": -0.2 * step,
        "entropy_mean": 0.1 * step,
        "entropy_std": 0.01 * step,
        "answer_span_mean_logprob": -0.3 * step,
        "answer_span_min_logprob": -0.5 * step,
        "answer_span_mean_entropy": 0.2 * step,
    }


def _fixture_panel(step: int, roster: tuple[str, ...], answers: tuple[str, ...]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    barrier_id = f"barrier-{step}"
    events = [
        _fixture_event(
            barrier_id=barrier_id,
            step=step,
            alias=alias,
            answer=answer,
            thought=f"Reasoning {alias} at step {step}",
            confidence=70 + index,
        )
        for index, (alias, answer) in enumerate(zip(roster, answers, strict=True))
    ]
    barrier = {
        "barrier_id": barrier_id,
        "barrier_complete": True,
        "complete_roster": True,
        "expected_aliases": list(roster),
        "completed_aliases": list(roster),
        "main_event_ids": [event["event_id"] for event in events],
        "task_id": "opaque_task",
        "replica_id": 0,
        "step": step,
    }
    return barrier, events


def self_test() -> None:
    roster = ("alpha", "beta", "gamma")
    panels = [
        _fixture_panel(1, roster, ("x", "x", "y")),
        _fixture_panel(2, roster, ("x", "z", "x")),
    ]
    batch = build_closed_barrier_features(
        panels,
        roster=roster,
        max_new_tokens=64,
        extended_observables=True,
    )
    if batch.values.shape != (3, FULL_NUMERIC_FEATURE_COUNT):
        raise AssertionError("Feature adapter did not emit the frozen full anonymous shape.")
    alpha = batch.values.iloc[0]
    if not np.isclose(float(alpha["peer_support_count"]), 1.0):
        raise AssertionError("The feature adapter included the scored response in peer support.")
    if set(batch.values.columns) & FORBIDDEN_OUTPUT_COLUMNS:
        raise AssertionError("Feature adapter exposed forbidden identifiers in its output.")
    if len(batch.payload_for_row(0)) != FULL_NUMERIC_FEATURE_COUNT:
        raise AssertionError("Serialized payload lost frozen contract columns.")
    try:
        build_closed_barrier_features(
            panels,
            roster=roster,
            max_new_tokens=64,
            extended_observables=False,
        )
    except ProspectivePeerFeatureError:
        pass
    else:
        raise AssertionError("Adapter accepted a full profile without extended observables.")
    leaky = [(dict(panels[0][0]), [dict(event) for event in panels[0][1]])]
    leaky[0][1][0]["correct"] = 1
    try:
        build_closed_barrier_features(
            leaky,
            roster=roster,
            max_new_tokens=64,
            extended_observables=True,
        )
    except ProspectivePeerFeatureError:
        pass
    else:
        raise AssertionError("Adapter accepted a runtime label field.")
    print("Prospective peer-dynamics feature adapter self-test passed.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    raise ProspectivePeerFeatureError("This is a library module; use --self-test or import build_closed_barrier_features.")


if __name__ == "__main__":
    raise SystemExit(main())
