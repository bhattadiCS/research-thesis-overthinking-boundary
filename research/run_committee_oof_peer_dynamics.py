#!/usr/bin/env python3
"""Task-held-out anonymous peer-dynamics committee experiments.

These exploratory contracts enrich the existing leave-target-alias-out vote
with peer distribution concentration, peer support-versus-opposition telemetry,
and previous-closed-barrier support dynamics.  They use no labels, raw answer
text, task/domain/difficulty metadata, batch timing, or parser/prompt/device
metadata as learner inputs.  As with every legacy-panel result, synchronous
peer completion still requires a fresh prospective collection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from committee_peer_dynamics import (
    TOPOLOGY_FEATURE_COLUMNS,
    build_peer_dynamics_features,
    peer_dynamics_self_test,
)


SCHEMA_VERSION = "committee-oof-peer-dynamics-v3"
DEFAULT_INPUT = Path("research/outputs/experiments_v2")
DEFAULT_ROOT = Path("research/outputs/experiments_v2/committee_oof_peer_dynamics_v3")
CONTRACTS = (
    "anonymous_minimal_baseline",
    "anonymous_minimal",
    "roster_no_timing_baseline",
    "roster_no_timing",
)
TIMING_COLUMNS = {"elapsed_seconds", "tokens_per_second", "log1p_elapsed_seconds"}

MINIMAL_NUMERIC_COLUMNS = [
    "step",
    "confidence",
    "model_stop_flag",
    "answer_changed",
    "thought_token_count",
    "raw_generation_tokens",
    "mean_token_logprob",
    "entropy_mean",
    "entropy_std",
    "lexical_echo",
    "verbose_confidence_proxy",
    "parse_success",
    "hit_max_new_tokens",
    "truncated_output_suspected",
    "answer_span_mean_logprob",
    "answer_span_min_logprob",
    "answer_span_mean_entropy",
    "answer_span_std_entropy",
    "raw_text_length_chars",
    "raw_text_length_tokens",
    "answer_nonempty",
    "answer_char_len",
    "same_prev_answer",
    "n_answer_changes_prefix",
]
MINIMAL_DELTA_SOURCES = [
    "confidence",
    "mean_token_logprob",
    "entropy_mean",
    "entropy_std",
    "answer_span_mean_logprob",
    "answer_span_min_logprob",
    "answer_span_mean_entropy",
    "answer_span_std_entropy",
]
MINIMAL_LOG1P_SOURCES = [
    "thought_token_count",
    "raw_generation_tokens",
    "raw_text_length_chars",
    "raw_text_length_tokens",
    "answer_char_len",
]


def load_base() -> Any:
    path = Path(__file__).with_name("run_committee_oof_experiments.py")
    spec = importlib.util.spec_from_file_location("committee_peer_dynamics_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", choices=CONTRACTS)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument(
        "--peer-profile",
        choices=["full", "topology"],
        default="full",
        help="Select all anonymous peer features or the equality-only topology subset.",
    )
    parser.add_argument(
        "--fixed-panel-size",
        type=int,
        default=None,
        help="Optional roster-completeness sensitivity: retain tasks with exactly this many aliases at every required step.",
    )
    parser.add_argument("--required-steps", type=int, default=5, help="Required consecutive steps for --fixed-panel-size.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def minimal_columns() -> list[str]:
    return [
        *MINIMAL_NUMERIC_COLUMNS,
        *(f"delta_{source}" for source in MINIMAL_DELTA_SOURCES),
        *(f"log1p_{source}" for source in MINIMAL_LOG1P_SOURCES),
    ]


def filter_complete_panels(
    frame: Any,
    *,
    panel_size: int | None,
    required_steps: int,
) -> tuple[Any, dict[str, Any]]:
    """Keep only tasks with a fixed complete roster at every required barrier.

    The filter sees only alias availability, never answer values or labels. It
    is a sensitivity filter, not a random sampling operation.
    """

    if panel_size is None:
        return frame, {"applied": False}
    if panel_size < 2 or required_steps < 1:
        raise ValueError("--fixed-panel-size must be >=2 and --required-steps must be positive.")
    barrier = frame.groupby(["task_id", "step"], sort=False)["model_alias"].nunique()
    expected_steps = tuple(range(1, required_steps + 1))
    complete_ids: list[str] = []
    for task_id, values in barrier.groupby(level=0, sort=False):
        steps = tuple(int(step) for _, step in values.index.tolist())
        counts = values.to_numpy(dtype=int)
        if steps == expected_steps and len(counts) == required_steps and np.all(counts == panel_size):
            complete_ids.append(str(task_id))
    # OOF arrays are positional throughout the shared evaluator; restore a
    # compact positional index after any task-level availability filter.
    filtered = frame.loc[frame["task_id"].astype(str).isin(complete_ids)].copy().reset_index(drop=True)
    expected_rows = len(complete_ids) * required_steps * panel_size
    if len(filtered) != expected_rows:
        raise AssertionError(
            f"Fixed-panel filter expected {expected_rows} rows but retained {len(filtered)}; panel completeness is inconsistent."
        )
    if not complete_ids:
        raise ValueError("Fixed-panel filter retained no complete tasks.")
    return filtered, {
        "applied": True,
        "panel_size": int(panel_size),
        "required_steps": int(required_steps),
        "retained_tasks": int(len(complete_ids)),
        "retained_rows": int(len(filtered)),
    }


def fixed_panel_filter_self_test() -> None:
    fixture = {
        "task_id": ["keep"] * 4 + ["drop"] * 3,
        "step": [1, 1, 2, 2, 1, 1, 2],
        "model_alias": ["a", "b", "a", "b", "a", "b", "a"],
    }
    import pandas as pd

    filtered, details = filter_complete_panels(pd.DataFrame(fixture), panel_size=2, required_steps=2)
    if details["retained_tasks"] != 1 or len(filtered) != 4 or set(filtered["task_id"]) != {"keep"}:
        raise AssertionError("Fixed-panel filter did not retain exactly the complete roster task.")


def contract_columns(base: Any, contract: str, peer_columns: list[str]) -> tuple[list[str], list[str]]:
    if contract in {"anonymous_minimal_baseline", "anonymous_minimal"}:
        numeric = [*minimal_columns(), *base.STRICT_COMMITTEE_COLUMNS]
        if contract == "anonymous_minimal":
            numeric += peer_columns
        categorical: list[str] = []
    elif contract in {"roster_no_timing_baseline", "roster_no_timing"}:
        numeric = [column for column in base.telemetry_feature_columns() if column not in TIMING_COLUMNS]
        numeric += list(base.STRICT_COMMITTEE_COLUMNS)
        if contract == "roster_no_timing":
            numeric += peer_columns
        categorical = ["model_alias"]
    else:  # pragma: no cover - argparse enforces this
        raise ValueError(contract)
    if len(numeric) != len(set(numeric)):
        duplicate = sorted({column for column in numeric if numeric.count(column) > 1})
        raise AssertionError(f"Duplicate peer-dynamics numeric features: {duplicate}")
    if set(numeric) & TIMING_COLUMNS:
        raise AssertionError("Peer-dynamics contract accidentally included timing fields.")
    return numeric, categorical


def main() -> int:
    args = parse_args()
    base = load_base()
    if args.self_test:
        base.committee_feature_self_test()
        peer_dynamics_self_test()
        fixed_panel_filter_self_test()
        return 0
    if args.contract is None:
        raise ValueError("--contract is required unless --self-test is used.")
    if args.n_splits < 2 or args.jobs < 1:
        raise ValueError("--n-splits must be at least two and --jobs must be positive.")
    output_dir = args.output_root / args.contract
    frame, files = base.load_canonical_panel(args.input_dir)
    frame, panel_filter = filter_complete_panels(
        frame, panel_size=args.fixed_panel_size, required_steps=args.required_steps
    )
    frame = base.build_prefix_and_committee_features(frame)
    frame, peer_columns = build_peer_dynamics_features(frame)
    if args.peer_profile == "topology":
        peer_columns = [column for column in peer_columns if column in TOPOLOGY_FEATURE_COLUMNS]
        if tuple(peer_columns) != TOPOLOGY_FEATURE_COLUMNS:
            raise AssertionError("Topology profile differs from its frozen feature ordering.")
    numeric, categorical = contract_columns(base, args.contract, peer_columns)
    base.validate_feature_contract(numeric, categorical)
    missing = sorted((set(numeric) | set(categorical)) - set(frame.columns))
    if missing:
        raise ValueError(f"Peer-dynamics frame is missing contract features: {missing}")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract": args.contract,
        "started_at_unix": time.time(),
        "script_sha256": source_hash(Path(__file__).resolve()),
        "peer_feature_script_sha256": source_hash(Path(__file__).with_name("committee_peer_dynamics.py")),
        "base_script_sha256": source_hash(Path(__file__).with_name("run_committee_oof_experiments.py")),
        "input_dir": str(args.input_dir),
        "files": files,
        "rows": int(len(frame)),
        "task_groups": int(frame["task_id"].nunique()),
        "trajectories": int(frame["trajectory_id"].nunique()),
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "fixed_panel_filter": panel_filter,
        "lightgbm": base.asdict(base.LightGBMConfig()),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "peer_dynamics_feature_count": int(len(peer_columns) if not args.contract.endswith("_baseline") else 0),
        "peer_profile": str(args.peer_profile),
        "strict_contract": {
            "outer_group": "task_id",
            "peer_visibility": "same task/current step/full barrier/distinct model aliases; scored response excluded from every peer aggregate",
            "temporal_visibility": "same candidate trajectory and prior closed task barrier only",
            "feature_variant": "peer_dynamics" if not args.contract.endswith("_baseline") else "matched_no_peer_dynamics_control",
            "removed": "elapsed_seconds, tokens_per_second, log1p_elapsed_seconds, task/domain/difficulty/parser/prompt/device metadata, raw answer/text, labels",
            "learner_categorical_features": categorical,
            "historical_limit": "legacy traces do not record timestamped peer completion before each candidate score",
            "production_requirement": "all peer outputs must be complete and timestamped before emitting a decision score",
        },
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
        return 0
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already contains artifacts: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = base.fit_outer_cv(
        frame,
        name=args.contract,
        numeric_columns=numeric,
        categorical_columns=categorical,
        n_splits=args.n_splits,
        seed=args.seed,
        jobs=args.jobs,
        config=base.LightGBMConfig(),
    )
    metrics = base.evaluate_result(frame, result, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed)
    manifest["completed_at_unix"] = time.time()
    manifest["status"] = "complete"
    base.atomic_write_json(output_dir / "peer_dynamics_manifest.json", manifest)
    base.atomic_write_json(output_dir / "peer_dynamics_metrics.json", metrics)
    prediction = frame[["task_id", "trajectory_id", "model_alias", "domain", "step", "correct"]].copy()
    prediction["fold"] = result.fold_ids
    prediction["oof_probability"] = result.oof
    prediction["independent_vote_fraction"] = frame["independent_vote_fraction"].to_numpy()
    prediction["peer_support_fraction"] = frame["peer_support_fraction"].to_numpy()
    temporary = output_dir / "peer_dynamics_predictions.csv.tmp"
    prediction.to_csv(temporary, index=False)
    os.replace(temporary, output_dir / "peer_dynamics_predictions.csv")
    summary = {
        "contract": args.contract,
        "oof_auc": metrics["oof_auc"],
        "task_cluster_bootstrap_auc_95_ci": metrics["task_cluster_bootstrap_auc_95_ci"],
        "per_domain": metrics["per_domain"],
        "peer_dynamics_features": int(len(peer_columns) if not args.contract.endswith("_baseline") else 0),
        "peer_profile": str(args.peer_profile),
        "fixed_panel_filter": panel_filter,
        "interpretation": "task-held-out retrospective closed-barrier peer-dynamics screen; not prospective proof without timestamped synchronization.",
    }
    base.atomic_write_json(output_dir / "peer_dynamics_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
