#!/usr/bin/env python3
"""Timing-free, identity-free fleet-committee OOF sensitivity analysis.

This script tests whether the historical committee result survives removal of
batch timing and all learner-visible categorical metadata (model, domain,
difficulty, parser, prompt, device, and run configuration).  The only new
signal remains a leave-target-alias-out same-barrier agreement statistic.

It is still retrospective: legacy traces do not prove that peer outputs were
synchronously complete when a candidate decision was made.  It therefore
measures a necessary historical upper-bound condition, not a deployment claim.
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
import pandas as pd


SCHEMA_VERSION = "committee-oof-minimal-v1"
DEFAULT_INPUT = Path("research/outputs/experiments_v2")
DEFAULT_OUTPUT = Path("research/outputs/experiments_v2/committee_oof_minimal_v1")

# No time measurement, no model/domain/run metadata, no hidden-state origin or
# scale proxy.  Every field is emitted by the current candidate response.
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
    spec = importlib.util.spec_from_file_location("committee_minimal_base", path)
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
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def feature_columns() -> tuple[list[str], list[str]]:
    telemetry = [
        *MINIMAL_NUMERIC_COLUMNS,
        *(f"delta_{source}" for source in MINIMAL_DELTA_SOURCES),
        *(f"log1p_{source}" for source in MINIMAL_LOG1P_SOURCES),
    ]
    committee = [*telemetry, "committee_panel_nonempty", "independent_vote_count", "independent_peer_count", "independent_vote_fraction", "committee_has_independent_match", "committee_nonempty_fraction"]
    return telemetry, committee


def main() -> int:
    args = parse_args()
    base = load_base()
    if args.self_test:
        base.committee_feature_self_test()
        return 0
    if args.n_splits < 2 or args.jobs < 1:
        raise ValueError("--n-splits must be at least two and --jobs must be positive.")
    frame, files = base.load_canonical_panel(args.input_dir)
    frame = base.build_prefix_and_committee_features(frame)
    telemetry_columns, committee_columns = feature_columns()
    base.validate_feature_contract(telemetry_columns, [])
    base.validate_feature_contract(committee_columns, [])
    missing = sorted(set(committee_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Input corpus is missing minimal committee features: {missing}")
    if any(column in {"elapsed_seconds", "tokens_per_second", "temperature"} for column in committee_columns):
        raise AssertionError("Minimal committee contract accidentally retained timing/configuration fields.")
    task_groups = int(frame["task_id"].nunique())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "started_at_unix": time.time(),
        "script_sha256": source_hash(Path(__file__).resolve()),
        "base_script_sha256": source_hash(Path(__file__).with_name("run_committee_oof_experiments.py")),
        "input_dir": str(args.input_dir),
        "files": files,
        "rows": int(len(frame)),
        "trajectories": int(frame["trajectory_id"].nunique()),
        "task_groups": task_groups,
        "model_aliases": int(frame["model_alias"].nunique()),
        "n_splits": args.n_splits,
        "seed": args.seed,
        "lightgbm": base.asdict(base.LightGBMConfig()),
        "strict_contract": {
            "outer_group": "task_id",
            "committee_vote": "same task and current step; scored model alias excluded",
            "learner_categorical_features": [],
            "removed": "elapsed/tokens-per-second, temperature, all categorical model/domain/parser/prompt/device/run metadata, hidden-state norm/shift/cosine fields",
            "retained": "candidate current/prefix token-probability and output-shape telemetry plus leave-one-alias-out agreement",
            "excluded": sorted(base.FORBIDDEN_MODEL_COLUMNS),
            "required_production_barrier": "all peer outputs timestamped complete at or before candidate decision",
            "historical_limit": "legacy traces lack synchronized peer completion timestamps",
        },
        "minimal_telemetry_numeric": telemetry_columns,
        "minimal_committee_numeric": committee_columns,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
        return 0
    if task_groups < args.n_splits:
        raise ValueError("Insufficient task groups for GroupKFold.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory already contains artifacts: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = base.LightGBMConfig()
    telemetry = base.fit_outer_cv(
        frame,
        name="minimal_telemetry",
        numeric_columns=telemetry_columns,
        categorical_columns=[],
        n_splits=args.n_splits,
        seed=args.seed,
        jobs=args.jobs,
        config=config,
    )
    committee = base.fit_outer_cv(
        frame,
        name="minimal_committee",
        numeric_columns=committee_columns,
        categorical_columns=[],
        n_splits=args.n_splits,
        seed=args.seed,
        jobs=args.jobs,
        config=config,
    )
    metrics = {
        "minimal_telemetry": base.evaluate_result(
            frame, telemetry, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed
        ),
        "minimal_committee": base.evaluate_result(
            frame, committee, bootstrap_replicates=args.bootstrap_replicates, seed=args.seed + 1
        ),
    }
    manifest["completed_at_unix"] = time.time()
    manifest["status"] = "complete"
    base.atomic_write_json(args.output_dir / "committee_minimal_manifest.json", manifest)
    base.atomic_write_json(args.output_dir / "committee_minimal_metrics.json", metrics)
    prediction = frame[["task_id", "trajectory_id", "model_alias", "domain", "step", "correct"]].copy()
    prediction["fold"] = committee.fold_ids
    prediction["minimal_telemetry_score"] = telemetry.oof
    prediction["minimal_committee_score"] = committee.oof
    prediction["independent_vote_count"] = frame["independent_vote_count"].to_numpy()
    prediction["independent_peer_count"] = frame["independent_peer_count"].to_numpy()
    prediction["independent_vote_fraction"] = frame["independent_vote_fraction"].to_numpy()
    temporary = args.output_dir / "committee_minimal_predictions.csv.tmp"
    prediction.to_csv(temporary, index=False)
    os.replace(temporary, args.output_dir / "committee_minimal_predictions.csv")
    report = {
        "minimal_telemetry_oof_auc": metrics["minimal_telemetry"]["oof_auc"],
        "minimal_committee_oof_auc": metrics["minimal_committee"]["oof_auc"],
        "minimal_committee_task_cluster_bootstrap_auc_95_ci": metrics["minimal_committee"]["task_cluster_bootstrap_auc_95_ci"],
        "minimal_committee_per_domain": metrics["minimal_committee"]["per_domain"],
        "interpretation": "timing-free, identity-free historical committee sensitivity; synchronization remains unproven.",
    }
    base.atomic_write_json(args.output_dir / "committee_minimal_summary.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
