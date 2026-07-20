#!/usr/bin/env python3
"""Task-held-out, no-wall-clock committee sensitivity contracts.

The historical committee OOF score is above .95 but includes batch elapsed time
and broad categorical metadata.  This runner isolates timing-free variants that
add task metadata one factor at a time:

* ``roster_no_timing``: only the fixed model alias is learner-visible metadata.
* ``roster_domain_no_timing``: model alias and public benchmark domain are
  learner-visible.
* ``roster_domain_difficulty_no_timing``: model alias, public domain, and
  benchmark difficulty are learner-visible.
* ``metadata_no_timing``: historical categorical metadata is retained, but all
  wall-clock timing is removed.

Both retain leave-target-alias-out same-barrier agreement.  They remain
retrospective until a synchronized timestamped committee run proves peer
availability online.
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

import pandas as pd


SCHEMA_VERSION = "committee-oof-no-timing-v2"
DEFAULT_INPUT = Path("research/outputs/experiments_v2")
DEFAULT_ROOT = Path("research/outputs/experiments_v2/committee_oof_no_timing_v2")
CONTRACTS = (
    "roster_no_timing",
    "roster_domain_no_timing",
    "roster_domain_difficulty_no_timing",
    "metadata_no_timing",
)
TIMING_COLUMNS = {"elapsed_seconds", "tokens_per_second", "log1p_elapsed_seconds"}


def load_base() -> Any:
    path = Path(__file__).with_name("run_committee_oof_experiments.py")
    spec = importlib.util.spec_from_file_location("committee_no_timing_base", path)
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
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def contract_features(base: Any, contract: str) -> tuple[list[str], list[str]]:
    numeric = [column for column in base.telemetry_feature_columns() if column not in TIMING_COLUMNS]
    numeric += list(base.STRICT_COMMITTEE_COLUMNS)
    categorical_by_contract = {
        "roster_no_timing": ["model_alias"],
        "roster_domain_no_timing": ["model_alias", "domain"],
        "roster_domain_difficulty_no_timing": ["model_alias", "domain", "difficulty"],
        "metadata_no_timing": list(base.CATEGORICAL_COLUMNS),
    }
    categorical = categorical_by_contract[contract]
    if any(column in TIMING_COLUMNS for column in numeric) or len(numeric) != len(set(numeric)):
        raise AssertionError("No-timing numeric feature contract is malformed.")
    return numeric, categorical


def main() -> int:
    args = parse_args()
    base = load_base()
    if args.self_test:
        base.committee_feature_self_test()
        return 0
    if args.contract is None:
        raise ValueError("--contract is required unless --self-test is used.")
    if args.n_splits < 2 or args.jobs < 1:
        raise ValueError("--n-splits must be at least two and --jobs must be positive.")
    output_dir = args.output_root / args.contract
    frame, files = base.load_canonical_panel(args.input_dir)
    frame = base.build_prefix_and_committee_features(frame)
    numeric, categorical = contract_features(base, args.contract)
    base.validate_feature_contract(numeric, categorical)
    missing = sorted((set(numeric) | set(categorical)) - set(frame.columns))
    if missing:
        raise ValueError(f"Input corpus is missing no-timing contract columns: {missing}")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract": args.contract,
        "started_at_unix": time.time(),
        "script_sha256": source_hash(Path(__file__).resolve()),
        "base_script_sha256": source_hash(Path(__file__).with_name("run_committee_oof_experiments.py")),
        "input_dir": str(args.input_dir),
        "files": files,
        "rows": int(len(frame)),
        "task_groups": int(frame["task_id"].nunique()),
        "trajectories": int(frame["trajectory_id"].nunique()),
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "lightgbm": base.asdict(base.LightGBMConfig()),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "strict_contract": {
            "outer_group": "task_id",
            "committee_vote": "same task/current step/distinct model alias; scored alias excluded",
            "removed": "elapsed_seconds, tokens_per_second, log1p_elapsed_seconds",
            "categorical": categorical,
            "required_production_barrier": "all peer outputs timestamped complete before consuming the candidate score",
            "historical_limit": "peer completion timestamps are unavailable in source traces",
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
    base.atomic_write_json(output_dir / "no_timing_manifest.json", manifest)
    base.atomic_write_json(output_dir / "no_timing_metrics.json", metrics)
    prediction = frame[["task_id", "trajectory_id", "model_alias", "domain", "step", "correct"]].copy()
    prediction["fold"] = result.fold_ids
    prediction["oof_probability"] = result.oof
    prediction["independent_vote_fraction"] = frame["independent_vote_fraction"].to_numpy()
    temporary = output_dir / "no_timing_predictions.csv.tmp"
    prediction.to_csv(temporary, index=False)
    os.replace(temporary, output_dir / "no_timing_predictions.csv")
    summary = {
        "contract": args.contract,
        "oof_auc": metrics["oof_auc"],
        "task_cluster_bootstrap_auc_95_ci": metrics["task_cluster_bootstrap_auc_95_ci"],
        "per_domain": metrics["per_domain"],
        "interpretation": "task-held-out no-wall-clock committee sensitivity; not prospective proof without timestamped barrier synchronization.",
    }
    base.atomic_write_json(output_dir / "no_timing_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
