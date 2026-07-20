#!/usr/bin/env python3
"""Compare nested timing-free committee contracts on paired task resamples.

The four contracts are deliberately nested in their learner-visible categorical
features while retaining the same numeric telemetry and leave-target-alias-out
committee features.  This utility refuses to compare predictions unless their
row keys, labels, and folds match exactly.  It reports conditional predictive
increments, not a prospective deployment claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


SCHEMA_VERSION = "committee-factor-ablation-v1"
DEFAULT_ROOT = Path("research/outputs/experiments_v2/committee_oof_no_timing_v2")
CONTRACTS = (
    "roster_no_timing",
    "roster_domain_no_timing",
    "roster_domain_difficulty_no_timing",
    "metadata_no_timing",
)
KEY_COLUMNS = ("task_id", "trajectory_id", "model_alias", "domain", "step")
REQUIRED_COLUMNS = set(KEY_COLUMNS) | {"correct", "fold", "oof_probability"}
PAIRWISE_COMPARISONS = (
    ("roster_domain_no_timing", "roster_no_timing"),
    ("roster_domain_difficulty_no_timing", "roster_domain_no_timing"),
    ("metadata_no_timing", "roster_domain_difficulty_no_timing"),
    ("metadata_no_timing", "roster_domain_no_timing"),
    ("metadata_no_timing", "roster_no_timing"),
)
CONTRACT_DESCRIPTIONS = {
    "roster_no_timing": "model alias only",
    "roster_domain_no_timing": "model alias + public benchmark domain",
    "roster_domain_difficulty_no_timing": "model alias + domain + benchmark difficulty",
    "metadata_no_timing": "all historical categorical response metadata, no wall-clock timing",
}
EXPECTED_CATEGORICAL_FEATURES = {
    "roster_no_timing": ["model_alias"],
    "roster_domain_no_timing": ["model_alias", "domain"],
    "roster_domain_difficulty_no_timing": ["model_alias", "domain", "difficulty"],
    "metadata_no_timing": [
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
    ],
}
SHARED_MANIFEST_FIELDS = (
    "schema_version",
    "script_sha256",
    "base_script_sha256",
    "input_dir",
    "files",
    "rows",
    "task_groups",
    "trajectories",
    "n_splits",
    "seed",
    "lightgbm",
    "numeric_features",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def prediction_path(root: Path, contract: str) -> Path:
    return root / contract / "no_timing_predictions.csv"


def manifest_path(root: Path, contract: str) -> Path:
    return root / contract / "no_timing_manifest.json"


def load_predictions(root: Path, contract: str) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    path = prediction_path(root, contract)
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction artifact for {contract}: {path}")
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{path} contains no predictions.")
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{path} has duplicate comparison keys.")
    if not np.isfinite(frame["oof_probability"].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{path} contains a non-finite OOF probability.")
    labels = frame["correct"].to_numpy(dtype=np.int8)
    if np.setdiff1d(np.unique(labels), np.array([0, 1], dtype=np.int8)).size:
        raise ValueError(f"{path} has non-binary correctness labels.")
    manifest_file = manifest_path(root, contract)
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Missing no-timing manifest for {contract}: {manifest_file}")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{manifest_file} is not valid JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_file} must contain an object.")
    missing_manifest = [field for field in (*SHARED_MANIFEST_FIELDS, "contract", "categorical_features") if field not in manifest]
    if missing_manifest:
        raise ValueError(f"{manifest_file} is missing required provenance fields: {missing_manifest}")
    if manifest["schema_version"] != "committee-oof-no-timing-v2":
        raise ValueError(f"{manifest_file} does not use the v2 no-timing schema.")
    if manifest["contract"] != contract:
        raise ValueError(f"{manifest_file} declares contract {manifest['contract']!r}, expected {contract!r}.")
    if list(manifest["categorical_features"]) != EXPECTED_CATEGORICAL_FEATURES[contract]:
        raise ValueError(f"{manifest_file} does not match the frozen categorical feature contract for {contract}.")
    if int(manifest["rows"]) != len(frame) or int(manifest["task_groups"]) != int(frame["task_id"].nunique()):
        raise ValueError(f"{manifest_file} row or task totals do not match its prediction file.")
    ordered = frame.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)
    return ordered, {
        "path": str(path),
        "sha256": sha256(path),
        "manifest_path": str(manifest_file),
        "manifest_sha256": sha256(manifest_file),
        "shared_manifest_sha256": canonical_json_hash({field: manifest[field] for field in SHARED_MANIFEST_FIELDS}),
        "rows": int(len(ordered)),
        "task_groups": int(ordered["task_id"].nunique()),
    }, manifest


def align_contract_predictions(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    reference, reference_info, reference_manifest = load_predictions(root, CONTRACTS[0])
    aligned = reference[list(KEY_COLUMNS) + ["correct", "fold"]].copy()
    aligned[f"score__{CONTRACTS[0]}"] = reference["oof_probability"].to_numpy(dtype=np.float64)
    provenance: dict[str, Any] = {CONTRACTS[0]: reference_info}
    for contract in CONTRACTS[1:]:
        frame, info, manifest = load_predictions(root, contract)
        provenance[contract] = info
        mismatched_fields = [
            field
            for field in SHARED_MANIFEST_FIELDS
            if manifest[field] != reference_manifest[field]
        ]
        if mismatched_fields:
            raise ValueError(
                f"{contract} differs from the roster control outside its categorical contract: {mismatched_fields}"
            )
        if len(frame) != len(reference):
            raise ValueError(f"{contract} has {len(frame)} rows, expected {len(reference)}.")
        candidate = frame[list(KEY_COLUMNS) + ["correct", "fold", "oof_probability"]].copy()
        candidate = candidate.rename(
            columns={
                "correct": "candidate_correct",
                "fold": "candidate_fold",
                "oof_probability": f"score__{contract}",
            }
        )
        merged = aligned.merge(candidate, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
        if len(merged) != len(aligned) or merged[f"score__{contract}"].isna().any():
            raise ValueError(f"{contract} does not have an exact one-to-one key match with the roster control.")
        if not np.array_equal(
            merged["correct"].to_numpy(dtype=np.int8),
            merged["candidate_correct"].to_numpy(dtype=np.int8),
        ):
            raise ValueError(f"{contract} correctness labels differ from the roster control.")
        if not np.array_equal(
            merged["fold"].to_numpy(dtype=np.int8),
            merged["candidate_fold"].to_numpy(dtype=np.int8),
        ):
            raise ValueError(f"{contract} outer-fold assignment differs from the roster control.")
        aligned = merged.drop(columns=["candidate_correct", "candidate_fold"])
    return aligned, provenance


def task_index_blocks(frame: pd.DataFrame) -> list[np.ndarray]:
    groups = frame.groupby("task_id", sort=False).indices
    blocks = [np.asarray(indices, dtype=np.int64) for indices in groups.values()]
    if len(blocks) < 2:
        raise ValueError("At least two task groups are required for cluster bootstrap.")
    return blocks


def paired_task_bootstrap(
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
    blocks: list[np.ndarray],
    *,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float | list[float]]]:
    """Bootstrap all arms on the same sampled task clusters."""

    if replicates <= 0:
        raise ValueError("--replicates must be positive.")
    generator = np.random.default_rng(seed)
    arm_names = list(scores)
    values = {name: np.empty(replicates, dtype=np.float64) for name in arm_names}
    filled = 0
    while filled < replicates:
        chosen = generator.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[index] for index in chosen])
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size != 2:
            continue
        for name in arm_names:
            values[name][filled] = float(roc_auc_score(sampled_labels, scores[name][indices]))
        filled += 1
    result: dict[str, dict[str, float | list[float]]] = {}
    for higher, lower in PAIRWISE_COMPARISONS:
        delta = values[higher] - values[lower]
        result[f"{higher}_minus_{lower}"] = {
            "observed_delta_auc": float(roc_auc_score(labels, scores[higher]) - roc_auc_score(labels, scores[lower])),
            "task_cluster_bootstrap_delta_auc_95_ci": [
                float(np.quantile(delta, 0.025)),
                float(np.quantile(delta, 0.975)),
            ],
            "bootstrap_probability_delta_gt_zero": float(np.mean(delta > 0.0)),
        }
    return result


def per_domain_auc(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for domain, group in frame.groupby("domain", sort=True):
        indices = group.index.to_numpy(dtype=np.int64)
        labels = group["correct"].to_numpy(dtype=np.int8)
        if np.unique(labels).size == 2:
            result[str(domain)] = float(roc_auc_score(labels, score[indices]))
    return result


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Timing-Free Committee Factor Ablation",
        "",
        "All arms use the same task-held-out folds, numeric telemetry, and leave-target-alias-out committee features.",
        "The increments are retrospective conditional prediction differences; source traces do not provide timestamped peer completion barriers.",
        "",
        "| Contract | Learner-visible categorical features | OOF ROC-AUC |",
        "| :--- | :--- | ---: |",
    ]
    for contract in CONTRACTS:
        arm = report["arms"][contract]
        lines.append(
            f"| {contract} | {CONTRACT_DESCRIPTIONS[contract]} | {arm['oof_auc']:.6f} |"
        )
    lines += ["", "| Paired comparison | Δ OOF ROC-AUC | 95% task-cluster bootstrap CI | P(Δ > 0) |", "| :--- | ---: | :--- | ---: |"]
    for name, value in report["paired_deltas"].items():
        ci = value["task_cluster_bootstrap_delta_auc_95_ci"]
        lines.append(
            f"| {name} | {value['observed_delta_auc']:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] | "
            f"{value['bootstrap_probability_delta_gt_zero']:.4f} |"
        )
    lines += [
        "",
        "Caveats: domain/difficulty must be known at decision time; after domain, difficulty only varies within MATH here. "
        "The contracts retain numerical hidden-state telemetry, so this is not a pure metadata-only causal attribution.",
        "",
    ]
    return "\n".join(lines)


def run_self_test() -> None:
    frame = pd.DataFrame(
        {
            "task_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "correct": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    labels = frame["correct"].to_numpy(dtype=np.int8)
    blocks = task_index_blocks(frame)
    weak = np.full(8, 0.5)
    strong = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
    scores = {
        "roster_no_timing": weak,
        "roster_domain_no_timing": strong,
        "roster_domain_difficulty_no_timing": strong,
        "metadata_no_timing": strong,
    }
    result = paired_task_bootstrap(labels, scores, blocks, replicates=32, seed=7)
    first = result["roster_domain_no_timing_minus_roster_no_timing"]
    if first["observed_delta_auc"] <= 0.0 or first["bootstrap_probability_delta_gt_zero"] < 0.99:
        raise AssertionError("Paired bootstrap self-test did not preserve the known positive increment.")
    print("Committee factor ablation self-test passed.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    output = args.output or (args.root / "factor_ablation_report.json")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {output}")
    aligned, provenance = align_contract_predictions(args.root)
    labels = aligned["correct"].to_numpy(dtype=np.int8)
    scores = {contract: aligned[f"score__{contract}"].to_numpy(dtype=np.float64) for contract in CONTRACTS}
    if np.unique(labels).size != 2:
        raise ValueError("Aligned predictions do not contain both correctness classes.")
    arms = {
        contract: {
            "oof_auc": float(roc_auc_score(labels, scores[contract])),
            "per_domain_auc": per_domain_auc(aligned, scores[contract]),
            "learner_visible_categorical_features": CONTRACT_DESCRIPTIONS[contract],
        }
        for contract in CONTRACTS
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "root": str(args.root),
        "rows": int(len(aligned)),
        "task_groups": int(aligned["task_id"].nunique()),
        "bootstrap": {"replicates": int(args.replicates), "seed": int(args.seed), "unit": "task_id"},
        "provenance": provenance,
        "preflight_checks": {
            "comparison_key": list(KEY_COLUMNS),
            "identical_rows_labels_and_outer_folds": True,
            "identical_shared_manifest_fields": list(SHARED_MANIFEST_FIELDS),
            "only_categorical_contract_differs": True,
        },
        "arms": arms,
        "paired_deltas": paired_task_bootstrap(
            labels, scores, task_index_blocks(aligned), replicates=args.replicates, seed=args.seed
        ),
        "interpretation": (
            "Nested, task-held-out retrospective sensitivity analysis. Deltas are conditional predictive increments, "
            "not causal effects or prospective committee validation."
        ),
        "limitations": [
            "Historical traces do not record timestamped peer completion before each candidate decision.",
            "Domain and difficulty are valid only when supplied at the deployed decision barrier.",
            "After domain is present, difficulty varies only for MATH benchmark rows in this corpus.",
            "All arms retain numerical hidden-state telemetry and agreement features; this is not a metadata-only experiment.",
        ],
    }
    atomic_write(output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path = output.with_suffix(".md")
    atomic_write(markdown_path, markdown_report(report))
    print(json.dumps({"output": str(output), "oof_auc": {k: v["oof_auc"] for k, v in arms.items()}, "paired_deltas": report["paired_deltas"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
