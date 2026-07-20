#!/usr/bin/env python3
"""Paired task-bootstrap analysis for matched peer-dynamics committee runs."""

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


SCHEMA_VERSION = "committee-peer-dynamics-ablation-v1"
DEFAULT_ROOT = Path("research/outputs/experiments_v2/committee_oof_peer_dynamics_v3")
PAIR_BY_NAME = {
    "anonymous": ("anonymous_minimal", "anonymous_minimal_baseline"),
    "roster": ("roster_no_timing", "roster_no_timing_baseline"),
}
DEFAULT_PAIR_NAMES = ("anonymous", "roster")
KEY_COLUMNS = ("task_id", "trajectory_id", "model_alias", "domain", "step")
REQUIRED_SHARED_MANIFEST_FIELDS = (
    "schema_version",
    "script_sha256",
    "peer_feature_script_sha256",
    "base_script_sha256",
    "input_dir",
    "files",
    "rows",
    "task_groups",
    "trajectories",
    "n_splits",
    "seed",
    "lightgbm",
)
OPTIONAL_SHARED_MANIFEST_FIELDS = (
    "fixed_panel_filter",
    "peer_profile",
)
SHARED_MANIFEST_FIELDS = REQUIRED_SHARED_MANIFEST_FIELDS + OPTIONAL_SHARED_MANIFEST_FIELDS
LEGACY_FIELD_ABSENT = {"__manifest_field_absent__": True}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--pairs",
        nargs="+",
        choices=sorted(PAIR_BY_NAME),
        default=list(DEFAULT_PAIR_NAMES),
        help="Matched treatment/control pairs to analyze.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def pairs_from_names(names: list[str]) -> tuple[tuple[str, str], ...]:
    if not names:
        raise ValueError("At least one matched pair is required.")
    return tuple(PAIR_BY_NAME[name] for name in names)


def contracts_for_pairs(pairs: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(contract for pair in pairs for contract in pair))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def shared_manifest_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    """Normalize a manifest for exact matched-arm provenance comparison.

    ``fixed_panel_filter`` and ``peer_profile`` were added after the completed
    v3 full-corpus artifacts.  An absent field remains an explicit legacy
    sentinel: a report can only align legacy arms when every arm is equally
    absent, never by treating absence as a guessed default.
    """

    return {
        field: manifest[field] if field in manifest else LEGACY_FIELD_ABSENT
        for field in SHARED_MANIFEST_FIELDS
    }


def optional_field_status(manifests: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Describe whether optional provenance fields are declared or legacy-absent."""

    return {
        field: (
            "declared_and_matched"
            if all(field in manifest for manifest in manifests.values())
            else "uniformly_absent_legacy"
        )
        for field in OPTIONAL_SHARED_MANIFEST_FIELDS
    }


def load_contract(root: Path, contract: str) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    directory = root / contract
    predictions_path = directory / "peer_dynamics_predictions.csv"
    manifest_path = directory / "peer_dynamics_manifest.json"
    if not predictions_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing completed peer-dynamics artifacts for {contract}: {directory}")
    frame = pd.read_csv(predictions_path)
    required = set(KEY_COLUMNS) | {"correct", "fold", "oof_probability"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{predictions_path} is missing: {missing}")
    if frame.duplicated(list(KEY_COLUMNS)).any() or frame.empty:
        raise ValueError(f"{predictions_path} has duplicate or empty comparison keys.")
    if not np.isfinite(frame["oof_probability"].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{predictions_path} has a non-finite OOF score.")
    labels = frame["correct"].to_numpy(dtype=np.int8)
    if not np.array_equal(np.unique(labels), np.array([0, 1], dtype=np.int8)):
        raise ValueError(f"{predictions_path} must contain both binary correctness classes.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{manifest_path} must contain an object.")
    missing_manifest = [
        field for field in (*REQUIRED_SHARED_MANIFEST_FIELDS, "contract", "status") if field not in manifest
    ]
    if missing_manifest or manifest["status"] != "complete" or manifest["contract"] != contract:
        raise ValueError(f"{manifest_path} is not a complete matching contract artifact.")
    return (
        frame.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True),
        {
            "prediction_path": str(predictions_path),
            "prediction_sha256": sha256(predictions_path),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "shared_manifest_sha256": canonical_hash(shared_manifest_contract(manifest)),
        },
        manifest,
    )


def align(root: Path, contracts: tuple[str, ...]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    reference_contract = contracts[0]
    reference, reference_info, reference_manifest = load_contract(root, reference_contract)
    aligned = reference[list(KEY_COLUMNS) + ["correct", "fold"]].copy()
    aligned[f"score__{reference_contract}"] = reference["oof_probability"].to_numpy(dtype=np.float64)
    provenance = {reference_contract: reference_info}
    manifests = {reference_contract: reference_manifest}
    for contract in contracts[1:]:
        frame, info, manifest = load_contract(root, contract)
        candidate_contract = shared_manifest_contract(manifest)
        reference_contract_values = shared_manifest_contract(reference_manifest)
        mismatch = [
            field
            for field in SHARED_MANIFEST_FIELDS
            if candidate_contract[field] != reference_contract_values[field]
        ]
        if mismatch:
            raise ValueError(f"{contract} differs from the reference outside its declared feature contract: {mismatch}")
        candidate = frame[list(KEY_COLUMNS) + ["correct", "fold", "oof_probability"]].rename(
            columns={
                "correct": "candidate_correct",
                "fold": "candidate_fold",
                "oof_probability": f"score__{contract}",
            }
        )
        merged = aligned.merge(candidate, on=list(KEY_COLUMNS), how="left", validate="one_to_one")
        if len(merged) != len(aligned) or merged[f"score__{contract}"].isna().any():
            raise ValueError(f"{contract} does not exactly align with {reference_contract} keys.")
        if not np.array_equal(merged["correct"].to_numpy(dtype=np.int8), merged["candidate_correct"].to_numpy(dtype=np.int8)):
            raise ValueError(f"{contract} labels differ from the matched baseline.")
        if not np.array_equal(merged["fold"].to_numpy(dtype=np.int8), merged["candidate_fold"].to_numpy(dtype=np.int8)):
            raise ValueError(f"{contract} folds differ from the matched baseline.")
        aligned = merged.drop(columns=["candidate_correct", "candidate_fold"])
        provenance[contract] = info
        manifests[contract] = manifest
    return aligned, provenance, manifests


def paired_task_bootstrap(
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
    task_blocks: list[np.ndarray],
    *,
    pairs: tuple[tuple[str, str], ...],
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float | list[float]]]:
    if replicates <= 0:
        raise ValueError("--replicates must be positive.")
    generator = np.random.default_rng(seed)
    values = {name: np.empty(replicates, dtype=np.float64) for name in scores}
    for replicate in range(replicates):
        chosen = generator.integers(0, len(task_blocks), size=len(task_blocks))
        indices = np.concatenate([task_blocks[index] for index in chosen])
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size != 2:
            raise AssertionError("A full task bootstrap sample unexpectedly contained one label class.")
        for name, value in scores.items():
            values[name][replicate] = float(roc_auc_score(sampled_labels, value[indices]))
    result: dict[str, dict[str, float | list[float]]] = {}
    for treatment, baseline in pairs:
        delta = values[treatment] - values[baseline]
        point = float(roc_auc_score(labels, scores[treatment]) - roc_auc_score(labels, scores[baseline]))
        result[f"{treatment}_minus_{baseline}"] = {
            "observed_delta_auc": point,
            "task_cluster_bootstrap_delta_auc_95_ci": [
                float(np.quantile(delta, 0.025)),
                float(np.quantile(delta, 0.975)),
            ],
            "bootstrap_probability_delta_gt_zero": float(np.mean(delta > 0.0)),
        }
    return result


def per_domain(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for domain, group in frame.groupby("domain", sort=True):
        indices = group.index.to_numpy(dtype=np.int64)
        labels = group["correct"].to_numpy(dtype=np.int8)
        if np.unique(labels).size == 2:
            result[str(domain)] = float(roc_auc_score(labels, scores[indices]))
    return result


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Matched Peer-Dynamics Committee Ablation",
        "",
        "Every pair uses identical task-held-out folds, LightGBM seed/configuration, corpus hashes, and preprocessing code. Only the peer-dynamics feature block differs.",
        "",
        "| Contract | OOF ROC-AUC | Peer features |",
        "| :--- | ---: | ---: |",
    ]
    for contract in report["contracts"]:
        arm = report["arms"][contract]
        lines.append(f"| {contract} | {arm['oof_auc']:.6f} | {arm['peer_dynamics_features']} |")
    lines += ["", "| Matched treatment | Delta AUC | 95% paired task-bootstrap CI | P(Delta > 0) |", "| :--- | ---: | :--- | ---: |"]
    for name, value in report["paired_deltas"].items():
        ci = value["task_cluster_bootstrap_delta_auc_95_ci"]
        lines.append(
            f"| {name} | {value['observed_delta_auc']:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] | {value['bootstrap_probability_delta_gt_zero']:.4f} |"
        )
    lines += [
        "",
        "This is a retrospective closed-barrier analysis. It remains non-prospective until a fixed roster is synchronously collected with timestamped peer completion before scoring.",
        "",
    ]
    return "\n".join(lines)


def self_test() -> None:
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    blocks = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5]), np.array([6, 7])]
    weak = np.full(len(labels), 0.5)
    strong = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
    scores = {
        "anonymous_minimal": strong,
        "anonymous_minimal_baseline": weak,
        "roster_no_timing": strong,
        "roster_no_timing_baseline": weak,
    }
    result = paired_task_bootstrap(
        labels,
        scores,
        blocks,
        pairs=tuple(PAIR_BY_NAME.values()),
        replicates=32,
        seed=1,
    )
    if result["anonymous_minimal_minus_anonymous_minimal_baseline"]["observed_delta_auc"] <= 0.0:
        raise AssertionError("Paired bootstrap self-test did not preserve the known positive delta.")
    print("Peer-dynamics paired-analysis self-test passed.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    output = args.output or (args.root / "peer_dynamics_ablation_report.json")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {output}")
    pairs = pairs_from_names(args.pairs)
    contracts = contracts_for_pairs(pairs)
    frame, provenance, manifests = align(args.root, contracts)
    labels = frame["correct"].to_numpy(dtype=np.int8)
    scores = {contract: frame[f"score__{contract}"].to_numpy(dtype=np.float64) for contract in contracts}
    task_blocks = [np.asarray(indices, dtype=np.int64) for indices in frame.groupby("task_id", sort=False).indices.values()]
    arms = {
        contract: {
            "oof_auc": float(roc_auc_score(labels, scores[contract])),
            "per_domain_auc": per_domain(frame, scores[contract]),
            "peer_dynamics_features": int(manifests[contract]["peer_dynamics_feature_count"]),
            "numeric_feature_count": int(len(manifests[contract]["numeric_features"])),
            "categorical_features": manifests[contract]["categorical_features"],
        }
        for contract in contracts
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "root": str(args.root),
        "rows": int(len(frame)),
        "task_groups": int(frame["task_id"].nunique()),
        "contracts": list(contracts),
        "pair_names": list(args.pairs),
        "bootstrap": {"replicates": int(args.replicates), "seed": int(args.seed), "unit": "task_id"},
        "preflight": {
            "comparison_key": list(KEY_COLUMNS),
            "identical_labels_and_folds": True,
            "identical_shared_manifest_fields": list(SHARED_MANIFEST_FIELDS),
            "optional_manifest_field_status": optional_field_status(manifests),
        },
        "provenance": provenance,
        "arms": arms,
        "paired_deltas": paired_task_bootstrap(
            labels, scores, task_blocks, pairs=pairs, replicates=args.replicates, seed=args.seed
        ),
        "interpretation": "Matched task-held-out retrospective feature ablation; not a synchronized deployment claim.",
    }
    atomic_write(output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    atomic_write(output.with_suffix(".md"), markdown(report))
    print(json.dumps({"output": str(output), "arms": {key: value["oof_auc"] for key, value in arms.items()}, "paired_deltas": report["paired_deltas"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
