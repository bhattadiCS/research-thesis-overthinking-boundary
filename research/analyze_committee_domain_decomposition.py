#!/usr/bin/env python3
"""Decompose pooled committee AUC into within- and cross-domain comparisons.

For a partition by benchmark domain, pooled ROC-AUC is exactly a pair-count
weighted mixture of within-domain and cross-domain concordance.  This matters
when domain is a learner-visible categorical feature: a pooled improvement may
be primarily a cross-domain score-calibration effect rather than stronger
within-domain discrimination.
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

from analyze_committee_factor_ablation import CONTRACTS, DEFAULT_ROOT, align_contract_predictions


SCHEMA_VERSION = "committee-domain-auc-decomposition-v1"
COMPARISONS = (
    ("roster_domain_no_timing", "roster_no_timing"),
    ("roster_domain_difficulty_no_timing", "roster_domain_no_timing"),
    ("metadata_no_timing", "roster_domain_difficulty_no_timing"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def decomposition(frame: pd.DataFrame, score: np.ndarray) -> dict[str, Any]:
    labels = frame["correct"].to_numpy(dtype=np.int8)
    total_positive = int(labels.sum())
    total_negative = int(len(labels) - total_positive)
    total_pairs = total_positive * total_negative
    if total_pairs <= 0:
        raise ValueError("Both correctness classes are required for ROC-AUC decomposition.")
    pooled_auc = float(roc_auc_score(labels, score))
    within_pairs = 0
    within_concordant = 0.0
    domains: dict[str, dict[str, float | int]] = {}
    for domain, group in frame.groupby("domain", sort=True):
        indices = group.index.to_numpy(dtype=np.int64)
        domain_labels = labels[indices]
        positive = int(domain_labels.sum())
        negative = int(len(domain_labels) - positive)
        pairs = positive * negative
        if pairs == 0:
            continue
        auc = float(roc_auc_score(domain_labels, score[indices]))
        within_pairs += pairs
        within_concordant += auc * pairs
        domains[str(domain)] = {
            "rows": int(len(group)),
            "positive_rate": float(domain_labels.mean()),
            "positive_negative_pairs": int(pairs),
            "auc": auc,
        }
    cross_pairs = total_pairs - within_pairs
    if cross_pairs <= 0:
        raise ValueError("At least two domains with positive-negative pairs are required.")
    within_auc = within_concordant / within_pairs
    cross_auc = (pooled_auc * total_pairs - within_concordant) / cross_pairs
    reconstructed = (within_pairs * within_auc + cross_pairs * cross_auc) / total_pairs
    if not np.isclose(reconstructed, pooled_auc, rtol=0.0, atol=1e-12):
        raise AssertionError("Within/cross AUC decomposition did not reconstruct pooled AUC.")
    return {
        "pooled_auc": pooled_auc,
        "within_domain_pair_weighted_auc": float(within_auc),
        "cross_domain_pair_weighted_auc": float(cross_auc),
        "within_pair_fraction": float(within_pairs / total_pairs),
        "cross_pair_fraction": float(cross_pairs / total_pairs),
        "total_positive_negative_pairs": int(total_pairs),
        "within_domain_positive_negative_pairs": int(within_pairs),
        "cross_domain_positive_negative_pairs": int(cross_pairs),
        "domains": domains,
    }


def component_delta(treatment: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    within_fraction = float(reference["within_pair_fraction"])
    if not np.isclose(within_fraction, float(treatment["within_pair_fraction"]), rtol=0.0, atol=0.0):
        raise AssertionError("Comparison arms do not share the same domain pair composition.")
    pooled_delta = float(treatment["pooled_auc"]) - float(reference["pooled_auc"])
    within_component = within_fraction * (
        float(treatment["within_domain_pair_weighted_auc"])
        - float(reference["within_domain_pair_weighted_auc"])
    )
    cross_component = (1.0 - within_fraction) * (
        float(treatment["cross_domain_pair_weighted_auc"])
        - float(reference["cross_domain_pair_weighted_auc"])
    )
    if not np.isclose(within_component + cross_component, pooled_delta, rtol=0.0, atol=1e-12):
        raise AssertionError("AUC delta components did not reconstruct the pooled AUC delta.")
    return {
        "pooled_auc_delta": pooled_delta,
        "within_domain_auc_component_delta": float(within_component),
        "cross_domain_auc_component_delta": float(cross_component),
        "cross_domain_share_of_pooled_delta": float(cross_component / pooled_delta) if pooled_delta else float("nan"),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Committee AUC Domain Decomposition",
        "",
        "Pooled AUC is decomposed exactly by positive-negative pair counts. This is descriptive and uses the same audited OOF predictions as the factor-ablation report.",
        "",
        "| Contract | Pooled AUC | Within-domain AUC | Cross-domain AUC | Within-pair share |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for contract in CONTRACTS:
        value = report["arms"][contract]
        lines.append(
            f"| {contract} | {value['pooled_auc']:.6f} | {value['within_domain_pair_weighted_auc']:.6f} | "
            f"{value['cross_domain_pair_weighted_auc']:.6f} | {value['within_pair_fraction']:.4%} |"
        )
    lines += ["", "| Nested increment | Pooled delta | Within component | Cross component | Cross share |", "| :--- | ---: | ---: | ---: | ---: |"]
    for name, value in report["nested_delta_components"].items():
        lines.append(
            f"| {name} | {value['pooled_auc_delta']:+.6f} | {value['within_domain_auc_component_delta']:+.6f} | "
            f"{value['cross_domain_auc_component_delta']:+.6f} | {value['cross_domain_share_of_pooled_delta']:.2%} |"
        )
    lines += [
        "",
        "Interpretation: a cross-domain component is legitimate only if domain is known at the real decision barrier and global pooled ranking is the intended metric. It is not evidence of better within-domain discrimination.",
        "",
    ]
    return "\n".join(lines)


def run_self_test() -> None:
    frame = pd.DataFrame(
        {
            "domain": ["a", "a", "b", "b"],
            "correct": [0, 1, 0, 1],
        }
    )
    scores = np.array([0.1, 0.9, 0.4, 0.6])
    result = decomposition(frame, scores)
    if not np.isclose(result["pooled_auc"], 1.0):
        raise AssertionError("Synthetic perfect ranking did not have unit AUC.")
    if not np.isclose(result["within_domain_pair_weighted_auc"], 1.0):
        raise AssertionError("Synthetic within-domain decomposition failed.")
    if not np.isclose(result["cross_domain_pair_weighted_auc"], 1.0):
        raise AssertionError("Synthetic cross-domain decomposition failed.")
    print("Committee domain decomposition self-test passed.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    output = args.output or (args.root / "domain_auc_decomposition.json")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {output}")
    frame, provenance = align_contract_predictions(args.root)
    arms = {
        contract: decomposition(frame, frame[f"score__{contract}"].to_numpy(dtype=np.float64))
        for contract in CONTRACTS
    }
    nested = {
        f"{treatment}_minus_{reference}": component_delta(arms[treatment], arms[reference])
        for treatment, reference in COMPARISONS
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "root": str(args.root),
        "rows": int(len(frame)),
        "task_groups": int(frame["task_id"].nunique()),
        "provenance": provenance,
        "arms": arms,
        "nested_delta_components": nested,
        "interpretation": (
            "Exact pair-count decomposition of pooled retrospective OOF AUC. It diagnoses score calibration across domains; "
            "it does not establish prospective causality or within-domain generalization."
        ),
    }
    atomic_write(output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    atomic_write(output.with_suffix(".md"), markdown_report(report))
    print(json.dumps({"output": str(output), "nested_delta_components": nested}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
