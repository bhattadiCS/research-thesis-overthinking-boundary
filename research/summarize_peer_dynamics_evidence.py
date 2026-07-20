#!/usr/bin/env python3
"""Create a fail-closed decision brief from completed peer-dynamics reports.

This program performs no fitting, calibration, or threshold selection.  Its
purpose is to freeze what the historical task-held-out experiments support and
to prevent a later prospective run from silently substituting a weaker or more
identity-dependent feature profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "peer-dynamics-evidence-brief-v1"
EXPERIMENT_ROOT = Path("research/outputs/experiments_v2")
DEFAULT_FULL_ROOT = EXPERIMENT_ROOT / "committee_oof_peer_dynamics_v3"
DEFAULT_FIXED_ROOT = EXPERIMENT_ROOT / "committee_oof_peer_dynamics_fixed13_v2"
DEFAULT_TOPOLOGY_ROOT = EXPERIMENT_ROOT / "committee_oof_peer_dynamics_topology_v1"
REPORT_NAME = "peer_dynamics_ablation_report.json"
FULL_ANONYMOUS = "anonymous_minimal"
FULL_BASELINE = "anonymous_minimal_baseline"
ROSTER = "roster_no_timing"
ROSTER_BASELINE = "roster_no_timing_baseline"
TARGET_AUC = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-root", type=Path, default=DEFAULT_FULL_ROOT)
    parser.add_argument("--fixed-root", type=Path, default=DEFAULT_FIXED_ROOT)
    parser.add_argument("--topology-root", type=Path, default=DEFAULT_TOPOLOGY_ROOT)
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


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def read_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not boolean.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not (-float("inf") < numeric < float("inf")):
        raise ValueError(f"{label} must be finite.")
    return numeric


def arm(report: dict[str, Any], name: str, *, label: str) -> dict[str, Any]:
    arms = report.get("arms")
    if not isinstance(arms, dict) or not isinstance(arms.get(name), dict):
        raise ValueError(f"{label} is missing arm {name!r}.")
    result = arms[name]
    finite_number(result.get("oof_auc"), label=f"{label}.{name}.oof_auc")
    return result


def delta(report: dict[str, Any], treatment: str, baseline: str, *, label: str) -> dict[str, Any]:
    key = f"{treatment}_minus_{baseline}"
    deltas = report.get("paired_deltas")
    if not isinstance(deltas, dict) or not isinstance(deltas.get(key), dict):
        raise ValueError(f"{label} is missing paired delta {key!r}.")
    result = deltas[key]
    finite_number(result.get("observed_delta_auc"), label=f"{label}.{key}.observed_delta_auc")
    interval = result.get("task_cluster_bootstrap_delta_auc_95_ci")
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError(f"{label}.{key} must contain a two-sided task bootstrap interval.")
    lower = finite_number(interval[0], label=f"{label}.{key}.ci_lower")
    upper = finite_number(interval[1], label=f"{label}.{key}.ci_upper")
    if lower > upper:
        raise ValueError(f"{label}.{key} has a reversed confidence interval.")
    return result


def compact_arm(arm_value: dict[str, Any]) -> dict[str, Any]:
    return {
        "oof_auc": finite_number(arm_value["oof_auc"], label="arm.oof_auc"),
        "peer_dynamics_features": int(arm_value["peer_dynamics_features"]),
        "numeric_feature_count": int(arm_value["numeric_feature_count"]),
    }


def compact_delta(delta_value: dict[str, Any]) -> dict[str, Any]:
    interval = delta_value["task_cluster_bootstrap_delta_auc_95_ci"]
    return {
        "observed_delta_auc": finite_number(delta_value["observed_delta_auc"], label="delta.observed_delta_auc"),
        "task_cluster_bootstrap_delta_auc_95_ci": [
            finite_number(interval[0], label="delta.ci_lower"),
            finite_number(interval[1], label="delta.ci_upper"),
        ],
        "bootstrap_probability_delta_gt_zero": finite_number(
            delta_value["bootstrap_probability_delta_gt_zero"], label="delta.probability"
        ),
    }


def derive_brief(
    *,
    full: dict[str, Any],
    fixed: dict[str, Any],
    topology: dict[str, Any],
    full_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Validate the three completed ablations and freeze a single candidate."""

    if full.get("preflight", {}).get("identical_labels_and_folds") is not True:
        raise ValueError("Full-profile report does not attest matched labels and folds.")
    if int(full.get("rows", 0)) != int(topology.get("rows", -1)) or int(full.get("task_groups", 0)) != int(
        topology.get("task_groups", -1)
    ):
        raise ValueError("Compact topology report is not on the same corpus/task grouping as the full-profile report.")

    full_anonymous = arm(full, FULL_ANONYMOUS, label="full")
    full_baseline = arm(full, FULL_BASELINE, label="full")
    roster = arm(full, ROSTER, label="full")
    roster_baseline = arm(full, ROSTER_BASELINE, label="full")
    fixed_anonymous = arm(fixed, FULL_ANONYMOUS, label="fixed-panel")
    fixed_baseline = arm(fixed, FULL_BASELINE, label="fixed-panel")
    topology_anonymous = arm(topology, FULL_ANONYMOUS, label="topology")
    topology_baseline = arm(topology, FULL_BASELINE, label="topology")
    full_delta = delta(full, FULL_ANONYMOUS, FULL_BASELINE, label="full")
    roster_delta = delta(full, ROSTER, ROSTER_BASELINE, label="full")
    fixed_delta = delta(fixed, FULL_ANONYMOUS, FULL_BASELINE, label="fixed-panel")
    topology_delta = delta(topology, FULL_ANONYMOUS, FULL_BASELINE, label="topology")

    if abs(float(full_baseline["oof_auc"]) - float(topology_baseline["oof_auc"])) > 1.0e-12:
        raise ValueError("Topology ablation does not retain the full-corpus anonymous baseline score.")
    if int(full_anonymous["peer_dynamics_features"]) <= int(topology_anonymous["peer_dynamics_features"]):
        raise ValueError("Full profile does not contain strictly more peer features than the topology profile.")
    if float(full_anonymous["oof_auc"]) < TARGET_AUC:
        raise ValueError("The proposed anonymous historical candidate does not meet the configured ROC-AUC gate.")
    if float(topology_anonymous["oof_auc"]) >= TARGET_AUC:
        raise ValueError("The topology profile unexpectedly meets the gate; candidate selection requires review.")
    for name, observed in (("full", full_delta), ("fixed-panel", fixed_delta)):
        interval = observed["task_cluster_bootstrap_delta_auc_95_ci"]
        if float(observed["observed_delta_auc"]) <= 0.0 or float(interval[0]) <= 0.0:
            raise ValueError(f"{name} peer-dynamics gain is not robustly positive at the stated task-bootstrap interval.")

    strict_contract = full_manifest.get("strict_contract")
    if not isinstance(strict_contract, dict):
        raise ValueError("The selected full-profile manifest lacks its strict feature-visibility contract.")
    optional_status = full.get("preflight", {}).get("optional_manifest_field_status", {})
    if not isinstance(optional_status, dict):
        optional_status = {}

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": {
            "historical_candidate": FULL_ANONYMOUS,
            "candidate_description": "anonymous closed-barrier full peer-telemetry profile",
            "roc_auc_gate": TARGET_AUC,
            "historical_oof_auc": float(full_anonymous["oof_auc"]),
            "historical_gate_met": True,
            "identity_free_learner_inputs": True,
            "timing_metadata_excluded": True,
            "status": "retrospective_candidate_only",
            "threshold_or_deployed_stop_rule": None,
            "threshold_status": "intentionally_unassigned_pending_preregistered_development_split",
        },
        "full_corpus_matched_ablation": {
            "rows": int(full["rows"]),
            "task_groups": int(full["task_groups"]),
            "anonymous_full": compact_arm(full_anonymous),
            "anonymous_baseline": compact_arm(full_baseline),
            "anonymous_delta": compact_delta(full_delta),
            "roster_full": compact_arm(roster),
            "roster_baseline": compact_arm(roster_baseline),
            "roster_delta": compact_delta(roster_delta),
        },
        "fixed_13_member_panel_sensitivity": {
            "rows": int(fixed["rows"]),
            "task_groups": int(fixed["task_groups"]),
            "anonymous_full": compact_arm(fixed_anonymous),
            "anonymous_baseline": compact_arm(fixed_baseline),
            "anonymous_delta": compact_delta(fixed_delta),
            "interpretation": "The incremental peer signal persists after fixing the panel size; this harder subset does not independently clear the absolute 0.95 gate.",
        },
        "compact_topology_rejection": {
            "rows": int(topology["rows"]),
            "task_groups": int(topology["task_groups"]),
            "topology_profile": compact_arm(topology_anonymous),
            "baseline": compact_arm(topology_baseline),
            "delta": compact_delta(topology_delta),
            "reason": "The equality/count-only profile improves over baseline but remains below the 0.95 historical ROC-AUC gate.",
        },
        "visibility_contract": strict_contract,
        "legacy_manifest_disclosure": {
            "optional_manifest_field_status": optional_status,
            "meaning": "Uniformly absent legacy fields prove no cross-arm mismatch, but do not retroactively prove their values for already-completed v3 runs.",
        },
        "fresh_confirmation_requirements": [
            "Pre-register the anonymous full peer-telemetry feature contract, model-training split, calibration method, and stopping threshold before accessing fresh evaluation labels.",
            "Use a frozen roster with pinned model/tokenizer and dataset revisions on a fresh task set disjoint from the historical corpus.",
            "Close and timestamp every full peer barrier before calculating a candidate score; record immutable event, barrier, decision, and token ledgers.",
            "Run the hardware preflight on the actual target GPU and retain its device, VRAM, temperature, power, and software-version provenance.",
            "Report the fresh task-held-out ROC-AUC, calibration, token utility, and audit status separately from this retrospective evidence.",
        ],
        "non_claims": [
            "This brief does not establish a deployed early-stopping policy or a prospective ROC-AUC result.",
            "This brief does not assign a stopping threshold from the historical evaluation corpus.",
            "The roster-aware arm is corroborating evidence, not the selected candidate, because the selected candidate avoids model-identity learner inputs.",
        ],
    }


def markdown(brief: dict[str, Any]) -> str:
    full = brief["full_corpus_matched_ablation"]
    fixed = brief["fixed_13_member_panel_sensitivity"]
    topology = brief["compact_topology_rejection"]
    decision = brief["decision"]
    full_ci = full["anonymous_delta"]["task_cluster_bootstrap_delta_auc_95_ci"]
    fixed_ci = fixed["anonymous_delta"]["task_cluster_bootstrap_delta_auc_95_ci"]
    topology_ci = topology["delta"]["task_cluster_bootstrap_delta_auc_95_ci"]
    lines = [
        "# Peer-Dynamics Evidence Decision Brief",
        "",
        f"**Historical candidate only:** `{decision['historical_candidate']}` - {decision['historical_oof_auc']:.6f} OOF ROC-AUC, above the {decision['roc_auc_gate']:.2f} gate without learner-visible model identity or timing metadata.",
        "",
        "| Evidence set | Treatment AUC | Baseline AUC | Delta AUC (95% task-bootstrap CI) | Decision |",
        "| :--- | ---: | ---: | :--- | :--- |",
        f"| Full corpus, anonymous 110-feature profile | {full['anonymous_full']['oof_auc']:.6f} | {full['anonymous_baseline']['oof_auc']:.6f} | {full['anonymous_delta']['observed_delta_auc']:+.6f} [{full_ci[0]:+.6f}, {full_ci[1]:+.6f}] | Selected historical candidate |",
        f"| Fixed 13-member panels | {fixed['anonymous_full']['oof_auc']:.6f} | {fixed['anonymous_baseline']['oof_auc']:.6f} | {fixed['anonymous_delta']['observed_delta_auc']:+.6f} [{fixed_ci[0]:+.6f}, {fixed_ci[1]:+.6f}] | Gain survives; absolute gate not met |",
        f"| Compact topology-only profile | {topology['topology_profile']['oof_auc']:.6f} | {topology['baseline']['oof_auc']:.6f} | {topology['delta']['observed_delta_auc']:+.6f} [{topology_ci[0]:+.6f}, {topology_ci[1]:+.6f}] | Rejected for < 0.95 AUC |",
        "",
        "The roster-aware full profile reaches "
        f"{full['roster_full']['oof_auc']:.6f}, but it is corroborating evidence rather than the selected candidate because the selected candidate avoids model-identity learner inputs.",
        "",
        "This is not a prospective stopping claim. The stopping threshold is intentionally unassigned until it is chosen on a preregistered development split and tested once on a fresh, synchronized fixed-roster collection.",
        "",
        "## Fresh confirmation gate",
        "",
    ]
    lines.extend(f"- {requirement}" for requirement in brief["fresh_confirmation_requirements"])
    lines += [
        "",
        "## Legacy manifest disclosure",
        "",
        brief["legacy_manifest_disclosure"]["meaning"],
        "",
    ]
    return "\n".join(lines)


def synthetic_report(*, full_auc: float, baseline_auc: float, features: int, rows: int = 10, tasks: int = 2) -> dict[str, Any]:
    delta_auc = full_auc - baseline_auc
    return {
        "rows": rows,
        "task_groups": tasks,
        "preflight": {"identical_labels_and_folds": True, "optional_manifest_field_status": {}},
        "arms": {
            FULL_ANONYMOUS: {"oof_auc": full_auc, "peer_dynamics_features": features, "numeric_feature_count": features},
            FULL_BASELINE: {"oof_auc": baseline_auc, "peer_dynamics_features": 0, "numeric_feature_count": 1},
            ROSTER: {"oof_auc": full_auc + 0.001, "peer_dynamics_features": features, "numeric_feature_count": features},
            ROSTER_BASELINE: {"oof_auc": baseline_auc + 0.001, "peer_dynamics_features": 0, "numeric_feature_count": 1},
        },
        "paired_deltas": {
            f"{FULL_ANONYMOUS}_minus_{FULL_BASELINE}": {
                "observed_delta_auc": delta_auc,
                "task_cluster_bootstrap_delta_auc_95_ci": [delta_auc / 2.0, delta_auc * 1.5],
                "bootstrap_probability_delta_gt_zero": 1.0,
            },
            f"{ROSTER}_minus_{ROSTER_BASELINE}": {
                "observed_delta_auc": delta_auc,
                "task_cluster_bootstrap_delta_auc_95_ci": [delta_auc / 2.0, delta_auc * 1.5],
                "bootstrap_probability_delta_gt_zero": 1.0,
            },
        },
    }


def self_test() -> None:
    full = synthetic_report(full_auc=0.955, baseline_auc=0.945, features=110)
    fixed = synthetic_report(full_auc=0.94, baseline_auc=0.93, features=110, rows=8, tasks=1)
    topology = synthetic_report(full_auc=0.948, baseline_auc=0.945, features=20)
    brief = derive_brief(
        full=full,
        fixed=fixed,
        topology=topology,
        full_manifest={"strict_contract": {"peer_visibility": "same closed barrier"}},
    )
    if brief["decision"]["historical_candidate"] != FULL_ANONYMOUS:
        raise AssertionError("Self-test did not freeze the anonymous candidate.")
    if brief["compact_topology_rejection"]["topology_profile"]["oof_auc"] >= TARGET_AUC:
        raise AssertionError("Self-test did not retain compact-profile rejection.")
    print("Peer-dynamics evidence-brief self-test passed.")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    output = args.output or (args.full_root / "peer_dynamics_evidence_brief.json")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing evidence brief: {output}")

    full_report_path = args.full_root / REPORT_NAME
    fixed_report_path = args.fixed_root / REPORT_NAME
    topology_report_path = args.topology_root / REPORT_NAME
    full_manifest_path = args.full_root / FULL_ANONYMOUS / "peer_dynamics_manifest.json"
    full = read_object(full_report_path, label="full-corpus report")
    fixed = read_object(fixed_report_path, label="fixed-panel report")
    topology = read_object(topology_report_path, label="topology report")
    full_manifest = read_object(full_manifest_path, label="selected candidate manifest")
    brief = derive_brief(full=full, fixed=fixed, topology=topology, full_manifest=full_manifest)
    brief["provenance"] = {
        "generator_sha256": sha256(Path(__file__).resolve()),
        "inputs": {
            "full_report": {"path": str(full_report_path), "sha256": sha256(full_report_path)},
            "fixed_panel_report": {"path": str(fixed_report_path), "sha256": sha256(fixed_report_path)},
            "topology_report": {"path": str(topology_report_path), "sha256": sha256(topology_report_path)},
            "selected_candidate_manifest": {"path": str(full_manifest_path), "sha256": sha256(full_manifest_path)},
        },
    }
    atomic_write(output, json.dumps(brief, indent=2, sort_keys=True) + "\n")
    atomic_write(output.with_suffix(".md"), markdown(brief))
    print(
        json.dumps(
            {
                "output": str(output),
                "historical_candidate": brief["decision"]["historical_candidate"],
                "historical_oof_auc": brief["decision"]["historical_oof_auc"],
                "status": brief["decision"]["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
