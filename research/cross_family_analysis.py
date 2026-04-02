from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_thesis_artifacts import (
    FEATURE_LABELS,
    capability_gate,
    corrected_drift_column,
    empirical_drift_column,
    feature_name,
    first_zero_crossing,
    fitted_drift_column,
    format_boundary,
    pooled_proxy_column,
    safe_float,
)
from trace_analysis import add_temporal_features, fit_global_models


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "cross_family"
DEFAULT_REPORT_PATH = Path(__file__).resolve().parent / "CROSS_FAMILY_REPORT.md"
DEFAULT_OPEN_QUESTIONS_PATH = Path(__file__).resolve().parent / "CROSS_FAMILY_OPEN_QUESTIONS.md"
SUMMARY_CSV = "cross_family_summary.csv"
DETECTOR_CSV = "cross_family_detector_comparison.csv"
SIGNAL_CSV = "cross_family_signal_summary.csv"
BOUNDARY_FIGURE = "cross_family_boundary_comparison.png"
DETECTOR_FIGURE = "cross_family_detector_gaps.png"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_corrected_hazard(run_dir: Path) -> pd.DataFrame:
    for artifact_name in ["hazard_decomposition_by_step.csv", "hazard_drift_summary.csv"]:
        artifact_path = run_dir / artifact_name
        if artifact_path.exists():
            frame = pd.read_csv(artifact_path)
            if "conditional_hazard_drift" in frame.columns or artifact_name == "hazard_decomposition_by_step.csv":
                return frame

    step_frame = read_csv(run_dir / "trace_steps.csv")
    step_frame = add_temporal_features(step_frame)
    hazard_frame, _, _, _, _ = fit_global_models(step_frame)
    return hazard_frame


def run_label(summary: pd.Series, metadata: dict[str, Any]) -> str:
    family = str(summary.get("family", "")).strip()
    parameter_count = str(summary.get("parameter_count", "")).strip()
    if "DeepSeek" in family:
        label = f"DeepSeek {parameter_count}"
    elif "Qwen" in family:
        label = f"Qwen {parameter_count}"
    else:
        label = f"{family} {parameter_count}".strip()
    quantization = str(metadata.get("quantization", "none"))
    if quantization != "none":
        label = f"{label} {quantization}"
    return label.strip()


def top_signal_row(weight_frame: pd.DataFrame, model_name: str, positive_only: bool = False) -> dict[str, Any]:
    model_rows = weight_frame[weight_frame["model"] == model_name].copy()
    model_rows = model_rows[model_rows["feature"] != "constant_probability"]
    if positive_only:
        model_rows = model_rows[model_rows["coefficient"] > 0.0]
        if model_rows.empty:
            return {"feature": "none", "label": "none", "coefficient": float("nan"), "display": "none"}
        winner = model_rows.sort_values("coefficient", ascending=False).iloc[0]
    else:
        if model_rows.empty:
            return {"feature": "none", "label": "none", "coefficient": float("nan"), "display": "none"}
        winner = model_rows.assign(abs_coefficient=model_rows["coefficient"].abs()).sort_values("abs_coefficient", ascending=False).iloc[0]
    feature = str(winner["feature"])
    label = FEATURE_LABELS.get(feature, feature)
    coefficient = float(winner["coefficient"])
    return {
        "feature": feature,
        "label": label,
        "coefficient": coefficient,
        "display": f"{label} ({feature}, coeff={coefficient:.3f})",
    }


def best_detector(detector_frame: pd.DataFrame) -> str:
    eligible = detector_frame[detector_frame["detector"] != "oracle"].copy()
    if eligible.empty:
        return "none"
    return str(eligible.sort_values("mean_oracle_gap").iloc[0]["detector"])


def classify_late_boundary(summary_row: dict[str, Any]) -> str:
    if not summary_row["capability_gate_met"]:
        return "No late-boundary replication"
    if summary_row["never_stop_gap"] <= 0.15:
        return "No late-boundary replication"
    if summary_row["peak_step"] <= 1:
        return "No late-boundary replication"
    if summary_row["total_repairs"] <= 0 or summary_row["total_corruptions"] <= 0:
        return "No late-boundary replication"

    boundary_step = summary_row["corrected_boundary_step"]
    if pd.isna(boundary_step):
        return "No late-boundary replication"
    boundary_step = int(boundary_step)
    if boundary_step == 7:
        return "Exact step-7 replication"
    if 5 <= boundary_step <= 9 and summary_row["peak_step"] >= 5:
        return "Late-boundary replication"
    if boundary_step > 2:
        return "Weakened late-boundary support"
    return "No late-boundary replication"


def task_ids_from_metadata(metadata: dict[str, Any], step_frame: pd.DataFrame) -> list[str]:
    tasks = metadata.get("tasks") or []
    if tasks:
        return [str(task["task_id"]) for task in tasks]
    return [str(task_id) for task_id in step_frame["task_id"].drop_duplicates().tolist()]


def compare_task_alignment(run_records: list[dict[str, Any]]) -> tuple[bool, str]:
    if not run_records:
        return True, "No runs loaded."
    baseline = run_records[0]["task_ids"]
    mismatches: list[str] = []
    for record in run_records[1:]:
        if record["task_ids"] != baseline:
            mismatches.append(record["run_name"])
    if mismatches:
        return False, f"Task IDs differ for: {', '.join(mismatches)}. Cross-run comparability is weakened until the task sets are aligned."
    return True, f"Task IDs align across all {len(run_records)} runs under the shared GSM8K train split and shuffle seed 17 protocol."


def load_run_record(run_dir: Path) -> dict[str, Any]:
    pilot = read_csv(run_dir / "pilot_summary.csv")
    detectors = read_csv(run_dir / "detector_comparison.csv")
    probe = read_csv(run_dir / "correctness_probe_metrics.csv")
    weights = read_csv(run_dir / "feature_weights.csv")
    metadata = read_json(run_dir / "metadata.json")
    steps = read_csv(run_dir / "trace_steps.csv")
    hazard = load_corrected_hazard(run_dir)

    summary = pilot.iloc[0]
    q_curve = steps.groupby("step")["correct"].mean().reset_index(name="q_t")
    corrected_column = corrected_drift_column(hazard)
    empirical_column = empirical_drift_column(hazard)
    pooled_column = pooled_proxy_column(hazard)
    fitted_column = fitted_drift_column(hazard)
    corrected_boundary = first_zero_crossing(hazard, corrected_column)
    empirical_boundary = first_zero_crossing(hazard, empirical_column)
    pooled_boundary = first_zero_crossing(hazard, pooled_column) if pooled_column else None
    fitted_boundary = first_zero_crossing(hazard, fitted_column) if fitted_column else None
    correct_signal = top_signal_row(weights, "correctness_probe")
    corrupt_signal = top_signal_row(weights, "corruption_hazard", positive_only=True)
    detector_rows = detectors.set_index("detector")

    summary_row: dict[str, Any] = {
        "run_name": run_dir.name,
        "run_label": run_label(summary, metadata),
        "model_alias": str(summary["model_alias"]),
        "model_name": str(summary["model_name"]),
        "family": str(summary["family"]),
        "parameter_count": str(summary["parameter_count"]),
        "backend": str(summary["backend"]),
        "device": str(summary["device"]),
        "quantization": str(metadata.get("quantization", "none")),
        "device_map": metadata.get("device_map"),
        "task_source": str(metadata.get("task_source", "unknown")),
        "dataset_split": str(metadata.get("dataset_split", "unknown")),
        "dataset_shuffle_seed": int(metadata.get("dataset_shuffle_seed", -1)),
        "max_tasks": int(metadata.get("max_tasks", q_curve.shape[0])),
        "max_steps": int(metadata.get("max_steps", int(steps["step"].max()))),
        "max_new_tokens": int(metadata.get("max_new_tokens", 0)),
        "temperatures": " ".join(str(value) for value in metadata.get("temperatures", [])),
        "seeds": " ".join(str(value) for value in metadata.get("seeds", [])),
        "step_cost": float(metadata.get("step_cost", 0.05)),
        "prompt_mode": str(metadata.get("prompt_mode", "unknown")),
        "system_prompt_mode": str(metadata.get("system_prompt_mode", "unknown")),
        "step1_accuracy": float(q_curve.loc[q_curve["step"] == 1, "q_t"].iloc[0]),
        "peak_accuracy": float(q_curve["q_t"].max()),
        "peak_step": int(q_curve.loc[q_curve["q_t"].idxmax(), "step"]),
        "corrected_boundary_step": float(corrected_boundary) if corrected_boundary is not None else float("nan"),
        "empirical_boundary_step": float(empirical_boundary) if empirical_boundary is not None else float("nan"),
        "pooled_proxy_boundary_step": float(pooled_boundary) if pooled_boundary is not None else float("nan"),
        "fitted_boundary_step": float(fitted_boundary) if fitted_boundary is not None else float("nan"),
        "repair_rate_overall": float(summary["repair_rate_overall"]),
        "corruption_rate_overall": float(summary["corruption_rate_overall"]),
        "hazard_rule_gap": float(detector_rows.loc["hazard_drift", "mean_oracle_gap"]),
        "e_process_gap": float(detector_rows.loc["e_process", "mean_oracle_gap"]) if "e_process" in detector_rows.index else float("nan"),
        "never_stop_gap": float(detector_rows.loc["never_stop", "mean_oracle_gap"]),
        "probe_brier": float(probe["brier"].mean()),
        "probe_auc": float(probe["auc"].dropna().mean()) if probe["auc"].dropna().size else float("nan"),
        "runs_ever_correct": int(summary["runs_ever_correct"]),
        "capability_gate_met": capability_gate(float(q_curve.loc[q_curve["step"] == 1, "q_t"].iloc[0]), int(summary["runs_ever_correct"])),
        "total_repairs": int(hazard["n_repairs"].fillna(0).sum()) if "n_repairs" in hazard.columns else int(steps.get("repair", pd.Series(dtype=int)).sum()),
        "total_corruptions": int(hazard["n_corruptions"].fillna(0).sum()) if "n_corruptions" in hazard.columns else int(steps.get("corruption", pd.Series(dtype=int)).sum()),
        "best_detector": best_detector(detectors),
        "strongest_correctness_signal": correct_signal["display"],
        "strongest_correctness_feature": correct_signal["feature"],
        "strongest_corruption_signal": corrupt_signal["display"],
        "strongest_corruption_feature": corrupt_signal["feature"],
        "legacy_proxy_mismatch": bool(
            not pd.isna(float(corrected_boundary) if corrected_boundary is not None else float("nan"))
            and not pd.isna(float(pooled_boundary) if pooled_boundary is not None else float("nan"))
            and corrected_boundary != pooled_boundary
        ),
    }
    summary_row["late_boundary_assessment"] = classify_late_boundary(summary_row)

    signal_row = {
        "run_name": summary_row["run_name"],
        "run_label": summary_row["run_label"],
        "family": summary_row["family"],
        "parameter_count": summary_row["parameter_count"],
        "correctness_feature": correct_signal["feature"],
        "correctness_label": correct_signal["label"],
        "correctness_coefficient": correct_signal["coefficient"],
        "correctness_display": correct_signal["display"],
        "corruption_feature": corrupt_signal["feature"],
        "corruption_label": corrupt_signal["label"],
        "corruption_coefficient": corrupt_signal["coefficient"],
        "corruption_display": corrupt_signal["display"],
    }

    return {
        "run_dir": run_dir,
        "run_name": summary_row["run_name"],
        "run_label": summary_row["run_label"],
        "family": summary_row["family"],
        "parameter_count": summary_row["parameter_count"],
        "summary_row": summary_row,
        "signal_row": signal_row,
        "detectors": detectors,
        "hazard": hazard,
        "q_curve": q_curve,
        "task_ids": task_ids_from_metadata(metadata, steps),
    }


def build_detector_comparison(run_records: list[dict[str, Any]]) -> pd.DataFrame:
    per_run_frames: list[pd.DataFrame] = []
    for record in run_records:
        frame = record["detectors"].copy()
        frame["aggregation_level"] = "run"
        frame["group_name"] = record["run_name"]
        frame["run_name"] = record["run_name"]
        frame["run_label"] = record["run_label"]
        frame["family"] = record["family"]
        frame["parameter_count"] = record["parameter_count"]
        per_run_frames.append(frame)
    per_run = pd.concat(per_run_frames, ignore_index=True) if per_run_frames else pd.DataFrame()

    family = (
        per_run.groupby(["family", "detector"], as_index=False)
        .agg(
            mean_stop_step=("mean_stop_step", "mean"),
            mean_stop_utility=("mean_stop_utility", "mean"),
            mean_oracle_utility=("mean_oracle_utility", "mean"),
            mean_oracle_gap=("mean_oracle_gap", "mean"),
            false_early_rate=("false_early_rate", "mean"),
            false_late_rate=("false_late_rate", "mean"),
            mean_false_late_severity=("mean_false_late_severity", "mean"),
        )
        if not per_run.empty
        else pd.DataFrame()
    )
    if not family.empty:
        family["aggregation_level"] = "family"
        family["group_name"] = family["family"]
        family["run_name"] = ""
        family["run_label"] = family["family"]
        family["parameter_count"] = "family-average"

    combined = pd.concat([per_run, family], ignore_index=True, sort=False) if not family.empty else per_run
    if combined.empty:
        return combined
    combined["rank"] = (
        combined.groupby(["aggregation_level", "group_name"])["mean_oracle_gap"].rank(method="dense", ascending=True).astype(int)
    )
    return combined.sort_values(["aggregation_level", "group_name", "rank", "detector"])


def plot_boundary_comparison(run_records: list[dict[str, Any]], output_path: Path) -> None:
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(run_records), 3)))
    fig, axes = plt.subplots(3, 1, figsize=(11, 12))

    for index, record in enumerate(run_records):
        color = colors[index]
        q_curve = record["q_curve"]
        hazard = record["hazard"]
        corrected_column = corrected_drift_column(hazard)
        axes[0].plot(q_curve["step"], q_curve["q_t"], marker="o", linewidth=2, color=color, label=record["run_label"])
        axes[1].plot(hazard["step"], hazard[corrected_column], marker="o", linewidth=2, color=color, label=record["run_label"])

    axes[0].set_title("Correctness trajectories by run")
    axes[0].set_ylabel("q_t")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title("Corrected conditional hazard drift by run")
    axes[1].set_ylabel("drift")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    labels = [record["run_label"] for record in run_records]
    values = [record["summary_row"]["corrected_boundary_step"] for record in run_records]
    axes[2].bar(labels, values, color=colors[: len(run_records)])
    axes[2].set_title("Corrected boundary step by run")
    axes[2].set_ylabel("step")
    axes[2].grid(axis="y", alpha=0.25)
    plt.setp(axes[2].get_xticklabels(), rotation=20, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_detector_gaps(run_records: list[dict[str, Any]], output_path: Path) -> None:
    detectors = ["hazard_drift", "e_process", "empirical_bernstein", "never_stop"]
    colors = plt.cm.Set2(np.linspace(0.0, 1.0, max(len(run_records), 3)))
    x = np.arange(len(detectors))
    width = 0.8 / max(len(run_records), 1)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for index, record in enumerate(run_records):
        detector_rows = record["detectors"].set_index("detector")
        values = [float(detector_rows.loc[name, "mean_oracle_gap"]) for name in detectors]
        ax.bar(x + index * width - (width * (len(run_records) - 1) / 2), values, width=width, color=colors[index], label=record["run_label"])

    ax.set_xticks(x)
    ax.set_xticklabels(detectors, rotation=20, ha="right")
    ax.set_ylabel("mean oracle gap")
    ax.set_title("Detector gap comparison by run")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def strongest_cross_family_conclusion(summary_df: pd.DataFrame) -> str:
    capable = summary_df[summary_df["capability_gate_met"] == True]
    late = capable[capable["late_boundary_assessment"].isin(["Exact step-7 replication", "Late-boundary replication", "Weakened late-boundary support"])]
    if capable.empty:
        return "No run currently clears the capability gate, so the repo still lacks a theorem-facing cross-family boundary witness."
    if late["family"].nunique() >= 2:
        return "A late overthinking boundary now appears in at least two capable families under the matched GSM8K protocol, so cross-family support is materially stronger."
    if late["family"].nunique() == 1:
        return "Late-boundary evidence is still confined to a single capable family, so cross-family robustness remains unproven."
    return "After the hazard audit, the current matched GSM8K evidence does not support a late conditional-hazard boundary in any available run."


def build_report_markdown(
    summary_df: pd.DataFrame,
    detector_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    task_alignment_note: str,
    output_dir: Path,
) -> str:
    report_rows = [
        [
            row["run_label"],
            row["family"],
            row["parameter_count"],
            row["backend"],
            row["quantization"],
            safe_float(row["step1_accuracy"], 4),
            safe_float(row["peak_accuracy"], 4),
            str(int(row["peak_step"])),
            format_boundary(None if pd.isna(row["corrected_boundary_step"]) else int(row["corrected_boundary_step"])),
            safe_float(row["repair_rate_overall"], 4),
            safe_float(row["corruption_rate_overall"], 4),
            safe_float(row["hazard_rule_gap"], 4),
            safe_float(row["e_process_gap"], 4),
            safe_float(row["never_stop_gap"], 4),
            safe_float(row["probe_brier"], 4),
            safe_float(row["probe_auc"], 4),
            row["late_boundary_assessment"],
        ]
        for _, row in summary_df.iterrows()
    ]
    signal_rows = [
        [
            row["run_label"],
            row["correctness_display"],
            row["corruption_display"],
        ]
        for _, row in signal_df.iterrows()
    ]
    detector_rows = []
    run_only = detector_df[detector_df["aggregation_level"] == "run"]
    for run_name in summary_df["run_name"].tolist():
        subset = run_only[run_only["group_name"] == run_name].sort_values(["rank", "detector"]).head(3)
        for _, row in subset.iterrows():
            detector_rows.append(
                [
                    str(row["run_label"]),
                    str(row["detector"]),
                    str(int(row["rank"])),
                    safe_float(row["mean_oracle_gap"], 4),
                    safe_float(row["false_late_rate"], 3),
                ]
            )

    legacy_rows = [
        [
            row["run_label"],
            format_boundary(None if pd.isna(row["empirical_boundary_step"]) else int(row["empirical_boundary_step"])),
            format_boundary(None if pd.isna(row["corrected_boundary_step"]) else int(row["corrected_boundary_step"])),
            format_boundary(None if pd.isna(row["fitted_boundary_step"]) else int(row["fitted_boundary_step"])),
            format_boundary(None if pd.isna(row["pooled_proxy_boundary_step"]) else int(row["pooled_proxy_boundary_step"])),
            "yes" if bool(row["legacy_proxy_mismatch"]) else "no",
        ]
        for _, row in summary_df.iterrows()
    ]

    lines = [
        "# Cross-Family Report",
        "",
        "## Executive Summary",
        strongest_cross_family_conclusion(summary_df),
        "",
        task_alignment_note,
        "",
        "## Run Summary",
        *markdown_table(
            [
                "Run",
                "Family",
                "Params",
                "Backend",
                "Quant",
                "Step-1 acc",
                "Peak acc",
                "Peak step",
                "Corrected boundary",
                "Repair",
                "Corruption",
                "Hazard gap",
                "E-process gap",
                "Never-stop gap",
                "Probe Brier",
                "Probe AUC",
                "Assessment",
            ],
            report_rows,
        ),
        "",
        "## Drift Audit",
        *markdown_table(
            ["Run", "Empirical boundary", "Corrected boundary", "Fitted boundary", "Legacy pooled proxy", "Mismatch"],
            legacy_rows,
        ),
        "",
        "## Detector Rankings",
        *markdown_table(["Run", "Detector", "Rank", "Mean oracle gap", "False-late rate"], detector_rows),
        "",
        "## Signal Comparison",
        *markdown_table(["Run", "Strongest correctness signal", "Strongest corruption signal"], signal_rows),
        "",
        "## Figures",
        f"![Cross-family boundary comparison](outputs/cross_family/{BOUNDARY_FIGURE})",
        "",
        f"![Cross-family detector gaps](outputs/cross_family/{DETECTOR_FIGURE})",
        "",
    ]
    return "\n".join(lines)


def open_question_rows(summary_df: pd.DataFrame) -> list[dict[str, str]]:
    capable = summary_df[summary_df["capability_gate_met"] == True]
    qwen_family = summary_df[summary_df["family"].str.contains("Qwen", case=False, na=False)]
    qwen_capable = qwen_family[qwen_family["capability_gate_met"] == True]
    late_runs = capable[capable["late_boundary_assessment"].isin(["Exact step-7 replication", "Late-boundary replication", "Weakened late-boundary support"])]
    detector_changes = summary_df["best_detector"].nunique() > 1
    signal_features = summary_df["strongest_corruption_feature"].dropna().unique().tolist()
    has_qwen7 = any(summary_df["model_alias"] == "qwen2p5_7b")

    if capable.empty:
        boundary_status = "still unresolved"
        boundary_answer = "No run currently clears the capability gate, so boundary existence is not yet robust across families."
    elif late_runs["family"].nunique() >= 2:
        boundary_status = "answered"
        boundary_answer = "A late boundary is present in at least two capable families under the matched GSM8K protocol."
    elif late_runs["family"].nunique() == 1:
        boundary_status = "partially answered"
        boundary_answer = "A late boundary is only supported in one capable family so far, so cross-family robustness is still unproven."
    else:
        boundary_status = "answered"
        boundary_answer = "After correcting the hazard analysis, no available run shows a late conditional-hazard boundary."

    if not has_qwen7:
        capability_status = "partially answered"
        capability_answer = "Boundary location appears capability-linked in the weak sense that Qwen 0.5B stays early and low-skill, but there is no matched higher-capability Qwen run yet."
    elif not qwen_capable.empty and any(qwen_capable["late_boundary_assessment"].isin(["Exact step-7 replication", "Late-boundary replication", "Weakened late-boundary support"])):
        capability_status = "answered"
        capability_answer = "The weak Qwen control stays early while the higher-capability Qwen run moves later, which supports a capability-linked boundary location."
    else:
        capability_status = "partially answered"
        capability_answer = "The weak Qwen control stays early, but the higher-capability Qwen follow-up does not show a matching late boundary, so capability linkage is still incomplete."

    detector_status = "answered" if detector_changes else "partially answered"
    detector_answer = (
        "Detector ranking changes across runs, so ranking is not invariant across capability regimes."
        if detector_changes
        else "The current runs do not yet show a clear change in best detector ranking across regimes."
    )

    if len(set(signal_features)) == 1 and signal_features:
        signal_status = "answered"
        signal_answer = f"The same corruption-side observable ({signal_features[0]}) leads across all current runs."
    else:
        signal_status = "partially answered"
        signal_answer = "Signal leadership is not stable yet: DeepSeek emphasizes answer revision, while Qwen-family evidence includes entropy or verbosity proxies."

    if not has_qwen7:
        family_status = "still unresolved"
        family_answer = "Without a capable second-family follow-up, the current data cannot separate family effects from capability effects."
    elif late_runs["family"].nunique() >= 2:
        family_status = "partially answered"
        family_answer = "A late boundary in multiple capable families weakens the case for a DeepSeek-only effect, but one benchmark and one capable run per family still leave family-versus-capability attribution incomplete."
    else:
        family_status = "partially answered"
        family_answer = "The matched benchmark now includes multiple families, but the evidence still cannot cleanly isolate family effects from capability effects."

    if not has_qwen7:
        blocked_status = "answered"
        blocked_answer = "Without the Qwen 7B second-family run, the repo cannot claim cross-family robustness, capability-linked boundary placement within Qwen, or stable detector/signal behavior in a capable second family."
    else:
        blocked_status = "answered"
        blocked_answer = "Even with the Qwen 7B run, the repo still cannot claim benchmark-invariant behavior, clean family-versus-capability separation, or universal observable stability without additional families or benchmarks."

    return [
        {
            "question": "Is the boundary robust across model families?",
            "status": boundary_status,
            "answer": boundary_answer,
        },
        {
            "question": "Does boundary location appear capability-linked?",
            "status": capability_status,
            "answer": capability_answer,
        },
        {
            "question": "Does detector ranking change with capability?",
            "status": detector_status,
            "answer": detector_answer,
        },
        {
            "question": "Is answer revision or entropy more cross-family stable?",
            "status": signal_status,
            "answer": signal_answer,
        },
        {
            "question": "Do the data support a family effect or mostly a capability effect?",
            "status": family_status,
            "answer": family_answer,
        },
        {
            "question": "What cannot yet be claimed without the stronger second-family run?",
            "status": blocked_status,
            "answer": blocked_answer,
        },
    ]


def build_open_questions_markdown(summary_df: pd.DataFrame) -> str:
    rows = open_question_rows(summary_df)
    table_rows = [[row["question"], row["status"], row["answer"]] for row in rows]
    lines = [
        "# Cross-Family Open Questions",
        "",
        *markdown_table(["Question", "Status", "Joint answer"], table_rows),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multiple real-trace runs into a cross-family report.")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--open-questions-output", default=str(DEFAULT_OPEN_QUESTIONS_PATH))
    args = parser.parse_args()

    run_dirs = [Path(value) for value in args.run_dirs]
    output_dir = Path(args.output_dir)
    report_output = Path(args.report_output)
    open_questions_output = Path(args.open_questions_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_records = [load_run_record(run_dir) for run_dir in run_dirs]
    summary_df = pd.DataFrame([record["summary_row"] for record in run_records])
    signal_df = pd.DataFrame([record["signal_row"] for record in run_records])
    detector_df = build_detector_comparison(run_records)
    task_alignment_ok, task_alignment_note = compare_task_alignment(run_records)
    summary_df["task_alignment_ok"] = task_alignment_ok
    summary_df["task_alignment_note"] = task_alignment_note

    summary_df.to_csv(output_dir / SUMMARY_CSV, index=False)
    detector_df.to_csv(output_dir / DETECTOR_CSV, index=False)
    signal_df.to_csv(output_dir / SIGNAL_CSV, index=False)
    plot_boundary_comparison(run_records, output_dir / BOUNDARY_FIGURE)
    plot_detector_gaps(run_records, output_dir / DETECTOR_FIGURE)

    report_output.write_text(build_report_markdown(summary_df, detector_df, signal_df, task_alignment_note, output_dir), encoding="utf-8")
    open_questions_output.write_text(build_open_questions_markdown(summary_df), encoding="utf-8")

    print(f"Wrote summary csv to: {output_dir / SUMMARY_CSV}")
    print(f"Wrote detector csv to: {output_dir / DETECTOR_CSV}")
    print(f"Wrote signal csv to: {output_dir / SIGNAL_CSV}")
    print(f"Wrote boundary figure to: {output_dir / BOUNDARY_FIGURE}")
    print(f"Wrote detector figure to: {output_dir / DETECTOR_FIGURE}")
    print(f"Wrote report to: {report_output}")
    print(f"Wrote open questions to: {open_questions_output}")


if __name__ == "__main__":
    main()