from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from universal_feature_analysis import (
    CAPABLE_FAMILIES,
    MODEL_SPECS,
    fit_phase2_models,
    load_traces as load_phase1_traces,
    safe_auc,
    zscore_per_family,
)


RESEARCH_ROOT = Path(__file__).resolve().parent
OUTPUT_BASE = RESEARCH_ROOT / "outputs"
REPORTS_DIR = RESEARCH_ROOT / "reports"
DEFAULT_REPORT_PATH = REPORTS_DIR / "frontier_validation_report.md"
DEFAULT_SUMMARY_PATH = REPORTS_DIR / "frontier_validation_summary.csv"
DEFAULT_INTEGRITY_PATH = REPORTS_DIR / "frontier_validation_integrity.csv"
DEFAULT_METADATA_PATH = OUTPUT_BASE / "universal_feature_analysis" / "universal_hazard_model_metadata.json"
DEFAULT_FRONTIER_RUN_DIRS = [
    OUTPUT_BASE / "real_traces_colab_gemma_4_e4b_it",
    OUTPUT_BASE / "real_traces_colab_qwen_3p5_9b",
    OUTPUT_BASE / "real_traces_colab_gemma_4_31b_it",
    OUTPUT_BASE / "real_traces_colab_llama_3p1_8b_instruct",
]
DEFAULT_SELECTED_MODEL = "quadratic_top4"
DEFAULT_STEP_COST = 0.05


def load_selected_spec(metadata_path: Path) -> tuple[dict[str, Any], Any]:
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected_name = str(metadata.get("selected_model", DEFAULT_SELECTED_MODEL))
    selected_spec = next((spec for spec in MODEL_SPECS if spec.name == selected_name), None)
    if selected_spec is None:
        selected_spec = next(spec for spec in MODEL_SPECS if spec.name == DEFAULT_SELECTED_MODEL)
        metadata["selected_model"] = selected_spec.name
    return metadata, selected_spec


def predict_probabilities(model: Any, frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    if frame.empty:
        return np.empty((0, 2), dtype=float)
    return model.predict_proba(frame[list(features)])


def first_stop_step(group: pd.DataFrame, column: str) -> int:
    ordered = group.sort_values("step")
    candidate = ordered.loc[(ordered["step"] >= 2) & (ordered[column] <= 0.0)]
    if not candidate.empty:
        return int(candidate.iloc[0]["step"])
    return int(ordered.iloc[-1]["step"])


def first_answer_step(group: pd.DataFrame) -> int:
    ordered = group.sort_values("step")
    has_answer = ordered["answer"].fillna("").astype(str).str.strip() != ""
    if has_answer.any():
        return int(ordered.loc[has_answer].iloc[0]["step"])
    return int(ordered.iloc[-1]["step"])


def stop_metrics(group: pd.DataFrame, step: int, detector: str) -> dict[str, Any]:
    ordered = group.sort_values("step")
    stop_row = ordered.loc[ordered["step"] == step].iloc[0]
    oracle_row = ordered.loc[ordered["utility"].idxmax()]
    return {
        "run_id": str(ordered.iloc[0]["run_id"]),
        "task_id": str(ordered.iloc[0]["task_id"]),
        "detector": detector,
        "stop_step": int(step),
        "stop_utility": float(stop_row["utility"]),
        "oracle_step": int(oracle_row["step"]),
        "oracle_utility": float(oracle_row["utility"]),
        "oracle_gap": float(oracle_row["utility"] - stop_row["utility"]),
        "false_early": int(step < int(oracle_row["step"])),
        "false_late": int(step > int(oracle_row["step"])),
        "false_late_severity": int(max(step - int(oracle_row["step"]), 0)),
        "stop_correct": int(stop_row["correct"]),
        "stop_tokens": float(stop_row["cumulative_generation_tokens"]),
    }


def summarize_detector(detector_frame: pd.DataFrame) -> pd.DataFrame:
    summary = detector_frame.groupby("detector").agg(
        mean_stop_step=("stop_step", "mean"),
        mean_stop_utility=("stop_utility", "mean"),
        mean_oracle_utility=("oracle_utility", "mean"),
        mean_oracle_gap=("oracle_gap", "mean"),
        false_early_rate=("false_early", "mean"),
        false_late_rate=("false_late", "mean"),
        mean_false_late_severity=("false_late_severity", "mean"),
        stop_accuracy=("stop_correct", "mean"),
        mean_stop_tokens=("stop_tokens", "mean"),
    ).reset_index()
    summary["accuracy_per_1k_tokens"] = np.where(
        summary["mean_stop_tokens"] > 0.0,
        1000.0 * summary["stop_accuracy"] / summary["mean_stop_tokens"],
        np.nan,
    )
    return summary.sort_values("mean_oracle_gap").reset_index(drop=True)


def relative_gain_pct(current: float, baseline: float) -> float:
    if not np.isfinite(current) or not np.isfinite(baseline) or baseline <= 0.0:
        return float("nan")
    return 100.0 * (current - baseline) / baseline


def first_zero_crossing(frame: pd.DataFrame, column: str) -> int | None:
    valid = frame.dropna(subset=[column]).sort_values("step")
    if valid.empty:
        return None
    previous = None
    for _, row in valid.iterrows():
        value = float(row[column])
        if previous is not None and previous > 0.0 and value <= 0.0:
            return int(row["step"])
        previous = value
    if float(valid.iloc[0][column]) <= 0.0:
        return int(valid.iloc[0]["step"])
    return None


def validate_hidden_states(hidden_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    if not hidden_dir.exists():
        return (
            {
                "hidden_dir": str(hidden_dir),
                "npz_file_count": 0,
                "invalid_npz_count": 1,
                "nan_file_count": 0,
                "inf_file_count": 0,
                "ndim_mismatch_count": 0,
                "zero_shift_count": 0,
                "all_npz_valid": False,
                "mean_l2_shift": float("nan"),
                "min_l2_shift": float("nan"),
                "max_l2_shift": float("nan"),
            },
            pd.DataFrame(
                [
                    {
                        "file": "<missing hidden_states directory>",
                        "valid": False,
                        "has_nan": False,
                        "has_inf": False,
                        "ndim": float("nan"),
                        "shape": "missing",
                        "l2_shift": float("nan"),
                        "reason": "missing hidden_states directory",
                    }
                ]
            ),
        )

    for npz_path in sorted(hidden_dir.glob("*.npz")):
        try:
            with np.load(npz_path) as payload:
                if "hidden_states" not in payload:
                    raise KeyError("missing hidden_states array")
                hidden_states = np.asarray(payload["hidden_states"])
            has_nan = bool(np.isnan(hidden_states).any())
            has_inf = bool(np.isinf(hidden_states).any())
            valid_ndim = hidden_states.ndim == 2
            l2_shift = float(np.linalg.norm(hidden_states[-1] - hidden_states[0])) if valid_ndim and hidden_states.shape[0] >= 2 else float("nan")
            zero_shift = bool(valid_ndim and hidden_states.shape[0] >= 2 and l2_shift <= 0.0)
            valid = valid_ndim and not has_nan and not has_inf and not zero_shift
            reason = ""
            if not valid_ndim:
                reason = f"expected 2D, got {hidden_states.ndim}D"
            elif has_nan:
                reason = "contains NaN"
            elif has_inf:
                reason = "contains Inf"
            elif zero_shift:
                reason = "zero L2 shift"
            rows.append(
                {
                    "file": npz_path.name,
                    "valid": valid,
                    "has_nan": has_nan,
                    "has_inf": has_inf,
                    "ndim": int(hidden_states.ndim),
                    "shape": "x".join(str(dim) for dim in hidden_states.shape),
                    "l2_shift": l2_shift,
                    "reason": reason,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "file": npz_path.name,
                    "valid": False,
                    "has_nan": False,
                    "has_inf": False,
                    "ndim": float("nan"),
                    "shape": "error",
                    "l2_shift": float("nan"),
                    "reason": f"load_error: {type(exc).__name__}: {exc}",
                }
            )

    integrity_frame = pd.DataFrame(rows)
    valid_shift_rows = integrity_frame.loc[integrity_frame["valid"] & integrity_frame["l2_shift"].notna(), "l2_shift"]
    summary = {
        "hidden_dir": str(hidden_dir),
        "npz_file_count": int(len(integrity_frame)),
        "invalid_npz_count": int((~integrity_frame["valid"]).sum()),
        "nan_file_count": int(integrity_frame["has_nan"].sum()),
        "inf_file_count": int(integrity_frame["has_inf"].sum()),
        "ndim_mismatch_count": int((integrity_frame["reason"].astype(str).str.startswith("expected 2D")).sum()),
        "zero_shift_count": int((integrity_frame["reason"] == "zero L2 shift").sum()),
        "all_npz_valid": bool((integrity_frame["valid"]).all()) if not integrity_frame.empty else False,
        "mean_l2_shift": float(valid_shift_rows.mean()) if not valid_shift_rows.empty else float("nan"),
        "min_l2_shift": float(valid_shift_rows.min()) if not valid_shift_rows.empty else float("nan"),
        "max_l2_shift": float(valid_shift_rows.max()) if not valid_shift_rows.empty else float("nan"),
    }
    return summary, integrity_frame


def load_run_metadata(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame | None, pd.DataFrame | None]:
    metadata_path = run_dir / "metadata.json"
    pilot_path = run_dir / "pilot_summary.csv"
    detector_path = run_dir / "detector_comparison.csv"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    pilot = pd.read_csv(pilot_path) if pilot_path.exists() else None
    detector = pd.read_csv(detector_path) if detector_path.exists() else None
    return metadata, pilot, detector


def prepare_frontier_frame(run_dir: Path, features: tuple[str, ...], step_cost: float) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "trace_steps.csv").sort_values(["run_id", "step"]).copy()
    frame["correct"] = pd.to_numeric(frame["correct"], errors="coerce").fillna(0).astype(int)
    frame["utility"] = pd.to_numeric(frame.get("utility"), errors="coerce")
    if frame["utility"].isna().any():
        frame["utility"] = frame["correct"].astype(float) - step_cost * (frame["step"].astype(float) - 1.0)
    frame["raw_generation_tokens"] = pd.to_numeric(frame.get("raw_generation_tokens"), errors="coerce")
    if frame["raw_generation_tokens"].isna().all():
        frame["raw_generation_tokens"] = pd.to_numeric(frame.get("thought_token_count"), errors="coerce").fillna(0.0)
    frame["cumulative_generation_tokens"] = frame.groupby("run_id")["raw_generation_tokens"].cumsum()
    frame["next_correct"] = frame.groupby("run_id")["correct"].shift(-1)
    frame["has_next"] = frame["next_correct"].notna()
    frame["event_repair"] = ((frame["correct"] == 0) & (frame["next_correct"] == 1)).astype(int)
    frame["event_corruption"] = ((frame["correct"] == 1) & (frame["next_correct"] == 0)).astype(int)
    frame["valid_repair"] = (frame["correct"] == 0) & frame["has_next"]
    frame["valid_corruption"] = (frame["correct"] == 1) & frame["has_next"]
    for feature in features:
        frame[feature] = zscore_per_family(frame[feature])
    return frame


def model_label(metadata: dict[str, Any], pilot: pd.DataFrame | None, run_dir: Path) -> str:
    if metadata.get("model"):
        model_meta = metadata["model"]
        family = str(model_meta.get("family", "")).strip()
        parameter_count = str(model_meta.get("parameter_count", "")).strip()
        if family and parameter_count:
            return f"{family} {parameter_count}"
        alias = str(model_meta.get("alias", "")).strip()
        if alias:
            return alias
    if pilot is not None and not pilot.empty:
        family = str(pilot.iloc[0].get("family", "")).strip()
        parameter_count = str(pilot.iloc[0].get("parameter_count", "")).strip()
        if family and parameter_count:
            return f"{family} {parameter_count}"
        alias = str(pilot.iloc[0].get("model_alias", "")).strip()
        if alias:
            return alias
    return run_dir.name


def evaluate_run_dir(
    run_dir: Path,
    selected_spec: Any,
    q_model: Any,
    alpha_model: Any,
    beta_model: Any,
    step_cost: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    metadata, pilot, detector_summary = load_run_metadata(run_dir)
    frame = prepare_frontier_frame(run_dir, selected_spec.features, step_cost)
    display_name = model_label(metadata, pilot, run_dir)

    frame["q_hat_universal"] = predict_probabilities(q_model, frame, selected_spec.features)[:, 1]
    frame["alpha_hat_universal"] = predict_probabilities(alpha_model, frame, selected_spec.features)[:, 1]
    frame["beta_hat_universal"] = predict_probabilities(beta_model, frame, selected_spec.features)[:, 1]
    frame["mu_hat_universal"] = (
        (1.0 - frame["q_hat_universal"]) * frame["alpha_hat_universal"]
        - frame["q_hat_universal"] * frame["beta_hat_universal"]
        - step_cost
    )

    q_auc = safe_auc(frame["correct"], frame["q_hat_universal"].to_numpy())
    alpha_frame = frame[frame["valid_repair"]].copy()
    beta_frame = frame[frame["valid_corruption"]].copy()
    alpha_auc = safe_auc(alpha_frame["event_repair"], predict_probabilities(alpha_model, alpha_frame, selected_spec.features)[:, 1]) if not alpha_frame.empty else float("nan")
    beta_auc = safe_auc(beta_frame["event_corruption"], predict_probabilities(beta_model, beta_frame, selected_spec.features)[:, 1]) if not beta_frame.empty else float("nan")

    detector_rows: list[dict[str, Any]] = []
    for _, group in frame.groupby("run_id"):
        detector_rows.append(stop_metrics(group, first_stop_step(group, "mu_hat_universal"), "universal_algorithm_x"))
        detector_rows.append(stop_metrics(group, first_answer_step(group), "first_answer"))
        detector_rows.append(stop_metrics(group, int(group["step"].max()), "never_stop"))
    detector_frame = pd.DataFrame(detector_rows)
    summary = summarize_detector(detector_frame)
    universal_row = summary.set_index("detector").loc["universal_algorithm_x"]
    never_stop_row = summary.set_index("detector").loc["never_stop"]
    first_answer_row = summary.set_index("detector").loc["first_answer"]

    hidden_summary, integrity_frame = validate_hidden_states(run_dir / "hidden_states")
    integrity_frame.insert(0, "run_dir", run_dir.name)

    by_step = frame.groupby("step", as_index=False).agg(
        q_hat_universal=("q_hat_universal", "mean"),
        alpha_hat_universal=("alpha_hat_universal", "mean"),
        beta_hat_universal=("beta_hat_universal", "mean"),
        mu_hat_universal=("mu_hat_universal", "mean"),
    )
    universal_boundary_step = first_zero_crossing(by_step, "mu_hat_universal")

    hazard_gap = float("nan")
    if detector_summary is not None and not detector_summary.empty:
        hazard_row = detector_summary.loc[detector_summary["detector"] == "hazard_drift"]
        if not hazard_row.empty:
            hazard_gap = float(hazard_row.iloc[0]["mean_oracle_gap"])

    parse_success_rate = float(pd.to_numeric(frame.get("parse_success"), errors="coerce").mean()) if "parse_success" in frame.columns else float("nan")
    success_gain = relative_gain_pct(
        float(universal_row["accuracy_per_1k_tokens"]),
        float(never_stop_row["accuracy_per_1k_tokens"]),
    )

    result = {
        "run_dir": run_dir.name,
        "model_label": display_name,
        "model_alias": str(frame.iloc[0].get("model_alias", "")),
        "n_runs": int(frame["run_id"].nunique()),
        "n_tasks": int(frame["task_id"].nunique()),
        "q_auc_zero_shot": float(q_auc),
        "alpha_auc_zero_shot": float(alpha_auc),
        "beta_auc_zero_shot": float(beta_auc),
        "universal_boundary_step": universal_boundary_step if universal_boundary_step is not None else "not_observed",
        "universal_mean_stop_step": float(universal_row["mean_stop_step"]),
        "universal_mean_stop_utility": float(universal_row["mean_stop_utility"]),
        "universal_mean_oracle_gap": float(universal_row["mean_oracle_gap"]),
        "universal_false_late_rate": float(universal_row["false_late_rate"]),
        "universal_stop_accuracy": float(universal_row["stop_accuracy"]),
        "universal_mean_stop_tokens": float(universal_row["mean_stop_tokens"]),
        "universal_accuracy_per_1k_tokens": float(universal_row["accuracy_per_1k_tokens"]),
        "never_stop_accuracy_per_1k_tokens": float(never_stop_row["accuracy_per_1k_tokens"]),
        "first_answer_accuracy_per_1k_tokens": float(first_answer_row["accuracy_per_1k_tokens"]),
        "efficiency_gain_vs_never_stop_pct": float(success_gain),
        "utility_gain_vs_never_stop": float(universal_row["mean_stop_utility"] - never_stop_row["mean_stop_utility"]),
        "hazard_drift_mean_oracle_gap": hazard_gap,
        "parse_success_rate": parse_success_rate,
        **hidden_summary,
    }
    result["data_quality_pass"] = bool(
        result["all_npz_valid"]
        and np.isfinite(result["parse_success_rate"])
        and result["parse_success_rate"] >= 0.95
        and result["npz_file_count"] > 0
    )
    result["efficiency_pass"] = bool(np.isfinite(success_gain) and success_gain > 25.0)
    return result, detector_frame, integrity_frame


def to_markdown_table(frame: pd.DataFrame, float_columns: set[str]) -> str:
    if frame.empty:
        return "| none |\n| --- |"
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in headers:
            value = row[column]
            if column in float_columns and pd.notna(value):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(
    results_frame: pd.DataFrame,
    missing_run_dirs: list[Path],
    requested_run_count: int,
    selected_name: str,
    step_cost: float,
    protocol_name: str,
) -> str:
    completed = int(len(results_frame))
    data_quality_pass = bool(results_frame["data_quality_pass"].all()) if completed else False
    efficiency_pass = bool(results_frame["efficiency_pass"].all()) if completed else False
    complete_frontier_set = completed == requested_run_count and not missing_run_dirs

    verdict_parts = []
    if complete_frontier_set and data_quality_pass and efficiency_pass:
        verdict_parts.append("The available frontier runs satisfy the prompt-level data quality gates and the >25% zero-shot efficiency target versus never-stop.")
    elif completed:
        verdict_parts.append("Frontier validation is partially complete, but at least one required gate is still unmet or one target run is missing.")
    else:
        verdict_parts.append("No completed frontier run directories were available, so the frontier claim remains untested in this workspace.")
    if missing_run_dirs:
        verdict_parts.append("Missing run directories: " + ", ".join(path.name for path in missing_run_dirs) + ".")

    score_table = pd.DataFrame(
        [
            {
                "criterion": f"All requested protocol runs present ({requested_run_count})",
                "status": "pass" if complete_frontier_set else "fail",
            },
            {
                "criterion": "All hidden-state .npz files valid",
                "status": "pass" if data_quality_pass else "fail",
            },
            {
                "criterion": "Zero-shot efficiency gain > 25% vs never-stop",
                "status": "pass" if efficiency_pass else "fail",
            },
        ]
    )

    result_columns = [
        "model_label",
        "q_auc_zero_shot",
        "alpha_auc_zero_shot",
        "beta_auc_zero_shot",
        "universal_mean_stop_step",
        "universal_mean_oracle_gap",
        "universal_stop_accuracy",
        "universal_accuracy_per_1k_tokens",
        "never_stop_accuracy_per_1k_tokens",
        "efficiency_gain_vs_never_stop_pct",
        "parse_success_rate",
        "hazard_drift_mean_oracle_gap",
    ]
    integrity_columns = [
        "model_label",
        "npz_file_count",
        "invalid_npz_count",
        "nan_file_count",
        "inf_file_count",
        "zero_shift_count",
        "mean_l2_shift",
        "min_l2_shift",
        "max_l2_shift",
        "data_quality_pass",
    ]

    lines = [
        "# Frontier Validation Report",
        "",
        "## Verdict",
        "",
        " ".join(verdict_parts),
        "",
        "## Protocol",
        "",
        f"- Protocol label: `{protocol_name}`.",
        f"- Phase 1 intake: `{selected_name}` fit on capable legacy families only ({', '.join(CAPABLE_FAMILIES)}).",
        f"- Step cost: `{step_cost:.2f}` utility units per extra reasoning step.",
        "- Efficiency metric: stop accuracy divided by mean cumulative generated tokens, reported as accuracy per 1k generated tokens.",
        "- Baseline for the >25% gate: `never_stop` on the same frontier trace set.",
        "",
        "## Success Matrix",
        "",
        to_markdown_table(score_table, float_columns=set()),
        "",
        "## Zero-Shot Frontier Results",
        "",
        to_markdown_table(
            results_frame[result_columns] if not results_frame.empty else results_frame,
            float_columns={
                "q_auc_zero_shot",
                "alpha_auc_zero_shot",
                "beta_auc_zero_shot",
                "universal_mean_stop_step",
                "universal_mean_oracle_gap",
                "universal_stop_accuracy",
                "universal_accuracy_per_1k_tokens",
                "never_stop_accuracy_per_1k_tokens",
                "efficiency_gain_vs_never_stop_pct",
                "parse_success_rate",
                "hazard_drift_mean_oracle_gap",
            },
        ),
        "",
        "## Hidden-State Integrity",
        "",
        to_markdown_table(
            results_frame[integrity_columns] if not results_frame.empty else results_frame,
            float_columns={"mean_l2_shift", "min_l2_shift", "max_l2_shift"},
        ),
    ]
    if missing_run_dirs:
        lines.extend(
            [
                "",
                "## Missing Runs",
                "",
                "| expected_run_dir |",
                "| --- |",
            ]
        )
        lines.extend(f"| {path.name} |" for path in missing_run_dirs)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="*", default=None, help="Completed frontier run directories to evaluate.")
    parser.add_argument("--metadata-path", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--integrity-path", default=str(DEFAULT_INTEGRITY_PATH))
    parser.add_argument("--protocol-name", default="frontier_validation")
    parser.add_argument("--random-state", type=int, default=7)
    args = parser.parse_args()

    metadata, selected_spec = load_selected_spec(Path(args.metadata_path))
    step_cost = float(metadata.get("step_cost", DEFAULT_STEP_COST))
    phase1_frame, _ = load_phase1_traces()
    q_model, alpha_model, beta_model = fit_phase2_models(
        phase1_frame,
        selected_spec,
        families=tuple(metadata.get("capable_families", CAPABLE_FAMILIES)),
        random_state=args.random_state,
    )

    requested_run_dirs = [Path(path) for path in args.run_dirs] if args.run_dirs else list(DEFAULT_FRONTIER_RUN_DIRS)
    existing_run_dirs = [path for path in requested_run_dirs if path.exists()]
    missing_run_dirs = [path for path in requested_run_dirs if not path.exists()]

    results: list[dict[str, Any]] = []
    integrity_frames: list[pd.DataFrame] = []
    if existing_run_dirs:
        for run_dir in existing_run_dirs:
            result, _, integrity_frame = evaluate_run_dir(
                run_dir,
                selected_spec,
                q_model,
                alpha_model,
                beta_model,
                step_cost,
            )
            results.append(result)
            integrity_frames.append(integrity_frame)

    results_frame = pd.DataFrame(results)
    if results_frame.empty:
        results_frame = pd.DataFrame(
            columns=[
                "run_dir",
                "model_label",
                "model_alias",
                "n_runs",
                "n_tasks",
                "q_auc_zero_shot",
                "alpha_auc_zero_shot",
                "beta_auc_zero_shot",
                "universal_boundary_step",
                "universal_mean_stop_step",
                "universal_mean_stop_utility",
                "universal_mean_oracle_gap",
                "universal_false_late_rate",
                "universal_stop_accuracy",
                "universal_mean_stop_tokens",
                "universal_accuracy_per_1k_tokens",
                "never_stop_accuracy_per_1k_tokens",
                "first_answer_accuracy_per_1k_tokens",
                "efficiency_gain_vs_never_stop_pct",
                "utility_gain_vs_never_stop",
                "hazard_drift_mean_oracle_gap",
                "parse_success_rate",
                "hidden_dir",
                "npz_file_count",
                "invalid_npz_count",
                "nan_file_count",
                "inf_file_count",
                "ndim_mismatch_count",
                "zero_shift_count",
                "all_npz_valid",
                "mean_l2_shift",
                "min_l2_shift",
                "max_l2_shift",
                "data_quality_pass",
                "efficiency_pass",
            ]
        )
    integrity_output = pd.concat(integrity_frames, ignore_index=True) if integrity_frames else pd.DataFrame(
        columns=["run_dir", "file", "valid", "has_nan", "has_inf", "ndim", "shape", "l2_shift", "reason"]
    )

    report_path = Path(args.report_path)
    summary_path = Path(args.summary_path)
    integrity_path = Path(args.integrity_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    integrity_path.parent.mkdir(parents=True, exist_ok=True)

    if not results_frame.empty:
        results_frame = results_frame.sort_values("model_label").reset_index(drop=True)
    results_frame.to_csv(summary_path, index=False)
    integrity_output.to_csv(integrity_path, index=False)
    report_path.write_text(
        build_report(
            results_frame=results_frame,
            missing_run_dirs=missing_run_dirs,
            requested_run_count=len(requested_run_dirs),
            selected_name=selected_spec.name,
            step_cost=step_cost,
            protocol_name=args.protocol_name,
        ),
        encoding="utf-8",
    )

    print(f"Selected Algorithm X intake: {selected_spec.name}")
    print(f"Completed frontier run directories: {len(existing_run_dirs)}")
    if missing_run_dirs:
        print("Missing run directories:")
        for missing in missing_run_dirs:
            print(f"- {missing}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote summary CSV: {summary_path}")
    print(f"Wrote integrity CSV: {integrity_path}")


if __name__ == "__main__":
    main()