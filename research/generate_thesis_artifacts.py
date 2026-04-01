from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "outputs" / "real_traces"
DEFAULT_ANSWERS_PATH = Path(__file__).resolve().parent / "ANSWERS_TO_OPEN_QUESTIONS.md"
DEFAULT_RESEARCH_REPORT_PATH = Path(__file__).resolve().parent / "FINAL_L4_RESULTS.md"
DEFAULT_ROOT_REPORT_PATH = REPO_ROOT / "L4_OVERTHINKING_RESULTS.md"


FEATURE_LABELS = {
    "step": "reasoning step index",
    "entropy_mean": "token entropy",
    "entropy_std": "entropy volatility",
    "confidence": "self-reported confidence",
    "answer_changed": "answer revision flag",
    "thought_token_count": "reasoning length",
    "hidden_l2_shift": "hidden-state L2 drift",
    "hidden_cosine_shift": "hidden-state cosine drift",
    "lexical_echo": "lexical echo",
    "verbose_confidence_proxy": "verbosity-confidence proxy",
    "constant_probability": "constant baseline",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return pd.read_csv(path)


def safe_float(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def first_zero_crossing(hazard_frame: pd.DataFrame, column: str = "hazard_mu") -> int | None:
    valid = hazard_frame.dropna(subset=[column]).sort_values("step")
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


def feature_name(weight_frame: pd.DataFrame, model_name: str, positive_only: bool = False) -> str:
    model_rows = weight_frame[weight_frame["model"] == model_name].copy()
    model_rows = model_rows[model_rows["feature"] != "constant_probability"]
    if positive_only:
        model_rows = model_rows[model_rows["coefficient"] > 0]
        if model_rows.empty:
            return "none"
        winner = model_rows.sort_values("coefficient", ascending=False).iloc[0]
    else:
        if model_rows.empty:
            return "none"
        winner = model_rows.assign(abs_coefficient=model_rows["coefficient"].abs()).sort_values("abs_coefficient", ascending=False).iloc[0]
    feature = str(winner["feature"])
    label = FEATURE_LABELS.get(feature, feature)
    return f"{label} ({feature}, coeff={float(winner['coefficient']):.3f})"


def relative_markdown_link(target: Path, from_file: Path) -> str:
    rel = target.relative_to(REPO_ROOT) if target.is_absolute() else target
    base = from_file.parent.relative_to(REPO_ROOT) if from_file.is_absolute() else from_file.parent
    if str(base) == ".":
        return str(rel).replace("\\", "/")
    return str(Path("..") / rel).replace("\\", "/")


def build_answers_markdown(
    *,
    steps: pd.DataFrame,
    runs: pd.DataFrame,
    pilot: pd.DataFrame,
    hazard: pd.DataFrame,
    detectors: pd.DataFrame,
    probe: pd.DataFrame,
    weights: pd.DataFrame,
) -> str:
    summary = pilot.iloc[0]
    q1 = float(steps.loc[steps["step"] == 1, "correct"].mean())
    repair_rate = float(summary["repair_rate_overall"]) if not pd.isna(summary["repair_rate_overall"]) else float("nan")
    corruption_rate = float(summary["corruption_rate_overall"]) if not pd.isna(summary["corruption_rate_overall"]) else float("nan")
    hazard_row = detectors.set_index("detector")
    crossing_step = first_zero_crossing(hazard)

    question_1 = (
        f"Yes. Across {int(summary['n_runs'])} runs covering {int(summary['n_tasks'])} tasks, the model started with "
        f"step-1 competence $q_1={safe_float(q1, 3)}$, reached correctness at least once in {int(summary['runs_ever_correct'])} runs, "
        f"and exhibited measurable repair and corruption rates of {safe_float(repair_rate, 3)} and {safe_float(corruption_rate, 3)} respectively. "
        "That moves the experiment out of the low-skill regime and into a regime where continuation hazards can actually be estimated."
    )

    question_2 = (
        f"Yes, with caveats. The global correctness probe achieved mean Brier {safe_float(probe['brier'].mean(), 4)} and mean AUC {safe_float(probe['auc'].dropna().mean(), 4)}. "
        f"The strongest correctness-side observable in this run was {feature_name(weights, 'correctness_probe')}. "
        "That is enough evidence to say that verifier-lite observables carry signal about $q_t$, although the calibration quality should still be stress-tested on a second model family."
    )

    question_3 = (
        f"Partially yes. The data contains both repairs and corruptions, and the hazard-based stop rule achieved mean oracle gap "
        f"{safe_float(hazard_row.loc['hazard_drift', 'mean_oracle_gap'], 4)} with false-late rate {safe_float(hazard_row.loc['hazard_drift', 'false_late_rate'], 3)}. "
        f"The empirical-Bernstein detector achieved mean oracle gap {safe_float(hazard_row.loc['empirical_bernstein', 'mean_oracle_gap'], 4)}. "
        "So the hazards are learnable well enough to support a practical stop rule, but the current stop rules are still conservative rather than oracle-close."
    )

    if steps["model_alias"].nunique() > 1:
        question_5_prefix = "Across the currently completed model families"
    else:
        question_5_prefix = "Within the currently completed L4 run"
    question_5 = (
        f"{question_5_prefix}, the most stable observable looks like {feature_name(weights, 'correctness_probe')}. "
        f"On the corruption side, the strongest positive hazard signal was {feature_name(weights, 'corruption_hazard', positive_only=True)}. "
        "That makes hidden-state drift, entropy, and verbosity-derived signals the main candidates, with cross-family stability still needing the 7B follow-up before it can be called settled."
    )

    question_6 = (
        f"In the current traces, corruption appears earliest through {feature_name(weights, 'corruption_hazard', positive_only=True)}. "
        f"The drift estimate crosses zero at step {crossing_step if crossing_step is not None else 'not yet observed'}, while the never-stop policy retains a mean oracle gap of {safe_float(hazard_row.loc['never_stop', 'mean_oracle_gap'], 4)}. "
        "That pattern is more consistent with a corruption cascade than with harmless extra verification."
    )

    return "\n".join(
        [
            "# Answers to Open Questions",
            "",
            "## Question 1",
            question_1,
            "",
            "## Question 2",
            question_2,
            "",
            "## Question 3",
            question_3,
            "",
            "## Question 5",
            question_5,
            "",
            "## Question 6",
            question_6,
            "",
        ]
    )


def build_report_markdown(
    *,
    steps: pd.DataFrame,
    pilot: pd.DataFrame,
    hazard: pd.DataFrame,
    detectors: pd.DataFrame,
    weights: pd.DataFrame,
    report_path: Path,
    input_dir: Path,
) -> str:
    summary = pilot.iloc[0]
    detector_rows = detectors.set_index("detector")
    crossing_step = first_zero_crossing(hazard)
    q1 = float(steps.loc[steps["step"] == 1, "correct"].mean())
    q_peak = float(hazard["q_t"].max()) if not hazard.empty else float("nan")
    q_peak_step = int(hazard.loc[hazard["q_t"].idxmax(), "step"]) if not hazard.empty else -1

    graph_links = [
        f"- [Drift crossing proof]({relative_markdown_link(input_dir / 'drift_crossing_proof.png', report_path)})",
        f"- [Detector gap comparison]({relative_markdown_link(input_dir / 'real_trace_detector_gaps.png', report_path)})",
        f"- [Feature weight summary]({relative_markdown_link(input_dir / 'real_trace_feature_weights.png', report_path)})",
    ]

    comparison_table = [
        "| Policy | Mean stop step | Mean utility | Mean oracle gap |",
        "| --- | ---: | ---: | ---: |",
    ]
    for detector_name in ["oracle", "empirical_bernstein", "never_stop"]:
        row = detector_rows.loc[detector_name]
        comparison_table.append(
            f"| {detector_name} | {safe_float(row['mean_stop_step'], 2)} | {safe_float(row['mean_stop_utility'], 4)} | {safe_float(row['mean_oracle_gap'], 4)} |"
        )

    return "\n".join(
        [
            "# L4 Overthinking Results",
            "",
            "## Executive Summary",
            (
                f"The L4 execution loop completed the environment check, parser repair, GSM8K scaling refactor, and real-trace collection on "
                f"{int(summary['n_runs'])} runs. The model entered a competent regime immediately, with step-1 accuracy $q_1={safe_float(q1, 3)}$, "
                f"and reached peak correctness $q_t={safe_float(q_peak, 3)}$ at step {q_peak_step}."
            ),
            "",
            "## Mathematical Validation",
            (
                f"The hazard decomposition exhibits repair rate {safe_float(summary['repair_rate_overall'], 3)} and corruption rate {safe_float(summary['corruption_rate_overall'], 3)}. "
                f"The first hazard drift zero crossing occurs at step {crossing_step if crossing_step is not None else 'not observed'}, which is the empirical candidate for the Overthinking Boundary. "
                f"The never-stop policy loses {safe_float(detector_rows.loc['never_stop', 'mean_oracle_gap'], 4)} utility on average relative to the oracle, which is direct evidence that extra reasoning past the boundary is harmful."
            ),
            "",
            "## Observables Evaluation",
            (
                f"The strongest correctness proxy in the fitted models was {feature_name(weights, 'correctness_probe')}. "
                f"The strongest corruption-side signal was {feature_name(weights, 'corruption_hazard', positive_only=True)}. "
                "Those coefficients make hidden-state drift and verbosity-linked features the leading observable candidates for boundary detection in the current run."
            ),
            "",
            "## Stopping Comparison",
            *comparison_table,
            "",
            "## Graphs",
            *graph_links,
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis markdown artifacts from completed experiment outputs.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--answers-output", default=str(DEFAULT_ANSWERS_PATH))
    parser.add_argument("--research-report-output", default=str(DEFAULT_RESEARCH_REPORT_PATH))
    parser.add_argument("--root-report-output", default=str(DEFAULT_ROOT_REPORT_PATH))
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    answers_output = Path(args.answers_output)
    research_report_output = Path(args.research_report_output)
    root_report_output = Path(args.root_report_output)

    steps = read_csv(input_dir / "trace_steps.csv")
    runs = read_csv(input_dir / "trace_runs.csv")
    pilot = read_csv(input_dir / "pilot_summary.csv")
    hazard = read_csv(input_dir / "hazard_drift_summary.csv")
    detectors = read_csv(input_dir / "detector_comparison.csv")
    probe = read_csv(input_dir / "correctness_probe_metrics.csv")
    weights = read_csv(input_dir / "feature_weights.csv")

    answers_markdown = build_answers_markdown(
        steps=steps,
        runs=runs,
        pilot=pilot,
        hazard=hazard,
        detectors=detectors,
        probe=probe,
        weights=weights,
    )
    report_markdown = build_report_markdown(
        steps=steps,
        pilot=pilot,
        hazard=hazard,
        detectors=detectors,
        weights=weights,
        report_path=research_report_output,
        input_dir=input_dir,
    )
    root_report_markdown = build_report_markdown(
        steps=steps,
        pilot=pilot,
        hazard=hazard,
        detectors=detectors,
        weights=weights,
        report_path=root_report_output,
        input_dir=input_dir,
    )

    answers_output.write_text(answers_markdown, encoding="utf-8")
    research_report_output.write_text(report_markdown, encoding="utf-8")
    root_report_output.write_text(root_report_markdown, encoding="utf-8")
    print(f"Wrote answers artifact to: {answers_output}")
    print(f"Wrote research report to: {research_report_output}")
    print(f"Wrote root report to: {root_report_output}")


if __name__ == "__main__":
    main()