from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "real_traces"
STEP_COST = 0.05
FEATURE_COLUMNS = [
    "step",
    "entropy_mean",
    "entropy_std",
    "confidence",
    "answer_changed",
    "thought_token_count",
    "hidden_l2_shift",
    "hidden_cosine_shift",
    "lexical_echo",
    "verbose_confidence_proxy",
]


class ConstantProbabilityModel:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        probabilities = np.full(len(features), self.probability, dtype=float)
        return np.column_stack([1.0 - probabilities, probabilities])


def add_temporal_features(step_frame: pd.DataFrame) -> pd.DataFrame:
    ordered = step_frame.sort_values(["run_id", "step"]).copy()
    ordered["entropy_delta"] = ordered.groupby("run_id")["entropy_mean"].diff().fillna(0.0)
    ordered["proxy_delta"] = ordered.groupby("run_id")["verbose_confidence_proxy"].diff().fillna(0.0)
    ordered["next_correct"] = ordered.groupby("run_id")["correct"].shift(-1)
    ordered["next_utility"] = ordered.groupby("run_id")["utility"].shift(-1)
    ordered["delta_utility"] = ordered["next_utility"] - ordered["utility"]
    ordered["answer_streak"] = (
        ordered.groupby("run_id")["answer_normalized"]
        .transform(lambda values: _answer_streak(values.tolist()))
        .astype(int)
    )
    ordered["has_next"] = ordered["next_correct"].notna().astype(int)
    return ordered


def _answer_streak(values: list[str]) -> list[int]:
    streaks: list[int] = []
    current = 0
    previous = None
    for value in values:
        if value and value == previous:
            current += 1
        else:
            current = 1
        streaks.append(current)
        previous = value
    return streaks


def fit_binary_model(train_frame: pd.DataFrame, target_column: str) -> Any:
    target = train_frame[target_column].astype(int)
    if target.nunique() < 2:
        return ConstantProbabilityModel(float(target.mean()))
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
            ),
        ]
    ).fit(train_frame[FEATURE_COLUMNS], target)


def first_step_matching(group: pd.DataFrame, mask: pd.Series) -> int:
    candidates = group.loc[mask]
    if len(candidates):
        return int(candidates.iloc[0]["step"])
    return int(group.iloc[-1]["step"])


def utility_at_stop(group: pd.DataFrame, step: int) -> float:
    return float(group.loc[group["step"] == step, "utility"].iloc[0])


def evaluate_detector_on_group(group: pd.DataFrame, detector_name: str, step: int) -> dict[str, Any]:
    oracle_utility = float(group["utility"].max())
    oracle_step = int(group.loc[group["utility"].idxmax(), "step"])
    stop_utility = utility_at_stop(group, step)
    return {
        "run_id": group.iloc[0]["run_id"],
        "task_id": group.iloc[0]["task_id"],
        "detector": detector_name,
        "stop_step": step,
        "oracle_step": oracle_step,
        "oracle_gap": oracle_utility - stop_utility,
        "false_early": int(step < oracle_step),
        "false_late": int(step > oracle_step),
        "false_late_severity": max(step - oracle_step, 0),
    }


def select_entropy_threshold(train_frame: pd.DataFrame) -> float:
    candidates = np.quantile(train_frame["entropy_mean"].dropna(), [0.2, 0.35, 0.5, 0.65, 0.8])
    best_threshold = float(candidates[0])
    best_gap = math.inf
    for threshold in candidates:
        detector_rows: list[dict[str, Any]] = []
        for _, group in train_frame.groupby("run_id"):
            step = first_step_matching(group, (group["step"] >= 2) & (group["entropy_mean"] <= threshold) & (group["answer_streak"] >= 2))
            detector_rows.append(evaluate_detector_on_group(group, detector_name="entropy_plateau", step=step))
        mean_gap = float(pd.DataFrame(detector_rows)["oracle_gap"].mean())
        if mean_gap < best_gap:
            best_gap = mean_gap
            best_threshold = float(threshold)
    return best_threshold


def hazard_stop_for_group(group: pd.DataFrame, train_frame: pd.DataFrame) -> int:
    train_rows = train_frame[train_frame["has_next"] == 1].copy()
    train_rows["repair"] = ((train_rows["correct"] == 0) & (train_rows["next_correct"] == 1)).astype(int)
    train_rows["corruption"] = ((train_rows["correct"] == 1) & (train_rows["next_correct"] == 0)).astype(int)

    repair_model = fit_binary_model(train_rows[train_rows["correct"] == 0], target_column="repair")
    corruption_model = fit_binary_model(train_rows[train_rows["correct"] == 1], target_column="corruption")

    q_model = fit_binary_model(train_frame, target_column="correct")
    scoring = group.copy()
    q_hat = q_model.predict_proba(scoring[FEATURE_COLUMNS])[:, 1]
    alpha_hat = repair_model.predict_proba(scoring[FEATURE_COLUMNS])[:, 1]
    beta_hat = corruption_model.predict_proba(scoring[FEATURE_COLUMNS])[:, 1]
    scoring["mu_hat"] = (1.0 - q_hat) * alpha_hat - q_hat * beta_hat - STEP_COST
    candidate = scoring.loc[(scoring["step"] >= 2) & (scoring["mu_hat"] <= 0.0)]
    if len(candidate):
        return int(candidate.iloc[0]["step"])
    return int(scoring.iloc[-1]["step"])


def build_detector_frame(step_frame: pd.DataFrame) -> pd.DataFrame:
    detector_rows: list[dict[str, Any]] = []
    for run_id, group in step_frame.groupby("run_id"):
        holdout = group.sort_values("step")
        train_frame = step_frame[step_frame["run_id"] != run_id]
        entropy_threshold = select_entropy_threshold(train_frame)

        detector_rows.append(
            evaluate_detector_on_group(
                holdout,
                detector_name="first_answer",
                step=first_step_matching(holdout, holdout["answer_normalized"] != ""),
            )
        )
        detector_rows.append(
            evaluate_detector_on_group(
                holdout,
                detector_name="verifier_first_correct",
                step=first_step_matching(holdout, holdout["correct"] == 1),
            )
        )
        detector_rows.append(
            evaluate_detector_on_group(
                holdout,
                detector_name="answer_stability",
                step=first_step_matching(holdout, (holdout["step"] >= 2) & (holdout["answer_streak"] >= 2) & (holdout["confidence"] >= 55)),
            )
        )
        detector_rows.append(
            evaluate_detector_on_group(
                holdout,
                detector_name="entropy_plateau",
                step=first_step_matching(holdout, (holdout["step"] >= 2) & (holdout["entropy_mean"] <= entropy_threshold) & (holdout["answer_streak"] >= 2)),
            )
        )
        detector_rows.append(
            evaluate_detector_on_group(
                holdout,
                detector_name="hazard_drift",
                step=hazard_stop_for_group(holdout, train_frame),
            )
        )
    return pd.DataFrame(detector_rows)


def evaluate_correctness_probe(step_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_id, group in step_frame.groupby("run_id"):
        train_frame = step_frame[step_frame["run_id"] != run_id]
        probe = fit_binary_model(train_frame, target_column="correct")
        predictions = probe.predict_proba(group[FEATURE_COLUMNS])[:, 1]
        y_true = group["correct"].astype(int).to_numpy()
        brier = brier_score_loss(y_true, predictions)
        if np.unique(y_true).size < 2:
            auc = float("nan")
        else:
            auc = roc_auc_score(y_true, predictions)
        rows.append({"run_id": run_id, "brier": brier, "auc": auc})
    return pd.DataFrame(rows)


def fit_full_models(step_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_probe = fit_binary_model(step_frame, target_column="correct")
    transition_rows = step_frame[step_frame["has_next"] == 1].copy()
    transition_rows["repair"] = ((transition_rows["correct"] == 0) & (transition_rows["next_correct"] == 1)).astype(int)
    transition_rows["corruption"] = ((transition_rows["correct"] == 1) & (transition_rows["next_correct"] == 0)).astype(int)
    repair_model = fit_binary_model(transition_rows[transition_rows["correct"] == 0], target_column="repair")
    corruption_model = fit_binary_model(transition_rows[transition_rows["correct"] == 1], target_column="corruption")

    weights: list[dict[str, Any]] = []
    for label, model in [
        ("correctness_probe", full_probe),
        ("repair_hazard", repair_model),
        ("corruption_hazard", corruption_model),
    ]:
        if isinstance(model, ConstantProbabilityModel):
            weights.append({"model": label, "feature": "constant_probability", "coefficient": model.probability})
            continue
        coefficients = model.named_steps["model"].coef_[0]
        for feature, coefficient in zip(FEATURE_COLUMNS, coefficients, strict=True):
            weights.append({"model": label, "feature": feature, "coefficient": float(coefficient)})

    by_step = step_frame[step_frame["has_next"] == 1].groupby("step").agg(
        q_t=("correct", "mean"),
        repair_rate=("repair", "mean"),
        corruption_rate=("corruption", "mean"),
        empirical_mu=("delta_utility", "mean"),
        entropy_mean=("entropy_mean", "mean"),
        answer_changed_rate=("answer_changed", "mean"),
        hidden_shift_mean=("hidden_l2_shift", "mean"),
        n_examples=("run_id", "count"),
    )
    by_step = by_step.reset_index()
    by_step["hazard_mu"] = (1.0 - by_step["q_t"]) * by_step["repair_rate"] - by_step["q_t"] * by_step["corruption_rate"] - STEP_COST
    return by_step, pd.DataFrame(weights)


def plot_detector_comparison(detector_frame: pd.DataFrame, output_dir: Path) -> None:
    summary = detector_frame.groupby("detector")["oracle_gap"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(summary.index, summary.values, color=["#0f766e", "#1d4ed8", "#d97706", "#7c3aed", "#b91c1c"])
    ax.set_ylabel("Mean oracle gap")
    ax.set_title("Real-Trace Detector Comparison")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "real_trace_detector_gaps.png", dpi=200)
    plt.close(fig)


def plot_hazard_curves(hazard_frame: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(hazard_frame["step"], hazard_frame["empirical_mu"], label="Empirical continuation utility", color="#0f766e", linewidth=2)
    ax.plot(hazard_frame["step"], hazard_frame["hazard_mu"], label="Hazard decomposition", color="#1d4ed8", linestyle="--", linewidth=2)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Drift / continuation value")
    ax.set_title("Real-Trace Drift Sign by Step")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "real_trace_hazard_curves.png", dpi=200)
    plt.close(fig)


def plot_feature_weights(weight_frame: pd.DataFrame, output_dir: Path) -> None:
    filtered = weight_frame[weight_frame["feature"] != "constant_probability"].copy()
    if filtered.empty:
        return
    filtered["abs_coefficient"] = filtered["coefficient"].abs()
    top = filtered.sort_values("abs_coefficient", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top["model"] + " :: " + top["feature"], top["coefficient"], color="#ea580c")
    ax.set_xlabel("Logistic coefficient")
    ax.set_title("Largest Probe and Hazard Coefficients")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "real_trace_feature_weights.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze small real-trace pilot artifacts.")
    parser.add_argument("--input-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    step_frame = pd.read_csv(input_dir / "trace_steps.csv")
    step_frame = add_temporal_features(step_frame)
    step_frame["repair"] = ((step_frame["correct"] == 0) & (step_frame["next_correct"] == 1)).astype(int)
    step_frame["corruption"] = ((step_frame["correct"] == 1) & (step_frame["next_correct"] == 0)).astype(int)

    detector_frame = build_detector_frame(step_frame)
    detector_summary = detector_frame.groupby("detector").agg(
        mean_stop_step=("stop_step", "mean"),
        mean_oracle_gap=("oracle_gap", "mean"),
        false_early_rate=("false_early", "mean"),
        false_late_rate=("false_late", "mean"),
        mean_false_late_severity=("false_late_severity", "mean"),
    )
    detector_summary = detector_summary.reset_index().sort_values("mean_oracle_gap")

    correctness_probe = evaluate_correctness_probe(step_frame)
    hazard_frame, weight_frame = fit_full_models(step_frame)

    detector_frame.to_csv(input_dir / "detector_comparison_by_run.csv", index=False)
    detector_summary.to_csv(input_dir / "detector_comparison.csv", index=False)
    correctness_probe.to_csv(input_dir / "correctness_probe_metrics.csv", index=False)
    hazard_frame.to_csv(input_dir / "hazard_drift_summary.csv", index=False)
    weight_frame.to_csv(input_dir / "feature_weights.csv", index=False)

    plot_detector_comparison(detector_frame, input_dir)
    plot_hazard_curves(hazard_frame, input_dir)
    plot_feature_weights(weight_frame, input_dir)

    print(
        "correctness probe: "
        f"brier={correctness_probe['brier'].mean():.4f}, "
        f"auc={correctness_probe['auc'].dropna().mean():.4f}"
    )
    for _, row in detector_summary.iterrows():
        print(
            f"{row['detector']}: stop={row['mean_stop_step']:.2f}, "
            f"gap={row['mean_oracle_gap']:.4f}, "
            f"false_early={row['false_early_rate']:.3f}, "
            f"false_late={row['false_late_rate']:.3f}"
        )
    print(f"Wrote analysis artifacts to: {input_dir}")


if __name__ == "__main__":
    main()