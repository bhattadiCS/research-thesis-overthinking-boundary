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
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "real_traces"
STEP_COST = 0.05
T_MIN = 2  # earliest admissible stop; step 1 is the forced-commit init, not a decision point
EB_DELTA = 0.05
DELTA_LOWER_BOUND = -(1.0 + STEP_COST)
DELTA_UPPER_BOUND = 1.0 - STEP_COST
E_PROCESS_SCALE = max(abs(DELTA_LOWER_BOUND), abs(DELTA_UPPER_BOUND))
E_PROCESS_LAMBDAS = np.linspace(0.05, 0.95, 19)
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
    if train_frame.empty:
        return ConstantProbabilityModel(0.0)
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


def oracle_stop(group: pd.DataFrame, t_min: int = T_MIN) -> tuple[int, float]:
    # AUDIT FIX: the oracle must respect the same protocol floor (t >= T_min) as
    # every detector. Previously the oracle could "stop" at step 1 (where
    # correct@1 forces utility=1.0 for already-correct traces) while detectors
    # were floored at step 2, baking a structural >=0.05 oracle_gap into every
    # comparison and overstating overthinking cost.
    feasible = group[group["step"] >= t_min]
    if feasible.empty:
        feasible = group
    idx = feasible["utility"].idxmax()
    return int(feasible.loc[idx, "step"]), float(feasible.loc[idx, "utility"])


def evaluate_detector_on_group(group: pd.DataFrame, detector_name: str, step: int) -> dict[str, Any]:
    oracle_step, oracle_utility = oracle_stop(group)
    stop_utility = utility_at_stop(group, step)
    return {
        "run_id": group.iloc[0]["run_id"],
        "task_id": group.iloc[0]["task_id"],
        "detector": detector_name,
        "stop_step": step,
        "stop_utility": stop_utility,
        "oracle_step": oracle_step,
        "oracle_utility": oracle_utility,
        "oracle_gap": oracle_utility - stop_utility,
        "false_early": int(step < oracle_step),
        "false_late": int(step > oracle_step),
        "false_late_severity": max(step - oracle_step, 0),
    }


def predict_probabilities(model: Any, frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.empty((0, 2), dtype=float)
    return model.predict_proba(frame[FEATURE_COLUMNS])


def conditional_probability(successes: int, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return float(successes / denominator)


def select_entropy_threshold(step_frame: pd.DataFrame) -> float:
    candidates = np.quantile(step_frame["entropy_mean"].dropna(), [0.2, 0.35, 0.5, 0.65, 0.8])
    best_threshold = float(candidates[0])
    best_gap = math.inf
    for threshold in candidates:
        detector_rows: list[dict[str, Any]] = []
        for _, group in step_frame.groupby("run_id"):
            ordered = group.sort_values("step")
            step = first_step_matching(
                ordered,
                (ordered["step"] >= 2) & (ordered["entropy_mean"] <= threshold) & (ordered["answer_streak"] >= 2),
            )
            detector_rows.append(evaluate_detector_on_group(ordered, detector_name="entropy_plateau", step=step))
        mean_gap = float(pd.DataFrame(detector_rows)["oracle_gap"].mean())
        if mean_gap < best_gap:
            best_gap = mean_gap
            best_threshold = float(threshold)
    return best_threshold


def hazard_stop_for_group(group: pd.DataFrame, q_model: Any, repair_model: Any, corruption_model: Any) -> int:
    scoring = group.sort_values("step").copy()
    q_hat = predict_probabilities(q_model, scoring)[:, 1]
    alpha_hat = predict_probabilities(repair_model, scoring)[:, 1]
    beta_hat = predict_probabilities(corruption_model, scoring)[:, 1]
    scoring["mu_hat"] = (1.0 - q_hat) * alpha_hat - q_hat * beta_hat - STEP_COST
    candidate = scoring.loc[(scoring["step"] >= 2) & (scoring["mu_hat"] <= 0.0)]
    if len(candidate):
        return int(candidate.iloc[0]["step"])
    return int(scoring.iloc[-1]["step"])


def mixture_e_process_value(observations: pd.Series | np.ndarray) -> float:
    values = np.asarray(observations, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 1.0

    scaled = np.clip(values / E_PROCESS_SCALE, -1.0, 1.0)
    log_e_values: list[float] = []
    for lam in E_PROCESS_LAMBDAS:
        factors = 1.0 - lam * scaled
        if np.any(factors <= 0.0):
            return float("inf")
        log_e_values.append(float(np.log(factors).sum()))

    log_e_array = np.asarray(log_e_values, dtype=float)
    max_log_e = float(log_e_array.max())
    return float(np.exp(max_log_e) * np.mean(np.exp(log_e_array - max_log_e)))


def fit_global_models(step_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Any, Any, Any]:
    transition_rows = step_frame[step_frame["has_next"] == 1].copy()
    transition_rows["repair"] = ((transition_rows["correct"] == 0) & (transition_rows["next_correct"] == 1)).astype(int)
    transition_rows["corruption"] = ((transition_rows["correct"] == 1) & (transition_rows["next_correct"] == 0)).astype(int)

    correctness_probe = fit_binary_model(step_frame, target_column="correct")
    # AUDIT FIX: fit the repair/corruption hazards on genuine state transitions
    # only (t >= T_MIN). The step 1->2 transition is a forced commit out of the
    # uncommitted init state, not a real repair/corruption, and contaminated
    # alpha/beta/mu_1 (4-59% of transitions across models).
    hazard_train = transition_rows[transition_rows["step"] >= T_MIN]
    repair_model = fit_binary_model(hazard_train[hazard_train["correct"] == 0], target_column="repair")
    corruption_model = fit_binary_model(hazard_train[hazard_train["correct"] == 1], target_column="corruption")

    weights: list[dict[str, Any]] = []
    for label, model in [
        ("correctness_probe", correctness_probe),
        ("repair_hazard", repair_model),
        ("corruption_hazard", corruption_model),
    ]:
        if isinstance(model, ConstantProbabilityModel):
            weights.append({"model": label, "feature": "constant_probability", "coefficient": model.probability})
            continue
        coefficients = model.named_steps["model"].coef_[0]
        for feature, coefficient in zip(FEATURE_COLUMNS, coefficients, strict=True):
            weights.append({"model": label, "feature": feature, "coefficient": float(coefficient)})

    transition_rows["q_hat"] = predict_probabilities(correctness_probe, transition_rows)[:, 1]
    transition_rows["repair_hazard_hat"] = predict_probabilities(repair_model, transition_rows)[:, 1]
    transition_rows["corruption_hazard_hat"] = predict_probabilities(corruption_model, transition_rows)[:, 1]
    transition_rows["fitted_hazard_drift"] = (
        (1.0 - transition_rows["q_hat"]) * transition_rows["repair_hazard_hat"]
        - transition_rows["q_hat"] * transition_rows["corruption_hazard_hat"]
        - STEP_COST
    )

    rows: list[dict[str, Any]] = []
    for step, group in transition_rows.groupby("step"):
        q_t = float(group["correct"].mean())
        n_transitions = int(len(group))
        n_correct_states = int(group["correct"].sum())
        n_incorrect_states = int(n_transitions - n_correct_states)
        n_repairs = int(group["repair"].sum())
        n_corruptions = int(group["corruption"].sum())

        pooled_repair_frequency = float(group["repair"].mean())
        pooled_corruption_frequency = float(group["corruption"].mean())
        conditional_repair_hazard = conditional_probability(n_repairs, n_incorrect_states)
        conditional_corruption_hazard = conditional_probability(n_corruptions, n_correct_states)
        empirical_utility_drift = float(group["delta_utility"].mean())
        conditional_hazard_drift = (
            (1.0 - q_t) * conditional_repair_hazard - q_t * conditional_corruption_hazard - STEP_COST
            if not math.isnan(conditional_repair_hazard) and not math.isnan(conditional_corruption_hazard)
            else float("nan")
        )
        # AUDIT FIX: pooled (marginal) frequencies already equal (1-q)*alpha and
        # q*beta, so the proxy drift is prf - pcf - lambda. The previous form
        # multiplied by (1-q)/q a second time -> (1-q)^2 alpha - q^2 beta, which
        # has no theoretical meaning.
        pooled_proxy_drift = pooled_repair_frequency - pooled_corruption_frequency - STEP_COST

        rows.append(
            {
                "step": int(step),
                "q_t": q_t,
                "n_transitions": n_transitions,
                "n_correct_states": n_correct_states,
                "n_incorrect_states": n_incorrect_states,
                "n_repairs": n_repairs,
                "n_corruptions": n_corruptions,
                "pooled_repair_frequency": pooled_repair_frequency,
                "pooled_corruption_frequency": pooled_corruption_frequency,
                "pooled_proxy_drift": pooled_proxy_drift,
                "repair_rate": conditional_repair_hazard,
                "corruption_rate": conditional_corruption_hazard,
                "empirical_utility_drift": empirical_utility_drift,
                "conditional_hazard_drift": conditional_hazard_drift,
                "hazard_mu": conditional_hazard_drift,
                "empirical_mu": empirical_utility_drift,
                "empirical_variance": float(group["delta_utility"].var(ddof=0)),
                "conditional_empirical_gap": conditional_hazard_drift - empirical_utility_drift,
                "fitted_q_t": float(group["q_hat"].mean()),
                "fitted_repair_hazard": float(group["repair_hazard_hat"].mean()),
                "fitted_corruption_hazard": float(group["corruption_hazard_hat"].mean()),
                "fitted_hazard_drift": float(group["fitted_hazard_drift"].mean()),
                "entropy_mean": float(group["entropy_mean"].mean()),
                "answer_changed_rate": float(group["answer_changed"].mean()),
                "hidden_shift_mean": float(group["hidden_l2_shift"].mean()),
                "confidence_mean": float(group["confidence"].mean()),
                "n_examples": n_transitions,
            }
        )

    by_step = pd.DataFrame(rows)
    e_process_frame = (
        transition_rows.groupby("step")["delta_utility"]
        .apply(mixture_e_process_value)
        .rename("e_process_value")
        .reset_index()
    )
    by_step = by_step.merge(e_process_frame, on="step", how="left")
    by_step["empirical_variance"] = by_step["empirical_variance"].fillna(0.0)
    by_step["delta_t"] = 6.0 * EB_DELTA / (np.pi**2 * np.square(by_step["step"]))
    by_step["eb_upper_bound"] = (
        by_step["empirical_utility_drift"]
        + np.sqrt(2.0 * by_step["empirical_variance"] * np.log(3.0 / by_step["delta_t"]) / by_step["n_examples"])
        + 3.0 * (DELTA_UPPER_BOUND - DELTA_LOWER_BOUND) * np.log(3.0 / by_step["delta_t"]) / by_step["n_examples"]
    )
    by_step["eb_stop_signal"] = (by_step["eb_upper_bound"] <= 0.0).astype(int)
    by_step["e_process_threshold"] = 1.0 / by_step["delta_t"]
    by_step["e_process_stop_signal"] = (by_step["e_process_value"] >= by_step["e_process_threshold"]).astype(int)
    return by_step, pd.DataFrame(weights), correctness_probe, repair_model, corruption_model


def corrected_drift_column(hazard_frame: pd.DataFrame) -> str:
    if "conditional_hazard_drift" in hazard_frame.columns:
        return "conditional_hazard_drift"
    return "hazard_mu"


def first_zero_crossing(hazard_frame: pd.DataFrame, column: str, t_min: int = T_MIN) -> int:
    # AUDIT FIX: implement the stated optimal-stop rule T* = inf{t >= T_min : col(t) <= 0},
    # else the last step. The previous version required a strict positive->nonpositive
    # transition and had no T_min floor, so it returned the theory-forbidden step 1
    # for models whose drift starts non-positive (4/6 shipped models). This now
    # matches hazard_stop_for_group.
    valid = hazard_frame.dropna(subset=[column]).sort_values("step")
    if valid.empty:
        return t_min
    eligible = valid[valid["step"] >= t_min]
    if eligible.empty:
        return int(valid.iloc[-1]["step"])
    crossed = eligible[eligible[column] <= 0.0]
    if len(crossed):
        return int(crossed.iloc[0]["step"])
    return int(valid.iloc[-1]["step"])


def build_detector_frame(
    step_frame: pd.DataFrame,
    q_model: Any,
    repair_model: Any,
    corruption_model: Any,
    entropy_threshold: float,
    eb_stop_step: int,
    e_process_stop_step: int,
) -> pd.DataFrame:
    detector_rows: list[dict[str, Any]] = []
    for _, group in step_frame.groupby("run_id"):
        ordered = group.sort_values("step")
        detector_rows.append(
            evaluate_detector_on_group(
                ordered,
                detector_name="oracle",
                step=oracle_stop(ordered)[0],
            )
        )
        detector_rows.append(
            evaluate_detector_on_group(
                ordered,
                detector_name="first_answer",
                step=first_step_matching(ordered, ordered["answer_normalized"] != ""),
            )
        )
        detector_rows.append(
            evaluate_detector_on_group(
                ordered,
                detector_name="verifier_first_correct",
                step=first_step_matching(ordered, ordered["correct"] == 1),
            )
        )
        detector_rows.append(
            evaluate_detector_on_group(
                ordered,
                detector_name="answer_stability",
                step=first_step_matching(
                    ordered,
                    (ordered["step"] >= 2) & (ordered["answer_streak"] >= 2) & (ordered["confidence"] >= 55),
                ),
            )
        )
        detector_rows.append(
            evaluate_detector_on_group(
                ordered,
                detector_name="entropy_plateau",
                step=first_step_matching(
                    ordered,
                    (ordered["step"] >= 2) & (ordered["entropy_mean"] <= entropy_threshold) & (ordered["answer_streak"] >= 2),
                ),
            )
        )
        detector_rows.append(
            evaluate_detector_on_group(
                ordered,
                detector_name="hazard_drift",
                step=hazard_stop_for_group(ordered, q_model, repair_model, corruption_model),
            )
        )
        detector_rows.append(
            evaluate_detector_on_group(
                ordered,
                detector_name="empirical_bernstein",
                step=min(eb_stop_step, int(ordered.iloc[-1]["step"])),
            )
        )
        detector_rows.append(
            evaluate_detector_on_group(
                ordered,
                detector_name="e_process",
                step=min(e_process_stop_step, int(ordered.iloc[-1]["step"])),
            )
        )
        detector_rows.append(
            evaluate_detector_on_group(
                ordered,
                detector_name="never_stop",
                step=int(ordered.iloc[-1]["step"]),
            )
        )
    return pd.DataFrame(detector_rows)


def evaluate_correctness_probe(step_frame: pd.DataFrame, correctness_probe: Any) -> pd.DataFrame:
    probabilities = predict_probabilities(correctness_probe, step_frame)[:, 1]
    scored = step_frame[["run_id", "correct"]].copy()
    scored["score"] = probabilities
    rows: list[dict[str, Any]] = []
    for run_id, group in scored.groupby("run_id"):
        y_true = group["correct"].astype(int).to_numpy()
        predictions = group["score"].to_numpy()
        brier = brier_score_loss(y_true, predictions)
        if np.unique(y_true).size < 2:
            auc = float("nan")
        else:
            auc = roc_auc_score(y_true, predictions)
        rows.append({"run_id": run_id, "brier": brier, "auc": auc})
    return pd.DataFrame(rows)


def _fit_fold_models(train_frame: pd.DataFrame) -> tuple[Any, Any, Any]:
    transitions = train_frame[train_frame["has_next"] == 1].copy()
    transitions["repair"] = ((transitions["correct"] == 0) & (transitions["next_correct"] == 1)).astype(int)
    transitions["corruption"] = ((transitions["correct"] == 1) & (transitions["next_correct"] == 0)).astype(int)
    haz = transitions[transitions["step"] >= T_MIN]
    probe = fit_binary_model(train_frame, target_column="correct")
    repair = fit_binary_model(haz[haz["correct"] == 0], target_column="repair")
    corruption = fit_binary_model(haz[haz["correct"] == 1], target_column="corruption")
    return probe, repair, corruption


def _probe_metrics(scored_frame: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold probe AUC/Brier: per-run rows plus a pooled sample-weighted row."""
    sf = scored_frame.dropna(subset=["oof_score"])
    rows: list[dict[str, Any]] = []
    for run_id, group in sf.groupby("run_id"):
        y = group["correct"].astype(int).to_numpy()
        p = group["oof_score"].to_numpy()
        auc = roc_auc_score(y, p) if np.unique(y).size >= 2 else float("nan")
        rows.append({"run_id": run_id, "brier": brier_score_loss(y, p), "auc": auc,
                     "n": int(len(y)), "scope": "per_run_oof"})
    y_all = sf["correct"].astype(int).to_numpy()
    p_all = sf["oof_score"].to_numpy()
    pooled_auc = roc_auc_score(y_all, p_all) if np.unique(y_all).size >= 2 else float("nan")
    rows.append({"run_id": "__pooled__", "brier": brier_score_loss(y_all, p_all), "auc": pooled_auc,
                 "n": int(len(y_all)), "scope": "pooled_oof"})
    return pd.DataFrame(rows)


def out_of_sample_eval(step_frame: pd.DataFrame, n_splits: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """AUDIT FIX: honest out-of-sample evaluation via GroupKFold by run_id, so no
    trajectory is scored by a probe / hazard model / entropy threshold / EB /
    e-process step that was fit on it. Returns (out-of-fold detector frame,
    out-of-fold probe metrics). The previous pipeline fit and scored everything
    on the same rows, inflating every learned-detector and probe metric."""
    sf = step_frame.reset_index(drop=True)
    n_groups = sf["run_id"].nunique()

    def _fold_steps(train_haz: pd.DataFrame) -> tuple[int, int]:
        eb = first_step_matching(train_haz, (train_haz["step"] >= T_MIN) & (train_haz["eb_upper_bound"] <= 0.0))
        ep = first_step_matching(train_haz, (train_haz["step"] >= T_MIN) & (train_haz["e_process_stop_signal"] == 1))
        return eb, ep

    if n_groups < 2:  # too few runs to split (pilot); fall back to in-sample, flagged
        probe, repair, corruption = _fit_fold_models(sf)
        train_haz = fit_global_models(sf)[0]
        eb, ep = _fold_steps(train_haz)
        det = build_detector_frame(sf, probe, repair, corruption, select_entropy_threshold(sf), eb, ep)
        det["out_of_sample"] = 0
        sf["oof_score"] = predict_probabilities(probe, sf)[:, 1]
        return det, _probe_metrics(sf)

    n_splits = max(2, min(n_splits, n_groups))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(sf), np.nan)
    det_parts: list[pd.DataFrame] = []
    for train_idx, test_idx in gkf.split(sf, groups=sf["run_id"]):
        train = sf.iloc[train_idx]
        test = sf.iloc[test_idx]
        probe, repair, corruption = _fit_fold_models(train)
        threshold = select_entropy_threshold(train)
        eb, ep = _fold_steps(fit_global_models(train)[0])
        oof[test_idx] = predict_probabilities(probe, test)[:, 1]
        det_parts.append(build_detector_frame(test, probe, repair, corruption, threshold, eb, ep))
    sf["oof_score"] = oof
    det = pd.concat(det_parts, ignore_index=True)
    det["out_of_sample"] = 1
    return det, _probe_metrics(sf)


def summarize_detector_frame(detector_frame: pd.DataFrame) -> pd.DataFrame:
    summary = detector_frame.groupby("detector").agg(
        mean_stop_step=("stop_step", "mean"),
        mean_stop_utility=("stop_utility", "mean"),
        mean_oracle_utility=("oracle_utility", "mean"),
        mean_oracle_gap=("oracle_gap", "mean"),
        false_early_rate=("false_early", "mean"),
        false_late_rate=("false_late", "mean"),
        mean_false_late_severity=("false_late_severity", "mean"),
    )
    return summary.reset_index().sort_values("mean_oracle_gap")


def plot_detector_comparison(detector_summary: pd.DataFrame, output_dir: Path) -> None:
    summary = detector_summary.sort_values("mean_oracle_gap")
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(summary)))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(summary["detector"], summary["mean_oracle_gap"], color=colors)
    ax.set_ylabel("Mean oracle gap")
    ax.set_title("Real-Trace Detector Comparison")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "real_trace_detector_gaps.png", dpi=200)
    plt.close(fig)


def plot_drift_crossing_proof(hazard_frame: pd.DataFrame, output_dir: Path) -> None:
    corrected_column = corrected_drift_column(hazard_frame)
    crossing_step = first_zero_crossing(hazard_frame, corrected_column)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(hazard_frame["step"], hazard_frame["q_t"], color="#1d4ed8", linewidth=2)
    axes[0].set_ylabel("q_t")
    axes[0].set_title("Overthinking Boundary: Belief and Drift")
    axes[0].grid(alpha=0.25)
    axes[0].axvline(crossing_step, color="#dc2626", linestyle="--", linewidth=2)

    axes[1].plot(
        hazard_frame["step"],
        hazard_frame["empirical_utility_drift"],
        label="Empirical utility drift",
        color="#0f766e",
        linewidth=2,
    )
    axes[1].plot(
        hazard_frame["step"],
        hazard_frame[corrected_column],
        label="Conditional hazard drift",
        color="#7c3aed",
        linestyle="--",
        linewidth=2,
    )
    if "pooled_proxy_drift" in hazard_frame.columns:
        axes[1].plot(
            hazard_frame["step"],
            hazard_frame["pooled_proxy_drift"],
            label="Pooled proxy drift",
            color="#b45309",
            linestyle=":",
            linewidth=2,
        )
    if "fitted_hazard_drift" in hazard_frame.columns:
        axes[1].plot(
            hazard_frame["step"],
            hazard_frame["fitted_hazard_drift"],
            label="Fitted hazard drift",
            color="#2563eb",
            linestyle="-.",
            linewidth=2,
        )
    axes[1].plot(hazard_frame["step"], hazard_frame["eb_upper_bound"], label="Empirical-Bernstein upper bound", color="#ea580c", linestyle=":", linewidth=2)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].axvline(crossing_step, color="#dc2626", linestyle="--", linewidth=2, label=f"Corrected boundary step {crossing_step}")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Drift")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "drift_crossing_proof.png", dpi=220)
    fig.savefig(output_dir / "real_trace_hazard_curves.png", dpi=220)
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


def _sanitize_step_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Defend against CSV field-shift from a hard kill mid-write. A SIGKILL (the
    autostart watchdog / -9 on a hung run) can split an append-only row so that
    string values like 'cuda' / 'minimal_json' / a raw JSON blob bleed into numeric
    columns (hidden_l2_shift, utility, seed, ...). Such a row is garbage across every
    column, and the stray strings later blow up arithmetic as 'str' - 'str'. Drop any
    row holding a non-numeric string in a must-be-numeric column, then re-type columns
    that are now cleanly numeric so downstream math works. Returns (clean, n_dropped)."""
    numeric_guard = [c for c in ("step", "correct", "confidence", "hidden_l2_shift",
                                 "hidden_cosine_shift", "hidden_norm", "entropy_mean",
                                 "utility", "elapsed_seconds", "seed") if c in frame.columns]
    # step/correct are the run index and the label: every analyzable row must have
    # both as clean numbers. A field-shift can leave them empty (NaN) rather than a
    # string, so guard them on NaN too (correct.astype(int) later can't take NaN).
    essential = [c for c in ("step", "correct") if c in frame.columns]
    bad = pd.Series(False, index=frame.index)
    for c in numeric_guard:
        coerced = pd.to_numeric(frame[c], errors="coerce")
        bad |= coerced.isna() & frame[c].notna()  # a real string sitting in a numeric column
    for c in essential:
        bad |= pd.to_numeric(frame[c], errors="coerce").isna()  # missing/garbled index or label
    # Drop the WHOLE run for any run touched by corruption: removing single rows would
    # leave step-gaps, but the detector logic assumes contiguous per-run step sequences
    # (utility_at_stop looks up the chosen stop step within the run and would IndexError).
    if bad.any() and "run_id" in frame.columns:
        bad = frame["run_id"].isin(frame.loc[bad, "run_id"].unique())
    n_dropped = int(bad.sum())
    frame = frame.loc[~bad].reset_index(drop=True)
    # Stray strings forced these columns to object dtype; now that the garbage rows
    # are gone, restore numeric dtype for any column whose non-null values all parse.
    for c in frame.columns:
        if frame[c].dtype == object:
            coerced = pd.to_numeric(frame[c], errors="coerce")
            non_null = frame[c].notna().sum()
            if non_null > 0 and coerced.notna().sum() == non_null:
                frame[c] = coerced
    return frame, n_dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze real-trace pilot or full-run artifacts.")
    parser.add_argument("--input-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    step_frame = pd.read_csv(input_dir / "trace_steps.csv")
    step_frame, n_corrupt = _sanitize_step_frame(step_frame)
    if n_corrupt:
        print(f"[analyze] dropped {n_corrupt} CSV-corrupt row(s) (hard-kill field-shift) before analysis.")
    step_frame = add_temporal_features(step_frame)
    step_frame["repair"] = ((step_frame["correct"] == 0) & (step_frame["next_correct"] == 1)).astype(int)
    step_frame["corruption"] = ((step_frame["correct"] == 1) & (step_frame["next_correct"] == 0)).astype(int)

    hazard_frame, weight_frame, correctness_probe, repair_model, corruption_model = fit_global_models(step_frame)
    # AUDIT FIX: honest out-of-sample evaluation (GroupKFold by run_id) for the
    # detector comparison and the probe. The in-sample models above feed only the
    # descriptive drift/boundary curve and feature weights, not any benchmark.
    detector_frame, correctness_probe_frame = out_of_sample_eval(step_frame)
    detector_summary = summarize_detector_frame(detector_frame)

    # Descriptive (in-sample) anytime stop steps, for the drift report/prints only.
    eb_stop_step = first_step_matching(hazard_frame, (hazard_frame["step"] >= T_MIN) & (hazard_frame["eb_upper_bound"] <= 0.0))
    e_process_stop_step = first_step_matching(
        hazard_frame, (hazard_frame["step"] >= T_MIN) & (hazard_frame["e_process_stop_signal"] == 1),
    )
    eb_summary = detector_summary[detector_summary["detector"].isin(["oracle", "empirical_bernstein", "never_stop"])]
    sequential_summary = detector_summary[
        detector_summary["detector"].isin(["oracle", "hazard_drift", "e_process", "empirical_bernstein", "never_stop"])
    ]

    detector_frame.to_csv(input_dir / "detector_comparison_by_run.csv", index=False)
    detector_summary.to_csv(input_dir / "detector_comparison.csv", index=False)
    correctness_probe_frame.to_csv(input_dir / "correctness_probe_metrics.csv", index=False)
    hazard_frame.to_csv(input_dir / "hazard_drift_summary.csv", index=False)
    hazard_frame.to_csv(input_dir / "hazard_decomposition_by_step.csv", index=False)
    weight_frame.to_csv(input_dir / "feature_weights.csv", index=False)
    eb_summary.to_csv(input_dir / "empirical_bernstein_summary.csv", index=False)
    sequential_summary.to_csv(input_dir / "sequential_detector_summary.csv", index=False)

    plot_detector_comparison(detector_summary, input_dir)
    plot_drift_crossing_proof(hazard_frame, input_dir)
    plot_feature_weights(weight_frame, input_dir)

    pooled = (correctness_probe_frame[correctness_probe_frame["scope"] == "pooled_oof"]
              if "scope" in correctness_probe_frame.columns else correctness_probe_frame.iloc[0:0])
    if not pooled.empty:
        print(f"correctness probe (out-of-fold, pooled): "
              f"brier={pooled['brier'].iloc[0]:.4f}, auc={pooled['auc'].iloc[0]:.4f}")
    else:
        print(f"correctness probe (out-of-fold): brier={correctness_probe_frame['brier'].mean():.4f}, "
              f"auc={correctness_probe_frame['auc'].dropna().mean():.4f}")
    for _, row in detector_summary.iterrows():
        print(
            f"{row['detector']}: stop={row['mean_stop_step']:.2f}, "
            f"utility={row['mean_stop_utility']:.4f}, "
            f"gap={row['mean_oracle_gap']:.4f}, "
            f"false_early={row['false_early_rate']:.3f}, "
            f"false_late={row['false_late_rate']:.3f}"
        )
    print(f"E-process stop step: {e_process_stop_step}")
    print(f"Empirical-Bernstein stop step: {eb_stop_step}")
    print(f"Wrote analysis artifacts to: {input_dir}")


if __name__ == "__main__":
    main()