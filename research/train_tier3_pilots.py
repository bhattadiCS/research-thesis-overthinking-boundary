#!/usr/bin/env python3
"""Offline training and evaluation script for Tier-3 pilots (N7 & N8).
Compares baseline features with N7 (k=2 self-consistency), N8a (answer-span metrics),
N8b (mid-layer hidden projections), and combined models.
"""
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss
from typing import Any

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Global constraints
T_MIN = 2
STEP_COST = 0.05
# Approximate average tokens per step to scale token costs consistently with step costs
AVG_TOKENS_PER_STEP = 250.0
TOKEN_PRICE = STEP_COST / AVG_TOKENS_PER_STEP  # ~0.0002 per token

BASELINE_FEATURES = [
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
        return np.column_stack((1.0 - probabilities, probabilities))


def fit_binary_model(train_frame: pd.DataFrame, target_column: str, feature_cols: list[str]) -> Any:
    if train_frame.empty:
        return ConstantProbabilityModel(0.0)
    target = train_frame[target_column].astype(int)
    if target.nunique() < 2:
        return ConstantProbabilityModel(float(target.mean()))
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
        ]
    ).fit(train_frame[feature_cols], target)


def parse_projections(frame: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Parses projection strings like '0.123,-0.456,...' into columns."""
    if col_name not in frame.columns:
        # Return a DataFrame of zeros
        zeros = np.zeros((len(frame), 64))
        cols = [f"{col_name}_{i}" for i in range(64)]
        return pd.DataFrame(zeros, columns=cols, index=frame.index)
        
    def parse_row(val):
        if not isinstance(val, str) or not val.strip():
            return [0.0] * 64
        try:
            return [float(x) for x in val.split(',')]
        except Exception:
            return [0.0] * 64
            
    parsed = frame[col_name].apply(parse_row).tolist()
    cols = [f"{col_name}_{i}" for i in range(64)]
    return pd.DataFrame(parsed, columns=cols, index=frame.index)


def evaluate_policy(
    test_runs: list[pd.DataFrame],
    probe: Any,
    repair_model: Any,
    corruption_model: Any,
    feature_cols: list[str],
    is_n7: bool = False
) -> pd.DataFrame:
    """Simulates the stopping policy under GroupKFold testing."""
    results = []
    
    for run in test_runs:
        run = run.sort_values("step").reset_index(drop=True)
        # Predict correctness and hazards
        q = probe.predict_proba(run[feature_cols])[:, 1]
        
        # Hazard models are fit on step >= T_MIN subset, so we pad or guard
        run_haz = run[run["step"] >= T_MIN]
        if not run_haz.empty:
            alpha = repair_model.predict_proba(run_haz[feature_cols])[:, 1]
            beta = corruption_model.predict_proba(run_haz[feature_cols])[:, 1]
        else:
            alpha = np.zeros(0)
            beta = np.zeros(0)
            
        # Simulate stopping
        stop_step = int(run.iloc[-1]["step"])  # default never_stop
        
        for idx in range(len(run)):
            step = int(run.iloc[idx]["step"])
            if step < T_MIN:
                continue
                
            q_t = q[idx]
            # Retrieve hazard predictions for this step
            haz_idx = step - T_MIN
            alpha_t = alpha[haz_idx] if haz_idx < len(alpha) else 0.0
            beta_t = beta[haz_idx] if haz_idx < len(beta) else 0.0
            
            # Continuation drift value
            mu = (1.0 - q_t) * alpha_t - q_t * beta_t - STEP_COST
            if mu <= 0.0:
                stop_step = step
                break
                
        # Calculate utilities
        final_step = int(run.iloc[-1]["step"])
        never_stop_correct = int(run.iloc[-1]["correct"])
        
        stop_row = run[run["step"] == stop_step].iloc[0]
        stop_correct = int(stop_row["correct"])
        
        # Standard step-based utilities
        u_stop = stop_correct - STEP_COST * (stop_step - 1)
        u_ns = never_stop_correct - STEP_COST * (final_step - 1)
        
        # Honest token-cost accounting
        # Cumulative tokens up to stop
        tokens_stop = run[run["step"] <= stop_step]["thought_token_count"].sum()
        tokens_ns = run["thought_token_count"].sum()
        
        if is_n7:
            # Add secondary path tokens
            tokens_stop += run[run["step"] <= stop_step]["k2_raw_generation_tokens"].sum()
            tokens_ns += run["k2_raw_generation_tokens"].sum()
            
        u_stop_token = stop_correct - TOKEN_PRICE * tokens_stop
        u_ns_token = never_stop_correct - TOKEN_PRICE * tokens_ns
        
        # In hindsight, find the optimal oracle stopping step
        oracle_step = T_MIN
        max_oracle_u = -999.0
        for idx in range(len(run)):
            s = int(run.iloc[idx]["step"])
            if s >= T_MIN:
                u = int(run.iloc[idx]["correct"]) - STEP_COST * (s - 1)
                if u > max_oracle_u:
                    max_oracle_u = u
                    oracle_step = s
                    
        results.append({
            "run_id": run.iloc[0]["run_id"],
            "stop_step": stop_step,
            "never_stop_step": final_step,
            "oracle_step": oracle_step,
            "stop_correct": stop_correct,
            "never_stop_correct": never_stop_correct,
            "stop_utility": u_stop,
            "never_stop_utility": u_ns,
            "stop_utility_token": u_stop_token,
            "never_stop_utility_token": u_ns_token,
        })
        
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate stopping rules on Tier-3 pilots.")
    parser.add_argument("--dir", default="research/outputs/experiments_v2", help="Matrix experiments output directory")
    args = parser.parse_args()

    v2_dir = Path(args.dir)
    gsm8k_path = v2_dir / "tier3_pilot_gsm8k" / "trace_steps.csv"
    math_path = v2_dir / "tier3_pilot_math" / "trace_steps.csv"

    if not gsm8k_path.exists() or not math_path.exists():
        logging.error(f"Pilot trace step files not found in {v2_dir}.")
        logging.error("Please run the telemetry collection commands first.")
        return

    # 1. Load and combine the pilot datasets
    logging.info("Loading pilot traces...")
    df_gsm = pd.read_csv(gsm8k_path)
    df_math = pd.read_csv(math_path)
    df = pd.concat([df_gsm, df_math], ignore_index=True)
    logging.info(f"Loaded {len(df)} step-level rows across both pilots.")

    # 2. Parse mid-layer projections (N8b)
    logging.info("Parsing mid-layer projections (N8b)...")
    proj1 = parse_projections(df, "mid_hidden_1_proj")
    proj2 = parse_projections(df, "mid_hidden_2_proj")
    proj_cols = list(proj1.columns) + list(proj2.columns)
    
    # Merge projections back to df
    df = pd.concat([df, proj1, proj2], axis=1)

    # Clean CSV shifts and missing values
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int)
    df["k2_agreement"] = pd.to_numeric(df["k2_agreement"], errors="coerce").fillna(0).astype(int)
    df["k2_raw_generation_tokens"] = pd.to_numeric(df["k2_raw_generation_tokens"], errors="coerce").fillna(0).astype(int)

    # Fill NaNs on all feature columns
    all_numeric_features = list(set(
        BASELINE_FEATURES + ["k2_agreement"] + 
        ["answer_span_mean_logprob", "answer_span_min_logprob", "answer_span_mean_entropy", "answer_span_std_entropy"] +
        proj_cols
    ))
    for col in all_numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Group transitions
    df["has_next"] = (df.groupby("run_id")["step"].shift(-1).notna()).astype(int)
    df["next_correct"] = df.groupby("run_id")["correct"].shift(-1).fillna(0).astype(int)

    # 3. Setup feature sets to compare
    n7_features = BASELINE_FEATURES + ["k2_agreement"]
    n8a_features = BASELINE_FEATURES + [
        "answer_span_mean_logprob",
        "answer_span_min_logprob",
        "answer_span_mean_entropy",
        "answer_span_std_entropy",
    ]
    n8b_features = BASELINE_FEATURES + proj_cols
    combined_features = BASELINE_FEATURES + ["k2_agreement"] + [
        "answer_span_mean_logprob",
        "answer_span_min_logprob",
        "answer_span_mean_entropy",
        "answer_span_std_entropy",
    ] + proj_cols

    feature_sets = {
        "Baseline (10-feat)": (BASELINE_FEATURES, False),
        "N7 (SC Agreement)": (n7_features, True),
        "N8a (Answer-Span)": (n8a_features, False),
        "N8b (Mid-Layer Proj)": (n8b_features, False),
        "Combined (All Enriched)": (combined_features, True),
    }

    # 4. Perform 5-Fold GroupKFold Cross-Validation
    logging.info("Running 5-fold cross-validation on all feature configurations...")
    gkf = GroupKFold(n_splits=5)
    
    # Run level results container
    fold_summaries = {name: [] for name in feature_sets}
    
    for train_idx, test_idx in gkf.split(df, groups=df["run_id"]):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        
        # Extract runs in test set
        test_runs = [run for _, run in test.groupby("run_id")]
        
        for name, (cols, is_n7) in feature_sets.items():
            # Prepare transition markers for hazards
            transitions = train[train["has_next"] == 1].copy()
            transitions["repair"] = ((transitions["correct"] == 0) & (transitions["next_correct"] == 1)).astype(int)
            transitions["corruption"] = ((transitions["correct"] == 1) & (transitions["next_correct"] == 0)).astype(int)
            haz = transitions[transitions["step"] >= T_MIN]
            
            # Fit models
            probe = fit_binary_model(train, "correct", cols)
            repair_model = fit_binary_model(haz[haz["correct"] == 0], "repair", cols)
            corruption_model = fit_binary_model(haz[haz["correct"] == 1], "corruption", cols)
            
            # Evaluate stopping rule OOF
            scored = evaluate_policy(test_runs, probe, repair_model, corruption_model, cols, is_n7)
            fold_summaries[name].append(scored)

    # 5. Summarize and print table
    print("\n" + "=" * 80)
    print("                 TIER-3 PILOTS OFFLINE TOURNAMENT SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<25} | {'OOF AUC':<8} | {'Utility (Step)':<14} | {'Utility (Token)':<15} | {'Win/Tie/Loss':<12}")
    print("-" * 80)

    for name, (cols, is_n7) in feature_sets.items():
        # Compute average correctness AUC across all step rows
        # (Fit global model OOF scores first)
        oof_scores = np.full(len(df), np.nan)
        for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df["run_id"])):
            train = df.iloc[train_idx]
            test = df.iloc[test_idx]
            probe = fit_binary_model(train, "correct", cols)
            oof_scores[test_idx] = probe.predict_proba(test[cols])[:, 1]
            
        y_all = df["correct"].astype(int).to_numpy()
        auc = roc_auc_score(y_all, oof_scores)
        
        # Aggregate run-level metrics
        runs_df = pd.concat(fold_summaries[name], ignore_index=True)
        
        win_count = (runs_df["stop_utility"] > runs_df["never_stop_utility"]).sum()
        tie_count = (runs_df["stop_utility"] == runs_df["never_stop_utility"]).sum()
        loss_count = (runs_df["stop_utility"] < runs_df["never_stop_utility"]).sum()
        
        mean_u_step = runs_df["stop_utility"].mean()
        mean_u_token = runs_df["stop_utility_token"].mean()
        
        # Display results
        u_step_str = f"{mean_u_step:+.4f}"
        u_tok_str = f"{mean_u_token:+.4f}"
        wtl_str = f"{win_count}/{tie_count}/{loss_count}"
        
        print(f"{name:<25} | {auc:<8.4f} | {u_step_str:<14} | {u_tok_str:<15} | {wtl_str:<12}")

    print("=" * 80)
    print("Standard step cost uses: correct - 0.05 * (step - 1)")
    print("Honest token cost uses:  correct - 0.0002 * total_tokens (N7 charges double for k2)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
