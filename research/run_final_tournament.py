#!/usr/bin/env python3
"""Final Tournament Script (Ultimate Blackwell GPU Optimized):
1. Loads baseline and enriched telemetry features.
2. Vectorizes all Scikit-Learn predictions at the fold level.
3. Bypasses PyTorch DataLoader overhead by batch-slicing static GPU tensors directly.
4. Uses vectorized boolean masking to map sequence predictions in milliseconds.
5. Computes win/loss stopping evaluations across all folds in seconds.
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
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Global constraints
T_MIN = 2
STEP_COST = 0.05
AVG_TOKENS_PER_STEP = 250.0
TOKEN_PRICE = STEP_COST / AVG_TOKENS_PER_STEP

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
    if col_name not in frame.columns:
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


class TrajectoryRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, rnn_type: str = "GRU"):
        super().__init__()
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        out, _ = self.rnn(x)
        logits = self.fc(out)
        return logits


def train_rnn(
    train_indices: np.ndarray,
    features_tensor: torch.Tensor,
    targets_tensor: torch.Tensor,
    rnn_type: str = "GRU",
    epochs: int = 20,
    device: str = "cpu"
) -> TrajectoryRNN:
    """Trains a sequence model directly in GPU VRAM without DataLoader overhead."""
    # Slice train subset directly on GPU/CPU
    train_feat = features_tensor[train_indices].to(device)
    train_targ = targets_tensor[train_indices].to(device)
    
    input_dim = train_feat.shape[-1]
    model = TrajectoryRNN(input_dim, rnn_type=rnn_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    num_train = len(train_indices)
    batch_size = 1024
    
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(num_train, device=device)
        for i in range(0, num_train, batch_size):
            batch_idx = perm[i:i + batch_size]
            seqs = train_feat[batch_idx]
            targs = train_targ[batch_idx]
            
            optimizer.zero_grad()
            logits = model(seqs)
            
            loss = criterion(logits.view(-1, 2), targs.view(-1))
            loss.backward()
            optimizer.step()
            
    return model


def predict_rnn(
    model: TrajectoryRNN,
    test_indices: np.ndarray,
    features_tensor: torch.Tensor,
    device: str = "cpu"
) -> np.ndarray:
    """Generates predictions in a single batch on the GPU."""
    model.eval()
    with torch.no_grad():
        seqs = features_tensor[test_indices].to(device)
        logits = model(seqs)
        probs = torch.softmax(logits, dim=-1)[:, :, 1].cpu().numpy()
    return probs


def evaluate_policy(test_runs: list[pd.DataFrame], is_gated_sc: bool = False) -> pd.DataFrame:
    """Vectorized stop step simulation per trajectory."""
    results = []
    for run in test_runs:
        stop_step = int(run.iloc[-1]["step"])
        sc_triggered_steps = []
        
        for idx in range(len(run)):
            row = run.iloc[idx]
            step = int(row["step"])
            if step < T_MIN:
                continue
                
            q_t = float(row["q"])
            alpha_t = float(row["alpha"])
            beta_t = float(row["beta"])
            
            mu = (1.0 - q_t) * alpha_t - q_t * beta_t - STEP_COST
            
            if is_gated_sc:
                if 0.10 < q_t < 0.90:
                    sc_triggered_steps.append(step)
                    k2_agreement = int(row.get("k2_agreement", 0))
                    if k2_agreement == 0:
                        continue
                        
            if mu <= 0.0:
                stop_step = step
                break
                
        final_step = int(run.iloc[-1]["step"])
        never_stop_correct = int(run.iloc[-1]["correct"])
        stop_correct = int(run[run["step"] == stop_step].iloc[0]["correct"])
        
        tokens_stop = run[run["step"] <= stop_step]["thought_token_count"].sum()
        if is_gated_sc:
            for s in sc_triggered_steps:
                if s <= stop_step:
                    tokens_stop += run[run["step"] == s]["k2_raw_generation_tokens"].sum()
                    
        tokens_ns = run["thought_token_count"].sum()
        if is_gated_sc:
            tokens_ns += run["k2_raw_generation_tokens"].sum()
            
        results.append({
            "run_id": run.iloc[0]["run_id"],
            "stop_step": stop_step,
            "never_stop_step": final_step,
            "stop_correct": stop_correct,
            "never_stop_correct": never_stop_correct,
            "stop_utility": stop_correct - STEP_COST * (stop_step - 1),
            "never_stop_utility": never_stop_correct - STEP_COST * (final_step - 1),
            "stop_utility_token": stop_correct - TOKEN_PRICE * tokens_stop,
            "never_stop_utility_token": never_stop_correct - TOKEN_PRICE * tokens_ns
        })
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate advanced stopping tournament.")
    parser.add_argument("--dir", default="research/outputs/experiments_v2", help="Matrix experiments output directory")
    args = parser.parse_args()

    v2_dir = Path(args.dir)
    trace_paths = list(v2_dir.glob("**/trace_steps.csv"))
    if not trace_paths:
        logging.error(f"No trace steps CSVs found in {v2_dir}.")
        return

    logging.info(f"Scanning and loading {len(trace_paths)} dataset cells...")
    dfs = []
    for path in trace_paths:
        try:
            dfs.append(pd.read_csv(path))
        except Exception as e:
            logging.warning(f"Failed to read {path}: {e}")
            
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["run_id", "step"]).reset_index(drop=True)
    logging.info(f"Loaded {df['run_id'].nunique()} unique run trajectories ({len(df)} total steps).")

    # 1. Clean missing values, transitions, and fill NaNs on base df first
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int)
    df["k2_agreement"] = pd.to_numeric(df["k2_agreement"], errors="coerce").fillna(0).astype(int)
    df["k2_raw_generation_tokens"] = pd.to_numeric(df["k2_raw_generation_tokens"], errors="coerce").fillna(0).astype(int)
    df["has_next"] = (df.groupby("run_id")["step"].shift(-1).notna()).astype(int)
    df["next_correct"] = df.groupby("run_id")["correct"].shift(-1).fillna(0).astype(int)

    base_numeric = list(set(BASELINE_FEATURES + ["k2_agreement", "k2_raw_generation_tokens"] + 
                            ["answer_span_mean_logprob", "answer_span_min_logprob", 
                             "answer_span_mean_entropy", "answer_span_std_entropy"]))
    for col in base_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 2. Parse mid-layer projections (N8b)
    logging.info("Parsing mid-layer projections...")
    proj1 = parse_projections(df, "mid_hidden_1_proj")
    proj2 = parse_projections(df, "mid_hidden_2_proj")
    proj_cols = list(proj1.columns) + list(proj2.columns)
    
    # Merge projections back to df
    df = pd.concat([df, proj1, proj2], axis=1)
    df = df.copy()  # Defragment memory

    # 3. Setup device and build static sequences tensors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device for PyTorch sequence training: {device}")

    n8b_features = BASELINE_FEATURES + proj_cols
    unique_run_ids = df["run_id"].unique()
    num_runs = len(unique_run_ids)
    max_len = df.groupby("run_id")["step"].count().max()
    input_dim = len(n8b_features)

    # Pre-extract sequences once into memory tensors
    features_np = np.zeros((num_runs, max_len, input_dim), dtype=np.float32)
    targets_np = np.full((num_runs, max_len), -1, dtype=np.int64)
    lengths_np = np.zeros(num_runs, dtype=np.int64)

    run_id_to_idx = {rid: i for i, rid in enumerate(unique_run_ids)}
    grouped = df.groupby("run_id")
    for rid, group in grouped:
        idx = run_id_to_idx[rid]
        run_len = len(group)
        features_np[idx, :run_len, :] = group[n8b_features].to_numpy()
        targets_np[idx, :run_len] = group["correct"].to_numpy()
        lengths_np[idx] = run_len

    features_tensor = torch.tensor(features_np, dtype=torch.float32)
    targets_tensor = torch.tensor(targets_np, dtype=torch.long)

    # Arrays to accumulate all out-of-fold predictions
    oof_predictions = {
        "Baseline (Linear)": np.full(len(df), np.nan),
        "N8b (Linear Proj)": np.full(len(df), np.nan),
        "GRU (Sequence)": np.full(len(df), np.nan),
        "LSTM (Sequence)": np.full(len(df), np.nan)
    }

    results = {
        "Baseline (Linear)": [],
        "N8b (Linear Proj)": [],
        "GRU (Sequence)": [],
        "LSTM (Sequence)": [],
        "Gated SC (Hysteresis)": []
    }

    gkf = GroupKFold(n_splits=5)
    logging.info("Starting cross-validation tournament...")

    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df["run_id"])):
        logging.info(f"--- FOLD {fold+1}/5 ---")
        train = df.iloc[train_idx]
        test = df.iloc[test_idx].copy()
        
        # Hazard estimations
        transitions = train[train["has_next"] == 1].copy()
        transitions["repair"] = ((transitions["correct"] == 0) & (transitions["next_correct"] == 1)).astype(int)
        transitions["corruption"] = ((transitions["correct"] == 1) & (transitions["next_correct"] == 0)).astype(int)
        haz = transitions[transitions["step"] >= T_MIN]
        
        # Fit models
        rep_base = fit_binary_model(haz[haz["correct"] == 0], "repair", BASELINE_FEATURES)
        corr_base = fit_binary_model(haz[haz["correct"] == 1], "corruption", BASELINE_FEATURES)
        
        rep_n8b = fit_binary_model(haz[haz["correct"] == 0], "repair", n8b_features)
        corr_n8b = fit_binary_model(haz[haz["correct"] == 1], "corruption", n8b_features)
        
        # Identify run index mappings for sequence slicing
        test_run_ids = test["run_id"].unique()
        test_run_indices = np.array([run_id_to_idx[rid] for rid in test_run_ids])
        
        train_run_ids = train["run_id"].unique()
        train_run_indices = np.array([run_id_to_idx[rid] for rid in train_run_ids])
        
        # Vectorized hazard predictions for test fold
        test["alpha"] = 0.0
        test["beta"] = 0.0
        test_haz = test[test["step"] >= T_MIN]
        
        # 1. Baseline Linear
        probe_base = fit_binary_model(train, "correct", BASELINE_FEATURES)
        oof_predictions["Baseline (Linear)"][test_idx] = probe_base.predict_proba(test[BASELINE_FEATURES])[:, 1]
        
        test_b = test.copy()
        test_b["q"] = oof_predictions["Baseline (Linear)"][test_idx]
        if not test_haz.empty:
            test_b.loc[test_haz.index, "alpha"] = rep_base.predict_proba(test_haz[BASELINE_FEATURES])[:, 1]
            test_b.loc[test_haz.index, "beta"] = corr_base.predict_proba(test_haz[BASELINE_FEATURES])[:, 1]
        results["Baseline (Linear)"].append(evaluate_policy([g for _, g in test_b.groupby("run_id")]))
        
        # 2. N8b Linear Proj
        probe_n8b = fit_binary_model(train, "correct", n8b_features)
        oof_predictions["N8b (Linear Proj)"][test_idx] = probe_n8b.predict_proba(test[n8b_features])[:, 1]
        
        test_proj = test.copy()
        test_proj["q"] = oof_predictions["N8b (Linear Proj)"][test_idx]
        if not test_haz.empty:
            test_proj.loc[test_haz.index, "alpha"] = rep_n8b.predict_proba(test_haz[n8b_features])[:, 1]
            test_proj.loc[test_haz.index, "beta"] = corr_n8b.predict_proba(test_haz[n8b_features])[:, 1]
        results["N8b (Linear Proj)"].append(evaluate_policy([g for _, g in test_proj.groupby("run_id")]))
        
        # 3. PyTorch GRU Model
        gru_model = train_rnn(train_run_indices, features_tensor, targets_tensor, rnn_type="GRU", epochs=20, device=device)
        gru_probs = predict_rnn(gru_model, test_run_indices, features_tensor, device=device)
        
        mask = np.arange(max_len) < lengths_np[test_run_indices][:, None]
        oof_predictions["GRU (Sequence)"][test_idx] = gru_probs[mask]
        
        test_gru = test.copy()
        test_gru["q"] = oof_predictions["GRU (Sequence)"][test_idx]
        if not test_haz.empty:
            test_gru.loc[test_haz.index, "alpha"] = rep_n8b.predict_proba(test_haz[n8b_features])[:, 1]
            test_gru.loc[test_haz.index, "beta"] = corr_n8b.predict_proba(test_haz[n8b_features])[:, 1]
        results["GRU (Sequence)"].append(evaluate_policy([g for _, g in test_gru.groupby("run_id")]))
        
        # 4. PyTorch LSTM Model
        lstm_model = train_rnn(train_run_indices, features_tensor, targets_tensor, rnn_type="LSTM", epochs=20, device=device)
        lstm_probs = predict_rnn(lstm_model, test_run_indices, features_tensor, device=device)
        oof_predictions["LSTM (Sequence)"][test_idx] = lstm_probs[mask]
        
        test_lstm = test.copy()
        test_lstm["q"] = oof_predictions["LSTM (Sequence)"][test_idx]
        if not test_haz.empty:
            test_lstm.loc[test_haz.index, "alpha"] = rep_n8b.predict_proba(test_haz[n8b_features])[:, 1]
            test_lstm.loc[test_haz.index, "beta"] = corr_n8b.predict_proba(test_haz[n8b_features])[:, 1]
        results["LSTM (Sequence)"].append(evaluate_policy([g for _, g in test_lstm.groupby("run_id")]))
        
        # 5. Gated SC Policy (uses GRU scores)
        results["Gated SC (Hysteresis)"].append(evaluate_policy([g for _, g in test_gru.groupby("run_id")], is_gated_sc=True))

    # Summarize and Print results
    print("\n" + "=" * 80)
    print("                 FINAL DEEP RESEARCH TOURNAMENT SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<25} | {'OOF AUC':<8} | {'Utility (Step)':<14} | {'Utility (Token)':<15} | {'Win/Tie/Loss':<12}")
    print("-" * 80)

    y_all = df["correct"].astype(int).to_numpy()

    for name in results:
        runs_df = pd.concat(results[name], ignore_index=True)
        
        win_count = (runs_df["stop_utility"] > runs_df["never_stop_utility"]).sum()
        tie_count = (runs_df["stop_utility"] == runs_df["never_stop_utility"]).sum()
        loss_count = (runs_df["stop_utility"] < runs_df["never_stop_utility"]).sum()
        
        mean_u_step = runs_df["stop_utility"].mean()
        mean_u_token = runs_df["stop_utility_token"].mean()
        
        auc_name = "GRU (Sequence)" if "Gated SC" in name else name
        auc = roc_auc_score(y_all, oof_predictions[auc_name])
        
        u_step_str = f"{mean_u_step:+.4f}"
        u_tok_str = f"{mean_u_token:+.4f}"
        wtl_str = f"{win_count}/{tie_count}/{loss_count}"
        
        print(f"{name:<25} | {auc:<8.4f} | {u_step_str:<14} | {u_tok_str:<15} | {wtl_str:<12}")

    print("=" * 80)
    print("Standard step cost uses: correct - 0.05 * (step - 1)")
    print("Honest token cost uses:  correct - 0.0002 * total_tokens (Gated SC charges for k2 only when triggered)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
