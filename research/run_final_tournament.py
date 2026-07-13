#!/usr/bin/env python3
"""Final Tournament Script:
1. Loads baseline and enriched telemetry features.
2. Trains step-level Logistic Regression models (Baseline and N8b).
3. Trains sequence models (GRU & LSTM) in PyTorch to capture trace-level dynamics.
4. Evaluates the hybrid Gated Self-Consistency Policy.
5. Prints the final OOF expected utility and decision accuracy comparison.
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
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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


# --- PyTorch Sequence Models ---

class SequenceDataset(Dataset):
    def __init__(self, sequences: list[np.ndarray], targets: list[np.ndarray], lengths: list[int]):
        self.sequences = [torch.tensor(s, dtype=torch.float32) for s in sequences]
        self.targets = [torch.tensor(t, dtype=torch.long) for t in targets]
        self.lengths = lengths

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.lengths[idx]


def pad_collate_fn(batch):
    sequences, targets, lengths = zip(*batch)
    max_len = max(lengths)
    
    padded_seqs = torch.zeros(len(sequences), max_len, sequences[0].shape[-1])
    padded_targets = torch.zeros(len(targets), max_len).long()
    
    for i, (seq, target) in enumerate(zip(sequences, targets)):
        padded_seqs[i, :len(seq), :] = seq
        padded_targets[i, :len(target)] = target
        
    return padded_seqs, padded_targets, torch.tensor(lengths)


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
    train_runs: list[pd.DataFrame],
    feature_cols: list[str],
    rnn_type: str = "GRU",
    epochs: int = 20,
    device: str = "cpu"
) -> TrajectoryRNN:
    """Trains a sequence model on run trajectories."""
    sequences = []
    targets = []
    lengths = []
    
    for run in train_runs:
        run = run.sort_values("step")
        seq = run[feature_cols].to_numpy()
        tar = run["correct"].astype(int).to_numpy()
        sequences.append(seq)
        targets.append(tar)
        lengths.append(len(seq))
        
    dataset = SequenceDataset(sequences, targets, lengths)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, collate_fn=pad_collate_fn)
    
    input_dim = len(feature_cols)
    model = TrajectoryRNN(input_dim, rnn_type=rnn_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # ignore padding
    
    model.train()
    for epoch in range(epochs):
        for seqs, targs, _ in dataloader:
            seqs, targs = seqs.to(device), targs.to(device)
            optimizer.zero_grad()
            logits = model(seqs)
            
            # Flatten for cross-entropy
            loss = criterion(logits.view(-1, 2), targs.view(-1))
            loss.backward()
            optimizer.step()
            
    return model


def predict_rnn(model: TrajectoryRNN, runs: list[pd.DataFrame], feature_cols: list[str], device: str = "cpu") -> list[np.ndarray]:
    model.eval()
    predictions = []
    with torch.no_grad():
        for run in runs:
            run = run.sort_values("step")
            seq = torch.tensor(run[feature_cols].to_numpy(), dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(seq).squeeze(0)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            predictions.append(probs)
    return predictions


# --- Evaluation Policies ---

def evaluate_rnn_policy(
    test_runs: list[pd.DataFrame],
    oof_q: list[np.ndarray],
    repair_model: Any,
    corruption_model: Any,
    feature_cols: list[str]
) -> pd.DataFrame:
    """Evaluates stopping policy using sequence model correctness belief."""
    results = []
    for run_idx, run in enumerate(test_runs):
        run = run.sort_values("step").reset_index(drop=True)
        q = oof_q[run_idx]
        
        run_haz = run[run["step"] >= T_MIN]
        if not run_haz.empty:
            alpha = repair_model.predict_proba(run_haz[feature_cols])[:, 1]
            beta = corruption_model.predict_proba(run_haz[feature_cols])[:, 1]
        else:
            alpha = np.zeros(0)
            beta = np.zeros(0)
            
        stop_step = int(run.iloc[-1]["step"])
        for idx in range(len(run)):
            step = int(run.iloc[idx]["step"])
            if step < T_MIN:
                continue
            q_t = q[idx]
            haz_idx = step - T_MIN
            alpha_t = alpha[haz_idx] if haz_idx < len(alpha) else 0.0
            beta_t = beta[haz_idx] if haz_idx < len(beta) else 0.0
            
            mu = (1.0 - q_t) * alpha_t - q_t * beta_t - STEP_COST
            if mu <= 0.0:
                stop_step = step
                break
                
        final_step = int(run.iloc[-1]["step"])
        never_stop_correct = int(run.iloc[-1]["correct"])
        stop_correct = int(run[run["step"] == stop_step].iloc[0]["correct"])
        
        results.append({
            "run_id": run.iloc[0]["run_id"],
            "stop_step": stop_step,
            "never_stop_step": final_step,
            "stop_correct": stop_correct,
            "never_stop_correct": never_stop_correct,
            "stop_utility": stop_correct - STEP_COST * (stop_step - 1),
            "never_stop_utility": never_stop_correct - STEP_COST * (final_step - 1),
            "stop_utility_token": stop_correct - TOKEN_PRICE * run[run["step"] <= stop_step]["thought_token_count"].sum(),
            "never_stop_utility_token": never_stop_correct - TOKEN_PRICE * run["thought_token_count"].sum()
        })
    return pd.DataFrame(results)


def evaluate_gated_sc_policy(
    test_runs: list[pd.DataFrame],
    oof_q: list[np.ndarray],
    repair_model: Any,
    corruption_model: Any,
    feature_cols: list[str]
) -> pd.DataFrame:
    """Evaluates the Gated Self-Consistency Policy."""
    results = []
    for run_idx, run in enumerate(test_runs):
        run = run.sort_values("step").reset_index(drop=True)
        q = oof_q[run_idx]
        
        run_haz = run[run["step"] >= T_MIN]
        if not run_haz.empty:
            alpha = repair_model.predict_proba(run_haz[feature_cols])[:, 1]
            beta = corruption_model.predict_proba(run_haz[feature_cols])[:, 1]
        else:
            alpha = np.zeros(0)
            beta = np.zeros(0)
            
        stop_step = int(run.iloc[-1]["step"])
        sc_triggered_steps = []
        
        for idx in range(len(run)):
            step = int(run.iloc[idx]["step"])
            if step < T_MIN:
                continue
                
            q_t = q[idx]
            haz_idx = step - T_MIN
            alpha_t = alpha[haz_idx] if haz_idx < len(alpha) else 0.0
            beta_t = beta[haz_idx] if haz_idx < len(beta) else 0.0
            
            # Decide stopping base mu
            mu = (1.0 - q_t) * alpha_t - q_t * beta_t - STEP_COST
            
            # Gating Logic: Trigger SC if model is confused (0.1 < q_t < 0.9)
            if 0.10 < q_t < 0.90:
                sc_triggered_steps.append(step)
                k2_agreement = int(run.iloc[idx].get("k2_agreement", 0))
                
                # If agreement is 0, model is unstable => force continuation (do not stop)
                if k2_agreement == 0:
                    continue
                    
            if mu <= 0.0:
                stop_step = step
                break
                
        final_step = int(run.iloc[-1]["step"])
        never_stop_correct = int(run.iloc[-1]["correct"])
        stop_correct = int(run[run["step"] == stop_step].iloc[0]["correct"])
        
        # Cost accounting: charge extra path only on steps where SC was triggered
        tokens_stop = run[run["step"] <= stop_step]["thought_token_count"].sum()
        for s in sc_triggered_steps:
            if s <= stop_step:
                tokens_stop += run[run["step"] == s]["k2_raw_generation_tokens"].sum()
                
        tokens_ns = run["thought_token_count"].sum() + run[run["step"] <= final_step]["k2_raw_generation_tokens"].sum()
        
        u_stop_token = stop_correct - TOKEN_PRICE * tokens_stop
        u_ns_token = never_stop_correct - TOKEN_PRICE * tokens_ns
        
        results.append({
            "run_id": run.iloc[0]["run_id"],
            "stop_step": stop_step,
            "never_stop_step": final_step,
            "stop_correct": stop_correct,
            "never_stop_correct": never_stop_correct,
            "stop_utility": stop_correct - STEP_COST * (stop_step - 1),
            "never_stop_utility": never_stop_correct - STEP_COST * (final_step - 1),
            "stop_utility_token": u_stop_token,
            "never_stop_utility_token": u_ns_token
        })
    return pd.DataFrame(results)


def evaluate_standard_policy(
    test_runs: list[pd.DataFrame],
    probe: Any,
    repair_model: Any,
    corruption_model: Any,
    feature_cols: list[str]
) -> pd.DataFrame:
    """Standard tabular stopping policy baseline simulation."""
    results = []
    for run in test_runs:
        run = run.sort_values("step").reset_index(drop=True)
        q = probe.predict_proba(run[feature_cols])[:, 1]
        
        run_haz = run[run["step"] >= T_MIN]
        if not run_haz.empty:
            alpha = repair_model.predict_proba(run_haz[feature_cols])[:, 1]
            beta = corruption_model.predict_proba(run_haz[feature_cols])[:, 1]
        else:
            alpha = np.zeros(0)
            beta = np.zeros(0)
            
        stop_step = int(run.iloc[-1]["step"])
        for idx in range(len(run)):
            step = int(run.iloc[idx]["step"])
            if step < T_MIN:
                continue
            q_t = q[idx]
            haz_idx = step - T_MIN
            alpha_t = alpha[haz_idx] if haz_idx < len(alpha) else 0.0
            beta_t = beta[haz_idx] if haz_idx < len(beta) else 0.0
            
            mu = (1.0 - q_t) * alpha_t - q_t * beta_t - STEP_COST
            if mu <= 0.0:
                stop_step = step
                break
                
        final_step = int(run.iloc[-1]["step"])
        never_stop_correct = int(run.iloc[-1]["correct"])
        stop_correct = int(run[run["step"] == stop_step].iloc[0]["correct"])
        
        results.append({
            "run_id": run.iloc[0]["run_id"],
            "stop_step": stop_step,
            "never_stop_step": final_step,
            "stop_correct": stop_correct,
            "never_stop_correct": never_stop_correct,
            "stop_utility": stop_correct - STEP_COST * (stop_step - 1),
            "never_stop_utility": never_stop_correct - STEP_COST * (final_step - 1),
            "stop_utility_token": stop_correct - TOKEN_PRICE * run[run["step"] <= stop_step]["thought_token_count"].sum(),
            "never_stop_utility_token": never_stop_correct - TOKEN_PRICE * run["thought_token_count"].sum()
        })
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate advanced stopping tournament.")
    parser.add_argument("--dir", default="research/outputs/experiments_v2", help="Matrix experiments output directory")
    args = parser.parse_args()

    v2_dir = Path(args.dir)
    
    # Locate all collected trace steps (including pilots and final sweep targets)
    trace_paths = list(v2_dir.glob("**/trace_steps.csv"))
    if not trace_paths:
        logging.error(f"No trace steps CSVs found in {v2_dir}.")
        return

    # Load and combine all available trace steps
    logging.info(f"Scanning and loading {len(trace_paths)} dataset cells...")
    dfs = []
    for path in trace_paths:
        try:
            dfs.append(pd.read_csv(path))
        except Exception as e:
            logging.warning(f"Failed to read {path}: {e}")
            
    df = pd.concat(dfs, ignore_index=True)
    logging.info(f"Loaded {df['run_id'].nunique()} unique run trajectories ({len(df)} total steps).")

    # 1. Clean missing values and transitions on base df first (prevents fragmentation warning)
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int)
    df["k2_agreement"] = pd.to_numeric(df["k2_agreement"], errors="coerce").fillna(0).astype(int)
    df["k2_raw_generation_tokens"] = pd.to_numeric(df["k2_raw_generation_tokens"], errors="coerce").fillna(0).astype(int)
    df["has_next"] = (df.groupby("run_id")["step"].shift(-1).notna()).astype(int)
    df["next_correct"] = df.groupby("run_id")["correct"].shift(-1).fillna(0).astype(int)

    # 2. Parse mid-layer projections (N8b)
    logging.info("Parsing mid-layer projections...")
    proj1 = parse_projections(df, "mid_hidden_1_proj")
    proj2 = parse_projections(df, "mid_hidden_2_proj")
    proj_cols = list(proj1.columns) + list(proj2.columns)
    
    # Merge projections back to df
    df = pd.concat([df, proj1, proj2], axis=1)
    df = df.copy()  # Defragment memory

    # Fill NaNs on all features
    all_numeric = list(set(BASELINE_FEATURES + ["k2_agreement"] + 
                           ["answer_span_mean_logprob", "answer_span_min_logprob", 
                            "answer_span_mean_entropy", "answer_span_std_entropy"] + proj_cols))
    for col in all_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device for PyTorch sequence training: {device}")

    # 3. Setup configurations
    n8b_features = BASELINE_FEATURES + proj_cols
    
    # Folds setup
    gkf = GroupKFold(n_splits=5)
    
    # Store fold outputs
    results = {
        "Baseline (Linear)": [],
        "N8b (Linear Proj)": [],
        "GRU (Sequence)": [],
        "LSTM (Sequence)": [],
        "Gated SC (Hysteresis)": []
    }

    logging.info("Starting cross-validation tournament...")
    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df["run_id"])):
        logging.info(f"--- FOLD {fold+1}/5 ---")
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        
        # Base datasets
        test_runs = [run for _, run in test.groupby("run_id")]
        
        # Prepare transitions for hazard estimation
        transitions = train[train["has_next"] == 1].copy()
        transitions["repair"] = ((transitions["correct"] == 0) & (transitions["next_correct"] == 1)).astype(int)
        transitions["corruption"] = ((transitions["correct"] == 1) & (transitions["next_correct"] == 0)).astype(int)
        haz = transitions[transitions["step"] >= T_MIN]
        
        # 1. Baseline Linear
        probe_base = fit_binary_model(train, "correct", BASELINE_FEATURES)
        rep_base = fit_binary_model(haz[haz["correct"] == 0], "repair", BASELINE_FEATURES)
        corr_base = fit_binary_model(haz[haz["correct"] == 1], "corruption", BASELINE_FEATURES)
        results["Baseline (Linear)"].append(
            evaluate_standard_policy(test_runs, probe_base, rep_base, corr_base, BASELINE_FEATURES)
        )
        
        # 2. N8b Linear Proj
        probe_n8b = fit_binary_model(train, "correct", n8b_features)
        rep_n8b = fit_binary_model(haz[haz["correct"] == 0], "repair", n8b_features)
        corr_n8b = fit_binary_model(haz[haz["correct"] == 1], "corruption", n8b_features)
        results["N8b (Linear Proj)"].append(
            evaluate_standard_policy(test_runs, probe_n8b, rep_n8b, corr_n8b, n8b_features)
        )
        
        # 3. PyTorch GRU Model
        train_runs = [run for _, run in train.groupby("run_id")]
        gru_model = train_rnn(train_runs, n8b_features, rnn_type="GRU", epochs=20, device=device)
        gru_oof_q = predict_rnn(gru_model, test_runs, n8b_features, device=device)
        results["GRU (Sequence)"].append(
            evaluate_rnn_policy(test_runs, gru_oof_q, rep_n8b, corr_n8b, n8b_features)
        )
        
        # 4. PyTorch LSTM Model
        lstm_model = train_rnn(train_runs, n8b_features, rnn_type="LSTM", epochs=20, device=device)
        lstm_oof_q = predict_rnn(lstm_model, test_runs, n8b_features, device=device)
        results["LSTM (Sequence)"].append(
            evaluate_rnn_policy(test_runs, lstm_oof_q, rep_n8b, corr_n8b, n8b_features)
        )
        
        # 5. Gated SC Policy (uses GRU correctness scores)
        results["Gated SC (Hysteresis)"].append(
            evaluate_gated_sc_policy(test_runs, gru_oof_q, rep_n8b, corr_n8b, n8b_features)
        )

    # 4. Summarize and Print results
    print("\n" + "=" * 80)
    print("                 FINAL DEEP RESEARCH TOURNAMENT SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<25} | {'OOF AUC':<8} | {'Utility (Step)':<14} | {'Utility (Token)':<15} | {'Win/Tie/Loss':<12}")
    print("-" * 80)

    for name in results:
        runs_df = pd.concat(results[name], ignore_index=True)
        
        win_count = (runs_df["stop_utility"] > runs_df["never_stop_utility"]).sum()
        tie_count = (runs_df["stop_utility"] == runs_df["never_stop_utility"]).sum()
        loss_count = (runs_df["stop_utility"] < runs_df["never_stop_utility"]).sum()
        
        mean_u_step = runs_df["stop_utility"].mean()
        mean_u_token = runs_df["stop_utility_token"].mean()
        
        # Standard accuracy AUC
        # We can extract the corresponding fold scores
        oof_scores = np.full(len(df), np.nan)
        for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df["run_id"])):
            train = df.iloc[train_idx]
            test = df.iloc[test_idx]
            test_runs = [run for _, run in test.groupby("run_id")]
            
            if "Linear" in name:
                cols = BASELINE_FEATURES if "Baseline" in name else n8b_features
                probe = fit_binary_model(train, "correct", cols)
                oof_scores[test_idx] = probe.predict_proba(test[cols])[:, 1]
            elif "SC" in name or "GRU" in name:
                train_runs = [run for _, run in train.groupby("run_id")]
                gru_model = train_rnn(train_runs, n8b_features, rnn_type="GRU", epochs=20, device=device)
                gru_oof_q = predict_rnn(gru_model, test_runs, n8b_features, device=device)
                # Populate oof_scores
                pos = 0
                for q_run in gru_oof_q:
                    run_len = len(q_run)
                    oof_scores[test_idx[pos:pos+run_len]] = q_run
                    pos += run_len
            elif "LSTM" in name:
                train_runs = [run for _, run in train.groupby("run_id")]
                lstm_model = train_rnn(train_runs, n8b_features, rnn_type="LSTM", epochs=20, device=device)
                lstm_oof_q = predict_rnn(lstm_model, test_runs, n8b_features, device=device)
                pos = 0
                for q_run in lstm_oof_q:
                    run_len = len(q_run)
                    oof_scores[test_idx[pos:pos+run_len]] = q_run
                    pos += run_len

        y_all = df["correct"].astype(int).to_numpy()
        auc = roc_auc_score(y_all, oof_scores)
        
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
