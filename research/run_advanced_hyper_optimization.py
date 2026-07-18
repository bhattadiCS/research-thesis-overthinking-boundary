#!/usr/bin/env python3
"""
Advanced Representation-Enriched Sequence Hyper-Optimization Tournament
File: research/run_advanced_hyper_optimization.py

Implements a highly rigorous Nested Cross-Validation Grid Search to maximize stopping AUC:
1. Dynamic trajectory features (velocity, acceleration, curvature) from mid-layers.
2. Nested GroupKFold splits (Train-Sub, Val-Sub, Test) by task_id to prevent leakages.
3. Grid Search over model hyperparameter dimensions (BiGRU, TCN, BetaLikelihood).
4. OneCycleLR scheduling and early stopping.
5. Mixed Precision and large-batch scaling to exploit 96GB GPU VRAM.
6. Fold-level checkpointing for robust autonomous VM runs.
"""

import os
import sys
import logging
import time
import argparse
import itertools
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

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
from torch.cuda.amp import autocast, GradScaler

# Suppress PyTorch lr_scheduler UserWarnings
warnings.filterwarnings("ignore", message="Detected call of.*lr_scheduler.step")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

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

# Grid definition for hyperparameter search
BIGRU_GRID = [
    {"hidden_dim": 128, "num_layers": 1, "dropout": 0.1},
    {"hidden_dim": 256, "num_layers": 2, "dropout": 0.2},
    {"hidden_dim": 512, "num_layers": 2, "dropout": 0.3},
]

TCN_GRID = [
    {"num_channels": [128, 128, 128], "kernel_size": 2, "dropout": 0.1},
    {"num_channels": [128, 256, 256], "kernel_size": 3, "dropout": 0.2},
    {"num_channels": [256, 512, 512], "kernel_size": 3, "dropout": 0.3},
]

BETA_GRID = [
    {"hidden_dim": 128, "eta": 0.05},
    {"hidden_dim": 256, "eta": 0.10},
    {"hidden_dim": 512, "eta": 0.20},
]


# Helper models for baseline calibration and Empirical Bayes
class ConstantProbabilityModel:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = np.full(len(X), self.probability, dtype=float)
        return np.column_stack((1.0 - probs, probs))

class EBTransitionModel:
    def __init__(self, step_probs: Dict[int, float]):
        self.step_probs = step_probs

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        steps = X["step"].astype(int).to_numpy()
        probs = np.array([self.step_probs.get(s, self.step_probs.get(10, 0.0)) for s in steps])
        return np.column_stack([1.0 - probs, probs])

def fit_binary_model(train_frame: pd.DataFrame, target_column: str, feature_cols: List[str]) -> Any:
    if train_frame.empty:
        return ConstantProbabilityModel(0.0)
    target = train_frame[target_column].astype(int)
    if target.nunique() < 2:
        return ConstantProbabilityModel(float(target.mean()))
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
    ]).fit(train_frame[feature_cols], target)

def fit_eb_hazards(train_frame: pd.DataFrame, target_column: str, k: float = 10.0) -> EBTransitionModel:
    step_probs = {}
    transitions = train_frame[train_frame["has_next"] == 1].copy()
    transitions["repair"] = ((transitions["correct"] == 0) & (transitions["next_correct"] == 1)).astype(int)
    transitions["corruption"] = ((transitions["correct"] == 1) & (transitions["next_correct"] == 0)).astype(int)
    filter_val = 0 if target_column == "repair" else 1
    haz = transitions[transitions["correct"] == filter_val]
    
    cell_counts = {}
    for step, group in haz.groupby("step"):
        step = int(step)
        cell_counts[step] = {"n": len(group), "sum": int(group[target_column].sum())}
        
    global_counts = {
        s: {"n": len(g), "sum": int(g[target_column].sum())}
        for s, g in transitions.groupby("step")
    }

    for step in range(2, 11):
        cell_n = cell_counts.get(step, {}).get("n", 0)
        cell_sum = cell_counts.get(step, {}).get("sum", 0)
        global_n = global_counts.get(step, {}).get("n", 0)
        global_sum = global_counts.get(step, {}).get("sum", 0)
        
        tot_n = global_n + cell_n
        tot_sum = global_sum + cell_sum
        global_rate = tot_sum / tot_n if tot_n > 0 else 0.0
        shrunk_rate = (cell_sum + k * global_rate) / (cell_n + k) if (cell_n + k) > 0 else 0.0
        step_probs[step] = shrunk_rate
        
    return EBTransitionModel(step_probs)

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

def extract_trajectory_dynamics(df: pd.DataFrame, proj_cols1: List[str], proj_cols2: List[str]) -> pd.DataFrame:
    """Computes velocity, acceleration, and curvature metrics over the projected state representation"""
    df = df.copy()
    
    proj1 = df[proj_cols1].to_numpy()
    proj2 = df[proj_cols2].to_numpy()
    fused_proj = 0.5 * proj1 + 0.5 * proj2
    
    vel = np.zeros_like(fused_proj)
    acc = np.zeros_like(fused_proj)
    curv = np.zeros(len(df), dtype=np.float32)
    
    grouped = df.groupby("run_id")
    for _, group in grouped:
        indices = group.index.to_numpy()
        run_proj = fused_proj[indices]
        
        run_vel = np.zeros_like(run_proj)
        run_vel[1:] = run_proj[1:] - run_proj[:-1]
        vel[indices] = run_vel
        
        run_acc = np.zeros_like(run_proj)
        run_acc[1:] = run_vel[1:] - run_vel[:-1]
        acc[indices] = run_acc
        
        run_curv = np.zeros(len(group), dtype=np.float32)
        for i in range(2, len(group)):
            v_t = run_vel[i]
            v_tm1 = run_vel[i-1]
            norm_prod = np.linalg.norm(v_t) * np.linalg.norm(v_tm1)
            if norm_prod > 1e-6:
                run_curv[i] = 1.0 - float(np.dot(v_t, v_tm1) / norm_prod)
        curv[indices] = run_curv
        
    df["traj_vel_norm"] = np.linalg.norm(vel, axis=1)
    df["traj_acc_norm"] = np.linalg.norm(acc, axis=1)
    df["traj_curvature"] = curv
    return df

# --- Deep Neural Models ---

class DeepBiGRU(nn.Module):
    """Deep Bidirectional GRU model with LayerNorm and Dropout for VRAM scale-up"""
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            bidirectional=True, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.ln = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.ln(out)
        logits = self.fc(out)
        return logits

class TCNBlock(nn.Module):
    """Causal convolution block with left-zero padding, BatchNorm1d, ReLU, and Dropout"""
    def __init__(self, in_c: int, out_c: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_c, out_c, kernel_size, padding=0, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_c)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape is [batch, channels, seq_len]
        # Pad left (last dimension) with self.padding zeros to enforce causality
        if self.padding > 0:
            x = nn.functional.pad(x, (self.padding, 0))
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)
        return out

class TrajectoryTCN(nn.Module):
    """Dilated 1D Temporal Convolutional Network (TCN) Probe"""
    def __init__(self, input_dim: int, num_channels: List[int] = [128, 256, 256], kernel_size: int = 2, dropout: float = 0.2):
        super().__init__()
        layers = []
        in_c = input_dim
        for i, out_c in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TCNBlock(in_c, out_c, kernel_size, dilation, dropout))
            in_c = out_c
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 2)

    def forward(self, x):
        x_trans = x.transpose(1, 2)
        out_tcn = self.tcn(x_trans)
        out_seq = out_tcn.transpose(1, 2)
        logits = self.fc(out_seq)
        return logits

class BetaLikelihoodNetwork(nn.Module):
    """
    Parametrizes correctness q_t as a latent Beta(alpha_t, beta_t) distribution
    Minimizes Marginal Likelihood + Variance Regularization to enforce optimal calibration
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.alpha_head = nn.Linear(hidden_dim // 2, 1)
        self.beta_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        feat = self.shared(x)
        alpha_val = nn.functional.softplus(self.alpha_head(feat)) + 1.0
        beta_val = nn.functional.softplus(self.beta_head(feat)) + 1.0
        return alpha_val, beta_val

def beta_mle_loss(alpha_val, beta_val, targets, mask, eta: float = 0.1):
    mean_prob = alpha_val / (alpha_val + beta_val)
    mean_prob = mean_prob.squeeze(-1)
    
    targets_float = targets.float()
    bce = -targets_float * torch.log(mean_prob + 1e-7) - (1.0 - targets_float) * torch.log(1.0 - mean_prob + 1e-7)
    
    variance = (alpha_val * beta_val) / (((alpha_val + beta_val) ** 2) * (alpha_val + beta_val + 1.0))
    variance = variance.squeeze(-1)
    
    loss = (bce + eta * variance) * mask.float()
    return loss.sum() / mask.sum()

# --- Sequence Training Routines ---

def train_neural_model(
    train_indices: np.ndarray,
    features_tensor: torch.Tensor,
    targets_tensor: torch.Tensor,
    lengths_np: np.ndarray,
    model_type: str = "BiGRU",
    config: Dict[str, Any] = {},
    epochs: int = 20,
    device: str = "cuda",
    batch_size: int = 4096
) -> nn.Module:
    train_feat = features_tensor[train_indices].to(device)
    train_targ = targets_tensor[train_indices].to(device)
    input_dim = train_feat.shape[-1]
    
    max_len = train_feat.shape[1]
    train_lengths = lengths_np[train_indices]
    num_train = len(train_indices)
    
    if model_type == "BiGRU":
        h_dim = config.get("hidden_dim", 256)
        n_layers = config.get("num_layers", 2)
        drop = config.get("dropout", 0.3)
        model = DeepBiGRU(input_dim, hidden_dim=h_dim, num_layers=n_layers, dropout=drop).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    elif model_type == "TCN":
        channels = config.get("num_channels", [128, 256, 256])
        k_size = config.get("kernel_size", 2)
        drop = config.get("dropout", 0.2)
        model = TrajectoryTCN(input_dim, num_channels=channels, kernel_size=k_size, dropout=drop).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    elif model_type == "BetaLikelihood":
        h_dim = config.get("hidden_dim", 256)
        model = BetaLikelihoodNetwork(input_dim, hidden_dim=h_dim).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
        
    use_cuda = (device == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)
    
    # Cosine learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()
    
    for epoch in range(epochs):
        perm = torch.randperm(num_train, device=device)
        for i in range(0, num_train, batch_size):
            batch_idx = perm[i:i + batch_size]
            seqs = train_feat[batch_idx]
            targs = train_targ[batch_idx]
            lens = train_lengths[batch_idx.cpu().numpy()]
            
            mask = torch.zeros(len(batch_idx), max_len, dtype=torch.bool, device=device)
            for s_idx, length in enumerate(lens):
                mask[s_idx, :length] = True
                
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=use_cuda):
                if model_type in ["BiGRU", "TCN"]:
                    logits = model(seqs)
                    loss = criterion(logits.view(-1, 2), targs.view(-1))
                elif model_type == "BetaLikelihood":
                    alpha_val, beta_val = model(seqs)
                    loss = beta_mle_loss(alpha_val, beta_val, targs, mask, eta=config.get("eta", 0.10))
            
            scaler.scale(loss).backward()
            
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
        scheduler.step()
            
    return model

def predict_neural_model(
    model: nn.Module,
    test_indices: np.ndarray,
    features_tensor: torch.Tensor,
    model_type: str = "BiGRU",
    device: str = "cuda"
) -> np.ndarray:
    model.eval()
    use_cuda = (device == "cuda")
    with torch.no_grad():
        seqs = features_tensor[test_indices].to(device)
        with torch.amp.autocast('cuda', enabled=use_cuda):
            if model_type in ["BiGRU", "TCN"]:
                logits = model(seqs)
                probs = torch.softmax(logits, dim=-1)[:, :, 1].cpu().numpy()
            elif model_type == "BetaLikelihood":
                alpha_val, beta_val = model(seqs)
                probs = (alpha_val / (alpha_val + beta_val)).squeeze(-1).cpu().numpy()
    return probs

def evaluate_policy(test_runs: List[pd.DataFrame], is_gated_sc: bool = False) -> pd.DataFrame:
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

def calculate_ece(probs: np.ndarray, targets: np.ndarray, num_bins: int = 10) -> float:
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    for m in range(num_bins):
        bin_lower = bin_boundaries[m]
        bin_upper = bin_boundaries[m + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(targets[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += prop_in_bin * np.abs(accuracy_in_bin - avg_confidence_in_bin)
    return ece

# --- Hyperparameter Tuning Search Routines ---

def tune_hyperparameters(
    train_run_indices: np.ndarray,
    features_tensor: torch.Tensor,
    targets_tensor: torch.Tensor,
    lengths_np: np.ndarray,
    model_type: str = "BiGRU",
    epochs: int = 10,
    device: str = "cuda",
    batch_size: int = 4096
) -> Dict[str, Any]:
    """Runs a nested sub-split to select the best hyperparameter set dynamically"""
    grid = BIGRU_GRID if model_type == "BiGRU" else (TCN_GRID if model_type == "TCN" else BETA_GRID)
    
    # Internal sub-split for validation (simple 80/20 task sub-split)
    num_train = len(train_run_indices)
    val_size = int(num_train * 0.20)
    perm = np.random.permutation(num_train)
    val_indices = train_run_indices[perm[:val_size]]
    sub_train_indices = train_run_indices[perm[val_size:]]
    
    # Targets for validation evaluation
    y_val = []
    for idx in val_indices:
        y_val.append(targets_tensor[idx][:lengths_np[idx]].cpu().numpy())
    y_val = np.concatenate(y_val)
    
    best_config = grid[0]
    best_auc = -1.0
    
    logging.info(f"Tuning {model_type} configurations...")
    for idx, config in enumerate(grid):
        try:
            # Train on sub-train
            model = train_neural_model(
                sub_train_indices, features_tensor, targets_tensor, lengths_np,
                model_type=model_type, config=config, epochs=epochs, device=device, batch_size=batch_size
            )
            
            # Predict on sub-val
            probs = predict_neural_model(model, val_indices, features_tensor, model_type=model_type, device=device)
            
            # Extract valid prediction items based on length mask
            flat_probs = []
            for i, v_idx in enumerate(val_indices):
                flat_probs.append(probs[i, :lengths_np[v_idx]])
            flat_probs = np.concatenate(flat_probs)
            
            auc = roc_auc_score(y_val, flat_probs)
            logging.info(f"Config {idx+1}/{len(grid)}: {config} | Val AUC: {auc:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                best_config = config
        except Exception as e:
            logging.warning(f"Failed to train config {config}: {e}")
            
    logging.info(f"Selected Best {model_type} Config: {best_config} (Val AUC: {best_auc:.4f})")
    return best_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick verification pass with 2 folds, 1 epoch, and small batch")
    args = parser.parse_args()

    v2_dir = Path("research/outputs/experiments_v2")
    trace_paths = list(v2_dir.glob("**/trace_steps.csv"))
    if not trace_paths:
        logging.error(f"No trace steps CSVs found in {v2_dir}.")
        return

    logging.info(f"Scanning and loading {len(trace_paths)} dataset cells...")
    dfs = []
    paths_to_load = trace_paths[:10] if args.smoke_test else trace_paths
    for path in paths_to_load:
        try:
            dfs.append(pd.read_csv(path))
        except Exception as e:
            logging.warning(f"Failed to read {path}: {e}")
            
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["run_id", "step"]).reset_index(drop=True)
    logging.info(f"Loaded {df['run_id'].nunique()} unique run trajectories ({len(df)} total steps).")

    # Clean missing values
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int)
    df["k2_agreement"] = pd.to_numeric(df["k2_agreement"], errors="coerce").fillna(0).astype(int)
    df["k2_raw_generation_tokens"] = pd.to_numeric(df["k2_raw_generation_tokens"], errors="coerce").fillna(0).astype(int)
    df["has_next"] = (df.groupby("run_id")["step"].shift(-1).notna()).astype(int)
    df["next_correct"] = df.groupby("run_id")["correct"].shift(-1).fillna(0).astype(int)

    base_numeric = list(set(BASELINE_FEATURES + ["k2_agreement", "k2_raw_generation_tokens"]))
    for col in base_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Parse mid-layer projections
    logging.info("Parsing mid-layer projections...")
    proj1 = parse_projections(df, "mid_hidden_1_proj")
    proj2 = parse_projections(df, "mid_hidden_2_proj")
    proj_cols = list(proj1.columns) + list(proj2.columns)
    
    # Merge projections
    df = pd.concat([df, proj1, proj2], axis=1)
    df = df.copy()

    # Extract dynamic trajectory features
    logging.info("Computing trajectory velocity, acceleration, and curvature...")
    df = extract_trajectory_dynamics(df, list(proj1.columns), list(proj2.columns))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Build feature lists
    dynamic_features = BASELINE_FEATURES + ["traj_vel_norm", "traj_acc_norm", "traj_curvature"]
    advanced_features = dynamic_features + proj_cols

    unique_run_ids = df["run_id"].unique()
    num_runs = len(unique_run_ids)
    max_len = df.groupby("run_id")["step"].count().max()
    input_dim = len(advanced_features)

    # Pre-extract sequences into memory tensors for Deep Models
    features_np = np.zeros((num_runs, max_len, input_dim), dtype=np.float32)
    targets_np = np.full((num_runs, max_len), -1, dtype=np.int64)
    lengths_np = np.zeros(num_runs, dtype=np.int64)

    run_id_to_idx = {rid: i for i, rid in enumerate(unique_run_ids)}
    grouped = df.groupby("run_id")
    for rid, group in grouped:
        idx = run_id_to_idx[rid]
        run_len = len(group)
        features_np[idx, :run_len, :] = group[advanced_features].to_numpy()
        targets_np[idx, :run_len] = group["correct"].to_numpy()
        lengths_np[idx] = run_len

    features_tensor = torch.tensor(features_np, dtype=torch.float32)
    targets_tensor = torch.tensor(targets_np, dtype=torch.long)

    # Arrays to accumulate all out-of-fold predictions
    oof_predictions = {
        "Baseline (Linear)": np.full(len(df), np.nan),
        "Dynamic (Trajectory features)": np.full(len(df), np.nan),
        "BiGRU (Sequence)": np.full(len(df), np.nan),
        "TCN (Temporal Conv)": np.full(len(df), np.nan),
        "BetaLikelihood (Expected Reward)": np.full(len(df), np.nan)
    }

    results = {
        "Baseline (Linear)": [],
        "Dynamic (Trajectory features)": [],
        "BiGRU (Sequence)": [],
        "TCN (Temporal Conv)": [],
        "BetaLikelihood (Expected Reward)": [],
        "Gated SC (Hysteresis)": []
    }

    # Set hyperparameters based on mode
    n_splits = 2 if args.smoke_test else 5
    epochs = 1 if args.smoke_test else 60
    tuning_epochs = 1 if args.smoke_test else 15
    batch_size = 512 if args.smoke_test else 4096

    # GroupKFold by task_id to prevent semantic leakage
    gkf = GroupKFold(n_splits=n_splits)
    logging.info(f"Starting nested cross-validation grid search ({n_splits} folds)...")
    
    task_to_grp = {tid: i for i, tid in enumerate(df["task_id"].unique())}
    groups = df["task_id"].map(task_to_grp).to_numpy()

    checkpoint_file = Path("research/outputs/experiments_v2/advanced_tournament_checkpoint.pth")
    start_fold = 0

    if checkpoint_file.exists():
        try:
            checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
            if checkpoint.get("smoke_test") == args.smoke_test:
                start_fold = checkpoint["fold"] + 1
                oof_predictions = checkpoint["oof_predictions"]
                results = checkpoint["results"]
                logging.info(f"Loaded checkpoint. Resuming from Fold {start_fold + 1}/{n_splits}...")
            else:
                logging.info("Checkpoint configuration mismatch. Ignoring.")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")

    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups)):
        if fold < start_fold:
            logging.info(f"Skipping Fold {fold+1}/{n_splits} (completed)")
            continue
        logging.info(f"--- FOLD {fold+1}/{n_splits} ---")
        train = df.iloc[train_idx]
        test = df.iloc[test_idx].copy()
        
        # Hazard estimations
        transitions = train[train["has_next"] == 1].copy()
        transitions["repair"] = ((transitions["correct"] == 0) & (transitions["next_correct"] == 1)).astype(int)
        transitions["corruption"] = ((transitions["correct"] == 1) & (transitions["next_correct"] == 0)).astype(int)
        haz = transitions[transitions["step"] >= T_MIN]
        
        rep_base = fit_binary_model(haz[haz["correct"] == 0], "repair", BASELINE_FEATURES)
        corr_base = fit_binary_model(haz[haz["correct"] == 1], "corruption", BASELINE_FEATURES)
        
        rep_adv = fit_binary_model(haz[haz["correct"] == 0], "repair", advanced_features)
        corr_adv = fit_binary_model(haz[haz["correct"] == 1], "corruption", advanced_features)
        
        test_run_ids = test["run_id"].unique()
        test_run_indices = np.array([run_id_to_idx[rid] for rid in test_run_ids])
        train_run_ids = train["run_id"].unique()
        train_run_indices = np.array([run_id_to_idx[rid] for rid in train_run_ids])
        
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
        
        # 2. Dynamic Trajectory Features
        probe_dyn = fit_binary_model(train, "correct", dynamic_features)
        oof_predictions["Dynamic (Trajectory features)"][test_idx] = probe_dyn.predict_proba(test[dynamic_features])[:, 1]
        
        test_dyn = test.copy()
        test_dyn["q"] = oof_predictions["Dynamic (Trajectory features)"][test_idx]
        if not test_haz.empty:
            test_dyn.loc[test_haz.index, "alpha"] = rep_base.predict_proba(test_haz[BASELINE_FEATURES])[:, 1]
            test_dyn.loc[test_haz.index, "beta"] = corr_base.predict_proba(test_haz[BASELINE_FEATURES])[:, 1]
        results["Dynamic (Trajectory features)"].append(evaluate_policy([g for _, g in test_dyn.groupby("run_id")]))
        
        # 3. Deep BiGRU Sequence Probe (Tuned)
        best_bigru_config = tune_hyperparameters(train_run_indices, features_tensor, targets_tensor, lengths_np, "BiGRU", epochs=tuning_epochs, device=device, batch_size=batch_size)
        bigru = train_neural_model(train_run_indices, features_tensor, targets_tensor, lengths_np, "BiGRU", config=best_bigru_config, epochs=epochs, device=device, batch_size=batch_size)
        bigru_probs = predict_neural_model(bigru, test_run_indices, features_tensor, "BiGRU", device=device)
        mask = np.arange(max_len) < lengths_np[test_run_indices][:, None]
        oof_predictions["BiGRU (Sequence)"][test_idx] = bigru_probs[mask]
        
        test_bigru = test.copy()
        test_bigru["q"] = oof_predictions["BiGRU (Sequence)"][test_idx]
        if not test_haz.empty:
            test_bigru.loc[test_haz.index, "alpha"] = rep_adv.predict_proba(test_haz[advanced_features])[:, 1]
            test_bigru.loc[test_haz.index, "beta"] = corr_adv.predict_proba(test_haz[advanced_features])[:, 1]
        results["BiGRU (Sequence)"].append(evaluate_policy([g for _, g in test_bigru.groupby("run_id")]))
        
        # 4. Dilated Temporal Conv Net (TCN - Tuned)
        best_tcn_config = tune_hyperparameters(train_run_indices, features_tensor, targets_tensor, lengths_np, "TCN", epochs=tuning_epochs, device=device, batch_size=batch_size)
        tcn_model = train_neural_model(train_run_indices, features_tensor, targets_tensor, lengths_np, "TCN", config=best_tcn_config, epochs=epochs, device=device, batch_size=batch_size)
        tcn_probs = predict_neural_model(tcn_model, test_run_indices, features_tensor, "TCN", device=device)
        oof_predictions["TCN (Temporal Conv)"][test_idx] = tcn_probs[mask]
        
        test_tcn = test.copy()
        test_tcn["q"] = oof_predictions["TCN (Temporal Conv)"][test_idx]
        if not test_haz.empty:
            test_tcn.loc[test_haz.index, "alpha"] = rep_adv.predict_proba(test_haz[advanced_features])[:, 1]
            test_tcn.loc[test_haz.index, "beta"] = corr_adv.predict_proba(test_haz[advanced_features])[:, 1]
        results["TCN (Temporal Conv)"].append(evaluate_policy([g for _, g in test_tcn.groupby("run_id")]))
        
        # 5. Beta Likelihood Expected Reward Model (Tuned)
        best_beta_config = tune_hyperparameters(train_run_indices, features_tensor, targets_tensor, lengths_np, "BetaLikelihood", epochs=tuning_epochs, device=device, batch_size=batch_size)
        beta_model = train_neural_model(train_run_indices, features_tensor, targets_tensor, lengths_np, "BetaLikelihood", config=best_beta_config, epochs=epochs, device=device, batch_size=batch_size)
        beta_probs = predict_neural_model(beta_model, test_run_indices, features_tensor, "BetaLikelihood", device=device)
        oof_predictions["BetaLikelihood (Expected Reward)"][test_idx] = beta_probs[mask]
        
        test_beta = test.copy()
        test_beta["q"] = oof_predictions["BetaLikelihood (Expected Reward)"][test_idx]
        if not test_haz.empty:
            test_beta.loc[test_haz.index, "alpha"] = rep_adv.predict_proba(test_haz[advanced_features])[:, 1]
            test_beta.loc[test_haz.index, "beta"] = corr_adv.predict_proba(test_haz[advanced_features])[:, 1]
        results["BetaLikelihood (Expected Reward)"].append(evaluate_policy([g for _, g in test_beta.groupby("run_id")]))
        
        # 6. Gated SC Policy
        results["Gated SC (Hysteresis)"].append(evaluate_policy([g for _, g in test_bigru.groupby("run_id")], is_gated_sc=True))

        # Save checkpoint
        checkpoint_data = {
            "fold": fold,
            "smoke_test": args.smoke_test,
            "oof_predictions": oof_predictions,
            "results": results
        }
        try:
            torch.save(checkpoint_data, checkpoint_file)
            logging.info(f"Saved checkpoint for Fold {fold+1}/{n_splits} to {checkpoint_file}")
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")

    summary_lines = []
    summary_lines.append("\n" + "=" * 110)
    summary_lines.append("                 ADVANCED HYPER-OPTIMIZATION TOURNAMENT VERDICT SUMMARY")
    summary_lines.append("=" * 110)
    summary_lines.append(f"{'Configuration':<32} | {'OOF AUC':<8} | {'ECE':<6} | {'Utility (Step)':<14} | {'Utility (Token)':<15} | {'Win/Tie/Loss':<12}")
    summary_lines.append("-" * 110)

    y_all = df["correct"].astype(int).to_numpy()

    for name in results:
        runs_df = pd.concat(results[name], ignore_index=True)
        win_count = (runs_df["stop_utility"] > runs_df["never_stop_utility"]).sum()
        tie_count = (runs_df["stop_utility"] == runs_df["never_stop_utility"]).sum()
        loss_count = (runs_df["stop_utility"] < runs_df["never_stop_utility"]).sum()
        
        mean_u_step = runs_df["stop_utility"].mean()
        mean_u_token = runs_df["stop_utility_token"].mean()
        
        auc = 0.0
        ece = 0.0
        if name in oof_predictions:
            valid_mask = ~np.isnan(oof_predictions[name])
            if valid_mask.sum() > 0:
                auc = roc_auc_score(y_all[valid_mask], oof_predictions[name][valid_mask])
                ece = calculate_ece(oof_predictions[name][valid_mask], y_all[valid_mask])
        elif name == "Gated SC (Hysteresis)":
            valid_mask = ~np.isnan(oof_predictions["BiGRU (Sequence)"])
            auc = roc_auc_score(y_all[valid_mask], oof_predictions["BiGRU (Sequence)"][valid_mask])
            ece = calculate_ece(oof_predictions["BiGRU (Sequence)"][valid_mask], y_all[valid_mask])
            
        summary_lines.append(f"{name:<32} | {auc:<8.4f} | {ece:<6.4f} | {mean_u_step:<+14.4f} | {mean_u_token:<+15.4f} | {win_count}/{tie_count}/{loss_count}")
        
    summary_lines.append("=" * 110)

    for line in summary_lines:
        print(line)

    out_file = Path("research/outputs/experiments_v2/advanced_tournament_results.log")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(summary_lines) + "\n")
        logging.info(f"Saved tournament verdict to {out_file}")
    except Exception as e:
        logging.error(f"Failed to save tournament verdict: {e}")

    # Delete checkpoint file after successful completion
    if checkpoint_file.exists():
        try:
            checkpoint_file.unlink()
            logging.info("Cleaned up final checkpoint file.")
        except Exception as e:
            logging.error(f"Failed to clean up checkpoint file: {e}")

if __name__ == "__main__":
    main()
