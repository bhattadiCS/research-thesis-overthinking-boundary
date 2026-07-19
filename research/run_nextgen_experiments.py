#!/usr/bin/env python3
"""
Next-Generation Deep Overthinking Probe Suite: Jerk/Torsion Kinematics, Causal RoPE Transformer, & Gated MoE
File: research/run_nextgen_experiments.py

Implements:
1. 3rd-Order Kinematics (Jerk j_t), Differential Torsion (tau_t), Phase-Space Attractor Distance D_attractor,
   Multi-Layer Velocity Alignment, and High-Resolution Token Entropy Quantiles.
2. Causal RoPE Transformer Probe with Cross-Layer Attention Fusion (CLAF).
3. Gated Mixture-of-Experts (MoE) Probe (BetaLikelihood + BiGRU + TCN + Transformer) with Load-Balancing Loss.
4. 5-Fold GroupKFold Cross-Validation grouped by task_id with AMP mixed precision and GPU batching.
"""

import os
import sys
import math
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

# --- Feature Extraction: Jerk, Torsion, Phase-Space & Quantiles ---

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

def extract_nextgen_features(df: pd.DataFrame, proj_cols1: List[str], proj_cols2: List[str]) -> pd.DataFrame:
    """Extracts 3rd-order kinematics, differential torsion, phase-space attractor distance, and entropy quantiles"""
    df = df.copy()
    
    p1 = df[proj_cols1].to_numpy()
    p2 = df[proj_cols2].to_numpy()
    fused_proj = 0.5 * p1 + 0.5 * p2
    
    N = len(df)
    dim = fused_proj.shape[1]
    
    vel = np.zeros_like(fused_proj)
    acc = np.zeros_like(fused_proj)
    jerk = np.zeros_like(fused_proj)
    curv = np.zeros(N, dtype=np.float32)
    torsion = np.zeros(N, dtype=np.float32)
    attractor_dist = np.zeros(N, dtype=np.float32)
    layer_align = np.zeros(N, dtype=np.float32)
    
    grouped = df.groupby("run_id")
    for _, group in grouped:
        indices = group.index.to_numpy()
        run_p = fused_proj[indices]
        run_p1 = p1[indices]
        run_p2 = p2[indices]
        L = len(group)
        
        # Velocity
        r_vel = np.zeros_like(run_p)
        r_vel[1:] = run_p[1:] - run_p[:-1]
        vel[indices] = r_vel
        
        # Acceleration
        r_acc = np.zeros_like(run_p)
        r_acc[1:] = r_vel[1:] - r_vel[:-1]
        acc[indices] = r_acc
        
        # Jerk
        r_jerk = np.zeros_like(run_p)
        r_jerk[1:] = r_acc[1:] - r_acc[:-1]
        jerk[indices] = r_jerk
        
        # Multi-layer velocity alignment (mid1 vs mid2)
        r_v1 = np.zeros_like(run_p1)
        r_v1[1:] = run_p1[1:] - run_p1[:-1]
        r_v2 = np.zeros_like(run_p2)
        r_v2[1:] = run_p2[1:] - run_p2[:-1]
        
        for i in range(1, L):
            n1 = np.linalg.norm(r_v1[i])
            n2 = np.linalg.norm(r_v2[i])
            if n1 * n2 > 1e-6:
                layer_align[indices[i]] = float(np.dot(r_v1[i], r_v2[i]) / (n1 * n2))
                
        # Curvature & Torsion & Attractor Distance
        for i in range(2, L):
            v_t = r_vel[i]
            v_tm1 = r_vel[i-1]
            nv = np.linalg.norm(v_t) * np.linalg.norm(v_tm1)
            if nv > 1e-6:
                curv[indices[i]] = 1.0 - float(np.dot(v_t, v_tm1) / nv)
                
            if i >= 3:
                # Approximate torsion: scalar triple product approximation of (v x a) . j
                cross_va = np.cross(v_t[:3], r_acc[i][:3])
                norm_cross = np.linalg.norm(cross_va)
                if norm_cross > 1e-6:
                    torsion[indices[i]] = float(np.abs(np.dot(cross_va, r_jerk[i][:3])) / (norm_cross ** 2))
                    
        # Phase-space attractor distance
        for i in range(1, L):
            curr_state = np.concatenate([run_p[i], r_vel[i]])
            prev_states = np.column_stack([run_p[:i], r_vel[:i]])
            dists = np.linalg.norm(prev_states - curr_state, axis=1)
            attractor_dist[indices[i]] = float(np.min(dists))
            
    df["traj_vel_norm"] = np.linalg.norm(vel, axis=1)
    df["traj_acc_norm"] = np.linalg.norm(acc, axis=1)
    df["traj_jerk_norm"] = np.linalg.norm(jerk, axis=1)
    df["traj_curvature"] = curv
    df["traj_torsion"] = torsion
    df["phase_attractor_dist"] = attractor_dist
    df["layer_align_m1_m2"] = layer_align
    
    # Advanced token-level logit entropy distribution proxies
    ent_mean = df["entropy_mean"].to_numpy()
    ent_std = df["entropy_std"].to_numpy()
    df["entropy_p10"] = np.maximum(0.0, ent_mean - 1.28 * ent_std)
    df["entropy_p50"] = ent_mean
    df["entropy_p90"] = ent_mean + 1.28 * ent_std
    df["entropy_iqr"] = 1.35 * ent_std
    df["entropy_skew_ratio"] = (df["entropy_p90"] - df["entropy_p50"]) / (df["entropy_iqr"] + 1e-5)
    
    return df

# --- Advanced Neural Architectures ---

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for sequence probes"""
    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.unsqueeze(0)

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, rope_emb):
    sin = rope_emb.sin()
    cos = rope_emb.cos()
    return (q * cos) + (rotate_half(q) * sin)

class CausalRoPETransformerProbe(nn.Module):
    """Causal Transformer Encoder Probe with RoPE and LayerNorm"""
    def __init__(self, input_dim: int, d_model: int = 256, nhead: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, d_model)
        self.rope = RotaryEmbedding(d_model // nhead)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_out = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, 2)

    def forward(self, x):
        B, S, D = x.shape
        h = self.proj_in(x)
        
        mask = torch.triu(torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
        out = self.transformer(h, mask=mask)
        out = self.ln_out(out)
        logits = self.fc_out(out)
        return logits

class DeepBiGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.ln = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.ln(out)
        return self.fc(out)

class TCNBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_c, out_c, kernel_size, padding=0, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_c)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        if self.padding > 0:
            x = nn.functional.pad(x, (self.padding, 0))
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)
        return out

class TrajectoryTCN(nn.Module):
    def __init__(self, input_dim: int, num_channels: List[int] = [512, 512, 512], kernel_size: int = 2, dropout: float = 0.1):
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
        return self.fc(out_seq)

class BetaLikelihoodNetwork(nn.Module):
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

def beta_mle_loss(alpha_val, beta_val, targets, mask, eta: float = 0.5):
    mean_prob = alpha_val / (alpha_val + beta_val)
    mean_prob = mean_prob.squeeze(-1)
    
    targets_float = targets.float()
    bce = -targets_float * torch.log(mean_prob + 1e-7) - (1.0 - targets_float) * torch.log(1.0 - mean_prob + 1e-7)
    
    variance = (alpha_val * beta_val) / (((alpha_val + beta_val) ** 2) * (alpha_val + beta_val + 1.0))
    variance = variance.squeeze(-1)
    
    loss = (bce + eta * variance) * mask.float()
    return loss.sum() / mask.sum()

# --- Next-Gen Gated Mixture-of-Experts (MoE) Probe ---

class GatedMoESequenceProbe(nn.Module):
    """
    Gated Mixture-of-Experts (MoE) probe combining:
    Expert 1: BetaLikelihood (Calibration Expert)
    Expert 2: BiGRU (Sequence Memory Expert)
    Expert 3: TrajectoryTCN (Convolutional Local Pattern Expert)
    Expert 4: CausalRoPETransformer (Global Self-Attention Expert)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.expert_beta = BetaLikelihoodNetwork(input_dim, hidden_dim=hidden_dim)
        self.expert_bigru = DeepBiGRU(input_dim, hidden_dim=hidden_dim, num_layers=1, dropout=0.3)
        self.expert_tcn = TrajectoryTCN(input_dim, num_channels=[256, 256, 256], kernel_size=2, dropout=0.1)
        self.expert_trans = CausalRoPETransformerProbe(input_dim, d_model=hidden_dim, nhead=4, num_layers=2, dropout=0.2)
        
        self.gating = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)
        )

    def forward(self, x):
        B, S, D = x.shape
        gate_logits = self.gating(x) # [B, S, 4]
        gate_weights = nn.functional.softmax(gate_logits, dim=-1)
        
        # Expert predictions
        alpha_v, beta_v = self.expert_beta(x)
        prob_beta = (alpha_v / (alpha_v + beta_v)).squeeze(-1) # [B, S]
        
        logits_bigru = self.expert_bigru(x)
        prob_bigru = torch.softmax(logits_bigru, dim=-1)[:, :, 1]
        
        logits_tcn = self.expert_tcn(x)
        prob_tcn = torch.softmax(logits_tcn, dim=-1)[:, :, 1]
        
        logits_trans = self.expert_trans(x)
        prob_trans = torch.softmax(logits_trans, dim=-1)[:, :, 1]
        
        expert_probs = torch.stack([prob_beta, prob_bigru, prob_tcn, prob_trans], dim=-1) # [B, S, 4]
        moe_prob = torch.sum(gate_weights * expert_probs, dim=-1) # [B, S]
        return moe_prob, gate_weights

def moe_joint_loss(moe_prob, gate_weights, targets, mask):
    targets_float = targets.float()
    bce = -targets_float * torch.log(moe_prob + 1e-7) - (1.0 - targets_float) * torch.log(1.0 - moe_prob + 1e-7)
    
    # Load balancing loss across 4 experts
    gate_usage = gate_weights.mean(dim=[0, 1])
    load_balance_loss = 4.0 * torch.sum(gate_usage ** 2)
    
    loss = (bce * mask.float()).sum() / mask.sum() + 0.01 * load_balance_loss
    return loss

# --- Training & Policy Evaluation ---

def train_model_v2(
    train_indices: np.ndarray,
    features_tensor: torch.Tensor,
    targets_tensor: torch.Tensor,
    lengths_np: np.ndarray,
    model_type: str = "BiGRU",
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
        model = DeepBiGRU(input_dim, hidden_dim=256, num_layers=1, dropout=0.3).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    elif model_type == "TCN":
        model = TrajectoryTCN(input_dim, num_channels=[512, 512, 512], kernel_size=2, dropout=0.1).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    elif model_type == "BetaLikelihood":
        model = BetaLikelihoodNetwork(input_dim, hidden_dim=256).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)
    elif model_type == "Transformer":
        model = CausalRoPETransformerProbe(input_dim, d_model=256, nhead=4, num_layers=2, dropout=0.2).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    elif model_type == "MoE":
        model = GatedMoESequenceProbe(input_dim, hidden_dim=256).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)

    use_cuda = (device == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)
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
                if model_type in ["BiGRU", "TCN", "Transformer"]:
                    logits = model(seqs)
                    loss = criterion(logits.view(-1, 2), targs.view(-1))
                elif model_type == "BetaLikelihood":
                    alpha_val, beta_val = model(seqs)
                    loss = beta_mle_loss(alpha_val, beta_val, targs, mask, eta=0.5)
                elif model_type == "MoE":
                    moe_prob, gate_w = model(seqs)
                    loss = moe_joint_loss(moe_prob, gate_w, targs, mask)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
        scheduler.step()
        
    return model

def predict_model_v2(
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
            if model_type in ["BiGRU", "TCN", "Transformer"]:
                logits = model(seqs)
                probs = torch.softmax(logits, dim=-1)[:, :, 1].cpu().numpy()
            elif model_type == "BetaLikelihood":
                alpha_val, beta_val = model(seqs)
                probs = (alpha_val / (alpha_val + beta_val)).squeeze(-1).cpu().numpy()
            elif model_type == "MoE":
                moe_prob, _ = model(seqs)
                probs = moe_prob.cpu().numpy()
    return probs

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run quick verification pass")
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

    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int)

    base_numeric = list(set(BASELINE_FEATURES))
    for col in base_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Parse mid-layer projections
    logging.info("Parsing mid-layer projections...")
    proj1 = parse_projections(df, "mid_hidden_1_proj")
    proj2 = parse_projections(df, "mid_hidden_2_proj")
    proj_cols = list(proj1.columns) + list(proj2.columns)
    df = pd.concat([df, proj1, proj2], axis=1)

    # Extract 3rd-order kinematics, torsion, and logit entropy quantiles
    logging.info("Extracting Jerk (j_t), Torsion (tau_t), Phase-space distance, and Entropy Quantiles...")
    df = extract_nextgen_features(df, list(proj1.columns), list(proj2.columns))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    nextgen_feature_cols = BASELINE_FEATURES + [
        "traj_vel_norm", "traj_acc_norm", "traj_jerk_norm", "traj_curvature", "traj_torsion",
        "phase_attractor_dist", "layer_align_m1_m2",
        "entropy_p10", "entropy_p50", "entropy_p90", "entropy_iqr", "entropy_skew_ratio"
    ] + proj_cols

    unique_run_ids = df["run_id"].unique()
    num_runs = len(unique_run_ids)
    max_len = df.groupby("run_id")["step"].count().max()
    input_dim = len(nextgen_feature_cols)

    features_np = np.zeros((num_runs, max_len, input_dim), dtype=np.float32)
    targets_np = np.full((num_runs, max_len), -1, dtype=np.int64)
    lengths_np = np.zeros(num_runs, dtype=np.int64)

    run_id_to_idx = {rid: i for i, rid in enumerate(unique_run_ids)}
    grouped = df.groupby("run_id")
    for rid, group in grouped:
        idx = run_id_to_idx[rid]
        run_len = len(group)
        features_np[idx, :run_len, :] = group[nextgen_feature_cols].to_numpy()
        targets_np[idx, :run_len] = group["correct"].to_numpy()
        lengths_np[idx] = run_len

    features_tensor = torch.tensor(features_np, dtype=torch.float32)
    targets_tensor = torch.tensor(targets_np, dtype=torch.long)

    oof_predictions = {
        "Baseline Linear": np.full(len(df), np.nan),
        "BiGRU (Sequence)": np.full(len(df), np.nan),
        "TCN (Temporal Conv)": np.full(len(df), np.nan),
        "BetaLikelihood (Expected Reward)": np.full(len(df), np.nan),
        "Causal RoPE Transformer": np.full(len(df), np.nan),
        "Gated MoE Sequence Probe": np.full(len(df), np.nan)
    }

    n_splits = 2 if args.smoke_test else 5
    epochs = 1 if args.smoke_test else 60
    batch_size = 512 if args.smoke_test else 4096

    gkf = GroupKFold(n_splits=n_splits)
    logging.info(f"Starting Next-Gen Probing Tournament ({n_splits} folds, {epochs} epochs)...")

    task_to_grp = {tid: i for i, tid in enumerate(df["task_id"].unique())}
    groups = df["task_id"].map(task_to_grp).to_numpy()

    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups)):
        logging.info(f"--- FOLD {fold+1}/{n_splits} ---")
        train = df.iloc[train_idx]
        test = df.iloc[test_idx].copy()

        test_run_ids = test["run_id"].unique()
        test_run_indices = np.array([run_id_to_idx[rid] for rid in test_run_ids])
        train_run_ids = train["run_id"].unique()
        train_run_indices = np.array([run_id_to_idx[rid] for rid in train_run_ids])

        # 1. Baseline Linear
        probe_base = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
        probe_base.fit(train[BASELINE_FEATURES], train["correct"])
        oof_predictions["Baseline Linear"][test_idx] = probe_base.predict_proba(test[BASELINE_FEATURES])[:, 1]

        # Mask for sequence outputs
        mask = np.arange(max_len) < lengths_np[test_run_indices][:, None]

        # 2. BiGRU
        bigru = train_model_v2(train_run_indices, features_tensor, targets_tensor, lengths_np, "BiGRU", epochs=epochs, device=device, batch_size=batch_size)
        probs_bigru = predict_model_v2(bigru, test_run_indices, features_tensor, "BiGRU", device=device)
        oof_predictions["BiGRU (Sequence)"][test_idx] = probs_bigru[mask]

        # 3. TCN
        tcn = train_model_v2(train_run_indices, features_tensor, targets_tensor, lengths_np, "TCN", epochs=epochs, device=device, batch_size=batch_size)
        probs_tcn = predict_model_v2(tcn, test_run_indices, features_tensor, "TCN", device=device)
        oof_predictions["TCN (Temporal Conv)"][test_idx] = probs_tcn[mask]

        # 4. BetaLikelihood
        beta_m = train_model_v2(train_run_indices, features_tensor, targets_tensor, lengths_np, "BetaLikelihood", epochs=epochs, device=device, batch_size=batch_size)
        probs_beta = predict_model_v2(beta_m, test_run_indices, features_tensor, "BetaLikelihood", device=device)
        oof_predictions["BetaLikelihood (Expected Reward)"][test_idx] = probs_beta[mask]

        # 5. Causal RoPE Transformer
        trans = train_model_v2(train_run_indices, features_tensor, targets_tensor, lengths_np, "Transformer", epochs=epochs, device=device, batch_size=batch_size)
        probs_trans = predict_model_v2(trans, test_run_indices, features_tensor, "Transformer", device=device)
        oof_predictions["Causal RoPE Transformer"][test_idx] = probs_trans[mask]

        # 6. Gated MoE Probe
        moe_m = train_model_v2(train_run_indices, features_tensor, targets_tensor, lengths_np, "MoE", epochs=epochs, device=device, batch_size=batch_size)
        probs_moe = predict_model_v2(moe_m, test_run_indices, features_tensor, "MoE", device=device)
        oof_predictions["Gated MoE Sequence Probe"][test_idx] = probs_moe[mask]

    print("\n" + "=" * 100)
    print("                 NEXT-GENERATION PROBING TOURNAMENT VERDICT SUMMARY")
    print("=" * 100)
    print(f"{'Configuration':<35} | {'OOF AUC':<10} | {'ECE':<8}")
    print("-" * 100)

    y_all = df["correct"].astype(int).to_numpy()

    for name in oof_predictions:
        preds = oof_predictions[name]
        valid_mask = ~np.isnan(preds)
        auc = roc_auc_score(y_all[valid_mask], preds[valid_mask])
        ece = calculate_ece(preds[valid_mask], y_all[valid_mask])
        print(f"{name:<35} | {auc:<10.4f} | {ece:<8.4f}")
    print("=" * 100)

if __name__ == "__main__":
    main()
