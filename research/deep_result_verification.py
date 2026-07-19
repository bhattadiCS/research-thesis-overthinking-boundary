#!/usr/bin/env python3
"""
Deep Data Science Verification & Model/Dataset Breakdown Analysis
File: research/deep_result_verification.py

Parses all 52 dataset cells and evaluates model performance stratified by:
1. Dataset source (GSM8K, MATH, ARC, GPQA)
2. Model Family & Parameter Scale (Qwen, DeepSeek-R1, Mistral, Llama, Phi, Yi)
3. Reasoning Trajectory Length & Overthinking Rate
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def main():
    v2_dir = Path("research/outputs/experiments_v2")
    trace_paths = list(v2_dir.glob("**/trace_steps.csv"))
    
    if not trace_paths:
        logging.error("No trace steps found.")
        return
        
    logging.info(f"Loading {len(trace_paths)} cells for stratified data science audit...")
    cell_summaries = []
    
    for path in trace_paths:
        cell_name = path.parent.name
        try:
            df = pd.read_csv(path)
            df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int)
            df["step"] = pd.to_numeric(df["step"], errors="coerce").fillna(1).astype(int)
            
            # Identify model and task
            parts = cell_name.replace("global_", "").rsplit("_", 1)
            model_name = parts[0] if len(parts) == 2 else cell_name
            dataset_name = parts[1] if len(parts) == 2 else "unknown"
            
            # Compute trajectory-level overthinking dynamics
            num_runs = df["run_id"].nunique()
            
            # First step vs step 2 vs final step correctness
            step1_df = df[df["step"] == 1]
            step2_df = df[df["step"] == 2]
            final_df = df.groupby("run_id").last()
            
            acc_step1 = step1_df["correct"].mean() if not step1_df.empty else 0.0
            acc_step2 = step2_df["correct"].mean() if not step2_df.empty else 0.0
            acc_final = final_df["correct"].mean() if not final_df.empty else 0.0
            
            overthinking_decay = acc_step2 - acc_final
            
            cell_summaries.append({
                "cell": cell_name,
                "model": model_name,
                "dataset": dataset_name,
                "num_runs": num_runs,
                "total_steps": len(df),
                "avg_steps": len(df) / num_runs if num_runs > 0 else 0,
                "acc_step1": acc_step1,
                "acc_step2": acc_step2,
                "acc_final": acc_final,
                "overthinking_decay": overthinking_decay
            })
        except Exception as e:
            logging.warning(f"Error processing {path}: {e}")
            
    summary_df = pd.DataFrame(cell_summaries)
    
    print("\n" + "=" * 90)
    print("                    DATASET BREAKDOWN & OVERTHINKING DYNAMICS AUDIT")
    print("=" * 90)
    print(f"{'Dataset':<10} | {'Runs':<8} | {'Avg Steps':<10} | {'Step 2 Acc':<12} | {'Final Acc':<12} | {'Decay (Overthinking)':<20}")
    print("-" * 90)
    
    for ds, g in summary_df.groupby("dataset"):
        runs = g["num_runs"].sum()
        avg_steps = (g["total_steps"].sum() / runs) if runs > 0 else 0.0
        acc_s2 = g["acc_step2"].mean()
        acc_fin = g["acc_final"].mean()
        decay = acc_s2 - acc_fin
        print(f"{ds:<10} | {runs:<8} | {avg_steps:<10.2f} | {acc_s2:<12.4f} | {acc_fin:<12.4f} | {decay:<+20.4f}")
        
    print("\n" + "=" * 90)
    print("                    MODEL FAMILY BREAKDOWN & OVERTHINKING DYNAMICS")
    print("=" * 90)
    print(f"{'Model Family':<25} | {'Runs':<8} | {'Avg Steps':<10} | {'Step 2 Acc':<12} | {'Final Acc':<12} | {'Decay (Overthinking)':<20}")
    print("-" * 90)
    
    for model_name, g in summary_df.groupby("model"):
        runs = g["num_runs"].sum()
        avg_steps = (g["total_steps"].sum() / runs) if runs > 0 else 0.0
        acc_s2 = g["acc_step2"].mean()
        acc_fin = g["acc_final"].mean()
        decay = acc_s2 - acc_fin
        print(f"{model_name:<25} | {runs:<8} | {avg_steps:<10.2f} | {acc_s2:<12.4f} | {acc_fin:<12.4f} | {decay:<+20.4f}")
        
    print("=" * 90)

if __name__ == "__main__":
    main()
