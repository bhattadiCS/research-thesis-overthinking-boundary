import os
import re
import csv
import pandas as pd
import numpy as np

ROOT = "research/outputs/experiment_matrix"
MODELS = [
    "deepseek_r1_distill_1p5b", "deepseek_r1_distill_7b", "qwen2p5_0p5b", "qwen2p5_3b",
    "qwen2p5_7b", "qwen2p5_14b", "qwen2p5_32b", "mistral_7b_instruct_v0p3",
    "phi_4_mini_instruct", "internlm3_8b_instruct", "yi_1p5_9b_chat",
    "mistral_small_24b_2409", "llama_3p1_8b_instruct"
]
DATASETS = ["gsm8k", "math", "arc", "gpqa"]

def parse_temp(run_id):
    # Search for tempXX.XX or tempXX.X
    match = re.search(r'temp(\d+\.\d+)', run_id)
    if match:
        val = float(match.group(1))
        # Round/format to standard 0.1, 0.6, 1.0
        if abs(val - 0.1) < 0.05:
            return 0.1
        elif abs(val - 0.6) < 0.05:
            return 0.6
        elif abs(val - 1.0) < 0.05:
            return 1.0
        return val
    return None

def analyze():
    records = []
    missing_cells = []
    
    for dataset in DATASETS:
        for model in MODELS:
            cell_dir = os.path.join(ROOT, f"{model}__{dataset}")
            csv_path = os.path.join(cell_dir, "detector_comparison_by_run.csv")
            if not os.path.exists(csv_path):
                missing_cells.append(f"{model}__{dataset}")
                continue
                
            # Read the CSV
            df = pd.read_csv(csv_path)
            # Group by run_id
            # Pivot the detector stop_utility
            # We want hazard_drift and never_stop
            df_filtered = df[df['detector'].isin(['hazard_drift', 'never_stop'])].copy()
            if df_filtered.empty:
                continue
            
            # Pivot table
            pivoted = df_filtered.pivot(index='run_id', columns='detector', values='stop_utility').reset_index()
            
            if 'hazard_drift' not in pivoted.columns or 'never_stop' not in pivoted.columns:
                continue
                
            for _, row in pivoted.iterrows():
                run_id = row['run_id']
                temp = parse_temp(run_id)
                if temp is None:
                    continue
                
                hd_util = row['hazard_drift']
                ns_util = row['never_stop']
                
                # Check for NaN
                if pd.isna(hd_util) or pd.isna(ns_util):
                    continue
                    
                records.append({
                    'dataset': dataset.upper(),
                    'model': model,
                    'temperature': temp,
                    'hazard_drift': hd_util,
                    'never_stop': ns_util,
                    'strictly_useful': 1 if hd_util > ns_util else 0
                })
                
    if missing_cells:
        print(f"Warning: Missing cells: {missing_cells}")
        
    df_all = pd.DataFrame(records)
    print(f"Loaded {len(df_all)} runs across all valid cells.")
    print("Unique temperatures parsed:", df_all['temperature'].unique())
    print("Unique datasets:", df_all['dataset'].unique())
    
    # Compute mean utility grouped by dataset and temperature
    grouped = df_all.groupby(['dataset', 'temperature']).agg(
        mean_hazard_drift=('hazard_drift', 'mean'),
        mean_never_stop=('never_stop', 'mean'),
        pct_strictly_useful=('strictly_useful', lambda x: x.mean() * 100),
        total_runs=('strictly_useful', 'count')
    ).reset_index()
    
    print("\n--- RESULTS BY DATASET AND TEMPERATURE ---")
    print(grouped.to_string(index=False))
    
    # Also print grouped just by dataset
    grouped_ds = df_all.groupby(['dataset']).agg(
        mean_hazard_drift=('hazard_drift', 'mean'),
        mean_never_stop=('never_stop', 'mean'),
        pct_strictly_useful=('strictly_useful', lambda x: x.mean() * 100),
        total_runs=('strictly_useful', 'count')
    ).reset_index()
    print("\n--- RESULTS BY DATASET ONLY ---")
    print(grouped_ds.to_string(index=False))
    
    # Also print grouped just by temperature
    grouped_temp = df_all.groupby(['temperature']).agg(
        mean_hazard_drift=('hazard_drift', 'mean'),
        mean_never_stop=('never_stop', 'mean'),
        pct_strictly_useful=('strictly_useful', lambda x: x.mean() * 100),
        total_runs=('strictly_useful', 'count')
    ).reset_index()
    print("\n--- RESULTS BY TEMPERATURE ONLY ---")
    print(grouped_temp.to_string(index=False))

if __name__ == "__main__":
    analyze()
