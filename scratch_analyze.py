import os
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research", "outputs", "experiment_matrix")
MODELS = [
    "deepseek_r1_distill_1p5b", "deepseek_r1_distill_7b", "qwen2p5_0p5b", "qwen2p5_3b",
    "qwen2p5_7b", "qwen2p5_14b", "qwen2p5_32b", "mistral_7b_instruct_v0p3",
    "phi_4_mini_instruct", "internlm3_8b_instruct", "yi_1p5_9b_chat",
    "mistral_small_24b_2409", "llama_3p1_8b_instruct"
]
DATASETS = ["gsm8k", "math", "arc", "gpqa"]

records = []
for dataset in DATASETS:
    for model in MODELS:
        csv_path = os.path.join(ROOT, f"{model}__{dataset}", "detector_comparison_by_run.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        df_filtered = df[df['detector'].isin(['hazard_drift', 'never_stop'])].copy()
        if df_filtered.empty:
            continue
        pivoted = df_filtered.pivot(index='run_id', columns='detector', values='stop_utility').reset_index()
        if 'hazard_drift' not in pivoted.columns or 'never_stop' not in pivoted.columns:
            continue
        for _, row in pivoted.iterrows():
            hd_util = row['hazard_drift']
            ns_util = row['never_stop']
            if pd.isna(hd_util) or pd.isna(ns_util):
                continue
            if hd_util > ns_util: status = "win"
            elif hd_util == ns_util: status = "tie"
            else: status = "loss"
            records.append(status)

df_all = pd.Series(records)
print(f"Total runs: {len(df_all)}")
counts = df_all.value_counts()
print(counts)
print(counts / len(df_all) * 100)
