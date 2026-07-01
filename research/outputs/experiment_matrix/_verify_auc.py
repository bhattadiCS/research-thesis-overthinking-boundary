import os
import pandas as pd, numpy as np

ROOT = r"C:/Aditya_Data/Personal/ResearchThesis/research/outputs/experiment_matrix"
MODELS = ["deepseek_r1_distill_1p5b","deepseek_r1_distill_7b","qwen2p5_0p5b","qwen2p5_3b",
"qwen2p5_7b","qwen2p5_14b","qwen2p5_32b","mistral_7b_instruct_v0p3","phi_4_mini_instruct",
"internlm3_8b_instruct","yi_1p5_9b_chat","mistral_small_24b_2409","llama_3p1_8b_instruct"]
DATASETS = ["gsm8k","math","arc","gpqa"]

rows=[]
for m in MODELS:
    for d in DATASETS:
        f=os.path.join(ROOT,f"{m}__{d}","correctness_probe_metrics.csv")
        if not os.path.exists(f): continue
        df=pd.read_csv(f)
        pooled=df[df.scope=="pooled_oof"]
        per_run=df[df.scope=="per_run_oof"]
        pooled_auc=pooled.auc.values[0] if len(pooled) else np.nan
        per_run_mean=per_run.auc.mean()  # noisy n=10 per-run
        rows.append(dict(model=m,dataset=d,pooled_auc=pooled_auc,per_run_mean_auc=per_run_mean,
                         per_run_nonnull=per_run.auc.notna().sum()))
A=pd.DataFrame(rows)
print(f"cells: {len(A)}")
print("\n=== POOLED OOF AUC (the cell-level OOS AUC; matches CROSS_FAMILY_REPORT) ===")
print(f"mean={A.pooled_auc.mean():.3f}  median={A.pooled_auc.median():.3f}  min={A.pooled_auc.min():.3f}  max={A.pooled_auc.max():.3f}")
print(f"AUC<0.65 in {int((A.pooled_auc<0.65).sum())}/{len(A)}; AUC>=0.8 in {int((A.pooled_auc>=0.8).sum())}/{len(A)}")
print(f"AUC<0.55 (near chance) in {int((A.pooled_auc<0.55).sum())}/{len(A)}")
print("\n=== per-run mean AUC (noisy, n=10/run, many NaN) ===")
print(f"mean={A.per_run_mean_auc.mean():.3f}  min={A.per_run_mean_auc.min():.3f}  max={A.per_run_mean_auc.max():.3f}")

print("\n=== POOLED AUC by dataset ===")
print(A.groupby("dataset").pooled_auc.agg(["mean","min","max","count"]).round(3).to_string())

print("\n=== full pooled AUC sorted ===")
print(A.sort_values("pooled_auc")[["model","dataset","pooled_auc"]].round(3).to_string(index=False))
