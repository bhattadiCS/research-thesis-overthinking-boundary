import os
import pandas as pd
import numpy as np

ROOT = "C:/Aditya_Data/Personal/ResearchThesis/research/outputs/experiment_matrix"
MODELS = ["deepseek_r1_distill_1p5b","deepseek_r1_distill_7b","qwen2p5_0p5b","qwen2p5_3b",
          "qwen2p5_7b","qwen2p5_14b","qwen2p5_32b","mistral_7b_instruct_v0p3",
          "phi_4_mini_instruct","internlm3_8b_instruct","yi_1p5_9b_chat",
          "mistral_small_24b_2409","llama_3p1_8b_instruct"]
DATASETS = ["gsm8k","math","arc","gpqa"]

rows = []
for m in MODELS:
    for ds in DATASETS:
        path = os.path.join(ROOT, f"{m}__{ds}", "detector_comparison.csv")
        df = pd.read_csv(path)
        df["model"]=m; df["dataset"]=ds
        rows.append(df)
allc = pd.concat(rows, ignore_index=True)

# pivot: per cell, gap by detector
piv = allc.pivot_table(index=["model","dataset"], columns="detector", values="mean_oracle_gap")
print("Pivot shape (cells x detectors):", piv.shape)

# Head-to-head: hazard_drift LOWER gap (better) than competitor
def h2h(a, b):
    wins = (piv[a] < piv[b]).sum()
    ties = (piv[a] == piv[b]).sum()
    n = len(piv)
    return wins, ties, n
for comp in ["entropy_plateau","answer_stability","first_answer","e_process"]:
    w,t,n = h2h("hazard_drift", comp)
    print(f"hazard_drift < {comp:18s}: {w}/{n} ({100*w/n:.0f}%)  ties={t}")

# Best GENUINE detector per cell.
# "Genuine" = deployable, label-free, non-degenerate, non-oracle.
# Exclude: oracle, verifier_first_correct (cheat), first_answer (degenerate fe=1), never_stop (degenerate).
GENUINE = ["hazard_drift","e_process","empirical_bernstein","entropy_plateau","answer_stability"]
print("\n=== Best genuine detector per cell (lowest oracle_gap among GENUINE) ===")
best_counts = {d:0 for d in GENUINE}
for (m,ds), r in piv.iterrows():
    vals = {d:r[d] for d in GENUINE}
    best = min(vals, key=vals.get)
    best_counts[best]+=1
for d,c in sorted(best_counts.items(), key=lambda x:-x[1]):
    print(f"  {d:20s} best in {c}/52 cells")

# Per-dataset: is hazard_drift the best genuine detector on the dataset-mean?
print("\n=== Per-dataset best genuine detector (by dataset-mean gap) ===")
for ds in DATASETS:
    sub = allc[allc["dataset"]==ds]
    g = sub.groupby("detector")["mean_oracle_gap"].mean()
    gg = g[g.index.isin(GENUINE)].sort_values()
    print(f"  {ds:6s}: " + ", ".join(f"{d}={v:.3f}" for d,v in gg.items()))

# Also: per-dataset best genuine detector by per-cell majority vote
print("\n=== Per-dataset best genuine detector (per-cell win count) ===")
for ds in DATASETS:
    sub = piv.xs(ds, level="dataset")
    cnt = {d:0 for d in GENUINE}
    for _, r in sub.iterrows():
        vals = {d:r[d] for d in GENUINE}
        cnt[min(vals,key=vals.get)]+=1
    print(f"  {ds:6s}: " + ", ".join(f"{d}={c}" for d,c in sorted(cnt.items(),key=lambda x:-x[1])))

# Sanity: confirm first_answer is degenerate (stop=1, fe=1) in EVERY cell
fa = allc[allc["detector"]=="first_answer"]
print("\nfirst_answer: stop_step unique=", sorted(fa["mean_stop_step"].unique()),
      " false_early unique=", sorted(fa["false_early_rate"].unique()))
ns = allc[allc["detector"]=="never_stop"]
print("never_stop: false_late min/max=", ns["false_late_rate"].min(), ns["false_late_rate"].max())

# false_late range for hazard_drift across datasets (verdict says misses 24-59%)
hd = allc[allc["detector"]=="hazard_drift"]
print("\nhazard_drift false_late by dataset (mean):")
print(hd.groupby("dataset")["false_late_rate"].mean().round(3).to_dict())
print("hazard_drift false_late pooled mean:", round(hd["false_late_rate"].mean(),3))
print("hazard_drift false_early by dataset (mean):")
print(hd.groupby("dataset")["false_early_rate"].mean().round(3).to_dict())
