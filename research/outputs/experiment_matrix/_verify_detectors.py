import os, glob
import pandas as pd
import numpy as np

ROOT = "C:/Aditya_Data/Personal/ResearchThesis/research/outputs/experiment_matrix"
MODELS = ["deepseek_r1_distill_1p5b","deepseek_r1_distill_7b","qwen2p5_0p5b","qwen2p5_3b",
          "qwen2p5_7b","qwen2p5_14b","qwen2p5_32b","mistral_7b_instruct_v0p3",
          "phi_4_mini_instruct","internlm3_8b_instruct","yi_1p5_9b_chat",
          "mistral_small_24b_2409","llama_3p1_8b_instruct"]
DATASETS = ["gsm8k","math","arc","gpqa"]

rows = []
missing = []
for m in MODELS:
    for ds in DATASETS:
        cell = f"{m}__{ds}"
        path = os.path.join(ROOT, cell, "detector_comparison.csv")
        if not os.path.exists(path):
            missing.append(cell); continue
        df = pd.read_csv(path)
        df["model"] = m; df["dataset"] = ds
        rows.append(df)

print("Cells loaded:", len(rows), "Missing:", missing)
allc = pd.concat(rows, ignore_index=True)
print("Total rows:", len(allc))
print("Detectors present:", sorted(allc["detector"].unique()))
print("Cells per detector:\n", allc.groupby("detector").size())

# columns: detector, mean_stop_step, mean_stop_utility, mean_oracle_utility,
#          mean_oracle_gap, false_early_rate, false_late_rate, mean_false_late_severity
METRICS = ["mean_oracle_gap","false_late_rate","false_early_rate","mean_stop_step",
           "mean_stop_utility","mean_oracle_utility"]

def agg(sub, name):
    g = sub.groupby("detector")[METRICS].mean()
    g = g.sort_values("mean_oracle_gap")
    print(f"\n===== {name} (mean over {sub['model'].nunique()} models) =====")
    for det, r in g.iterrows():
        print(f"  {det:24s} gap={r['mean_oracle_gap']:.3f} fl={r['false_late_rate']:.2f} "
              f"fe={r['false_early_rate']:.2f} stop={r['mean_stop_step']:.2f} "
              f"util={r['mean_stop_utility']:.3f} orac={r['mean_oracle_utility']:.3f}")
    return g

# POOLED across all 52 cells
pooled = agg(allc, "POOLED 52 cells")

# Per-dataset
per_ds = {}
for ds in DATASETS:
    per_ds[ds] = agg(allc[allc["dataset"]==ds], ds.upper())

# Fraction of never_stop->oracle gap closed by best deployable theory detector
print("\n===== Fraction of never_stop-to-oracle UTILITY gap closed =====")
for ds in DATASETS:
    g = per_ds[ds]
    ns = g.loc["never_stop","mean_stop_utility"]
    orac = g.loc["oracle","mean_stop_utility"]
    denom = orac - ns
    for det in ["hazard_drift","e_process","empirical_bernstein"]:
        u = g.loc[det,"mean_stop_utility"]
        frac = (u - ns)/denom*100 if denom!=0 else float('nan')
        resid = orac - u
        print(f"  {ds:6s} {det:20s} util={u:.3f} closed={frac:.1f}% resid_gap={resid:.3f}")

# anytime-valid stop_step normalized to never_stop horizon
print("\n===== Anytime-valid stop_step / never_stop horizon =====")
for ds in DATASETS:
    g = per_ds[ds]
    horizon = g.loc["never_stop","mean_stop_step"]
    for det in ["e_process","empirical_bernstein"]:
        ss = g.loc[det,"mean_stop_step"]
        print(f"  {ds:6s} {det:20s} stop={ss:.2f} horizon={horizon:.1f} ratio={ss/horizon:.2f} fl={g.loc[det,'false_late_rate']:.2f}")
