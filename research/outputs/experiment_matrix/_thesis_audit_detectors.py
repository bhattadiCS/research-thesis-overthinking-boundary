import os, glob, math
import pandas as pd, numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = ["deepseek_r1_distill_1p5b","deepseek_r1_distill_7b","qwen2p5_0p5b","qwen2p5_3b",
"qwen2p5_7b","qwen2p5_14b","qwen2p5_32b","mistral_7b_instruct_v0p3","phi_4_mini_instruct",
"internlm3_8b_instruct","yi_1p5_9b_chat","mistral_small_24b_2409","llama_3p1_8b_instruct"]
DATASETS = ["gsm8k","math","arc","gpqa"]

THEORY = ["hazard_drift","e_process","empirical_bernstein"]
HEUR   = ["answer_stability","entropy_plateau","first_answer"]
CHEAT  = ["verifier_first_correct"]
BOUNDS = ["oracle","never_stop"]

rows=[]
missing=[]
for m in MODELS:
    for d in DATASETS:
        f=os.path.join(ROOT,f"{m}__{d}","detector_comparison.csv")
        if not os.path.exists(f):
            missing.append(f"{m}__{d}"); continue
        df=pd.read_csv(f)
        df["model"]=m; df["dataset"]=d
        rows.append(df)
print("CELLS LOADED:",len(rows),"  MISSING:",missing)
alld=pd.concat(rows,ignore_index=True)
print("unique detectors:",sorted(alld.detector.unique()))

# Per-detector mean across all 52 cells
print("\n===== MEAN ACROSS ALL 52 CELLS (per detector) =====")
g=alld.groupby("detector").agg(
    n_cells=("mean_oracle_gap","size"),
    oracle_gap=("mean_oracle_gap","mean"),
    false_late=("false_late_rate","mean"),
    false_early=("false_early_rate","mean"),
    stop_util=("mean_stop_utility","mean"),
    stop_step=("mean_stop_step","mean"),
).sort_values("oracle_gap")
pd.set_option("display.width",200); pd.set_option("display.max_columns",20)
print(g.round(4))

# Same but ONLY gsm8k (the sweet-spot where boundaries exist) and ONLY late-boundary cells
print("\n===== MEAN ACROSS GSM8K CELLS ONLY (13 cells) =====")
g2=alld[alld.dataset=="gsm8k"].groupby("detector").agg(
    n_cells=("mean_oracle_gap","size"),
    oracle_gap=("mean_oracle_gap","mean"),
    false_late=("false_late_rate","mean"),
    false_early=("false_early_rate","mean"),
    stop_util=("mean_stop_utility","mean"),
).sort_values("oracle_gap")
print(g2.round(4))

# Head-to-head: for each cell, rank deployable detectors only (exclude oracle, cheater)
deployable = THEORY + HEUR + ["never_stop"]
print("\n===== HEAD-TO-HEAD: best DEPLOYABLE detector per cell by oracle_gap =====")
sub=alld[alld.detector.isin(deployable)].copy()
idx=sub.groupby(["model","dataset"])["mean_oracle_gap"].idxmin()
best=sub.loc[idx]
print("Times each detector is the BEST (lowest oracle_gap) deployable across 52 cells:")
print(best.detector.value_counts())

print("\n===== HEAD-TO-HEAD on GSM8K only =====")
subg=sub[sub.dataset=="gsm8k"]
idxg=subg.groupby(["model"])["mean_oracle_gap"].idxmin()
print(subg.loc[idxg].detector.value_counts())

# Direct theory-vs-heuristic: is hazard_drift better than the BEST heuristic per cell?
print("\n===== hazard_drift vs best-heuristic, per cell (oracle_gap) =====")
piv=alld.pivot_table(index=["model","dataset"],columns="detector",values="mean_oracle_gap")
piv["best_heur"]=piv[HEUR].min(axis=1)
piv["hd_minus_bestheur"]=piv["hazard_drift"]-piv["best_heur"]
wins=(piv["hd_minus_bestheur"]<0).sum()
print(f"hazard_drift beats best heuristic on oracle_gap in {wins}/{len(piv)} cells")
print("mean(hazard_drift gap - best heuristic gap) =", round(piv["hd_minus_bestheur"].mean(),4))

# false_late comparison (catching overthinking) - theory vs heuristics
print("\n===== false_late_rate: theory vs heuristics (mean across 52) =====")
fl=alld.pivot_table(index=["model","dataset"],columns="detector",values="false_late_rate")
print("hazard_drift mean false_late:",round(fl["hazard_drift"].mean(),4))
print("e_process mean false_late:",round(fl["e_process"].mean(),4))
print("empirical_bernstein mean false_late:",round(fl["empirical_bernstein"].mean(),4))
print("first_answer mean false_late:",round(fl["first_answer"].mean(),4))
print("entropy_plateau mean false_late:",round(fl["entropy_plateau"].mean(),4))
print("answer_stability mean false_late:",round(fl["answer_stability"].mean(),4))

# How often does each detector improve over never_stop (the do-nothing lower bound)?
print("\n===== utility improvement over never_stop =====")
util=alld.pivot_table(index=["model","dataset"],columns="detector",values="mean_stop_utility")
for det in THEORY+HEUR+CHEAT:
    better=(util[det]>util["never_stop"]).sum()
    meandelta=(util[det]-util["never_stop"]).mean()
    print(f"{det:24s}: beats never_stop in {better:2d}/52 cells, mean util gain {meandelta:+.4f}")

# Oracle gap as fraction of (never_stop gap): normalized regret. 0=oracle, 1=never_stop
print("\n===== NORMALIZED REGRET vs never_stop (0=oracle,1=do-nothing); mean over 52 =====")
ns_gap=alld[alld.detector=="never_stop"].set_index(["model","dataset"])["mean_oracle_gap"]
for det in THEORY+HEUR+CHEAT:
    dg=alld[alld.detector==det].set_index(["model","dataset"])["mean_oracle_gap"]
    norm=(dg/ns_gap.reindex(dg.index)).replace([np.inf,-np.inf],np.nan)
    print(f"{det:24s}: mean normalized regret = {norm.mean():.3f}  (median {norm.median():.3f})")
