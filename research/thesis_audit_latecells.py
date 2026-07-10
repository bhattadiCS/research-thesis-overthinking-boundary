import os
import pandas as pd, numpy as np

# Relocated out of research/outputs/experiment_matrix/ (audit lever H6):
# analysis code must not live inside the outputs tree.
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "experiment_matrix")
late=[("qwen2p5_3b","gsm8k"),("qwen2p5_7b","gsm8k"),("qwen2p5_7b","math"),("qwen2p5_14b","gsm8k"),
("qwen2p5_32b","gsm8k"),("qwen2p5_32b","math"),("mistral_7b_instruct_v0p3","gsm8k"),
("internlm3_8b_instruct","gsm8k"),("yi_1p5_9b_chat","gsm8k"),("mistral_small_24b_2409","gsm8k"),
("mistral_small_24b_2409","arc"),("llama_3p1_8b_instruct","gsm8k")]
THEORY=["hazard_drift","e_process","empirical_bernstein"]
HEUR=["answer_stability","entropy_plateau","first_answer"]

rows=[]
for m,d in late:
    f=os.path.join(ROOT,f"{m}__{d}","detector_comparison.csv")
    df=pd.read_csv(f); df["model"]=m; df["dataset"]=d; rows.append(df)
L=pd.concat(rows,ignore_index=True)

print("===== DETECTOR PERFORMANCE ON THE 12 LATE-BOUNDARY CELLS ONLY =====")
print("(this is the ONLY regime where the optimal-stopping theory has a nontrivial boundary to find)\n")
g=L.groupby("detector").agg(
    oracle_gap=("mean_oracle_gap","mean"),
    false_late=("false_late_rate","mean"),
    false_early=("false_early_rate","mean"),
    stop_util=("mean_stop_utility","mean"),
    stop_step=("mean_stop_step","mean"),
).sort_values("oracle_gap")
print(g.round(4).to_string())

# head to head deployable on late cells
deployable=THEORY+HEUR+["never_stop"]
sub=L[L.detector.isin(deployable)]
idx=sub.groupby(["model","dataset"])["mean_oracle_gap"].idxmin()
print("\nBest deployable detector per late-cell (by oracle_gap):")
print(sub.loc[idx].groupby(["model","dataset"]).apply(lambda x:x.iloc[0]["detector"]).to_string())
print("\ncounts:",dict(sub.loc[idx].detector.value_counts()))

# hazard_drift vs first_answer head-to-head on late cells (the key comparison)
piv=L.pivot_table(index=["model","dataset"],columns="detector",values="mean_oracle_gap")
print("\n===== hazard_drift vs first_answer (oracle_gap) on late cells =====")
cmp=piv[["hazard_drift","first_answer","answer_stability","entropy_plateau"]].copy()
cmp["hd_beats_first"]=cmp["hazard_drift"]<cmp["first_answer"]
print(cmp.round(3).to_string())
print(f"\nhazard_drift beats first_answer on {int(cmp['hd_beats_first'].sum())}/12 late cells")
print(f"mean oracle_gap: hazard_drift={cmp['hazard_drift'].mean():.4f}  first_answer={cmp['first_answer'].mean():.4f}  "
      f"answer_stability={cmp['answer_stability'].mean():.4f}  entropy_plateau={cmp['entropy_plateau'].mean():.4f}")

# utility view on late cells
pu=L.pivot_table(index=["model","dataset"],columns="detector",values="mean_stop_utility")
print("\n===== mean_stop_utility on late cells (higher=better) =====")
print("oracle=%.4f  hazard_drift=%.4f  first_answer=%.4f  answer_stability=%.4f  entropy_plateau=%.4f  e_process=%.4f  EB=%.4f  never_stop=%.4f"%(
 pu["oracle"].mean(),pu["hazard_drift"].mean(),pu["first_answer"].mean(),pu["answer_stability"].mean(),
 pu["entropy_plateau"].mean(),pu["e_process"].mean(),pu["empirical_bernstein"].mean(),pu["never_stop"].mean()))
# fraction of oracle utility captured (normalized)
print("\nfraction of oracle's utility-over-neverstop captured:")
denom=(pu["oracle"]-pu["never_stop"])
for det in ["hazard_drift","first_answer","answer_stability","entropy_plateau","e_process","empirical_bernstein"]:
    frac=((pu[det]-pu["never_stop"])/denom).mean()
    print(f"  {det:20s}: {frac:.3f}")
