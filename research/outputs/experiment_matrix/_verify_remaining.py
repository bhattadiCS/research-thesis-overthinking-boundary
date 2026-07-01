import os
import pandas as pd, numpy as np

ROOT = r"C:/Aditya_Data/Personal/ResearchThesis/research/outputs/experiment_matrix"
MODELS = ["deepseek_r1_distill_1p5b","deepseek_r1_distill_7b","qwen2p5_0p5b","qwen2p5_3b",
"qwen2p5_7b","qwen2p5_14b","qwen2p5_32b","mistral_7b_instruct_v0p3","phi_4_mini_instruct",
"internlm3_8b_instruct","yi_1p5_9b_chat","mistral_small_24b_2409","llama_3p1_8b_instruct"]
DATASETS = ["gsm8k","math","arc","gpqa"]

dc_rows=[]
for m in MODELS:
    for d in DATASETS:
        f=os.path.join(ROOT,f"{m}__{d}","detector_comparison.csv")
        df=pd.read_csv(f); df["model"]=m; df["dataset"]=d; dc_rows.append(df)
DC=pd.concat(dc_rows,ignore_index=True)

late=[("qwen2p5_3b","gsm8k"),("qwen2p5_7b","gsm8k"),("qwen2p5_7b","math"),("qwen2p5_14b","gsm8k"),
("qwen2p5_32b","gsm8k"),("qwen2p5_32b","math"),("mistral_7b_instruct_v0p3","gsm8k"),
("internlm3_8b_instruct","gsm8k"),("yi_1p5_9b_chat","gsm8k"),("mistral_small_24b_2409","gsm8k"),
("mistral_small_24b_2409","arc"),("llama_3p1_8b_instruct","gsm8k")]
lateset=set(late)
DC["is_late"]=DC.apply(lambda r:(r.model,r.dataset) in lateset,axis=1)

def col(det,c,mask=None):
    s=DC[DC.detector==det]
    if mask is not None: s=s[mask(s)]
    return s[c]

print("=== e_process false_early in the 12 LATE cells ===")
ep_late=DC[(DC.detector=="e_process")&(DC.is_late)]
print(f"e_process false_early>0.05 in late cells: {int((ep_late.false_early_rate>0.05).sum())}/12; values: {ep_late.false_early_rate.round(3).tolist()}")

print("\n=== EB worse than first_answer? (oracle_gap, all 52) ===")
eb=DC[DC.detector=="empirical_bernstein"].set_index(["model","dataset"])["mean_oracle_gap"]
fa=DC[DC.detector=="first_answer"].set_index(["model","dataset"])["mean_oracle_gap"]
ns=DC[DC.detector=="never_stop"].set_index(["model","dataset"])["mean_oracle_gap"]
print(f"EB mean gap={eb.mean():.4f}  first_answer mean gap={fa.mean():.4f}  -> EB worse by {eb.mean()-fa.mean():+.4f}")
print(f"EB gap as fraction of never_stop gap: mean={ (eb/ns).mean():.3f}  (verdict: 76%)")
print(f"EB worse than first_answer in {int((eb>fa).sum())}/52 cells")

print("\n=== detector means across all 52 (recap) ===")
g=DC.groupby("detector").agg(gap=("mean_oracle_gap","mean"),fl=("false_late_rate","mean"),
    fe=("false_early_rate","mean"),step=("mean_stop_step","mean")).sort_values("gap")
print(g.round(4).to_string())

print("\n=== false_late: deployables (which catches overthinking best) ===")
for det in ["hazard_drift","answer_stability","entropy_plateau","first_answer","e_process","empirical_bernstein","never_stop"]:
    print(f"  {det:22s} false_late mean={DC[DC.detector==det].false_late_rate.mean():.4f}")

print("\n=== Mechanism: WHY does mu<=0 at t=2? alpha vs lambda at step2 ===")
LAM=0.05
n_alpha_decay=0; n_beta_low=0
a2_list=[]; b2_list=[]; mu2_list=[]
for m in MODELS:
    for d in DATASETS:
        f=os.path.join(ROOT,f"{m}__{d}","hazard_drift_summary.csv")
        df=pd.read_csv(f).sort_values("step")
        s2=df[df.step==2]
        a2=s2.repair_rate.values[0]; b2=s2.corruption_rate.values[0]; q2=s2.q_t.values[0]
        mu2=s2.hazard_mu.values[0]
        a2_list.append(a2); b2_list.append(b2); mu2_list.append(mu2)
print(f"at step2: mean alpha(repair)={np.mean(a2_list):.3f}  mean beta(corruption)={np.mean(b2_list):.3f}  mean mu={np.mean(mu2_list):.3f}")
# decompose: is mu<=0 at t=2 driven by (1-q)alpha < q*beta+lambda?
# count cells where corruption term q*beta dominates vs lambda dominates the negativity
print("at step2: among the 40 cells with mu<=0, is it lambda or beta that kills mu?")
lam_kills=0; beta_kills=0
for m in MODELS:
    for d in DATASETS:
        f=os.path.join(ROOT,f"{m}__{d}","hazard_drift_summary.csv")
        df=pd.read_csv(f).sort_values("step")
        s2=df[df.step==2]
        a2=s2.repair_rate.values[0]; b2=s2.corruption_rate.values[0]; q2=s2.q_t.values[0]
        mu2=s2.hazard_mu.values[0]
        if mu2<=0:
            gain=(1-q2)*a2
            corrupt=q2*b2
            # would mu be >0 without lambda? gain - corrupt > 0 ?
            if gain-corrupt>0:  # only lambda makes it negative
                lam_kills+=1
            else:  # corruption term alone already exceeds repair
                beta_kills+=1
print(f"  lambda alone makes mu<=0 (repair>corruption but <lambda): {lam_kills}")
print(f"  corruption>=repair even before lambda: {beta_kills}")

print("\n=== oracle stop step in late cells (the 'late boundary at 5-6' claim) ===")
orc=DC[(DC.detector=="oracle")].set_index(["model","dataset"])["mean_stop_step"]
orc_late=orc[orc.index.isin(lateset)]
print(f"oracle per-trace mean stop step in late cells: mean={orc_late.mean():.3f} max={orc_late.max():.3f}")
print(f"oracle per-trace mean stop step ALL 52: mean={orc.mean():.3f} max={orc.max():.3f}")
