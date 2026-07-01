import os
import pandas as pd, numpy as np

ROOT = r"C:/Aditya_Data/Personal/ResearchThesis/research/outputs/experiment_matrix"
MODELS = ["deepseek_r1_distill_1p5b","deepseek_r1_distill_7b","qwen2p5_0p5b","qwen2p5_3b",
"qwen2p5_7b","qwen2p5_14b","qwen2p5_32b","mistral_7b_instruct_v0p3","phi_4_mini_instruct",
"internlm3_8b_instruct","yi_1p5_9b_chat","mistral_small_24b_2409","llama_3p1_8b_instruct"]
DATASETS = ["gsm8k","math","arc","gpqa"]

# Independently derive late cells: corrected boundary T* = inf{t>=2: hazard_mu<=0}, late iff T*>2
def boundary(df,floor=2):
    sub=df[df.step>=floor].sort_values("step")
    neg=sub[sub.hazard_mu<=0]
    if len(neg)==0: return int(df.step.max())
    return int(neg.iloc[0].step)

late_derived=[]
allB=[]
for m in MODELS:
    for d in DATASETS:
        f=os.path.join(ROOT,f"{m}__{d}","hazard_drift_summary.csv")
        if not os.path.exists(f): continue
        df=pd.read_csv(f).sort_values("step")
        T=boundary(df)
        allB.append((m,d,T))
        if T>2: late_derived.append((m,d))
print(f"=== INDEPENDENTLY DERIVED late cells (corrected T*>2): {len(late_derived)} ===")
for m,d in late_derived: print(f"   {m}__{d}  (T*={dict(((a,b),t) for a,b,t in allB)[(m,d)]})")

# prior analyst hardcoded list
prior_late=[("qwen2p5_3b","gsm8k"),("qwen2p5_7b","gsm8k"),("qwen2p5_7b","math"),("qwen2p5_14b","gsm8k"),
("qwen2p5_32b","gsm8k"),("qwen2p5_32b","math"),("mistral_7b_instruct_v0p3","gsm8k"),
("internlm3_8b_instruct","gsm8k"),("yi_1p5_9b_chat","gsm8k"),("mistral_small_24b_2409","gsm8k"),
("mistral_small_24b_2409","arc"),("llama_3p1_8b_instruct","gsm8k")]
sd=set(late_derived); sp=set(prior_late)
print(f"\nprior hardcoded list size={len(sp)}; derived size={len(sd)}")
print(f"in derived but NOT prior: {sd-sp}")
print(f"in prior but NOT derived: {sp-sd}")

# Load detector_comparison for the DERIVED late cells and re-run head-to-head
def load_dc(cells):
    rows=[]
    for m,d in cells:
        f=os.path.join(ROOT,f"{m}__{d}","detector_comparison.csv")
        df=pd.read_csv(f); df["model"]=m; df["dataset"]=d; rows.append(df)
    return pd.concat(rows,ignore_index=True)

THEORY=["hazard_drift","e_process","empirical_bernstein"]
HEUR=["answer_stability","entropy_plateau","first_answer"]

for label,cells in [("DERIVED late cells",late_derived),("PRIOR hardcoded late cells",prior_late)]:
    L=load_dc(cells)
    print("\n"+"="*70)
    print(f"HEAD-TO-HEAD on {label} (n={len(cells)})")
    print("="*70)
    piv=L.pivot_table(index=["model","dataset"],columns="detector",values="mean_oracle_gap")
    # hazard_drift vs answer_stability
    hd=piv["hazard_drift"]; ans=piv["answer_stability"]; fa=piv["first_answer"]; ent=piv["entropy_plateau"]
    print(f"hazard_drift mean gap={hd.mean():.4f}  answer_stability={ans.mean():.4f}  first_answer={fa.mean():.4f}  entropy_plateau={ent.mean():.4f}")
    print(f"hazard_drift beats answer_stability: {int((hd<ans).sum())}W / {int((hd>ans).sum())}L (mean gap diff hd-ans = {(hd-ans).mean():+.4f})")
    print(f"hazard_drift beats first_answer:     {int((hd<fa).sum())}W / {int((hd>fa).sum())}L")
    print(f"hazard_drift beats entropy_plateau:  {int((hd<ent).sum())}W / {int((hd>ent).sum())}L")
    # best deployable
    deployable=THEORY+HEUR+["never_stop"]
    sub=L[L.detector.isin(deployable)].copy()
    idx=sub.groupby(["model","dataset"])["mean_oracle_gap"].idxmin()
    best=sub.loc[idx]
    print(f"best deployable counts: {dict(best.detector.value_counts())}")
    # hazard_drift dominates all 3 heuristics
    dom=((hd<fa)&(hd<ans)&(hd<ent)).sum()
    print(f"hazard_drift dominates ALL 3 heuristics simultaneously: {int(dom)}/{len(cells)}")
    # hazard_drift is single best deployable
    hd_best=int((best.detector=="hazard_drift").sum())
    ans_best=int((best.detector=="answer_stability").sum())
    print(f"hazard_drift single best deployable: {hd_best}/{len(cells)}; answer_stability best: {ans_best}/{len(cells)}")
    # verifier (cheat) gap
    vfc=L[L.detector=="verifier_first_correct"].set_index(["model","dataset"])["mean_oracle_gap"]
    print(f"verifier_first_correct (CHEATS) mean gap: {vfc.mean():.4f}")
