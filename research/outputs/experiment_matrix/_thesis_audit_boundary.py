import os, glob
import pandas as pd, numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = ["deepseek_r1_distill_1p5b","deepseek_r1_distill_7b","qwen2p5_0p5b","qwen2p5_3b",
"qwen2p5_7b","qwen2p5_14b","qwen2p5_32b","mistral_7b_instruct_v0p3","phi_4_mini_instruct",
"internlm3_8b_instruct","yi_1p5_9b_chat","mistral_small_24b_2409","llama_3p1_8b_instruct"]
DATASETS = ["gsm8k","math","arc","gpqa"]
LAM=0.05

# Compute corrected boundary T* = inf{t>=2 : hazard_mu <= 0} from hazard_by_step.csv (conditional hazard_mu column)
def boundary_from(df, col="hazard_mu", floor=2):
    sub=df[df.step>=floor].sort_values("step")
    neg=sub[sub[col]<=0]
    if len(neg)==0: return df.step.max()
    return int(neg.iloc[0].step)

recs=[]
for m in MODELS:
    for d in DATASETS:
        f=os.path.join(ROOT,f"{m}__{d}","hazard_drift_summary.csv")
        if not os.path.exists(f): continue
        df=pd.read_csv(f)
        # corrected boundary uses conditional hazard drift (hazard_mu == conditional_hazard_drift here)
        T=boundary_from(df,"hazard_mu",2)
        # alpha/beta at step 2 (first valid)
        s2=df[df.step==2]
        a=s2.repair_rate.values[0] if len(s2) else np.nan
        b=s2.corruption_rate.values[0] if len(s2) else np.nan
        # peak q_t and step1 q
        q1=df[df.step==1].q_t.values[0] if len(df[df.step==1]) else np.nan
        qpeak=df.q_t.max()
        recs.append(dict(model=m,dataset=d,boundary=T,maxstep=int(df.step.max()),
                         alpha2=a,beta2=b,ab_ratio=(a/b if b and b>0 else np.nan),
                         q1=q1,qpeak=qpeak,headroom=qpeak-q1,late=(T>2)))
B=pd.DataFrame(recs)
pd.set_option("display.width",220); pd.set_option("display.max_columns",30); pd.set_option("display.max_rows",100)
print("===== CORRECTED BOUNDARY per cell (from conditional hazard_mu, floor t>=2) =====")
print(B.round(3).to_string(index=False))

print("\n===== boundary distribution =====")
print("late-boundary cells (T>2):", int(B.late.sum()), "of", len(B))
print(B.groupby("dataset").late.sum())
print("\nlate cells list:")
print(B[B.late][["model","dataset","boundary","alpha2","beta2","headroom"]].round(3).to_string(index=False))

# Monotone one-crossing check: does conditional hazard_mu actually cross zero ONCE and stay negative?
print("\n===== ONE-CROSSING / MONOTONICITY CHECK (Theorem 1 empirical) =====")
viol=[]
nonmono=[]
for m in MODELS:
    for d in DATASETS:
        f=os.path.join(ROOT,f"{m}__{d}","hazard_drift_summary.csv")
        if not os.path.exists(f): continue
        df=pd.read_csv(f).sort_values("step")
        mu=df[df.step>=2]["hazard_mu"].values
        # count sign changes pos->neg and neg->pos among steps>=2
        signs=np.sign(mu)
        crossings_down=int(np.sum((signs[:-1]>0)&(signs[1:]<=0)))
        crossings_up=int(np.sum((signs[:-1]<=0)&(signs[1:]>0)))
        is_monotone=bool(np.all(np.diff(mu)<=1e-9))
        if crossings_up>0: viol.append((m,d,crossings_down,crossings_up))
        if not is_monotone: nonmono.append((m,d))
print(f"cells where mu (t>=2) is exactly nonincreasing (Theorem1 holds empirically): {52-len(nonmono)}/52")
print(f"cells with >=1 neg->pos crossing of mu after t>=2 (one-crossing VIOLATED): {len(viol)}/52")
print("examples of one-crossing violations (model,dataset,down,up):", viol[:12])

# Check the three monotonicity PREMISES of Theorem 1: q nondecr, alpha nonincr, beta nondecr
print("\n===== THEOREM 1 PREMISE CHECK (q up, alpha down, beta up), t>=2 =====")
pq=pa=pb=0; tot=0
for m in MODELS:
    for d in DATASETS:
        f=os.path.join(ROOT,f"{m}__{d}","hazard_drift_summary.csv")
        if not os.path.exists(f): continue
        df=pd.read_csv(f).sort_values("step")
        sub=df[df.step>=2]
        tot+=1
        if np.all(np.diff(sub.q_t.values)>=-1e-9): pq+=1
        if np.all(np.diff(sub.repair_rate.values)<=1e-9): pa+=1
        if np.all(np.diff(sub.corruption_rate.values)>=-1e-9): pb+=1
print(f"q_t nondecreasing: {pq}/{tot}")
print(f"alpha (repair_rate) nonincreasing: {pa}/{tot}")
print(f"beta (corruption_rate) nondecreasing: {pb}/{tot}")
B.to_csv(os.path.join(ROOT,"_thesis_boundary_table.csv"),index=False)
