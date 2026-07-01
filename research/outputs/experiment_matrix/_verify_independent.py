import os
import pandas as pd, numpy as np

ROOT = r"C:/Aditya_Data/Personal/ResearchThesis/research/outputs/experiment_matrix"
MODELS = ["deepseek_r1_distill_1p5b","deepseek_r1_distill_7b","qwen2p5_0p5b","qwen2p5_3b",
"qwen2p5_7b","qwen2p5_14b","qwen2p5_32b","mistral_7b_instruct_v0p3","phi_4_mini_instruct",
"internlm3_8b_instruct","yi_1p5_9b_chat","mistral_small_24b_2409","llama_3p1_8b_instruct"]
DATASETS = ["gsm8k","math","arc","gpqa"]
LAM=0.05

# ---- Load detector_comparison for all cells ----
dc_rows=[]
missing_dc=[]
for m in MODELS:
    for d in DATASETS:
        f=os.path.join(ROOT,f"{m}__{d}","detector_comparison.csv")
        if not os.path.exists(f):
            missing_dc.append(f"{m}__{d}"); continue
        df=pd.read_csv(f); df["model"]=m; df["dataset"]=d
        dc_rows.append(df)
DC=pd.concat(dc_rows,ignore_index=True)
ncells=DC[["model","dataset"]].drop_duplicates().shape[0]
print(f"=== COVERAGE: {ncells} cells with detector_comparison.csv; missing: {missing_dc} ===")
# check for extra dirs
import glob
alldirs=[os.path.basename(p) for p in glob.glob(os.path.join(ROOT,"*__*")) if os.path.isdir(p)]
expected=set(f"{m}__{d}" for m in MODELS for d in DATASETS)
extra=[x for x in alldirs if x not in expected and "__" in x]
print(f"extra cell-like dirs not in 13x4 list: {extra}")
for x in extra:
    has_dc=os.path.exists(os.path.join(ROOT,x,"detector_comparison.csv"))
    print(f"   {x}: detector_comparison.csv exists={has_dc}")

# ---- Load hazard summaries for all cells ----
HZ={}
for m in MODELS:
    for d in DATASETS:
        f=os.path.join(ROOT,f"{m}__{d}","hazard_drift_summary.csv")
        if os.path.exists(f):
            HZ[(m,d)]=pd.read_csv(f).sort_values("step")

print("\n"+"#"*80)
print("# CLAIM (a) MECHANISM: beta direction, alpha direction, Theorem-1 premises")
print("#"*80)

beta_falls=beta_rises=0
alpha_falls=0
q_up_full=0; a_down_full=0; b_up_full=0; mu_down_full=0
beta_pair_frac=[]; q_pair_frac=[]; a_pair_frac=[]; mu_pair_frac=[]
fitted_beta_falls=0; fitted_b_up_full=0
for (m,d),df in HZ.items():
    sub=df[df.step>=2]
    # fitted beta direction (overall step1->last)
    fb=df.sort_values("step")["fitted_corruption_hazard"].values
    if fb[-1] < fb[0]: fitted_beta_falls+=1
    # empirical beta direction step2 -> last (use corruption_rate = conditional beta)
    b=sub["corruption_rate"].values
    if b[-1] < b[0]: beta_falls+=1
    elif b[-1] > b[0]: beta_rises+=1
    a=sub["repair_rate"].values
    if a[-1] < a[0]: alpha_falls+=1
    q=sub["q_t"].values
    mu=sub["hazard_mu"].values
    # full monotonicity (t>=2)
    if np.all(np.diff(q)>=-1e-9): q_up_full+=1
    if np.all(np.diff(a)<=1e-9): a_down_full+=1
    if np.all(np.diff(b)>=-1e-9): b_up_full+=1
    if np.all(np.diff(mu)<=1e-9): mu_down_full+=1
    if np.all(np.diff(df.sort_values("step")["fitted_corruption_hazard"].values)>=-1e-9): fitted_b_up_full+=1
    # fraction of step-pairs satisfying each
    q_pair_frac.append(np.mean(np.diff(q)>=-1e-9))
    a_pair_frac.append(np.mean(np.diff(a)<=1e-9))
    beta_pair_frac.append(np.mean(np.diff(b)>=-1e-9))
    mu_pair_frac.append(np.mean(np.diff(mu)<=1e-9))
N=len(HZ)
print(f"cells with hazard summary: {N}")
print(f"empirical beta (corruption_rate) step2->last FALLS in {beta_falls}/{N}, RISES in {beta_rises}/{N}")
print(f"fitted_corruption_hazard step1->last FALLS in {fitted_beta_falls}/{N}")
print(f"empirical alpha (repair_rate) step2->last FALLS in {alpha_falls}/{N}")
print(f"--- Theorem 1 premises (fully holds across t>=2 step-pairs) ---")
print(f"q_t nondecreasing fully: {q_up_full}/{N}  (mean pair-frac {np.mean(q_pair_frac):.3f})")
print(f"alpha nonincreasing fully: {a_down_full}/{N}  (mean pair-frac {np.mean(a_pair_frac):.3f})")
print(f"beta nondecreasing fully (empirical): {b_up_full}/{N}  (mean pair-frac {np.mean(beta_pair_frac):.3f})")
print(f"beta nondecreasing fully (fitted): {fitted_b_up_full}/{N}")
print(f"mu nonincreasing fully (conclusion): {mu_down_full}/{N}  (mean pair-frac {np.mean(mu_pair_frac):.3f})")

print("\n"+"#"*80)
print("# CLAIM (a) CONTINUATION WINDOW: mu<=0 already at t=2? one-crossing?")
print("#"*80)
mu_neg_at2=0; has_pos_after2=0
sign_changes_dist={0:0,1:0,2:0,"ge2":0}
one_crossing_ok=0
for (m,d),df in HZ.items():
    sub=df[df.step>=2].sort_values("step")
    mu=sub["hazard_mu"].values
    if mu[0]<=0: mu_neg_at2+=1
    if np.any(mu>0): has_pos_after2+=1
    signs=np.sign(mu)
    # count down and up crossings
    down=int(np.sum((signs[:-1]>0)&(signs[1:]<=0)))
    up=int(np.sum((signs[:-1]<=0)&(signs[1:]>0)))
    total_cross=down+up
    if total_cross==0: sign_changes_dist[0]+=1
    elif total_cross==1: sign_changes_dist[1]+=1
    elif total_cross==2: sign_changes_dist[2]+=1
    else: sign_changes_dist["ge2"]+=1
    if total_cross<=1: one_crossing_ok+=1
print(f"hazard_mu <= 0 already at t=2 in {mu_neg_at2}/{N} cells")
print(f"cells with ANY step>=2 where mu>0 (genuine continuation window): {has_pos_after2}/{N}")
print(f"sign-change distribution of mu (t>=2): {sign_changes_dist}")
print(f"one-crossing (<=1 sign change) holds in {one_crossing_ok}/{N}")

# Oracle mean stop step across cells
orac=DC[DC.detector=="oracle"]
print(f"\noracle mean_stop_step: mean={orac.mean_stop_step.mean():.3f} max={orac.mean_stop_step.max():.3f} min={orac.mean_stop_step.min():.3f}")

print("\n"+"#"*80)
print("# CLAIM (b) ANYTIME-VALID: e_process & EB false_early vs delta=0.05")
print("#"*80)
ep=DC[DC.detector=="e_process"].set_index(["model","dataset"])
eb=DC[DC.detector=="empirical_bernstein"].set_index(["model","dataset"])
hd=DC[DC.detector=="hazard_drift"].set_index(["model","dataset"])
ns=DC[DC.detector=="never_stop"].set_index(["model","dataset"])
fa=DC[DC.detector=="first_answer"].set_index(["model","dataset"])
orc=DC[DC.detector=="oracle"].set_index(["model","dataset"])

ep_viol=(ep.false_early_rate>0.05).sum()
print(f"e_process false_early > 0.05 in {ep_viol}/{len(ep)} cells; mean={ep.false_early_rate.mean():.4f} max={ep.false_early_rate.max():.4f}")
eb_viol=(eb.false_early_rate>0.05).sum()
print(f"EB false_early > 0.05 in {eb_viol}/{len(eb)} cells; mean={eb.false_early_rate.mean():.4f} max={eb.false_early_rate.max():.4f}")
hd_viol=(hd.false_early_rate>0.05).sum()
print(f"hazard_drift false_early > 0.05 in {hd_viol}/{len(hd)} cells; mean={hd.false_early_rate.mean():.4f} max={hd.false_early_rate.max():.4f}")

print(f"\n--- EB vacuity check ---")
print(f"EB mean_stop_step={eb.mean_stop_step.mean():.3f}  oracle={orc.mean_stop_step.mean():.3f}  never_stop={ns.mean_stop_step.mean():.3f}")
print(f"EB false_late={eb.false_late_rate.mean():.4f}  never_stop false_late={ns.false_late_rate.mean():.4f}")
print(f"EB oracle_gap={eb.mean_oracle_gap.mean():.4f}  first_answer oracle_gap={fa.mean_oracle_gap.mean():.4f}  never_stop gap={ns.mean_oracle_gap.mean():.4f}")
print(f"EB gap as % of never_stop gap: mean={100*(eb.mean_oracle_gap/ns.mean_oracle_gap).mean():.1f}%")
# forced to horizon
maxsteps={(m,d):int(df.step.max()) for (m,d),df in HZ.items()}
eb_forced=sum(1 for k in eb.index if k in maxsteps and abs(eb.loc[k,"mean_stop_step"]-maxsteps[k])<0.5)
print(f"EB forced to max horizon (stop==maxstep) in {eb_forced}/{len(eb)} cells")

print("\n"+"#"*80)
print("# CLAIM (c) BEST DEPLOYABLE DETECTOR (by oracle_gap), all 52")
print("#"*80)
THEORY=["hazard_drift","e_process","empirical_bernstein"]
HEUR=["answer_stability","entropy_plateau","first_answer"]
deployable=THEORY+HEUR+["never_stop"]
sub=DC[DC.detector.isin(deployable)].copy()
idx=sub.groupby(["model","dataset"])["mean_oracle_gap"].idxmin()
best=sub.loc[idx]
print("best deployable detector counts (all 52):")
print(best.detector.value_counts().to_string())
for det in ["hazard_drift","answer_stability","first_answer","entropy_plateau","e_process","empirical_bernstein"]:
    g=DC[DC.detector==det].mean_oracle_gap.mean()
    print(f"  mean oracle_gap {det}: {g:.4f}")

# probe AUC
print("\n"+"#"*80)
print("# CLAIM (c) PROBE AUC out-of-sample")
print("#"*80)
aucs=[]
for m in MODELS:
    for d in DATASETS:
        f=os.path.join(ROOT,f"{m}__{d}","correctness_probe_metrics.csv")
        if os.path.exists(f):
            pm=pd.read_csv(f)
            # find an auc column
            cols=[c for c in pm.columns if "auc" in c.lower()]
            if cols:
                # take mean oos auc if multiple rows; prefer column literally 'auc' or containing 'oos'/'test'
                val=None
                for pref in ["oos","test","val","cv"]:
                    cc=[c for c in cols if pref in c.lower()]
                    if cc: val=pm[cc[0]].mean(); break
                if val is None: val=pm[cols[0]].mean()
                aucs.append((m,d,val))
A=pd.DataFrame(aucs,columns=["model","dataset","auc"])
if len(A):
    print(f"probe AUC mean={A.auc.mean():.3f} min={A.auc.min():.3f} max={A.auc.max():.3f}")
    print(f"AUC<0.65 in {int((A.auc<0.65).sum())}/{len(A)} cells; AUC>=0.8 in {int((A.auc>=0.8).sum())}/{len(A)}")
else:
    print("no AUC columns found; columns sample:")
    f=os.path.join(ROOT,"qwen2p5_7b__gsm8k","correctness_probe_metrics.csv")
    print(pd.read_csv(f).columns.tolist())

DC.to_csv(os.path.join(ROOT,"_verify_DC_all.csv"),index=False)
print("\nsaved _verify_DC_all.csv")
