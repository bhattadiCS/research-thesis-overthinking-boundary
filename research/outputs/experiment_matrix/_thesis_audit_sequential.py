import os
import pandas as pd, numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = ["deepseek_r1_distill_1p5b","deepseek_r1_distill_7b","qwen2p5_0p5b","qwen2p5_3b",
"qwen2p5_7b","qwen2p5_14b","qwen2p5_32b","mistral_7b_instruct_v0p3","phi_4_mini_instruct",
"internlm3_8b_instruct","yi_1p5_9b_chat","mistral_small_24b_2409","llama_3p1_8b_instruct"]
DATASETS = ["gsm8k","math","arc","gpqa"]

# ---- Claim (b): anytime-valid false-early control at level delta ----
# detector_comparison reports false_early_rate per detector. Theory promises EB & e_process
# control false-early at delta. What delta? schedule delta_t = 6*delta/(pi^2 (t+1)^2). Whatever delta,
# the practical question: do EB / e_process achieve LOW false-early? and at what false-late cost?
rows=[]
for m in MODELS:
    for d in DATASETS:
        f=os.path.join(ROOT,f"{m}__{d}","detector_comparison.csv")
        if not os.path.exists(f): continue
        df=pd.read_csv(f); df=df.set_index("detector")
        for det in ["empirical_bernstein","e_process","hazard_drift","oracle","never_stop"]:
            if det in df.index:
                rows.append(dict(model=m,dataset=d,detector=det,
                    false_early=df.loc[det,"false_early_rate"],
                    false_late=df.loc[det,"false_late_rate"],
                    oracle_gap=df.loc[det,"mean_oracle_gap"],
                    stop_step=df.loc[det,"mean_stop_step"]))
S=pd.DataFrame(rows)
print("===== CLAIM (b): false-early control of anytime-valid rules =====")
for det in ["empirical_bernstein","e_process","hazard_drift"]:
    sub=S[S.detector==det]
    print(f"\n{det}:")
    print(f"  false_early: mean={sub.false_early.mean():.4f} max={sub.false_early.max():.4f} "
          f"(cells with FE>0.05: {int((sub.false_early>0.05).sum())}/{len(sub)}, FE>0.10: {int((sub.false_early>0.10).sum())})")
    print(f"  false_late : mean={sub.false_late.mean():.4f} (this is the COST: missed overthinking)")
    print(f"  oracle_gap : mean={sub.oracle_gap.mean():.4f}")

# Is EB effectively never-stop? compare stop_step & oracle_gap to never_stop
print("\n===== Is empirical_bernstein effectively do-nothing? =====")
piv=S.pivot_table(index=["model","dataset"],columns="detector",values="stop_step")
print("mean stop_step: EB=%.2f  never_stop=%.2f  (max possible step ~ 7-13)"%(piv["empirical_bernstein"].mean(),piv["never_stop"].mean()))
gappiv=S.pivot_table(index=["model","dataset"],columns="detector",values="oracle_gap")
print("EB oracle_gap as %% of never_stop gap: mean=%.1f%%"%(100*(gappiv["empirical_bernstein"]/gappiv["never_stop"]).mean()))

# eb_stop_signal / e_process_stop_signal: when does the bound first fire (from hazard_drift_summary)?
print("\n===== When do EB / e-process bounds FIRST fire vs corrected boundary? =====")
recs=[]
for m in MODELS:
    for d in DATASETS:
        f=os.path.join(ROOT,f"{m}__{d}","hazard_drift_summary.csv")
        if not os.path.exists(f): continue
        df=pd.read_csv(f).sort_values("step")
        sub=df[df.step>=2]
        ebfire=sub[sub.eb_stop_signal==1].step.min() if (sub.eb_stop_signal==1).any() else np.nan
        epfire=sub[sub.e_process_stop_signal==1].step.min() if (sub.e_process_stop_signal==1).any() else np.nan
        mufire=sub[sub.hazard_mu<=0].step.min() if (sub.hazard_mu<=0).any() else np.nan
        recs.append(dict(model=m,dataset=d,mu_cross=mufire,eb_fire=ebfire,ep_fire=epfire,maxstep=int(df.step.max())))
F=pd.DataFrame(recs)
print("mean first-fire step: mu_cross=%.2f  eb=%.2f  ep=%.2f"%(F.mu_cross.mean(),F.eb_fire.mean(),F.ep_fire.mean()))
print("EB fires LATER than mu-crossing by (steps): mean=%.2f"%((F.eb_fire-F.mu_cross).mean()))
print("e-process fires vs mu-crossing by (steps): mean=%.2f"%((F.ep_fire-F.mu_cross).mean()))
# on late cells specifically
late=[("qwen2p5_3b","gsm8k"),("qwen2p5_7b","gsm8k"),("qwen2p5_7b","math"),("qwen2p5_14b","gsm8k"),
("qwen2p5_32b","gsm8k"),("qwen2p5_32b","math"),("mistral_7b_instruct_v0p3","gsm8k"),
("internlm3_8b_instruct","gsm8k"),("yi_1p5_9b_chat","gsm8k"),("mistral_small_24b_2409","gsm8k"),
("mistral_small_24b_2409","arc"),("llama_3p1_8b_instruct","gsm8k")]
Fl=F.set_index(["model","dataset"]).loc[late]
print("\nON 12 LATE-BOUNDARY CELLS:")
print(Fl[["mu_cross","eb_fire","ep_fire"]].round(2).to_string())

# ---- verify thesis boundary table numbers (peak acc & AUC) ----
print("\n===== VERIFY thesis Table 3 boundary/peak/AUC against source =====")
chk=[]
for m in MODELS:
    for d in DATASETS:
        hf=os.path.join(ROOT,f"{m}__{d}","hazard_drift_summary.csv")
        pf=os.path.join(ROOT,f"{m}__{d}","correctness_probe_metrics.csv")
        if not (os.path.exists(hf) and os.path.exists(pf)): continue
        hd=pd.read_csv(hf); pm=pd.read_csv(pf)
        peak=hd.q_t.max()
        pooled=pm[pm.scope=="pooled_oof"]
        auc=pooled.auc.values[0] if len(pooled) else np.nan
        chk.append(dict(model=m,dataset=d,peak_acc=round(peak,2),pooled_auc=round(auc,2)))
C=pd.DataFrame(chk)
print(C.pivot(index="model",columns="dataset",values="pooled_auc").to_string())
print("\nPeak accuracy by cell (compare to thesis table):")
print(C.pivot(index="model",columns="dataset",values="peak_acc").to_string())
