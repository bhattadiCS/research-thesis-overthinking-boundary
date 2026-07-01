import os, glob
import pandas as pd, numpy as np

ROOT = r"C:/Aditya_Data/Personal/ResearchThesis/research/outputs/experiment_matrix"
MODELS = ["deepseek_r1_distill_1p5b","deepseek_r1_distill_7b","qwen2p5_0p5b","qwen2p5_3b",
"qwen2p5_7b","qwen2p5_14b","qwen2p5_32b","mistral_7b_instruct_v0p3","phi_4_mini_instruct",
"internlm3_8b_instruct","yi_1p5_9b_chat","mistral_small_24b_2409","llama_3p1_8b_instruct"]
DATASETS = ["gsm8k","math","arc","gpqa"]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

rows=[]
for m in MODELS:
    for d in DATASETS:
        cell=os.path.join(ROOT,f"{m}__{d}")
        scf=os.path.join(cell,"sequential_detector_summary.csv")
        hf=os.path.join(cell,"hazard_drift_summary.csv")
        if not (os.path.exists(scf) and os.path.exists(hf)):
            continue
        sc=pd.read_csv(scf).set_index("detector")
        hd=pd.read_csv(hf).sort_values("step")
        maxstep=int(hd.step.max())
        # boundary T* = first step>=2 with hazard_mu<=0
        sub=hd[hd.step>=2]
        mu_cross=sub[sub.hazard_mu<=0].step.min() if (sub.hazard_mu<=0).any() else np.nan
        # first-fire step for eb / e_process signals (>=2)
        eb_fire=sub[sub.eb_stop_signal==1].step.min() if (sub.eb_stop_signal==1).any() else np.nan
        ep_fire=sub[sub.e_process_stop_signal==1].step.min() if (sub.e_process_stop_signal==1).any() else np.nan
        def g(det,col):
            return sc.loc[det,col] if det in sc.index else np.nan
        rows.append(dict(
            model=m, dataset=d, maxstep=maxstep,
            mu_cross=mu_cross, eb_fire=eb_fire, ep_fire=ep_fire,
            # oracle
            oracle_stop=g("oracle","mean_stop_step"),
            oracle_util=g("oracle","mean_stop_utility"),
            # e_process
            ep_stop=g("e_process","mean_stop_step"),
            ep_gap=g("e_process","mean_oracle_gap"),
            ep_fe=g("e_process","false_early_rate"),
            ep_fl=g("e_process","false_late_rate"),
            ep_fls=g("e_process","mean_false_late_severity"),
            ep_util=g("e_process","mean_stop_utility"),
            # empirical_bernstein
            eb_stop=g("empirical_bernstein","mean_stop_step"),
            eb_gap=g("empirical_bernstein","mean_oracle_gap"),
            eb_fe=g("empirical_bernstein","false_early_rate"),
            eb_fl=g("empirical_bernstein","false_late_rate"),
            eb_fls=g("empirical_bernstein","mean_false_late_severity"),
            eb_util=g("empirical_bernstein","mean_stop_utility"),
            # never_stop baseline
            ns_stop=g("never_stop","mean_stop_step"),
            ns_gap=g("never_stop","mean_oracle_gap"),
            ns_fl=g("never_stop","false_late_rate"),
            ns_util=g("never_stop","mean_stop_utility"),
            # hazard_drift (theory drift rule, for contrast)
            hd_stop=g("hazard_drift","mean_stop_step"),
            hd_gap=g("hazard_drift","mean_oracle_gap"),
            hd_fe=g("hazard_drift","false_early_rate"),
            hd_fl=g("hazard_drift","false_late_rate"),
            hd_util=g("hazard_drift","mean_stop_utility"),
        ))
S=pd.DataFrame(rows)
print(f"Cells loaded: {len(S)} / 52")
print(f"Max horizon distribution (maxstep): min={S.maxstep.min()}, max={S.maxstep.max()}, "
      f"mean={S.maxstep.mean():.2f}, mode={S.maxstep.mode().tolist()}")
print(f"  per value: {S.maxstep.value_counts().sort_index().to_dict()}")

print("\n" + "="*78)
print("Q1: DO THE BOUNDS STOP MEANINGFULLY EARLY, OR AT/NEAR MAX HORIZON?")
print("="*78)
# fraction of horizon used: mean_stop_step / maxstep
S["ep_frac"]=S.ep_stop/S.maxstep
S["eb_frac"]=S.eb_stop/S.maxstep
S["ns_frac"]=S.ns_stop/S.maxstep
S["oracle_frac"]=S.oracle_stop/S.maxstep
for det,fr,st in [("e_process","ep_frac","ep_stop"),("empirical_bernstein","eb_frac","eb_stop"),
                  ("never_stop","ns_frac","ns_stop"),("oracle","oracle_frac","oracle_stop")]:
    print(f"  {det:22s}: mean_stop={S[st].mean():.2f}  frac_of_horizon={S[fr].mean():.3f}  "
          f"(median frac={S[fr].median():.3f})")
# how often does EB stop == maxstep (i.e. effectively never stop / forced to horizon)?
print(f"\n  EB mean_stop == maxstep (forced to horizon) in {int((np.abs(S.eb_stop-S.maxstep)<0.5).sum())}/{len(S)} cells")
print(f"  EB mean_stop >= 0.9*maxstep in {int((S.eb_frac>=0.9).sum())}/{len(S)} cells")
print(f"  e_process mean_stop >= 0.9*maxstep in {int((S.ep_frac>=0.9).sum())}/{len(S)} cells")
print(f"  e_process mean_stop >= 0.8*maxstep in {int((S.ep_frac>=0.8).sum())}/{len(S)} cells")

print("\n" + "="*78)
print("Q2: FALSE-LATE RATE -- do bounds actually catch overthinking?")
print("="*78)
for det,fl,sev in [("e_process","ep_fl","ep_fls"),("empirical_bernstein","eb_fl","eb_fls"),
                   ("never_stop","ns_fl",None),("hazard_drift","hd_fl",None)]:
    extra=f"  mean_severity={S[sev].mean():.2f}" if sev else ""
    print(f"  {det:22s}: false_late mean={S[fl].mean():.4f}  median={S[fl].median():.4f}  "
          f"min={S[fl].min():.4f}  max={S[fl].max():.4f}{extra}")
    print(f"  {'':22s}  cells with FL>=0.95: {int((S[fl]>=0.95).sum())}/{len(S)};  FL>=0.99: {int((S[fl]>=0.99).sum())}/{len(S)}")

print("\n" + "="*78)
print("Q3: IS EB EFFECTIVELY never_stop (do-nothing)?")
print("="*78)
print(f"  mean_stop_step:    EB={S.eb_stop.mean():.3f}   never_stop={S.ns_stop.mean():.3f}")
print(f"  mean_oracle_gap:   EB={S.eb_gap.mean():.4f}  never_stop={S.ns_gap.mean():.4f}")
print(f"  EB gap as %% of never_stop gap: mean={100*(S.eb_gap/S.ns_gap).mean():.1f}%  median={100*(S.eb_gap/S.ns_gap).median():.1f}%")
print(f"  EB false_late:     EB={S.eb_fl.mean():.4f}  never_stop={S.ns_fl.mean():.4f}")
print(f"  EB utility:        EB={S.eb_util.mean():.4f}  never_stop={S.ns_util.mean():.4f}  oracle={S.oracle_util.mean():.4f}")
# how many cells is EB stop within 0.5 step of never_stop
print(f"  cells where |EB_stop - never_stop_stop| < 0.5: {int((np.abs(S.eb_stop-S.ns_stop)<0.5).sum())}/{len(S)}")
print(f"  cells where |EB_stop - never_stop_stop| < 1.0: {int((np.abs(S.eb_stop-S.ns_stop)<1.0).sum())}/{len(S)}")

print("\n" + "="*78)
print("Q4: DOES EITHER BOUND BEAT never_stop ON UTILITY / GAP? (the real test)")
print("="*78)
S["ep_beats_ns_util"]=S.ep_util>S.ns_util
S["eb_beats_ns_util"]=S.eb_util>S.ns_util
S["ep_gap_lt_ns"]=S.ep_gap<S.ns_gap
S["eb_gap_lt_ns"]=S.eb_gap<S.ns_gap
print(f"  e_process utility > never_stop utility: {int(S.ep_beats_ns_util.sum())}/{len(S)} cells "
      f"(mean util improvement={100*((S.ep_util-S.ns_util)/np.abs(S.ns_util)).mean():.1f}%)")
print(f"  EB        utility > never_stop utility: {int(S.eb_beats_ns_util.sum())}/{len(S)} cells "
      f"(mean util improvement={100*((S.eb_util-S.ns_util)/np.abs(S.ns_util)).mean():.1f}%)")
print(f"  e_process oracle_gap < never_stop gap : {int(S.ep_gap_lt_ns.sum())}/{len(S)} cells")
print(f"  EB        oracle_gap < never_stop gap : {int(S.eb_gap_lt_ns.sum())}/{len(S)} cells")
# vs oracle, fraction of achievable gap closed
S["ep_gap_closed"]=(S.ns_gap-S.ep_gap)/S.ns_gap   # 1.0 = closes all gap to oracle, 0 = same as never_stop
S["eb_gap_closed"]=(S.ns_gap-S.eb_gap)/S.ns_gap
print(f"\n  Fraction of never_stop->oracle gap CLOSED (1.0=oracle, 0=never_stop, <0=worse):")
print(f"    e_process: mean={S.ep_gap_closed.mean():.3f}  median={S.ep_gap_closed.median():.3f}")
print(f"    EB:        mean={S.eb_gap_closed.mean():.3f}  median={S.eb_gap_closed.median():.3f}")

print("\n" + "="*78)
print("Q5: WHEN DO BOUNDS FIRST FIRE vs boundary T* (mu-crossing)? (per-step signal)")
print("="*78)
valid=S.dropna(subset=["mu_cross"])
print(f"  cells with a mu-crossing (T* exists): {len(valid)}/{len(S)}")
print(f"  mean first-fire step:  mu_cross(T*)={valid.mu_cross.mean():.2f}  "
      f"eb_fire={valid.eb_fire.mean():.2f}  ep_fire={valid.ep_fire.mean():.2f}  (mean maxstep={valid.maxstep.mean():.2f})")
print(f"  EB fires LATER than T* by: mean={ (valid.eb_fire-valid.mu_cross).mean():.2f} steps "
      f"(median={ (valid.eb_fire-valid.mu_cross).median():.2f})")
print(f"  e_process fires vs T* by:  mean={ (valid.ep_fire-valid.mu_cross).mean():.2f} steps "
      f"(median={ (valid.ep_fire-valid.mu_cross).median():.2f})")
print(f"  EB never fires (no eb_stop_signal at any step>=2): {int(S.eb_fire.isna().sum())}/{len(S)} cells")
print(f"  e_process never fires: {int(S.ep_fire.isna().sum())}/{len(S)} cells")

# Save full table
out=os.path.join(ROOT,"_anytime_valid_audit_table.csv")
S.round(4).to_csv(out,index=False)
print(f"\nFull per-cell table -> {out}")

# Per-dataset breakdown
print("\n" + "="*78)
print("PER-DATASET BREAKDOWN (mean across models)")
print("="*78)
agg=S.groupby("dataset").agg(
    maxstep=("maxstep","mean"),
    ep_stop=("ep_stop","mean"), ep_fl=("ep_fl","mean"), ep_gap=("ep_gap","mean"), ep_fe=("ep_fe","mean"),
    eb_stop=("eb_stop","mean"), eb_fl=("eb_fl","mean"), eb_gap=("eb_gap","mean"), eb_fe=("eb_fe","mean"),
    ns_stop=("ns_stop","mean"), ns_gap=("ns_gap","mean"),
    oracle_stop=("oracle_stop","mean"),
).round(3)
print(agg.to_string())
