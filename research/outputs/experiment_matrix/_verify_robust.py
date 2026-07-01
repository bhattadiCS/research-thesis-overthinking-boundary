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
        df = pd.read_csv(path); df["model"]=m; df["dataset"]=ds
        rows.append(df)
allc = pd.concat(rows, ignore_index=True)

# Is there an n column to weight? check columns
print("Columns:", list(allc.columns))

# GSM8K verdict: "hazard_drift achieves a LOW oracle_gap (best genuine ... on gsm8k/arc/gpqa, #2 on math;
#   pooled gap 0.185, just behind degenerate first_answer at 0.182)"
# verify pooled hazard_drift vs first_answer
pooled = allc.groupby("detector")["mean_oracle_gap"].mean().sort_values()
print("\nPooled gap (re-derived):")
print(pooled.round(4).to_dict())

# Check the claim "best genuine detector in 18/52 individual cells" -- already 18 confirmed.
# But which definition of genuine? If e_process is included it's 23 for e_process. Verdict says hazard 18.
# So verdict counts hazard_drift=18 best cells -- but does NOT prominently report e_process=23 best cells.

# verifier_first_correct: verdict says "NOT a clean upper bound ... pooled gap 0.166 ... on MATH worse (0.402, stop 11.08)"
vfc = allc[allc["detector"]=="verifier_first_correct"]
print("\nverifier_first_correct gap by dataset:", vfc.groupby("dataset")["mean_oracle_gap"].mean().round(3).to_dict())
print("verifier_first_correct stop by dataset:", vfc.groupby("dataset")["mean_stop_step"].mean().round(2).to_dict())
print("verifier_first_correct pooled gap:", round(vfc["mean_oracle_gap"].mean(),4))

# On MATH, how many deployable (genuine) detectors beat verifier on gap?
math = allc[allc["dataset"]=="math"].groupby("detector")["mean_oracle_gap"].mean().sort_values()
print("\nMATH gap ranking:", math.round(3).to_dict())
vfc_math = math["verifier_first_correct"]
GENUINE = ["hazard_drift","e_process","empirical_bernstein","entropy_plateau","answer_stability"]
beat = [d for d in GENUINE if math[d] < vfc_math]
print(f"Genuine detectors beating verifier on MATH gap: {beat}")

# ARC: verifier has NEGATIVE gap (-0.009) and fe=0.66. Note arc verifier gap < oracle? gap is mean over models.
# That's possible if utility-stop > oracle in some cells (oracle is per-task argmax of EXPECTED utility, realized can differ)
arc = allc[allc["dataset"]=="arc"]
print("\nARC verifier gap:", round(arc[arc.detector=='verifier_first_correct']['mean_oracle_gap'].mean(),4),
      "fe:", round(arc[arc.detector=='verifier_first_correct']['false_early_rate'].mean(),3))

# Claim: e_process "best on math". Confirm e_process is best GENUINE on math (0.203 < hazard 0.225). YES.
# Claim: "empirical_bernstein barely stops". stop ratio 0.59-0.85. fl 0.62-0.92. confirmed earlier.

# Claim "misses 24-59% of overthinking" = false_late range across datasets for best deployable detector.
# hazard_drift fl: gpqa 0.245, arc 0.329, math 0.564, gsm8k 0.591. Range 0.24-0.59. matches.
# But on MATH e_process is the 'best' detector and its fl=0.512. So if best-per-dataset, range is
# e_process(math) fl=0.51, hazard gpqa 0.245 ... -> 0.24-0.59 still holds via gsm8k hazard 0.591.
print("\nBest-genuine-per-dataset false_late:")
for ds in DATASETS:
    sub = allc[allc.dataset==ds].groupby("detector")[["mean_oracle_gap","false_late_rate"]].mean()
    sub = sub[sub.index.isin(GENUINE)].sort_values("mean_oracle_gap")
    bestdet = sub.index[0]
    print(f"  {ds:6s} best={bestdet:16s} fl={sub.iloc[0]['false_late_rate']:.3f}")

# Robustness: pooled ranking is unweighted mean over 52 cells. Each cell already a mean over tasks.
# Verify equal-ish task counts via trace_steps if available -> check pilot_summary for n
print("\n--- check a pilot_summary for task count ---")
for c in ["qwen2p5_7b__gsm8k","qwen2p5_7b__math"]:
    p = os.path.join(ROOT,c,"pilot_summary.csv")
    if os.path.exists(p):
        ps = pd.read_csv(p)
        print(c, "cols:", list(ps.columns)[:8])
        print(ps.head(2).to_string())
