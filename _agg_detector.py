import csv, os, statistics as st

ROOT = "research/outputs/experiment_matrix"
MODELS = ["deepseek_r1_distill_1p5b","deepseek_r1_distill_7b","qwen2p5_0p5b","qwen2p5_3b",
          "qwen2p5_7b","qwen2p5_14b","qwen2p5_32b","mistral_7b_instruct_v0p3",
          "phi_4_mini_instruct","internlm3_8b_instruct","yi_1p5_9b_chat",
          "mistral_small_24b_2409","llama_3p1_8b_instruct"]
DATASETS = ["gsm8k","math","arc","gpqa"]
METRICS = ["mean_oracle_gap","false_late_rate","false_early_rate","mean_stop_step",
           "mean_stop_utility","mean_oracle_utility","mean_false_late_severity"]

# data[dataset][detector][metric] = list over models
data = {d:{} for d in DATASETS}
ncells = {d:0 for d in DATASETS}
for d in DATASETS:
    for m in MODELS:
        f = os.path.join(ROOT, f"{m}__{d}", "detector_comparison.csv")
        if not os.path.exists(f):
            print("MISSING", f); continue
        ncells[d]+=1
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                det = row["detector"]
                dd = data[d].setdefault(det, {k:[] for k in METRICS})
                for k in METRICS:
                    try: dd[k].append(float(row[k]))
                    except: pass

def mean(xs): return sum(xs)/len(xs) if xs else float("nan")

for d in DATASETS:
    print("\n" + "="*100)
    print(f"DATASET: {d.upper()}   (n_models aggregated = {ncells[d]})")
    print("="*100)
    rows=[]
    for det, mm in data[d].items():
        rows.append((det, mean(mm["mean_oracle_gap"]), mean(mm["false_late_rate"]),
                     mean(mm["false_early_rate"]), mean(mm["mean_stop_step"]),
                     mean(mm["mean_stop_utility"]), len(mm["mean_oracle_gap"])))
    rows.sort(key=lambda r: r[1])  # by oracle gap asc
    hdr = f"{'detector':<22}{'oracle_gap':>11}{'false_late':>11}{'false_early':>12}{'stop_step':>11}{'stop_util':>11}{'n':>4}"
    print(hdr); print("-"*len(hdr))
    for det,g,fl,fe,ss,su,n in rows:
        print(f"{det:<22}{g:>11.4f}{fl:>11.4f}{fe:>12.4f}{ss:>11.3f}{su:>11.4f}{n:>4}")

# Cross-dataset summary for key detectors
print("\n\n" + "#"*100)
print("CROSS-DATASET SUMMARY (mean over 4 datasets of the per-dataset model-mean)")
print("#"*100)
KEY = ["oracle","hazard_drift","e_process","empirical_bernstein","answer_stability",
       "entropy_plateau","first_answer","verifier_first_correct","never_stop"]
for det in KEY:
    gaps=[]; fls=[]; fes=[]; sss=[]
    for d in DATASETS:
        if det in data[d]:
            gaps.append(mean(data[d][det]["mean_oracle_gap"]))
            fls.append(mean(data[d][det]["false_late_rate"]))
            fes.append(mean(data[d][det]["false_early_rate"]))
            sss.append(mean(data[d][det]["mean_stop_step"]))
    print(f"{det:<22} gap={mean(gaps):.4f}  false_late={mean(fls):.4f}  false_early={mean(fes):.4f}  stop_step={mean(sss):.3f}")
