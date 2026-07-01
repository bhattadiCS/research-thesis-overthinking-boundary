import csv, os

ROOT = "research/outputs/experiment_matrix"
MODELS = ["deepseek_r1_distill_1p5b","deepseek_r1_distill_7b","qwen2p5_0p5b","qwen2p5_3b",
          "qwen2p5_7b","qwen2p5_14b","qwen2p5_32b","mistral_7b_instruct_v0p3",
          "phi_4_mini_instruct","internlm3_8b_instruct","yi_1p5_9b_chat",
          "mistral_small_24b_2409","llama_3p1_8b_instruct"]
DATASETS = ["gsm8k","math","arc","gpqa"]

# per-cell gap: cell[(model,dataset)][detector] = oracle_gap
cell = {}
for d in DATASETS:
    for m in MODELS:
        f = os.path.join(ROOT, f"{m}__{d}", "detector_comparison.csv")
        if not os.path.exists(f): continue
        dd = cell.setdefault((m,d), {})
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                dd[row["detector"]] = (float(row["mean_oracle_gap"]),
                                       float(row["false_late_rate"]),
                                       float(row["false_early_rate"]))

DEPLOYABLE_HEURS = ["answer_stability","entropy_plateau","first_answer"]
THEORY = ["hazard_drift","e_process","empirical_bernstein"]

print("HEAD-TO-HEAD: hazard_drift vs each deployable heuristic, per-cell oracle_gap (lower=better)")
print("(win = hazard_drift strictly lower gap). 52 cells total.\n")
for h in DEPLOYABLE_HEURS:
    win=loss=tie=0; gap_diff=[]
    for k,v in cell.items():
        if "hazard_drift" in v and h in v:
            a=v["hazard_drift"][0]; b=v[h][0]
            gap_diff.append(a-b)
            if a<b-1e-9: win+=1
            elif a>b+1e-9: loss+=1
            else: tie+=1
    print(f"hazard_drift vs {h:<18}: win={win:>2} loss={loss:>2} tie={tie:>2}  mean(gap_hd - gap_{h})={sum(gap_diff)/len(gap_diff):+.4f}")

print("\nBest deployable theory detector = hazard_drift. Compare to best heuristic per cell:")
win=loss=tie=0
for k,v in cell.items():
    if "hazard_drift" not in v: continue
    hd=v["hazard_drift"][0]
    best_heur=min(v[h][0] for h in DEPLOYABLE_HEURS if h in v)
    if hd<best_heur-1e-9: win+=1
    elif hd>best_heur+1e-9: loss+=1
    else: tie+=1
print(f"hazard_drift beats the BEST-of-3 heuristic in {win}/{win+loss+tie} cells (loss={loss}, tie={tie})")

# hazard_drift vs first_answer specifically (first_answer is the strongest heuristic by mean gap)
print("\nhazard_drift vs first_answer detail by dataset (mean gap, mean false_late):")
import statistics as stx
for d in DATASETS:
    hd=[cell[(m,d)]["hazard_drift"] for m in MODELS if (m,d) in cell]
    fa=[cell[(m,d)]["first_answer"] for m in MODELS if (m,d) in cell]
    print(f"  {d:<6} hd_gap={stx.mean(x[0] for x in hd):.4f} fa_gap={stx.mean(x[0] for x in fa):.4f} | hd_flate={stx.mean(x[1] for x in hd):.4f} fa_flate={stx.mean(x[1] for x in fa):.4f} fa_fearly={stx.mean(x[2] for x in fa):.4f}")

# How often does empirical_bernstein essentially never stop? false_late near 1
print("\nempirical_bernstein & e_process 'never stops' check (false_late):")
for det in ["e_process","empirical_bernstein"]:
    fl=[v[det][1] for v in cell.values() if det in v]
    hi=sum(1 for x in fl if x>=0.9)
    print(f"  {det:<20} mean_false_late={sum(fl)/len(fl):.4f}  cells_with_false_late>=0.90: {hi}/{len(fl)}  min={min(fl):.3f} max={max(fl):.3f}")

# Gap reduction vs never_stop: how much of the never_stop->oracle gap does hazard_drift close?
print("\nFraction of never_stop oracle-gap closed by best deployable theory (hazard_drift), per dataset:")
import statistics as stx2
for d in DATASETS:
    hd=stx2.mean(cell[(m,d)]["hazard_drift"][0] for m in MODELS if (m,d) in cell)
    ns=stx2.mean(cell[(m,d)]["never_stop"][0] for m in MODELS if (m,d) in cell)
    print(f"  {d:<6} never_stop_gap={ns:.4f} hazard_drift_gap={hd:.4f} -> closed {100*(ns-hd)/ns:.1f}% of the do-nothing gap")
