#!/usr/bin/env bash
# Autonomous algorithm-v2 run: N1/N4 (CPU) alongside N5/P4b + N6/P8 (GPU).
# Pre-registrations: ThesisDocs/rigor_audit/07_algorithm_v2_protocols.md.
#
# Checkpointing / crash-safety:
#   * every stage's outputs live under research/outputs/experiments_v2/
#   * a background loop commits+pushes that directory every 20 minutes
#   * the GPU collector resumes interrupted runs by default (append-only CSVs
#     + reconcile pass); the N1/N4 harness caches per-cell JSONs; re-running
#     this script simply continues where the last run stopped.
#
# Usage (from the repo root on the box, python env active):
#   bash tools/run_autonomous_v2.sh
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PYTHON:-python}"
V2="research/outputs/experiments_v2"
mkdir -p "$V2"
LOG="$V2/autonomous_run.log"
SUMMARY="$V2/success_checks.log"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

ckpt() {
    git add "$V2" >/dev/null 2>&1
    if ! git diff --cached --quiet 2>/dev/null; then
        git commit -q -m "wip: autonomous v2 checkpoint ($(date '+%F %T')) [skip ci]" \
            -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" || say "ckpt commit failed"
        git push -q origin main 2>>"$LOG" || say "ckpt push failed (continuing; will retry next cycle)"
    fi
}

# ---- background checkpointer (every 20 min) --------------------------------
( while true; do sleep 1200; ckpt; done ) &
CKPT_PID=$!
trap 'kill "$CKPT_PID" 2>/dev/null; ckpt' EXIT

say "=== autonomous v2 run starting on $(hostname); HEAD=$(git rev-parse --short HEAD) ==="
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" 2>&1 | tee -a "$LOG"

# ---- Stage 1: N1/N4 (CPU) in the background --------------------------------
say "Stage 1: launching N1/N4 harness (CPU, background; cache under $V2/algov2_cache)"
nohup "$PY" research/algorithm_v2_experiments.py \
    --matrix-root research/outputs/experiment_matrix \
    --cache-dir "$V2/algov2_cache" > "$V2/n1n4_results.log" 2>&1 &
N1_PID=$!

# ---- Stage 2: N5 (= P4b) token budget 512, worst cell ----------------------
N5_DIR="$V2/p4b_mistral_small_24b_2409__gsm8k_tok512"
say "Stage 2: N5/P4b collection -> $N5_DIR (resumes automatically if interrupted)"
"$PY" research/real_trace_experiments.py --model mistral_small_24b_2409 --device cuda \
    --quantization none --attn-implementation sdpa --task-source gsm8k --dataset-split train \
    --dataset-shuffle-seed 17 --max-steps 10 --max-new-tokens 512 --batch-size 16 \
    --prompt-mode minimal_json --system-prompt-mode default --temperatures 0.1 0.6 1.0 --seeds 7 \
    --max-tasks 500 --output-dir "$N5_DIR" >> "$LOG" 2>&1
N5_RC=$?
say "N5 collect rc=$N5_RC"
if [ "$N5_RC" -eq 0 ]; then
    "$PY" research/trace_analysis.py --input-dir "$N5_DIR" >> "$LOG" 2>&1 || say "N5 analyze FAILED"
    "$PY" - "$N5_DIR" <<'EOF' | tee -a "$SUMMARY"
import sys, pandas as pd
d = pd.read_csv(sys.argv[1] + "/detector_comparison_by_run.csv")
p = d[d.detector.isin(["hazard_drift","never_stop"])].pivot(index="run_id", columns="detector", values="stop_utility")
loss = 100*(p.hazard_drift < p.never_stop).mean()
print(f"N5/P4b: loss {loss:.2f}% on {len(p)} runs | pre-registered PASS(<=25.3%): {loss <= 25.3}")
EOF
fi
ckpt

# ---- Stage 3: N6 (= P8) clean bf16 vs 4-bit pair ---------------------------
for Q in none 4bit; do
    ARM_DIR="$V2/p8_qwen7b_$Q"
    say "Stage 3: N6/P8 arm quantization=$Q -> $ARM_DIR"
    "$PY" research/real_trace_experiments.py --model qwen2p5_7b --device cuda \
        --quantization "$Q" --attn-implementation sdpa --task-source gsm8k --dataset-split train \
        --dataset-shuffle-seed 17 --max-steps 10 --max-new-tokens 256 --batch-size 32 \
        --prompt-mode minimal_json --system-prompt-mode default --temperatures 0.1 0.6 1.0 --seeds 7 \
        --max-tasks 500 --output-dir "$ARM_DIR" >> "$LOG" 2>&1
    say "N6 arm $Q collect rc=$?"
    "$PY" research/trace_analysis.py --input-dir "$ARM_DIR" >> "$LOG" 2>&1 || say "N6 arm $Q analyze FAILED"
    ckpt
done
if [ -f "$V2/p8_qwen7b_none/trace_steps.csv" ] && [ -f "$V2/p8_qwen7b_4bit/trace_steps.csv" ]; then
    "$PY" - "$V2" <<'EOF' | tee -a "$SUMMARY"
import sys, math, pandas as pd
v2 = sys.argv[1]; q = {}
for arm in ("none", "4bit"):
    ts = pd.read_csv(f"{v2}/p8_qwen7b_{arm}/trace_steps.csv", usecols=["step","correct"])
    s2 = ts[ts.step==2].correct.fillna(0); q[arm] = (float(s2.mean()), len(s2))
gap = q["none"][0]-q["4bit"][0]
se = math.sqrt(sum(p*(1-p)/n for p,n in q.values()))
print(f"N6/P8: q2 bf16={q['none'][0]:.4f} (n={q['none'][1]}) 4bit={q['4bit'][0]:.4f} (n={q['4bit'][1]}) "
      f"gap={gap:.4f} Z={gap/se:.2f} | pre-registered PASS(gap>=0.10, Z>=1.96): {gap>=0.10 and gap/se>=1.96}")
EOF
fi

# ---- Stage 4: wait for N1/N4, record results -------------------------------
say "Stage 4: waiting for N1/N4 harness (pid $N1_PID)"
wait "$N1_PID"
say "N1/N4 finished rc=$? — verdicts:"
tail -25 "$V2/n1n4_results.log" | tee -a "$SUMMARY"

# ---- final commit ------------------------------------------------------------
kill "$CKPT_PID" 2>/dev/null
git add "$V2"
git commit -q -m "results: autonomous algorithm-v2 run (N1/N4 + N5/P4b + N6/P8)" \
    -m "Success-check verdicts in research/outputs/experiments_v2/success_checks.log; full log in autonomous_run.log. Pre-registrations: ThesisDocs/rigor_audit/07_algorithm_v2_protocols.md." \
    -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" || say "final commit: nothing new"
git push origin main 2>>"$LOG" || say "FINAL PUSH FAILED — push manually: git push origin main"
say "=== run complete. Read $SUMMARY for the pre-registered verdicts. ==="
