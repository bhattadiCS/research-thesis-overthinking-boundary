#!/usr/bin/env bash
# Autonomous algorithm-v2 run: N1/N4 (CPU) alongside N5/P4b + N6/P8 (GPU).
# Pre-registrations: ThesisDocs/rigor_audit/07_algorithm_v2_protocols.md
#
#   bash tools/run_autonomous_v2.sh              # detaches; safe to close the terminal
#   bash tools/run_autonomous_v2.sh --foreground # stay attached (Ctrl-C stops it)
#
# Crash-safety / checkpointing:
#   * self-detaches (setsid/nohup + SIGHUP ignored) => survives disconnects
#   * all outputs under research/outputs/experiments_v2/
#   * background loop commits + pushes that dir every 20 min; each stage also
#     commits; git add/commit/push only READ the working tree, so checkpointing
#     during an active CSV append is safe (the script never checks out, stashes,
#     or rebases while data is being written)
#   * resumable: the collector reconciles and continues interrupted runs; the
#     N1/N4 harness caches per-cell JSONs. Re-running this script continues.
set -u

FOREGROUND=0
[ "${1:-}" = "--foreground" ] && FOREGROUND=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PYTHON:-python}"
V2="research/outputs/experiments_v2"
LOG="$V2/autonomous_run.log"
SUMMARY="$V2/success_checks.log"
PIDFILE="$V2/.run.pid"
GIT_LOCK="${TMPDIR:-/tmp}/v2_gitlock_$(id -u 2>/dev/null || echo 0)"
mkdir -p "$V2"

# ---- self-detach: one command, survives terminal close ----------------------
if [ "$FOREGROUND" -eq 0 ] && [ "${V2_DETACHED:-0}" != "1" ]; then
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
        echo "A run is already active (pid $(cat "$PIDFILE")). Follow it with:"
        echo "  tail -f $REPO/$LOG"
        exit 0
    fi
    if command -v setsid >/dev/null 2>&1; then
        V2_DETACHED=1 setsid nohup bash "$0" --foreground >> "$V2/nohup.out" 2>&1 < /dev/null &
    else
        V2_DETACHED=1 nohup bash "$0" --foreground >> "$V2/nohup.out" 2>&1 < /dev/null &
    fi
    echo "Autonomous v2 run detached (pid $!). Safe to close this terminal."
    echo "  progress : tail -f $REPO/$LOG"
    echo "  verdicts : cat  $REPO/$SUMMARY   (written as each stage finishes)"
    echo "  stop     : kill \$(cat $REPO/$PIDFILE)"
    exit 0
fi
trap '' HUP                      # inherited by children: terminal close cannot kill the run
echo $$ > "$PIDFILE"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---- git checkpointing (serialized; never mutates the working tree) ---------
git_lock() {                     # break a stale lock (>15 min) left by a crashed run
    if [ -d "$GIT_LOCK" ] && [ -z "$(find "$GIT_LOCK" -maxdepth 0 -mmin -15 2>/dev/null)" ]; then
        rmdir "$GIT_LOCK" 2>/dev/null
    fi
    local tries=0
    while ! mkdir "$GIT_LOCK" 2>/dev/null; do
        tries=$((tries + 1))
        [ "$tries" -gt 60 ] && { say "git lock busy 5min, skipping this checkpoint"; return 1; }
        sleep 5
    done
    return 0
}
git_unlock() { rmdir "$GIT_LOCK" 2>/dev/null; }

ckpt() {
    local msg="${1:-wip: autonomous v2 checkpoint}"
    git_lock || return 0
    git add "$V2" >/dev/null 2>&1
    if git diff --cached --quiet 2>/dev/null; then git_unlock; return 0; fi
    git commit -q -m "$msg ($(date '+%F %T'))" \
        -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" >>"$LOG" 2>&1 \
        || say "checkpoint commit failed"
    git push -q origin main >>"$LOG" 2>&1 \
        || say "push failed (commits are safe locally; retrying next cycle)"
    git_unlock
}

( while true; do sleep 1200; ckpt "wip: autonomous v2 checkpoint"; done ) &
CKPT_PID=$!
cleanup() { kill "$CKPT_PID" 2>/dev/null; ckpt "wip: autonomous v2 checkpoint (exit)"; rm -f "$PIDFILE"; }
trap cleanup EXIT INT TERM

say "=== autonomous v2 run starting on $(hostname) | HEAD=$(git rev-parse --short HEAD) | pid $$ ==="
"$PY" -c "import torch;print('torch',torch.__version__,'| cuda:',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" 2>&1 | tee -a "$LOG"

# ---- Stage 1: N1/N4 (CPU, background, niced so it never starves the GPU feed) ----
say "Stage 1: N1/N4 harness -> $V2/n1n4_results.log (cache: $V2/algov2_cache)"
nohup nice -n 10 "$PY" research/algorithm_v2_experiments.py \
    --matrix-root research/outputs/experiment_matrix \
    --cache-dir "$V2/algov2_cache" > "$V2/n1n4_results.log" 2>&1 &
N1_PID=$!

# ---- Stage 2: N5 (= P4b) token budget 256 -> 512 on the worst cell ----------
N5_DIR="$V2/p4b_mistral_small_24b_2409__gsm8k_tok512"
say "Stage 2: N5/P4b collect -> $N5_DIR"
"$PY" research/real_trace_experiments.py --model mistral_small_24b_2409 --device cuda \
    --quantization none --attn-implementation sdpa --task-source gsm8k --dataset-split train \
    --dataset-shuffle-seed 17 --max-steps 10 --max-new-tokens 512 --batch-size 16 \
    --prompt-mode minimal_json --system-prompt-mode default --temperatures 0.1 0.6 1.0 --seeds 7 \
    --max-tasks 500 --output-dir "$N5_DIR" >>"$LOG" 2>&1
say "N5 collect rc=$?"
ckpt "wip: N5 collection"
if [ -f "$N5_DIR/trace_steps.csv" ]; then
    "$PY" research/trace_analysis.py --input-dir "$N5_DIR" >>"$LOG" 2>&1 || say "N5 analyze FAILED"
    "$PY" - "$N5_DIR" <<'EOF' 2>&1 | tee -a "$SUMMARY" | tee -a "$LOG"
import sys, pandas as pd
d = pd.read_csv(sys.argv[1] + "/detector_comparison_by_run.csv")
p = d[d.detector.isin(["hazard_drift","never_stop"])].pivot(index="run_id", columns="detector", values="stop_utility")
loss = 100*(p.hazard_drift < p.never_stop).mean()
print(f"N5/P4b (max_new_tokens 256->512): loss {loss:.2f}% on {len(p)} runs | "
      f"baseline 30.27% | pre-registered PASS (<=25.3%): {loss <= 25.3}")
EOF
fi
ckpt "wip: N5 analyzed"

# ---- Stage 3: N6 (= P8) clean bf16 vs 4-bit pair ----------------------------
for Q in none 4bit; do
    ARM_DIR="$V2/p8_qwen7b_$Q"
    say "Stage 3: N6/P8 arm quantization=$Q -> $ARM_DIR"
    "$PY" research/real_trace_experiments.py --model qwen2p5_7b --device cuda \
        --quantization "$Q" --attn-implementation sdpa --task-source gsm8k --dataset-split train \
        --dataset-shuffle-seed 17 --max-steps 10 --max-new-tokens 256 --batch-size 32 \
        --prompt-mode minimal_json --system-prompt-mode default --temperatures 0.1 0.6 1.0 --seeds 7 \
        --max-tasks 500 --output-dir "$ARM_DIR" >>"$LOG" 2>&1
    say "N6 arm $Q collect rc=$?"
    [ -f "$ARM_DIR/trace_steps.csv" ] && { "$PY" research/trace_analysis.py --input-dir "$ARM_DIR" >>"$LOG" 2>&1 || say "N6 arm $Q analyze FAILED"; }
    ckpt "wip: N6 arm $Q"
done
if [ -f "$V2/p8_qwen7b_none/trace_steps.csv" ] && [ -f "$V2/p8_qwen7b_4bit/trace_steps.csv" ]; then
    "$PY" - "$V2" <<'EOF' 2>&1 | tee -a "$SUMMARY" | tee -a "$LOG"
import sys, math, pandas as pd
v2 = sys.argv[1]; q = {}
for arm in ("none", "4bit"):
    ts = pd.read_csv(f"{v2}/p8_qwen7b_{arm}/trace_steps.csv", usecols=["step","correct"])
    s2 = ts[ts.step == 2].correct.fillna(0)
    q[arm] = (float(s2.mean()), int(len(s2)))
gap = q["none"][0] - q["4bit"][0]
se = math.sqrt(sum(p*(1-p)/n for p, n in q.values())) or float("nan")
print(f"N6/P8 (bf16 vs 4-bit): q2 bf16={q['none'][0]:.4f} (n={q['none'][1]}) 4bit={q['4bit'][0]:.4f} (n={q['4bit'][1]}) "
      f"gap={gap:.4f} Z={gap/se:.2f} | pre-registered PASS (gap>=0.10 and Z>=1.96): {gap >= 0.10 and gap/se >= 1.96}")
EOF
fi
ckpt "wip: N6 analyzed"

# ---- Stage 4: N1/N4 verdicts ------------------------------------------------
say "Stage 4: waiting for N1/N4 (pid $N1_PID)"
wait "$N1_PID"
say "N1/N4 finished rc=$?"
{ echo "--- N1/N4 (pre-registered: N1 LOCO >= +927.5 ; N4 paired gain > +150) ---"
  tail -25 "$V2/n1n4_results.log"; } | tee -a "$SUMMARY" >/dev/null

kill "$CKPT_PID" 2>/dev/null
trap - EXIT INT TERM
git_lock
git add "$V2" >/dev/null 2>&1
git commit -q -m "results: autonomous algorithm-v2 run (N1/N4 + N5/P4b + N6/P8)" \
    -m "Pre-registered verdicts in research/outputs/experiments_v2/success_checks.log; protocols in ThesisDocs/rigor_audit/07_algorithm_v2_protocols.md." \
    -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" >>"$LOG" 2>&1 || say "final commit: nothing new"
git push origin main >>"$LOG" 2>&1 || say "FINAL PUSH FAILED — run: git push origin main"
git_unlock
rm -f "$PIDFILE"
say "=== run complete — verdicts: ==="
cat "$SUMMARY" | tee -a "$LOG"
