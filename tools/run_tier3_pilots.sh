#!/usr/bin/env bash
# Run Tier-3 N7/N8 pilots on GSM8K and MATH sequentially.
set -eu

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PYTHON:-python}"
V2="research/outputs/experiments_v2"
LOG="$V2/tier3_pilot_run.log"

mkdir -p "$V2"
echo "=== Tier-3 Pilot run starting on $(date '+%F %T') ===" | tee -a "$LOG"

# 1. Pull the new code changes
git pull | tee -a "$LOG"

# 2. Run the GSM8K Pilot
echo "=== Running GSM8K Pilot ===" | tee -a "$LOG"
"$PY" research/real_trace_experiments.py \
    --model qwen2p5_7b \
    --task-source gsm8k \
    --max-tasks 500 \
    --temperatures 0.6 \
    --seeds 7 \
    --enable-k2-agreement \
    --enable-extended-observables \
    --attn-implementation sdpa \
    --output-dir "$V2/tier3_pilot_gsm8k" 2>&1 | tee -a "$LOG"

# 3. Run the MATH Pilot
echo "=== Running MATH Pilot ===" | tee -a "$LOG"
"$PY" research/real_trace_experiments.py \
    --model qwen2p5_7b \
    --task-source math \
    --max-tasks 500 \
    --temperatures 0.6 \
    --seeds 7 \
    --enable-k2-agreement \
    --enable-extended-observables \
    --attn-implementation sdpa \
    --output-dir "$V2/tier3_pilot_math" 2>&1 | tee -a "$LOG"

echo "=== Pilots complete. Committing and pushing results... ===" | tee -a "$LOG"

# 4. Stage and push results
git add "$V2/tier3_pilot_gsm8k/trace_steps.csv" \
        "$V2/tier3_pilot_gsm8k/trace_runs.csv" \
        "$V2/tier3_pilot_gsm8k/trace_batch_metrics.csv" \
        "$V2/tier3_pilot_gsm8k/runtime_metadata.json" \
        "$V2/tier3_pilot_math/trace_steps.csv" \
        "$V2/tier3_pilot_math/trace_runs.csv" \
        "$V2/tier3_pilot_math/trace_batch_metrics.csv" \
        "$V2/tier3_pilot_math/runtime_metadata.json"

git commit -m "results: Tier-3 Qwen-7B GSM8K/MATH pilot telemetry collected" | tee -a "$LOG"
git push origin main 2>&1 | tee -a "$LOG"

echo "=== run complete ===" | tee -a "$LOG"
