#!/usr/bin/env bash
# Unified Overnight Deep Research Orchestrator.
# Runs:
# 1. Enriched Telemetry Sweep (N8a, N8b, N7 k=2 SC) on 4 models x 2 tasks.
# 2. Sequential GRU/LSTM training and evaluation (N8b sequence upgrade).
# 3. Hysteresis Gated Self-Consistency simulation.
# 4. Automatic Git push of data and summary tournament log.

set -eu

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PYTHON:-python}"
V2="research/outputs/experiments_v2"
LOG="$V2/overnight_run.log"
TOURNAMENT_LOG="$V2/overnight_tournament_results.log"

mkdir -p "$V2"
echo "=== Overnight Deep Research starting on $(date '+%F %T') ===" | tee -a "$LOG"

# 1. Pull latest code updates
git pull | tee -a "$LOG"

# 2. Run Enriched Telemetry Sweep (4 Models x 2 Datasets x 200 Tasks)
MODELS=("qwen2p5_7b" "qwen2p5_14b" "mistral_small_24b_2409" "deepseek_r1_distill_1p5b")
DATASETS=("gsm8k" "math")

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "=== Running ${model} on ${dataset} ===" | tee -a "$LOG"
        
        SPLIT="train"
        if [ "$dataset" = "math" ]; then
            SPLIT="test"
        fi
        
        # Run generation
        "$PY" research/real_trace_experiments.py \
            --model "$model" \
            --task-source "$dataset" \
            --dataset-split "$SPLIT" \
            --max-tasks 200 \
            --temperatures 0.6 \
            --seeds 7 \
            --enable-k2-agreement \
            --enable-extended-observables \
            --attn-implementation sdpa \
            --output-dir "$V2/sweep_${model}_${dataset}" 2>&1 | tee -a "$LOG"
    done
done

echo "=== Telemetry Sweep Complete. Starting Advanced Tournament... ===" | tee -a "$LOG"

# 3. Run final sequence model and gated SC tournament
"$PY" research/run_final_tournament.py --dir "$V2" 2>&1 | tee "$TOURNAMENT_LOG" | tee -a "$LOG"

echo "=== Tournament Complete. Committing and pushing all results to GitHub... ===" | tee -a "$LOG"

# 4. Git stage and push
git add "$V2"
git commit -m "results: overnight deep research tournament results collected" | tee -a "$LOG"
git push origin main 2>&1 | tee -a "$LOG"

echo "=== overnight run complete ===" | tee -a "$LOG"
