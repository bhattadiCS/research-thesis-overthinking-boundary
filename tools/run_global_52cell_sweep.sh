#!/usr/bin/env bash
# Global 52-Cell Telemetry Sweep Orchestrator.
# Generates enriched trace telemetry across 13 models and 4 reasoning benchmarks.
# Uses optimal batch size parallelization for the NVIDIA RTX Pro 6000 Blackwell GPU.

set -eu

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Automatically load HF_TOKEN from .hf_token if it exists on disk
if [ -z "${HF_TOKEN:-}" ] && [ -f "$REPO/.hf_token" ]; then
    export HF_TOKEN="$(tr -d '[:space:]' < "$REPO/.hf_token")"
    echo "Loaded HF_TOKEN from .hf_token file"
fi

PY="${PYTHON:-python}"
V2="research/outputs/experiments_v2"
LOG="$V2/global_sweep.log"

mkdir -p "$V2"
echo "=== Global 52-Cell Telemetry Sweep starting on $(date '+%F %T') ===" | tee -a "$LOG"

# 13 Models in the census
MODELS=(
    "qwen2p5_0p5b"
    "qwen2p5_3b"
    "qwen2p5_7b"
    "qwen2p5_14b"
    "qwen2p5_32b"
    "deepseek_r1_distill_1p5b"
    "deepseek_r1_distill_7b"
    "llama_3p1_8b_instruct"
    "mistral_7b_instruct_v0p3"
    "mistral_small_24b_2409"
    "phi_4_mini_instruct"
    "yi_1p5_9b_chat"
    "qwen_3p5_9b"
)


# 4 Reasoning Datasets
DATASETS=(
    "gsm8k"
    "math"
    "arc"
    "gpqa"
)

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        CELL_DIR="$V2/global_${model}_${dataset}"
        
        # Checkpoint: Skip if cell has already completed successfully
        if [ -f "$CELL_DIR/trace_steps.csv" ] && [ -f "$CELL_DIR/batch_metrics.csv" ]; then
            echo "Checkpoint found: skipping completed cell global_${model}_${dataset}" | tee -a "$LOG"
            continue
        fi
        
        echo "=== Launching Cell: ${model} on ${dataset} ===" | tee -a "$LOG"
        
        # Determine split (train for gsm8k, test for others matching census)
        SPLIT="train"
        if [ "$dataset" != "gsm8k" ]; then
            SPLIT="test"
        fi
        
        # Blackwell GPU Batch Size tuning: Use safer batch sizes to prevent container cgroup freezes
        BATCH_SIZE=16
        if [[ "$model" == *"0p5b"* ]]; then
            BATCH_SIZE=128
        elif [[ "$model" == *"3b"* || "$model" == *"1p5b"* ]]; then
            BATCH_SIZE=64
        elif [[ "$model" == *"7b"* || "$model" == *"8b"* || "$model" == *"9b"* || "$model" == *"14b"* ]]; then
            BATCH_SIZE=32
        fi
        
        # Run generation
        # Note: --attn-implementation sdpa is faster and avoids flash_attn install errors
        "$PY" research/real_trace_experiments.py \
            --model "$model" \
            --task-source "$dataset" \
            --dataset-split "$SPLIT" \
            --max-tasks 500 \
            --temperatures 0.6 \
            --seeds 7 \
            --batch-size "$BATCH_SIZE" \
            --enable-k2-agreement \
            --enable-extended-observables \
            --attn-implementation sdpa \
            --output-dir "$CELL_DIR" 2>&1 | tee -a "$LOG"
            
        echo "Cell complete: global_${model}_${dataset}" | tee -a "$LOG"
        
        # Commit results cell-by-cell to prevent git conflicts or data loss
        git add "$CELL_DIR"
        git commit -m "results: auto-commit cell global_${model}_${dataset}" || true
        git push origin main || true
    done
done

echo "=== Global Sweep Complete on $(date '+%F %T') ===" | tee -a "$LOG"
# Run final tournament solver on the full database
"$PY" research/run_final_tournament.py --dir "$V2" 2>&1 | tee "$V2/global_tournament_results.log" | tee -a "$LOG"
git add "$V2"
git commit -m "results: global 52-cell sweep final tournament completed" || true
git push origin main || true
