#!/usr/bin/env bash
# Run Tier-1 N2/N3 experiments on the 52-cell matrix data.
set -eu

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PYTHON:-python}"
V2="research/outputs/experiments_v2"
LOG="$V2/n2_n3_run.log"
SUMMARY="$V2/n2_n3_summary.log"

mkdir -p "$V2"
echo "=== N2/N3 experiment run starting on $(date '+%F %T') ===" | tee -a "$LOG"

"$PY" research/algorithm_v2_experiments_n2_n3.py \
    --matrix-root research/outputs/experiment_matrix \
    --cache-dir "$V2/algov2_cache_n2_n3" 2>&1 | tee -a "$LOG" | tee "$SUMMARY"

echo "=== run complete ===" | tee -a "$LOG"
