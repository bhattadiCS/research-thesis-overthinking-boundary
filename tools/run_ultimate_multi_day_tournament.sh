#!/usr/bin/env bash
# Blackwell-safe, resumable launcher for research/run_ultimate_multi_day_tournament.py.
#
# Usage (after an intentional, user-controlled `git pull origin main`):
#   bash tools/run_ultimate_multi_day_tournament.sh
#
# This launcher deliberately never pulls while a tournament is active.  It only
# pushes immutable fold summaries / final artifacts every 15 minutes; it excludes
# mutable SQLite WAL files and .pth checkpoints so a remote sync cannot capture a
# half-written state.

set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_DIR="${INPUT_DIR:-research/outputs/experiments_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-research/outputs/experiments_v2}"
TRIALS_PER_FOLD="${TRIALS_PER_FOLD:-500}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8192}"
MAX_HOURS="${MAX_HOURS:-71.5}"
SYNC_SECONDS="${SYNC_SECONDS:-900}"
RUN_LOG="${RUN_LOG:-run_ultimate_multi_day_tournament.log}"
SMOKE_OUTPUT_DIR="${OUTPUT_DIR}/ultimate_smoke"
RUN_PID=""
SYNC_PID=""

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "[ERROR] Required file is missing: $1" >&2
        exit 2
    fi
}

cleanup() {
    if [[ -n "${SYNC_PID}" ]] && kill -0 "${SYNC_PID}" 2>/dev/null; then
        kill "${SYNC_PID}" 2>/dev/null || true
    fi
    if [[ -n "${RUN_PID}" ]] && kill -0 "${RUN_PID}" 2>/dev/null; then
        echo "[WARN] Interrupt received; tournament process remains checkpoint-safe but is being stopped." >&2
        kill "${RUN_PID}" 2>/dev/null || true
    fi
}
trap cleanup INT TERM

sync_git() {
    local phase="${1:-checkpoint}"
    local lock_dir="${OUTPUT_DIR}/.ultimate_git_sync.lock"
    [[ -d .git ]] || return 0
    mkdir -p "${OUTPUT_DIR}"
    if ! mkdir "${lock_dir}" 2>/dev/null; then
        return 0
    fi
    (
        trap 'rmdir "${lock_dir}" 2>/dev/null || true' EXIT
        local artifacts=()
        shopt -s nullglob
        artifacts+=("${OUTPUT_DIR}"/ultimate_tournament_manifest.json)
        artifacts+=("${OUTPUT_DIR}"/ultimate_tournament_status.json)
        artifacts+=("${OUTPUT_DIR}"/ultimate_fold_*_summary.json)
        if [[ "${phase}" == "final" ]]; then
            artifacts+=("${OUTPUT_DIR}"/ultimate_tournament_results.log)
            artifacts+=("${OUTPUT_DIR}"/ultimate_tournament_results.csv)
            artifacts+=("${OUTPUT_DIR}"/ultimate_oof_predictions.npz)
            artifacts+=("${OUTPUT_DIR}"/ultimate_research_graph.json)
        fi
        shopt -u nullglob
        if [[ "${#artifacts[@]}" -eq 0 ]]; then
            exit 0
        fi
        git add -- "${artifacts[@]}" || {
            echo "[WARN] Git staging skipped for ${phase} artifacts." >&2
            exit 0
        }
        if git diff --cached --quiet; then
            exit 0
        fi
        git commit -m "results: ultimate tournament ${phase} sync" || {
            echo "[WARN] Git commit failed; local artifacts are retained." >&2
            exit 0
        }
        git push origin main || echo "[WARN] Git push failed; the next sync will retry."
    )
}

monitor_sync() {
    while kill -0 "${RUN_PID}" 2>/dev/null; do
        sleep "${SYNC_SECONDS}"
        if kill -0 "${RUN_PID}" 2>/dev/null; then
            sync_git checkpoint
        fi
    done
}

require_file research/run_ultimate_multi_day_tournament.py

echo "==============================================================="
echo "  ULTIMATE CAUSAL STOPPING TOURNAMENT -- BLACKWELL 72H PROTOCOL"
echo "==============================================================="
echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Input: ${INPUT_DIR}"
echo "[INFO] Output: ${OUTPUT_DIR}"
echo "[INFO] No git pull is performed here; synchronize before launch."

"${PYTHON_BIN}" - <<'PY'
import importlib.util
import torch

print("Python torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA runtime:", torch.version.cuda)
print("Torch architectures:", ", ".join(torch.cuda.get_arch_list()) if torch.cuda.is_available() else "n/a")
print("Optuna installed:", importlib.util.find_spec("optuna") is not None)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the full Blackwell protocol")
props = torch.cuda.get_device_properties(0)
print("GPU:", props.name)
print("Compute capability:", f"sm_{props.major}{props.minor}")
print("Visible VRAM GiB:", round(props.total_memory / 2**30, 2))
if props.major < 12 or props.total_memory < 90 * 2**30:
    raise SystemExit("This launcher requires a Blackwell-class GPU with >=90 GiB visible VRAM")
if importlib.util.find_spec("optuna") is None:
    raise SystemExit("Install optuna>=4 before starting the full tournament")
PY

echo "[INFO] Running real-model VRAM / power / BF16 throughput audit..."
"${PYTHON_BIN}" research/run_ultimate_multi_day_tournament.py \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --vram-audit --audit-only \
    --batch-size "${BATCH_SIZE}" --max-batch-size "${MAX_BATCH_SIZE}" \
    --require-blackwell

echo "[INFO] Running a two-cell, one-epoch causal smoke test..."
"${PYTHON_BIN}" research/run_ultimate_multi_day_tournament.py \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${SMOKE_OUTPUT_DIR}" \
    --smoke-test --no-resume --no-compile --no-require-blackwell

echo "[INFO] Starting resumable 72-hour tournament; log: ${RUN_LOG}"
"${PYTHON_BIN}" -u research/run_ultimate_multi_day_tournament.py \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --trials-per-fold "${TRIALS_PER_FOLD}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" --max-batch-size "${MAX_BATCH_SIZE}" \
    --max-hours "${MAX_HOURS}" \
    --require-blackwell --resume \
    > "${RUN_LOG}" 2>&1 &
RUN_PID=$!
monitor_sync &
SYNC_PID=$!

if wait "${RUN_PID}"; then
    RUN_STATUS=0
else
    RUN_STATUS=$?
fi
RUN_PID=""

if kill -0 "${SYNC_PID}" 2>/dev/null; then
    kill "${SYNC_PID}" 2>/dev/null || true
fi
wait "${SYNC_PID}" 2>/dev/null || true
SYNC_PID=""

if [[ "${RUN_STATUS}" -ne 0 ]]; then
    echo "[ERROR] Tournament exited with status ${RUN_STATUS}. Check ${RUN_LOG}; checkpoints were retained." >&2
    tail -n 120 "${RUN_LOG}" || true
    exit "${RUN_STATUS}"
fi

if "${PYTHON_BIN}" -c "import json, sys; from pathlib import Path; status=json.loads((Path(sys.argv[1]) / 'ultimate_tournament_status.json').read_text()); raise SystemExit(0 if status.get('complete') else 1)" "${OUTPUT_DIR}"; then
    sync_git final
    echo "[INFO] Tournament completed successfully."
    echo "[INFO] Results: ${OUTPUT_DIR}/ultimate_tournament_results.log"
else
    sync_git checkpoint
    echo "[INFO] Time budget reached before all folds completed. Re-run this launcher to resume from checkpoints."
fi
