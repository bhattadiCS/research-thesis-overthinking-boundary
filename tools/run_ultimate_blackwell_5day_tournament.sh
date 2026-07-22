#!/usr/bin/env bash
# ==============================================================================
# ULTIMATE BLACKWELL 5-DAY CAUSAL STOPPING TOURNAMENT LAUNCHER
# Optimized for NVIDIA RTX PRO 6000 Blackwell Server Edition (98GB VRAM)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  elif command -v python &> /dev/null; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi

# Auto-install lightgbm if missing in the environment
"${PYTHON_BIN}" -c "import lightgbm" 2>/dev/null || "${PYTHON_BIN}" -m pip install --quiet lightgbm scikit-learn scipy pandas

INPUT_DIR="${INPUT_DIR:-research/outputs/experiments_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-research/outputs/experiments_v2/blackwell_5day_tournament_v1}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-512}"
BOOTSTRAP_DRAWS="${BOOTSTRAP_DRAWS:-10000}"
JOBS="${JOBS:-16}"

echo "==============================================================="
echo " ULTIMATE BLACKWELL 5-DAY TOURNAMENT LAUNCHER"
echo "==============================================================="
echo "[INFO] Python Binary: ${PYTHON_BIN}"
echo "[INFO] Input Directory: ${INPUT_DIR}"
echo "[INFO] Output Directory: ${OUTPUT_DIR}"
echo "[INFO] Epochs: ${EPOCHS} | Batch Size: ${BATCH_SIZE} | Jobs: ${JOBS}"
echo "[INFO] Bootstrap Draws: ${BOOTSTRAP_DRAWS}"
echo "==============================================================="

"${PYTHON_BIN}" -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('Compute Capability:', torch.cuda.get_device_capability(0))
    print('VRAM (GiB):', torch.cuda.get_device_properties(0).total_memory / (1024**3))
"

echo "[INFO] Running Self-Test..."
"${PYTHON_BIN}" research/run_ultimate_blackwell_5day_tournament.py --self-test

echo "[INFO] Launching Blackwell 5-Day Tournament..."
"${PYTHON_BIN}" research/run_ultimate_blackwell_5day_tournament.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --bootstrap-draws "${BOOTSTRAP_DRAWS}" \
  --jobs "${JOBS}"

echo "==============================================================="
echo "[SUCCESS] Tournament Complete! Results saved to ${OUTPUT_DIR}"
echo "==============================================================="
