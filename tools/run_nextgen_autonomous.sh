#!/usr/bin/env bash
# Autonomous Next-Gen Probing Runner & Periodic Git Commit Sync
# File: tools/run_nextgen_autonomous.sh

set -e

echo "=========================================================="
echo "      AUTONOMOUS NEXT-GEN PROBING RUNNER & SYNC"
echo "=========================================================="

# Detect virtual environment python
if [ -f ".venv/bin/python" ]; then
    PY=".venv/bin/python"
    echo "[INFO] Detected local virtualenv python: $PY"
elif [ -f "../.venv/bin/python" ]; then
    PY="../.venv/bin/python"
    echo "[INFO] Detected parent virtualenv python: $PY"
else
    if [ -f ".venv/bin/activate" ]; then
        echo "[INFO] Activating local virtualenv..."
        source .venv/bin/activate
    fi
    PY="python"
fi

# 1. Sync repository before running
echo "[INFO] Pulling latest changes from remote main..."
git pull origin main

# Check CUDA environment
echo "[INFO] Checking CUDA environment..."
"$PY" -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# 2. Run Preflight Smoke Test
echo "[INFO] Running Preflight Smoke Test..."
"$PY" research/run_nextgen_experiments.py --smoke-test

echo "[INFO] Next-Gen Smoke Test passed successfully. Staging results..."
git add research/run_nextgen_experiments.py
git commit -m "results: VM Next-Gen smoke test completed successfully" || echo "[INFO] No changes to commit for smoke test."
git push origin main || echo "[INFO] Git push skipped or failed."

# 3. Start full Next-Gen Probing Tournament in background
echo "[INFO] Starting full Next-Gen Probing Tournament in background..."
"$PY" -u research/run_nextgen_experiments.py > run_nextgen_opt.log 2>&1 &
PID=$!

echo "[INFO] Background process started with PID $PID. Monitoring progress..."

while kill -0 $PID 2>/dev/null; do
    sleep 60
done

wait $PID
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Next-Gen probing script failed with exit code $EXIT_CODE!"
    echo "[ERROR] Check run_nextgen_opt.log for error traceback."
    exit $EXIT_CODE
fi

echo "[INFO] Next-Gen probing script completed successfully."

# 4. Commit and push final results
git add run_nextgen_opt.log
git commit -m "results: Next-Gen Probing Tournament completed" || echo "[INFO] No changes to commit."
git push origin main || echo "[WARNING] Final push failed."

echo "=========================================================="
echo "                NEXT-GEN RUN COMPLETED"
echo "=========================================================="
