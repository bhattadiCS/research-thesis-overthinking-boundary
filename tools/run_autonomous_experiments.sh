#!/usr/bin/env bash
# Autonomous Hyper-Optimization Runner & Periodic Git Commit Sync
# File: tools/run_autonomous_experiments.sh

# Exit immediately if any command fails (except monitoring loop checks)
set -e

echo "=========================================================="
echo "      AUTONOMOUS HYPER-OPTIMIZATION RUNNER & SYNC"
echo "=========================================================="

# Detect virtual environment python
if [ -f ".venv/bin/python" ]; then
    PY=".venv/bin/python"
    echo "[INFO] Detected local virtualenv python: $PY"
elif [ -f "../.venv/bin/python" ]; then
    PY="../.venv/bin/python"
    echo "[INFO] Detected parent virtualenv python: $PY"
else
    # Try activating virtual env if present
    if [ -f ".venv/bin/activate" ]; then
        echo "[INFO] Activating local virtualenv..."
        source .venv/bin/activate
    fi
    PY="python"
fi

# 1. Sync repository before running
echo "[INFO] Pulling latest changes from remote main..."
git pull origin main

# Check if CUDA is available via python
echo "[INFO] Checking CUDA environment..."
"$PY" -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# 2. Run Smoke Test first to verify compiling and logic
echo "[INFO] Running Preflight Smoke Test..."
"$PY" research/run_advanced_hyper_optimization.py --smoke-test

echo "[INFO] Smoke Test passed successfully. Staging results..."
git add research/outputs/experiments_v2/advanced_tournament_results.log
git commit -m "results: VM smoke test completed successfully" || echo "[INFO] No changes to commit for smoke test."
git push origin main || echo "[INFO] Git push skipped or failed."

# 3. Start the main full hyper-optimization tournament in the background
echo "[INFO] Starting full hyper-optimization tournament in background..."
"$PY" -u research/run_advanced_hyper_optimization.py --deep-search > run_advanced_hyper_opt.log 2>&1 &
PID=$!

echo "[INFO] Background process started with PID $PID. Monitoring and syncing folds periodically..."

CHECKPOINT_FILE="research/outputs/experiments_v2/advanced_tournament_checkpoint.pth"
RESULTS_FILE="research/outputs/experiments_v2/advanced_tournament_results.log"

# Keep track of last seen fold in checkpoint to avoid redundant commits
LAST_FOLD=-1

while kill -0 $PID 2>/dev/null; do
    sleep 30
    
    # Check if checkpoint exists
    if [ -f "$CHECKPOINT_FILE" ]; then
        # Check if we can read the completed fold using a quick python print
        CURRENT_FOLD=$("$PY" -c "import torch; cp=torch.load('$CHECKPOINT_FILE', map_location='cpu', weights_only=False); print(cp.get('fold', -1))" 2>/dev/null || echo "-1")
        
        # If the fold has progressed, commit and push the checkpoint
        if [ "$CURRENT_FOLD" -ne "-1" ] && [ "$CURRENT_FOLD" -ne "$LAST_FOLD" ]; then
            echo "[SYNC] Fold $((CURRENT_FOLD + 1)) completed. Committing progress..."
            git add "$CHECKPOINT_FILE"
            git commit -m "results: VM completed Fold $((CURRENT_FOLD + 1))/5 checkpoint" || echo "[INFO] Nothing to commit."
            git push origin main || echo "[WARNING] Push failed, will retry on next fold."
            LAST_FOLD=$CURRENT_FOLD
        fi
    fi
done

# Wait for background process to finalize and get exit code
wait $PID
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Advanced tournament script failed with exit code $EXIT_CODE!"
    echo "[ERROR] Check run_advanced_hyper_opt.log for error traceback."
    exit $EXIT_CODE
fi

echo "[INFO] Advanced tournament script completed successfully."

# 4. Commit and push the final tournament verdict
if [ -f "$RESULTS_FILE" ]; then
    echo "[INFO] Staging final verdict summary..."
    git add "$RESULTS_FILE"
    # Stage any other log files created
    if [ -f "run_advanced_hyper_opt.log" ]; then
        git add run_advanced_hyper_opt.log
    fi
    git commit -m "results: VM advanced hyper-optimization tournament completed" || echo "[INFO] No changes to commit."
    git push origin main || echo "[WARNING] Final push failed."
    echo "[SUCCESS] Final tournament results synced to GitHub!"
else
    echo "[WARNING] Final results file $RESULTS_FILE not found."
fi

echo "=========================================================="
echo "                    RUN COMPLETED"
echo "=========================================================="
