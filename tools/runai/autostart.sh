#!/usr/bin/env bash
# ============================================================================
# tools/runai/autostart.sh
#
# Auto-resume the experiment matrix after a pod (re)start. Designed to survive
# RunAI's recurring pod recycling: everything it needs is on the persistent
# volume, so on each boot it pulls the latest code and re-launches the matrix,
# which resumes from per-cell + within-cell checkpoints.
#
# IDEMPOTENT: if a run is already alive (pgrep), it does nothing -- safe to call
# repeatedly and safe if you also launch manually.
#
# ---- How to make it fire automatically on boot ----
# Best: set your RunAI workspace's *startup command* (Runtime settings ->
# Command, when creating/editing the workspace) to:
#       bash /workspace-persist/research-thesis-overthinking-boundary/tools/runai/autostart.sh
#   (or ask whoever manages the RunAI template to add it). It persists across
#   restarts because it lives on the persistent volume.
# Fallback (always works): after any restart just run that one command yourself
#   -- it pulls, installs deps, and resumes in one shot.
#
# ---- Config (optional) ----
#   DATASETS   which benchmarks to sweep   (default: "gsm8k math arc gpqa")
#   HF_TOKEN   for gated models/datasets   (env, or a .hf_token file in the repo
#              root -- which is gitignored; create with:
#                  echo 'hf_xxx' > <repo>/.hf_token )
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

LOG_DIR="${LOG_DIR:-$REPO_DIR/run_logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S 2>/dev/null || echo now)"

log() { printf '[autostart] %s\n' "$1"; }

# 1. Don't double-launch.
if pgrep -f run_experiment_matrix.py >/dev/null 2>&1; then
  log "a matrix run is already alive (pgrep hit); nothing to do."
  exit 0
fi

# 2. Config.
export DATASETS="${DATASETS:-gsm8k math arc gpqa}"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$REPO_DIR/.hf_token" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$REPO_DIR/.hf_token")"
  log "loaded HF_TOKEN from .hf_token"
fi

# 3. Sync latest code (best-effort; never block the run on a git hiccup). Stash
#    the constantly-rewritten manifest so the rebase applies; untracked partial
#    cells and finished cells are untouched.
git stash -q 2>/dev/null || true
if git pull --rebase >"$LOG_DIR/autostart_pull_${STAMP}.log" 2>&1; then
  log "git pull --rebase ok ($(git rev-parse --short HEAD))"
else
  git rebase --abort 2>/dev/null || true
  log "git pull failed; continuing on current local code (see autostart_pull_${STAMP}.log)"
fi
git stash drop -q 2>/dev/null || true

# 4. Launch detached. run_matrix.sh installs deps (fresh pod), runs the GPU
#    preflight + smoke gate, then the matrix -- which skips finished cells and
#    resumes partial ones.
RUN_LOG="$LOG_DIR/autostart_run_${STAMP}.log"
log "launching matrix for DATASETS='$DATASETS' -> $RUN_LOG"
nohup bash tools/runai/run_matrix.sh >"$RUN_LOG" 2>&1 &
log "launched (pid $!).  watch:  tail -f $RUN_LOG"
