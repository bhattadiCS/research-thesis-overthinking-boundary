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
#   STALE_MIN  minutes of no run output before a still-alive run is judged hung
#              and force-restarted (default: 30)
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

LOG_DIR="${LOG_DIR:-$REPO_DIR/run_logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S 2>/dev/null || echo now)"

log() { printf '[autostart] %s\n' "$1"; }

# 1. Don't double-launch -- BUT distinguish a healthy run from a hung one. A
#    matrix run that's alive AND still writing output is left alone; one that's
#    alive but silent for STALE_MIN minutes (GPU/CUDA stall, deadlock) is treated
#    as hung, killed, and relaunched. (pgrep alone can't tell the difference, so
#    a hang would otherwise block auto-resume forever -- which is exactly what
#    bit us on the 36h stall.)
STALE_MIN="${STALE_MIN:-30}"
running_pid="$(pgrep -f run_experiment_matrix.py | head -1)"
if [ -n "$running_pid" ]; then
  # Freshest progress signal: per-cell collect.log or per-run .npz both update
  # every few minutes during an active collect.
  newest_epoch="$(find "$REPO_DIR/research/outputs/experiment_matrix" -type f \
      \( -name 'collect.log' -o -name '*.npz' \) -printf '%T@\n' 2>/dev/null | sort -nr | head -1)"
  if [ -n "$newest_epoch" ]; then
    age_min=$(( ( $(date +%s) - ${newest_epoch%.*} ) / 60 ))
  else
    age_min=9999
  fi
  if [ "$age_min" -lt "$STALE_MIN" ]; then
    log "matrix run (pid $running_pid) alive and progressing (last output ${age_min}m ago); nothing to do."
    exit 0
  fi
  log "matrix run (pid $running_pid) alive but STALE (no output ${age_min}m >= ${STALE_MIN}m) -- treating as hung; killing."
  pkill -9 -f run_experiment_matrix.py 2>/dev/null || true
  pkill -9 -f real_trace_experiments.py 2>/dev/null || true
  sleep 5
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
