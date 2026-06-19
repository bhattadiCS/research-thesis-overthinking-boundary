#!/usr/bin/env bash
# ============================================================================
# tools/runai/run_matrix.sh
#
# Single RUN:AI entrypoint that runs the FULL experiment matrix on one GPU:
# every model x every dataset, each through collect -> analyze -> report,
# then a cross-family aggregate per dataset. Wraps research/run_experiment_matrix.py
# with environment preflight, efficient single-GPU settings, a fast smoke gate,
# and a final pass/fail summary that names exactly which steps failed.
#
# Run it from anywhere inside the workspace (it locates the repo from its own
# path):
#     bash tools/runai/run_matrix.sh                 # full run: all models x gsm8k
#     MODE=smoke bash tools/runai/run_matrix.sh      # ~2 min end-to-end sanity run
#     MODELS="qwen2p5_7b qwen2p5_14b qwen2p5_32b" bash tools/runai/run_matrix.sh
#     PHASES="collect" bash tools/runai/run_matrix.sh
#     HF_TOKEN=hf_xxx bash tools/runai/run_matrix.sh # required for gated Llama
#
# Optional env knobs (all have sane defaults):
#     MODE=full|smoke            MODELS="a b c"          DATASETS="gsm8k"
#     PHASES="collect analyze report aggregate"          MAX_TASKS=<n>
#     BATCH_SIZE=<n>             QUANTIZATION=none|4bit   MATRIX_ARGS="--force"
#     PREFLIGHT_SMOKE=1          FORCE_INSTALL=0          PULL_LATEST=0
#     HF_HOME=<dir>              OUTPUT_ROOT=<dir>        LOG_DIR=<dir>
#     OMP_NUM_THREADS=<n>        AUTO_INSTALL_TORCH=0
# ============================================================================
set -euo pipefail

# --- locate repo from this script's own path (tools/runai/run_matrix.sh) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# --- knobs -----------------------------------------------------------------
MODE="${MODE:-full}"
PULL_LATEST="${PULL_LATEST:-0}"
FORCE_INSTALL="${FORCE_INSTALL:-0}"
PREFLIGHT_SMOKE="${PREFLIGHT_SMOKE:-1}"
AUTO_INSTALL_TORCH="${AUTO_INSTALL_TORCH:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-auto}"        # concurrent collect jobs (VRAM-bounded); int or 'auto'
VRAM_BUDGET_FRAC="${VRAM_BUDGET_FRAC:-0.80}"
GIT_CHECKPOINT="${GIT_CHECKPOINT:-1}"       # commit light artifacts after each cell
GIT_PUSH="${GIT_PUSH:-1}"                   # also push each checkpoint
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/research/outputs/experiment_matrix}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/run_logs}"
HF_HOME="${HF_HOME:-$REPO_DIR/.hf_cache}"
SMOKE_MODEL="${SMOKE_MODEL:-qwen2p5_0p5b}"

log()  { printf '[runai-matrix] %s\n' "$1"; }
fail() { printf '[runai-matrix] ERROR: %s\n' "$1" >&2; exit 1; }
trap 'rc=$?; [[ $rc -ne 0 ]] && printf "\n[runai-matrix] FAILED at line %s (exit %s). Logs: %s\n" "$LINENO" "$rc" "$LOG_DIR" >&2' ERR

mkdir -p "$LOG_DIR" "$HF_HOME"
cd "$REPO_DIR"
[[ -f "research/run_experiment_matrix.py" ]] || fail "research/run_experiment_matrix.py not found under $REPO_DIR"

# --- repo update (off by default; never clobbers local changes) -------------
if [[ "$PULL_LATEST" == "1" ]]; then
  if [[ -n "$(git -C "$REPO_DIR" status --porcelain 2>/dev/null)" ]]; then
    log "Skipping git pull: worktree has local changes."
  else
    log "Pulling latest (--ff-only)"; git -C "$REPO_DIR" pull --ff-only || log "git pull failed; continuing with current tree."
  fi
fi

# --- interpreter -----------------------------------------------------------
# Default: use the CURRENT interpreter (e.g. conda base), where the Blackwell-
# ready torch already lives. requirements-colab.txt has no torch line on
# purpose, so a fresh venv would have no CUDA torch. Opt into an isolated venv
# with USE_VENV=1 (it inherits site-packages so it still sees that torch).
if [[ "${USE_VENV:-0}" == "1" ]]; then
  if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating virtualenv at $VENV_DIR (--system-site-packages, to inherit torch)"
    "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
  fi
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  log "Using venv interpreter: $(command -v python)"
else
  log "Using current interpreter: $(command -v python) (set USE_VENV=1 to isolate)"
fi

# --- dependencies (install only when missing, unless FORCE_INSTALL=1) -------
imports_ok() {
  python - <<'PY' >/dev/null 2>&1
import importlib
for m in ("torch","transformers","datasets","pandas","sklearn","matplotlib","numpy"):
    importlib.import_module(m)
PY
}
if [[ "$FORCE_INSTALL" == "1" ]] || ! imports_ok; then
  log "Installing dependencies from requirements-colab.txt"
  python -m pip install --upgrade pip
  python -m pip install -r "$REPO_DIR/requirements-colab.txt"
else
  log "Core dependencies present; skipping install (set FORCE_INSTALL=1 to force)."
fi

# --- efficiency env (must be set BEFORE any CUDA init in child processes) ---
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # less fragmentation -> bigger batches
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"                        # avoid fork-warning stalls
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"                                          # cap CPU oversubscription
export PYTHONUNBUFFERED=1
export HF_HOME                                                                          # persist model cache across pods
# Fast parallel downloads, but only if the helper package is importable.
python -m pip install -q hf_transfer >/dev/null 2>&1 || true
if python -c "import hf_transfer" >/dev/null 2>&1; then export HF_HUB_ENABLE_HF_TRANSFER=1; fi

# --- Hugging Face auth (needed for gated models e.g. Llama-3.1) ------------
if [[ -n "${HF_TOKEN:-}" ]]; then
  log "Logging into Hugging Face via HF_TOKEN"
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential >/dev/null 2>&1 || log "HF login failed; gated downloads may fail."
else
  log "HF_TOKEN not set. Gated models (e.g. llama_3p1_8b_instruct) will fail to download until you set it."
fi

# --- git readiness for periodic checkpoints --------------------------------
if [[ "$GIT_CHECKPOINT" == "1" ]]; then
  if ! git -C "$REPO_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    log "Not a git repo; disabling git checkpoints."; GIT_CHECKPOINT=0; GIT_PUSH=0
  else
    git -C "$REPO_DIR" config user.email >/dev/null 2>&1 || git -C "$REPO_DIR" config user.email "${GIT_EMAIL:-runai@overthinking.local}"
    git -C "$REPO_DIR" config user.name  >/dev/null 2>&1 || git -C "$REPO_DIR" config user.name  "${GIT_NAME:-RunAI Matrix}"
    if [[ "$GIT_PUSH" == "1" ]] && ! git -C "$REPO_DIR" push --dry-run >/dev/null 2>&1; then
      log "git push dry-run failed (no remote write access?). Will commit locally but NOT push."
      GIT_PUSH=0
    fi
    log "Git checkpoints ON (push=$GIT_PUSH). Large .npz excluded via .gitignore."
  fi
fi

# --- GPU preflight: fail fast with remediation if torch can't see CUDA -----
log "Environment check:"
python tools/runai/check_env.py || log "check_env.py reported issues (continuing to hard CUDA assertion)."
if ! python - <<'PY'
import sys
try:
    import torch
except Exception as e:  # noqa: BLE001
    print(f"[preflight] FATAL: torch not importable: {e}"); sys.exit(3)
print(f"[preflight] torch {torch.__version__} | CUDA build {torch.version.cuda}")
if not torch.cuda.is_available():
    print("[preflight] FATAL: torch cannot see CUDA (CPU-only torch in this env).")
    print("  Remediation: python tools/runai/jupyter_repo_gpu_sanity.py  (or re-run with AUTO_INSTALL_TORCH=1)")
    sys.exit(3)
i = torch.cuda.current_device()
p = torch.cuda.get_device_properties(i)
cap = torch.cuda.get_device_capability(i)
sm = f"sm_{cap[0]}{cap[1]}"
archs = torch.cuda.get_arch_list()
print(f"[preflight] GPU: {p.name} | {p.total_memory/1024**3:.0f} GiB | {sm} | "
      f"bf16={torch.cuda.is_bf16_supported()} | built for {archs}")
if archs and sm not in archs:
    print(f"[preflight] WARNING: this torch was NOT built for {sm}. "
          f"Blackwell (sm_120) needs a cu128+/sm_120 wheel.")
# Real compute test: catches the 'no kernel image is available' failure on
# Blackwell that torch.cuda.is_available() does not.
try:
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    _ = (x @ x).float().sum().item()
    torch.cuda.synchronize()
except Exception as e:  # noqa: BLE001
    print(f"[preflight] FATAL: CUDA compute test failed on {sm}: {e}")
    print("  Remediation: python tools/runai/jupyter_repo_gpu_sanity.py  (or re-run with AUTO_INSTALL_TORCH=1)")
    sys.exit(3)
print("[preflight] CUDA compute test passed.")
PY
then
  if [[ "$AUTO_INSTALL_TORCH" == "1" ]]; then
    log "AUTO_INSTALL_TORCH=1 -> attempting the repo torch/GPU repair path"
    python tools/runai/jupyter_repo_gpu_sanity.py || true
    python -c "import torch; assert torch.cuda.is_available()" \
      || fail "CUDA still unavailable after repair attempt. Inspect the image's torch/CUDA build."
  else
    fail "GPU preflight failed (see remediation above)."
  fi
fi

# --- assemble matrix arguments from env ------------------------------------
MATRIX_ARGV=(--output-root "$OUTPUT_ROOT")
add_list() { local flag="$1" val="$2"; [[ -z "$val" ]] && return 0; MATRIX_ARGV+=("$flag"); local a; for a in $val; do MATRIX_ARGV+=("$a"); done; }
add_list --models    "${MODELS:-}"
add_list --datasets  "${DATASETS:-}"
add_list --phases    "${PHASES:-}"
[[ -n "${MAX_TASKS:-}" ]]    && MATRIX_ARGV+=(--max-tasks "$MAX_TASKS")
[[ -n "${BATCH_SIZE:-}" ]]   && MATRIX_ARGV+=(--batch-size "$BATCH_SIZE")
[[ -n "${QUANTIZATION:-}" ]] && MATRIX_ARGV+=(--quantization "$QUANTIZATION")
MATRIX_ARGV+=(--max-parallel "$MAX_PARALLEL" --vram-budget-frac "$VRAM_BUDGET_FRAC")
[[ "$GIT_CHECKPOINT" == "1" ]] && MATRIX_ARGV+=(--git-checkpoint)
[[ "$GIT_PUSH" == "1" ]]       && MATRIX_ARGV+=(--git-push)
if [[ "$MODE" == "smoke" ]]; then
  MATRIX_ARGV+=(--smoke)
  [[ -z "${MODELS:-}" ]] && MATRIX_ARGV+=(--models "$SMOKE_MODEL")
fi
[[ -n "${MATRIX_ARGS:-}" ]] && { for a in $MATRIX_ARGS; do MATRIX_ARGV+=("$a"); done; }
MATRIX_ARGV+=("$@")

STAMP="$(date +%Y%m%d_%H%M%S)"

# --- fast smoke gate before the long full run ------------------------------
if [[ "$MODE" == "full" && "$PREFLIGHT_SMOKE" == "1" ]]; then
  log "Preflight smoke ($SMOKE_MODEL, tiny scope) -- must pass before the full run..."
  PRE_ROOT="$OUTPUT_ROOT/_preflight"
  rm -rf "$PRE_ROOT" 2>/dev/null || true      # force a fresh, real run (not a resume-skip)
  set +e
  python -u research/run_experiment_matrix.py --smoke --models "$SMOKE_MODEL" \
      --phases collect analyze report --max-parallel 1 --no-progress \
      --output-root "$PRE_ROOT" 2>&1 | tee "$LOG_DIR/smoke_${STAMP}.log"
  smoke_rc=${PIPESTATUS[0]}
  set -e
  [[ $smoke_rc -eq 0 ]] || fail "Preflight smoke failed (exit $smoke_rc). See $LOG_DIR/smoke_${STAMP}.log. Set PREFLIGHT_SMOKE=0 to override."
  # Exit 0 is not enough -- assert the smoke actually produced valid traces.
  python - "$PRE_ROOT/${SMOKE_MODEL}__gsm8k" <<'PY' || fail "Preflight smoke produced no valid traces. Inspect $LOG_DIR/smoke_${STAMP}.log."
import csv, json, sys
from pathlib import Path
cell = Path(sys.argv[1])
meta, steps = cell / "metadata.json", cell / "trace_steps.csv"
if not meta.exists() or not steps.exists():
    print(f"[smoke] FAIL: missing outputs in {cell}"); sys.exit(1)
data = json.loads(meta.read_text(encoding="utf-8"))
if data.get("pending_run_count") not in (0, None):
    print(f"[smoke] FAIL: collection incomplete (pending={data.get('pending_run_count')})"); sys.exit(1)
with steps.open(newline="", encoding="utf-8", errors="replace") as fh:
    rows = sum(1 for _ in csv.DictReader(fh))
if rows < 1:
    print("[smoke] FAIL: trace_steps.csv has no rows"); sys.exit(1)
print(f"[smoke] OK: {rows} step rows recorded, collection complete.")
PY
  rm -rf "$PRE_ROOT" 2>/dev/null || true
  log "Preflight smoke passed (data validated). Proceeding to full run."
fi

# --- run the full matrix ---------------------------------------------------
RUN_LOG="$LOG_DIR/matrix_${MODE}_${STAMP}.log"
log "Launching matrix:"
printf '$ python -u research/run_experiment_matrix.py'; printf ' %q' "${MATRIX_ARGV[@]}"; printf '\n'
set +e
python -u research/run_experiment_matrix.py "${MATRIX_ARGV[@]}" 2>&1 | tee "$RUN_LOG"
matrix_rc=${PIPESTATUS[0]}
set -e

# --- final summary: name every failed step, set the job's exit code --------
log "Result summary (from $OUTPUT_ROOT/matrix_manifest.json):"
set +e
python - "$OUTPUT_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
man = root / "matrix_manifest.json"
if not man.exists():
    print("  no manifest written (collection may not have produced any cell).")
    sys.exit(0)
data = json.loads(man.read_text(encoding="utf-8"))
failed, ok = [], 0
for key, cell in sorted(data.get("cells", {}).items()):
    for stage in ("collect", "analyze", "report"):
        st = cell.get(stage)
        if not st:
            continue
        if st.get("status") == "failed":
            failed.append((key, stage, st.get("return_code")))
        elif st.get("status") == "ok":
            ok += 1
for ds, agg in sorted(data.get("aggregate", {}).items()):
    if agg.get("status") == "failed":
        failed.append((f"_aggregate/{ds}", "aggregate", agg.get("return_code")))
    elif agg.get("status") == "ok":
        ok += 1
print(f"  {ok} step(s) OK, {len(failed)} failed.")
for key, stage, code in failed:
    print(f"  FAILED  {key:<42} [{stage}] exit={code}  -> see {root / key / (stage + '.log')}")
sys.exit(1 if failed else 0)
PY
summary_rc=$?
set -e

if [[ $matrix_rc -ne 0 ]]; then
  fail "Matrix runner exited non-zero (exit $matrix_rc). Full log: $RUN_LOG"
fi
if [[ $summary_rc -ne 0 ]]; then
  printf '[runai-matrix] Completed WITH FAILURES. Full log: %s\n' "$RUN_LOG" >&2
  exit 1
fi
log "All matrix steps completed successfully. Full log: $RUN_LOG"
log "Per-dataset cross-family reports: $OUTPUT_ROOT/_aggregate/<dataset>/CROSS_FAMILY_REPORT.md"
