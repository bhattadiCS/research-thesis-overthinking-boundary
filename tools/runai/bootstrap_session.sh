#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/bhattadiCS/research-thesis-overthinking-boundary.git}"
REPO_NAME="${REPO_NAME:-research-thesis-overthinking-boundary}"
WORKDIR="${WORKDIR:-$PWD}"
REPO_DIR="${REPO_DIR:-$WORKDIR/$REPO_NAME}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
MODEL="${MODEL:-deepseek_r1_distill_1p5b}"
MODE="${MODE:-smoke}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
PULL_LATEST="${PULL_LATEST:-1}"
RUN_SIMULATOR="${RUN_SIMULATOR:-0}"

EXTRA_ARGS=("$@")

log() {
  printf '[runai-bootstrap] %s\n' "$1"
}

fail() {
  printf '[runai-bootstrap] ERROR: %s\n' "$1" >&2
  exit 1
}

command -v git >/dev/null 2>&1 || fail 'git is required but not installed.'
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "${PYTHON_BIN} is required but not installed."

if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  fail "MODE must be 'smoke' or 'full' (got: $MODE)"
fi

if [[ ! -d "$REPO_DIR/.git" ]]; then
  log "Cloning repository into $REPO_DIR"
  git clone --depth 1 "$REPO_URL" "$REPO_DIR"
else
  log "Using existing repository at $REPO_DIR"
  remote_url="$(git -C "$REPO_DIR" remote get-url origin 2>/dev/null || true)"
  if [[ -n "$remote_url" && "$remote_url" != *"research-thesis-overthinking-boundary"* ]]; then
    fail "Existing repository remote does not look like the thesis repo: $remote_url"
  fi

  if [[ "$PULL_LATEST" == "1" ]]; then
    if [[ -n "$(git -C "$REPO_DIR" status --porcelain)" ]]; then
      log 'Skipping git pull because the worktree has local changes.'
    else
      log 'Pulling latest changes with --ff-only'
      git -C "$REPO_DIR" pull --ff-only
    fi
  fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

log 'Upgrading pip'
python -m pip install --upgrade pip

if [[ "$SKIP_INSTALL" != "1" ]]; then
  log 'Installing Python dependencies from requirements-colab.txt'
  python -m pip install -r "$REPO_DIR/requirements-colab.txt"
else
  log 'Skipping dependency installation because SKIP_INSTALL=1'
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  log 'Logging into Hugging Face using HF_TOKEN'
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
else
  log 'HF_TOKEN not set. If a gated model fails to download, run: huggingface-cli login'
fi

cd "$REPO_DIR"

log 'Running environment check'
python tools/runai/check_env.py

COMMAND=(python tools/runai/run_experiment.py --model "$MODEL" --mode "$MODE")
if [[ "$RUN_SIMULATOR" == "1" ]]; then
  COMMAND+=(--run-simulator)
fi
COMMAND+=("${EXTRA_ARGS[@]}")

log "Launching experiment wrapper in $MODE mode for model $MODEL"
printf '$'
for token in "${COMMAND[@]}"; do
  printf ' %q' "$token"
done
printf '\n'
"${COMMAND[@]}"

log 'Session bootstrap finished successfully.'
