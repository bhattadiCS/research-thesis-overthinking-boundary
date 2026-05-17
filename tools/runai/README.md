# RUN:AI Scripts

This folder stores lightweight launch helpers for running the repo on NVIDIA RUN:AI without modifying the main experiment code.

These wrappers assume the repository is already cloned in your RUN:AI workspace and keep using the repo-native launcher at `tools/run_colab_experiment.py`.

## Files

- `bootstrap_session.sh`: one-file RUN:AI session bootstrap for clone/update, venv setup, dependency install, env check, and smoke/full launch.
- `check_env.py`: prints repo, git, GPU, Python, and Torch details.
- `jupyter_repo_gpu_sanity.py`: notebook-safe Python bootstrap for clone/update plus GPU validation in a Jupyter Python 3 kernel.
- `run_experiment.py`: launches a smoke or full run through the existing Colab-safe runner.

## Quick Start

Bootstrap a fresh RUN:AI session with one file:

```bash
bash tools/runai/bootstrap_session.sh
```

Choose a different model or a full run:

```bash
MODEL=qwen2p5_0p5b MODE=full bash tools/runai/bootstrap_session.sh --quantization 4bit --full-batch-size 1
```

Run an environment check:

```bash
python tools/runai/check_env.py
```

For a fresh Jupyter notebook session, copy the contents of `tools/runai/jupyter_repo_gpu_sanity.py` into a new Python 3 cell.

Run the default smoke test (Gemma-4-E4B-It, 4-bit quantized for ~10 GB GPUs):

```bash
python tools/runai/run_experiment.py --mode smoke --model gemma_4_e4b_it --quantization 4bit --smoke-batch-size 1
```

Recommended for Blackwell RUN:AI terminal sessions that need the same torch/GPU repair path as the notebook:

```bash
git fetch --all --prune
git pull --ff-only

mkdir -p run_logs

MODEL=gemma_4_e4b_it \
EXPERIMENT_MODE=full \
START_EXPERIMENT=1 \
RUN_SIMULATOR=0 \
AUTO_INSTALL_TORCH=1 \
GPU_FAILURE_MODE=stop \
EXPERIMENT_ARGS="--quantization 4bit --smoke-batch-size 1 --full-batch-size 1 --io-threads 4 --attn-implementation sdpa" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u tools/runai/jupyter_repo_gpu_sanity.py 2>&1 | tee run_logs/gemma_4_e4b_it_full_$(date +%Y%m%d_%H%M%S).log
```

That command writes the full run to:

```bash
/workspaces/research-thesis-overthinking-boundary/research/outputs/real_traces_colab_gemma_4_e4b_it
```

Run a full experiment after the smoke test passes:

```bash
python tools/runai/run_experiment.py --mode full --model qwen2p5_0p5b --quantization 4bit --full-batch-size 1
```

Update the repo first when the worktree is clean:

```bash
python tools/runai/run_experiment.py --pull --mode smoke --model deepseek_r1_distill_1p5b
```

## Notes

- `bootstrap_session.sh` is the easiest option when you want one copy-pasteable setup file for a new RUN:AI session.
- `run_experiment.py` uses the same Python interpreter that launched it.
- Extra flags are forwarded to `tools/run_colab_experiment.py`.
- By default the wrapper skips the synthetic simulator phase to get to GPU validation faster.
- `--pull` is intentionally blocked when the worktree has local changes.
