# ResearchThesis

This repository contains the overthinking-boundary research package: theory notes, a synthetic simulator, a real-trace harness for open-weight reasoning models, and downstream analysis scripts.

## Local entry points

- `python research/simulate_overthinking_boundary.py`
- `python research/real_trace_experiments.py --model qwen2p5_0p5b --device cpu --max-tasks 3 --max-steps 3 --max-new-tokens 16 --temperatures 0.2 0.8 --seeds 7 --output-dir research/outputs/real_traces_qwen`
- `python research/trace_analysis.py --input-dir research/outputs/real_traces_qwen`

## Google Colab

The guarded Colab runner is `tools/run_colab_experiment.py`. It is designed to avoid wasting GPU credits:

1. It checks the Python environment and GPU.
2. It runs the cheap synthetic simulator.
3. It runs a small smoke test first.
4. Only if the smoke test succeeds does it launch the full real-trace experiment.
5. It prints the key summaries and creates a zipped results archive.

Example Colab command after cloning the repo:

```bash
python tools/run_colab_experiment.py --model deepseek_r1_distill_1p5b
```

Useful variants:

- Smoke test only: `python tools/run_colab_experiment.py --smoke-only`
- Skip dependency installation if the runtime already has them: `python tools/run_colab_experiment.py --skip-install`
- Use the smaller model end-to-end: `python tools/run_colab_experiment.py --model qwen2p5_0p5b`

Dependencies for Colab are listed in `requirements-colab.txt`. The runner intentionally does not reinstall PyTorch so it preserves the GPU-enabled Colab build.