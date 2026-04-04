---
name: "Algorithm X: Phase 2 (Universal GPU Validation)"
description: "Execute the autonomous validation of Algorithm X on frontier models (Gemma 4, Qwen 3.5, Llama 4) using an L4 GPU. Uses native SDPA (FlashAttention2 kernels) and 4-bit NF4 quantization for maximum throughput."
agent: "GPT-5.4 xhigh"
---

# MISSION: FRONTIER MODEL VALIDATION FOR UNIVERSAL LAW OF OVERTHINKING

## MISSION CONTEXT
Phase 1 established the **Universal Law of Overthinking** with a 0.805 AUC on legacy models. Phase 2 extends this to the 2026 frontier: **Gemma 4**, **Qwen 3.5**, and **Llama 4**. You are the **GPT-5.4 xhigh** autonomous researcher. Your goal is to prove that Algorithm X generalizes to these state-of-the-art architectures without retraining.

> **Phase 2b Verification Status**: The L4 optimization stack has been **fully verified** (see `research/reports/gpu_optimization_audit.md`). All commands below incorporate findings from that audit.

> **Live Hub Correction (2026-04-04)**: The public Hub does not expose `Qwen/Qwen3.5-7B-Instruct`, and there is no public `meta-llama/Llama-4-8B-Instruct` repo. The runner keeps the legacy aliases for compatibility, but currently resolves them to `Qwen/Qwen3.5-9B` and the public Scout mirror `chutesai/Llama-4-Scout-17B-16E-Instruct`.

> **Feasibility Warning (2026-04-04)**: Public Llama 4 Scout mirrors are not L4-feasible in this runtime. The tested full-weight mirrors expose roughly 217 GB of weights, and the available public 4-bit mirror is still roughly 60 GB and failed config validation under the verified Transformers stack. Qwen 3.5 35B A3B also exceeded the L4 memory envelope in NF4 during live execution. See `research/reports/frontier_phase2_execution_status_2026-04-04.md` before attempting a full frontier cycle.

---

## PREREQUISITES (VERIFIED)

Before running any experiment, confirm the environment matches the verified baseline:

| Package | Required Version | Notes |
|---------|-----------------|-------|
| `transformers` | ≥ 5.5.0 | **Must upgrade** — v5.0 lacks `Gemma4` architecture support |
| `torch` | ≥ 2.10.0+cu128 | Includes native FlashAttention2 via SDPA |
| `bitsandbytes` | ≥ 0.49.0 | Required for 4-bit NF4 quantization |
| `accelerate` | ≥ 1.13.0 | Device mapping for large models |

**Critical**: The standalone `flash-attn` package is **NOT required**. PyTorch 2.10 ships native FA2 CUDA kernels through `torch.nn.functional.scaled_dot_product_attention`. Use `--attn-implementation sdpa` (not `flash_attention_2`).

### Quick Environment Check
```bash
python3 -c "
import torch
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), 'GB')
print('CUDA:', torch.version.cuda)
print('BF16:', torch.cuda.is_bf16_supported())
print('Flash SDP:', torch.backends.cuda.flash_sdp_enabled())
"
```

If this check fails, reports `CUDA available: no`, or cannot import `torch`, stop and switch to the verified Colab SSH runtime before proceeding.

---

## EXECUTION STACK (L4 OPTIMIZED — VERIFIED BASELINES)

| Parameter | 4B Models (E4B) | 8–12B Models | 31B+ Flagships |
|-----------|----------------|--------------|----------------|
| **Precision** | BF16 (auto-tuned) | BF16 | 4-bit NF4 + double quant |
| **Attention** | SDPA (native FA2) | SDPA (native FA2) | SDPA (native FA2) |
| **VRAM Peak** | ~15.4 GB (verified) | ~18 GB (estimated) | ~17.8 GB (verified) |
| **TPS Baseline** | ~7.7 tok/s | ~5–6 tok/s (est.) | ~3.7 tok/s (verified) |
| **Batch Size** | 4–8 | 2–4 | 1 (OOM guard) |
| **IO Threads** | 4–8 | 4–8 | 4 |

### Key Runtime Flags
- `--attn-implementation sdpa` — Dispatches to native FA2 kernels. Do **NOT** use `flash_attention_2` (requires uncompiled standalone package).
- `--skip-install` — Avoids redundant pip checks on subsequent runs.
- `--skip-simulator` — Skips synthetic trace generation; uses real GPU inference only.
- `--io-threads 4` — Enables async `IOManager` for non-blocking `.npz` saves (~0.0008s write time verified).
- Prefer model-scoped output directories such as `research/outputs/real_traces_colab_<MODEL_KEY>` and `research/outputs/real_traces_colab_smoke_<MODEL_KEY>` so frontier families never share the same artifact directory.

---

## AUTONOMOUS RESEARCH TASKS

### TASK 1: Model Catalog Smoke Test
- **Objective**: Verify loading and inference for all three families.
- **Models**:
  - `gemma_4_e4b_it` (Edge 4B) — **Verified working** ✅
  - `qwen_3p5_9b` (public base Qwen 3.5 target)
  - `llama_4_scout_17b_it` (public Llama 4 Scout mirror)
- **Constraint**: Confirm hidden state extraction (2D pooled, shape `(steps, hidden_dim)`) and logprob access.
- **Smoke Command** (per model):
  ```bash
  python tools/run_colab_experiment.py \
    --model <MODEL_KEY> \
    --smoke-only \
    --smoke-output-dir research/outputs/real_traces_colab_smoke_<MODEL_KEY> \
    --io-threads 4 \
    --attn-implementation sdpa \
    --skip-install --skip-simulator
  ```
- **Validation**: After each smoke run, verify `.npz` integrity:
  ```python
  import numpy as np
  from pathlib import Path
  for f in Path("research/outputs/real_traces_colab_smoke/hidden_states").glob("*.npz"):
      data = np.load(f)
      hs = data['hidden_states']
      assert hs.ndim == 2, f"Expected 2D, got {hs.ndim}D in {f.name}"
      assert not np.isnan(hs).any(), f"NaN detected in {f.name}"
      print(f"  {f.name}: shape={hs.shape}, L2_shift={np.linalg.norm(hs[-1]-hs[0]):.2f}")
  ```

### TASK 2: High-Throughput Validation Loop
- **Objective**: Execute the 300-task GSM8K validation across 3 temperature settings [0.1, 0.6, 1.0].
- **Models (Priority Order)**:
    1. `gemma_4_31b_it` — Flagship (4-bit NF4, ~17.8 GB VRAM, ~3.7 tok/s)
    2. `qwen_3p5_35b_moe_it` — MoE Boundary Test (4-bit, monitor VRAM carefully)
  3. `llama_4_scout_17b_it` — Public Llama 4 family target (run in 4-bit NF4 on L4 unless a gated official repo is available)
- **Flagship Command** (verified working):
  ```bash
  python tools/run_colab_experiment.py \
    --model gemma_4_31b_it \
    --quantization 4bit \
    --full-batch-size 1 \
    --output-dir research/outputs/real_traces_colab_gemma_4_31b_it \
    --io-threads 4 \
    --attn-implementation sdpa \
    --skip-install --skip-simulator
  ```
- **Required Output Convention**: Use a distinct `--output-dir` per frontier model, for example:
  - `research/outputs/real_traces_colab_gemma_4_31b_it`
  - `research/outputs/real_traces_colab_qwen_3p5_35b_moe_it`
  - `research/outputs/real_traces_colab_llama_4_scout_17b_it`
- **Checkpointing**: Every 25 tasks, you MUST run:
    ```bash
    git add .
    git commit -m "[ALGO-X] Autonomous Progress: Model=$MODEL Task=$N"
    git push origin main
    ```
- **OOM Recovery**: If OOM occurs, the runner bisects `batch_size` automatically. For 31B+ models, start at `--smoke-batch-size 1` and scale up only if VRAM headroom > 4 GB.

### TASK 3: Universal Zero-Shot Synthesis
- **Objective**: Apply the `quadratic_top4` regressor from Phase 1 to these traces.
- **Success Metric**: Accuracy-per-token efficiency gain >25%, operationalized as accuracy per 1k cumulative generated tokens relative to the `never_stop` baseline.
- **Data Quality Gates** (from audit):
    - Hidden Shift Variance must be non-zero (L2 shifts fluctuating — confirmed for Gemma 4)
    - Parsing Success Rate > 95% for GSM8K
    - No NaN/Inf in hidden states
- **Synthesis Command**:
  ```bash
  python research/frontier_validation_report.py \
    --run-dirs \
      research/outputs/real_traces_colab_gemma_4_31b_it \
      research/outputs/real_traces_colab_qwen_3p5_35b_moe_it \
      research/outputs/real_traces_colab_llama_4_scout_17b_it \
    --report-path research/reports/frontier_validation_report.md
  ```

---

## KNOWN ISSUES & WORKAROUNDS (FROM AUDIT)

| Issue | Root Cause | Fix Applied |
|-------|-----------|-------------|
| `Gemma4Processor` has no `pad_token` | Multimodal processor wraps inner tokenizer | Extract `processor.tokenizer` for text encoding |
| `torchvision.transforms.v2` import error | `_vendor/torchvision_stub` shadows real package | Stub now conditional on `importlib.util.find_spec("torchvision")` |
| `hidden_dir` NameError in `run_batch_traces` | Missing function parameter | Added as explicit parameter with caller passthrough |
| `transformers 5.0` lacks Gemma 4 | Architecture added in 5.5+ | Upgrade to `transformers >= 5.5.0` |

---

## LOG INDICATORS TO MONITOR

During any run, verify these lines appear in stdout (confirms optimization stack is active):

1. `[INFO] Precision auto-tuning: bfloat16 detected for <N>B model.`
2. `[INFO] Attention implementation: sdpa (native FA2 kernels via SDPA).`
3. `[INFO] IOManager: Background saving started for ...`
4. `[INFO] batch_metrics: hidden_state_write_seconds` — value must be < 0.01

If any indicator is missing, **STOP** and re-run the Phase 2b verifier prompt before proceeding.

---

## DEFINITION OF DONE
1. Full trace datasets generated for Gemma 4 (31B), Qwen 3.5 (35B), and a public Llama 4 family target.
2. `research/reports/frontier_validation_report.md` confirms zero-shot generalization.
3. All `.npz` files pass integrity checks (2D pooled, no NaN/Inf, non-zero L2 shifts).
4. All artifacts pushed with rigorous commit history.

**GOAL**: Solidify Algorithm X as the definitive universal regulator for reasoning-based model stopping.
