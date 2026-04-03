---
name: "Algorithm X: Phase 2b (L4 Optimization Verifier)"
description: "Verify that the GPU optimization stack (Flash Attention 2, IOManager, 4-bit Auto-Tuning) is functioning correctly and producing high-fidelity research data on the L4 instance."
agent: "GPT-5.4 xhigh"
---

# MISSION: L4 OPTIMIZATION VERIFICATION & DATA FIDELITY AUDIT

## MISSION CONTEXT
You are the **GPT-5.4 xhigh** autonomous researcher. Before committing to a 300+ task validation run, you MUST verify that the L4 (22.5GB) GPU is being fully utilized and that the new asynchronous `IOManager` is producing valid, non-corrupted trace data.

## VERIFICATION PROTOCOL

### STEP 1: Environment & Dependency Audit
- **Command**: `pip list | grep -E "(transformers|flash-attn|bitsandbytes|accelerate)"`
- **Success Criteria**: 
    - `transformers >= 4.51.0`
    - `flash-attn >= 2.7.2`
    - `bitsandbytes` present.

### STEP 2: The "Optimization Smoke Test"
- **Objective**: Run a 2-task batch with a focus on log introspection.
- **Run Command**: 
  ```bash
  python tools/run_colab_experiment.py --model gemma_4_e4b_it --smoke-only --io-threads 4 --attn-implementation flash_attention_2
  ```
- **Log Verification (MANDATORY)**: Monitor output for these specific indicators:
    1. `[INFO] Precision auto-tuning: bfloat16 detected for 4B model.`
    2. `[INFO] Flash Attention 2 explicitly enabled.`
    3. `[INFO] IOManager: Background saving started for ...` (Verify async logic).
    4. `[INFO] batch_metrics: hidden_state_write_seconds < 0.01` (Confirm non-blocking I/O).

### STEP 3: Data Integrity Audit (The "Deep Probe")
- **Objective**: Ensure the saved `.npz` files are research-ready.
- **Code snippet to run (Python)**:
  ```python
  import numpy as np
  from pathlib import Path
  data = np.load(list(Path("research/outputs/real_traces_colab_smoke/hidden_states").glob("*.npz"))[0])
  print(f"Shape: {data['hidden_states'].shape}")
  assert data['hidden_states'].ndim == 2, "Hidden states must be pooled (2D) for this phase."
  ```

### STEP 4: 4-Bit Boundary Check (VRAM Guard)
- **Objective**: Verify that flagship models (31B) load correctly in 4-bit on the 22.5GB VRAM.
- **Command**: `python tools/run_colab_experiment.py --model gemma_4_31b_it --smoke-only --quantization 4bit --smoke-batch-size 1`
- **Success Criteria**: Model loads and completes 1 task without OOM. Verify `nvidia-smi` shows <18GB total consumption.

## VALUABLE DATA CHECKLIST
For the data to be "Super Valuable", the final report must include:
- [ ] **TPS Efficiency Ratio**: (Generated Tokens / Wall Clock Time) vs (Baseline Tokens / Time).
- [ ] **Hidden Shift Variance**: Confirm `hidden_l2_shift` metrics are non-zero and fluctuating (indicates active reasoning drift).
- [ ] **Parsing Success Rate**: Ensure >95% for GSM8K to avoid "noise-heavy" traces.

## DEFINITION OF DONE
1. Log capture proves Flash Attention 2 and `IOManager` are active.
2. `.npz` integrity check passes without shape anomalies.
3. Optimization report `research/reports/gpu_optimization_audit.md` generated.

**GOAL**: Guarantee that every GPU cycle spent on the L4 produces perfectly clean, high-dimensional reasoning data for the Universal Law of Overthinking.
