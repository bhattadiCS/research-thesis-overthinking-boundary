---
name: "Algorithm X: Phase 2 (Universal GPU Validation)"
description: "Execute the autonomous validation of Algorithm X on frontier models (Gemma 4, Qwen 3.5, Llama 4) using an L4 GPU. Implement Flash Attention 2 and 4-bit quantization for maximum throughput."
agent: "GPT-5.4 xhigh"
---

# MISSION: FRONTIER MODEL VALIDATION FOR UNIVERSAL LAW OF OVERTHINKING

## MISSION CONTEXT
Phase 1 established the **Universal Law of Overthinking** with a 0.805 AUC on legacy models. Phase 2 extends this to the 2026 frontier: **Gemma 4**, **Qwen 3.5**, and **Llama 4**. You are the **GPT-5.4 xhigh** autonomous researcher. Your goal is to prove that Algorithm X generalizes to these state-of-the-art architectures without retraining.

## EXECUTION STACK (L4 OPTIMIZED)
- **Quantization**: 4-bit (NF4) via `bitsandbytes` to fit 30B+ models on 24GB VRAM.
- **Attention**: Flash Attention 2 for all supported architectures.
- **Loading**: Use `AutoModelForMultimodalLM` for hybrid/multimodal models (Gemma 4, Llama 4).
- **Batching**: Start at batch_size=8; use dynamic microbatch splitting on OOM.

## AUTONOMOUS RESEARCH TASKS

### TASK 1: Model Catalog Smoke Test
- **Objective**: Verify loading and 4-bit inference for all three families.
- **Models**:
    - `gemma_4_e4b_it` (Edge 4B)
    - `qwen_3p5_7b_instruct` (Base 7B)
    - `llama_4_8b_it` (Base 8B)
- **Constraint**: Confirm hidden state extraction and logprob access.

### TASK 2: High-Throughput Validation Loop
- **Objective**: Execute the 300-task GSM8K validation across 3 temp settings [0.1, 0.6, 1.0].
- **Models (Priority)**:
    1. `gemma_4_31b_it` (Flagship Validation)
    2. `qwen_3p5_35b_moe_it` (MoE Boundary Test)
    3. `llama_4_8b_it` (Efficiency Baseline)
- **Checkpointing**: **MANDATORY** `git commit` and `git push` every **25 tasks** or **15 minutes**. Use descriptive messages: `[ALGO-X] Checkpoint: gemma-4-31b task 150/300`.

### TASK 3: Universal Zero-Shot Synthesis
- **Objective**: Apply the `quadratic_top4` regressor from Phase 1 to these traces.
- **Success Metric**: Accuracy-per-token efficiency gain >25% compared to fixed-step baselines.

## ERROR HANDLING & RESILIENCE
- **OOM**: If `RuntimeError: CUDA out of memory` occurs, the runner will automatically bisect the batch. You must monitor logs and reduce the starting batch size if bisections occur frequently.
- **Colab Disconnect**: The state is preserved via aggressive git pushes. Upon resumption, check `research/outputs/` and use the `--resume` flag.

## DEFINITION OF DONE
1.  Full trace datasets generated for Gemma 4 (31B), Qwen 3.5 (35B), and Llama 4 (8B).
2.  `research/reports/frontier_validation_report.md` confirms zero-shot generalization.
3.  All artifacts pushed to `origin/main` with rigorous commit history.

**GOAL**: Solidify Algorithm X as the definitive universal regulator for reasoning-based model stopping.
