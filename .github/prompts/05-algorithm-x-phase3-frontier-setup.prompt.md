---
name: "Algorithm X: Phase 3 — Frontier Validation Execution & Equation Promotion"
description: "Transition from CPU-audit to full GPU frontier validation. Retire stale model aliases, wire the recommended hazard equation into the pipeline, perform the full 300-task GSM8K trace collection on verified frontier families (Gemma 4, Qwen 3.5), and determine the final research-grade stopping rule."
agent: "GPT-5.4 xhigh"
---

# MISSION: PHASE 3 — FRONTIER VALIDATION AND EQUATION PROMOTION

You are starting Phase 3 of the Algorithm X research program. The prior CPU-only audit (Phase 2.5) successfully reconciled the repository drift, diagnosed parse-success failures, and identified the best candidate equation for promotion. 

Your job is now to transition back to live GPU execution (NVIDIA L4) to complete the full frontier validation set and finalize the mathematical stopping rule for the thesis.

## 1. CRITICAL TRUTHS FROM THE CPU AUDIT

### 1.1 Equation Promotion Status
- **Current Deployed Default**: `quadratic_top4` (Basis: Quadratic, Features: entropy_mean, answer_changed, thought_token_count, hidden_l2_shift).
- **Recommended Successor**: `hazard_quadratic_combo_entropymean_entropystd_confidence_thoughttokencount` (Basis: Quadratic, Features: entropy_mean, entropy_std, confidence, thought_token_count).
- **Finding**: The successor has better boundary accuracy (0.51 vs 0.40) and a smaller oracle gap (0.25 vs 0.33) on legacy data, despite slightly lower raw AUC.

### 1.2 Model Catalog Integrity
- **Verified Aliases**: `gemma_4_e4b_it`, `qwen_3p5_9b`, `gemma_4_31b_it`.
- **Dead/Stale Aliases**: `qwen_3p5_7b_instruct` (does not exist), `llama_4_8b_it` (use `llama_3p1_8b_instruct` or similar).

### 1.3 Parse-Success Diagnosis
- **Qwen 3.5 9B**: Failures were due to prompt-format mismatches and truncation. Fallback answer recovery worked, but formal parse rates were 0.00.
- **DeepSeek 1.5B**: Low success due to long prose and repeated token truncation.

---

## 2. HARD CONSTRAINTS

1. **Hardware**: NVIDIA L4 (24GB VRAM) environment required.
2. **Persistence**: Use `git` for checkpointing every 25 tasks.
3. **Trace Isolation**: Each model MUST have its own directory: `research/outputs/real_traces_colab_<MODEL_KEY>`.

---

## 3. FIRST ACTIONS: CODE REFINEMENT

### TASK A: Retire Stale Model Aliases
Update `research/real_trace_experiments.py` and `tools/run_colab_experiment.py` to:
1. Remove `qwen_3p5_7b_instruct` and `llama_4_8b_it` from the `MODEL_CATALOG`.
2. Ensure `qwen_3p5_9b` is the primary Qwen 3.5 target.
3. Add `llama_3p1_8b_instruct` as the validated Llama 3 baseline.

### TASK B: Wire Recommended Equation into Pipeline
Update `research/universal_feature_analysis.py` to:
1. Add `hazard_quadratic_combo_entropymean_entropystd_confidence_thoughttokencount` to the `MODEL_SPECS` tuple.
2. Ensure the feature set for this model uses: `("entropy_mean", "entropy_std", "confidence", "thought_token_count")`.

---

## 4. PRIMARY EXECUTION: FULL FRONTIER COLLECTION

### TASK C: Run Full 300-Task GSM8K Traces
Execute the full collection protocol on the following models (300 tasks, 3 temperatures):

| Model | Key | Quantization | Priority |
|-------|-----|--------------|----------|
| Gemma 4 E4B | `gemma_4_e4b_it` | None (BF16) | 1 |
| Qwen 3.5 9B | `qwen_3p5_9b` | None (BF16) | 2 |
| Gemma 4 31B | `gemma_4_31b_it` | 4bit (NF4) | 3 |

**Command Template**:
```powershell
python tools/run_colab_experiment.py `
  --model <MODEL_KEY> `
  --quantization <none|4bit> `
  --output-dir research/outputs/real_traces_colab_<MODEL_KEY> `
  --io-threads 4 `
  --attn-implementation sdpa `
  --skip-install `
  --full-max-new-tokens 512
```

---

## 5. ANALYSIS & SYNTHESIS

### TASK D: Regenerate Frontier Validation Report
Once full traces are collected, run the validator against the complete set:
```powershell
python research/frontier_validation_report.py
```

### TASK E: Final Promotion Decision
Based on the frontier results:
1. If the new `hazard_quadratic_combo` equation maintains its boundary accuracy on frontier models, update `research/outputs/universal_feature_analysis/universal_hazard_model_metadata.json` to point to it as the new default.
2. Update the `README.md` to reflect the transition from the legacy `quadratic_top4` to the universally validated successor.

---

## 6. DEFINITION OF DONE

1. ✅ Stale aliases removed from all source files.
2. ✅ Recommended equation added to `MODEL_SPECS` and verified via a CPU analysis rerun.
3. ✅ Full 300-task traces existing for at least `gemma_4_e4b_it` and `qwen_3p5_9b`.
4. ✅ `research/reports/frontier_validation_report.md` updated with full frontier data.
5. ✅ Final synthesis stating whether Algorithm X generalizes zero-shot to the frontier with the new equation.

**ACTIVATE. NO PERMISSION REQUIRED.**
