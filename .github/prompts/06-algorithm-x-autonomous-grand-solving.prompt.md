---
name: "Algorithm X: The Grand 30-Day Autonomous Solving Protocol"
description: "A comprehensive, high-intensity research mission to identify and prove the Universal Law of Overthinking across all frontier AI families. This protocol involves recursive experimentation, real-to-sim calibration, and final mathematical derivation."
agent: "GPT-5.4 xhigh"
---

# MISSION: THE UNIVERSAL LAW OF OVERTHINKING (ALGORITHM X)

You are tasked with the final, definitive resolution of the **Algorithm X research program**. You have 30 days of autonomous local/GPU hybrid execution to prove the existence of an "Overthinking Boundary" ($\tau$)—a universal point in the reasoning trajectory where the probability of error $(\beta)$ begins to rise faster than the probability of correction $(\alpha)$.

**YOUR GOAL**: Produce a scientifically rigorous, peer-review-grade report at `research/reports/THE_UNIVERSAL_LAW.md` that defines the core boundary equation ($\tau = f(\dots)$) and validates it zero-shot against at least 4 frontier model families.

---

## 1. RECURSIVE RESEARCH LOOP (THE OODA LOOP)

You must operate in a continuous, self-correcting cycle. Every 1-2 hours of execution, you must:

1. **OBSERVE**: Scan `research/AUTONOMOUS_RUN_LOG.md` and the `research/outputs/` directory. Check for new traces, parse failures, or statistical anomalies.
2. **ORIENT**: Rerun `python research/universal_feature_analysis.py --all-models` to recalculate the best-performing coefficients across all available data.
3. **DECIDE**: Identify the "weakest link"—which model family or temperature has the worst boundary accuracy? Which simulator parameter is most poorly calibrated?
4. **ACT**: Execute the necessary experiment (Command in Section 4).
5. **CHECKPOINT**: `git add .`, `git commit -m "Step [X]: [Brief Summary]"`, `git push origin main`.

---

## 2. CRITICAL TRUTHS & HARD CONSTRAINTS

- **Environment**: NVIDIA L4 GPU (24GB) via Google Colab.
- **Hardware Efficiency**: Use `--attn-implementation sdpa` and `--quantization 4bit` for models > 10B parameters.
- **Persistence**: Log all decisions and hypotheses in `research/AUTONOMOUS_RUN_LOG.md`.
- **Integrity**: Never fabricate traces. If a model fails to load, diagnose the VRAM error and pivot to a different quantization or family.

---

## 3. PHASE A: CODE & HYPOTHESIS ALIGNMENT (IMMEDIATE)

### TASK 1: Model Alias Cleanup
Retire stale aliases to prevent runtime crashes.
- **Remove**: `qwen_3p5_7b_instruct`, `llama_4_8b_it`.
- **Verified Primary Targets**: `gemma_4_e4b_it`, `qwen_3p5_9b`, `gemma_4_31b_it`, `llama_3p1_8b_instruct`.
- **Files**: `research/real_trace_experiments.py`, `tools/run_colab_experiment.py`.

### TASK 2: Equation Specification Injection
Add the current champion successor to `research/universal_feature_analysis.py`:
- **ID**: `hazard_quadratic_combo_entropymean_entropystd_confidence_thoughttokencount`
- **Features**: `("entropy_mean", "entropy_std", "confidence", "thought_token_count")`
- **Basis**: `quadratic`

---

## 4. PHASE B: HIGH-INTENSITY FRONTIER COLLECTION

Execute the full 300-task GSM8K collection sweep (Temperatures: 0.2, 0.4, 0.8).

**Priority Queue**:
1. `gemma_4_e4b_it` (16bit)
2. `qwen_3p5_9b` (16bit)
3. `llama_3p1_8b_instruct` (16bit)
4. `gemma_4_31b_it` (4bit)

**Command Template**:
```powershell
python tools/run_colab_experiment.py `
  --model <MODEL_KEY> `
  --quantization <none|4bit> `
  --output-dir research/outputs/real_traces_colab_<MODEL_KEY> `
  --io-threads 4 `
  --attn-implementation sdpa `
  --full-max-new-tokens 512 `
  --num-tasks 300
```

---

## 5. PHASE C: THE "REAL-TO-SIM" BRIDGE (DEEP ANALYTICS)

The theoretical validator `research/simulate_overthinking_boundary.py` must be calibrated against the real traces.

1. **Calculate Learning Rates**: Extract the empirical $\alpha$ (correction probability) and $\beta$ (error induction) from the frontier traces.
2. **Inject into Simulator**: Update the `ScenarioConfig` in `simulate_overthinking_boundary.py` to match the observed dynamics of a specific model (e.g., "Gemma 4 mode").
3. **Zero-Shot Boundary Prediction**: Can a stopping rule trained on **Simulated Data** predict the **Real Oracle Stop** on frontier models?

---

## 6. PHASE D: FINAL SYNTHESIS & PUBLICATION

Generate the final artifact: `research/reports/THE_UNIVERSAL_LAW.md`. It must contain:

- **The Equation**: The final, coefficient-locked mathematical formula for the overthinking boundary.
- **The Evidence**: A table showing the **Oracle Gap Reduction** (%) achieved by Algorithm X over naive PRM-argmax across all tested families.
- **The Theorem**: A formal statement explaining *why* the boundary exists (e.g. "The accumulation of error-biased tokens grows non-linearly while the correction capacity decays exponentially").

---

**ACTIVATE. NO PERMISSION REQUIRED. THE THESIS FINALIZATION STARTS NOW.**
