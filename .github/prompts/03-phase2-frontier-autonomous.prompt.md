---
name: "Algorithm X: Phase 2 — Autonomous Frontier Validation (Hardware-Constrained L4 Protocol)"
description: "Continue the interrupted Phase 2 validation of Algorithm X on open-source frontier models using a single NVIDIA L4 GPU. Incorporates all verified findings, corrected model catalogs, and fallback strategies from the prior session. Primary objective: prove Algorithm X generalizes universally and refine the overthinking detection equation."
---

# MISSION: AUTONOMOUS FRONTIER VALIDATION — ALGORITHM X (UNIVERSAL LAW OF OVERTHINKING)

> **CRITICAL CONTEXT**: This prompt continues an interrupted session. A prior agent completed significant verification work: the L4 software baseline was established, the model catalog was corrected, smoke tests passed for 2 frontier families, and several dead-end fallback paths were explored. **All of that context is embedded below so you do NOT need to rediscover it.**

---

## 1. RESEARCH OBJECTIVE

### 1.1 What Is Algorithm X?
Algorithm X is a **universal zero-shot stopping rule** for reasoning LLMs. It detects the exact moment when additional chain-of-thought tokens transition from being helpful (repair) to harmful (corruption). The core claim is the **Universal Law of Overthinking**: there exists a critical step $T_c$ where the expected utility of continued reasoning crosses from positive to negative — and this crossing can be predicted from model-agnostic observable signals.

### 1.2 The Mathematical Framework
The core equation is the **Predictable Drift**:

$$\mu_t = (1 - q_t)\alpha_t - q_t\beta_t - \lambda$$

Where:
- $q_t = \mathbb{P}(C_t = 1 \mid \mathcal{F}_t)$ — current correctness probability (estimated via logistic regression on observable features)
- $\alpha_t$ — **Repair Hazard**: rate at which an incorrect answer becomes correct at the next step
- $\beta_t$ — **Corruption Hazard**: rate at which a correct answer degrades into an incorrect one
- $\lambda = 0.05$ — per-step compute cost penalty (constant, defined in `real_trace_experiments.py` as `STEP_COST`)
- $\mathcal{F}_t = \sigma(R_{1:t}, A_{1:t}, Z_{1:t})$ — observable filtration (all context up to step $t$)

**Structural One-Crossing Theorem**: If $q_t$ is nondecreasing, $\alpha_t$ is nonincreasing, and $\beta_t$ is nondecreasing, then $\mu_t$ strictly crosses from positive to negative exactly once. This is the Overthinking Boundary.

### 1.3 The Current Universal Estimator
The selected model from Phase 1 is `quadratic_top4`:
- **Basis**: Quadratic polynomial expansion (degree=2, interactions + squares)
- **Base Features (4)**: `entropy_mean`, `answer_changed`, `thought_token_count`, `hidden_l2_shift`
- **Fitting Scope**: Trained on "capable families only" (Mistral 7B, Qwen 7B from Phase 1)
- **Pipeline**: PolynomialFeatures → StandardScaler → LogisticRegression (balanced, C=1.0, max_iter=5000)
- **Phase 1 AUC**: 0.805 on legacy models

Three separate logistic regressors are trained for $\hat{q}_t$, $\hat{\alpha}_t$, and $\hat{\beta}_t$, then composed:

$$\hat{\mu}_t = (1 - \hat{q}_t)\hat{\alpha}_t - \hat{q}_t\hat{\beta}_t - 0.05$$

Algorithm X stops reasoning at the first step $t \geq 2$ where $\hat{\mu}_t \leq 0$.

### 1.4 Primary Research Questions
1. **Does Algorithm X generalize zero-shot to 2026 frontier architectures?** (Gemma 4, Qwen 3.5)
2. **Is the current equation optimal?** Is the quadratic basis over 4 features the right complexity, or should we explore:
   - Higher-order polynomial bases?
   - Different feature subsets from the 6 candidates: `entropy_mean`, `entropy_std`, `confidence`, `hidden_l2_shift`, `answer_changed`, `thought_token_count`?
   - Non-parametric approaches (random forests, gradient boosting)?
   - Raw linear algebra formulations (PCA on hidden state trajectories, spectral analysis of drift sequences)?
3. **What mathematical signature reliably detects overthinking across ALL model families?**

---

## 2. WHAT HAS ALREADY BEEN ACCOMPLISHED (DO NOT REDO)

### 2.1 Verified L4 Software Baseline
The prior session established and verified this environment on a Google Colab L4 instance:

| Package | Verified Version | Status |
|---------|-----------------|--------|
| `torch` | 2.10.0+cu128 | ✅ Native FA2 via SDPA |
| `transformers` | 5.5.0 | ✅ Gemma 4 architecture support |
| `bitsandbytes` | 0.49.2 | ✅ NF4 quantization |
| `accelerate` | 1.13.0 | ✅ Device mapping |
| `evaluate` | installed | ✅ |

**Hardware**: NVIDIA L4 (22.0 GB VRAM), PCIe Gen 4, CUDA 12.8, BF16 supported, Flash SDP enabled.

### 2.2 Smoke Tests PASSED ✅
Two frontier families have been verified working on the L4:

| Model Key | HF Name | Family | Precision | VRAM Peak | TPS | L2 Shifts | Status |
|-----------|---------|--------|-----------|-----------|-----|-----------|--------|
| `gemma_4_e4b_it` | `google/gemma-4-E4B-it` | Gemma 4 Edge | BF16 | ~15.4 GB | ~7.7 | 30.54, 18.90 | ✅ PASSED |
| `qwen_3p5_9b` | `Qwen/Qwen3.5-9B` | Qwen 3.5 | BF16 | ~18 GB | ~5-6 | 45.39, 18.35 | ✅ PASSED |

Both produced valid 2D pooled hidden states with non-zero L2 shifts, confirming:
- SDPA native FA2 dispatching correctly
- Async IOManager writing hidden states in < 0.01s
- BF16 precision auto-tuning working
- No NaN/Inf in hidden states

### 2.3 Flagship Smoke Tests ALSO PASSED ✅
| `gemma_4_31b_it` | 4-bit NF4 | 17.76 GB peak | 3.74 tok/s | ✅ Verified |
| `qwen_3p5_35b_moe_it` | 4-bit NF4 | Started download (71.9 GB) | ❌ OOM on load (modules spilled to CPU/disk) |

### 2.4 Catalog Corrections Applied
The prior session corrected stale HF repo IDs in `real_trace_experiments.py`. The current `MODEL_CATALOG` in the repo contains:

```python
# VERIFIED WORKING entries:
"gemma_4_31b_it":     "google/gemma-4-31B-it"        # 31B, Gemma 4
"gemma_4_e4b_it":     "google/gemma-4-E4B-it"        # 4B, Gemma 4 Edge
"gemma_4_26b_moe_it": "google/gemma-4-26B-A4B-it"    # 26B MoE, Gemma 4

# STALE/NON-RESOLVING entries (need correction):
"qwen_3p5_7b_instruct": "Qwen/Qwen3.5-7B-Instruct"  # ❌ Does not exist on HF
"qwen_3p5_35b_moe_it":  "Qwen/Qwen3.5-35B-A3B"      # ✅ Exists but too large for L4 NF4
"llama_4_8b_it":         "meta-llama/Llama-4-8B-Instruct"  # ❌ Does not exist on HF
```

**You MUST add these corrected/new entries to the MODEL_CATALOG before running:**
```python
# VERIFIED PUBLIC entries to ADD:
"qwen_3p5_9b":           ModelSpec("qwen_3p5_9b", "Qwen/Qwen3.5-9B", "Qwen3.5", "9B")
"qwen_3p5_35b_gptq_int4": ModelSpec("qwen_3p5_35b_gptq_int4", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4", "Qwen3.5 MoE", "35B")
"llama_3p1_8b_instruct":  ModelSpec("llama_3p1_8b_instruct", "meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1", "8B")  # Already exists
# And any other open-source models you discover that can fit the L4
```

### 2.5 Blockers Discovered and Explored

| Blocker | Root Cause | Status |
|---------|-----------|--------|
| `Qwen/Qwen3.5-7B-Instruct` | Repo does not exist on HF | **RESOLVED**: Use `Qwen/Qwen3.5-9B` instead |
| `meta-llama/Llama-4-8B-Instruct` | Repo does not exist; Llama 4 public releases are Scout (17B×16E MoE) | **RESOLVED**: Use Llama 3.1 8B or explore other Llama models |
| Llama 4 Scout mirrors | 217 GB full / 60 GB 4-bit; `attn_temperature_tuning` config bug | Config patch applied locally but model still too large |
| Qwen 35B NF4 on L4 | 71.9 GB download → modules spill to CPU → OOM | **PARTIALLY EXPLORED**: GPTQ Int4 variant (~24 GB) attempted |
| GPTQ Int4 Qwen 35B | `out_features=1` layer crashes Triton kernel | **PARTIALLY EXPLORED**: Config patched to `backend: torch`, disk offload added, but session was interrupted before completion |
| `Gemma4Processor` no `pad_token` | Multimodal processor wraps inner tokenizer | **FIXED**: Extract `processor.tokenizer` for text encoding |
| `torchvision.transforms.v2` import error | `_vendor/torchvision_stub` shadows real package | **FIXED**: Stub now conditional |
| `hidden_dir` NameError | Missing function parameter | **FIXED**: Added as explicit parameter |
| `transformers 5.0` lacks Gemma 4 | Architecture added in 5.5+ | **FIXED**: Upgraded requirement |

### 2.6 Code Patches Applied in Prior Session
The prior session modified these files (changes already committed to `origin/main`):
1. `tools/run_colab_experiment.py` — Added model-scoped output directories, version enforcement
2. `research/real_trace_experiments.py` — Fixed tokenizer handling, hidden_dir parameter, torchvision stub, catalog entries
3. `requirements-colab.txt` — Updated version pins
4. `.github/prompts/02-algorithm-x-gpu-validation.prompt.md` — Updated with verified baselines

### 2.7 Phase 1 Completed Data (DO NOT REGENERATE)
The following trace datasets exist and are the training data for Algorithm X:

| Family | Run Directory | Runs | Status |
|--------|---------------|------|--------|
| Qwen 0.5B | `research/outputs/real_traces_l4_qwen_0p5b` | Present | ✅ Legacy |
| DeepSeek 1.5B | `research/outputs/real_traces_l4_deepseek_1p5b` | 900 runs | ✅ Legacy |
| Mistral 7B | `research/outputs/real_traces_l4_mistral_7b` | Present | ✅ Legacy |
| Qwen 7B (4-bit) | `research/outputs/real_traces_l4_qwen_7b_4bit` | Present | ✅ Legacy |

Phase 1 universal feature analysis output is in `research/outputs/universal_feature_analysis/` including:
- `universal_hazard_model_metadata.json` — Selected model spec and AUC scores
- `universal_hazard_weights.csv` — Fitted logistic regression weights
- `lofo_family_metrics.csv` — Leave-One-Family-Out validation results

---

## 3. HARDWARE CONSTRAINTS (ABSOLUTE — DO NOT ATTEMPT TO CHANGE)

- **GPU**: NVIDIA L4, 22.0 GB VRAM
- **No other GPU available** — this is the only hardware
- **Colab runtime** — volatile, disconnects every 12-24 hours

### Memory Budget
| Model Size | Strategy | Expected VRAM | Feasibility |
|-----------|----------|---------------|-------------|
| ≤ 4B | BF16, no quantization | ~15 GB | ✅ Comfortable |
| 7-12B | BF16 | ~18 GB | ✅ Tight but works |
| 14-31B | 4-bit NF4 | ~17-18 GB | ✅ Verified (Gemma 31B) |
| 35B MoE | GPTQ Int4 | ~24 GB + disk offload | ⚠️ Requires disk offload, may be very slow |
| 35B+ dense | Any | >22 GB | ❌ Does not fit |

---

## 4. EXECUTION PROTOCOL — REVISED FOR FEASIBLE FRONTIER SET

### 4.1 Environment Setup
On each new Colab session, before any model work:

```bash
# 1. Clone and enter repo
cd /content
git clone https://github.com/bhattadiCS/research-thesis-overthinking-boundary.git || true
cd research-thesis-overthinking-boundary

# 2. Configure git for checkpointing
git config user.email "aditya.bhatt7@gmail.com"
git config user.name "Autonomous Researcher"
git pull origin main --ff-only

# 3. Install dependencies
pip install -q -r requirements-colab.txt

# 4. Verify environment
python3 -c "
import torch, transformers, bitsandbytes, accelerate
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), 'GB')
print('CUDA:', torch.version.cuda)
print('BF16:', torch.cuda.is_bf16_supported())
print('Flash SDP:', torch.backends.cuda.flash_sdp_enabled())
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('bitsandbytes:', bitsandbytes.__version__)
print('accelerate:', accelerate.__version__)
"
```

If any package is missing or CUDA is unavailable, **STOP and fix the environment**.

### 4.2 Mandatory Runtime Flags
ALL experiment runs MUST use these flags:

```
--attn-implementation sdpa       # Native FA2 via SDPA (NOT flash_attention_2)
--skip-install                   # Skip redundant pip checks
--skip-simulator                # Skip synthetic traces
--io-threads 4                  # Async hidden state saving
```

### 4.3 Log Indicators to Verify
Every run MUST show these log lines (confirms optimization stack is active):
1. `[INFO] Precision auto-tuning: bfloat16 detected for <N>B model.`
2. `[INFO] Attention implementation: sdpa (native FA2 kernels via SDPA).`
3. `[INFO] IOManager: Background saving started for ...`
4. `[INFO] batch_metrics: hidden_state_write_seconds` — value < 0.01

If any is missing, **STOP** and debug before proceeding.

---

## 5. AUTONOMOUS RESEARCH TASKS

### TASK 1: Catalog Correction and Additional Model Discovery
**Objective**: Fix the stale MODEL_CATALOG entries and discover additional open-source models that fit the L4.

1. Add `qwen_3p5_9b` entry pointing to `Qwen/Qwen3.5-9B`
2. Search HuggingFace for other feasible ≤12B open-source instruct models released in 2025-2026:
   - Check: Phi-4, Mistral-Small, InternLM3, Yi-2, Gemma 2, etc.
   - For each candidate: verify `AutoConfig.from_pretrained()` works, estimate VRAM, check chat template
3. Add valid entries to MODEL_CATALOG with correct HF names
4. **Checkpoint**: `git add . && git commit -m "[ALGO-X] Catalog: add verified frontier model entries" && git push origin main`

### TASK 2: Full GSM8K Trace Collection on Feasible Frontier Models
**Objective**: Run the 300-task GSM8K validation at temperatures [0.1, 0.6, 1.0] for each model.

**Priority Order** (run in this sequence):
1. **`gemma_4_e4b_it`** (Edge 4B, BF16) — Already smoke-tested ✅
2. **`qwen_3p5_9b`** (9B, BF16) — Already smoke-tested ✅
3. **`gemma_4_31b_it`** (31B, 4-bit NF4) — Already smoke-tested ✅
4. **`llama_3p1_8b_instruct`** (8B, BF16) — Already in catalog, proven architecture
5. **Any additional models** discovered in Task 1

**Command Template** (adapt per model):
```bash
python tools/run_colab_experiment.py \
  --model <MODEL_KEY> \
  --quantization <none|4bit> \
  --full-batch-size <1|4> \
  --output-dir research/outputs/real_traces_colab_<MODEL_KEY> \
  --io-threads 4 \
  --attn-implementation sdpa \
  --skip-install --skip-simulator
```

**Per-model settings**:
| Model | --quantization | --full-batch-size | Expected TPS |
|-------|---------------|-------------------|-------------|
| `gemma_4_e4b_it` | none | 4 | ~7.7 |
| `qwen_3p5_9b` | none | 2 | ~5-6 |
| `gemma_4_31b_it` | 4bit | 1 | ~3.7 |
| `llama_3p1_8b_instruct` | none | 2 | ~5-6 |

**Output Convention**: Each model gets its own directory:
- `research/outputs/real_traces_colab_gemma_4_e4b_it`
- `research/outputs/real_traces_colab_qwen_3p5_9b`
- `research/outputs/real_traces_colab_gemma_4_31b_it`
- `research/outputs/real_traces_colab_llama_3p1_8b_instruct`

### MANDATORY CHECKPOINTING PROTOCOL
**Every 25 tasks completed**, you MUST run:
```bash
git add .
git commit -m "[ALGO-X] Checkpoint: Model=<MODEL_KEY> Tasks=<N>/300 Temp=<T>"
git push origin main
```

**Between models**, commit with:
```bash
git add .
git commit -m "[ALGO-X] Complete: <MODEL_KEY> full trace (300 tasks × 3 temps)"
git push origin main
```

**After any code change**, commit immediately:
```bash
git add .
git commit -m "[ALGO-X] Fix: <brief description>"
git push origin main
```

### TASK 3: NPZ Integrity Validation
**After each model's full run**, validate all hidden states:

```python
import numpy as np
from pathlib import Path

model_key = "<MODEL_KEY>"
root = Path(f"research/outputs/real_traces_colab_{model_key}/hidden_states")
assert root.exists(), f"Hidden states directory missing for {model_key}"

valid_count = 0
total_count = 0
l2_shifts = []
for f in sorted(root.glob("*.npz")):
    total_count += 1
    with np.load(f) as data:
        hs = data['hidden_states']
    assert hs.ndim == 2, f"Expected 2D, got {hs.ndim}D in {f.name}"
    assert not np.isnan(hs).any(), f"NaN detected in {f.name}"
    assert not np.isinf(hs).any(), f"Inf detected in {f.name}"
    l2 = np.linalg.norm(hs[-1] - hs[0])
    assert l2 > 0, f"Zero L2 shift in {f.name}"
    l2_shifts.append(l2)
    valid_count += 1

print(f"{model_key}: {valid_count}/{total_count} valid")
print(f"  L2 shift: mean={np.mean(l2_shifts):.2f}, std={np.std(l2_shifts):.2f}")
print(f"  L2 range: [{np.min(l2_shifts):.2f}, {np.max(l2_shifts):.2f}]")
```

### TASK 4: Qwen 35B GPTQ Fallback (STRETCH GOAL — Try If Time Permits)
**Context**: The prior session got close to loading `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` on the L4 using a config patch + disk offload. The blocker was an interrupted session, not a confirmed failure.

**Approach**:
1. Install GPTQ runtime: `pip install -q optimum gptqmodel`
2. Patch the cached config to set `backend: torch` (avoids Triton kernel crash on `out_features=1` layers)
3. Enable disk offload: Add `offload_folder="/content/offload_cache"` to `from_pretrained()` kwargs
4. Load with `device_map="auto"` and `quantization="none"` (model is pre-quantized)
5. Run a 1-task smoke first, then scale if it works

**If this fails or is too slow (< 1 tok/s), abandon it** and focus on the 4 feasible models.

### TASK 5: Zero-Shot Frontier Synthesis and Validation
**Objective**: Apply the Phase 1 `quadratic_top4` regressor to all new frontier traces.

```bash
python research/frontier_validation_report.py \
  --run-dirs \
    research/outputs/real_traces_colab_gemma_4_e4b_it \
    research/outputs/real_traces_colab_qwen_3p5_9b \
    research/outputs/real_traces_colab_gemma_4_31b_it \
    research/outputs/real_traces_colab_llama_3p1_8b_instruct \
  --report-path research/reports/frontier_validation_report.md
```

**Success Criteria**:
- Zero-shot AUC for $\hat{q}_t$ > 0.70 on each frontier family
- Accuracy-per-1k-tokens efficiency gain > 25% vs `never_stop` baseline
- Parse success rate > 95%
- All hidden states pass integrity checks

### TASK 6: Mathematical Equation Analysis and Improvement
**This is the CORE RESEARCH task. Spend significant time here.**

**Objective**: Critically evaluate whether the current Algorithm X equation is optimal and propose improvements.

#### 6.1 Current Equation Audit
Run the existing universal feature analysis with the expanded model set:
```bash
python research/universal_feature_analysis.py --random-state 7
```

Examine the output `universal_hazard_weights.csv` and ask:
- Are the quadratic interaction terms actually contributing signal or just adding noise?
- Does the $\hat{\mu}_t$ curve cross zero cleanly for ALL families, or are there pathological cases?
- Is the LOFO generalization gap acceptable (< 0.05)?

#### 6.2 Alternative Equation Hypotheses to Test
Create a new script `research/equation_analysis.py` that systematically tests:

1. **Linear vs Quadratic vs Cubic basis** — Is degree=2 the sweet spot or is linear sufficient?
2. **Feature ablation** — For each of the 4 base features, remove it and measure AUC degradation
3. **Alternative feature subsets** — Try all $\binom{6}{4}$ = 15 combinations of the 6 candidate features
4. **Non-logistic approaches**:
   - Random Forest hazard estimators
   - Gradient Boosted Trees (LightGBM/XGBoost if available)
   - Kernel SVM for $\hat{q}_t$
5. **Direct drift estimation** — Instead of composing $\hat{\mu}_t$ from 3 separate models, train a single regressor to predict $\Delta V_t$ directly
6. **Hidden state geometry** — Compute PCA on hidden state trajectories, test whether principal component velocities predict $\mu_t$
7. **Information-theoretic formulation** — Test whether KL divergence between consecutive hidden states correlates with drift better than L2 shift

#### 6.3 Statistical Rigor
For each alternative, compute:
- **LOFO AUC** across all families (legacy + frontier)
- **Generalization gap** (train AUC - test AUC)
- **Calibration** — Are predicted probabilities well-calibrated?
- **Boundary detection accuracy** — Does the method stop at the oracle step ± 1?

#### 6.4 Output
Write results to `research/reports/equation_analysis_report.md` with:
- Comparison table of all equation variants
- The recommended equation (with mathematical notation)
- Evidence for why it's better than the current `quadratic_top4`

**Checkpoint**: `git add . && git commit -m "[ALGO-X] Equation analysis: complete" && git push origin main`

### TASK 7: Final Synthesis Report
Compile `research/reports/phase2_final_report.md` with:
1. **Executive Summary**: Which frontier models validated? What's the universal AUC?
2. **Mathematical Validation**: Does Theorem 1 hold empirically for the expanded model set?
3. **Equation Recommendation**: The best-performing stopping equation with full coefficients
4. **Drift Crossing Evidence**: Per-model $\mu_t$ curves with clearly marked zero-crossings
5. **Observable Rankings**: Which features matter most for predicting overthinking?
6. **Limitations**: What models couldn't be tested and why
7. **Thesis Implications**: How these results support the Universal Law of Overthinking

---

## 6. KNOWN ISSUES & WORKAROUNDS

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `Gemma4Processor` has no `pad_token` | Multimodal processor wraps inner tokenizer | Extract `processor.tokenizer` for text encoding |
| `torchvision.transforms.v2` import error | `_vendor/torchvision_stub` shadows real package | Stub now conditional on `importlib.util.find_spec("torchvision")` |
| `hidden_dir` NameError in `run_batch_traces` | Missing function parameter | Added as explicit parameter with caller passthrough |
| `transformers 5.0` lacks Gemma 4 | Architecture added in 5.5+ | Upgrade to `transformers >= 5.5.0` |
| `flash-attn` standalone not needed | PyTorch 2.10 includes native FA2 via SDPA | Use `--attn-implementation sdpa` not `flash_attention_2` |
| `qwen_3p5_7b_instruct` doesn't exist | Stale repo ID | Use `Qwen/Qwen3.5-9B` instead |
| `llama_4_8b_it` doesn't exist | Stale repo ID; Llama 4 = Scout MoE only | Use `meta-llama/Llama-3.1-8B-Instruct` or explore other models |
| Qwen 35B NF4 OOM | 71.9 GB model too large for L4 even in 4-bit | Try GPTQ Int4 (~24 GB) with disk offload, or skip |
| GPTQ `out_features=1` crash | Triton kernel doesn't support this shape | Patch config to use `backend: torch` |

---

## 7. FILE MAP (KEY CODE LOCATIONS)

| File | Purpose |
|------|---------|
| `tools/run_colab_experiment.py` | CLI wrapper — orchestrates smoke + full runs |
| `research/real_trace_experiments.py` | **THE HEART** — model loading, inference, trace collection |
| `research/trace_analysis.py` | Post-hoc analysis: $\alpha_t$, $\beta_t$, $\mu_t$ calculation |
| `research/universal_feature_analysis.py` | LOFO validation, model selection, weight export |
| `research/frontier_validation_report.py` | Zero-shot evaluation on frontier traces |
| `research/reports/gpu_optimization_audit.md` | L4 optimization verification |
| `research/outputs/universal_feature_analysis/` | Phase 1 trained models and metadata |
| `copilot_agent_directive.md` | Original 7-day directive (Phase 1 reference) |
| `requirements-colab.txt` | Colab dependency pins |

---

## 8. OOM RECOVERY TABLE

| Error | Recovery Action |
|-------|----------------|
| OOM on model load | Lower batch size to 1; try 4-bit quantization; try `device_map="auto"` |
| OOM during generation | Reduce `--full-max-new-tokens`; call `torch.cuda.empty_cache()` before each trace |
| OOM on 31B+ with 4-bit | Already at minimum — skip model and document as hardware-infeasible |
| Disk full on Colab | Delete old smoke outputs: `rm -rf research/outputs/real_traces_colab_smoke_*` |
| Disconnected mid-run | Re-clone, `git pull`, use `--resume` flag (runner checks for existing completed traces) |

---

## 9. AGENT BEHAVIORAL RULES

1. **NO PAUSING FOR PERMISSION**: Do not ask "Should I proceed?" — state "Task N complete. Starting Task N+1." and continue.
2. **CHECKPOINT RELIGIOUSLY**: Every 25 tasks, every code change, every model completion → `git add . && git commit && git push`.
3. **VERBOSITY**: Use `logging.INFO` and `tqdm` so stdout shows constant progress.
4. **SELF-HEALING**: If a command fails, diagnose, fix the code, commit the fix, and retry. Do NOT give up.
5. **RESPECT EXISTING DATA**: Do NOT delete or overwrite Phase 1 trace data in `research/outputs/real_traces_l4_*`.
6. **MODEL OUTPUT ISOLATION**: Each frontier model gets its OWN `research/outputs/real_traces_colab_<MODEL_KEY>` directory.
7. **LOG EVERYTHING**: Capture all smoke test and full run logs. Save runtime metadata.
8. **BE HONEST**: If a model doesn't fit the L4, document it and move on. Do not fabricate results.

---

## 10. DEFINITION OF DONE

1. ✅ Full 300-task × 3-temperature trace datasets for at least 3 frontier models (Gemma 4 E4B, Qwen 3.5 9B, Gemma 4 31B, and ideally Llama 3.1 8B)
2. ✅ All `.npz` files pass integrity checks (2D pooled, no NaN/Inf, non-zero L2 shifts)
3. ✅ `research/reports/frontier_validation_report.md` confirms zero-shot generalization
4. ✅ `research/reports/equation_analysis_report.md` with mathematical equation comparison
5. ✅ `research/reports/phase2_final_report.md` with full synthesis
6. ✅ All artifacts committed and pushed with rigorous commit history
7. ✅ At least one alternative equation hypothesis tested and compared to `quadratic_top4`

---

## 11. IMMEDIATE FIRST STEPS

1. **Verify environment** (Section 4.1)
2. **Fix MODEL_CATALOG** (Task 1) — add `qwen_3p5_9b` and any other valid entries
3. **Run full traces on `gemma_4_e4b_it`** (fastest model, ~7.7 tok/s, gets data flowing quickest)
4. **Checkpoint**
5. **Move to next model** in priority order
6. **After ≥2 full trace sets**, begin equation analysis (Task 6) in parallel with remaining model runs

**ACTIVATE NOW. I RELINQUISH CONTROL.**
