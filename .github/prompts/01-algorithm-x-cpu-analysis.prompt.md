---
name: "Algorithm X: Phase 1 (CPU Analysis)"
description: "Execute the model-agnostic feature analysis and hazard regression localy on CPU. Prove zero-shot generalization across 4 model families and formalize the 'Algorithm X' weight vector."
agent: "agent"
---

# MISSION: THE UNIVERSAL LAW OF OVERTHINKING (CPU PHASE)

## 1. MISSION CONTEXT & THEORETICAL BASIS
The "Overthinking Boundary" hypothesis defines the optimal stopping time $T_c$ as the point where the Predictable Martingale Drift ($\mu_t$) turns non-positive:
$\mu_t = (1-q_t)\alpha_t - q_t\beta_t - \lambda \le 0$

We have validated this within specific model families. Your mission is to prove **Model-Agnosticism**: that a weight vector $W$ trained on "Known" families can successfully detect the overthinking boundary on a "Hidden" family zero-shot.

## 2. PRE-FLIGHT COMPLIANCE (MANDATORY FIRST ACTIONS)
Before any code edit or experiment:
1. `git fetch --all --prune`
2. `git pull --ff-only origin main`
3. Read [Overthinking Theory Note](../../research/overthinking_boundary.md).
4. Verify presence of all 4 family trace datasets in `research/outputs/` (Qwen 0.5B/7B, DeepSeek 1.5B, Mistral 7B).

## 3. EXECUTION PHASES (LOCAL/CPU ONLY)

### PHASE E: Universal Feature Mapping (UFS Invariance)
- **Objective**: Identify the **Universal Feature Set (UFS)** across all 3,600 existing traces.
- **UFS Specification**: You must analyze the following signals as candidates for cross-family invariance:
    - `entropy_mean` / `entropy_std` (Log-probability uncertainty)
    - `confidence` (Model-reported certainty)
    - `hidden_l2_shift` (Latent-state "wandering")
    - `answer_changed` (Strategic revisions)
    - `thought_token_count` (Verbosity/Search depth)
- **Constraint**: Implement **Per-Model Z-Score Normalization**. Scale every feature relative to its own model's distribution before pooling files.
- **Metric**: Compute the **Feature Correlation Matrix** for `event_repair` ($\alpha$) vs. `event_corruption` ($\beta$).

### PHASE F: Zero-Shot Hazard Regression (LOFO Validation)
- **The Protocol**: Execute **Leave-One-Family-Out (LOFO)** cross-validation.
- **Training**: Fit two Logistic Regressors using `class_weight='balanced'` on 3 families:
    1. $\sigma(W_\alpha \cdot Z_t) \rightarrow \text{Repair Probability}$
    2. $\sigma(W_\beta \cdot Z_t) \rightarrow \text{Corruption Probability}$
- **Testing**: Evaluate AUC on the 4th "Hidden" model family.
- **Calibration Targets**:
    - **Mistral 7B (Hidden)**: Zero-shot Repair AUC should be $\approx 0.68$.
    - **Qwen 7B (Hidden)**: Zero-shot Corruption AUC should be $\approx 0.80$.

### PHASE G: Formalize 'Algorithm X'
- **Target Output**: `research/outputs/universal_feature_analysis/universal_hazard_weights.csv`.
- **Thesis Formalization**: You must update `research/overthinking_boundary.md` with a new section titled **"Algorithm X: The Universal Law of Overthinking"**. 
- **Required Component**: Use the following LaTeX template for the universal drift estimate:
    $$\hat{\mu}_t^{U} = (1 - \hat{q}_t) \cdot \text{logistic}(W_\alpha \cdot Z_t) - \hat{q}_t \cdot \text{logistic}(W_\beta \cdot Z_t) - \lambda$$
- **Required Component**: A **Generalization Gap Table** comparing Training AUC vs. Zero-Shot Test AUC for all 4 families.

## 4. SCIENTIFIC SUCCESS MATRIX
Your output is considered "Thesis-Grade" only if:
- **Zero-Shot Generalization**: Total average cross-family LOFO AUC for $\beta$ (corruption) is $> 0.70$.
- **Feature Invariance**: The **Generalization Gap** ($\Delta AUC$) for the Mistral 7B family is $< 0.05$.
- **Statistical Significance**: You include a $p$-value or confidence interval check for the most influential feature (expected: `entropy_mean`).

## 5. DIAGNOSTICS & TROUBLESHOOTING
- **Low Signal Warning**: If LOFO AUC drops below 0.5, check for **Feature Leakage**. Ensure `correct` or `run_id` were not accidentally included as features.
- **Incapable Regime**: Models with $q_t \approx 0$ consistently (Qwen 0.5B) will produce noisy weights. Prioritize the **Capable Group** (Mistral 7B, Qwen 7B) for the final $W$ vector derivation.
- **Scaling Errors**: If weights are unexpectedly large (>10), verify that **Z-score normalization** was applied separately to each model before pooling.

## 6. DEFINITION OF DONE
1.  Validated LOFO results stored in `research/outputs/universal_feature_analysis/`.
2.  Weighted vector $W$ exported as a standardized CSV for Phase 2 intake.
3.  `research/overthinking_boundary.md` contains a high-authority, LaTeX-formatted documentation of the Universal Law.
4.  Detailed autonomous log summarizing the "Zero-Shot" proof of concept.

**STOP**: Once Algorithm X is formalized, yield to the user for the Phase 2 GPU trigger.
