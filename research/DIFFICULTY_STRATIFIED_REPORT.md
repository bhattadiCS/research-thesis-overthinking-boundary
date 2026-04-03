# Difficulty-Stratified Boundary Analysis

## Overview

The aggregate cross-family analysis grouped 300 diverse GSM8K problems into a single dataset. Because problem difficulty interacts with a model's intrinsic capabilities, aggregating across all problems confounds the capability-gated boundary determination. Phase A of the deep-dive analysis addresses this by classifying each problem into three difficulty strata and analyzing the resulting continuation value ($\mu_t$) trajectories independently.

## Methodology

Each of the 300 common GSM8K problems was classified into a difficulty stratum per model based on its performance across 3 distinct generation instances (T=0.1, 0.6, 1.0; seed=7).
  * **Easy:** The model solved the problem at step 1 in the majority of instances ($\geq 50\%$ solve rate).
  * **Medium:** The model failed to solve it reliably at step 1 ($< 50\%$ rate) but reached a correct answer at some point during the available steps.
  * **Hard:** The model never reached the correct answer in any instance.

After stratification, the base drift formulation was applied to derive the per-stratum conditional transition rates ($\alpha_t$ and $\beta_t$) and the drift computation $\mu_t = (1-q_t)\alpha_t - q_t\beta_t - \lambda$.

Two boundary steps were extracted:
* **$T_c^{first}$**: The chronological first step where $\mu_t \leq 0$.
* **$T_c^{late}$**: The final usable step immediately following the last positive-drift window (last positive-to-negative crossing).

## Results: Stratum Difficulty Distributions

As expected, models with higher peak capabilities have fundamentally different difficulty distributions:

| Model | Easy | Medium | Hard | Max Acc |
|---|---|---|---|---|
| Qwen 0.5B | 11 | 51 | 238 | 0.09 |
| DeepSeek 1.5B | 60 | 194 | 46 | 0.69 |
| Mistral 7B | 82 | 136 | 82 | 0.46 |
| Qwen 7B | 97 | 194 | 9 | 0.83 |

*Note: For Qwen 7B, nearly 65% of the dataset is "Medium", meaning it represents a highly repairable task subset. For Qwen 0.5B, 80% is simply "Hard" (unrepairable).*

## Results: The Boundary is Stratum-Dependent

The "Medium" stratum perfectly highlights the overthinking boundary mechanisms, capturing problems where repair outpaces corruption over the first few tokens.

**Medium Stratum Analysis:**
* **Qwen 7B** (194 problems):
  * **$\alpha/\beta$ ratio:** 0.918
  * **$T_c^{late}$ Boundary:** Step 6
  * **Acc gain from Step 1 to Peak:** +60.3pp (15% $\rightarrow$ 75%)
* **Mistral 7B** (136 problems):
  * **$\alpha/\beta$ ratio:** 0.435
  * **$T_c^{late}$ Boundary:** Step 3
  * **Acc gain from Step 1 to Peak:** +14.7pp (18% $\rightarrow$ 33%)

In both models, the `Medium` bounded tasks experienced extended continuation utility, but Mistral lacked the depth of task-aligned capability to sustain the repair rate, leading to an earlier boundary and substantially lower peak accuracy.

## Conclusion

Aggregating all problems effectively suppresses late-boundary signals for models positioned at a capability threshold (like Mistral 7B). By resolving traces by baseline difficulty, we expose that models retain task-specific active repair windows. Late reasoning boundaries explicitly require a deep, highly-populated "medium" difficulty stratum, where $\alpha \approx \beta$.
