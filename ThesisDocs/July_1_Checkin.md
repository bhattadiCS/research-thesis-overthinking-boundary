# Thesis Progress Guide: Overthinking Boundary in Reasoning LLMs
*Meeting Date: July 1, 2026*

This guide is designed to catch my advisor (Dr. Woods) up to speed on the progress, empirical findings, and resolved codebase issues from the last 3-4 weeks.

---

## 📌 Executive Summary (What to tell the Professor first)
*   **The Big Picture**: We are investigating **overthinking** in reasoning models—where letting a model think longer actually causes it to drift into wrong answers (corruption) or loop pointlessly (stagnation).
*   **Massive Scale-up**: In the last 3-4 weeks, we scaled our experiments from a single-model pilot to a **52-cell matrix** (13 models across 4 benchmarks: GSM8K, MATH, ARC, GPQA).
*   **Code Clean-up**: We completed a comprehensive audit of the code. We fixed critical bugs in the grading logic (which were causing correct answers to be marked wrong) and eliminated data leakage in our machine-learning stopping detectors.
*   **Core Finding**: Overthinking is highly **task-dependent**. On math word problems (GSM8K), models benefit from reasoning steps up to a peak, and then degrade. On shallow reasoning/multiple-choice tasks (ARC, GPQA), additional reasoning is useless, and the model should stop immediately.

---

## 📐 The Mathematical Framework (The Stopping Rule)

To determine the optimal stopping point $T^*$, we model the expected utility of the model's response at step $t$:
$$V(q_t) = q_t \cdot v - (1 - q_t) \cdot c = q_t(v + c) - c$$
where:
*   $q_t \in [0, 1]$ is the model's current belief that its intermediate answer is correct.
*   $v > 0$ is the positive utility of a correct final answer.
*   $c > 0$ is the penalty (negative utility) of an incorrect final answer.

Including the per-step computation cost $\lambda > 0$, the value process is:
$$V_t = q_t(v + c) - c - \lambda t$$

The **predictable drift** (expected change in value by taking one more step) is:
$$\mu_t = \mathbb{E}[V_{t+1} - V_t \mid \mathcal{F}_t] = \left[ (1 - q_t)\alpha_t - q_t\beta_t \right] (v + c) - \lambda$$
where we define:
*   **$\alpha_t$ (Repair Hazard)**: The probability that a wrong answer transitions to a correct answer at step $t+1$.
*   **$\beta_t$ (Corruption Hazard)**: The probability that a correct answer degrades into a wrong answer at step $t+1$.

### The Optimal Stopping Rule
Applying One-Step Look-Ahead (OSLA) theory, the optimal stopping time $T^*$ is when the expected marginal value of continuing reasoning turns non-positive:
$$T^* = \inf \{ t \ge 0 : \mu_t \le 0 \}$$

> [!NOTE]
> **Empirical Floor**: In our practical experiments, we force the model to take at least one reasoning step (so the minimum stopping step is floored at $t \ge 2$).

---

## 🧪 Empirical Matrix Results (The 52-Cell Sweep)

We ran 13 models on 4 datasets (52 cells total) to verify if our theoretical boundary matches real-world runs:

1.  **GSM8K (The Reasoning Sweet-Spot)**: 9 out of 13 models show a clear **late-boundary** ($T^* > 2$). They need time to think, peak in accuracy, and then decay. For example, Qwen 2.5 7B peaks in accuracy at Step 9, and the optimal boundary is Step 5.
2.  **ARC & GPQA (Shallow Reasoning)**: Almost all models cross the boundary at Step 2 (the floor). In multiple-choice questions, the model either knows the answer or it doesn't; extra steps do not "repair" mistakes ($\alpha_t \approx 0$).
3.  **Model Capability Dictates the Boundary**:
    *   **Weak Models (e.g., Qwen 0.5B)**: Stop at Step 2. They lack the capability to repair their errors.
    *   **Stronger Models (e.g., Qwen 32B, Mistral 24B)**: Have late boundaries (Step 5–6) on math, because they can systematically correct intermediate errors.

### Stopping Detector Performance (On the 12 Late-Boundary Cells)
We compared different deployable stopping detectors against the theoretical **Oracle** (which knows exactly when to stop for maximum utility) and the do-nothing baseline (**Never Stop**):

*   **Hazard Drift (Our Learned Detector)**: **Beats all other deployable detectors**, capturing **50.1%** of the potential oracle utility gain.
*   **Answer Stability (Heuristic)**: Captures **46.2%** of the gap. It works well but lacks theoretical safety guarantees.
*   **E-Process & Empirical Bernstein**: These anytime-valid statistical bounds are highly conservative. Empirical Bernstein is too strict (requires $N > 1000$ samples to fire), so it behaves similarly to "Never Stop".

---

## 🛠️ Main Codebase Fixes (Remediating the Audit)

I completed a deep audit of the codebase to make sure our results are statistically sound:

1.  **Fixed Silently Corrupted Graders**: The GSM8K grader had regex issues (e.g., picking up words like "third" as fractions instead of the final integer), and the MATH parser had latex equation bugs. This was corrupting our correctness labels ($C_t$). I rewrote them to be highly robust.
2.  **Eliminated Evaluation Leakage**: Previously, the regression model for the detectors and the thresholds were fit on the same data they were evaluated on. I refactored the analysis to run **GroupKFold cross-validation** (grouped by run/task) to ensure honest out-of-sample testing.
3.  **Consistent Oracle Baseline**: Floored the Oracle at $t \ge 2$ to match the detectors, removing a structural advantage that was skewing our utility gap plots.

---

## 💬 Next Steps / Key Questions for the Meeting

I want to discuss these three main directions with Dr. Woods:

1.  **Stakes Sweep**: Expose the stakes parameter $(v + c)$ in our inference experiments. If we increase the penalty for incorrect answers ($c \gg v$), how aggressively does the boundary move earlier?
2.  **In-Flight Stopping**: Transition our offline analysis to a real-time, in-flight stopping system that terminates reasoning tokens dynamically during generation.
3.  **Prompted vs. Distilled Reasoning**: DeepSeek-R1 (distilled reasoning) behaves differently than standard instruct models (Qwen 2.5). Even the 7B DeepSeek model stops early (Step 2) on GSM8K compared to Qwen 7B (Step 5). Is this because distilled models have their stopping rules "baked-in" during training?
