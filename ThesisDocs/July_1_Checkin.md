# Thesis Progress Guide: Overthinking Boundary in Reasoning LLMs
*Meeting Date: July 1, 2026*

This guide is a self-explanatory walkthrough of our research. It is structured to help explain the project's core concepts, terminology, data, and performance results to anyone from scratch.

---

## 📌 Part 1: The Core Problem (What is "LLM Overthinking"?)

When modern AI models (like Qwen 2.5 or DeepSeek-R1) solve complex math or logic problems, they use **Chain-of-Thought (CoT)**. Instead of outputting the final answer immediately, they generate intermediate reasoning steps (thinking tokens).

While thinking longer generally helps, there is a turning point where continuing to think becomes harmful or wasteful. We call this the **Overthinking Boundary**:
1.  **Corruption**: The model derives the correct answer early in its thinking, but keeps generating tokens. It over-complicates its own logic, hallucinates a detail, and edits its correct answer to a wrong one.
2.  **Token Waste**: The model gets stuck in a recursive loop or redundant thinking trace. It cannot find the correct answer, but continues generating tokens, wasting time and money.

**Our Goal**: Create a mathematical "stopping rule" that monitors the model's internal thinking process in real-time and halts generation at the exact moment the model is most likely to have a correct, cost-efficient answer.

---

## 📐 Part 2: What is a "Step" and What Does the Math Mean?

To understand our data, we first need to define our terms:

*   **What is a "Step"?**
    *   As the model writes its thoughts, we extract its temporary "candidate answer" at every newline or paragraph boundary. 
    *   **Step 1** is the model's first draft of an answer.
    *   **Step 10** means the model has written 10 segments of thinking and had 10 opportunities to revise its answer.
*   **What is a "Late-Boundary"?**
    *   If a model has a "late-boundary at Step 5," it means the model's accuracy increases as it thinks up to Step 5, but after Step 5, the model begins to overthink (accuracy decays or token costs dominate).
*   **The Utility Value ($V_t$)**:
    *   We evaluate the model's final response at step $t$ using a utility score:
        $$V_t = q_t(v + c) - c - \lambda t$$
    *   $q_t$ (Belief of Correctness): The probability (0.0 to 1.0) that the model's candidate answer at step $t$ is correct.
    *   $v$ (Correct Reward): The positive value we get for a correct final answer (set to $+1.0$).
    *   $c$ (Incorrect Penalty): The penalty/cost of giving a wrong answer (set to $0.0$).
    *   $\lambda$ (Step Cost): The cost of generating tokens. Every reasoning step costs a small penalty $\lambda$ (e.g., $0.01$).
*   **Predictable Drift ($\mu_t$)**:
    *   The expected change in utility if we generate one more step:
        $$\mu_t = \left[ (1 - q_t)\alpha_t - q_t\beta_t \right] (v + c) - \lambda$$
    *   **$\alpha_t$ (Repair Hazard)**: The probability that a model corrects a wrong answer on the next step.
    *   **$\beta_t$ (Corruption Hazard)**: The probability that a model ruins a correct answer on the next step.
*   **Optimal Stopping Rule**:
    *   We want to stop reasoning the exact moment the marginal gain of continuing ($\mu_t$) drops to zero or below:
        $$T^* = \inf \{ t \ge 0 : \mu_t \le 0 \}$$
    *   *(Note: In practice, we floor stopping at Step 2 to allow the model at least one step of reasoning).*

---

## 📊 Part 3: What Data Did We Collect & Why?

We ran **900 independent problem-solving trials** per model-dataset combination. For every single step $t$ of the model's reasoning chain, we recorded the following features:

*   **Token Entropy**: 
    *   *What it means*: The uncertainty of the model's token selections. 
    *   *Why it's useful*: If the model is confused, token entropy is high. When it finds a solid logical path, entropy drops.
*   **Log-probability Variance**:
    *   *What it means*: How spread out the model's confidence is over possible next tokens.
    *   *Why it's useful*: High variance indicates the model is highly confident in specific words.
*   **Answer Stability (`answer_changed`)**:
    *   *What it means*: A flag showing if the model changed its candidate answer from the previous step.
    *   *Why it's useful*: If the answer stops changing, the model has converged. If it keeps changing, the model is lost.
*   **Self-Reported Confidence**:
    *   *What it means*: The model's own numeric estimate of how sure it is (e.g., "Confidence: 0.8").

### Why this data is useful:
We cannot observe the model's true correctness belief ($q_t$) or hazards ($\alpha_t, \beta_t$) in-flight. Instead, we use this collected data to train a machine-learning model (a stopping detector). This detector looks at the active token entropy and answer stability in real-time, estimates $\mu_t$, and halts the model.

---

## 🧪 Part 4: How Did We Perform? (Reading the Results)

We ran this pipeline on **13 different models** across **4 benchmarks** (52 combinations, or "cells"):

*   **Task-Dependency**:
    *   On simple multiple-choice tasks (ARC, GPQA), reasoning does not help. The models perform best at Step 2 (stop immediately to save tokens).
    *   On math word problems (GSM8K), reasoning helps significantly before overthinking sets in. Models peak around Step 5 to 9, showing a clear late-boundary.

### Detector Performance on Late-Boundary Tasks
Here is how our stopping rules performed compared to a perfect "Oracle" and the baseline of "Never Stop":

| Stopping Policy | Mean Stop Step | Mean Stop Utility | Mean Oracle Gap | % Oracle Utility Captured |
| :--- | :---: | :---: | :---: | :---: |
| **Oracle** | 3.05 | 0.5563 | 0.0000 | 100.0% |
| **Hazard Drift (Ours)** | **4.34** | **0.3360** | **0.2203** | **50.1%** |
| **Answer Stability** | 4.65 | 0.3148 | 0.2415 | 46.2% |
| **First Answer** | 2.00 | 0.1432 | 0.4131 | 8.0% |
| **Never Stop** | 10.50 | 0.1077 | 0.4486 | 0.0% |

#### Explaining the Table Values:
1.  **Never Stop (Baseline)**: If we never stop the model, it runs to the maximum limit (Step 10+). It achieves a low utility of **0.1077** because it wastes massive tokens and sometimes corrupts its own answers.
2.  **Oracle (Theoretical Limit)**: A perfect Oracle that stops the model at the absolute best step for every single run achieves a utility of **0.5563**. The maximum possible utility gain is $0.5563 - 0.1077 = 0.4486$.
3.  **First Answer**: If we stop the model immediately at Step 2, we avoid wasting tokens, but we lose the massive accuracy gains of thinking. It achieves a utility of **0.1432** (capturing only **8%** of the potential gains).
4.  **Hazard Drift (Our Detector)**: By predicting when the predictable drift $\mu_t \le 0$ dynamically, our detector stops the model at an average of **Step 4.34**. It achieves a utility of **0.3360**, which bridges **50.1%** of the gap between Never Stop and the perfect Oracle.
5.  **Answer Stability (Heuristic)**: Stopping when the candidate answer hasn't changed for 2 steps is a solid heuristic (capturing **46.2%** of gains), but it is rigid and cannot adapt to different token costs ($\lambda$) or stakes.

---

## 🛠️ Part 5: Codebase Remediation (What We Cleaned Up)

*   **Grader Fixing**: The model's final answers were being graded incorrectly by regex bugs (e.g., scoring "a third" as a wrong answer even if the math was correct). We rewrote the grading logic, recovering hundreds of correct runs.
*   **Leakage Elimination**: We implemented **GroupKFold cross-validation**. This ensures that the detectors are trained and tested on separate subsets of math problems, proving they truly generalize out-of-sample.
*   **Fair Baselines**: We enforced a minimum Step 2 floor on both our detectors and the Oracle, making sure we compare them on equal footing.

---

## 🚀 Part 6: What We Plan to Do Next (Our Roadmap)

To wrap up the thesis and prepare for publication, we have outlined the following next steps:

1.  **Stakes Sweeping (Evaluating Safety-Critical Domains)**:
    *   *What we will do*: Expose and sweep the correct reward $v$ vs. incorrect penalty $c$ (e.g., $c = 10, v = 1$ representing a safety-critical setting like medical or legal Q&A).
    *   *Why*: Verify if our stopping equations correctly force aggressive early stopping when incorrect answers carry massive penalties.
2.  **In-Flight Inference Deployment (Active Stopping)**:
    *   *What we will do*: Build a real-time stopper that hooks into the generation loop (using vLLM or Hugging Face hooks) to evaluate the observable vector $\mathbf{x}_t$ dynamically.
    *   *Why*: Transition from offline analysis of logs to actively halting token generation, saving actual GPU runtime and token costs.
3.  **Prompted vs. Distilled Reasoning Analysis**:
    *   *What we will do*: Study why distilled reasoning models (like DeepSeek-R1 Distill 7B) exhibit early-boundary profiles (halting at Step 2 on GSM8K) compared to prompted instruct models (like Qwen 2.5 7B, which halts at Step 5).
    *   *Why*: Determine if distilled models have their stopping rules "pre-baked" during training, or if their internal hazards ($\alpha_t, \beta_t$) behave differently.
4.  **New Benchmark Domains**:
    *   *What we will do*: Evaluate stopping behavior on other reasoning domains (such as coding benchmarks or logical puzzles).
    *   *Why*: Verify if the late-boundary and corruption hazard behaviors generalize outside mathematical word problems.
