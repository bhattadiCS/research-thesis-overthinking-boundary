# Thesis Progress Check-in: Overthinking Boundary in Reasoning LLMs
*Date: July 1, 2026*

This document outlines the theoretical, empirical, and engineering progress made on the thesis project over the past 3–4 weeks, highlighting key achievements, resolved challenges, and next steps for discussion with the thesis advisor.

---

## 1. Context & Research Objective

The project focuses on modeling and identifying the **overthinking boundary** in reasoning language models (LLMs) utilizing Chain-of-Thought (CoT) prompting. We model optimal stopping using sequential decision theory under competing hazard rates:
*   **$\alpha_t$ (Repair Hazard)**: The probability that the reasoning state transitions from incorrect to correct at step $t+1$.
*   **$\beta_t$ (Corruption Hazard)**: The probability that a correct reasoning state degrades to incorrect at step $t+1$.

The expected marginal utility $\mu_t$ of generating step $t+1$ is defined as:
$$\mu_t = \left[ (1 - q_t)\alpha_t - q_t\beta_t \right] (v + c) - \lambda$$
where $q_t$ is the belief of correctness at step $t$, $v$ is utility of a correct answer, $c$ is the cost/penalty of an incorrect answer, and $\lambda$ is the token cost. The optimal boundary $T^*$ is the point where expected marginal utility turns non-positive:
$$T^* = \inf \{ t \ge 2 : \mu_t \le 0 \}$$

---

## 2. Key Accomplishments (Past 3–4 Weeks)

### A. The 52-Cell Empirical Matrix
We scaled up from pilot testing to run a massive validation grid of **13 models** across **4 benchmarks** (GSM8K, MATH, ARC, GPQA), resulting in **52 independent experiment cells**.
*   **Models Evaluated**: Qwen 2.5 Instruct (0.5B, 3B, 7B, 14B, 32B), DeepSeek-R1 Distilled (1.5B, 7B), InternLM3 (8B), Llama 3.1 (8B), Yi 1.5 (9B), Phi 4 Mini (4B), Mistral 7B Instruct v0.3, and Mistral Small (24B).
*   **Key Finding**: Overthinking is task-dependent. In reasoning-rich tasks (GSM8K), 9 out of 13 models exhibit a clear late-boundary ($T > 2$), where accuracy peaks before decaying or plateauing. In shallow reasoning tasks (ARC, GPQA), the optimal stop step is at the floor ($T=2$), meaning extra steps only waste tokens.

### B. Core Stopping Detector Rankings
We benchmarked several stopping policies across the **12 late-boundary cells** (where the optimal stopping boundary is non-trivial). Our learned **Hazard Drift** detector outperformed all other deployable baselines, capturing half of the maximum possible oracle utility:

| Detector | Mean Oracle Gap | Mean Stop Step | Mean Stop Utility | Utility Gain vs. Never-Stop | % Oracle Utility Captured |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Oracle ($T \ge 2$)** | 0.0000 | 3.05 | 0.5563 | +0.4486 | 100.0% |
| *Cheater: Verifier (First Correct)* | 0.1271 | 5.82 | 0.4292 | +0.3215 | 71.7% |
| **Hazard Drift (Learned)** | **0.2203** | **4.34** | **0.3360** | **+0.2283** | **50.1%** |
| **Answer Stability (Heuristic)** | 0.2415 | 4.65 | 0.3148 | +0.2071 | 46.2% |
| **Entropy Plateau (Heuristic)** | 0.2603 | 4.34 | 0.2960 | +0.1883 | 41.7% |
| **E-Process (Anytime-Valid)** | 0.2672 | 6.23 | 0.2891 | +0.1814 | 40.9% |
| **Empirical Bernstein (Bound)** | 0.4024 | 9.50 | 0.1539 | +0.0462 | 10.3% |
| **First Answer ($t=2$ fallback)** | 0.4131 | 2.00 | 0.1432 | +0.0355 | 8.0% |
| **Never Stop (Do Nothing)** | 0.4486 | 10.50 | 0.1077 | 0.0000 | 0.0% |

### C. Remediating Codebase Defects (79-Agent Audit)
We conducted a deep correctness audit and resolved all critical and high findings:
1.  **Grader Corrections**: Fixed GSM8K numeric candidate extraction (handling fractions and singular ordinals) and LaTeX MATH parsing (handling RHS-only splits), ensuring the target correctness label $S_t$ is highly accurate.
2.  **Out-of-Sample Validation**: Refactored the analysis to run GroupKFold cross-validation (grouped by `run_id` and `task_id`). The stopping thresholds and `hazard_drift` regression are now evaluated honestly on unseen trajectories, proving they generalize.
3.  **Boundary & Oracle Alignment**: Implemented the correct mathematical definition for the boundary operator ($T^* = \inf \{ t \ge 2 : \mu_t \le 0 \}$) and enforced a matching $t \ge 2$ floor on the Oracle to guarantee equal comparison footing.

---

## 3. Engineering Challenges & Solutions

*   **VRAM Optimization**: Managed co-running models on single nodes by tightening VRAM footprint estimations and optimizing batch sizing to double decode throughput.
*   **Auto-Resume Orchestration**: Developed an orchestrator (`run_experiment_matrix.py`) to launch parallel runs on preemptible cloud RunAI pods. The pipeline gracefully handles pod preemptions via automatic checkpoints and restarts.

---

## 4. Next Steps for Discussion

1.  **Stakes Sweeping**: Explore model behavior under varying stakes. How does the optimal boundary shift in high-penalty scenarios ($c \gg v$)?
2.  **Active Stopping Implementation**: Move from offline trace analysis to runtime active stopping (inference-time halting using the learned `hazard_drift` threshold).
3.  **Cross-Family Architecture Profiles**: Investigate the difference between pre-trained/instruct models (like Qwen 2.5) vs. distilled reasoning models (like DeepSeek-R1 Distill) regarding their hazard curves.
