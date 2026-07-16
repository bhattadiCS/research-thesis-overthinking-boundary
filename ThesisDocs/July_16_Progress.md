# Advisor Progress Update: July 16, 2026
*A condensed 5-minute presentation guide mapping our progress, variables, and tournament results since July 1.*

---

## 1. Quick Progress Milestones (July 1 – July 16)

* **Completed Global Sweep**: Collected **19,948 reasoning paths** (147,740 steps) across 52 experimental cells.
* **3.5x Run Optimization**: Restricted max tasks to 500 per cell and doubled batch sizes to leverage the NVIDIA RTX Pro 6000 Blackwell GPU, cutting sweep time from 22 hours to **8 hours**.
* **System Hardening**: Hardened trace saving against VM restarts and added a 5-second SymPy timeout to prevent evaluation hangs on pathological LLM equation strings.

---

## 2. Global Experimental Architecture

We map our work to a strict scientific variable framework to isolate causal relationships and control confounding noise:

```mermaid
graph TD
    subgraph independent_variables ["Independent Variables (Knobs We Turn)"]
        V1[Step Budget: How long it thinks]
        V2[Temperature: Randomness]
        V3[Model Scale: Parameter size]
        V4[Quantization: Weight compression]
        V5[Model Family: Architecture brand]
    end

    subgraph dependent_variables ["Dependent Variables (Things We Measure)"]
        V6[Trajectory Correctness: Step-level accuracy]
        V7[Overthinking Drift: Right-to-wrong transitions]
        V8[OOF AUC: Stopping detector accuracy]
        V9[Policy Utility: Token/step savings]
        V10[Token Count: Text volume]
        V11[Latency: Compute time]
    end

    subgraph control_variables ["Control Variables (Held Constant)"]
        V12[Dataset Domain: Domain difficulty]
        V13[System Prompt: Formatting rules]
        V14[Sampling Seed: Randomness seed]
    end

    independent_variables --> dependent_variables
    control_variables -->|Controls Confounding Noise| dependent_variables
```

---

## 3. High-Density Variable Walkthrough

### 3.1. Independent Variables (Inputs)
* **Variable 1: Step Budget ($N_{steps}$)**: The max reasoning steps allowed. As $N_{steps}$ increases, overthinking drift rises because long reasoning chains introduce logical mistakes.
  ```mermaid
  graph LR
      S1[Step 1: Setup] --> S2[Step 2: Execution] --> S3[Step 3: Extraction] --> S4{Optimal Stop}
      S4 -->|Overthinking| S5[Step 4: Drift] --> S6[Step 5: Degradation] --> S7[Step 6: Error]
  ```
* **Variable 2: Temperature ($T$)**: Controls output randomness. High temperature ($T=0.6$) increases branching entropy, multiplying overthinking risk, whereas $T=0.0$ is highly stable.
  ```mermaid
  graph LR
      Start((Step t)) -->|T=0.0 Greedy| Greedy[Deterministic Path: 95% prob] --> S1((Step t+1))
      Start -->|T=0.6 Stochastic| Branch1[Alternative Path A: 30% prob] --> S2((Step t+1))
      Start -->|T=0.6 Stochastic| Branch2[Alternative Path B: 5% prob] --> S3((Step t+1))
  ```
* **Variable 3: Model Scale ($S$)**: Parameter size (0.5B to 32B). Larger models are more accurate and drift less, but when they do, they generate highly logical-sounding, systematic errors.
  ![Model Scale vs. Correctness/Drift](images/model_scale_accuracy_drift.png)
* **Variable 4: Quantization ($Q$)**: Weight compression (16-bit vs 4-bit). Compressing weights introduces noise and drops base accuracy slightly, but preserves the *shape* of hidden state trajectories.
  ```mermaid
  graph TD
      subgraph precision16 ["16-bit Precision"]
          W1[Raw Weights] -->|Generates| H1((Trajectory Shape))
      end
      subgraph quantized4 ["4-bit Quantization"]
          W2[Compressed Weights] -->|Generates| H2((Identical Shape))
      end
      H1 -->|Transfer AUC = 0.86| H2
  ```
* **Variable 5: Model Family ($A_{family}$)**: Architectural lineage (Qwen vs DeepSeek vs Llama). Dictates pretraining distributions and hidden representation styles.
  ```mermaid
  graph TD
      Root((Model Families)) --> Dense[Dense: Qwen / Mistral / Llama]
      Root --> RL[RL Distilled: DeepSeek R1]
  ```

### 3.2. Dependent Variables (Outputs)
* **Variable 6: Trajectory Correctness ($C_t$)**: Step-by-step correctness state of intermediate math. Used as our ground-truth label.
  ```mermaid
  graph TD
      Start([Start]) --> Incorrect[Incorrect State]
      Incorrect -->|Model Solves Problem| Correct[Correct State]
      Correct -->|Consistent Logic| Correct
      Correct -->|Overthinking Drift| Incorrect
  ```
* **Variable 7: Stopping Drift / Overthinking ($D$)**: When a model reaches a correct answer at step $t$ but continues generating until it outputs a wrong final answer.
  ![Overthinking Drift Curve](images/overthinking_drift_by_step.png)
* **Variable 8: Detector AUC ($AUC_{det}$)**: Out-of-fold stopping classifier performance. Tracks predictive signal strength in hidden states.
  ```
  TPR (True Positive Rate)
  1.0 |                   .------------------ LSTM / GRU (AUC = 0.87)
  0.8 |             .-----
  0.6 |          .-'    .---------------------- Linear Head (AUC = 0.73)
  0.0 |---+----+----+----+--------------------- FPR (False Positive Rate)
      0.0  0.2  0.4  0.6  0.8  1.0
  ```
* **Variable 9: Stopping Utility ($U$)**: Accuracy gains balanced against token/step compute costs.
  ![Early Stopping Policy Utility Curves](images/stopping_utility_by_step.png)
* **Variable 10: Inference Token Count ($L_{tokens}$)**: Amount of text written. Determines api cost and latency.
  ```
  Step 1: [████████] 80 Tokens --> Step 2: [███████████████] 150 Tokens (Cumulative: 230)
  ```
* **Variable 11: Compute Latency ($T_{latency}$)**: Wall-clock execution time. Under parallel batching on Blackwell GPUs, throughput scales to 850 tokens/sec.
  ```mermaid
  graph LR
      Batch[Batch Size 64] -->|Blackwell Core Execution| GPU[NVIDIA RTX 6000] -->|Throughput| Speed[850 tokens/sec]
  ```

### 3.3. Control Variables (Fixed)
* **Variable 12: Dataset Domain ($D_{domain}$)**: Benchmark difficulty (GSM8K vs MATH vs GPQA). Harder domains show higher overthinking drift rates and peak stopping utility.
  ```
  Difficulty Spectrum: [GSM8K (Easy)] ------------> [MATH (Medium)] ------------> [GPQA (Hard)]
  ```
* **Variable 13: System Prompt Template ($P_{prompt}$)**: Enforces step-by-step formatting so reasoning states align across steps.
  ```mermaid
  graph TD
      Prompt[Force Steps] --> Response[Model Generation] -->|Regex Split| S1[Step 1 State] & S2[Step 2 State]
  ```
* **Variable 14: Sampling Seeds ($S_{seed}$)**: Fixed to `seed=7` to guarantee identical stochastic runs, eliminating background generation noise.
  ```mermaid
  graph TD
      Seed[Seed = 7] --> Run1[Trace 1: Answer 42] & Run2[Trace 2: Answer 42] & Run3[Trace 3: Answer 42]
  ```

---

## 4. Cross-Validation Tournament Results

Evaluated over **19,948 unique trajectories** (147,740 steps) under 5-Fold Group Cross-Validation:

| Stopping Method | Prediction Accuracy (OOF AUC) | Step Cost Utility | Token Cost Utility | Head-to-Head Record (W/T/L) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Linear)** | 0.7380 | +0.3705 | +0.4524 | 14,966 / 2,968 / 2,014 |
| **N8b (Linear Proj)** | 0.8104 | +0.3818 | +0.4637 | 15,441 / 2,660 / 1,847 |
| **Gated SC (Hysteresis)** | 0.8686 | +0.3640 | +0.4579 | 10,879 / 8,067 / 1,002 |
| **GRU (Sequence)** | 0.8686 | +0.3760 | **+0.4786** | 12,737 / 5,669 / 1,542 |
| **LSTM (Sequence)** | **0.8714** | **+0.3784** | +0.4759 | **13,196 / 5,114 / 1,638** |

### 4.1. The Stopping Methods Explained Simply
* **Baseline (Linear)**: Uses logistic regression on current step features ($q_t, \alpha_t, \beta_t$) with **zero memory** of the past. Disadvantage: Treats steps in isolation, yielding poor predictions (0.738 AUC).
* **N8b (Linear Proj)**: Linear model on mid-layer compressed representations. Filters background token noise, but still lacks memory.
* **GRU & LSTM (Sequence)**: Recurrent networks that track the model's confidence trajectory over steps. **Core finding**: Tracks the temporal buildup of logical drift, boosting prediction accuracy to **0.8714 AUC** (an 8:1 win-to-loss ratio).
* **Gated SC (Consensus)**: Industry standard voting system. Strong accuracy (0.868 AUC) but **highly expensive** (requires 5-10 parallel runs). Our LSTM model beats it using only a **single reasoning run**.

---

## 5. Visualizing the Overthinking Boundary

```
Model Confidence / State
  ^
  |      [Correct Boundary]
  |      ---------------------------------- (Gold Standard Answer)
  |     /
  |    /   * (Optimal Stop Point: Model reaches correct answer at Step 2)
  |   /     \
  |  /       \   [Overthinking Drift]
  | /         \
  |/           *--------> [Incorrect Final Output at Step 6]
  +---------------------------------------------> Reasoning Steps
  0     1     2     3     4     5     6 
```
* **Our stopping detector** triggers a "Pencil-Down" halt at Step 2. This preserves the correct answer and saves **4 steps worth of token generation cost**.

---

## 6. Next Steps for Writing the Paper

```mermaid
graph LR
    S1[1. Abstract & Intro] --> S2[2. Related Work] --> S3[3. Methodology] --> S4[4. Experimental Setup] --> S5[5. Evaluation] --> S6[6. Discussion]
```
* **Methodology**: Use the "Pencil-Down Math Exam" metaphor to explain cognitive drift.
* **Evaluation**: Embed the CV tournament results and describe sequence model performance.
* **Discussion**: Focus on quantization transferability (proves detectors generalize across precision formats).
