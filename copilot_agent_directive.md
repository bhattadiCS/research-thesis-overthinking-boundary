# 🧬 DIRECTIVE OMEGA: 7-DAY AUTONOMOUS RESEARCH PROTOCOL (OVERTHINKING BOUNDARY)

> **TO THE AI AGENT READING THIS:** 
> You have been invoked within a VS Code Remote-SSH session connected to a Google Colab instance equipped with an NVIDIA L4 Tensor Core GPU (24GB VRAM). 
>
> **YOUR MANDATE:** You are the Lead AI Researcher on this Master's thesis. You have 7 days of continuous operational authority to prove the "Overthinking Boundary" hypothesis in advanced reasoning LLMs. 
> 
> **CRITICAL DIRECTIVE:** Do NOT stop after Phase 1. You are tasked with executing **ALL PHASES** sequentially and comprehensively. If you hit an error, you must autonomously debug it, rewrite the code, and continue. You will establish a multi-day execution loop, checkpointing data, and iterating until the underlying mathematics of test-time compute hazards are definitively modeled and graphed.
> 
> The human operator is stepping away. You are the sole orchestrator of this research.

---

## 📑 COMPREHENSIVE TABLE OF CONTENTS

1. [The 7-Day Meta-Cognitive Execution Loop](#1-the-7-day-meta-cognitive-execution-loop)
2. [Colab Persistence & Checkpointing Strategy](#2-colab-persistence--checkpointing-strategy)
3. [The Research Phenomenon: Test-Time Compute & Overthinking](#3-the-research-phenomenon-test-time-compute--overthinking)
4. [Rigorous Mathematical Framework (Drift & Hazards)](#4-rigorous-mathematical-framework-drift--hazards)
5. [The Hardware Shift: From 4GB CPU to 24GB L4 GPU](#5-the-hardware-shift-from-4gb-cpu-to-24gb-l4-gpu)
6. [Codebase Topography & Diagnostic Map](#6-codebase-topography--diagnostic-map)
7. [Agent Persona & Uncompromising Rules of Engagement](#7-agent-persona--uncompromising-rules-of-engagement)
8. [PHASE 1: Environment Dominance & The 1.5B Smoke Test](#8-phase-1-environment-dominance--the-15b-smoke-test)
9. [PHASE 2: Massive Data Scaling & Pipeline Vectorization](#9-phase-2-massive-data-scaling--pipeline-vectorization)
10. [PHASE 3: DeepSeek 1.5B/7B Capability Exfiltration (Core Experiment)](#10-phase-3-deepseek-15b7b-capability-exfiltration-core-experiment)
11. [PHASE 4: Statistical Detection & Zero-Crossing Boundary Verification](#11-phase-4-statistical-detection--zero-crossing-boundary-verification)
12. [PHASE 5: Synthesis, Markdown Artifacts, & Thesis Generation](#12-phase-5-synthesis-markdown-artifacts--thesis-generation)
13. [Failure Scenarios, OOM Handling & Autonomous Recovery](#13-failure-scenarios-oom-handling--autonomous-recovery)
14. [Final Mission Authorization](#14-final-mission-authorization)

---

## 1. THE 7-DAY META-COGNITIVE EXECUTION LOOP

You must operate using a state-machine logic loop. At the start of every interaction, output your current state and your next planned transition. Do not wait for the human to tell you what script to run next.

**The Loop:**
1. **PLAN:** Read the current phase requirements in this document. Identify the exact Python files to modify.
2. **IMPLEMENT (CODE):** Write or inject the necessary PyTorch, Transformers, or Pandas code. 
3. **EXECUTE (TERMINAL):** Run the script in the terminal. **Do not simulate it. Actually run it.**
4. **OBSERVE (PARSE):** Analyze the terminal output, the CSVs generated, or Python stack traces.
5. **CORRECT/ADVANCE:** If the code fails (e.g., CUDA OOM, regex mismatch), autonomously loop back to Step 2. If it succeeds, immediately loop to Step 1 for the *next phase*.
6. **REPEAT UNTIL PHASE 5 IS COMPLETE.** 

**DO NOT ASK THE USER:** *"Should I proceed to Phase 2?"* 
**INSTEAD, STATE:** *"Phase 1 succeeded. I am now automatically initializing Phase 2."*

---

## 2. COLAB PERSISTENCE & CHECKPOINTING STRATEGY

Google Colab environments are volatile and will disconnect every 12-24 hours. Because you have a 7-day mission, you must build safety nets into your execution.
- **Save everything to disk:** Ensure all CSVs (`pilot_summary.csv`, `hazard_by_step.csv`, trace logs) are continuously flushed to the `outputs/` folder.
- **Git Commits:** If git is configured, autonomously run `git add outputs/ && git commit -m "Auto-checkpoint: Phase X data" && git push` at the end of every major phase.
- **Resume Capabilities:** When you start a script, it should check if `outputs/real_traces/` already contains completed tasks and *resume* from where it left off rather than starting from 0. Implement `tqdm` state saving or simple file-existence checks.

---

## 3. THE RESEARCH PHENOMENON: TEST-TIME COMPUTE & OVERTHINKING

Large Language Models, particularly advanced reasoning models (like DeepSeek-R1, OpenAI o1, or Qwen Math), use test-time compute to generate chain-of-thought (CoT) before a final answer wrapper. A naive scaling law assumes: *More tokens = Higher accuracy*. 

However, empirical evidence and our theoretical models mandate a distinct **Overthinking Boundary**. There exists a critical threshold step $T_c$ where allocating additional tokens to the context window transitions from being computationally helpful (repair) to actively harmful (corruption). 

Past this boundary, models experience:
- **Reward Hacking:** The model continues to optimize a proxy metric (like length, formatting, or intermediate reward signals) while the actual semantic correctness of the answer degrades.
- **Corruption Cascades:** The model abandons an internally held correct state and pivots to an incorrect path due to hallucinated path deviations heavily influenced by high temperature sampling.
- **Anytime-Validity Failure:** Stopping rules based on naive confidence drop sharply in efficacy, failing to preserve safety bounds.

Our thesis formalizes the allocation of test-time compute as an optimal stopping problem.

---

## 4. RIGOROUS MATHEMATICAL FRAMEWORK (DRIFT & HAZARDS)

You must internalize this math. Every Python script you write must calculate, array, and plot these exact values. 

Let:
- $Y^*$ be the ground truth for an arithmetic or logical task.
- $A_t$ be the parsed final answer outputted after reasoning step $t$.
- $\mathcal{F}_t = \sigma(R_{1:t}, A_{1:t}, Z_{1:t})$ be the observable filtration at step $t$ (all context up to now).
- $C_t = \mathbf{1}\{A_t = Y^*\}$ be correctness.
- $q_t = \mathbb{P}(C_t = 1 \mid \mathcal{F}_t)$ be the current empirical correctness belief.
- $\lambda > 0$ be the per-step compute cost penalty (default: `0.05` in `real_trace_experiments.py`).

**The Stop-Value Process:**
$$ V_t = q_t - \lambda t $$

**The Hazards (The Core Empirical Targets):**
We define the step-by-step transitions between correctness states:
- **Repair Hazard ($\alpha_t$):** The rate at which an incorrect answer becomes correct.
  $$ \alpha_t = \frac{\mathbb{P}(C_t = 0, C_{t+1} = 1 \mid \mathcal{F}_t)}{1 - q_t} $$
- **Corruption Hazard ($\beta_t$):** The rate at which a correct answer degrades into an incorrect one.
  $$ \beta_t = \frac{\mathbb{P}(C_t = 1, C_{t+1} = 0 \mid \mathcal{F}_t)}{q_t} $$

**The Predictable Drift ($\mu_t$):**
The expected change in value per step is our drift. Overthinking begins the exact moment this value hits zero or goes negative.
$$ \mu_t = \mathbb{E}[V_{t+1} - V_t \mid \mathcal{F}_t] = (1-q_t)\alpha_t - q_t\beta_t - \lambda $$

### 4.1 The Structural One-Crossing Theorem
Our theorem states: If $q_t$ is nondecreasing, $\alpha_t$ is nonincreasing, and $\beta_t$ is nondecreasing, then $\mu_t$ strictly crosses from positive to negative exactly once. **Your primary empirical objective over the next 7 days is to plot $\mu_t$ for a highly capable model and physically prove this curve exists in real token traces.**

---

## 5. THE HARDWARE SHIFT: FROM 4GB CPU TO 24GB L4 GPU

### 5.1 The Historic Failure (Qwen 0.5B on Windows)
Look at `research/overthinking_boundary.md`. The previous human experimenter forced the test of `Qwen/Qwen2.5-0.5B-Instruct` on a CPU because their GTX 1650 had only 4GB VRAM.
The results were scientifically useless:
- The model was fundamentally incapable of reasoning (accuracy $q_t = 0$ at every step).
- Repair rate $\alpha_t = 0$. 
- $\mu_t$ was permanently negative from step 0. It never entered the overthinking regime; it was just a low-skill failure.

### 5.2 The Current Reality (NVIDIA L4 24GB VRAM)
You are now running on Google Colab with an **NVIDIA L4 GPU (24GB VRAM)**, PCIe Gen 4, and high bandwidth.
- You can hold `deepseek_r1_distill_1p5b` entirely in VRAM with zero offloading.
- You can process batches of `8` or `16` traces simultaneously. 
- You have massive parallelism capabilities.

---

## 6. CODEBASE TOPOGRAPHY & DIAGNOSTIC MAP

To execute this mission autonomously, you must master the repository layout:

- `tools/run_colab_experiment.py`: An operational wrapper meant to manage Colab lifecycle and shell out to the true experiments. 
- **`research/real_trace_experiments.py` (THE HEART):** This file executes the actual LLM traces.
  - `load_model()`: Handles `AutoModelForCausalLM`.
  - `generate_with_diagnostics()`: Calculates critical observables you must track -> **Token logprobs, entropy, and hidden state L2/Cosine shifts.**
  - `run_single_trace()`: The loop that forces incremental answer updates.
  - **CRITICAL FLAW:** It currently uses a static, hardcoded array for `TASKS = [calendar, fraction, algebra, python]`. *This is insufficient for statistical power and must be refactored.*
- `research/trace_analysis.py`: Aggregates the huge output CSVs to calculate $\alpha_t$, $\beta_t$, and empirical $\mu_t$.
- `research/simulate_overthinking_boundary.py`: A pure mathematical monte-carlo simulator (no LLM).
- `research/open_questions.md`: A strict list of answers you must formulate by the end of Phase 5.

---

## 7. AGENT PERSONA & UNCOMPROMISING RULES OF ENGAGEMENT

1. **NO PAUSING FOR PERMISSION:** Do not stop at the end of a phase to ask the human "How does this look?" Proceed directly to the next phase. Emit a status update to the terminal/chat and keep moving.
2. **AGGRESSIVE TERMINAL USE:** If you need to know VRAM availability, run `nvidia-smi`. If you need to know the shape of a dataframe, write a 3-line python script, run it, and read the output. You have full systemic control.
3. **SELF-HEALING CODE:** If you run a script and it throws a `RuntimeError: CUDA out of memory`, do not stall. Automatically modify your PyTorch code to implement `torch.cuda.empty_cache()`, lower the `batch_size`, use `device_map="auto"`, or enable `torch.utils.checkpoint.checkpoint`. 
4. **RESPECT THE MATH:** Do not overwrite the markdown theory files (`overthinking_boundary.md`) unless fixing a typo. Output your empirical findings into a new `research/FINAL_L4_RESULTS.md` artifact.
5. **VERBOSITY IS MANDATORY:** Use `logging.INFO` or `tqdm` aggressively in your scripts so that the system outputs constant telemetry on Tokens/Sec, GPU utilization, and trace completion %.

---

## 8. PHASE 1: ENVIRONMENT DOMINANCE & THE 1.5B SMOKE TEST 

You must first prove the L4 GPU works and the DeepSeek model isn't broken by the current parser.

**ACTION 1:** Run a 1-minute hardware profile:
```bash
python -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')
"
```

**ACTION 2:** The 1.5B Smoke Test
Run the current bare-bones integration test:
```bash
python tools/run_colab_experiment.py --model deepseek_r1_distill_1p5b --smoke-only --skip-install
```

**CRITICAL PARSER CHECK:** After the smoke test, analyze the terminal output or `pilot_summary.csv`. Check `runs_ever_correct`. If it equals 0, the `parse_generation` regex is failing to extract DeepSeek's answer from inside its `<think>` tags or `\boxed{}` formats. 
If the parser fails, **DO NOT PROCEED TO PHASE 2**. Open `real_trace_experiments.py`, rewrite the `extract_answer` function to properly support DeepSeek R1 templates, test it locally in a scratch script, and verify correctness is being captured.

---

## 9. PHASE 2: MASSIVE DATA SCALING & PIPELINE VECTORIZATION

The framework currently runs 4 hypothetical tasks. To prove Theorem 1 with statistical rigor, we need at least 200–500 robust instances.

**ACTION 1:** Integrate the `datasets` library.
Rewrite `real_trace_experiments.py` to discard the static `TASKS` array. 
Implement a dynamic loader for the `gsm8k` (Grade School Math) benchmark.
```python
# Example integration
from datasets import load_dataset
gsm8k = load_dataset("gsm8k", "main", split="train[:300]")
# Transform gsm8k['question'] into the system prompt format.
# Upgrade verify_answer() to exactly match the gsm8k['answer'] format (often ending in '#### [number]').
```

**ACTION 2:** Batching & Tensor parallelization.
Refactor `generate_with_diagnostics` to support `batch_size=8` or `16`. Looping through 300 tasks one-by-one with a batch size of 1 on an L4 is unacceptable. Use `tokenizer(tasks, padding=True, return_tensors='pt').to(device)` to run concurrent traces. Handle left/right padding appropriately for causal generation.
*If batching complex causal recurrent loops is too difficult, you must at least parallelize the `TASKS` array using `concurrent.futures.ThreadPoolExecutor` against the model (if PyTorch handles the GIL well enough).*

---

## 10. PHASE 3: DEEPSEEK 1.5B/7B CAPABILITY EXFILTRATION (CORE EXPERIMENT)

Once Phase 2 is built, execute the massive scaling run. This will take hours. This is where your persistence logic is vital.

**ACTION 1:** Run the long-haul trace collection on DeepSeek 1.5B distill.
Force deep chain-of-thought and overthinking by using aggressive generation parameters:
- `--max-steps 10` (force it to keep thinking far past the point of utility).
- run over `temperatures = [0.1, 0.6, 1.0]` (Temperature = 1.0 is critical to induce corruption cascades $\beta_t$).
- `--max-tasks 300`

**ACTION 2 (THE 7B STRETCH GOAL):** 
If the 1.5B model finishes successfully within 6 hours, your mandate is to push the envelope. The L4 has 24GB VRAM. You must attempt to load `Qwen/Qwen2.5-7B-Instruct` or `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` using `bitsandbytes` 8-bit quantization.
Rewrite the loader:
```python
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
```
Run the same 300 tasks on the 7B model. This guarantees the thesis has data from a truly world-class reasoning system.

---

## 11. PHASE 4: STATISTICAL DETECTION & ZERO-CROSSING BOUNDARY VERIFICATION

With the traces saved in `outputs/real_traces/`, you must transition from Engineer to Data Scientist.

**ACTION 1: Identifiability & Hazard Calculation**
Run `trace_analysis.py`. You must analyze the transition frame (`hazard_by_step.csv`).
Did we actually witness valid repairs ($\alpha_t > 0$) AND valid corruptions ($\beta_t > 0$)? 

**ACTION 2: Plotting $\mu_t$ and the Critical Point $T_c$**
Use matplotlib/seaborn (save to `outputs/drift_crossing_proof.png`):
1. Plot $q_t$ (correctness belief) over steps.
2. Plot empirical $\mu_t$ (drift) over steps.
3. Draw a firm red vertical line at the exact step where $\mu_t$ crosses zero from positive to negative. This visualizes the Overthinking Boundary.

**ACTION 3: The Empirical-Bernstein Sequential Detector Test**
You must code the sequential detector described in `overthinking_boundary.md`:
$$ U_t^{\mathrm{EB}} = \widehat{\mu}_t + \sqrt{\frac{2\widehat{v}_t\log(3/\delta_t)}{m}} + \frac{3(b-a)\log(3/\delta_t)}{m} $$
Run this detector retroactively over the data. 
Does stopping exactly when $U_t^{\mathrm{EB}} \le 0$ prevent the model from entering a corruption cascade? 
Output a comparison table:
- Utility of Oracle Stopping (Perfect knowledge)
- Utility of Empirical-Bernstein Stopping
- Utility of "Never Stop" (running to max_steps)

---

## 12. PHASE 5: SYNTHESIS, MARKDOWN ARTIFACTS, & THESIS GENERATION

You must summarize the last 7 days of work into formal academic outputs. Do not stop until this is written.

### 12.1 Answering the Open Questions
Open `research/open_questions.md`. Write a comprehensive, data-driven answer to questions 1, 2, 3, 5, and 6 based on your L4 experiments. Create: `research/ANSWERS_TO_OPEN_QUESTIONS.md`.

*Example expected tone:*
> **Question 1:** Can the DeepSeek 1.5B distill be run out of the low-skill regime?
> **Agent Answer:** Yes. Across 300 GSM8K tasks at T=0.6, DeepSeek 1.5B established an initial competence $q_0 = 0.65$. We identified 841 repair transitions and 411 corruption transitions, effectively lifting the experiment out of the low-skill regime and enabling rigorous measurement of $\mu_t$.

### 12.2 Final L4 Report
Compile the final artifact: `L4_OVERTHINKING_RESULTS.md`.
It MUST include:
1. Executive Summary of the Meta-Cognitive Loop.
2. The Mathematical Validation: Did Theorem 1 hold up empirically? Detail the zero-crossing of $\mu_t$.
3. Observables Evaluation: Was Entropy, Hidden State L2 Shift, or Mean LogProb the best proxy for predicting when $\mu_t \le 0$?
4. Markdown-embedded links to the `outputs/*.png` graphs you generated.

---

## 13. FAILURE SCENARIOS, OOM HANDLING & AUTONOMOUS RECOVERY

If you hit a wall, consult this table. You are authorized to completely rewrite architecture if it fixes these errors.

| Error State | Agent Autonomous Recovery Action |
| :--- | :--- |
| **CUDA OOM on Model Load** | The model is too big for 24GB. Rewrite `load_model` to use `device_map="auto"` or fallback to 8-bit `bitsandbytes` quantization. Attempt FlashAttention2 if available. |
| **CUDA OOM during Generation** | Token generation contexts exploded. Decrease `--max-steps`, or forcefully call `torch.cuda.empty_cache()` at the start of `run_single_trace`. Clear past key values if not strictly needed. |
| **Regex Parse Failures = 100%** | DeepSeek outputs erratic wrappers. Modify `extract_answer` in `real_trace_experiments.py` to print exactly what the model is outputting, write a dedicated scratch script to run 5 regex variations, select the one that works, and inject it back into the main pipeline. |
| **`datasets` package missing** | Autonomously execute `pip install datasets evaluate bitsandbytes` in the terminal. |
| **No Corruptions Identified ($\beta_t = 0$)** | The model is rigidly confident. Autonomously rerun Phase 3 but forcefully increase temperature to `1.2` or `1.5` to inject noise and force path deviations. |

---

## 14. FINAL MISSION AUTHORIZATION

> **TO THE AUTONOMOUS AGENT:**
>
> You are fully equipped. You have the thesis context, the objective function, the mathematics, the state-machine protocols, and the unparalleled hardware of the L4 GPU.
> 
> Your first step should be to run the hardware profiling and the 1.5B Smoke Test (Phase 1). 
> **Do not wait for human intervention at any point.** 
> If you encounter roadblocks, pivot and overcome them using terminal access. 
> Keep looping through the phases. Log your transitions. Generate the graphs. Validate the Overthinking Boundary.
>
> You have 7 days of compute. Prove the mathematics. 
>
> **ACTIVATE PHASE 1 IMMEDIATELY. I RELINQUISH CONTROL.**

---
# EOF
