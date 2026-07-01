# Cross-Benchmark Results: The Overthinking Boundary is Difficulty-Conditioned

**Master's Thesis Research Note — Empirical Results (625.803–804)**
**Author:** Aditya Bhatt (M.S. in Applied and Computational Mathematics, Johns Hopkins University)
**Research Adviser:** Dr. Zerotti Woods
**Proposed Second Reader:** Dr. Moustapha Pemy

---

## 1. Purpose

This note reports the principal empirical result of the thesis: a 13-model × 4-benchmark
sweep that locates the **corrected overthinking boundary** $T^\* = \inf\{t \ge 2 : \mu_t \le 0\}$
for each (model, benchmark) pair, where the per-step drift
$\mu_t = (1 - q_t)\alpha_t - q_t\beta_t - \lambda$ balances the **repair hazard** $\alpha_t$
(incorrect $\to$ correct) against the **corruption hazard** $\beta_t$ (correct $\to$ incorrect)
net of a per-step cost $\lambda$. The headline finding is that the late boundary predicted by
the theory is **not a universal property of reasoning models** but emerges only inside a
**difficulty window** set by the interaction of task hardness and model capability.

## 2. Protocol

- **Models (13):** DeepSeek-R1-Distill 1.5B/7B; Qwen2.5-Instruct 0.5B/3B/7B/14B/32B;
  Mistral-7B-Instruct-v0.3; Phi-4-mini (4B); InternLM3-8B; Yi-1.5-9B; Mistral-Small-22B;
  Llama-3.1-8B-Instruct. All run at **full precision (bf16)** to remove the quantization
  confound identified earlier in the project.
- **Benchmarks (4):** GSM8K (grade-school arithmetic), MATH-500 (competition math),
  ARC-Challenge (multiple-choice science), GPQA (graduate-level multiple-choice).
- **Per cell:** 500 tasks × 3 temperatures = 1,500 reasoning traces; shared task split and
  shuffle seed (17) so task IDs align across all models.
- **Evaluation honesty (post-audit):** the correctness probe and detector comparison are
  evaluated **out-of-sample** via GroupKFold by `run_id`; the boundary operator is floored at
  $t \ge 2$ (the theory forbids a step-1 stop); $\alpha_t,\beta_t$ are fit on $t \ge 2$
  transitions; answer labels were re-graded from stored model outputs after a grader audit.
  Reported probe AUCs are therefore deflated relative to in-sample values and should be read
  as honest generalization estimates.

## 3. Cross-Benchmark Boundary Matrix

Each cell reports **corrected boundary step / peak accuracy / out-of-sample probe AUC**.
Boundary `2` denotes the floor — i.e. *no* late boundary (the model never sustains positive
repair drift, so stopping is optimal as early as the operator allows).

| Model | GSM8K | MATH | ARC | GPQA |
|---|---|---|---|---|
| DeepSeek-R1 1.5B | 2 / 0.36 / 0.70 | 2 / 0.05 / 0.61 | 2 / 0.29 / 0.56 | 2 / 0.23 / 0.53 |
| DeepSeek-R1 7B | 2 / 0.55 / 0.75 | 2 / 0.12 / 0.61 | 2 / 0.54 / 0.73 | 2 / 0.21 / 0.55 |
| Qwen2.5 0.5B | 2 / 0.08 / 0.58 | 2 / 0.06 / 0.63 | 2 / 0.40 / 0.53 | 2 / 0.20 / 0.55 |
| Qwen2.5 3B | **4** / 0.50 / 0.71 | 2 / 0.17 / 0.73 | 2 / 0.71 / 0.66 | 2 / 0.23 / 0.57 |
| Qwen2.5 7B | **5** / 0.71 / 0.75 | **5** / 0.39 / 0.82 | 2 / 0.90 / 0.64 | 2 / 0.33 / 0.53 |
| Qwen2.5 14B | **5** / 0.47 / 0.91 | 2 / 0.31 / 0.90 | 2 / 0.94 / 0.68 | 2 / 0.36 / 0.54 |
| Qwen2.5 32B | **5** / 0.88 / 0.83 | **6** / 0.59 / 0.82 | 2 / 0.96 / 0.69 | 2 / 0.37 / 0.54 |
| Mistral-7B | **3** / 0.32 / 0.65 | 2 / 0.07 / 0.71 | 2 / 0.72 / 0.66 | 2 / 0.29 / 0.52 |
| Phi-4-mini 4B | 2 / 0.34 / 0.65 | 2 / 0.14 / 0.74 | 2 / 0.71 / 0.75 | 2 / 0.29 / 0.55 |
| InternLM3 8B | **4** / 0.68 / 0.69 | 2 / 0.25 / 0.75 | 2 / 0.90 / 0.61 | 2 / 0.37 / 0.53 |
| Yi-1.5 9B | **3** / 0.38 / 0.69 | 2 / 0.16 / 0.70 | 2 / 0.82 / 0.68 | 2 / 0.30 / 0.52 |
| Mistral-Small 22B | **6** / 0.72 / 0.82 | 2 / 0.31 / 0.81 | **3** / 0.89 / 0.78 | 2 / 0.36 / 0.61 |
| Llama-3.1 8B | **5** / 0.47 / 0.74 | 2 / 0.21 / 0.75 | 2 / 0.77 / 0.75 | 2 / 0.34 / 0.57 |
| **Late boundaries** | **5 clear + 4 weak** | **2 clear** | **1 weak** | **0** |

## 4. Principal Finding: A Difficulty Window

The boundary's presence tracks neither model size nor benchmark alone, but the **realized
repair headroom** — how much accuracy a model can *recover* across steps. Define headroom
informally as `peak_accuracy − step1_accuracy`. A late boundary appears precisely when
headroom is both **large** (there is something to repair) and **realizable** (the model is
capable enough to repair it). The four benchmarks fall at different points of this window:

- **ARC — too easy (no headroom).** Capable models are already correct at step 1
  (Qwen-7B/14B/32B step-1 accuracy 0.87 / 0.93 / 0.96). With $q_1$ near the ceiling the
  repair *term* $(1-q_t)\alpha_t$ is small simply because $(1-q_t)$ is small — few traces
  remain to be repaired — while corruption is also tiny (Qwen-7B $\beta=0.005$). Net drift
  is non-positive from the start, so the boundary collapses to the floor even though the raw
  repair rate is nonzero. The lone exception, Mistral-Small (step 3), is the one capable
  model whose ARC step-1 accuracy leaves a little headroom.

- **GPQA — too hard (no realizable repair).** No model exceeds 0.37 peak accuracy; $q_t$
  stays low and flat, repair and corruption roughly cancel, and **no model of any size**
  shows a late boundary. The ceiling is below the level at which sustained repair can occur.

- **GSM8K — the sweet spot.** Capable models start near-zero at step 1 (Qwen-32B 0.04,
  Qwen-14B 0.03, Llama 0.01) yet climb to high peaks (0.47–0.88), realizing large repair
  headroom over many steps. Repair dominates corruption for several steps
  (Qwen-7B: $\alpha=0.121$ vs $\beta=0.063$), pushing the boundary out to step 5–6. The
  result is **cross-family**: 5 clear (Qwen-7B/14B/32B, Mistral-Small, Llama, across 3
  families) and 4 weaker (Qwen-3B, Mistral-7B, InternLM3, Yi).

- **MATH — sweet spot only for the strongest.** The headroom exists but is realizable only
  by the two most capable Qwen models (7B step 5, 32B step 6); weaker models cannot climb
  far enough for repair to dominate, so their boundaries sit at the floor. Late-boundary
  evidence is therefore confined to a single family — cross-family robustness is *not*
  established on MATH.

> [!IMPORTANT]
> **Empirical claim.** The late overthinking boundary is a *difficulty-conditioned*
> phenomenon, not an intrinsic property of chain-of-thought. It requires a task that is
> hard enough that the model is wrong at step 1 (excluding ARC for capable models) yet
> tractable enough that iterative reasoning can recover the answer (excluding GPQA and
> under-capacity models). This is exactly the regime in which the repair hazard $\alpha_t$
> can sustain above the corruption hazard $\beta_t$, which is the mechanism the drift
> equation $\mu_t = (1-q_t)\alpha_t - q_t\beta_t - \lambda$ predicts.

## 5. The Capability Ladder (Within-Family Control)

Holding family, precision, and benchmark fixed (Qwen2.5 on GSM8K) isolates capability as
the driver: the boundary moves monotonically later as model size grows.

| Qwen2.5 | Step-1 acc | Peak acc | Corrected boundary | Probe AUC |
|---|---|---|---|---|
| 0.5B | 0.07 | 0.08 | 2 (floor) | 0.58 |
| 3B | 0.05 | 0.50 | 4 | 0.71 |
| 7B | 0.27 | 0.71 | 5 | 0.75 |
| 14B | 0.03 | 0.47 | 5 | 0.91 |
| 32B | 0.04 | 0.88 | 5 | 0.83 |

The 0.5B model has essentially no repair headroom (peak 0.08) and sits at the floor; from
3B upward the boundary is late and the hidden-state correctness probe becomes informative
(AUC 0.71 → 0.91). *Caveat:* the 14B peak (0.47) dips below 7B (0.71), and step-1 accuracies
for 3B/14B/32B are near zero — both warrant a check for a step-1 answer-format artifact.

## 6. Limitations and Honest Caveats

1. **Probe AUCs are modest** (0.53–0.91, mostly 0.65–0.75 out-of-sample). The hidden-state
   correctness probe is informative for capable models on GSM8K/MATH but near chance on
   ARC/GPQA — consistent with the difficulty-window account, not a strong universal detector.
2. **The boundary is a population quantity** (a crossing of the pooled drift curve), not a
   per-trace certificate; the anytime-valid stopping rules (empirical-Bernstein, mixture
   e-process) remain comparatively conservative and are reported as upper-bound detectors.
3. **Data-integrity guards.** A handful of traces (14–41 rows out of ~13k–21k per affected
   cell) were dropped at analysis time after hard-kill SIGKILLs field-shifted append-only
   CSV rows; whole runs were dropped to preserve contiguous step sequences. This is <0.3%
   of data and does not affect any qualitative conclusion.
4. **MCQ benchmarks (ARC/GPQA)** use 4-way multiple choice; the low GPQA ceiling may partly
   reflect format rather than pure reasoning difficulty, though the flat $q_t$ curves argue
   against a late boundary regardless.

## 7. Takeaway for the Thesis

The optimal-stopping framework is **validated where its assumptions hold and falsified where
they do not**, in a way the theory itself explains. The drift equation predicts a late
boundary only when $\alpha_t > \beta_t$ is sustainable; the four benchmarks supply a natural
gradient of that condition, and the observed boundaries track it. This converts a single
"overthinking exists" observation into a **mechanistic, difficulty-conditioned law**, which
is the stronger and more defensible contribution.

*Source data:* `research/outputs/experiment_matrix/_aggregate/{gsm8k,math,arc,gpqa}/CROSS_FAMILY_REPORT.md`
(13/13 models each, regenerated 2026-06-27).
