# Thesis Project Outline — Advisor Meeting
*A detailed but readable guide for talking through where we are and where we're going.*

---

## Part 1: Where We Are Now

### What's been built so far
| Area | Status | Details |
|------|--------|---------|
| Core math | ✅ Done | The drift equation ($\mu_t$), repair hazard ($\alpha_t$), corruption hazard ($\beta_t$), and per-step cost ($\lambda$) are all formalized and verified. |
| Dual-boundary theory | ✅ Done | We found that models show *two* stopping points — an early warning ($T_c^{first}$) and a late breaking point ($T_c^{late}$). This was a genuine finding, not expected. |
| Trace pipeline | ✅ Working | We can feed a model a math problem, force it to reason step-by-step, and record every intermediate answer, entropy score, and token probability along the way. |
| Initial model results | ✅ 4 models done | Qwen 0.5B, DeepSeek 1.5B, Qwen 7B, Mistral 7B — all tested on GSM8K. |
| Semantic vs. format errors | ✅ Proven | We confirmed that overthinking is a *logic* problem, not a formatting problem. Models actually break their own reasoning. |
| Basic predictor | ✅ Prototype | A 4-feature observable vector (token count, entropy MA, logprob variance, entropy drop) with baseline regressions. |

### What the initial results showed
- **Qwen 7B (the "competent" model):** Starts at 36% accuracy, peaks at 78% at Step 9, then *degrades*. The late boundary is Step 6 — meaning after Step 6, more thinking hurts.
- **Mistral 7B:** Same dual-boundary pattern confirmed across a completely different model family.
- **DeepSeek 1.5B & Qwen 0.5B:** These weaker models cross the boundary at Step 1 — they can't benefit from extended reasoning at all.
- **The takeaway:** Overthinking is real, measurable, and the boundary depends on both the model's capability and the difficulty of the question.

---

## Part 2: The Experimental Design

### Why these specific models?
We're not just "running everything we can find." Each model serves a specific scientific purpose:

| Model | Size | Purpose | Status |
|-------|------|---------|--------|
| **Tier 1: Already Done** ||||
| Qwen2.5-0.5B-Instruct | 0.5B | Minimal capability baseline (too weak to reason) | ✅ Done |
| DeepSeek-R1-Distill-1.5B | 1.5B | Weak-regime control (early boundary expected) | ✅ Done |
| Mistral-7B-Instruct | 7.3B | Cross-family validation (non-Qwen witness) | ✅ Done |
| Qwen2.5-7B-Instruct | 7.6B | Primary late-boundary witness (our best result) | ✅ Done |
| **Tier 2: Mid-Weight (16-32 GB)** ||||
| Mathstral-7B-v0.1 | 7.3B | Does math-specific training move the boundary? | ❌ Pending |
| NuminaMath-7B-CoT | 7.0B | AIMO winner — does optimized CoT change the picture? | ❌ Pending |
| DeepSeek-R1-Distill-7B | 7.6B | Distilled reasoning: does knowledge distillation help? | ❌ Pending |
| Llama-3.1-8B-Instruct | 8.0B | Standard non-reasoning baseline from Meta | ❌ Pending |
| DeepSeek-R1-Distill-Llama-8B | 8.0B | Same distilled reasoning, but on Llama architecture | ❌ Pending |
| Qwen-3.5-9B | 9.0B | Next-gen Qwen — has the architecture improved? | 💨 Smoke only |
| Gemma-2-9B-It | 9.2B | Alternating local/global attention (different architecture) | ❌ Pending |
| Gemma-4-E4B-It | 4.0B | Edge-optimized model — boundary in constrained hardware | 💨 Smoke only |
| Phi-4 | 14B | Dense textbook-trained model — does data quality matter? | ❌ Pending |
| DeepSeek-R1-Distill-14B | 14.7B | Mid-weight capability gate | ❌ Pending |
| **Tier 3: Frontier (40-80 GB)** ||||
| InternLM2-20B | 20B | Independent family (Shanghai AI Lab) | ❌ Pending |
| GPT-OSS-20B | 21B | OpenAI open model with configurable reasoning | ❌ Pending |
| Mistral-Small-3.1-24B | 24B | High-density reasoning from Mistral | ❌ Pending |
| Qwen3.6-27B | 27B | Novel hybrid attention architecture | ❌ Pending |
| Gemma-2-27B-It | 27.2B | Google's alternating attention at scale | ❌ Pending |
| Qwen3-30B-A3B (MoE) | 30.5B | Think/non-think toggle — can MoE self-regulate? | ❌ Pending |
| Gemma-4-31B-It | 31B | Advanced non-Qwen frontier | ❌ Pending |
| QwQ-32B | 32.5B | Primary frontier reasoning model | ❌ Pending |
| DeepSeek-R1-Distill-32B | 32.5B | Distilled twin of QwQ — same knowledge, different process | ❌ Pending |
| Yi-34B-Chat | 34B | Independent 01.AI family | ❌ Pending |
| Command-R-08-2024 | 35B | RAG/tool-use reasoning style (different paradigm) | ❌ Pending |
| Qwen-3.5-35B-MoE | 35B | MoE architectural test at scale | 💨 Smoke only |

### What datasets and why?
| Dataset | Domain | Why we need it |
|---------|--------|----------------|
| **GSM8K** (1,319 problems) | Grade-school math | ✅ Already done. Our baseline. Simple enough that most models can engage. |
| **MATH** (5,000 problems) | Competition math (AMC/AIME level) | Longer reasoning chains required. Tests if the boundary shifts when problems are genuinely hard. |
| **HumanEval** (164 problems) | Python code generation | Completely different modality. Tests if overthinking exists outside of math (i.e., can a model "overcode"?). |
| **MBPP** (974 problems) | Simpler Python tasks | Paired with HumanEval. Gives us an easy-vs-hard comparison in the code domain. |
| **ARC-Challenge** (1,172 problems) | Grade-school science | Multi-hop reasoning over factual knowledge. Tests generalization beyond pure math. |
| **GPQA** (448 problems) | Graduate-level science | Extremely hard. Tests if the boundary even exists when models are operating near their absolute limit. |

### How many traces per model?
- **Target:** 500-1,000 prompts per model per dataset, at 3 temperatures (0.1, 0.6, 1.0).
- **Each trace:** Records the answer, correctness, entropy, and logprobs at every single reasoning step (up to ~15 steps).
- **Total estimated traces:** ~50,000-100,000 individual reasoning trajectories across the full model suite.
- **Why this many:** We need enough statistical power to compute confident hazard rates per difficulty stratum. Fewer than ~500 per cell makes the $\alpha_t$/$\beta_t$ estimates noisy.

---

## Part 3: Compute Plan

### Where are the GPUs coming from?
Dr. Woods is providing GPU access for this project. No out-of-pocket compute costs.

| Tier | What it covers | Hardware needed |
|------|----------------|-----------------|
| Tier 2 (mid-weight) | Models up to ~14B parameters | 16-32 GB VRAM (e.g., L4 / A100 40GB) |
| Tier 3 (frontier) | Models 20B-35B parameters | 40-80 GB VRAM (e.g., A100 80GB) |

---

## Part 4: Risk & Contingency

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GPU access delayed or limited | Low | High | Dr. Woods is providing access. If a specific tier is temporarily unavailable, we reorder the schedule and run smaller models first. |
| A model produces garbage traces | Low | Low | We already have smoke tests. Drop the model and document why. |
| Timeline slips on trace collection | Medium | Medium | Weeks 4-8 have buffer. We can run models in parallel or drop the lowest-priority ones (Command-R, Yi-34B). |
| Math proofs have a flaw | Low | Very High | Week 1 is specifically dedicated to re-derivation. Advisor reviews the proofs before we build on them. |

### What's the minimum viable thesis?
If everything goes wrong with compute, the absolute floor is:
- **4 models already done** + **3-4 mid-weight models** on **GSM8K + MATH**
- The math, the dual-boundary finding, and the baseline predictor
- This is still a defensible thesis — just narrower in scope

### What's the dream version?
- All 25+ models across all 6 datasets
- A live stopping system that actively controls generation
- A discovered mathematical law (via symbolic regression) that predicts the boundary from model size alone

---

## Part 5: Success Criteria

| Phase | We succeeded if... | We failed if... |
|-------|-------------------|-----------------|
| Semester 1 | We have traces from 15+ models, the feature extraction pipeline works, and a baseline predictor achieves >0.7 AUROC for predicting when to stop. | We have fewer than 8 models or can't extract clean features. |
| Semester 2 | The advanced predictor beats all baselines (never-stop, fixed-length) on both accuracy and compute savings, and the thesis is defended. | The stopping rule performs no better than "just stop at step 5 every time." |

---

## Part 6: Check-in Plan

| Cadence | What happens |
|---------|-------------|
| **Weekly** | Short email/Slack update to Dr. Woods — what got done, what's blocked, what's next. |
| **Biweekly** | 30-min meeting to review results, discuss any math questions, and course-correct the schedule. |
| **End of each phase** | Formal status document with charts and metrics. |

---

## Part 7: The Week-by-Week Plan

### SEMESTER 1 — Collect data, build baselines

---

#### Week 1: Math Lockdown & Project Audit
| Task | Est. Time | Details |
|------|-----------|---------|
| Re-derive the $\mu_t$ drift equation from scratch | 3 hrs | Walk through every step to make sure the math is airtight before building on it. |
| Audit existing trace data for leakage | 5 hrs | Check that training data never leaks into test evaluations in the current codebase. |
| Re-run a small Qwen 0.5B sample | 2 hrs | Confirm outputs match previous runs exactly — this verifies nothing is broken. |
| Set up a clean Git branch for Phase 1 | 1 hr | Separate all new semester work from the preliminary experiments. |
| **→ Done when:** Math is locked in. Advisor signs off on the core equations. ||

---

#### Week 2: Literature Review
| Task | Est. Time | Details |
|------|-----------|---------|
| Gather 20-30 recent papers | 4 hrs | Focus on test-time compute scaling, hazard models, sequential stopping rules, and adaptive inference. |
| Deep-read the core 10 papers | 8 hrs | Annotate each one: what they did, what they missed, and how our approach differs. |
| Draft the literature synthesis | 6 hrs | Write the "gap statement" showing exactly what no one has done yet. |
| **→ Done when:** We can clearly articulate why this thesis is novel in 2 sentences. ||

---

#### Week 3: Pipeline Hardening & New Datasets
| Task | Est. Time | Details |
|------|-----------|---------|
| Refactor trace collection for large models | 6 hrs | Add batched inference, gradient checkpointing, and graceful OOM recovery. |
| Write dataset parsers for MATH, HumanEval, MBPP, ARC, GPQA | 5 hrs | Each dataset has its own format — we need clean input/output parsing for all of them. |
| Smoke-test the pipeline on 10 prompts per dataset | 2 hrs | Make sure nothing crashes before we commit to multi-day runs. |
| Confirm university HPC access and GPU allocation | 1 hr | Email the department. Reserve hours for Weeks 6-8. |
| **→ Done when:** The pipeline can accept any model + any dataset and produce clean trace logs. ||

---

#### Week 4: Trace Collection — Standard Mid-Weight Models
| Task | Est. Time | Details |
|------|-----------|---------|
| Configure DeepSeek-Distill-7B, Llama-3.1-8B, Gemma-2-9B | 3 hrs | Download weights, set quantization, verify tokenizer configs. |
| Run traces: all 3 models × all datasets × 3 temperatures | 12 hrs (automated) | This mostly runs unattended. ~500 prompts per model per dataset. |
| Spot-check 50 random traces for sanity | 2 hrs | Read actual model outputs to make sure the pipeline is recording correctly. |
| **→ Done when:** Clean trace logs exist for 3 standard architecture models across all datasets. ||

---

#### Week 5: Trace Collection — Math-Specialized Models
| Task | Est. Time | Details |
|------|-----------|---------|
| Configure Mathstral-7B, NuminaMath-7B, Phi-4 | 2 hrs | These may need special prompt templates for their math-specific training. |
| Run traces: all 3 models × all datasets × 3 temperatures | 10 hrs (automated) | Same protocol as Week 4. |
| Quick comparison: do math-trained models have a later boundary? | 4 hrs | Generate a preliminary drift plot comparing math-trained vs. standard models. |
| **→ Done when:** We can answer "does math-specific training shift the overthinking boundary?" with initial evidence. ||

---

#### Week 6: Trace Collection — Frontier Models (Part 1)
| Task | Est. Time | Details |
|------|-----------|---------|
| Spin up 80GB GPU instances on HPC/cloud | 3 hrs | Environment setup, driver checks, dependency installation. |
| Run traces: QwQ-32B, DeepSeek-Distill-32B | 14 hrs (automated) | These are the two primary frontier reasoning models — the most important new data. |
| Monitor for OOM errors and restart failed jobs | 3 hrs | Big models are fragile. Expect some babysitting. |
| **→ Done when:** We have frontier-scale reasoning traces. This is the first time we see how 32B+ models overthink. ||

---

#### Week 7: Trace Collection — Frontier Models (Part 2)
| Task | Est. Time | Details |
|------|-----------|---------|
| Run Mistral-Small-24B, InternLM2-20B, GPT-OSS-20B | 10 hrs (automated) | Three independent model families — critical for proving the boundary isn't Qwen-specific. |
| Run Qwen3.6-27B, Gemma-2-27B | 8 hrs (automated) | Tests whether novel attention architectures change the overthinking dynamics. |
| Log any architectural quirks (MoE routing, attention patterns) | 4 hrs | Some of these models behave very differently internally — document anything unusual. |
| **→ Done when:** We have traces from 5+ independent model families at frontier scale. ||

---

#### Week 8: Trace Collection — Final Models & Data Consolidation
| Task | Est. Time | Details |
|------|-----------|---------|
| Run remaining models: Yi-34B, Command-R, Qwen-MoE variants, Gemma-4-31B | 12 hrs (automated) | Complete the full model matrix. |
| Consolidate all trace logs into a unified directory structure | 4 hrs | Standardize file naming, verify completeness, create a manifest CSV. |
| Run a data completeness check | 2 hrs | Ensure every model × dataset × temperature cell has the target trace count. |
| Shut down expensive GPU instances | 0.5 hrs | Stop the billing clock. |
| **→ Done when:** The complete, multi-family trace database is finalized. No more model inference needed this semester. ||

---

#### Week 9: Feature Extraction
| Task | Est. Time | Details |
|------|-----------|---------|
| Extract the 4D observable vector from every trace | 8 hrs (computation) | Token count, moving average entropy, logprob variance, entropy drop — for every step of every trace. |
| Build the ground-truth label matrix | 4 hrs | For every step $t$, label whether it was a repair ($0 \to 1$), corruption ($1 \to 0$), or no change. |
| Format everything into clean train/test CSVs | 3 hrs | One row per (model, prompt, step) with features + labels. |
| **→ Done when:** We have a clean, tabular dataset ready for machine learning. ||

---

#### Week 10: Hazardous Feature Discovery
| Task | Est. Time | Details |
|------|-----------|---------|
| Build a "semantic loop" detector | 5 hrs | Check if the model starts repeating the same logical argument — a sign it's stuck. |
| Compute hidden-state Euclidean shifts between steps | 6 hrs | Measure how much the model's internal representation changes step-to-step. Big jumps = unstable. |
| Test "answer revision frequency" as a predictor | 3 hrs | How many times does the model change its final answer? Does flip-flopping predict failure? |
| Append all new features to the feature matrix | 2 hrs | Expand from 4D to ~8-10D observable vector. |
| **→ Done when:** We've identified which features are the strongest predictors of an incoming mistake. ||

---

#### Week 11: Baseline Predictor Training
| Task | Est. Time | Details |
|------|-----------|---------|
| Train Ridge Regression to predict $P(\mu_t \le 0)$ | 4 hrs | The simplest possible baseline. If this works, great. If not, we know we need nonlinear models. |
| Train XGBoost classifier | 5 hrs | Nonlinear baseline. Tune with cross-validation. |
| Evaluate both using AUROC, Precision-Recall, and calibration plots | 3 hrs | We need to know not just "is it accurate" but "are the probabilities trustworthy?" |
| Feature importance analysis | 2 hrs | Which of our 8-10 features actually matter? Can we drop some? |
| **→ Done when:** We have a working "Stop Button" predictor with measured performance metrics. ||

---

#### Week 12: Generalization Testing
| Task | Est. Time | Details |
|------|-----------|---------|
| Cross-domain test: train on GSM8K, evaluate on MATH | 4 hrs | Does the stopping rule learned on easy math transfer to hard math? |
| Cross-domain test: train on math, evaluate on HumanEval | 4 hrs | Does the stopping rule transfer to a completely different task (coding)? |
| Cross-family test: train on Qwen, evaluate on Llama/Mistral | 4 hrs | Is the boundary universal across model architectures? |
| Document performance gaps and hypothesize why | 3 hrs | If transfer fails, explain *what* is different. This is a result either way. |
| **→ Done when:** We know whether the stopping rule is universal, domain-specific, or model-specific. ||

---

#### Week 13: Visualization & Analysis
| Task | Est. Time | Details |
|------|-----------|---------|
| Generate difficulty-stratified drift grids for all models | 6 hrs | The signature chart type of this thesis — showing how the boundary shifts with question difficulty. |
| Create $\alpha_t$ vs $\beta_t$ scatter plots across model families | 4 hrs | Visualize the "repair vs corruption" tradeoff for each architecture. |
| Build a "model leaderboard" summary table | 2 hrs | Rank all models by their boundary location, peak accuracy, and overthinking cost. |
| Format all figures for publication quality | 3 hrs | Proper axis labels, legends, font sizes, and color schemes. |
| **→ Done when:** Every key finding has a clear, professional visual. ||

---

#### Week 14: Semester 1 Report & Submission
| Task | Est. Time | Details |
|------|-----------|---------|
| Compile the literature review into the report | 4 hrs | Pull from Week 2 drafts, polish for formal submission. |
| Write the methodology and experimental design sections | 4 hrs | Document exactly what we did, why, and how. |
| Write the preliminary results and "open questions for Phase 2" | 4 hrs | Summarize findings and explicitly state what Semester 2 will tackle. |
| Present to the committee | 2 hrs | Walk through the report, answer questions, get approval to continue. |
| **→ Done when:** 625.803 report submitted. Green light for Semester 2. ||

---

### SEMESTER 2 — Advanced math, live testing, thesis defense

---

#### Week 15 (Sem 2, W1): Advanced Predictor — Symbolic Regression
| Task | Est. Time | Details |
|------|-----------|---------|
| Set up PySR (Symbolic Regression library) | 3 hrs | Install, configure, and test on a toy problem. |
| Run symbolic regression on the feature matrix | 8 hrs (computation) | Try to discover an actual mathematical formula that predicts the boundary. |
| Compare discovered formulas against XGBoost performance | 3 hrs | Is the explicit formula as accurate as the black-box model? |
| **→ Done when:** We either have a clean formula (huge win) or evidence that the relationship is too complex for symbolic regression. ||

---

#### Week 16: Sequential Analysis — Statistical Rigor
| Task | Est. Time | Details |
|------|-----------|---------|
| Implement empirical-Bernstein confidence bounds | 8 hrs | These provide "anytime-valid" guarantees — meaning our stopping rule is statistically valid no matter when we check. |
| Compare pointwise vs. sequential validity | 4 hrs | Show the difference between "checking once" and "checking at every step." |
| Prove the stopping rule controls the false-stop rate | 4 hrs | The mathematical heart of the thesis — this is what makes it an ACM thesis, not just an ML paper. |
| **→ Done when:** The stopping rule has a formal statistical guarantee, not just empirical accuracy. ||

---

#### Week 17: Building the Live Stopping System
| Task | Est. Time | Details |
|------|-----------|---------|
| Build a token-by-token generation loop with a predictor hook | 8 hrs | At each step, compute the observable vector and run the predictor. |
| Wire the "STOP" signal into the generation loop | 4 hrs | When $P(\mu_t \le 0)$ crosses the threshold, halt generation immediately. |
| Test on 100 prompts with a small model to verify it works | 2 hrs | Sanity check before scaling up. |
| **→ Done when:** We have a live system that actively prevents a model from overthinking in real-time. ||

---

#### Week 18: The Big Comparison
| Task | Est. Time | Details |
|------|-----------|---------|
| Run 1,000 prompts: dynamic stopping (our method) | 4 hrs | Record accuracy, stop step, and total tokens generated. |
| Run 1,000 prompts: "never stop" baseline | 4 hrs | Let the model run to the maximum step count. |
| Run 1,000 prompts: "fixed stop at step 5" baseline | 4 hrs | A naive but common heuristic. |
| Calculate accuracy gain and compute savings | 3 hrs | The headline numbers of the thesis: "X% more accurate, Y% fewer tokens." |
| **→ Done when:** We have definitive evidence that our method beats the baselines. ||

---

#### Week 19: Error Analysis
| Task | Est. Time | Details |
|------|-----------|---------|
| Isolate the 50 worst predictions | 4 hrs | Where did our system stop too early (killed a correct chain) or too late (let a hallucination run)? |
| Manually read and categorize the failure traces | 6 hrs | Understand *why* the predictor was wrong — was the feature misleading? Was the model doing something unusual? |
| Draft the "Limitations" section | 4 hrs | Honest assessment of when and why the method fails. |
| **→ Done when:** We understand our method's failure modes and can articulate them clearly. ||

---

#### Week 20: Final Data Cleanup
| Task | Est. Time | Details |
|------|-----------|---------|
| Identify missing or corrupted traces | 3 hrs | Final sweep of the entire dataset. |
| Run patch-up jobs for any gaps | 5 hrs | Fill holes with targeted re-runs. |
| Lock the dataset and create a cold-storage backup | 2 hrs | After this point, NO more data collection. |
| **→ Done when:** The dataset is frozen forever. ||

---

#### Week 21: Writing — Methods Chapters
| Task | Est. Time | Details |
|------|-----------|---------|
| Write Chapter 2: Mathematical Framework | 8 hrs | The formal derivation of the stopping boundary, hazard processes, and drift equation. |
| Write Chapter 3: Experimental Design | 5 hrs | Data collection protocol, model descriptions, dataset descriptions, hardware setup. |
| **→ Done when:** The technical backbone of the thesis is on paper. ||

---

#### Week 22: Writing — Results Chapter
| Task | Est. Time | Details |
|------|-----------|---------|
| Integrate all charts, drift grids, and comparison tables | 4 hrs | Every figure gets a home in the document. |
| Write the narrative: what happened, what surprised us, what confirmed our theory | 8 hrs | This is the storytelling — guiding the reader through the data. |
| **→ Done when:** Chapter 4 is drafted. ||

---

#### Week 23: Writing — Discussion, Intro, Conclusion
| Task | Est. Time | Details |
|------|-----------|---------|
| Write the Discussion: broader impacts, connection to AI scaling, future work | 6 hrs | Why does this matter beyond our specific experiments? |
| Write the Introduction and Abstract | 4 hrs | Often the last thing written — needs to summarize the whole story. |
| Write the Conclusion | 2 hrs | Clear, concise summary of contributions. |
| **→ Done when:** The full rough draft exists. ||

---

#### Week 24: Advisor Review Cycle 1
| Task | Est. Time | Details |
|------|-----------|---------|
| Submit draft to Dr. Woods and Dr. Pemy | 1 hr | Send it off. |
| Self-review: typos, citation formatting, figure labels | 6 hrs | Clean up the obvious stuff while waiting for advisor feedback. |
| **→ Done when:** Feedback is in hand. ||

---

#### Week 25: Revisions
| Task | Est. Time | Details |
|------|-----------|---------|
| Address all advisor comments — structural, technical, and editorial | 8 hrs | This could range from "expand this proof" to "this chart is confusing, redo it." |
| Re-generate any figures that were flagged | 5 hrs | Sometimes a chart that makes sense to us doesn't make sense to the reader. |
| **→ Done when:** A polished, defense-ready thesis document. ||

---

#### Week 26: Defense Slide Deck
| Task | Est. Time | Details |
|------|-----------|---------|
| Outline the 30-minute presentation | 2 hrs | Plan the story arc: problem → math → experiments → results → so what? |
| Build the slides with heavy emphasis on visuals | 8 hrs | The drift grids, the before/after accuracy charts, and the "zero crossing" moment. |
| Add backup slides for anticipated tough questions | 2 hrs | Prepare for "what about model X?" or "why not use method Y?" |
| **→ Done when:** The deck is complete and visually compelling. ||

---

#### Week 27: Practice Runs
| Task | Est. Time | Details |
|------|-----------|---------|
| Do 3 full timed rehearsals | 3 hrs | Practice hitting the 30-minute mark. |
| Brainstorm 20 hard questions the committee might ask | 3 hrs | "How does this scale to 70B models?" / "Why not use reinforcement learning?" / "Is $\lambda$ arbitrary?" |
| Write concise answers to each question | 2 hrs | Have a cheat sheet ready. |
| **→ Done when:** Confident and ready. ||

---

#### Week 28: Defense & Submission
| Task | Est. Time | Details |
|------|-----------|---------|
| Defend the thesis before the committee | 2 hrs | Present, answer questions, receive feedback. |
| Incorporate any final requested edits | 3 hrs | Usually minor at this stage. |
| Submit the final approved PDF to JHU | 1 hr | Done. |
| **→ Done when:** M.S. in Applied and Computational Mathematics. 🎓 ||
