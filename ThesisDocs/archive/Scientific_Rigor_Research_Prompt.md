# Scientific Rigor Audit & Failure Analysis — Research Operating Brief for Claude Fable 5
## Overthinking Boundary: Verifying Variable Isolation, and a Full-Scale Autopsy of the 7.57% Losses

*Target Runtime: **Claude Fable 5 for judgment/verification work, Claude Sonnet 5 for mechanical fact-finding**, run inside Claude Code or an equivalent agentic harness with subagent dispatch, multi-agent workflow orchestration, and file read/write/execute access. Model and reasoning-effort are set **per subagent**, not once for the whole session — see Section 1.*

*This is a research-and-verification mission, not an implementation mission. Its deliverables are findings, verdicts, and a ready-to-execute next-experiment protocol — not new production code, not a new GPU run, not a rebuilt pipeline. Small throwaway analysis/verification scripts are in scope and expected (see Section 4); editing `research/run_experiment_matrix.py`, `research/real_trace_experiments.py`, or any other production pipeline file, and launching any new GPU job, are explicitly out of scope for this pass.*

*Prepared 2026-07-09 by a Sonnet 5 brainstorming/planning pass, based on a direct read of the current repo state — not a rehash of what the existing reports claim about themselves. Verify and extend what's below; don't re-discover it from scratch.*

**Relationship to `ThesisDocs/Startup_Research_Prompt.md`:** that document is the GTM/YC operating brief and deliberately stays out of scope here (per its own §0.6, mixing the two is a live open question it hasn't resolved yet). This document is purely about whether the *research* is actually rigorous and what's causing the *research's* current error rate. Don't import commercialization framing into this pass, and don't let this pass's findings silently leak into a pitch deck without a human deciding that's appropriate — that's the other document's job, specifically its Agent 0a.

---

## ⏰ The One Fact That Changes How You Should Read This Prompt

**Your advisor already got a rigor report once, and it's already gone stale.** `ThesisDocs/Scientific_Method_Verification_Report.md` and `ThesisDocs/Scientific_Method_Deep_Dive_Report.md` (both dated the July 1, 2026 meeting) already claim a ceteris-paribus control system and already name three failure modes for the loss rate. **Treat every sentence in those two files as a claim to re-derive from code and data, not as an established fact to cite.** Three concrete reasons why:

1. Commit `8ce9b9f` ("fix: floor boundary detection at T_MIN=2 in ad-hoc scripts") landed **after** both reports and fixed exactly the class of bug (an unfloored boundary operator) that produced one of the two numeric contradictions flagged in `Startup_Research_Prompt.md` §0.3 — but neither Scientific Method report was regenerated afterward, and neither got the ⚠️ SUPERSEDED banner that three other stale reports got in the very next commit (`9819ee2`).
2. Commit `074bc70` ("fix: handle NaN in regrade_traces.py row parsing") touches the label-correction pipeline that feeds every downstream correctness/utility number — another concrete, recent change neither report reflects.
3. This codebase has a documented history of confidently-reported numbers turning out to be wrong: a 79-agent audit (`research/reports/deep_code_audit.md`, 2026-06-20) found corrupted `correct` labels, in-sample-inflated AUC, and a theory-forbidden step-1 boundary operator; a later SIGKILL-during-CSV-write bug field-shifted rows and silently bled strings into numeric columns. **The "7.57%" and "92.43%" figures your advisor heard are exactly the kind of number that has moved under this codebase's feet before.** Recompute them fresh; don't cite them.

This prompt exists because "we wrote a confident-sounding report saying we're rigorous" is not the same thing as "we verified we're rigorous," and the gap between those two is precisely what your advisor's feedback was pointing at.

---

## How to Run This Prompt

1. **Start your Claude Code session with working directory = `C:\Aditya_Data\Personal\ResearchThesis`** (this repo — not `algorithm-x-optimizer`). Paste this whole document as your first message.
2. **Read Section 0 in full before dispatching any subagent.** It contains specific, checkable leads (exact file paths, exact numbers, exact arithmetic) that took real repo-reading to find. Re-deriving them from zero wastes budget that should go toward the genuinely open questions.
3. **Run `git status` and `git diff` before trusting any pipeline file as ground truth for how data was actually produced.** As of 2026-07-09 there is one uncommitted change (§0.5) and an untracked `.agents/` directory (orchestration scratch state from a prior session — ignore its contents, it is not research material).
4. **Use real agent orchestration with per-agent model/effort tiers**, not a simulated one. Section 1 gives the roster. Reserve Fable 5 / max for judgment calls that require weighing evidence and writing a verdict; use Sonnet 5 for mechanical file-diffing, grepping, and status-mapping.
5. **All new outputs live under `ThesisDocs/rigor_audit/`.** This keeps `ThesisDocs/` as the human-facing narrative home without scattering a six-file audit across its root.
6. **Completion criterion (yes, applying the thesis's own stopping-rule discipline to itself is a little on the nose — the point still stands):** you are done when every file in the Section 3 manifest exists, every claim in it cites the exact file/line/command that supports it, the three reconciliation items in §0.3 are each explicitly closed or explicitly still-open-with-a-named-blocker, and Module 5's executive synthesis gives an explicit, defensible answer to "is this rigorous enough to tell my advisor 'yes, verified' " — not a hedge, an answer. Don't keep re-deriving numbers that already check out past the point of marginal value.

---

## Section 0: Context & Required Reading

### 0.1 What Already Exists (Read These First, Trust Them Least)

| File | What it claims | Status |
|---|---|---|
| `ThesisDocs/Scientific_Method_Verification_Report.md` | Ceteris-paribus control system; bf16-ladder results table; 3 named failure modes for the 7.57% | Pre-dates `8ce9b9f` and `074bc70`; not regenerated; no SUPERSEDED banner |
| `ThesisDocs/Scientific_Method_Deep_Dive_Report.md` | Per-factor isolation tables (6 factors) + same 3 failure modes | Same staleness exposure; its Factor 5 (stakes sweep) table is sourced from a script that was later bug-fixed |
| `ThesisDocs/July_1_Checkin.md` | Full from-scratch walkthrough: methodology, 75,996-run scale table, global win/loss breakdown (89.59% / 2.83% / 7.57%), roadmap | Most complete single source of the headline numbers; still the right onboarding doc, but the numbers inside it are exactly what Module 2 must recompute, not copy |
| `ThesisDocs/project_plan_advisor_meeting.md` | Original semester plan, week-by-week, 25+ model roster | **Largely superseded by what actually happened** — the plan lists models like QwQ-32B, Gemma-2/4, InternLM2-20B, GPT-OSS-20B that don't appear in the actual 13-model roster that got run (§0.3 of `Startup_Research_Prompt.md` has the real 13). Useful for *why* the experimental design looks the way it does, not for *what* was actually executed. Don't let a subagent cite its model roster as current without checking. |
| `ThesisDocs/Startup_Research_Prompt.md` | GTM operating brief; §0.3 flagged 3 numeric reconciliation items that are this document's direct ancestors | Out of scope here (see header) except for §0.3, which Module 0 must resolve using *current* files, not re-litigate from scratch |
| `research/reports/deep_code_audit.md` | The 79-agent audit that fixed grader bugs, in-sample leakage, and the unfloored boundary operator (2026-06-20) | Historical record — read it to know what's *already* been fixed so you don't re-flag it as new |

### 0.2 The Real Data Model — Where Ground Truth Actually Lives

- **Per-run policy comparison (the source of the 7.57%):** `research/outputs/experiment_matrix/{model}__{dataset}/detector_comparison_by_run.csv` — long format, columns include `run_id`, `detector`, `stop_utility`; filter `detector in ['hazard_drift', 'never_stop']` and pivot on `run_id` to get win/loss/tie per run. `research/analyze_runs.py` already does roughly this (it's committed but its output doesn't appear to be saved anywhere — confirm whether it's ever been run to completion, and if its logic is correct, before trusting it).
- **Per-step trace detail (needed for failure taxonomy):** trace-level CSVs with step-by-step `q_t`, entropy, answer, confidence — the two Scientific Method reports cite examples from `research/outputs/real_traces_bf16_ladder/qwen2p5_7b/trace_steps.csv`.
- **⚠️ Open question Module 0 must resolve first:** `research/outputs/real_traces_bf16_ladder/` and `research/outputs/experiment_matrix/` appear to be **two different output trees** — the former looks like the original 3-model (Qwen 7B/14B/32B), GSM8K-only capability-ladder sweep; the latter looks like the full 52-cell (13 models × 4 datasets) matrix. Before Module 2 builds a failure taxonomy, confirm **which tree is the authoritative source of the 74,540-run / 7.57% headline figure**, whether the per-cell trace-level CSV exists in that same tree for every cell (not just the 3-model ladder), and document this clearly — mixing the two trees by accident would quietly invalidate the whole exercise.
- `scratch_analyze.py` (repo root, not `research/`) and `research/analyze_runs.py` were added together in one commit (`32de197`, "chore: add ground-truth re-verification scripts") — this looks like a start on exactly this mission that didn't get finished. Read both before writing new analysis code; extend or fix them rather than duplicating.

### 0.3 The Three Known Reconciliation Items — Current Status (Verify, Don't Re-Litigate From Zero)

`Startup_Research_Prompt.md` §0.3 flagged three internal numeric contradictions. Direct reads during this planning pass found:

1. **Qwen-7B boundary: step 5 vs. step 1 — ✅ appears CLOSED.** The current (post-`8ce9b9f`) `research/outputs/real_traces_bf16_ladder/stakes_sweep_report.md` shows `qwen2p5_7b, c=0.0 → T*=5`, matching the main capability-ladder story. **Action: confirm this still holds (re-read the file — don't take this document's word for it either) and cite it as closed. Don't re-investigate from scratch.**
2. **`stakes_sweep_report.md`'s prose contradicting its own table — ✅ appears CLOSED.** The current file's "Key Insights" now correctly states the boundary shifts *later* as the penalty rises, matching the table (c=0→T\*=5, c=10→T\*=7, c=100→T\*=10 for qwen2p5_7b). **Action: confirm and cite as closed.**
3. **DeepSeek-R1-Distill-7B boundary: step 2 vs. step 1 — ⚠️ still genuinely open.** The current (post-fix) `research/outputs/real_traces_bf16_ladder/prompted_vs_distilled_report.md` shows `T*=2` (correctly floored at T_MIN=2). But a separate prior-session note describes this same model's boundary, from the *main* cross-family pipeline (not this ad-hoc script), as "step 1." Since the t≥2 floor is supposed to be a structural invariant enforced in **all three** canonical pipeline copies as of the June 20 audit (per `deep_code_audit.md`'s remediation), a report showing step 1 for a supposedly-floored model is itself a bug — either a stale report that predates the floor fix, or a genuine second unfloored code path that the June 20 audit and the `8ce9b9f` ad-hoc-script fix both missed. **Action: find the actual current DeepSeek-R1-Distill-7B GSM8K boundary in `research/outputs/experiment_matrix/deepseek_r1_distill_7b__gsm8k/` and in whatever the current canonical cross-family aggregate report is (see §0.2's open question about which tree is authoritative), and resolve the discrepancy with a named cause, not a guess.**
4. **Generalize the pattern, don't just close these three:** grep the repo for every file that states a `T*`/boundary-step value for any model, and confirm each is (a) generated from current code, (b) consistent with the t≥2 floor, and (c) not silently contradicted by a sibling report. List every straggler you find, even ones nobody has flagged yet.

### 0.4 What "the 8%" Actually Means — Don't Let This Get Blurry

The number your advisor and you are worried about is the **"Worse (Losses)" bucket from the global run-level win/loss/tie breakdown**: out of 74,540 completed runs, `hazard_drift` was a strict win in 89.59% (66,784), a harmless tie in 2.83% (2,110), and **strictly worse than the `never_stop` baseline in 7.57% (5,646)** (`ThesisDocs/July_1_Checkin.md`, Part 6). This is a claim about **the stopping policy's decision quality relative to a naive baseline** — it is *not* the same thing as "the model's raw accuracy was 8% worse than something," and it is *not* the same thing as raw per-benchmark accuracy (which is far lower than 92% on GPQA, for instance, regardless of any stopping policy). Module 2 must state this distinction explicitly in its output, because conflating "the stopping policy underperformed the do-nothing baseline on 7.57% of runs" with "the model got 8% of answers wrong" would itself be a rigor failure in the audit that's supposed to be catching rigor failures.

Two more things worth knowing before Module 2 starts:
- **The per-dataset numbers already published (ARC 94.63%, GPQA 92.05%, MATH 86.40%, GSM8K 85.32%) are labeled "Win Rates," not loss rates** — their complements include both losses *and* ties, so don't assume e.g. "GSM8K loss rate ≈ 14.68%." Compute the actual per-dataset (and per-model, and per-temperature) Worse-rate directly from the CSVs, the same way the global 7.57% should be recomputed rather than trusted.
- **A specific, testable prediction worth checking first:** the existing narrative (both Scientific Method reports) argues GSM8K/MATH are the "sweet spot" for late, missable corrections (high repair headroom) while ARC/GPQA collapse to the floor with little room to lose. If that story is right, the 5,646 losses should concentrate heavily in GSM8K/MATH and be comparatively rare in ARC/GPQA. Confirm or refute this with the actual per-dataset breakdown before building the failure taxonomy — it tells you where to spend classification effort.

### 0.5 Standing Hygiene Instructions

- **Uncommitted change as of 2026-07-09:** `research/real_trace_experiments.py`, inside `reconcile_existing_outputs`, currently has an uncommitted edit changing `is_complete = observed_steps == expected_step_sequence and run_id_str in hidden_run_ids` to `is_complete = observed_steps == expected_step_sequence` (dropping the `hidden_run_ids` membership check). Module 0 should determine exactly what this changes about completeness detection and flag it for explicit human review — **don't assume it's a bug, and don't assume it's a validated fix; just surface it precisely** so nobody downstream accidentally treats a mid-edit pipeline as the one that produced the committed data.
- **`.agents/` is untracked orchestration scratch** from a prior multi-agent GPU-setup session (BRIEFING.md/handoff.md/worker_* subdirectories, dated 2026-07-01/02). It is not research content and should not be cited as evidence of anything.
- Whenever a subagent is about to cite a number, it should be able to answer "which script, which commit, which file produced this" — if it can't, the number gets a `[UNVERIFIED — recompute]` tag instead of being stated as fact.

---

## Section 1: Agent Orchestration Plan

| # | Agent | Mission | Model | Effort | Depends on | Output |
|---|---|---|---|---|---|---|
| 0 | Repo State & Staleness Auditor | Resolve §0.3 and §0.5; map every report to current/stale; confirm the §0.2 output-tree question | Sonnet 5 | high | — | `ThesisDocs/rigor_audit/00_repo_state_and_staleness.md` |
| 1 | **Scientific-Method Adversarial Verifier** | Re-derive (not re-read) every variable-isolation claim across 7 axes directly from code and data | **Fable 5** | **max** | 0 | `ThesisDocs/rigor_audit/01_scientific_method_adversarial_verification.md` |
| 2 | **Failure Taxonomy at Scale** | Recompute the current loss rate fresh; classify every loss (not a handful of anecdotes) into a quantified, mutually-exclusive taxonomy | **Fable 5** | **max** (drive real classification code, don't hand-wave) | 0 | `ThesisDocs/rigor_audit/02_failure_taxonomy.md` + a committed classification script |
| 3 | Root-Cause → Improvement Levers | For each failure category, what would actually reduce it, and by how much | Fable 5 | max | 1, 2 | `ThesisDocs/rigor_audit/03_improvement_levers.md` |
| 4 | **Next-Experiment Protocol Designer** | Pre-registered, ready-to-launch experiment designs that close rigor gaps and test the levers | Fable 5 | max | 1, 2, 3 | `ThesisDocs/rigor_audit/04_next_experiment_protocols.md` |
| 5 | Executive Synthesis | One navigable answer: are we rigorous, what's the real failure rate and why, what do we do next | Fable 5 | max | all | `ThesisDocs/rigor_audit/00_EXECUTIVE_SUMMARY.md` |

**Phasing:**

```
Phase 1 (solo):      0  (Sonnet / high)
Phase 2 (parallel):  1  (Fable / max)   +   2  (Fable / max)
Phase 3 (solo):      3  (Fable / max)
Phase 4 (solo):      4  (Fable / max)
Phase 5 (solo):      5  (Fable / max)
```

If your harness exposes a `Workflow`-style tool with `parallel()`/`pipeline()` and per-call `model`/`effort` options:

```js
phase('Phase 1')
const groundTruth = await agent(REPO_STATE_PROMPT, {model: 'claude-sonnet-5', effort: 'high'})

phase('Phase 2')
const [rigor, taxonomy] = await parallel([
  () => agent(RIGOR_VERIFIER_PROMPT(groundTruth), {model: 'claude-fable-5', effort: 'max'}),
  () => agent(FAILURE_TAXONOMY_PROMPT(groundTruth), {model: 'claude-fable-5', effort: 'max'}),
])

phase('Phase 3')
const levers = await agent(LEVERS_PROMPT(rigor, taxonomy), {model: 'claude-fable-5', effort: 'max'})

phase('Phase 4')
const protocols = await agent(NEXT_EXPERIMENT_PROMPT(rigor, taxonomy, levers), {model: 'claude-fable-5', effort: 'max'})

phase('Phase 5')
const summary = await agent(SYNTHESIS_PROMPT(rigor, taxonomy, levers, protocols), {model: 'claude-fable-5', effort: 'max'})
```

If your harness doesn't expose per-call model selection, run Agent 0 as a plain cheaper-mode pass and reserve the Fable-5 session itself for Agents 1–5. **Every subagent prompt must be self-contained** — inline the relevant facts from Section 0, name the exact file to write, and state which other agent owns which question so work doesn't duplicate.

---

## Section 2: The Research Mission — Modules

### Module 0: Repo State & Staleness Audit
Do §0.3, §0.5, and the §0.2 output-tree question, in that order. This gates everything else — modules 1 and 2 need to know which files are current before they cite anything from them.

### Module 1: Scientific-Method Adversarial Verification
For **each** of the following seven axes, don't summarize what the existing reports say — go to the actual invocation (launcher script, CLI args, config) and the actual output data, and state a verdict of **CONFIRMED / CONFIRMED-WITH-CAVEAT / NOT SUPPORTED / NOT CONTROLLABLE (inherent)**, each with the exact file/line/command used to check it:

1. **Model parameter scale** (0.5B→32B within Qwen). Confirm dataset, temperature set, precision, prompt template, and seeds are identical across every size in the launcher configs — not just "should be by design."
2. **Precision/quantization** (bf16 vs. 4-bit, Qwen2.5-7B). Confirm the two invocations differ *only* in the `--quantization` flag.
3. **Temperature** (0.1/0.6/1.0, same model+dataset). This is the sharpest test available: pull the actual sequence of problem/question IDs seen at each temperature for the same model+dataset and confirm they are identical multisets in identical order. If they aren't, temperature isn't actually isolated — something else is riding along with it.
4. **Dataset/benchmark.** Confirm the grading function applied is the same for every model within a given benchmark (no per-model special-casing that could quietly advantage or disadvantage a specific model's output format).
5. **Question/trace-set identity — start here, this one has a concrete lead.** Across the 13-model × GPQA cells, 9 models show exactly 1,344 runs (= 448 GPQA problems × 3 temperatures — the complete set), but `llama_3p1_8b_instruct` shows 1,348, `qwen2p5_3b` shows 1,349, `yi_1p5_9b_chat` shows 1,349, and `qwen2p5_0p5b` shows 1,354 (`ThesisDocs/July_1_Checkin.md`, Part 3 table) — all four *above* the clean full-set count, not below it. Check `detector_comparison_by_run.csv` (or the underlying trace CSV) for these four model×GPQA cells for duplicate `run_id`s. If duplicates exist, confirm whether the win-rate/boundary computation already dedupes them — if it doesn't, a handful of GPQA problems are being double-counted for exactly these four models' statistics, which is a small but real and previously-unflagged violation of "same questions, same weight, every model."
6. **Stakes/penalty sweep** ($c \in \{0, 0.5, 1, 2, 5, 10, 20, 50, 100\}$). Confirm only $c$ (and derived $v$ if applicable) varies, and that no filtering logic downstream changes which traces are included as $c$ changes.
7. **Reasoning paradigm** (prompted Qwen2.5-7B vs. RL-distilled DeepSeek-R1-Distill-7B). **Be honest that this one is not a controlled experiment in the classical sense** — the two models differ in base architecture, training data, and tokenizer, not just "prompted vs. distilled." State explicitly which confounds are *not* and *cannot* be controlled here, and reframe the existing claim accordingly (suggestive case comparison, not isolated-factor causal claim). Correctly identifying what's fundamentally uncontrollable vs. what's controllable-but-wasn't is itself part of what "rigorous" means — don't chase all-green verdicts by overclaiming control where none is possible.

**Deliverable framing:** this report explicitly states, for each of the two July 1 Scientific Method reports, whether it is now SUPERSEDED (and by what), CONFIRMED-AS-IS, or CONFIRMED-WITH-NAMED-CORRECTIONS. Don't leave that judgment implicit.

### Module 2: Failure Taxonomy at Scale
1. **Recompute the global win/loss/tie breakdown fresh** from the current `research/outputs/experiment_matrix/*/detector_comparison_by_run.csv` (all cells, whichever tree §0.2 confirms is authoritative). Compare to the July 1 figure of 7.57% (5,646/74,540). If it's moved, say by how much and which fix likely moved it — you have three concrete candidates already on record (`8ce9b9f`, `074bc70`, and the still-uncommitted change from §0.5).
2. **Compute the real per-dataset, per-model, and per-temperature loss rates directly** — don't infer them from the published win-rate complements (§0.4). Check the specific prediction in §0.4 (losses concentrate in GSM8K/MATH) against real numbers.
3. **Build a quantified, mutually-exclusive, programmatic taxonomy of every loss row** — not 2-3 hand-picked trace examples generalized into "the" failure modes. For each loss, join `detector_comparison_by_run.csv` against the corresponding per-step trace CSV to get the full $q_t$/entropy/answer/confidence history plus the `never_stop` final answer, and classify based on the actual step-by-step data, not narrative pattern-matching. The three previously-named modes (arithmetic slip-then-late-fix, sub-problem/intermediate-answer confusion, parser lag/misextraction) are a **non-exhaustive starting point** — if they don't cleanly and completely partition the loss population, define additional or refined categories and justify them from the data. Every category's count must be reported, and the counts must sum to the total loss population for whatever slice you're analyzing.
4. **For each category, tag it** as (i) a fixable pipeline/parser/grader bug, (ii) fundamentally unpredictable from the currently-recorded observables (a genuine online-decision limit), or (iii) potentially fixable with a better feature/model — and justify the tag with evidence (e.g., category (iii) requires showing that *some* currently-recorded signal already carried the predictive information the policy failed to use).
5. Commit the classification script under `research/` (extending `analyze_runs.py`/`scratch_analyze.py` rather than duplicating them), clearly named, so this is re-runnable after future fixes rather than a one-off.

### Module 3: Root-Cause → Improvement Levers
Using Module 2's tagged categories:
1. For every (i) fixable-bug category: propose the specific fix and estimate the utility/accuracy recovery **using that category's actual row count** as an upper bound (fixing the bug removes that failure mode; it doesn't guarantee the policy then makes the *right* call every time).
2. For every (iii) fixable-with-better-features category: propose a specific new observable or model change, grounded in the actual mechanism found in Module 2 — not a generic "use a bigger model" suggestion.
3. For (ii) fundamental categories: quantify the realistic ceiling. The current policy captures 50.9% of the oracle-vs-baseline utility gap against an online (no-foresight) upper bound generally cited around 50%, with a Prophet-Inequality-style worst case near 1/e ≈ 36.8% — be honest about how much of the remaining gap is plausibly closeable at all versus a structural floor.
4. Rank every lever by (estimated recovery) × (implementation cost/risk). This ranking is the direct input to Module 4.

### Module 4: Next-Experiment Protocol Designer
1. **Reconcile against the known backlog before proposing anything new** — re-check current status of: MATH's GPU-verification completeness across the full model roster; the two previously-failed cells (`llama_3p1_8b_instruct` — prior notes say unblocked as of the 4-dataset sweep, confirm it now has real data in all 4 datasets; `mistral_small_3p1_24b` — prior notes say still failing as of 2026-06-27, confirm current status); the coding-domain roadmap item (HumanEval/MBPP, never wired in per `real_trace_experiments.py`'s current parser coverage); the e-process honest reframing and λ-sensitivity sweep (both noted as pending in prior sessions).
2. **For every proposed experiment** — whether closing a Module 1 rigor gap, testing a Module 3 lever, or picking up a backlog item — require a pre-registration block, written *before* anyone would run it:
   - Independent variable (exactly one).
   - Explicitly-held-constant variables, using the same 7-axis checklist from Module 1, so the new experiment doesn't repeat old confound mistakes.
   - Hypothesis, stated in advance, falsifiable.
   - Minimum sample size for statistical power — reuse the SE/Z-score method already established in `ThesisDocs/July_1_Checkin.md` Part 7 rather than inventing a new one.
   - Estimated GPU-hours on the RTX PRO 6000 Blackwell (96GB) box (the same hardware used for the 52-cell sweep).
   - A success/failure criterion decided in advance, not fitted after seeing results.
3. Sequence and prioritize the resulting experiment list.

### Module 5: Executive Synthesis
1. Explicit, non-hedged answer: is the scientific-method claim genuinely verified now? (Yes / Partially, with named exceptions / No, with named blockers.)
2. One table: the true current failure rate (from Module 2's fresh computation) and its full cause breakdown by category.
3. Top-N next actions, ranked, each traceable to a specific Module 3 lever or Module 4 protocol.
4. **What can be told to the advisor with full confidence at the next check-in, and what should be explicitly flagged as still-open** — don't let anything Module 1–4 found get smoothed over in the summary.

---

## Section 3: Output Manifest

| File | Module | Must contain |
|---|---|---|
| `ThesisDocs/rigor_audit/00_EXECUTIVE_SUMMARY.md` | 5 | Non-hedged rigor verdict + true failure rate/cause table + ranked next actions |
| `ThesisDocs/rigor_audit/00_repo_state_and_staleness.md` | 0 | §0.3 items resolved with citations; §0.5 uncommitted-change flagged; §0.2 output-tree question resolved |
| `ThesisDocs/rigor_audit/01_scientific_method_adversarial_verification.md` | 1 | Verdict (CONFIRMED / CONFIRMED-WITH-CAVEAT / NOT SUPPORTED / NOT CONTROLLABLE) per axis, with evidence; explicit supersession call on the two July 1 reports |
| `ThesisDocs/rigor_audit/02_failure_taxonomy.md` | 2 | Fresh loss-rate computation; quantified mutually-exclusive taxonomy with counts; per-category bug/feature/fundamental tags |
| `ThesisDocs/rigor_audit/03_improvement_levers.md` | 3 | Ranked levers with estimated recovery and cost |
| `ThesisDocs/rigor_audit/04_next_experiment_protocols.md` | 4 | Pre-registered, sequenced, ready-to-launch experiment designs |
| Classification script under `research/` | 2 | Re-runnable, extends `analyze_runs.py`/`scratch_analyze.py` rather than duplicating |

---

## Section 4: Operating Principles

- **Adversarial by default toward prior Claude-authored output**, including the two Scientific Method reports and this document's own §0.3/§0.5 claims — re-verify them, don't just cite them, even though this planning pass tried to get them right.
- **Analysis and verification scripts are in scope; production changes are not.** Write and run whatever throwaway code is needed to classify thousands of rows or recompute a statistic. Do not edit `research/run_experiment_matrix.py`, `research/real_trace_experiments.py`, or any other file that determines how GPU experiments are actually run, and do not launch any new GPU job — this pass produces the *design* for the next run, not the run itself.
- **Every empirical claim cites the exact file, line, or command that produced it.** A number without a citation gets `[UNVERIFIED — recompute]` instead of being stated as fact.
- **Distinguish statistically-confirmed from merely-suggestive**, especially for Module 1's axis 7 (prompted vs. distilled) and any small-sample slice of Module 2's taxonomy.
- **No placeholders.** If a module can't fully resolve something in the time available, it states exactly what's blocking it and what would resolve it, rather than filling the space with generic advice.
- **Self-review before declaring done:** re-read the Section 3 manifest against what actually exists on disk, re-read §0.3's three items for explicit closure status, and only then write the executive summary.
