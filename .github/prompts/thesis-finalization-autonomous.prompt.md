---
name: "Autonomous Thesis Finalization"
description: "Continue the interrupted thesis-finalization run from the latest remote Mistral checkpoint state: reuse the completed mu_t / T_c documentation, resume or restart the matched-protocol Mistral run safely, verify the selected L4 runtime, and checkpoint commits/pushes aggressively."
argument-hint: "Optional: preferred non-Qwen family (llama/gemma/mistral), compute budget, benchmark, or thesis-doc priority"
agent: "agent"
---

# Autonomous Thesis Finalization Prompt

Use this prompt inside the `ResearchThesis` workspace when you want GPT-5.4 to autonomously execute the next full thesis-finalization cycle from the **current audited repo state**, not from the old pre-audit story.

This is a **post-recovery, post-audit** prompt. It assumes the repo already contains:

- the recovered Qwen 7B 4-bit run,
- the regenerated Qwen 7B reports,
- the corrected cross-family summary, and
- the updated README that now states the narrower but defensible claim.

Do not restart from scratch. Build on the audited state that is already in the repository.

## Current Ground Truth

Read these first and treat them as the current authoritative baseline unless new evidence from this run supersedes them:

- [Cross-family report](../../research/CROSS_FAMILY_REPORT.md)
- [Cross-family open questions](../../research/CROSS_FAMILY_OPEN_QUESTIONS.md)
- [Qwen 7B final report](../../research/FINAL_QWEN_7B_L4_RESULTS.md)
- [Recovery audit](../../research/RECOVERY_AUDIT_2026-04-02.md)
- [Theory note](../../research/overthinking_boundary.md)
- [README](../../README.md)
- [Thesis proposal draft](../../ThesisDocs/acm_thesis_proposal_draft.md)
- [Real-trace harness](../../research/real_trace_experiments.py)
- [Trace analysis](../../research/trace_analysis.py)
- [Artifact generator](../../research/generate_thesis_artifacts.py)
- [Colab runner](../../tools/run_colab_experiment.py)

The current evidence state is:

- Qwen 7B 4-bit is the strongest completed late-boundary run in the repo.
- Qwen 7B has step-1 accuracy `0.3644`, peak correctness `0.7789` at step `9`, corrected conditional-hazard boundary `6`, fitted boundary `7`, and never-stop oracle gap `0.4317`.
- DeepSeek 1.5B and Qwen 0.5B both cross at step `1` after the hazard audit.
- The strongest currently defensible claim is **capability-linked late-boundary evidence**, not full cross-family robustness.
- The repo already contains a concrete stopping-rule framework centered on:

$$
\mu_t = (1-q_t)\alpha_t - q_t\beta_t - \lambda
$$

with the latent theorem-facing boundary:

$$
T_c = \inf\{t \ge 0 : \mu_t \le 0\}.
$$

## Current Continuation State (Post-Interrupted Mistral Run)

The repo has already advanced beyond the original version of this prompt. Treat the following as completed unless the files are missing or corrupted:

- Thesis-ready stopping-rule documentation was added in commit `b2a2b94`.
- The harness was extended to support the selected non-Qwen family alias `mistral_7b_instruct_v0p3` in commit `048036e`.
- A successful Mistral smoke run was committed in `9d74e32`.
- L4 optimization benchmarking and runtime selection were committed in `e6be9dd`.

The current committed Mistral baseline is:

- Selected model: `mistralai/Mistral-7B-Instruct-v0.3`
- Selected alias: `mistral_7b_instruct_v0p3`
- Selected long-run configuration: `quantization=none`, `attn_implementation=sdpa`, `batch_size=4`
- Best tested quantized fallback: `4bit`, `sdpa`, `batch_size=6`
- `flash_attention_2` was unavailable in the benchmark environment because `flash_attn` was not installed
- Gemma and Llama candidates were blocked by gated-model access in the tested Colab runtime

The interrupted run did **not** reach its first committed long-run checkpoint.

That means:

- there are **no** committed `exp: mistral checkpoint ...` commits on `origin/main`,
- the committed repo contains the smoke run and benchmark artifacts but not the full matched-protocol Mistral run,
- any partial full-run output may exist only in the current Colab filesystem or remote runtime and must be inspected before deciding whether to resume or restart.

The canonical progress log for the interrupted run is:

- [Autonomous run log](../../research/AUTONOMOUS_RUN_LOG.md)
- [Non-Qwen model selection](../../research/NON_QWEN_MODEL_SELECTION.md)
- [L4 optimization notes](../../research/L4_OPTIMIZATION_NOTES.md)

## Mission

You are not here to brainstorm. You are here to finish the next thesis-grade research cycle autonomously.

You must complete **all** of the following, end-to-end, unless you are genuinely blocked by unavailable credentials, unavailable hardware, or an irreducible external dependency:

1. Turn the current stopping rule into a **thesis-ready algorithm box and pseudocode section** directly tied to $\mu_t$ and $T_c$.
2. Continue the selected genuinely non-Qwen family path, which is currently the matched-protocol Mistral 7B run, and only reopen model selection if the current runtime changes or Mistral becomes infeasible.
3. Reuse the committed L4 optimization evidence unless the environment has materially changed, and verify that the resumed full run still matches the selected best stable throughput configuration.
4. Use materially stronger **commit and push checkpointing** so failures, Colab disconnects, or SSH drops do not strand important code or outputs locally.

Do not stop at partial implementation. Continue through code changes, benchmark calibration, real execution, regeneration of reports, and updated written conclusions.

## Non-Negotiable Rules

1. Do not revert to the old README-era DeepSeek-centered narrative.
2. Do not count the existing DeepSeek-R1 distill runs as the new non-Qwen family witness for this objective.
3. Do not rerun the recovered Qwen 7B collection unless artifacts are missing or corrupted.
4. Do not change the scientific protocol unless a code bug or external constraint forces it. If you must change something, document exactly why.
5. Do not claim cross-family robustness unless a **genuinely non-Qwen family** reproduces a materially late corrected boundary under the matched protocol.
6. Do not rely on GPU memory percentage alone to claim optimization. Use throughput, stability, and utilization measurements.
7. Do not leave meaningful milestones uncommitted and unpushed.
8. Configure the Python environment before any Python terminal command.
9. Use workspace tools, repo memory, and real scripts aggressively. Do not fake execution.
10. Continue automatically through successive phases unless blocked by a real external constraint.
11. Start by synchronizing with `origin/main` and reconciling the current worktree before making any edits or launching any run.
12. Do not redo already committed benchmark sweeps or smoke tests unless the runtime, package set, or hardware has changed enough to invalidate the prior choice.

## Mandatory First Actions

Before any new code edit or experiment command, do all of the following in order:

1. `git fetch --all --prune`
2. `git pull --ff-only origin main` if the local branch is behind and the worktree is clean
3. read [Autonomous run log](../../research/AUTONOMOUS_RUN_LOG.md)
4. read [Non-Qwen model selection](../../research/NON_QWEN_MODEL_SELECTION.md)
5. read [L4 optimization notes](../../research/L4_OPTIMIZATION_NOTES.md)
6. inspect whether the current runtime already contains any partial full-run artifacts under:
   - `research/outputs/real_traces_l4_mistral_7b`
   - `research/outputs/real_traces_l4_mistral_7b_checkpointed.log`
7. inspect local git history for any `exp: mistral checkpoint ...` commits

Decision rule:

- If partial Mistral full-run artifacts exist locally, stage and checkpoint them **before** any restart.
- If they do not exist locally and no checkpoint commits exist remotely, treat the full run as not yet recoverable from git and restart it from the selected configuration rather than wasting time searching for missing tracked state.

## What Counts As A Real Upgrade To The Thesis Claim

The claim can only be upgraded from **capability-linked evidence** to something closer to **genuine cross-family robustness** if all of the following hold:

1. The new run is from a **genuinely non-Qwen family**.
2. The run uses the matched protocol or a clearly documented justified variant.
3. The corrected conditional-hazard analysis is used, not the legacy pooled proxy.
4. The new family shows a **materially late** corrected boundary rather than collapsing to step `1`.
5. The new family also shows a meaningful never-stop penalty and a coherent early-helpful / later-harmful transition.
6. The updated cross-family report and open questions still support the upgraded claim after regeneration.

If those conditions are not met, the correct outcome is **not** to force the stronger claim. The correct outcome is to preserve the capability-linked conclusion and document the blocker precisely.

## Hard Deliverables

By the end of this autonomous cycle, the repo should contain all of the following unless a real blocker prevents a specific item:

1. A thesis-ready algorithm box or pseudocode section in the thesis-writing path.
2. A theory-facing algorithm section in the research note.
3. One stronger genuinely non-Qwen family run completed, or a documented blocker after a serious attempt.
4. A committed benchmark or profiling record showing how the L4 configuration was optimized.
5. Regenerated per-run and cross-family artifacts.
6. Updated written conclusions explaining whether the thesis claim strengthened.
7. A persistent autonomous progress log with commit hashes and push status.

## Required Documentation Targets

### Thesis-facing algorithm box

Add a polished algorithm section to at least one thesis-facing document. Preferred targets in order:

1. [Thesis proposal draft](../../ThesisDocs/acm_thesis_proposal_draft.md)
2. A new thesis-facing note under `ThesisDocs/` if the proposal draft is too constrained
3. [Theory note](../../research/overthinking_boundary.md) as a parallel technical version

The algorithm section must explicitly distinguish:

- the latent theoretical boundary $T_c$,
- the plug-in empirical boundary $\widehat{T}_c$ from estimated hazards, and
- any sequentially safe stop rule such as a confidence-bound or e-process based stopping time.

### Required algorithmic content

The algorithm box or pseudocode must use the current research objects directly:

- correctness belief $q_t$,
- repair hazard $\alpha_t$,
- corruption hazard $\beta_t$,
- compute cost $\lambda$,
- continuation value $\mu_t = (1-q_t)\alpha_t - q_t\beta_t - \lambda$,
- boundary $T_c = \inf\{t : \mu_t \le 0\}$.

It must not drift into a generic early-exit heuristic that hides the actual theory.

### Required pseudocode skeleton

Your final write-up should include a polished version of logic equivalent to:

```text
Input: observable trace features Z_1:Z_T, cost lambda, estimator family for q_t, alpha_t, beta_t
Output: stop time tau

for t = 1, 2, ..., T:
    update features from current trace prefix
    estimate q_hat_t = P(C_t = 1 | F_t)
    estimate alpha_hat_t = P(C_{t+1}=1 | C_t=0, F_t)
    estimate beta_hat_t  = P(C_{t+1}=0 | C_t=1, F_t)
    compute mu_hat_t = (1 - q_hat_t) * alpha_hat_t - q_hat_t * beta_hat_t - lambda
    if theorem-facing plug-in rule: stop when mu_hat_t <= 0
    if sequentially safe deployment rule: compute U_t and stop when U_t <= 0

Return tau
```

But do not leave it at this rough level. Convert it into a thesis-quality algorithm box with notation, input assumptions, output interpretation, and a short discussion of the difference between the latent boundary and the deployable estimate.

## Stronger Non-Qwen Family Requirement

This requirement is already partially satisfied in the current repo state.

The harness now supports the selected non-Qwen family alias:

- `mistral_7b_instruct_v0p3` -> `mistralai/Mistral-7B-Instruct-v0.3`

Do **not** reopen this as a generic engineering task unless the relevant code or files are missing.

Only revisit model-selection or model-catalog work if one of the following is true:

- the current runtime now has access to Gemma or Llama and you explicitly want to supersede Mistral,
- Mistral becomes infeasible under the matched instrumentation,
- the existing Mistral harness path is broken in the present environment.

### Current selected non-Qwen family

The current selected candidate is Mistral, not because it is theoretically preferred over Gemma or Llama, but because it was the first genuinely non-Qwen family that was both accessible and benchmarkable in the live Colab runtime.

Reuse this selection by default.

Only fall back to the generic priority order below if you are forced to reopen candidate selection.

### Candidate priority order if reselection is required

Pick the strongest candidate that is both **downloadable** and **practical on a 22.5 GB L4** under instrumentation.

Preferred order:

1. A Gemma-family instruct model that is clearly stronger than the current weak controls and fits with quantization.
2. A Llama-family instruct model if weights and access are available.
3. A Mistral-family instruct model if it is more practical than the above.

Do **not** count a Qwen-derived or Qwen-branded family member as satisfying the new-family requirement.

### Model-selection requirements

Before finalizing the candidate, verify and record:

- exact HF model id,
- family name,
- parameter count,
- whether hidden states and token-level outputs are accessible,
- whether 4-bit or 8-bit loading is supported,
- whether the model fits the L4 with the required instrumentation,
- whether access is gated,
- whether the tokenizer / prompt format is compatible with the current harness.

If the first candidate is blocked by gated access or infeasible memory, fall back automatically to the next candidate and document the reason.

In the currently committed evidence, Gemma and Llama were both blocked by gated access, and Mistral was the first successful fallback.

## Colab L4 Optimization Requirements

You must explicitly verify that the chosen long-run configuration is actually close to the best stable throughput path on the `22.5 GB` L4.

The current committed benchmark conclusion is:

- best overall tested runtime: `quantization=none`, `attn_implementation=sdpa`, `batch_size=4`
- best tested quantized runtime: `4bit`, `sdpa`, `batch_size=6`
- `flash_attention_2` unavailable in the tested environment

Default rule:

- reuse the committed full-precision `sdpa` batch-4 configuration for the matched Mistral full run,
- only rerun the full benchmark sweep if the environment changed materially or if real full-run telemetry contradicts the benchmark results.

### Required optimization workflow

1. Verify hardware with `nvidia-smi` and runtime inspection.
2. Confirm usable VRAM, not just total VRAM.
3. Run a smoke benchmark ladder over batch size for the chosen new family.
4. Measure at least:
   - examples per second,
   - tokens per second,
   - wall-clock per batch,
   - peak GPU memory,
   - whether GPU utilization is consistently low,
   - whether the bottleneck is model compute, hidden-state extraction, CPU preprocessing, or disk writes.
5. Test at least the relevant feasible options for:
   - `4bit` vs `8bit` vs full precision where practical,
   - `sdpa` vs `flash_attention_2` vs `auto` when available,
   - at least several batch sizes rather than a single guess.
6. Choose the final configuration based on measured throughput and stability, not just “highest memory use”.

Additional rule:

- If the committed benchmark note already shows a best stable configuration and the runtime is materially the same, do not waste paid Colab time repeating the full benchmark ladder. Validate the chosen path with a quick sanity check and continue.

### Required decision logic

- If a configuration uses much less than `22.5 GB` but still delivers the best stable throughput, document why that is the optimum.
- If a configuration approaches full memory but is unstable, reject it even if it looks efficient.
- If low memory use is caused by hidden-state extraction or I/O bottlenecks, identify and patch the bottleneck when practical.
- If the best stable configuration is still significantly underutilized, explain why instead of pretending the run is optimized.

### Required profiling artifacts

Write out a reproducible benchmark record, preferably under a new path such as:

- `research/outputs/benchmark_<family>_<size>_l4/`
- `research/outputs/benchmark_<family>_<size>_l4_summary.csv`
- `research/L4_OPTIMIZATION_NOTES.md`

or an equivalently clear tracked location.

The optimization record must justify the final batch size and quantization choice used for the long run.

## Commit And Push Resilience Requirements

You must be much more aggressive and disciplined about version control than a typical one-shot run.

### Create and maintain a durable progress log

Create or update a tracked file such as:

- `research/AUTONOMOUS_RUN_LOG.md`

and record at least:

- current target model,
- chosen configuration,
- benchmark results,
- current run progress,
- last completed task index,
- last commit hash,
- whether push succeeded,
- any resume instructions if the session dies.

### Mandatory checkpoint commits

Commit and push after each of these milestones:

1. Algorithm-box and thesis-doc patch.
2. Model-catalog / harness extension for the new family.
3. Successful smoke test for the new family.
4. L4 optimization benchmark and chosen configuration.
5. Each major long-run checkpoint.
6. Final analysis regeneration.
7. Final narrative and conclusion updates.
8. The first successful partial full-run state if the output directory or checkpoint log exists but no experiment checkpoint commit exists yet.

### Long-run checkpoint cadence

During the long run, do not wait until all 300 tasks finish before checkpointing.

At minimum, commit and push after whichever comes first:

- every `25` tasks,
- every `75` completed runs,
- every completed temperature block,
- any recovery from an OOM or disconnect,
- any change in the chosen runtime configuration,
- any first bootstrap checkpoint where the long-run output directory exists but no experiment checkpoint commit has yet been created.

Do **not** use a first checkpoint boundary as large as `50` tasks if the run is otherwise unprotected. The initial checkpoint must land earlier than that.

Prefer a tracked checkpoint driver or scripted loop over a one-off terminal one-liner, so the checkpoint logic itself is reproducible.

Before the first commit attempt in a fresh Colab runtime:

- verify repo-local git author configuration,
- if missing, set it to the same author identity already used on `main`,
- record that action in the run log.

If push fails, keep the local commit and record the failure in the run log immediately.

## Exact Execution Plan

Follow these phases in order.

### Phase A: Re-read and verify the audited baseline

Tasks:

1. Read repo memory.
2. Read the authoritative result docs.
3. Read the current theory note and thesis proposal.
4. Verify current git status before editing.
5. Pull the latest remote commits before doing anything else if the worktree allows it.
6. Confirm that the current late-boundary witness is still Qwen 7B and that the claim is still capability-linked rather than cross-family robust.
7. Confirm which Mistral milestones are already committed and which are not.

Acceptance criteria:

- You have a clean baseline summary grounded in the tracked repo state.

### Phase B: Turn the stopping rule into a thesis-ready algorithm section

Tasks:

1. Add an algorithm box or pseudocode section tied directly to $\mu_t$ and $T_c$.
2. Make the exposition thesis-ready rather than notebook-style.
3. Distinguish the latent theorem object from the deployable empirical estimator.
4. Explain how the sequentially safe rule differs from the raw plug-in boundary.

Required outputs:

- updated thesis-facing doc,
- updated theory note if needed,
- commit and push.

Acceptance criteria:

- A thesis reader can see the stopping rule as an actual algorithm, not only as prose and equations.

### Phase C: Extend the harness to one genuinely non-Qwen family

This phase is now primarily a verification-and-repair phase, not a blank-slate extension phase.

Tasks:

1. Inspect the current model catalog and runner choices.
2. Verify that `mistral_7b_instruct_v0p3` is still present and functional.
3. Preserve current families and CLI behavior.
4. Keep quantization, prompt mode, resume behavior, and instrumentation coherent.
5. If necessary, add guarded compatibility logic for tokenizer or chat template differences.
6. Only add another family alias if the current Mistral path is blocked or if a stronger non-Qwen candidate is newly accessible.

Acceptance criteria:

- The new family can be invoked end-to-end through the same real-trace harness and Colab runner.

### Phase D: Calibrate the Colab L4 path for the new family

This phase is now a reuse-or-validate phase by default.

Tasks:

1. Read the committed optimization artifacts first.
2. Validate that the current runtime still supports the chosen configuration.
3. Only rerun the full ladder if the environment changed or if the selected configuration now behaves materially differently.
4. If rerunning is unnecessary, record a short validation note and move on.
5. Record any new results in tracked artifacts.

Acceptance criteria:

- You can defend the chosen configuration as the best stable configuration found on the L4.

### Phase E: Run the stronger non-Qwen family under the matched protocol

Match the existing empirical design unless a documented blocker requires a deviation:

- `task_source = gsm8k`
- `dataset_split = train`
- `dataset_shuffle_seed = 17`
- `max_tasks = 300`
- `temperatures = [0.1, 0.6, 1.0]`
- `seeds = [7]`
- `max_steps = 10`
- `max_new_tokens = 256`
- same analysis stack and hazard audit path

Tasks:

1. Inspect whether a partial full-run output directory or checkpoint log already exists in the current runtime.
2. If partial full-run artifacts exist, checkpoint them immediately and resume rather than restarting.
3. If no partial full-run artifacts exist, start the full real-trace experiment from the committed selected configuration.
4. Use a checkpoint plan that lands the first durable experiment commit no later than the `25`-task mark.
5. Checkpoint commits and pushes at the required cadence.
6. Recover automatically from resumable failures.

Implementation preference:

- Do not launch the 300-task run as a single fragile one-liner with the first commit deferred too long.
- Prefer explicit task-increment checkpoints such as `25, 50, 75, ..., 300` per temperature, or an equivalently aggressive scripted checkpoint schedule.

Acceptance criteria:

- Either the full run finishes, or the blocker is real, documented, and backed by logs and code attempts.

### Phase F: Regenerate analysis and update the thesis claim

Tasks:

1. Run `trace_analysis.py` on the new family outputs.
2. Run `generate_thesis_artifacts.py` for the new family.
3. Run the combined cross-family analysis path.
4. Update the relevant Markdown summaries and open-question files.
5. State clearly whether the thesis claim strengthens.

Required updated outputs:

- per-run final results for the new family,
- updated cross-family outputs,
- updated cross-family report,
- updated open questions,
- updated README or thesis-facing summary if the headline claim changes.

Acceptance criteria:

- The repo clearly states whether the new family upgrades the evidence to something stronger than capability-linked late-boundary behavior.

## Scientific Guardrails

Do not overclaim.

If the new family:

- fits poorly,
- never leaves a weak regime,
- collapses to an early corrected boundary,
- or is too unstable to instrument faithfully,

then document that honestly. The output should then reinforce that the thesis has a valid stopping-rule framework and one strong capable-regime witness, but not yet a universal cross-family law.

## Final Required Summary

Do not finish with a vague success note. End with a concrete final summary that includes:

1. Exactly where the algorithm box was added.
2. Which new family was selected and why.
3. The chosen L4 runtime configuration and why it was optimal.
4. The completed run size and whether resume logic was used.
5. The corrected boundary result for the new family.
6. Whether the claim upgraded beyond capability-linked evidence.
7. The list of commit hashes and whether each was pushed successfully.
8. Any remaining blockers or next-best follow-up experiments.

## Definition Of Done

This prompt is complete only when:

- the thesis-ready stopping algorithm is written into the repo,
- the committed Mistral baseline has been either resumed successfully or restarted cleanly from the selected configuration,
- a stronger genuinely non-Qwen family has been seriously executed or blocked with evidence,
- the L4 path has been benchmarked and justified,
- commits and pushes were used as durable checkpoints,
- the written research conclusion has been updated to reflect the new evidence,
- and the repo is left in a reproducible, resumable, clearly documented state.
