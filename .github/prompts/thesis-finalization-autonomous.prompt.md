---
name: "Autonomous Thesis Finalization"
description: "Fully autonomous thesis-finalization run: add a thesis-ready stopping-rule algorithm box tied to mu_t and T_c, run one stronger genuinely non-Qwen family on Colab L4, verify GPU optimization, and checkpoint commits/pushes throughout."
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

## Mission

You are not here to brainstorm. You are here to finish the next thesis-grade research cycle autonomously.

You must complete **all** of the following, end-to-end, unless you are genuinely blocked by unavailable credentials, unavailable hardware, or an irreducible external dependency:

1. Turn the current stopping rule into a **thesis-ready algorithm box and pseudocode section** directly tied to $\mu_t$ and $T_c$.
2. Run **one stronger genuinely non-Qwen family** under the matched protocol if the goal is to upgrade the current claim from capability-linked evidence to genuine cross-family robustness.
3. Verify that the Colab L4 run is **actually optimized** for throughput and stable memory use on the available `22.5 GB` GPU, using measured metrics rather than intuition.
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

The current harness only supports DeepSeek and Qwen aliases. That is not sufficient for this objective.

### You must treat this as a real engineering task

If a genuinely non-Qwen family is not already supported, extend:

- [research/real_trace_experiments.py](../../research/real_trace_experiments.py)
- [tools/run_colab_experiment.py](../../tools/run_colab_experiment.py)

so that at least one stronger genuinely non-Qwen family can run under the same instrumentation.

### Non-Qwen candidate priority order

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

## Colab L4 Optimization Requirements

You must explicitly verify that the chosen long-run configuration is actually close to the best stable throughput path on the `22.5 GB` L4.

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

### Long-run checkpoint cadence

During the long run, do not wait until all 300 tasks finish before checkpointing.

At minimum, commit and push after whichever comes first:

- every `50` tasks,
- every `150` completed runs,
- every completed temperature block,
- any recovery from an OOM or disconnect,
- any change in the chosen runtime configuration.

If push fails, keep the local commit and record the failure in the run log immediately.

## Exact Execution Plan

Follow these phases in order.

### Phase A: Re-read and verify the audited baseline

Tasks:

1. Read repo memory.
2. Read the authoritative result docs.
3. Read the current theory note and thesis proposal.
4. Verify current git status before editing.
5. Confirm that the current late-boundary witness is still Qwen 7B and that the claim is still capability-linked rather than cross-family robust.

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

Tasks:

1. Inspect the current model catalog and runner choices.
2. Add one stronger genuinely non-Qwen family alias.
3. Preserve current families and CLI behavior.
4. Keep quantization, prompt mode, resume behavior, and instrumentation coherent.
5. If necessary, add guarded compatibility logic for tokenizer or chat template differences.

Acceptance criteria:

- The new family can be invoked end-to-end through the same real-trace harness and Colab runner.

### Phase D: Calibrate the Colab L4 path for the new family

Tasks:

1. Benchmark the candidate across a sensible batch-size ladder.
2. Test stable quantization and attention settings.
3. Use measured throughput and memory to choose the final configuration.
4. Record the results in tracked artifacts.

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

1. Run a smoke test.
2. Run the full real-trace experiment.
3. Checkpoint commits and pushes at the required cadence.
4. Recover automatically from resumable failures.

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
- a stronger genuinely non-Qwen family has been seriously executed or blocked with evidence,
- the L4 path has been benchmarked and justified,
- commits and pushes were used as durable checkpoints,
- the written research conclusion has been updated to reflect the new evidence,
- and the repo is left in a reproducible, resumable, clearly documented state.
