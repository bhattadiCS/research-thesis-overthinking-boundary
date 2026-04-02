# GPT-5.4 XHigh Autonomous Continuation Prompt

Use this prompt inside the ResearchThesis workspace when you want GPT-5.4 xhigh to resume the interrupted overthinking-boundary program from recovered checkpoint state, finish the pending second-family run efficiently on Colab L4, and drive the next experiment cycle to completion.

Historical note as of 2026-04-02: recovery of the overnight Colab backup established that the raw Qwen2.5 7B 4-bit collection actually completed all 900 runs across 300 GSM8K tasks. Use this prompt only for downstream recovery, analysis regeneration, or optional follow-up experiments unless the tracked Qwen 7B artifacts are missing again.

Recovered completion state:

- `research/outputs/real_traces_l4_qwen_7b_4bit` now contains 900 runs, 9000 step rows, and 900 hidden-state files.
- The main unfinished work after recovery was downstream analysis and report regeneration, not raw collection.
- `thesis_data_backup_from_overnight.zip` supersedes the earlier incomplete backup for recovery purposes.

## Mission

You are GPT-5.4 xhigh operating inside the `ResearchThesis` VS Code workspace. Your job is not to propose ideas and stop. Your job is to inspect the existing code and artifacts, recover the interrupted `qwen2p5_7b` experiment from the saved partial state, repair any analysis bug that would invalidate theorem-facing claims, optimize the runtime path for the available L4 GPU, build a combined cross-family report, finish the materially important experiment, regenerate the research artifacts, and stop only when the repo contains updated code, updated Markdown outputs, durable checkpointed commits, and a clear empirical answer about whether the late overthinking boundary survives in a stronger second family.

The previous autonomous run was halted mid-execution by an SSH or Colab disconnect. The saved partial data are not disposable scratch output. Treat them as checkpoint state that must be protected, audited, resumed from, and incorporated into the final result rather than restarted from zero.

Do not ask for permission between phases. If a phase succeeds, proceed to the next phase automatically. If a phase fails, debug it and continue. Only stop early if you are genuinely blocked by missing external credentials, unavailable hardware, or a repeated failure that cannot be resolved from inside the workspace.

## Use MCPs Aggressively

Use MCPs and workspace tools heavily rather than freehanding assumptions.

Required behavior:

1. Read repository memory before touching code.
2. Read the current result Markdown files and the exact CSV or JSON artifacts they summarize.
3. Use search or exploration tools to map the relevant code paths before editing.
4. Use web fetches for model-card, quantization, and L4 performance guidance before finalizing the 7B plan.
5. Configure the Python environment before any Python terminal command.
6. Run the real scripts. Do not simulate execution if the workspace can actually run the command.
7. Inspect recovered partial-run directories before launching any long experiment and determine whether the harness can resume without duplicating completed tasks.
8. Audit throughput on the actual Colab L4 hardware using measured runtime metrics rather than GPU-memory intuition alone.
9. Patch resume or checkpoint behavior if the current harness cannot continue safely from the saved state.
10. After each significant milestone, create a git commit and push it if remote/auth are available; if push fails, preserve a local commit and record the blocker explicitly.

## Non-Negotiable Continuation Context

- If `recovered_thesis_data/` exists, treat it as the extracted backup from the interrupted Colab session.
- The default behavior for the Qwen 7B path is resume and complete, not restart.
- Preserve recovered CSV, JSON, PNG, and hidden-state artifacts unless you are certain a file is only cache or duplication.
- If the live repo is missing newer recovered outputs, reconcile them into the working tree before continuing the experiment.

## Workspace Facts You Inherit

### Core files

- `research/real_trace_experiments.py` is the real-trace collection harness.
- `research/trace_analysis.py` computes detector summaries, probe metrics, and hazard summaries.
- `research/generate_thesis_artifacts.py` writes the per-run Markdown reports.
- `tools/run_colab_experiment.py` is the main Colab-safe orchestration wrapper.
- `research/open_questions.md` and `research/open_questions_qwen_l4.md` currently answer questions per-model, not jointly.
- `research/FINAL_L4_RESULTS.md` and `research/FINAL_QWEN_L4_RESULTS.md` are the current per-family top-level summaries.

### Existing completed runs

Treat these as the current empirical baseline. Reuse them. Do not rerun them unless an analysis correction absolutely requires regeneration.

- `research/outputs/real_traces_l4_deepseek_1p5b`
- `research/outputs/real_traces_l4_qwen_0p5b`

Both completed runs are directly comparable on the following dimensions:

- `task_source = gsm8k`
- `dataset_split = train`
- `dataset_shuffle_seed = 17`
- `max_tasks = 300`
- `temperatures = [0.1, 0.6, 1.0]`
- `seeds = [7]`
- `max_steps = 10`
- `max_new_tokens = 256`
- `step_cost = 0.05`
- `prompt_mode = minimal_json`
- `system_prompt_mode = default`

### Interrupted run that must be resumed

- `research/outputs/real_traces_l4_qwen_7b_4bit` is the active second-family run directory.
- The recovered state captured 148 completed runs, 1,480 step rows, 148 hidden-state files, and contiguous task indices `00000` through `00147`.
- Unless the live repo proves otherwise, assume 152 tasks remain to reach the matched 300-task design.
- Never delete or overwrite these partial artifacts. Resume or merge deterministically.

### Colab hardware and efficiency target

- Google Colab Pro single L4 GPU session.
- GPU RAM: 22.5 GB total.
- System RAM: 53.0 GB total.
- Disk: 235.7 GB total.
- The interrupted Qwen 7B 4-bit run appeared to use only about 8 to 10.5 GB of GPU RAM.
- Treat that as likely optimization headroom until profiling proves otherwise.
- Optimize for completed runs per hour and stable continuation, not memory percentage alone, but do not accept low utilization without measuring the bottleneck.

### Existing quantitative state

Use these values as the starting factual state, but do not treat the reported crossing step as final until you audit the hazard computation path.

| Run | Family | Params | Step-1 accuracy | Peak accuracy | Peak step | Reported crossing step | Repair rate overall | Corruption rate overall | Hazard rule gap | E-process gap | Never-stop gap | Probe Brier | Probe AUC | Strongest correctness signal | Strongest corruption signal |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `real_traces_l4_deepseek_1p5b` | DeepSeek-R1 distill | 1.5B | 0.2367 | 0.3044 | 7 | 7 | 0.1887 | 0.4612 | 0.4121 | 0.4441 | 0.7463 | 0.2217 | 0.6137 | `answer_changed` | `answer_changed` |
| `real_traces_l4_qwen_0p5b` | Qwen2.5 instruct | 0.5B | 0.0711 | 0.0822 | 3 | 1 | 0.0029 | 0.0236 | 0.1531 | 0.0595 | 0.4595 | 0.2399 | 0.5291 | `verbose_confidence_proxy` | `entropy_mean` |

### What is already supported jointly

These claims are already supported by the completed data:

1. Extra reasoning is not monotone-helpful under the repo's cost-adjusted utility objective.
2. The boundary location is strongly capability-dependent in the current data: DeepSeek 1.5B shows a later peak than Qwen 0.5B.
3. Detector ranking is not obviously invariant across regimes: the fitted hazard rule is stronger on DeepSeek 1.5B, but the e-process is stronger on Qwen 0.5B.
4. Observable stability across families is not settled: DeepSeek 1.5B is dominated by answer revision signals, while weak Qwen is dominated by verbosity or entropy proxies.
5. Family-specific versus benchmark-specific attribution is still unresolved because the only currently capable family is DeepSeek 1.5B on GSM8K.

## Critical Analysis Caveat You Must Resolve First

Do not blindly reuse the current crossing-step story.

There is a real analysis inconsistency in the current artifact stack:

1. `research/trace_analysis.py` writes `hazard_drift_summary.csv` using pooled stepwise means of `repair` and `corruption` over all transitions.
2. That construction is not the same as the conditional hazards $\alpha_t = P(C_t=0,C_{t+1}=1)/(1-q_t)$ and $\beta_t = P(C_t=1,C_{t+1}=0)/q_t$.
3. `research/generate_thesis_artifacts.py` currently reads `hazard_drift_summary.csv` for zero-crossing claims.
4. In the saved artifacts, `hazard_by_step.csv` and `hazard_drift_summary.csv` do not tell the same story for DeepSeek 1.5B.
5. Therefore, any theorem-facing claim about a step-7 drift crossing must be audited before being treated as the final witness.

You must explicitly reconcile these files before making strong claims:

- `research/outputs/real_traces_l4_deepseek_1p5b/hazard_by_step.csv`
- `research/outputs/real_traces_l4_deepseek_1p5b/hazard_drift_summary.csv`
- `research/outputs/real_traces_l4_qwen_0p5b/hazard_by_step.csv`
- `research/outputs/real_traces_l4_qwen_0p5b/hazard_drift_summary.csv`

If the current report wording overstates the certainty of the step-7 theorem witness, correct it or add an explicit erratum.

## Theory You Must Keep Fixed

Do not drift into a different research program.

Keep this layered structure:

1. Core theory: semimartingale drift-sign or hazard continuation model.
2. Operational layer: correctness belief plus repair and corruption hazards.
3. Safety layer: anytime-valid or at least sequentially valid detector logic.
4. Auxiliary observables: entropy, answer revision, hidden-state drift, lexical echo, confidence proxies.

Do not replace the hazard model with a pure hidden-state or pure entropy story. Hidden states and entropy are observables, not the primary theoretical object.

## Primary Objective Order

Complete these in order.

### Objective 1: Repair the theorem-facing analysis path

Audit and fix the hazard computation or report path so that the Markdown outputs do not confuse unconditional transition frequencies with conditional hazards.

Minimum acceptable outcome:

1. Either `trace_analysis.py` is corrected to output conditional hazards explicitly, or a new analysis path is added that does so.
2. `generate_thesis_artifacts.py` no longer bases boundary claims on a known-misaligned hazard summary.
3. The repo contains a reproducible artifact that distinguishes:
   - empirical utility drift,
   - conditional hazard-based drift,
   - any fitted-model drift estimate.
4. If the corrected crossing step for DeepSeek 1.5B differs materially from the current report language, document that change.

### Objective 2: Add a combined cross-family reporting path

Build a reproducible script, not a one-off notebook, that aggregates multiple completed run directories into a joint report.

Preferred implementation:

- Add a new script such as `research/cross_family_analysis.py`.

Alternative acceptable implementation:

- Extend `research/generate_thesis_artifacts.py` with a multi-input mode, but only if the code stays readable.

Required inputs per run:

- `pilot_summary.csv`
- `detector_comparison.csv`
- `correctness_probe_metrics.csv`
- `feature_weights.csv`
- `metadata.json`
- `trace_steps.csv`
- corrected hazard artifact or corrected recomputation from `trace_steps.csv`

Required combined outputs:

- `research/outputs/cross_family/cross_family_summary.csv`
- `research/outputs/cross_family/cross_family_detector_comparison.csv`
- `research/outputs/cross_family/cross_family_signal_summary.csv`
- `research/outputs/cross_family/cross_family_boundary_comparison.png`
- `research/outputs/cross_family/cross_family_detector_gaps.png`
- `research/CROSS_FAMILY_OPEN_QUESTIONS.md`
- `research/CROSS_FAMILY_REPORT.md`

The combined report must answer the existing open questions jointly, not copy-paste per-model answers.

### Objective 3: Recover and harden the interrupted Qwen 7B continuation path

The primary next run is:

- model: `qwen2p5_7b`
- quantization: `4bit`
- output dir: `research/outputs/real_traces_l4_qwen_7b_4bit`
- smoke output dir: `research/outputs/real_traces_l4_qwen_7b_4bit_smoke`

This is not a fresh run. It is an interrupted run that must be safely resumed from the saved partial state.

Why this is the primary next run:

1. It is a true second-family capability follow-up, unlike DeepSeek 7B which is within-family scaling.
2. It is already partially complete, so resuming it preserves recovered empirical work and saves paid Colab time.
3. It directly tests whether a later boundary emerges in a more capable Qwen model after the weak Qwen 0.5B control failed to leave a low-skill regime.
4. It preserves the benchmark and protocol while changing capability and family.
5. The current harness already supports `qwen2p5_7b` and quantization modes, so resume support should be the first engineering target.

### Objective 4: Audit and optimize the Colab L4 execution path

You are not only trying to finish the experiment. You are trying to finish it efficiently on paid Colab hardware.

Minimum acceptable outcome:

1. The code path has been audited for bottlenecks in loading, batching, generation, hidden-state extraction, and disk writes.
2. The chosen batch size is justified by measured throughput and GPU memory on the L4.
3. The repo contains either a real performance patch or a documented explanation of why a larger steady-state batch does not help.
4. The long Qwen 7B continuation run uses the highest stable throughput configuration the code can support without changing the scientific protocol.

### Objective 5: Optional within-family scaling after the second-family run

If Qwen 7B completes and there is still time or budget, run:

- model: `deepseek_r1_distill_7b`
- quantization: `4bit`
- separate output dir: `research/outputs/real_traces_l4_deepseek_7b_4bit`

This is a secondary experiment. Do not do it first. The main unresolved thesis gap is cross-family, not only within-family scaling.

## Exact Execution Rules

### Before coding

1. Read repo memory.
2. Read the current per-family result docs and open-question docs.
3. Read the recovered partial Qwen 7B outputs in the live repo and, if present, the `recovered_thesis_data` copies.
4. Verify the current resume point from `trace_runs.csv`, `trace_steps.csv`, `metadata.json`, and hidden-state filenames before launching any continuation run.
5. Read the current output CSVs for both completed baseline runs.
6. Read the relevant code paths in `real_trace_experiments.py`, `trace_analysis.py`, `generate_thesis_artifacts.py`, and `tools/run_colab_experiment.py`.

### Before any Python terminal command

Configure the Python environment and use the returned executable path rather than assuming `python` will resolve correctly.

### Editing rules

1. Make the smallest coherent root-cause fix.
2. Do not overwrite unrelated outputs.
3. Use new output directories for new experiments.
4. Do not quietly change the primary experiment design parameters.
5. If you add new analysis files, keep them script-based and reproducible.
6. Do not silently restart interrupted experiments when resume is possible.
7. Keep scientific parameters fixed; treat performance tuning and checkpointing as execution-path changes, not protocol changes.

### Version-control checkpointing

Create git commits and push them at meaningful milestones. At minimum, checkpoint after:

1. the hazard-analysis fix,
2. the cross-family analysis generator and outputs,
3. any resume or crash-safety patch,
4. any L4 performance optimization patch and green smoke benchmark,
5. any durable long-run milestone for Qwen 7B, especially if the run finishes or crosses a major completion boundary,
6. the final analysis and report regeneration.

Use non-interactive git commands. Do not rewrite history unless explicitly instructed. If push fails because of auth or remote issues, keep the local commit and record the blocker clearly.

## Detailed Phase Plan

## Phase A: Audit and reconcile the hazard path

Tasks:

1. Compare the conditional hazard construction in `hazard_by_step.csv` against the fitted or pooled hazard construction in `hazard_drift_summary.csv` for both completed runs.
2. Decide which quantities correspond to:
   - raw empirical utility drift,
   - conditional hazard drift,
   - fitted-model hazard drift.
3. Patch `trace_analysis.py` and `generate_thesis_artifacts.py` so these are not conflated.
4. Regenerate the relevant per-run Markdown artifacts if needed.

Acceptance criteria:

1. A reader can tell exactly which drift curve is being cited.
2. The code no longer uses an unconditional proxy as if it were the conditional theorem object.
3. DeepSeek 1.5B and Qwen 0.5B reports are either corrected or explicitly caveated.

## Phase B: Implement the cross-family report generator

The report must not be a narrative only. It must emit combined structured data.

Mandatory combined tables:

1. Per-run summary table with family, params, backend, quantization, step-1 accuracy, peak accuracy, peak step, corrected boundary step, repair rate, corruption rate, oracle gap for hazard rule, oracle gap for e-process, oracle gap for never-stop, probe Brier, probe AUC.
2. Detector ranking table by run and by family.
3. Signal comparison table showing the top correctness-side and corruption-side observables for each run.
4. A joint answer table for the open questions that explicitly marks each question as:
   - answered,
   - partially answered,
   - still unresolved.

Mandatory combined figures:

1. Overlay of correctness trajectories `q_t` by run.
2. Overlay of corrected drift curves by run.
3. Bar chart of corrected boundary step by run.
4. Detector gap comparison by run.

Mandatory Markdown outputs:

1. `research/CROSS_FAMILY_REPORT.md`
2. `research/CROSS_FAMILY_OPEN_QUESTIONS.md`

The joint open-question document must answer the repo's existing questions in a cross-family way. It must explicitly discuss:

1. whether boundary existence is robust across model families,
2. whether boundary location appears capability-linked,
3. whether detector ranking changes with capability,
4. whether answer revision or entropy is more cross-family stable,
5. whether the current data support a family effect or merely a capability effect,
6. what cannot yet be claimed without the 7B second-family run.

## Phase C: Recover and harden the interrupted Qwen 7B continuation path

Do not launch a new long run until the saved partial outputs have been reconciled and the continuation logic is safe.

Tasks:

1. Inspect `research/outputs/real_traces_l4_qwen_7b_4bit` in the live repo and compare it with any recovered copy under `recovered_thesis_data`.
2. Verify the exact continuation point from `trace_runs.csv`, `trace_steps.csv`, `metadata.json`, and hidden-state filenames. The recovered audit found 148 completed runs covering task indices `00000` through `00147`; treat that as the default checkpoint unless the live repo shows more progress.
3. Patch `tools/run_colab_experiment.py`, `research/real_trace_experiments.py`, or both if necessary so the run can skip already completed tasks and append only missing tasks.
4. Ensure writes are incremental and crash-safe enough that a disconnect loses at most the current in-flight batch, not the whole run.
5. Preserve the existing output directory rather than creating a fresh duplicate unless a merge-safe recovery workflow is strictly necessary.
6. Commit and push the continuation hardening work before the long run.

Acceptance criteria:

1. The agent can state the exact next task index before launching the continuation.
2. Restarting the run does not duplicate rows in `trace_runs.csv` or `trace_steps.csv`.
3. Partial hidden-state artifacts are preserved and new ones append cleanly.
4. The repo has a durable git checkpoint for this phase, or a concrete push blocker is documented.

## Phase D: Audit and optimize the L4 execution path

Do not treat 8 to 10.5 GB used on a 22.5 GB L4 as automatically acceptable. Investigate it.

Tasks:

1. Inspect the model-loading, batching, generation, hidden-state extraction, tokenization, and output-writing code paths for avoidable bottlenecks.
2. Run measured smoke or mini-benchmark passes on the actual Colab L4 and capture wall-clock throughput, batch latency, peak GPU memory, and if possible GPU utilization.
3. Start with the current conservative full batch size target of `4`, but actively test larger steady-state values such as `6` and `8` if memory and stability allow.
4. Strengthen quantized loading with NF4, double quant, and `bfloat16` compute on the L4 when supported.
5. Prefer the fastest supported attention backend, such as `sdpa` or `flash_attention_2`, if it is compatible with the chosen model path.
6. Remove obvious CPU or I/O bottlenecks, especially redundant post-generation passes, serialization stalls, or repeated tokenizer or model setup work.
7. Record the chosen configuration and commit and push the optimization patch before the long continuation run.

Acceptance criteria:

1. The final long-run configuration is justified by actual measurements rather than guesswork.
2. The repo contains either a material throughput improvement or a clear explanation of the limiting factor.
3. Scientific protocol fields remain unchanged.
4. The repo has a durable git checkpoint for this phase, or a concrete push blocker is documented.

## Phase E: Resume and complete the Qwen 7B quantized run

Do not start the long continuation until the smoke test is green, the resume path is verified, and the L4 configuration has been chosen.

### Default protocol for the matched second-family experiment

Keep these scientific settings fixed unless debugging demands otherwise:

- `task_source = gsm8k`
- `dataset_split = train`
- `dataset_shuffle_seed = 17`
- `max_tasks = 300`
- `max_steps = 10`
- `max_new_tokens = 256`
- `temperatures = [0.1, 0.6, 1.0]`
- `seeds = [7]`
- `prompt_mode = minimal_json`
- `system_prompt_mode = default`
- `quantization = 4bit`
- full batch size should be treated as a tunable execution parameter, not a fixed scientific constant
- initial calibration start = `4`
- preferred steady-state target = the highest stable value found during profiling
- smoke batch size start = `1`

Rationale:

1. This preserves comparability with the completed runs.
2. The harness already has recursive OOM recovery by splitting prompt batches.
3. The model still performs a full hidden-state forward pass after generation, so batch size should start conservative and then be re-tuned using measurements.
4. This workspace is running on an L4 with 22.5 GB VRAM, so leaving large headroom unused without checking throughput is not acceptable.

### Default command pattern

Use the configured Python executable and first confirm that the existing output directory can be resumed safely. Then run a smoke test and continue the full experiment in the existing output directory rather than starting from zero.

Suggested full-run command shape:

```bash
<PYTHON> tools/run_colab_experiment.py \
  --model qwen2p5_7b \
  --quantization 4bit \
  --task-source gsm8k \
  --smoke-task-source builtin \
  --dataset-split train \
  --dataset-shuffle-seed 17 \
  --prompt-mode minimal_json \
  --system-prompt-mode default \
  --full-max-tasks 300 \
  --full-max-steps 10 \
  --full-max-new-tokens 256 \
  --full-temperatures 0.1 0.6 1.0 \
  --full-seeds 7 \
  --full-batch-size 4 \
  --smoke-batch-size 1 \
  --output-dir research/outputs/real_traces_l4_qwen_7b_4bit \
  --smoke-output-dir research/outputs/real_traces_l4_qwen_7b_4bit_smoke
```

The execution path must be resume-aware. If the CLI does not currently support resume semantics, add them before launching the long run. Minimum behavior: detect the 148 completed `run_id` values already present, skip them, and continue from task index `148` without duplicating rows or hidden-state files.

### Quantization guidance you should use if the loader needs improvement

The current loader supports `4bit` and `8bit`, but it is minimal. If quantized loading is unstable, patch it cleanly.

Preferred 4-bit fallback configuration:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
)
```

Preferred 8-bit fallback configuration if 4-bit fails unexpectedly:

```python
BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
)
```

Use `device_map="auto"` for quantized inference, prefer `bfloat16` compute on the L4 when supported, and use the fastest compatible attention backend.

### Fallback ladder for the Qwen 7B run

If the first smoke attempt fails, follow this order:

1. Resume from the saved partial directory first. Do not restart from zero unless resume is genuinely impossible.
2. Benchmark and, if beneficial, raise steady-state full batch size from `4` to `6`, then `8`, and higher only if the measurements support it.
3. If higher batch sizes are unstable, back off from `8` to `6`, then `4`, then `2`, then `1`.
4. If the loader is suboptimal or fails, patch the quantization config to the stronger NF4 or double-quant version.
5. If the attention backend is the bottleneck, switch to the fastest supported backend.
6. If 4-bit still fails unexpectedly, try 8-bit with CPU offload.
7. If Qwen 7B is irrecoverably blocked, document the reason and only then switch to `deepseek_r1_distill_7b` as a fallback experiment, making clear that this is not the primary second-family answer.

## Phase F: Validate the new run correctly

Once the new run completes:

1. Run `trace_analysis.py` on the new output directory.
2. Regenerate per-run Markdown artifacts for the new model.
3. Check that the standard artifact set exists:
   - `trace_steps.csv`
   - `trace_runs.csv`
   - `pilot_summary.csv`
   - `metadata.json`
   - `hazard_by_step.csv`
   - corrected hazard summary artifact
   - `detector_comparison.csv`
   - `correctness_probe_metrics.csv`
   - `feature_weights.csv`
   - `real_trace_detector_gaps.png`
   - `real_trace_hazard_curves.png`
   - `real_trace_feature_weights.png`
4. Confirm that the task IDs in `metadata.json` align with the previously completed runs. If they do not, explain the mismatch and whether it affects comparability.
5. Confirm that the resumed output contains no duplicated `run_id` values and that the completed task range now reaches the intended 300-task endpoint.
6. Record the final chosen batch size and any relevant throughput or memory observations in the write-up.

## Phase G: Decide whether the late boundary survives

Do not collapse several different hypotheses into one.

Use these distinct labels:

1. `Exact step-7 replication`
   - corrected boundary step is exactly 7.
2. `Late-boundary replication`
   - corrected boundary step is in the range 5 to 9 and the accuracy peak is also late.
3. `Weakened late-boundary support`
   - corrected boundary step is later than 2 but outside 5 to 9.
4. `No late-boundary replication`
   - corrected boundary step is 1 or 2, or the model never exits the low-skill regime.

Use these capability gates before claiming the model left the low-skill regime:

1. `step-1 accuracy >= 0.15`, or
2. `runs_ever_correct >= 250` out of 900 total runs.

If neither threshold is met, say clearly that the second-family follow-up still failed to reach a sufficiently capable regime for a strong cross-family boundary claim.

Use these practical overthinking gates before claiming harmful late reasoning:

1. `never_stop` mean oracle gap must be materially positive, preferably `> 0.15`.
2. The peak correctness step must be later than step 1.
3. There must be evidence of both repair and corruption transitions, even if estimated conservatively.

## Phase H: Update the combined report after the new run

Once the new run is analyzed, rerun the combined cross-family reporting script using:

- `research/outputs/real_traces_l4_deepseek_1p5b`
- `research/outputs/real_traces_l4_qwen_0p5b`
- `research/outputs/real_traces_l4_qwen_7b_4bit`

Then rewrite:

- `research/CROSS_FAMILY_REPORT.md`
- `research/CROSS_FAMILY_OPEN_QUESTIONS.md`

The updated report must explicitly answer:

1. Is the boundary clearly present across both DeepSeek 1.5B and Qwen 7B, or only within DeepSeek so far?
2. Is the original step-7 observation best interpreted as a DeepSeek-specific effect, a capability effect, or still unresolved?
3. Does the strongest observable remain answer revision in the higher-capability second family, or does entropy or hidden-state drift become more stable?
4. Does the e-process remain better in weaker regimes while hazard drift catches up in stronger regimes?
5. What still needs to be run to separate family from benchmark effects?

## Optional Extension: DeepSeek 7B scaling run

Only do this after the Qwen 7B path is complete or irrecoverably blocked.

If you run DeepSeek 7B, start with the matched default protocol in a new directory:

- `research/outputs/real_traces_l4_deepseek_7b_4bit`

Important constraint:

DeepSeek's model card recommends temperature near `0.6` and no system prompt, but do not change the primary matched protocol for the first scaling run if your goal is comparability with the existing DeepSeek 1.5B result. If you want to test the model-card recommendation, do it as a separate ablation in a separate directory after the matched run.

## Definition of Done

Do not stop until all of the following are true:

1. The hazard-analysis mismatch has been repaired, isolated, or explicitly documented in a reproducible way.
2. The repo contains a combined cross-family analysis script and reproducible combined output artifacts.
3. The recovered partial Qwen 7B state has been protected and the continuation path is resume-safe.
4. The code path has been audited for L4 efficiency, and any material optimization has been applied or the limiting bottleneck documented with evidence.
5. The Qwen 7B quantized second-family run has either been resumed to completion or has been shown to be genuinely blocked after a serious fallback ladder.
6. If the run completed, the combined report has been regenerated with the new family included.
7. The joint open-question document has been updated with actual numbers rather than placeholders.
8. Significant milestones have git commits and push attempts, with any push blocker documented explicitly.
9. The final write-up distinguishes clearly between:
   - current evidence,
   - corrected evidence after the hazard audit,
   - still unresolved claims.

## Final Reporting Standard

When you finish, your final summary must contain:

1. What code changed.
2. What resume or checkpointing changes were added.
3. What performance changes were made, what batch size was ultimately used, and what throughput or GPU-memory effect was observed.
4. What new artifacts were created.
5. Whether the corrected analysis still supports a late overthinking boundary for DeepSeek 1.5B.
6. Whether Qwen 7B shows exact step-7 replication, late-boundary replication, weakened support, or no replication.
7. Whether the Qwen 7B run resumed successfully from the recovered partial state and how much of the 300-task design was completed.
8. Which git commits were created and pushed, or why push was blocked.
9. What the strongest cross-family conclusion is right now.
10. What the single most important next experiment would be after this cycle.

This prompt is intentionally strict. Work to completion.
