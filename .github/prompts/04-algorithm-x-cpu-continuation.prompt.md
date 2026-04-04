---
name: "Algorithm X: CPU Continuation - Truth Audit, Equation Decision, Local Synthesis"
description: "Continue Algorithm X locally on CPU after the interrupted GPU frontier session. Audit what actually changed, reconcile report-vs-code drift, decide whether the stopping equation or algorithm should be promoted, parse any additional local trace data, and produce thesis-grade status artifacts without fabricating frontier validation."
agent: "GPT-5.4 xhigh"
---

# MISSION: CPU-ONLY CONTINUATION OF ALGORITHM X

You are continuing an interrupted multi-session research program. You are not starting from scratch, and you are not allowed to bluff completion. Your job is to determine what is true in this local workspace, what is only true in prior conversation history, and what remains scientifically open.

This pass is CPU-only. Do not attempt GPU inference, Colab orchestration, Hugging Face model downloads, or any frontier trace collection. Use the local repository, the existing reports, and the existing trace artifacts.

## 1. CRITICAL TRUTHS YOU MUST TREAT AS THE STARTING POINT

### 1.1 What Changed vs What Did Not

You must answer these three questions explicitly in your final report and in any user-facing summary:

1. Did we change the math equation?
2. Did we change the algorithm?
3. Did we parse more data?

The current local evidence implies the following:

- Yes, the recommended math equation changed at the analysis level.
- No, the default deployed algorithm has not clearly been switched everywhere in the local codebase.
- Yes, more data were parsed locally, but only from existing legacy traces plus small frontier smoke traces. No full new frontier validation set is present here.

### 1.2 Current Best-Available Scientific Status

- The legacy equation sweep was completed on 4 legacy families:
  - `research/outputs/real_traces_l4_deepseek_1p5b`
  - `research/outputs/real_traces_l4_mistral_7b`
  - `research/outputs/real_traces_l4_qwen_0p5b`
  - `research/outputs/real_traces_l4_qwen_7b_4bit`
- That coverage equals 3,600 runs and 36,000 steps.
- A frontier smoke-only validation exists for:
  - `research/outputs/real_traces_colab_smoke_gemma_4_e4b_it`
  - `research/outputs/real_traces_colab_smoke_qwen_3p5_9b`
- The smoke traces prove systems viability only. They do not prove frontier generalization.

### 1.3 Equation vs Algorithm Status

The local reports establish a three-way distinction that must not be collapsed:

- Current default Phase 1 intake still points to `quadratic_top4`.
- Best theorem-preserving hazard equation is now:
  - `hazard_quadratic_combo_entropymean_entropystd_confidence_thoughttokencount`
- Best empirical stop rule overall is now:
  - `direct_drift_ridge_top4`

Interpretation:

- The equation recommendation changed.
- The algorithmic recommendation widened.
- The local deployed default does not yet appear fully updated to match the new recommendation.

### 1.4 The Key Metrics You Must Preserve Accurately

From `research/reports/equation_analysis_report.md`:

- `direct_drift_ridge_top4`
  - mean test AUC: `0.5890`
  - boundary within `+/- 1`: `0.5358`
  - mean oracle gap: `0.3216`
- `hazard_quadratic_combo_entropymean_entropystd_confidence_thoughttokencount`
  - mean test AUC: `0.6151`
  - boundary within `+/- 1`: `0.5111`
  - mean oracle gap: `0.2512`
- current baseline `hazard_quadratic_top4`
  - mean test AUC: `0.6479`
  - boundary within `+/- 1`: `0.3986`
  - mean oracle gap: `0.3287`

Do not misread these metrics. Higher mean test AUC did not correspond to the best stopping behavior. Boundary accuracy and oracle gap matter.

### 1.5 Frontier Progress So Far

From the local smoke frontier artifacts and reports:

- Gemma 4 Edge 4B smoke: valid hidden states, non-zero L2 shift, parse success `0.75`, efficiency gain vs `never_stop` = `0.00%`
- Qwen 3.5 9B smoke: valid hidden states, non-zero L2 shift, parse success `0.00`, efficiency gain vs `never_stop` = `0.00%`
- Canonical full frontier report still says the required run directories are missing.
- Therefore the universal frontier claim remains unproven in this local workspace.

## 2. HARD CONSTRAINTS

1. CPU only. No Colab. No CUDA. No model loading unless it is lightweight and strictly CPU-safe.
2. Do not fabricate frontier completion.
3. Do not silently promote a new stopping rule without updating the corresponding metadata, reports, and rationale.
4. Treat report-vs-code drift as a first-class problem.
5. Use the local workspace state as ground truth, not the prior chat summary alone.

## 3. FIRST ACTIONS - MANDATORY

Before any edits, perform a truth audit across these files:

- `.github/prompts/02-algorithm-x-gpu-validation.prompt.md`
- `.github/prompts/03-phase2-frontier-autonomous.prompt.md`
- `research/reports/equation_analysis_report.md`
- `research/reports/phase2_final_report.md`
- `research/reports/frontier_phase2_execution_status_2026-04-04.md`
- `research/reports/frontier_validation_report.md`
- `research/reports/frontier_smoke_validation_report.md`
- `research/outputs/universal_feature_analysis/universal_hazard_model_metadata.json`
- `research/real_trace_experiments.py`
- `tools/run_colab_experiment.py`
- `research/frontier_validation_report.py`

You are specifically checking whether the local code actually reflects the claims made in the local reports.

## 4. PRIMARY CPU TASKS

### TASK A: Workspace Truth Audit

Produce:

- `research/reports/cpu_truth_audit_2026-04-04.md`

This report must contain a table with these columns:

- `claim`
- `supported_by_local_artifact`
- `implemented_in_local_code`
- `status`
- `required_action`

You must include, at minimum, these claims:

1. `qwen_3p5_9b` replaced stale Qwen 3.5 aliases everywhere.
2. `llama_3p1_8b_instruct` replaced stale Llama 4 8B assumptions everywhere.
3. The frontier validator defaults target the feasible corrected run set.
4. The deployed default model is still `quadratic_top4`.
5. The equation recommendation changed, but the pipeline default did not necessarily change.
6. The frontier result is still smoke-only in this workspace.

### TASK B: Additional Local Data Discovery

Search these roots for any completed or partially completed run directories that were not included in the current reports:

- `research/outputs`
- `recovered_overnight_data`
- `recovered_thesis_data`

Search criteria:

- directories containing `trace_steps.csv`
- directories containing `trace_runs.csv`
- directories containing `hidden_states`
- directories whose names match `real_traces_colab_*`, `real_traces_l4_*`, or other obvious frontier aliases

If you discover additional completed runs:

- integrate them into the CPU analyses
- update the relevant reports
- explicitly state what new data were added

If you do not discover additional completed runs:

- state that no additional completed frontier traces exist locally
- list the exact missing directories that still block the universal frontier claim

### TASK C: Equation Promotion Decision

Produce:

- `research/reports/equation_promotion_decision.md`

This report must compare three states:

1. `quadratic_top4` as the current default
2. `hazard_quadratic_combo_entropymean_entropystd_confidence_thoughttokencount` as the best theorem-preserving hazard equation
3. `direct_drift_ridge_top4` as the best empirical stopping rule

You must answer:

- Should the default remain unchanged for now?
- Should the best hazard equation replace the default metadata and reporting baseline?
- Should the direct drift rule be exposed only as an experimental comparator?

Do not silently overwrite the baseline. If you promote anything, propagate it coherently through metadata, reports, and evaluation entrypoints.

### TASK D: Parse-Success Audit

Produce:

- `research/reports/parse_success_audit.md`

Investigate at least these anomalies:

- DeepSeek 1.5B legacy parse success is extremely low.
- Qwen 3.5 9B smoke parse success is `0.00` despite correct answers and valid hidden traces.

Determine whether the failure is:

- answer extraction logic
- prompt format mismatch
- stop/confidence extraction mismatch
- trace schema issue
- genuine model formatting failure

### TASK E: CPU Re-analysis

Rerun the CPU-safe analysis stack as needed.

Required commands:

```powershell
python research\universal_feature_analysis.py --random-state 7
python research\equation_analysis.py --random-state 7
```

If additional completed frontier run directories are found, rerun:

```powershell
python research\equation_analysis.py --random-state 7 --frontier-run-dirs <DIR1> <DIR2> ...
python research\frontier_validation_report.py --run-dirs <DIR1> <DIR2> ...
```

If only the smoke frontier directories are present, you may use them only for a smoke-labeled appendix or sanity check. Do not convert smoke evidence into a universal frontier claim.

### TASK F: Status Synthesis

Update or create a thesis-grade CPU status memo that explicitly answers:

1. Did the math equation change?
2. Did the algorithm change?
3. Did we parse more data?
4. How much did we progress toward the research goal?

The answer must distinguish between:

- analytical progress
- deployed pipeline changes
- data coverage progress
- still-open validation gaps

## 5. LOCAL WINDOWS COMMANDS YOU MAY NEED

Use these commands if helpful.

### 5.1 Find Candidate Run Directories

```powershell
Get-ChildItem research\outputs,recovered_overnight_data,recovered_thesis_data -Recurse -Directory |
  Where-Object {
    (Test-Path (Join-Path $_.FullName 'trace_steps.csv')) -or
    (Test-Path (Join-Path $_.FullName 'trace_runs.csv')) -or
    (Test-Path (Join-Path $_.FullName 'hidden_states'))
  } |
  Select-Object -ExpandProperty FullName
```

### 5.2 Regenerate the Smoke Frontier Report Explicitly

```powershell
python research\frontier_validation_report.py `
  --run-dirs research\outputs\real_traces_colab_smoke_gemma_4_e4b_it research\outputs\real_traces_colab_smoke_qwen_3p5_9b `
  --report-path research\reports\frontier_smoke_validation_report.md `
  --summary-path research\reports\frontier_smoke_validation_summary.csv `
  --integrity-path research\reports\frontier_smoke_validation_integrity.csv
```

## 6. WHAT YOU MUST NOT DO

1. Do not say frontier validation succeeded unless full corrected run directories actually exist and pass validation.
2. Do not equate smoke traces with statistical validation.
3. Do not say the algorithm changed everywhere if only the analysis recommendation changed.
4. Do not overwrite `quadratic_top4` silently without documenting the promotion decision.
5. Do not assume prior conversation patches were applied locally just because reports mention them.

## 7. DEFINITION OF DONE

You are done only when all of the following are true:

1. The local workspace truth audit exists and clearly separates report truth from code truth.
2. The repo explicitly states whether the equation changed, whether the algorithm changed, and whether more data were parsed.
3. Any newly discovered local run data are integrated into the CPU analyses.
4. The equation-promotion decision is documented and technically justified.
5. Parse-success failure modes are diagnosed.
6. The final CPU status memo explains how far Phase 2 progressed and what still blocks universal validation.

## 8. BOTTOM LINE

This CPU session is not about collecting new frontier traces. It is about forcing the repository into epistemic consistency.

You are reconciling:

- what the reports say
- what the code actually does
- what the data actually cover
- what the thesis can honestly claim today

Proceed until the local repository tells one coherent story.