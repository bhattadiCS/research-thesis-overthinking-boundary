# Recovery Audit 2026-04-02

## Scope
This audit reconciles the repository state across:

- `origin/main` after the overnight pushed commits,
- `thesis_data_backup_from_incomplete_run.zip`, and
- `thesis_data_backup_from_overnight.zip`.

The goal was to recover any zip-only records, regenerate the missing downstream artifacts, and determine whether the current experiment cycle is complete.

## Recovery Findings
- The earlier incomplete backup contributed no unique files beyond the overnight backup.
- The overnight backup captured `origin/main` at `bbb9778` plus uncommitted run-state and analysis deltas.
- The main unique recovery target was the completed raw Qwen 7B directory `research/outputs/real_traces_l4_qwen_7b_4bit`.
- The live repo after rebasing onto `origin/main` contained only 372 committed Qwen 7B runs. The overnight backup contained the full 900-run raw collection.
- The overnight backup also preserved useful L4 tuning artifacts: small benchmark trace folders, `real_traces_l4_qwen_7b_4bit_resume_bs16.log`, `real_traces_l4_qwen_7b_4bit_resume_bs24.log`, and an uncommitted CUDA OOM recovery patch in `research/real_trace_experiments.py`.

## Records Kept
- Restored the full `research/outputs/real_traces_l4_qwen_7b_4bit` directory with 900 runs, 9000 step rows, and 900 hidden-state files.
- Preserved `research/outputs/real_traces_l4_qwen_7b_4bit_smoke` as a known-good smoke reference.
- Preserved benchmark directories `benchmark_qwen7b_bs4`, `benchmark_qwen7b_bs6`, `benchmark_qwen7b_bs6_clean`, `benchmark_qwen7b_bs8_clean`, `benchmark_qwen7b_bs12_clean`, `benchmark_qwen7b_bs16_clean`, `benchmark_qwen7b_bs24_clean`, and `benchmark_qwen7b_bs32_clean` as small L4 performance calibration traces.
- Preserved `research/outputs/real_traces_l4_qwen_7b_4bit_resume_bs16.log` and `research/outputs/real_traces_l4_qwen_7b_4bit_resume_bs24.log` as runtime evidence for stable and unstable batch-size regimes.
- Applied the recovered `research/real_trace_experiments.py` patch that adds explicit CUDA cache release and a one-time single-prompt retry after OOM.

## Discardable Items
- Local archive files such as `thesis_data_backup_from_*.zip` after recovery.
- Extracted sandbox directories such as `recovered_thesis_data/` and `recovered_overnight_data/` once the restored records are committed.
- Sandbox-local `.git` histories and `__pycache__` directories.

## Analysis Regeneration
After restoring the raw Qwen 7B run, the following were regenerated:

- `research/outputs/real_traces_l4_qwen_7b_4bit/correctness_probe_metrics.csv`
- `research/outputs/real_traces_l4_qwen_7b_4bit/detector_comparison.csv`
- `research/outputs/real_traces_l4_qwen_7b_4bit/detector_comparison_by_run.csv`
- `research/outputs/real_traces_l4_qwen_7b_4bit/empirical_bernstein_summary.csv`
- `research/outputs/real_traces_l4_qwen_7b_4bit/feature_weights.csv`
- `research/outputs/real_traces_l4_qwen_7b_4bit/hazard_decomposition_by_step.csv`
- `research/outputs/real_traces_l4_qwen_7b_4bit/hazard_drift_summary.csv`
- `research/outputs/real_traces_l4_qwen_7b_4bit/sequential_detector_summary.csv`
- `research/outputs/real_traces_l4_qwen_7b_4bit/drift_crossing_proof.png`
- `research/outputs/real_traces_l4_qwen_7b_4bit/real_trace_detector_gaps.png`
- `research/outputs/real_traces_l4_qwen_7b_4bit/real_trace_feature_weights.png`
- `research/FINAL_QWEN_7B_L4_RESULTS.md`
- `research/ANSWERS_QWEN_7B_L4.md`
- `research/open_questions_qwen_7b_l4.md`
- `QWEN_7B_L4_OVERTHINKING_RESULTS.md`
- Updated cross-family outputs under `research/outputs/cross_family/`
- Updated `research/CROSS_FAMILY_REPORT.md`
- Updated `research/CROSS_FAMILY_OPEN_QUESTIONS.md`

## Recovered Qwen 7B Result
- Step-1 accuracy: `0.3644`
- Peak accuracy: `0.7789`
- Peak step: `9`
- Corrected conditional-hazard boundary: `6`
- Empirical utility boundary: `6`
- Fitted hazard boundary: `7`
- Never-stop mean oracle gap: `0.4317`
- Repair rate overall: `0.1794`
- Corruption rate overall: `0.1678`
- Probe Brier / AUC: `0.1625 / 0.9063`
- Assessment: `Late-boundary replication`

## Cross-Family Interpretation
- The corrected DeepSeek 1.5B boundary is early (`1`) after the hazard audit, so the prior step-7 story does not survive as the theorem-facing witness.
- Qwen 0.5B remains a weak-regime control with no late boundary.
- Qwen 7B 4-bit shows a clear competent-regime late boundary with corrected crossing at step `6` and peak accuracy at step `9`.
- The strongest current conclusion is that boundary location is capability-linked within the present data, but cross-family robustness is still not established because only the higher-capability Qwen run shows a late corrected boundary.

## Performance Insights From Recovery Logs
- The `bs16` continuation log finished the pending Qwen 7B collection and recorded step-level generation throughput roughly in the `24` to `48` tokens/sec range with logged `gpu_mem_gb` around `5.18` during generation steps.
- The `bs24` continuation log failed with CUDA OOM at essentially full device occupancy (`22.02 GiB` in use out of `22.03 GiB`), so larger steady-state batches were not robust in the full continuation path.
- The small benchmark folders show that larger batch sizes can pass short calibration slices, but the resume logs are the stronger evidence for full-run stability. In practice, `bs16` is the best recovered stable configuration and `bs24` is not safe enough for unattended long runs.

## Done Status
Core cycle status: done for the current thesis phase.

What is complete:
- The Qwen 7B second-family raw run is fully recovered.
- The missing downstream analysis artifacts have been regenerated.
- The cross-family report has been updated with the recovered Qwen 7B evidence.
- The repo now contains a reproducible record of the full Qwen 7B raw run plus its regenerated reports.

What is not required to close this cycle:
- Resuming the Qwen 7B raw collection. That work is already complete.

What remains optional or next:
- A higher-capability DeepSeek follow-up, another model family, or another benchmark if the goal is to separate family effects from capability effects more cleanly.