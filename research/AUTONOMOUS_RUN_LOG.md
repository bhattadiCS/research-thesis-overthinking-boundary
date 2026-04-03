# Autonomous Thesis Finalization Run Log

## Session Header

- Started: 2026-04-02
- Baseline branch: `main`
- Baseline remote: `origin`
- Baseline status: clean worktree at start of run
- Verified hardware: NVIDIA L4 with 23034 MiB total VRAM and about 22564 MiB free before workload
- Current audited conclusion at start: capability-linked late-boundary evidence only; Qwen2.5 7B 4-bit remains the sole late corrected-boundary witness

## Current Working State

- Current target model: `mistral_7b_instruct_v0p3` (`mistralai/Mistral-7B-Instruct-v0.3`)
- Current benchmark configuration: selected `quantization=none`, `attn_implementation=sdpa`, `batch_size=4`
- Current benchmark validation: short rerun under `research/outputs/benchmark_mistral7b_l4_validation_20260402` reconfirmed `none + sdpa + batch_size=4` as the fastest stable path in the live runtime; `flash_attn` remains unavailable
- Current full-run output directory: `research/outputs/real_traces_l4_mistral_7b`
- Current last completed task index: full matched-protocol run complete at `900/900` runs (`300` tasks across temperatures `0.1`, `0.6`, and `1.0` with seed `7`)
- Current runtime validation: live NVIDIA L4 was re-verified on 2026-04-02; repo-local `.venv` reused the system CUDA stack successfully for the completed Mistral run
- Current git checkpoint state: repo-local git author remained `Aditya Bhatt <bhattadiCS@users.noreply.github.com>` throughout the run; aggressive checkpoint commits landed and pushed across every temperature block, followed by a final analysis commit
- Current harness compatibility state: the text-only Mistral path required the local torchvision shim and `torch_dtype` load fix, both now committed on `main`
- Current resume instructions: no resume is required for this cycle; the completed raw run, per-run analysis, and updated cross-family artifacts are all on `origin/main`

## Checkpoints

| Timestamp | Phase | Summary | Commit | Push |
| --- | --- | --- | --- | --- |
| 2026-04-02 14:44 UTC | A | Verified clean audited baseline, git tracking, and live L4 availability. | pending | pending |
| 2026-04-02 14:56 UTC | B | Added thesis-ready stopping-rule documentation, a theory-note algorithm section, and the autonomous run log. | `b2a2b94` | pushed |
| 2026-04-02 15:13 UTC | C | Added the Mistral harness path, L4 benchmark runner, and non-Qwen selection note. | `048036e` | pushed |
| 2026-04-02 15:18 UTC | D | Fixed the quantized Colab runner argument-order bug and checkpointed the successful Mistral smoke run artifacts. | `9d74e32` | pushed |
| 2026-04-02 15:31 UTC | D | Committed the benchmark-selection checkpoint and recorded the chosen long-run Mistral L4 runtime. | `e6be9dd` | pushed |
| 2026-04-02 23:16 UTC | D | Added the checkpointed Mistral long-run driver and pinned the repo-local dependency baseline. | `cd28243` | pushed |
| 2026-04-02 23:39 UTC | D | Restored the live Mistral L4 runtime, revalidated the selected configuration, and committed the validation artifacts. | `58b3a3f` | pushed |
| 2026-04-02 23:41 UTC | E | Fixed checkpoint-driver path normalization before restarting the full matched-protocol run. | `e70eedf` | pushed |
| 2026-04-03 00:09 UTC | F | Added the watcher that waits for run completion and finalizes the analysis stack automatically. | `53b2dc8` | pushed |
| 2026-04-03 overnight | E | Full matched-protocol Mistral run completed with checkpoint commits from `f7697f4` through `3a651ce`; per-checkpoint hashes and push status are recorded in `research/outputs/real_traces_l4_mistral_7b_checkpoint_history.jsonl`. | multiple | pushed |
| 2026-04-03 completion | F | Regenerated the Mistral artifacts, reran the cross-family analysis, and committed the completed thesis-cycle outputs. | `968a96b` | pushed |

## Candidate Access Notes

- Gemma attempt: `google/gemma-2-9b-it` returned a gated-repo `401` error in this runtime.
- Llama attempt: `meta-llama/Llama-3.1-8B-Instruct` returned a gated-repo `401` error in this runtime.
- Selected fallback: `mistralai/Mistral-7B-Instruct-v0.3` loaded config and tokenizer successfully with chat-template support.
- Reason for selection: Mistral is the strongest practical genuinely non-Qwen family that is open-access from this Colab environment without waiting on Hugging Face approval.

## Smoke Result

- Smoke output directory: `research/outputs/real_traces_l4_mistral_7b_4bit_smoke`
- Smoke configuration: `4bit`, `device_map=auto`, `attn_implementation=sdpa`, `batch_size=1`
- Outcome: success on the live NVIDIA L4 with no OOM retries and no emergency microbatch splits
- Measured batch profile: step batches ran at about `0.11` to `0.13` examples/s and `10.95` to `13.75` tokens/s with peak reserved memory about `13.48 GB`
- Scientific caveat: the smoke workload used only 2 builtin tasks, so it validates the path but is not the optimization study or the matched 300-task GSM8K run

## L4 Benchmark Result

- Backend scan path: `research/outputs/benchmark_mistral7b_l4_attn_scan`
- 4-bit ladder path: `research/outputs/benchmark_mistral7b_l4_4bit_ladder`
- Full-precision ladder path: `research/outputs/benchmark_mistral7b_l4_fp_ladder`
- `flash_attention_2` is not available in this environment because the `flash_attn` package is not installed; all such runs failed fast after the CUDA fallback fix.
- 8-bit was screened out after the backend scan because it was materially slower than both 4-bit and full precision at matched batch size `2` while also showing much lower mean GPU utilization.
- Best 4-bit point: `sdpa`, batch size `6`, about `0.2346` examples/s, `25.95` tokens/s, and `13.42 GB` peak reserved memory.
- Best full-precision point: `sdpa`, batch size `4`, about `0.3497` examples/s, `35.61` tokens/s, and `15.27 GB` peak reserved memory.
- Bottleneck diagnosis: successful runs were compute-bound rather than I/O-bound; generation time dominated, hidden-state extraction was secondary, and CPU preprocessing plus hidden-state writes were negligible.
- Chosen long-run configuration: `quantization=none`, `attn_implementation=sdpa`, `batch_size=4`.
- Rationale: this is the fastest stable configuration tested by examples/s, it keeps enough VRAM headroom for long-run stability, and increasing batch size beyond `4` in full precision reduced example throughput even though memory was still available.

## Pending Milestones

None. This autonomous cycle is complete.

## Completed Full Mistral Result

- Total matched-protocol runs: `900`
- Temperatures and seed: `0.1`, `0.6`, `1.0` with seed `7`
- Step-1 accuracy: `0.3022`
- Peak accuracy: `0.3189` at step `10`
- Corrected conditional-hazard boundary: `3`
- Fitted hazard boundary: `5`
- Never-stop oracle gap: `0.5696`
- Assessment: `Weakened late-boundary support`

## Final Claim State

- Qwen 7B remains the only clearly late corrected-boundary witness in the repo.
- Mistral 7B now provides a genuinely non-Qwen matched-protocol replication of early-helpful / later-harmful behavior with a nontrivial step-`3` corrected boundary and a large never-stop penalty.
- The cross-family evidence is therefore materially stronger than the earlier single-family story, but full late-boundary robustness is still not established.

## 2026-04-03 Algorithm X CPU Phase

- Detailed phase log: `research/outputs/universal_feature_analysis/autonomous_zero_shot_log.md`
- Mode: local CPU-only universal feature analysis and leave-one-family-out hazard regression
- Candidate scan: `linear_required6`, `linear_top4`, and `quadratic_top4`
- Selected Algorithm X basis: `quadratic_top4` over `entropy_mean`, `answer_changed`, `thought_token_count`, and `hidden_l2_shift`
- Mean zero-shot repair AUC: `0.6070`
- Mean zero-shot corruption AUC: `0.6930`
- Mistral hidden repair AUC: `0.7089`
- Qwen 7B hidden corruption AUC: `0.8055`
- Entropy significance check: `r = 0.1243`, 95% CI `[0.1047, 0.1439]`, `p = 1.17e-34`
- Outcome: strong zero-shot proof of concept with both hidden-family calibration targets satisfied, but the all-family corruption target remains narrowly missed by `0.0070`
- Phase-2 intake file: `research/outputs/universal_feature_analysis/universal_hazard_weights.csv`
