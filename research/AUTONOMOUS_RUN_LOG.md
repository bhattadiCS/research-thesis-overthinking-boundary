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
- Current full-run output directory: `research/outputs/real_traces_l4_mistral_7b`
- Current last completed task index: successful Mistral smoke run and L4 benchmark ladder complete; no recoverable full matched-protocol output directory or experiment checkpoint commit exists yet
- Current runtime validation: live NVIDIA L4 re-verified on 2026-04-02; repo-local `.venv` now reuses the system CUDA stack and imports the required experiment packages successfully
- Current git checkpoint state: repo-local git author is configured as `Aditya Bhatt <bhattadiCS@users.noreply.github.com>` and the next run should use the tracked checkpoint driver in `tools/run_checkpointed_real_trace.py`
- Current resume instructions: restart from repo root with the selected `none + sdpa + batch_size=4` configuration, keep the recovered Qwen2.5 7B artifacts untouched, and land the first durable Mistral experiment checkpoint no later than the first completed 25-task / temperature block

## Checkpoints

| Timestamp | Phase | Summary | Commit | Push |
| --- | --- | --- | --- | --- |
| 2026-04-02 14:44 UTC | A | Verified clean audited baseline, git tracking, and live L4 availability. | pending | pending |
| 2026-04-02 14:56 UTC | B | Added thesis-ready stopping-rule documentation, a theory-note algorithm section, and the autonomous run log. | `b2a2b94` | pushed |
| 2026-04-02 15:13 UTC | C | Added the Mistral harness path, L4 benchmark runner, and non-Qwen selection note. | `048036e` | pushed |
| 2026-04-02 15:18 UTC | D | Fixed the quantized Colab runner argument-order bug and checkpointed the successful Mistral smoke run artifacts. | `9d74e32` | pushed |
| 2026-04-02 15:31 UTC | D | Committed the benchmark-selection checkpoint and recorded the chosen long-run Mistral L4 runtime. | `e6be9dd` | pushed |

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

1. Validate the selected `none + sdpa + batch_size=4` path on the current runtime without repeating the full ladder.
2. Run the full matched-protocol non-Qwen experiment with aggressive checkpoint commits.
3. Regenerate per-run and cross-family reports and update the claim.
