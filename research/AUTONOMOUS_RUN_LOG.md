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
- Current benchmark configuration: smoke-pass on `4bit + sdpa + batch_size=1`; benchmark ladder pending
- Current full-run output directory: `research/outputs/real_traces_l4_mistral_7b_4bit`
- Current last completed task index: successful Mistral smoke run complete; L4 benchmark ladder pending
- Current resume instructions: rerun from repo root and resume from the latest committed output directory metadata; do not rerun the recovered Qwen2.5 7B artifacts

## Checkpoints

| Timestamp | Phase | Summary | Commit | Push |
| --- | --- | --- | --- | --- |
| 2026-04-02 14:44 UTC | A | Verified clean audited baseline, git tracking, and live L4 availability. | pending | pending |
| 2026-04-02 14:56 UTC | B | Added thesis-ready stopping-rule documentation, a theory-note algorithm section, and the autonomous run log. | `b2a2b94` | pushed |
| 2026-04-02 15:13 UTC | C | Added the Mistral harness path, L4 benchmark runner, and non-Qwen selection note. | `048036e` | pushed |

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

## Pending Milestones

1. Commit and push the smoke-test checkpoint.
2. Run and record the L4 benchmark ladder.
3. Run the full matched-protocol non-Qwen experiment with checkpoint commits.
4. Regenerate per-run and cross-family reports and update the claim.
