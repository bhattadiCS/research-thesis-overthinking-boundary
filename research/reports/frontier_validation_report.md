# Frontier Validation Report

## Verdict

No completed frontier run directories were available, so the frontier claim remains untested in this workspace. Missing run directories: real_traces_colab_gemma_4_e4b_it, real_traces_colab_qwen_3p5_9b, real_traces_colab_gemma_4_31b_it, real_traces_colab_llama_3p1_8b_instruct.

## Protocol

- Protocol label: `frontier_validation`.
- Phase 1 intake: `quadratic_top4` fit on capable legacy families only (Mistral 7B, Qwen 7B).
- Step cost: `0.05` utility units per extra reasoning step.
- Efficiency metric: stop accuracy divided by mean cumulative generated tokens, reported as accuracy per 1k generated tokens.
- Baseline for the >25% gate: `never_stop` on the same frontier trace set.

## Success Matrix

| criterion | status |
| --- | --- |
| All requested protocol runs present (4) | fail |
| All hidden-state .npz files valid | fail |
| Zero-shot efficiency gain > 25% vs never-stop | fail |

## Zero-Shot Frontier Results

| none |
| --- |

## Hidden-State Integrity

| none |
| --- |

## Missing Runs

| expected_run_dir |
| --- |
| real_traces_colab_gemma_4_e4b_it |
| real_traces_colab_qwen_3p5_9b |
| real_traces_colab_gemma_4_31b_it |
| real_traces_colab_llama_3p1_8b_instruct |
