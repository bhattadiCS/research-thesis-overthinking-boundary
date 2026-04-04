# CPU Truth Audit — 2026-04-04

## Scope

- Audited prompt files, reports, code, metadata, and local trace directories requested in the CPU continuation brief.
- Reran the CPU-safe legacy analysis commands with `--random-state 7`.
- Treated the local workspace as ground truth.

## Required Claims

| claim | supported_by_local_artifact | implemented_in_local_code | status | required_action |
| --- | --- | --- | --- | --- |
| `quadratic_top4` is still the deployed Phase 1 / frontier intake baseline. | yes — `research/outputs/universal_feature_analysis/universal_hazard_model_metadata.json` and the CPU rerun of `research/universal_feature_analysis.py` both select `quadratic_top4`. | yes | supported | Leave unchanged until a promotion decision is executed coherently through metadata, reports, and entrypoints. |
| `qwen_3p5_9b` replaced stale Qwen 3.5 aliases everywhere. | partial — smoke traces and reports use `qwen_3p5_9b`, but legacy `qwen_3p5_7b_instruct` artifacts and prompt text still exist. | partial — runner and catalog now expose `qwen_3p5_9b`, but stale aliases remain for historical context. | drift remains | Do not describe `qwen_3p5_7b_instruct` as the current public target; keep it clearly historical or remove it later. |
| `llama_3p1_8b_instruct` replaced stale `llama_4_8b_it` assumptions everywhere. | partial — `research/reports/frontier_validation_report.md` expects `llama_3p1_8b_instruct`, but prompt/report history still references `llama_4_8b_it`. | partial — frontier validator defaults now target `llama_3p1_8b_instruct`, but `llama_4_8b_it` still exists in the model catalog and runner choices. | drift remains | Treat `llama_4_8b_it` as stale history, not the active feasible frontier target. |
| The frontier validator defaults target the feasible corrected frontier run set. | yes — `research/reports/frontier_validation_report.md` names the corrected missing directories. | yes — `research/frontier_validation_report.py` now defaults to `gemma_4_e4b_it`, `qwen_3p5_9b`, `gemma_4_31b_it`, and `llama_3p1_8b_instruct`. | resolved | Future validator runs can use defaults without silently drifting back to the impossible legacy set. |
| The best theorem-preserving hazard equation changed. | yes — `research/reports/equation_analysis_report.md` recommends `hazard_quadratic_combo_entropymean_entropystd_confidence_thoughttokencount`. | no — metadata and deployed validator intake remain `quadratic_top4`. | recommendation only | Keep this as the documented next hazard candidate until the universal training/metadata pipeline is explicitly upgraded. |
| The best overall empirical stop rule changed to `direct_drift_ridge_top4`. | yes — `research/reports/equation_analysis_report.md` ranks it first by boundary accuracy then oracle gap. | no | experimental only | Keep it as a comparator until the thesis explicitly chooses a non-hazard algorithmic formulation. |
| Additional completed full frontier run directories exist locally. | no — only smoke frontier traces and empty failed placeholder directories were found. | no | unsupported | Keep the frontier generalization claim pending. |
| Additional local trace data beyond the current frontier reports exist. | yes — auxiliary directories `real_traces_colab`, `real_traces_colab_smoke`, `real_traces_l4_mistral_7b_4bit_smoke`, and `real_traces_l4_qwen_7b_4bit_smoke` are present. | n/a | true but non-frontier | Treat these as auxiliary smoke/debug evidence only, not as full frontier validation. |
| Current parse failures imply hidden-state corruption or unusable traces. | no — smoke `.npz` files are valid and DeepSeek/Qwen traces still recover answers via fallback extraction. | no | false | Report the issue as strict exact-format parse failure plus truncation, not as hidden-state corruption. |

## Local Run Discovery

### Completed legacy run directories

- `research/outputs/real_traces_l4_deepseek_1p5b`
- `research/outputs/real_traces_l4_mistral_7b`
- `research/outputs/real_traces_l4_qwen_0p5b`
- `research/outputs/real_traces_l4_qwen_7b_4bit`

### Frontier smoke directories with actual trace data

- `research/outputs/real_traces_colab_smoke_gemma_4_e4b_it` — 2 runs, 4 steps, 2 `.npz`
- `research/outputs/real_traces_colab_smoke_qwen_3p5_9b` — 2 runs, 4 steps, 2 `.npz`

### Additional auxiliary trace directories with data

- `research/outputs/real_traces_colab` — Qwen 2.5 0.5B auxiliary traces
- `research/outputs/real_traces_colab_smoke` — Qwen 2.5 0.5B auxiliary smoke traces
- `research/outputs/real_traces_l4_mistral_7b_4bit_smoke`
- `research/outputs/real_traces_l4_qwen_7b_4bit_smoke`

### Empty failed placeholder directories

- `research/outputs/real_traces_colab_smoke_qwen_3p5_35b_gptq_int4`
- `research/outputs/real_traces_colab_smoke_qwen_3p5_35b_gptq_int4_torch_backend`
- `research/outputs/real_traces_colab_smoke_qwen_3p5_35b_moe_it`
- `research/outputs/real_traces_colab_smoke_qwen_3p5_7b_instruct`

### Recovered mirrors

- `recovered_overnight_data` and `recovered_thesis_data` contain mirrored legacy/recovery artifacts, but no additional completed full frontier run directories.

## CPU Re-analysis Result

- `research/universal_feature_analysis.py --random-state 7` reran successfully and still selected `quadratic_top4`.
- `research/equation_analysis.py --random-state 7` regenerated `research/reports/equation_analysis_report.md` with the same ranking used in this audit.
- `research/frontier_validation_report.py` remains frontier-pending because the corrected full-run directories do not exist locally.