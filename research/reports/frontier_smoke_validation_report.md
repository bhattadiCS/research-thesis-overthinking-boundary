# Frontier Validation Report

## Verdict

Frontier validation is partially complete, but at least one required gate is still unmet or one target run is missing.

## Protocol

- Protocol label: `frontier_smoke_validation`.
- Phase 1 intake: `quadratic_top4` fit on capable legacy families only (Mistral 7B, Qwen 7B).
- Step cost: `0.05` utility units per extra reasoning step.
- Efficiency metric: stop accuracy divided by mean cumulative generated tokens, reported as accuracy per 1k generated tokens.
- Baseline for the >25% gate: `never_stop` on the same frontier trace set.

## Success Matrix

| criterion | status |
| --- | --- |
| All requested protocol runs present (2) | pass |
| All hidden-state .npz files valid | fail |
| Zero-shot efficiency gain > 25% vs never-stop | fail |

## Zero-Shot Frontier Results

| model_label | q_auc_zero_shot | alpha_auc_zero_shot | beta_auc_zero_shot | universal_mean_stop_step | universal_mean_oracle_gap | universal_stop_accuracy | universal_accuracy_per_1k_tokens | never_stop_accuracy_per_1k_tokens | efficiency_gain_vs_never_stop_pct | parse_success_rate | hazard_drift_mean_oracle_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 Edge 4B | 0.5000 | nan | 0.5000 | 2.0000 | 0.0500 | 1.0000 | 5.4496 | 5.4496 | 0.0000 | 0.7500 | 0.0500 |
| Qwen3.5 instruct 9B | 0.5000 | nan | 0.5000 | 2.0000 | 0.0500 | 1.0000 | 3.9062 | 3.9062 | 0.0000 | 0.0000 | 0.0500 |

## Hidden-State Integrity

| model_label | npz_file_count | invalid_npz_count | nan_file_count | inf_file_count | zero_shift_count | mean_l2_shift | min_l2_shift | max_l2_shift | data_quality_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 Edge 4B | 2 | 0 | 0 | 0 | 0 | 24.7225 | 18.9005 | 30.5445 | False |
| Qwen3.5 instruct 9B | 2 | 0 | 0 | 0 | 0 | 31.8711 | 18.3495 | 45.3927 | False |
