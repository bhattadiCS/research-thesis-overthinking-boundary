# Frozen prefix probability model

Artifact: prefix_model.json; file-byte SHA-256: 92fe0af86ac0f204d514a6938d0a29dace2b3cdffa24800cc4e2c85a2d51879f.
This is a CPU-trained, standard-library deployable pair of unweighted logistic models.
Each probability calibrator uses separate calibration tasks; evaluation tasks enter neither fit.

| Target | Held-out rows | AUC | Calibrated Brier | Raw Brier | ECE (10 bins) |
|---|---:|---:|---:|---:|---:|
| q_current | 1380 | 0.710094 | 0.097599 | 0.098540 | 0.022227 |
| p_next | 1104 | 0.692308 | 0.099697 | 0.100677 | 0.020928 |

AUC measures ranking; held-out Brier/ECE describe marginal prediction quality, not conditional coverage.

| Replay policy | Accuracy | Paired accuracy change | Step utility change | Completion-token saving | Mean stop |
|---|---:|---:|---:|---:|---:|
| fixed_2 | 0.1196 | +0.0145 | +0.1645 | 51.33% | 2.000 |
| fixed_3 | 0.1304 | +0.0254 | +0.1254 | 32.71% | 3.000 |
| fixed_4 | 0.1159 | +0.0109 | +0.0609 | 16.26% | 4.000 |
| learned_drift | 0.1196 | +0.0145 | +0.1643 | 51.29% | 2.004 |
| never | 0.1051 | +0.0000 | +0.0000 | 0.00% | 5.000 |

Replay tokens are archived completion costs, not measured prospective model compute. Full task-cluster intervals and domain diagnostics are in evaluation.json.

Limits:

- Only archived Qwen2.5-0.5B GSM8K training and MATH traces enter fitting.
- Archived four-line output and live JSON prompts differ; this is a material transport limitation.
- Strict JSON confidence is reconstructed from raw output; fallback defaults are missing, not trusted confidence.
- Archives omit emitted EOS tokens while the live generator charges them.
- Reconstructed candidates are regraded against archived gold offline; source labels/data are unchanged.
- A row-level audit preserves old/reconstructed candidates and labels, including parser disagreements.
- Task-disjoint archive calibration is marginal validation, not a guarantee of conditional live calibration.
- The direct one-step drift policy is myopic; it does not approximate a proven Bellman continuation value.
- No live benchmark or adversarial trap outcomes select coefficients, calibrators, features, cost or threshold.
- Bootstrap intervals describe repeated task-cluster sampling from this held-out development panel.
