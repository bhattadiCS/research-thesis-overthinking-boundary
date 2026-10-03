# Frozen prefix probability model

Artifact: prefix_model.json; file-byte SHA-256: f8aad312fdc9abd024e47084efe091addd376f95cb01d7b43932e75dce0764ab.
This is a CPU-trained, standard-library deployable pair of unweighted logistic models.
Each probability calibrator uses separate calibration tasks; evaluation tasks enter neither fit.

| Target | Held-out rows | AUC | Calibrated Brier | Raw Brier | ECE (10 bins) |
|---|---:|---:|---:|---:|---:|
| q_current | 280 | 0.720966 | 0.111227 | 0.114681 | 0.037224 |
| p_next | 224 | 0.715636 | 0.110321 | 0.113175 | 0.037886 |

AUC measures ranking; held-out Brier/ECE describe marginal prediction quality, not conditional coverage.

| Replay policy | Accuracy | Paired accuracy change | Step utility change | Completion-token saving | Mean stop |
|---|---:|---:|---:|---:|---:|
| fixed_2 | 0.1607 | +0.0536 | +0.2036 | 49.59% | 2.000 |
| fixed_3 | 0.1429 | +0.0357 | +0.1357 | 28.65% | 3.000 |
| fixed_4 | 0.1250 | +0.0179 | +0.0679 | 13.61% | 4.000 |
| learned_drift | 0.1607 | +0.0536 | +0.2036 | 49.59% | 2.000 |
| never | 0.1071 | +0.0000 | +0.0000 | 0.00% | 5.000 |

Replay tokens are archived completion costs, not measured prospective model compute. Full task-cluster intervals and domain diagnostics are in evaluation.json.

Limits:

- Only archived Qwen2.5-0.5B GSM8K training and MATH traces enter fitting.
- Archived four-line output and live JSON prompts differ; this is a material transport limitation.
- Strict JSON confidence is reconstructed from raw output; fallback defaults are missing, not trusted confidence.
- Archives omit emitted EOS tokens while the live generator charges them.
- Trajectories with reconstructed/saved candidate disagreements are excluded, not regraded.
- Task-disjoint archive calibration is marginal validation, not a guarantee of conditional live calibration.
- The direct one-step drift policy is myopic; it does not approximate a proven Bellman continuation value.
- No live benchmark or adversarial trap outcomes select coefficients, calibrators, features, cost or threshold.
- Bootstrap intervals describe repeated task-cluster sampling from this held-out development panel.
