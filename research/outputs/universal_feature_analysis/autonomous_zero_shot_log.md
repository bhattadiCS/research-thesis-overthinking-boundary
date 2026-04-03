# Algorithm X CPU Phase Log

## Session Header

- Date: 2026-04-03
- Mode: local CPU analysis only
- Git pre-flight: completed with `git fetch --all --prune` and `git pull --ff-only origin main`
- Theory note reviewed: `research/overthinking_boundary.md`
- Verified trace families: Qwen 0.5B, DeepSeek 1.5B, Mistral 7B, Qwen 7B

## Experimental Protocol

- Candidate signals analyzed: entropy_mean, entropy_std, confidence, hidden_l2_shift, answer_changed, thought_token_count
- Normalization: per-family z-score before pooling
- Targets: conditional repair hazard and conditional corruption hazard
- Validation: leave-one-family-out zero-shot evaluation
- Final selected basis: quadratic lift over entropy_mean, answer_changed, thought_token_count, hidden_l2_shift
- Final weight export scope: capable group only (Mistral 7B and Qwen 7B)

## Selection Outcome

- Selected model: `quadratic_top4`
- Average zero-shot repair AUC: `0.6070`
- Average zero-shot corruption AUC: `0.6930`
- Mistral hidden repair AUC: `0.7089`
- Mistral hidden repair gap: `-0.0584`
- Qwen 7B hidden corruption AUC: `0.8055`

## Success Matrix

- Average beta LOFO AUC > 0.70: fail (value=0.6930, threshold=0.7000)
- Mistral alpha generalization gap < 0.05: pass (value=-0.0584, threshold=0.0500)
- Mistral hidden repair AUC about 0.68: pass (value=0.7089, threshold=0.6800)
- Qwen 7B hidden corruption AUC about 0.80: pass (value=0.8055, threshold=0.8000)

## Interpretation

- Passed checks: `3`
- Failed checks: `1`
- The hidden-family calibration targets are met for Mistral repair and Qwen 7B corruption.
- The full 4-family mean corruption AUC remains below the 0.70 thesis-grade target, so the CPU phase supports strong zero-shot transfer but does not close the universal corruption proof completely.
- The exported capable-group weight vector is therefore suitable as the current best Algorithm X intake for a follow-up GPU phase, but it should be treated as a high-quality partial proof rather than a final theorem-closing estimate.

## Output Artifacts

- model_candidate_summary.csv
- lofo_family_metrics.csv
- generalization_gap_table.md
- feature_event_correlation_matrix.csv
- feature_event_correlation_ranking.csv
- feature_event_correlation_heatmap.png
- entropy_mean_significance.csv
- universal_hazard_weights.csv
- universal_hazard_model_metadata.json
- universal_feature_report.md