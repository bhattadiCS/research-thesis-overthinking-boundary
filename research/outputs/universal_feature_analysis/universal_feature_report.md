# Universal Feature Analysis

## Data Summary

| family | run_directory | runs | steps | repair_eligible_steps | repair_events | corruption_eligible_steps | corruption_events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen 0.5B | real_traces_l4_qwen_0p5b | 900 | 9000 | 7456 | 22 | 644 | 15 |
| DeepSeek 1.5B | real_traces_l4_deepseek_1p5b | 900 | 9000 | 5853 | 1105 | 2247 | 1030 |
| Mistral 7B | real_traces_l4_mistral_7b | 900 | 9000 | 5761 | 326 | 2339 | 311 |
| Qwen 7B | real_traces_l4_qwen_7b_4bit | 900 | 9000 | 3646 | 782 | 4454 | 409 |

## Candidate Comparison

| model_name | basis | feature_count | alpha_mean_test_auc | beta_mean_test_auc | q_mean_test_auc | mistral_alpha_test_auc | mistral_alpha_gap | qwen7_beta_test_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| quadratic_top4 | quadratic | 4 | 0.6070 | 0.6930 | 0.6435 | 0.7089 | -0.0584 | 0.8055 |
| linear_top4 | linear | 4 | 0.5972 | 0.6736 | 0.6376 | 0.6992 | -0.0720 | 0.8082 |
| linear_required6 | linear | 6 | 0.5848 | 0.6319 | 0.6339 | 0.7008 | -0.0105 | 0.8181 |

## Final LOFO Generalization Table

| holdout_family | alpha_train_auc | alpha_test_auc | alpha_generalization_gap | beta_train_auc | beta_test_auc | beta_generalization_gap | q_train_auc | q_test_auc | q_generalization_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen 0.5B | 0.6077 | 0.6457 | -0.0381 | 0.7607 | 0.5987 | 0.1620 | 0.6975 | 0.5291 | 0.1683 |
| DeepSeek 1.5B | 0.7614 | 0.5207 | 0.2407 | 0.8682 | 0.5776 | 0.2905 | 0.7135 | 0.6369 | 0.0766 |
| Mistral 7B | 0.6505 | 0.7089 | -0.0584 | 0.7371 | 0.7902 | -0.0532 | 0.6983 | 0.6506 | 0.0476 |
| Qwen 7B | 0.6884 | 0.5528 | 0.1355 | 0.7351 | 0.8055 | -0.0703 | 0.6638 | 0.7575 | -0.0937 |

## Correlation Ranking

| feature | corr_repair | corr_corruption | max_abs_corr |
| --- | --- | --- | --- |
| confidence | -0.1271 | -0.0325 | 0.1271 |
| answer_changed | 0.1197 | -0.0318 | 0.1197 |
| thought_token_count | 0.0801 | 0.0409 | 0.0801 |
| hidden_l2_shift | 0.0700 | -0.0460 | 0.0700 |
| entropy_mean | 0.0674 | -0.0028 | 0.0674 |
| entropy_std | 0.0548 | -0.0074 | 0.0548 |

## Entropy Mean Stability

| feature | target | statistic | point_estimate | ci_lower | ci_upper | p_value | eligible_steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| entropy_mean | event_corruption | pearson_correlation | 0.1243 | 0.1047 | 0.1439 | 0.0000 | 9684 |

## Success Matrix

| criterion | value | threshold | status |
| --- | --- | --- | --- |
| Average beta LOFO AUC > 0.70 | 0.6930 | 0.7000 | fail |
| Mistral alpha generalization gap < 0.05 | -0.0584 | 0.0500 | pass |
| Mistral hidden repair AUC about 0.68 | 0.7089 | 0.6800 | pass |
| Qwen 7B hidden corruption AUC about 0.80 | 0.8055 | 0.8000 | pass |

## Selected Algorithm X Variant

Selected model: `quadratic_top4`

Basis: `quadratic`

Base features: `entropy_mean, answer_changed, thought_token_count, hidden_l2_shift`

Average beta zero-shot AUC: `0.6930`

Mistral hidden repair AUC: `0.7089`

Qwen 7B hidden corruption AUC: `0.8055`