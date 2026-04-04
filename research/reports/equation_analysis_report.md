# Equation Analysis Report

## Scope

- Families analyzed: DeepSeek 1.5B, Mistral 7B, Qwen 0.5B, Qwen 7B.
- Run directories analyzed: real_traces_l4_deepseek_1p5b, real_traces_l4_mistral_7b, real_traces_l4_qwen_0p5b, real_traces_l4_qwen_7b_4bit.
- Evaluation protocol: leave-one-family-out over all loaded families.
- Stopping objective: stop at the first step $t \ge 2$ where the estimated continuation value is non-positive.
- Note: no frontier full-trace directories were included unless explicitly supplied at runtime.

## Recommendation

Best overall stop-rule variant by boundary accuracy then oracle gap: `direct_drift_ridge_top4`.
Best hazard-preserving equation: `hazard_quadratic_combo_entropymean_entropystd_confidence_thoughttokencount`.
$$
\hat{\mu}_t = (1 - \hat{q}_t)\hat{\alpha}_t - \hat{q}_t\hat{\beta}_t - 0.05,\qquad \phi(x_t) = \text{quadratic}(entropy_mean, entropy_std, confidence, thought_token_count)
$$
The recommended hazard equation achieved LOFO mean test AUC `0.6151`, boundary accuracy within $\pm 1$ step `0.5111`, and mean oracle gap `0.2512`.
The best overall stop rule `direct_drift_ridge_top4` reaches boundary accuracy `0.5358` with oracle gap `0.3216`, but it departs from the original q/alpha/beta hazard decomposition.

## Coverage

| family | run_dir | runs | steps | tasks | parse_success_rate |
| --- | --- | --- | --- | --- | --- |
| DeepSeek 1.5B | real_traces_l4_deepseek_1p5b | 900 | 9000 | 300 | 0.010888888888888889 |
| Mistral 7B | real_traces_l4_mistral_7b | 900 | 9000 | 300 | 0.6938888888888889 |
| Qwen 0.5B | real_traces_l4_qwen_0p5b | 900 | 9000 | 300 | 0.9871111111111112 |
| Qwen 7B | real_traces_l4_qwen_7b_4bit | 900 | 9000 | 300 | 0.566 |

## Variant Comparison

| rank | variant | estimator_family | basis | feature_count | mean_test_auc | mean_generalization_gap | boundary_within_one | mean_oracle_gap | mean_q_test_auc | mean_alpha_test_auc | mean_beta_test_auc | mean_drift_test_auc | mean_ece | closed_form |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | direct_drift_ridge_top4 | direct_drift | quadratic | 4 | 0.5890 | 0.0821 | 0.5358 | 0.3216 | nan | nan | nan | 0.5890 | nan | True |
| 2 | hazard_quadratic_combo_entropymean_entropystd_confidence_thoughttokencount | hazard | quadratic | 4 | 0.6151 | 0.1064 | 0.5111 | 0.2512 | 0.5987 | 0.6091 | 0.6374 | nan | 0.3048 | True |
| 3 | direct_drift_hgb_top4 | direct_drift | linear | 4 | 0.6107 | 0.1494 | 0.5061 | 0.3767 | nan | nan | nan | 0.6107 | nan | False |
| 4 | hazard_linear_top4 | hazard | linear | 4 | 0.6361 | 0.0346 | 0.4978 | 0.3199 | 0.6376 | 0.5972 | 0.6736 | nan | 0.3134 | True |
| 5 | hazard_quadratic_top4_kl | hazard | quadratic | 4 | 0.6307 | 0.0621 | 0.4661 | 0.3271 | 0.6457 | 0.5960 | 0.6504 | nan | 0.3020 | True |
| 6 | hazard_quadratic_combo_entropystd_confidence_answerchanged_thoughttokencount | hazard | quadratic | 4 | 0.6183 | 0.1177 | 0.4628 | 0.2787 | 0.6304 | 0.6001 | 0.6243 | nan | 0.2968 | True |
| 7 | hazard_quadratic_combo_confidence_hiddenl2shift_answerchanged_thoughttokencount | hazard | quadratic | 4 | 0.6050 | 0.1413 | 0.4622 | 0.3042 | 0.6159 | 0.5978 | 0.6014 | nan | 0.2947 | True |
| 8 | hazard_quadratic_combo_entropymean_confidence_answerchanged_thoughttokencount | hazard | quadratic | 4 | 0.6248 | 0.1120 | 0.4594 | 0.2865 | 0.6325 | 0.6197 | 0.6222 | nan | 0.2975 | True |
| 9 | hazard_quadratic_combo_entropystd_confidence_hiddenl2shift_thoughttokencount | hazard | quadratic | 4 | 0.6031 | 0.1284 | 0.4467 | 0.2593 | 0.6037 | 0.5851 | 0.6206 | nan | 0.2991 | True |
| 10 | hazard_quadratic_combo_entropystd_confidence_hiddenl2shift_answerchanged | hazard | quadratic | 4 | 0.6126 | 0.1323 | 0.4439 | 0.3181 | 0.6276 | 0.5954 | 0.6148 | nan | 0.2865 | True |
| 11 | hazard_quadratic_combo_entropymean_entropystd_confidence_answerchanged | hazard | quadratic | 4 | 0.6065 | 0.1229 | 0.4425 | 0.3132 | 0.6238 | 0.6168 | 0.5788 | nan | 0.2976 | True |
| 12 | hazard_quadratic_combo_entropymean_confidence_hiddenl2shift_thoughttokencount | hazard | quadratic | 4 | 0.6105 | 0.1203 | 0.4381 | 0.2625 | 0.6060 | 0.6027 | 0.6229 | nan | 0.3010 | True |
| 13 | hazard_cubic_top4 | hazard | cubic | 4 | 0.6111 | 0.1655 | 0.4344 | 0.3534 | 0.6331 | 0.5485 | 0.6517 | nan | 0.2893 | True |
| 14 | hazard_quadratic_combo_entropymean_entropystd_answerchanged_thoughttokencount | hazard | quadratic | 4 | 0.6338 | 0.0612 | 0.4331 | 0.3247 | 0.6364 | 0.6075 | 0.6574 | nan | 0.3049 | True |
| 15 | hazard_quadratic_drop_hidden_l2_shift | hazard | quadratic | 3 | 0.6371 | 0.0469 | 0.4328 | 0.3205 | 0.6454 | 0.6105 | 0.6555 | nan | 0.3055 | True |
| 16 | hazard_quadratic_combo_entropymean_confidence_hiddenl2shift_answerchanged | hazard | quadratic | 4 | 0.6186 | 0.1287 | 0.4322 | 0.3181 | 0.6308 | 0.6080 | 0.6171 | nan | 0.2888 | True |
| 17 | hazard_quadratic_combo_entropymean_entropystd_hiddenl2shift_answerchanged | hazard | quadratic | 4 | 0.6459 | 0.0656 | 0.4208 | 0.3304 | 0.6316 | 0.6212 | 0.6850 | nan | 0.3040 | True |
| 18 | hazard_quadratic_top4_pca | hazard | quadratic | 4 | 0.6425 | 0.0634 | 0.4156 | 0.3268 | 0.6449 | 0.6007 | 0.6820 | nan | 0.3019 | True |
| 19 | hazard_quadratic_drop_entropy_mean | hazard | quadratic | 3 | 0.6202 | 0.0699 | 0.4139 | 0.3227 | 0.6091 | 0.5816 | 0.6700 | nan | 0.3034 | True |
| 20 | hazard_quadratic_drop_thought_token_count | hazard | quadratic | 3 | 0.6490 | 0.0573 | 0.4128 | 0.3317 | 0.6389 | 0.6265 | 0.6818 | nan | 0.3066 | True |
| 21 | hazard_quadratic_combo_entropystd_hiddenl2shift_answerchanged_thoughttokencount | hazard | quadratic | 4 | 0.6432 | 0.0695 | 0.4081 | 0.3295 | 0.6409 | 0.5967 | 0.6921 | nan | 0.2997 | True |
| 22 | hazard_quadratic_combo_entropymean_entropystd_hiddenl2shift_thoughttokencount | hazard | quadratic | 4 | 0.6211 | 0.0534 | 0.4003 | 0.3459 | 0.6009 | 0.5771 | 0.6853 | nan | 0.3059 | True |
| 23 | hazard_quadratic_combo_entropymean_hiddenl2shift_answerchanged_thoughttokencount | hazard | quadratic | 4 | 0.6479 | 0.0673 | 0.3986 | 0.3287 | 0.6435 | 0.6070 | 0.6930 | nan | 0.3007 | True |
| 24 | hazard_quadratic_top4 | hazard | quadratic | 4 | 0.6479 | 0.0673 | 0.3986 | 0.3287 | 0.6435 | 0.6070 | 0.6930 | nan | 0.3007 | True |
| 25 | hazard_quadratic_drop_answer_changed | hazard | quadratic | 3 | 0.6249 | 0.0337 | 0.3956 | 0.3452 | 0.6142 | 0.5840 | 0.6763 | nan | 0.3122 | True |
| 26 | hazard_quadratic_combo_entropymean_entropystd_confidence_hiddenl2shift | hazard | quadratic | 4 | 0.5855 | 0.1297 | 0.3819 | 0.3343 | 0.5873 | 0.5811 | 0.5881 | nan | 0.2950 | True |
| 27 | hazard_rf_top4 | hazard | linear | 4 | 0.6436 | 0.2293 | 0.3489 | 0.3586 | 0.6441 | 0.5642 | 0.7225 | nan | 0.2276 | False |
| 28 | hazard_hgb_top4 | hazard | linear | 4 | 0.6342 | 0.2367 | 0.3478 | 0.3638 | 0.6336 | 0.5700 | 0.6991 | nan | 0.2355 | False |
| 29 | hazard_q_svm_top4 | hazard | linear | 4 | 0.6316 | 0.0669 | 0.3342 | 0.3170 | 0.6241 | 0.5972 | 0.6736 | nan | 0.3379 | False |

## Feature Ablation

| variant | features | mean_test_auc | boundary_within_one | mean_oracle_gap |
| --- | --- | --- | --- | --- |
| hazard_quadratic_drop_hidden_l2_shift | entropy_mean, answer_changed, thought_token_count | 0.6371 | 0.4328 | 0.3205 |
| hazard_quadratic_drop_entropy_mean | answer_changed, thought_token_count, hidden_l2_shift | 0.6202 | 0.4139 | 0.3227 |
| hazard_quadratic_drop_thought_token_count | entropy_mean, answer_changed, hidden_l2_shift | 0.6490 | 0.4128 | 0.3317 |
| hazard_quadratic_drop_answer_changed | entropy_mean, thought_token_count, hidden_l2_shift | 0.6249 | 0.3956 | 0.3452 |

## Best 4-of-6 Feature Sets

| variant | features | mean_test_auc | boundary_within_one | mean_oracle_gap |
| --- | --- | --- | --- | --- |
| hazard_quadratic_combo_entropymean_entropystd_confidence_thoughttokencount | entropy_mean, entropy_std, confidence, thought_token_count | 0.6151 | 0.5111 | 0.2512 |
| hazard_quadratic_combo_entropystd_confidence_answerchanged_thoughttokencount | entropy_std, confidence, answer_changed, thought_token_count | 0.6183 | 0.4628 | 0.2787 |
| hazard_quadratic_combo_confidence_hiddenl2shift_answerchanged_thoughttokencount | confidence, hidden_l2_shift, answer_changed, thought_token_count | 0.6050 | 0.4622 | 0.3042 |
| hazard_quadratic_combo_entropymean_confidence_answerchanged_thoughttokencount | entropy_mean, confidence, answer_changed, thought_token_count | 0.6248 | 0.4594 | 0.2865 |
| hazard_quadratic_combo_entropystd_confidence_hiddenl2shift_thoughttokencount | entropy_std, confidence, hidden_l2_shift, thought_token_count | 0.6031 | 0.4467 | 0.2593 |
| hazard_quadratic_combo_entropystd_confidence_hiddenl2shift_answerchanged | entropy_std, confidence, hidden_l2_shift, answer_changed | 0.6126 | 0.4439 | 0.3181 |
| hazard_quadratic_combo_entropymean_entropystd_confidence_answerchanged | entropy_mean, entropy_std, confidence, answer_changed | 0.6065 | 0.4425 | 0.3132 |
| hazard_quadratic_combo_entropymean_confidence_hiddenl2shift_thoughttokencount | entropy_mean, confidence, hidden_l2_shift, thought_token_count | 0.6105 | 0.4381 | 0.2625 |
| hazard_quadratic_combo_entropymean_entropystd_answerchanged_thoughttokencount | entropy_mean, entropy_std, answer_changed, thought_token_count | 0.6338 | 0.4331 | 0.3247 |
| hazard_quadratic_combo_entropymean_confidence_hiddenl2shift_answerchanged | entropy_mean, confidence, hidden_l2_shift, answer_changed | 0.6186 | 0.4322 | 0.3181 |

## Hidden-State Geometry

| feature | mean_abs_corr | mean_auc |
| --- | --- | --- |
| hidden_l2_shift | 0.0844 | 0.5671 |
| pca_velocity_norm | 0.0592 | 0.5470 |
| hidden_kl_divergence | 0.0394 | 0.4461 |

## Interpretation

Quadratic structure should be retained only if it materially improves LOFO boundary accuracy without widening the generalization gap. Feature ablations identify whether any of the four current observables are acting as passengers rather than signal carriers. The geometry table tests whether PCA trajectory velocity or KL divergence consistently outrun raw hidden-state L2 drift as a family-normalized observable.
The sweep indicates that the strongest hazard-form replacement for the current equation is not the original top-4 set: entropy variance and confidence improve boundary accuracy more consistently than answer change and hidden-state L2 shift in the best LOFO hazard variant.

## Current Baseline Audit

Current selected baseline `hazard_quadratic_top4` scored LOFO mean test AUC `0.6479`, boundary accuracy `0.3986`, and oracle gap `0.3287`.
