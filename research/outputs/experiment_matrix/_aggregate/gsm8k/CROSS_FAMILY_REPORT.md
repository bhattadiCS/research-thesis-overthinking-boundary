# Cross-Family Report

## Executive Summary
A clearly late corrected boundary replicates in 4 capable run(s) across 2 family/families: Qwen 7B (step 5); Qwen 14B (step 5); Qwen 32B (step 5); Mistral Small 22B (step 6). An additional 4 run(s) across 4 family/families (InternLM3, Mistral instruct, Qwen2.5 instruct, Yi 1.5) add weaker late-boundary support: Qwen 3B (step 4); Mistral instruct 7B (step 3); InternLM3 8B (step 4); Yi 1.5 9B (step 3). Cross-family support for a late overthinking boundary is now strong.

Task IDs align across all 12 runs under the shared GSM8K train split and shuffle seed 17 protocol.

## Run Summary
| Run | Family | Params | Backend | Quant | Step-1 acc | Peak acc | Peak step | Corrected boundary | Repair | Corruption | Hazard gap | E-process gap | Never-stop gap | Probe Brier | Probe AUC | Assessment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek 1.5B | DeepSeek-R1 distill | 1.5B | transformers+torch(cuda) | none | 0.3113 | 0.3600 | 9 | 2 | 0.2231 | 0.4307 | 0.3870 | 0.4604 | 0.7338 | 0.2144 | 0.7017 | No late-boundary replication |
| DeepSeek 7B | DeepSeek-R1 distill | 7B | transformers+torch(cuda) | none | 0.4780 | 0.5460 | 6 | 2 | 0.3636 | 0.3499 | 0.2815 | 0.6344 | 0.7771 | 0.1976 | 0.7535 | No late-boundary replication |
| Qwen 0.5B | Qwen2.5 instruct | 0.5B | transformers+torch(cuda) | none | 0.0740 | 0.0813 | 5 | 2 | 0.0045 | 0.0404 | 0.0493 | 0.0121 | 0.4067 | 0.2413 | 0.5787 | No late-boundary replication |
| Qwen 3B | Qwen2.5 instruct | 3B | transformers+torch(cuda) | none | 0.0533 | 0.5040 | 10 | 4 | 0.0907 | 0.0830 | 0.2380 | 0.2369 | 0.4609 | 0.2166 | 0.7087 | Weakened late-boundary support |
| Qwen 7B | Qwen2.5 instruct | 7B | transformers+torch(cuda) | none | 0.2740 | 0.7053 | 10 | 5 | 0.1214 | 0.0634 | 0.1672 | 0.1978 | 0.3878 | 0.1960 | 0.7525 | Late-boundary replication |
| Qwen 14B | Qwen2.5 instruct | 14B | transformers+torch(cuda) | none | 0.0300 | 0.4733 | 9 | 5 | 0.1427 | 0.2577 | 0.3535 | 0.3990 | 0.5676 | 0.1026 | 0.9122 | Late-boundary replication |
| Qwen 32B | Qwen2.5 instruct | 32B | transformers+torch(cuda) | none | 0.0413 | 0.8787 | 10 | 5 | 0.2188 | 0.0611 | 0.1835 | 0.1889 | 0.3529 | 0.1500 | 0.8346 | Late-boundary replication |
| Mistral instruct 7B | Mistral instruct | 7B | transformers+torch(cuda) | none | 0.3040 | 0.3207 | 10 | 3 | 0.0526 | 0.1339 | 0.1440 | 0.1550 | 0.4470 | 0.2316 | 0.6504 | Weakened late-boundary support |
| Phi 4 4B | Phi 4 | 4B | transformers+torch(cuda) | none | 0.3333 | 0.3413 | 10 | 2 | 0.1589 | 0.3427 | 0.2935 | 0.3407 | 0.6594 | 0.2314 | 0.6521 | No late-boundary replication |
| InternLM3 8B | InternLM3 | 8B | transformers+torch(cuda) | none | 0.2707 | 0.6807 | 9 | 4 | 0.1073 | 0.0243 | 0.1569 | 0.1725 | 0.4092 | 0.2170 | 0.6942 | Weakened late-boundary support |
| Yi 1.5 9B | Yi 1.5 | 9B | transformers+torch(cuda) | none | 0.0953 | 0.3807 | 10 | 3 | 0.1001 | 0.1550 | 0.2077 | 0.2341 | 0.4968 | 0.2215 | 0.6923 | Weakened late-boundary support |
| Mistral Small 22B | Mistral Small | 22B | transformers+torch(cuda) | none | 0.0140 | 0.7247 | 10 | 6 | 0.2025 | 0.2266 | 0.3638 | 0.3251 | 0.3338 | 0.1713 | 0.8246 | Late-boundary replication |

## Drift Audit
| Run | Empirical boundary | Corrected boundary | Fitted boundary | Legacy pooled proxy | Mismatch |
| --- | --- | --- | --- | --- | --- |
| DeepSeek 1.5B | 2 | 2 | 2 | 2 | no |
| DeepSeek 7B | 2 | 2 | 4 | 2 | no |
| Qwen 0.5B | 2 | 2 | 2 | 2 | no |
| Qwen 3B | 4 | 4 | 4 | 4 | no |
| Qwen 7B | 5 | 5 | 5 | 5 | no |
| Qwen 14B | 5 | 5 | 6 | 5 | no |
| Qwen 32B | 5 | 5 | 5 | 5 | no |
| Mistral instruct 7B | 3 | 3 | 4 | 3 | no |
| Phi 4 4B | 2 | 2 | 3 | 2 | no |
| InternLM3 8B | 4 | 4 | 4 | 4 | no |
| Yi 1.5 9B | 3 | 3 | 4 | 3 | no |
| Mistral Small 22B | 6 | 6 | 5 | 6 | no |

## Detector Rankings
| Run | Detector | Rank | Mean oracle gap | False-late rate |
| --- | --- | --- | --- | --- |
| DeepSeek 1.5B | oracle | 1 | 0.0000 | 0.000 |
| DeepSeek 1.5B | verifier_first_correct | 2 | 0.0544 | 0.243 |
| DeepSeek 1.5B | first_answer | 3 | 0.3218 | 0.000 |
| DeepSeek 7B | verifier_first_correct | 1 | -0.0093 | 0.091 |
| DeepSeek 7B | oracle | 2 | 0.0000 | 0.000 |
| DeepSeek 7B | hazard_drift | 3 | 0.2815 | 0.477 |
| Qwen 0.5B | first_answer | 1 | -0.0359 | 0.000 |
| Qwen 0.5B | oracle | 2 | 0.0000 | 0.000 |
| Qwen 0.5B | e_process | 3 | 0.0121 | 0.000 |
| Qwen 3B | oracle | 1 | 0.0000 | 0.000 |
| Qwen 3B | verifier_first_correct | 2 | 0.1456 | 0.395 |
| Qwen 3B | answer_stability | 3 | 0.2240 | 0.769 |
| Qwen 7B | oracle | 1 | 0.0000 | 0.000 |
| Qwen 7B | verifier_first_correct | 2 | 0.0508 | 0.239 |
| Qwen 7B | answer_stability | 3 | 0.1543 | 0.892 |
| Qwen 14B | oracle | 1 | 0.0000 | 0.000 |
| Qwen 14B | verifier_first_correct | 2 | 0.0956 | 0.277 |
| Qwen 14B | hazard_drift | 3 | 0.3535 | 0.438 |
| Qwen 32B | oracle | 1 | 0.0000 | 0.000 |
| Qwen 32B | verifier_first_correct | 2 | 0.0376 | 0.102 |
| Qwen 32B | hazard_drift | 3 | 0.1835 | 0.783 |
| Mistral instruct 7B | oracle | 1 | 0.0000 | 0.000 |
| Mistral instruct 7B | first_answer | 2 | 0.0137 | 0.000 |
| Mistral instruct 7B | verifier_first_correct | 3 | 0.1423 | 0.558 |
| Phi 4 4B | oracle | 1 | 0.0000 | 0.000 |
| Phi 4 4B | verifier_first_correct | 2 | 0.0046 | 0.279 |
| Phi 4 4B | first_answer | 3 | 0.2174 | 0.000 |
| InternLM3 8B | oracle | 1 | 0.0000 | 0.000 |
| InternLM3 8B | verifier_first_correct | 2 | 0.0834 | 0.273 |
| InternLM3 8B | answer_stability | 3 | 0.1478 | 0.544 |
| Yi 1.5 9B | oracle | 1 | 0.0000 | 0.000 |
| Yi 1.5 9B | verifier_first_correct | 2 | 0.1763 | 0.483 |
| Yi 1.5 9B | answer_stability | 3 | 0.2045 | 0.753 |
| Mistral Small 22B | oracle | 1 | 0.0000 | 0.000 |
| Mistral Small 22B | verifier_first_correct | 2 | 0.0907 | 0.233 |
| Mistral Small 22B | e_process | 3 | 0.3251 | 0.945 |

## Signal Comparison
| Run | Strongest correctness signal | Strongest corruption signal |
| --- | --- | --- |
| DeepSeek 1.5B | answer revision flag (answer_changed, coeff=-0.720) | entropy volatility (entropy_std, coeff=0.363) |
| DeepSeek 7B | answer revision flag (answer_changed, coeff=-0.925) | answer revision flag (answer_changed, coeff=0.240) |
| Qwen 0.5B | token entropy (entropy_mean, coeff=-0.368) | verbosity-confidence proxy (verbose_confidence_proxy, coeff=0.625) |
| Qwen 3B | self-reported confidence (confidence, coeff=0.717) | answer revision flag (answer_changed, coeff=0.443) |
| Qwen 7B | self-reported confidence (confidence, coeff=0.740) | token entropy (entropy_mean, coeff=1.431) |
| Qwen 14B | hidden-state L2 drift (hidden_l2_shift, coeff=1.755) | answer revision flag (answer_changed, coeff=0.942) |
| Qwen 32B | hidden-state L2 drift (hidden_l2_shift, coeff=1.152) | hidden-state L2 drift (hidden_l2_shift, coeff=1.537) |
| Mistral instruct 7B | answer revision flag (answer_changed, coeff=-0.554) | hidden-state L2 drift (hidden_l2_shift, coeff=0.908) |
| Phi 4 4B | answer revision flag (answer_changed, coeff=-0.565) | answer revision flag (answer_changed, coeff=0.355) |
| InternLM3 8B | hidden-state L2 drift (hidden_l2_shift, coeff=0.395) | hidden-state L2 drift (hidden_l2_shift, coeff=1.309) |
| Yi 1.5 9B | self-reported confidence (confidence, coeff=0.994) | token entropy (entropy_mean, coeff=0.718) |
| Mistral Small 22B | hidden-state L2 drift (hidden_l2_shift, coeff=1.736) | token entropy (entropy_mean, coeff=0.480) |

## Figures
![Cross-family boundary comparison](outputs/cross_family/cross_family_boundary_comparison.png)

![Cross-family detector gaps](outputs/cross_family/cross_family_detector_gaps.png)
