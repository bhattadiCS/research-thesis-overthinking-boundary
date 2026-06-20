# Cross-Family Report

## Executive Summary
A clearly late corrected boundary replicates in 3 capable run(s) across 1 family/families: Qwen 14B (step 5); Qwen 32B (step 5); Qwen 7B (step 5). An additional 4 run(s) across 4 family/families (InternLM3, Mistral instruct, Qwen2.5 instruct, Yi 1.5) add weaker late-boundary support: InternLM3 8B (step 4); Mistral instruct 7B (step 3); Qwen 3B (step 4); Yi 1.5 9B (step 3). The clearly-late evidence is concentrated in the Qwen2.5 instruct family, with weaker corroboration from 3 other family/families; cross-family support is materially stronger than a single-witness story, though a clearly-late non-Qwen2.5 instruct witness is still pending.

Task IDs align across all 11 runs under the shared GSM8K train split and shuffle seed 17 protocol.

## Run Summary
| Run | Family | Params | Backend | Quant | Step-1 acc | Peak acc | Peak step | Corrected boundary | Repair | Corruption | Hazard gap | E-process gap | Never-stop gap | Probe Brier | Probe AUC | Assessment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek 1.5B | DeepSeek-R1 distill | 1.5B | transformers+torch(cuda) | none | 0.3113 | 0.3600 | 9 | 1 | 0.2231 | 0.4307 | 0.4436 | 0.4533 | 0.7886 | 0.2140 | 0.6264 | No late-boundary replication |
| DeepSeek 7B | DeepSeek-R1 distill | 7B | transformers+torch(cuda) | none | 0.4780 | 0.5460 | 6 | 1 | 0.3636 | 0.3499 | 0.3297 | 0.6262 | 0.8275 | 0.1971 | 0.7278 | No late-boundary replication |
| InternLM3 8B | InternLM3 | 8B | transformers+torch(cuda) | none | 0.2533 | 0.6753 | 9 | 4 | 0.1073 | 0.0243 | 0.1914 | 0.2046 | 0.4466 | 0.2159 | 0.8639 | Weakened late-boundary support |
| Mistral instruct 7B | Mistral instruct | 7B | transformers+torch(cuda) | none | 0.2953 | 0.3200 | 10 | 3 | 0.0526 | 0.1339 | 0.2647 | 0.2506 | 0.5533 | 0.2308 | 0.7294 | Weakened late-boundary support |
| Phi 4 4B | Phi 4 | 4B | transformers+torch(cuda) | none | 0.3240 | 0.3407 | 10 | 1 | 0.1589 | 0.3427 | 0.4204 | 0.4609 | 0.7795 | 0.2309 | 0.6387 | No late-boundary replication |
| Qwen 0.5B | Qwen2.5 instruct | 0.5B | transformers+torch(cuda) | none | 0.0720 | 0.0813 | 5 | 1 | 0.0045 | 0.0404 | 0.0870 | 0.0655 | 0.4602 | 0.2411 | 0.5716 | No late-boundary replication |
| Qwen 14B | Qwen2.5 instruct | 14B | transformers+torch(cuda) | none | 0.0280 | 0.4693 | 9 | 5 | 0.1427 | 0.2577 | 0.3842 | 0.4294 | 0.5981 | 0.1035 | 0.8948 | Late-boundary replication |
| Qwen 32B | Qwen2.5 instruct | 32B | transformers+torch(cuda) | none | 0.0413 | 0.8680 | 10 | 5 | 0.2188 | 0.0611 | 0.1814 | 0.1966 | 0.3626 | 0.1526 | 0.9609 | Late-boundary replication |
| Qwen 3B | Qwen2.5 instruct | 3B | transformers+torch(cuda) | none | 0.0520 | 0.5013 | 10 | 4 | 0.0907 | 0.0830 | 0.2634 | 0.2691 | 0.4918 | 0.2152 | 0.8130 | Weakened late-boundary support |
| Qwen 7B | Qwen2.5 instruct | 7B | transformers+torch(cuda) | none | 0.2740 | 0.7053 | 10 | 5 | 0.1214 | 0.0634 | 0.2224 | 0.2349 | 0.4442 | 0.1955 | 0.8756 | Late-boundary replication |
| Yi 1.5 9B | Yi 1.5 | 9B | transformers+torch(cuda) | none | 0.0913 | 0.3767 | 10 | 3 | 0.1001 | 0.1550 | 0.2582 | 0.2392 | 0.5358 | 0.2206 | 0.7609 | Weakened late-boundary support |

## Drift Audit
| Run | Empirical boundary | Corrected boundary | Fitted boundary | Legacy pooled proxy | Mismatch |
| --- | --- | --- | --- | --- | --- |
| DeepSeek 1.5B | 1 | 1 | 1 | 3 | yes |
| DeepSeek 7B | 1 | 1 | 3 | 1 | no |
| InternLM3 8B | 4 | 4 | 4 | 3 | yes |
| Mistral instruct 7B | 3 | 3 | 5 | 3 | no |
| Phi 4 4B | 1 | 1 | 3 | 3 | yes |
| Qwen 0.5B | 1 | 1 | 1 | 1 | no |
| Qwen 14B | 5 | 5 | 7 | 5 | no |
| Qwen 32B | 5 | 5 | 5 | 4 | yes |
| Qwen 3B | 4 | 4 | 3 | 3 | yes |
| Qwen 7B | 5 | 5 | 5 | 4 | yes |
| Yi 1.5 9B | 3 | 3 | 3 | 3 | no |

## Detector Rankings
| Run | Detector | Rank | Mean oracle gap | False-late rate |
| --- | --- | --- | --- | --- |
| DeepSeek 1.5B | oracle | 1 | 0.0000 | 0.000 |
| DeepSeek 1.5B | verifier_first_correct | 2 | 0.1092 | 0.243 |
| DeepSeek 1.5B | first_answer | 3 | 0.3766 | 0.000 |
| DeepSeek 7B | oracle | 1 | 0.0000 | 0.000 |
| DeepSeek 7B | verifier_first_correct | 2 | 0.0411 | 0.091 |
| DeepSeek 7B | hazard_drift | 3 | 0.3297 | 0.851 |
| InternLM3 8B | oracle | 1 | 0.0000 | 0.000 |
| InternLM3 8B | verifier_first_correct | 2 | 0.1257 | 0.279 |
| InternLM3 8B | answer_stability | 3 | 0.1875 | 0.896 |
| Mistral instruct 7B | oracle | 1 | 0.0000 | 0.000 |
| Mistral instruct 7B | first_answer | 2 | 0.1280 | 0.000 |
| Mistral instruct 7B | e_process | 3 | 0.2506 | 0.912 |
| Phi 4 4B | oracle | 1 | 0.0000 | 0.000 |
| Phi 4 4B | verifier_first_correct | 2 | 0.1257 | 0.279 |
| Phi 4 4B | first_answer | 3 | 0.3462 | 0.000 |
| Qwen 0.5B | oracle | 1 | 0.0000 | 0.000 |
| Qwen 0.5B | first_answer | 2 | 0.0195 | 0.000 |
| Qwen 0.5B | e_process | 3 | 0.0655 | 0.979 |
| Qwen 14B | oracle | 1 | 0.0000 | 0.000 |
| Qwen 14B | verifier_first_correct | 2 | 0.1257 | 0.279 |
| Qwen 14B | hazard_drift | 3 | 0.3842 | 0.479 |
| Qwen 32B | oracle | 1 | 0.0000 | 0.000 |
| Qwen 32B | verifier_first_correct | 2 | 0.0504 | 0.112 |
| Qwen 32B | hazard_drift | 3 | 0.1814 | 0.851 |
| Qwen 3B | oracle | 1 | 0.0000 | 0.000 |
| Qwen 3B | verifier_first_correct | 2 | 0.1794 | 0.399 |
| Qwen 3B | answer_stability | 3 | 0.2570 | 0.877 |
| Qwen 7B | oracle | 1 | 0.0000 | 0.000 |
| Qwen 7B | verifier_first_correct | 2 | 0.1074 | 0.239 |
| Qwen 7B | answer_stability | 3 | 0.2131 | 0.947 |
| Yi 1.5 9B | oracle | 1 | 0.0000 | 0.000 |
| Yi 1.5 9B | verifier_first_correct | 2 | 0.2196 | 0.488 |
| Yi 1.5 9B | e_process | 3 | 0.2392 | 0.749 |

## Signal Comparison
| Run | Strongest correctness signal | Strongest corruption signal |
| --- | --- | --- |
| DeepSeek 1.5B | answer revision flag (answer_changed, coeff=-0.720) | answer revision flag (answer_changed, coeff=0.355) |
| DeepSeek 7B | answer revision flag (answer_changed, coeff=-0.925) | answer revision flag (answer_changed, coeff=0.224) |
| InternLM3 8B | hidden-state L2 drift (hidden_l2_shift, coeff=0.411) | answer revision flag (answer_changed, coeff=0.503) |
| Mistral instruct 7B | answer revision flag (answer_changed, coeff=-0.552) | answer revision flag (answer_changed, coeff=0.581) |
| Phi 4 4B | answer revision flag (answer_changed, coeff=-0.567) | answer revision flag (answer_changed, coeff=0.325) |
| Qwen 0.5B | token entropy (entropy_mean, coeff=-0.364) | answer revision flag (answer_changed, coeff=0.551) |
| Qwen 14B | hidden-state L2 drift (hidden_l2_shift, coeff=1.753) | answer revision flag (answer_changed, coeff=0.980) |
| Qwen 32B | hidden-state L2 drift (hidden_l2_shift, coeff=1.157) | token entropy (entropy_mean, coeff=1.700) |
| Qwen 3B | self-reported confidence (confidence, coeff=0.738) | answer revision flag (answer_changed, coeff=0.414) |
| Qwen 7B | self-reported confidence (confidence, coeff=0.741) | token entropy (entropy_mean, coeff=1.116) |
| Yi 1.5 9B | self-reported confidence (confidence, coeff=1.049) | token entropy (entropy_mean, coeff=0.710) |

## Figures
![Cross-family boundary comparison](outputs/cross_family/cross_family_boundary_comparison.png)

![Cross-family detector gaps](outputs/cross_family/cross_family_detector_gaps.png)
