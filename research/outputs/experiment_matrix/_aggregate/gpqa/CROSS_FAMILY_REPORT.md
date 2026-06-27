# Cross-Family Report

## Executive Summary
No capable run shows a late corrected boundary under the matched protocol.

Task IDs align across all 13 runs under the shared GSM8K train split and shuffle seed 17 protocol.

## Run Summary
| Run | Family | Params | Backend | Quant | Step-1 acc | Peak acc | Peak step | Corrected boundary | Repair | Corruption | Hazard gap | E-process gap | Never-stop gap | Probe Brier | Probe AUC | Assessment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek 1.5B | DeepSeek-R1 distill | 1.5B | transformers+torch(cuda) | none | 0.2068 | 0.2269 | 10 | 2 | 0.1016 | 0.3663 | 0.2029 | 0.2034 | 0.5900 | 0.2489 | 0.5325 | No late-boundary replication |
| DeepSeek 7B | DeepSeek-R1 distill | 7B | transformers+torch(cuda) | none | 0.2135 | 0.2143 | 5 | 2 | 0.1059 | 0.4183 | 0.2505 | 0.3407 | 0.6399 | 0.2476 | 0.5502 | No late-boundary replication |
| Qwen 0.5B | Qwen2.5 instruct | 0.5B | transformers+torch(cuda) | none | 0.2034 | 0.2034 | 1 | 2 | 0.0271 | 0.1342 | 0.0666 | 0.0544 | 0.4626 | 0.2485 | 0.5452 | No late-boundary replication |
| Qwen 3B | Qwen2.5 instruct | 3B | transformers+torch(cuda) | none | 0.1690 | 0.2278 | 10 | 2 | 0.0345 | 0.1053 | 0.1324 | 0.1375 | 0.4860 | 0.2459 | 0.5729 | No late-boundary replication |
| Qwen 7B | Qwen2.5 instruct | 7B | transformers+torch(cuda) | none | 0.3118 | 0.3296 | 5 | 2 | 0.0245 | 0.0473 | 0.0777 | 0.0631 | 0.4445 | 0.2493 | 0.5316 | No late-boundary replication |
| Qwen 14B | Qwen2.5 instruct | 14B | transformers+torch(cuda) | none | 0.3326 | 0.3579 | 9 | 2 | 0.0176 | 0.0257 | 0.0791 | 0.0491 | 0.4320 | 0.2493 | 0.5373 | No late-boundary replication |
| Qwen 32B | Qwen2.5 instruct | 32B | transformers+torch(cuda) | none | 0.2827 | 0.3705 | 9 | 2 | 0.0276 | 0.0289 | 0.0923 | 0.1058 | 0.4358 | 0.2480 | 0.5370 | No late-boundary replication |
| Mistral instruct 7B | Mistral instruct | 7B | transformers+torch(cuda) | none | 0.2775 | 0.2879 | 4 | 2 | 0.0518 | 0.1309 | 0.1599 | 0.1671 | 0.5355 | 0.2497 | 0.5209 | No late-boundary replication |
| Phi 4 4B | Phi 4 | 4B | transformers+torch(cuda) | none | 0.2641 | 0.2865 | 7 | 2 | 0.0889 | 0.2255 | 0.2210 | 0.2629 | 0.5881 | 0.2480 | 0.5486 | No late-boundary replication |
| InternLM3 8B | InternLM3 | 8B | transformers+torch(cuda) | none | 0.3519 | 0.3743 | 3 | 2 | 0.0310 | 0.0491 | 0.0837 | 0.0633 | 0.4573 | 0.2494 | 0.5266 | No late-boundary replication |
| Yi 1.5 9B | Yi 1.5 | 9B | transformers+torch(cuda) | none | 0.2695 | 0.3023 | 10 | 2 | 0.0452 | 0.1023 | 0.1306 | 0.1125 | 0.4954 | 0.2488 | 0.5167 | No late-boundary replication |
| Mistral Small 22B | Mistral Small | 22B | transformers+torch(cuda) | none | 0.1704 | 0.3557 | 10 | 2 | 0.0556 | 0.0793 | 0.2228 | 0.2613 | 0.4828 | 0.2405 | 0.6138 | No late-boundary replication |
| Llama 3.1 8B | Llama 3.1 | 8B | transformers+torch(cuda) | none | 0.1721 | 0.3256 | 10 | 2 | 0.1005 | 0.2105 | 0.3382 | 0.4358 | 0.6298 | 0.2446 | 0.5715 | No late-boundary replication |

## Drift Audit
| Run | Empirical boundary | Corrected boundary | Fitted boundary | Legacy pooled proxy | Mismatch |
| --- | --- | --- | --- | --- | --- |
| DeepSeek 1.5B | 2 | 2 | 2 | 2 | no |
| DeepSeek 7B | 2 | 2 | 2 | 2 | no |
| Qwen 0.5B | 2 | 2 | 2 | 2 | no |
| Qwen 3B | 2 | 2 | 2 | 2 | no |
| Qwen 7B | 2 | 2 | 2 | 2 | no |
| Qwen 14B | 2 | 2 | 2 | 2 | no |
| Qwen 32B | 2 | 2 | 3 | 2 | no |
| Mistral instruct 7B | 2 | 2 | 2 | 2 | no |
| Phi 4 4B | 2 | 2 | 2 | 2 | no |
| InternLM3 8B | 2 | 2 | 2 | 2 | no |
| Yi 1.5 9B | 2 | 2 | 2 | 2 | no |
| Mistral Small 22B | 2 | 2 | 3 | 2 | no |
| Llama 3.1 8B | 2 | 2 | 3 | 2 | no |

## Detector Rankings
| Run | Detector | Rank | Mean oracle gap | False-late rate |
| --- | --- | --- | --- | --- |
| DeepSeek 1.5B | oracle | 1 | 0.0000 | 0.000 |
| DeepSeek 1.5B | first_answer | 2 | 0.1601 | 0.000 |
| DeepSeek 1.5B | verifier_first_correct | 3 | 0.1723 | 0.523 |
| DeepSeek 7B | oracle | 1 | 0.0000 | 0.000 |
| DeepSeek 7B | verifier_first_correct | 2 | 0.1588 | 0.491 |
| DeepSeek 7B | first_answer | 3 | 0.1802 | 0.000 |
| Qwen 0.5B | first_answer | 1 | -0.0143 | 0.000 |
| Qwen 0.5B | oracle | 2 | 0.0000 | 0.000 |
| Qwen 0.5B | e_process | 3 | 0.0544 | 0.000 |
| Qwen 3B | oracle | 1 | 0.0000 | 0.000 |
| Qwen 3B | first_answer | 2 | 0.0948 | 0.000 |
| Qwen 3B | hazard_drift | 3 | 0.1324 | 0.263 |
| Qwen 7B | oracle | 1 | 0.0000 | 0.000 |
| Qwen 7B | first_answer | 2 | 0.0116 | 0.000 |
| Qwen 7B | e_process | 3 | 0.0631 | 0.000 |
| Qwen 14B | oracle | 1 | 0.0000 | 0.000 |
| Qwen 14B | first_answer | 2 | 0.0073 | 0.000 |
| Qwen 14B | e_process | 3 | 0.0491 | 0.000 |
| Qwen 32B | oracle | 1 | 0.0000 | 0.000 |
| Qwen 32B | first_answer | 2 | 0.0735 | 0.000 |
| Qwen 32B | answer_stability | 3 | 0.0894 | 0.500 |
| Mistral instruct 7B | oracle | 1 | 0.0000 | 0.000 |
| Mistral instruct 7B | first_answer | 2 | 0.0952 | 0.000 |
| Mistral instruct 7B | verifier_first_correct | 3 | 0.1307 | 0.504 |
| Phi 4 4B | oracle | 1 | 0.0000 | 0.000 |
| Phi 4 4B | verifier_first_correct | 2 | 0.1166 | 0.447 |
| Phi 4 4B | first_answer | 3 | 0.1604 | 0.000 |
| InternLM3 8B | oracle | 1 | 0.0000 | 0.000 |
| InternLM3 8B | first_answer | 2 | 0.0244 | 0.000 |
| InternLM3 8B | e_process | 3 | 0.0633 | 0.000 |
| Yi 1.5 9B | oracle | 1 | 0.0000 | 0.000 |
| Yi 1.5 9B | first_answer | 2 | 0.0782 | 0.000 |
| Yi 1.5 9B | e_process | 3 | 0.1125 | 0.000 |
| Mistral Small 22B | oracle | 1 | 0.0000 | 0.000 |
| Mistral Small 22B | verifier_first_correct | 2 | 0.1775 | 0.505 |
| Mistral Small 22B | first_answer | 3 | 0.2181 | 0.000 |
| Llama 3.1 8B | oracle | 1 | 0.0000 | 0.000 |
| Llama 3.1 8B | verifier_first_correct | 2 | 0.1081 | 0.366 |
| Llama 3.1 8B | first_answer | 3 | 0.3333 | 0.000 |

## Signal Comparison
| Run | Strongest correctness signal | Strongest corruption signal |
| --- | --- | --- |
| DeepSeek 1.5B | answer revision flag (answer_changed, coeff=-0.188) | answer revision flag (answer_changed, coeff=0.659) |
| DeepSeek 7B | token entropy (entropy_mean, coeff=0.148) | answer revision flag (answer_changed, coeff=0.692) |
| Qwen 0.5B | verbosity-confidence proxy (verbose_confidence_proxy, coeff=-0.175) | answer revision flag (answer_changed, coeff=0.844) |
| Qwen 3B | token entropy (entropy_mean, coeff=0.311) | answer revision flag (answer_changed, coeff=0.579) |
| Qwen 7B | entropy volatility (entropy_std, coeff=0.221) | hidden-state L2 drift (hidden_l2_shift, coeff=0.922) |
| Qwen 14B | token entropy (entropy_mean, coeff=0.132) | hidden-state L2 drift (hidden_l2_shift, coeff=0.702) |
| Qwen 32B | self-reported confidence (confidence, coeff=0.192) | answer revision flag (answer_changed, coeff=0.398) |
| Mistral instruct 7B | lexical echo (lexical_echo, coeff=0.139) | answer revision flag (answer_changed, coeff=0.672) |
| Phi 4 4B | self-reported confidence (confidence, coeff=0.132) | answer revision flag (answer_changed, coeff=0.515) |
| InternLM3 8B | answer revision flag (answer_changed, coeff=-0.100) | hidden-state L2 drift (hidden_l2_shift, coeff=0.996) |
| Yi 1.5 9B | self-reported confidence (confidence, coeff=0.215) | answer revision flag (answer_changed, coeff=0.864) |
| Mistral Small 22B | entropy volatility (entropy_std, coeff=0.380) | hidden-state L2 drift (hidden_l2_shift, coeff=0.804) |
| Llama 3.1 8B | self-reported confidence (confidence, coeff=0.346) | hidden-state L2 drift (hidden_l2_shift, coeff=0.674) |

## Figures
![Cross-family boundary comparison](outputs/cross_family/cross_family_boundary_comparison.png)

![Cross-family detector gaps](outputs/cross_family/cross_family_detector_gaps.png)
