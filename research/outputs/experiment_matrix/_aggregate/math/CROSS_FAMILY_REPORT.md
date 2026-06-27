# Cross-Family Report

## Executive Summary
A clearly late corrected boundary replicates in 2 capable run(s) across 1 family/families: Qwen 7B (step 5); Qwen 32B (step 6). Late-boundary evidence remains confined to a single capable family, so cross-family robustness is not yet established.

Task IDs align across all 13 runs under the shared GSM8K train split and shuffle seed 17 protocol.

## Run Summary
| Run | Family | Params | Backend | Quant | Step-1 acc | Peak acc | Peak step | Corrected boundary | Repair | Corruption | Hazard gap | E-process gap | Never-stop gap | Probe Brier | Probe AUC | Assessment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek 1.5B | DeepSeek-R1 distill | 1.5B | transformers+torch(cuda) | none | 0.0433 | 0.0540 | 13 | 2 | 0.0318 | 0.6611 | 0.1358 | 0.1188 | 0.7041 | 0.2383 | 0.6080 | No late-boundary replication |
| DeepSeek 7B | DeepSeek-R1 distill | 7B | transformers+torch(cuda) | none | 0.0867 | 0.1193 | 4 | 2 | 0.0582 | 0.5520 | 0.2479 | 0.2248 | 0.8075 | 0.2286 | 0.6058 | No late-boundary replication |
| Qwen 0.5B | Qwen2.5 instruct | 0.5B | transformers+torch(cuda) | none | 0.0513 | 0.0573 | 4 | 2 | 0.0022 | 0.0328 | 0.0532 | 0.0072 | 0.6045 | 0.2361 | 0.6317 | No late-boundary replication |
| Qwen 3B | Qwen2.5 instruct | 3B | transformers+torch(cuda) | none | 0.0440 | 0.1720 | 14 | 2 | 0.0207 | 0.0915 | 0.2464 | 0.1463 | 0.6250 | 0.2138 | 0.7281 | No late-boundary replication |
| Qwen 7B | Qwen2.5 instruct | 7B | transformers+torch(cuda) | none | 0.1093 | 0.3893 | 13 | 5 | 0.0425 | 0.0820 | 0.1820 | 0.5093 | 0.5727 | 0.1760 | 0.8202 | Late-boundary replication |
| Qwen 14B | Qwen2.5 instruct | 14B | transformers+torch(cuda) | none | 0.0647 | 0.3100 | 14 | 2 | 0.0488 | 0.1420 | 0.3824 | 0.3480 | 0.6220 | 0.1319 | 0.8965 | No late-boundary replication |
| Qwen 32B | Qwen2.5 instruct | 32B | transformers+torch(cuda) | none | 0.0880 | 0.5853 | 14 | 6 | 0.0657 | 0.0210 | 0.2616 | 0.2865 | 0.5345 | 0.1754 | 0.8186 | Late-boundary replication |
| Mistral instruct 7B | Mistral instruct | 7B | transformers+torch(cuda) | none | 0.0733 | 0.0733 | 1 | 2 | 0.0087 | 0.1423 | 0.1490 | 0.0523 | 0.6257 | 0.2209 | 0.7097 | No late-boundary replication |
| Phi 4 4B | Phi 4 | 4B | transformers+torch(cuda) | none | 0.0887 | 0.1373 | 10 | 2 | 0.0594 | 0.4521 | 0.2536 | 0.2105 | 0.7399 | 0.2002 | 0.7356 | No late-boundary replication |
| InternLM3 8B | InternLM3 | 8B | transformers+torch(cuda) | none | 0.1120 | 0.2540 | 14 | 2 | 0.0272 | 0.0588 | 0.1893 | 0.1284 | 0.6117 | 0.2072 | 0.7463 | No late-boundary replication |
| Yi 1.5 9B | Yi 1.5 | 9B | transformers+torch(cuda) | none | 0.1167 | 0.1553 | 14 | 2 | 0.0232 | 0.1248 | 0.1953 | 0.0946 | 0.6592 | 0.2242 | 0.7044 | No late-boundary replication |
| Mistral Small 22B | Mistral Small | 22B | transformers+torch(cuda) | none | 0.0200 | 0.3113 | 14 | 2 | 0.0395 | 0.1047 | 0.3503 | 0.2286 | 0.5479 | 0.1815 | 0.8146 | No late-boundary replication |
| Llama 3.1 8B | Llama 3.1 | 8B | transformers+torch(cuda) | none | 0.0287 | 0.2113 | 14 | 2 | 0.0371 | 0.1405 | 0.2811 | 0.2800 | 0.6760 | 0.2109 | 0.7496 | No late-boundary replication |

## Drift Audit
| Run | Empirical boundary | Corrected boundary | Fitted boundary | Legacy pooled proxy | Mismatch |
| --- | --- | --- | --- | --- | --- |
| DeepSeek 1.5B | 2 | 2 | 2 | 2 | no |
| DeepSeek 7B | 2 | 2 | 3 | 2 | no |
| Qwen 0.5B | 2 | 2 | 3 | 2 | no |
| Qwen 3B | 2 | 2 | 7 | 2 | no |
| Qwen 7B | 5 | 5 | 5 | 5 | no |
| Qwen 14B | 2 | 2 | not observed | 2 | no |
| Qwen 32B | 6 | 6 | 6 | 6 | no |
| Mistral instruct 7B | 2 | 2 | 5 | 2 | no |
| Phi 4 4B | 2 | 2 | 6 | 2 | no |
| InternLM3 8B | 2 | 2 | 6 | 2 | no |
| Yi 1.5 9B | 2 | 2 | 6 | 2 | no |
| Mistral Small 22B | 2 | 2 | 10 | 2 | no |
| Llama 3.1 8B | 2 | 2 | 7 | 2 | no |

## Detector Rankings
| Run | Detector | Rank | Mean oracle gap | False-late rate |
| --- | --- | --- | --- | --- |
| DeepSeek 1.5B | oracle | 1 | 0.0000 | 0.000 |
| DeepSeek 1.5B | first_answer | 2 | 0.0648 | 0.000 |
| DeepSeek 1.5B | e_process | 3 | 0.1188 | 0.000 |
| DeepSeek 7B | oracle | 1 | 0.0000 | 0.000 |
| DeepSeek 7B | first_answer | 2 | 0.1501 | 0.000 |
| DeepSeek 7B | e_process | 3 | 0.2248 | 0.759 |
| Qwen 0.5B | first_answer | 1 | -0.0415 | 0.000 |
| Qwen 0.5B | oracle | 2 | 0.0000 | 0.000 |
| Qwen 0.5B | e_process | 3 | 0.0072 | 0.000 |
| Qwen 3B | oracle | 1 | 0.0000 | 0.000 |
| Qwen 3B | first_answer | 2 | 0.1030 | 0.000 |
| Qwen 3B | e_process | 3 | 0.1463 | 0.856 |
| Qwen 7B | oracle | 1 | 0.0000 | 0.000 |
| Qwen 7B | hazard_drift | 2 | 0.1820 | 0.667 |
| Qwen 7B | answer_stability | 3 | 0.2360 | 1.000 |
| Qwen 14B | oracle | 1 | 0.0000 | 0.000 |
| Qwen 14B | first_answer | 2 | 0.2173 | 0.000 |
| Qwen 14B | verifier_first_correct | 3 | 0.3225 | 0.574 |
| Qwen 32B | oracle | 1 | 0.0000 | 0.000 |
| Qwen 32B | verifier_first_correct | 2 | 0.2306 | 0.395 |
| Qwen 32B | hazard_drift | 3 | 0.2616 | 0.644 |
| Mistral instruct 7B | first_answer | 1 | -0.0330 | 0.000 |
| Mistral instruct 7B | oracle | 2 | 0.0000 | 0.000 |
| Mistral instruct 7B | e_process | 3 | 0.0523 | 0.000 |
| Phi 4 4B | oracle | 1 | 0.0000 | 0.000 |
| Phi 4 4B | first_answer | 2 | 0.1265 | 0.000 |
| Phi 4 4B | e_process | 3 | 0.2105 | 0.618 |
| InternLM3 8B | oracle | 1 | 0.0000 | 0.000 |
| InternLM3 8B | first_answer | 2 | 0.1037 | 0.000 |
| InternLM3 8B | e_process | 3 | 0.1284 | 0.869 |
| Yi 1.5 9B | oracle | 1 | 0.0000 | 0.000 |
| Yi 1.5 9B | first_answer | 2 | 0.0479 | 0.000 |
| Yi 1.5 9B | e_process | 3 | 0.0946 | 0.000 |
| Mistral Small 22B | oracle | 1 | 0.0000 | 0.000 |
| Mistral Small 22B | first_answer | 2 | 0.1892 | 0.000 |
| Mistral Small 22B | e_process | 3 | 0.2286 | 0.000 |
| Llama 3.1 8B | oracle | 1 | 0.0000 | 0.000 |
| Llama 3.1 8B | first_answer | 2 | 0.2087 | 0.000 |
| Llama 3.1 8B | e_process | 3 | 0.2800 | 0.801 |

## Signal Comparison
| Run | Strongest correctness signal | Strongest corruption signal |
| --- | --- | --- |
| DeepSeek 1.5B | hidden-state cosine drift (hidden_cosine_shift, coeff=-0.804) | verbosity-confidence proxy (verbose_confidence_proxy, coeff=0.209) |
| DeepSeek 7B | entropy volatility (entropy_std, coeff=0.586) | entropy volatility (entropy_std, coeff=0.664) |
| Qwen 0.5B | verbosity-confidence proxy (verbose_confidence_proxy, coeff=-0.591) | hidden-state L2 drift (hidden_l2_shift, coeff=1.852) |
| Qwen 3B | self-reported confidence (confidence, coeff=1.093) | token entropy (entropy_mean, coeff=0.636) |
| Qwen 7B | self-reported confidence (confidence, coeff=2.596) | answer revision flag (answer_changed, coeff=0.993) |
| Qwen 14B | self-reported confidence (confidence, coeff=1.855) | hidden-state L2 drift (hidden_l2_shift, coeff=1.010) |
| Qwen 32B | self-reported confidence (confidence, coeff=1.810) | answer revision flag (answer_changed, coeff=0.663) |
| Mistral instruct 7B | self-reported confidence (confidence, coeff=0.790) | lexical echo (lexical_echo, coeff=0.664) |
| Phi 4 4B | self-reported confidence (confidence, coeff=0.884) | hidden-state L2 drift (hidden_l2_shift, coeff=0.513) |
| InternLM3 8B | entropy volatility (entropy_std, coeff=0.527) | hidden-state L2 drift (hidden_l2_shift, coeff=0.870) |
| Yi 1.5 9B | self-reported confidence (confidence, coeff=1.349) | hidden-state L2 drift (hidden_l2_shift, coeff=0.618) |
| Mistral Small 22B | self-reported confidence (confidence, coeff=1.099) | answer revision flag (answer_changed, coeff=0.764) |
| Llama 3.1 8B | self-reported confidence (confidence, coeff=1.299) | answer revision flag (answer_changed, coeff=0.575) |

## Figures
![Cross-family boundary comparison](outputs/cross_family/cross_family_boundary_comparison.png)

![Cross-family detector gaps](outputs/cross_family/cross_family_detector_gaps.png)
