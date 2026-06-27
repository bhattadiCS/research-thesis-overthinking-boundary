# Cross-Family Report

## Executive Summary
No run shows a clearly late corrected boundary, but 1 capable run(s) across 1 family/families (Mistral Small) show weaker late-boundary support: Mistral Small 22B (step 3).

Task IDs align across all 13 runs under the shared GSM8K train split and shuffle seed 17 protocol.

## Run Summary
| Run | Family | Params | Backend | Quant | Step-1 acc | Peak acc | Peak step | Corrected boundary | Repair | Corruption | Hazard gap | E-process gap | Never-stop gap | Probe Brier | Probe AUC | Assessment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek 1.5B | DeepSeek-R1 distill | 1.5B | transformers+torch(cuda) | none | 0.2607 | 0.2853 | 8 | 2 | 0.1735 | 0.4576 | 0.3139 | 0.3844 | 0.6017 | 0.2459 | 0.5566 | No late-boundary replication |
| DeepSeek 7B | DeepSeek-R1 distill | 7B | transformers+torch(cuda) | none | 0.3560 | 0.5447 | 5 | 2 | 0.3233 | 0.2907 | 0.2976 | 0.4878 | 0.5864 | 0.2073 | 0.7316 | No late-boundary replication |
| Qwen 0.5B | Qwen2.5 instruct | 0.5B | transformers+torch(cuda) | none | 0.3873 | 0.3987 | 6 | 2 | 0.0532 | 0.0796 | 0.0958 | 0.0774 | 0.3687 | 0.2500 | 0.5266 | No late-boundary replication |
| Qwen 3B | Qwen2.5 instruct | 3B | transformers+torch(cuda) | none | 0.7100 | 0.7100 | 1 | 2 | 0.0928 | 0.0453 | 0.0824 | 0.0801 | 0.3754 | 0.2297 | 0.6626 | No late-boundary replication |
| Qwen 7B | Qwen2.5 instruct | 7B | transformers+torch(cuda) | none | 0.8720 | 0.9040 | 5 | 2 | 0.0763 | 0.0052 | 0.0523 | 0.0166 | 0.3086 | 0.2294 | 0.6425 | No late-boundary replication |
| Qwen 14B | Qwen2.5 instruct | 14B | transformers+torch(cuda) | none | 0.9340 | 0.9433 | 3 | 2 | 0.0471 | 0.0019 | 0.0473 | 0.0063 | 0.3056 | 0.2198 | 0.6791 | No late-boundary replication |
| Qwen 32B | Qwen2.5 instruct | 32B | transformers+torch(cuda) | none | 0.9053 | 0.9567 | 6 | 2 | 0.1161 | 0.0014 | 0.0839 | 0.0179 | 0.3039 | 0.2009 | 0.6945 | No late-boundary replication |
| Mistral instruct 7B | Mistral instruct | 7B | transformers+torch(cuda) | none | 0.7233 | 0.7233 | 1 | 2 | 0.0969 | 0.0463 | 0.0753 | 0.0709 | 0.3762 | 0.2261 | 0.6601 | No late-boundary replication |
| Phi 4 4B | Phi 4 | 4B | transformers+torch(cuda) | none | 0.6147 | 0.7087 | 6 | 2 | 0.4144 | 0.1862 | 0.1530 | 0.4209 | 0.4736 | 0.2006 | 0.7468 | No late-boundary replication |
| InternLM3 8B | InternLM3 | 8B | transformers+torch(cuda) | none | 0.8833 | 0.8960 | 7 | 2 | 0.0505 | 0.0044 | 0.0388 | 0.0134 | 0.3041 | 0.2355 | 0.6099 | No late-boundary replication |
| Yi 1.5 9B | Yi 1.5 | 9B | transformers+torch(cuda) | none | 0.8140 | 0.8167 | 2 | 2 | 0.1287 | 0.0319 | 0.0540 | 0.0536 | 0.3590 | 0.2146 | 0.6755 | No late-boundary replication |
| Mistral Small 22B | Mistral Small | 22B | transformers+torch(cuda) | none | 0.5360 | 0.8907 | 8 | 3 | 0.2347 | 0.0159 | 0.1010 | 0.1663 | 0.3163 | 0.1814 | 0.7762 | Weakened late-boundary support |
| Llama 3.1 8B | Llama 3.1 | 8B | transformers+torch(cuda) | none | 0.5833 | 0.7720 | 7 | 2 | 0.2629 | 0.0667 | 0.1332 | 0.2082 | 0.4322 | 0.1962 | 0.7481 | No late-boundary replication |

## Drift Audit
| Run | Empirical boundary | Corrected boundary | Fitted boundary | Legacy pooled proxy | Mismatch |
| --- | --- | --- | --- | --- | --- |
| DeepSeek 1.5B | 2 | 2 | 2 | 2 | no |
| DeepSeek 7B | 2 | 2 | 3 | 2 | no |
| Qwen 0.5B | 2 | 2 | 2 | 2 | no |
| Qwen 3B | 2 | 2 | 2 | 2 | no |
| Qwen 7B | 2 | 2 | 3 | 2 | no |
| Qwen 14B | 2 | 2 | 3 | 2 | no |
| Qwen 32B | 2 | 2 | 4 | 2 | no |
| Mistral instruct 7B | 2 | 2 | 2 | 2 | no |
| Phi 4 4B | 2 | 2 | 2 | 2 | no |
| InternLM3 8B | 2 | 2 | 2 | 2 | no |
| Yi 1.5 9B | 2 | 2 | 2 | 2 | no |
| Mistral Small 22B | 3 | 3 | 3 | 3 | no |
| Llama 3.1 8B | 2 | 2 | 2 | 2 | no |

## Detector Rankings
| Run | Detector | Rank | Mean oracle gap | False-late rate |
| --- | --- | --- | --- | --- |
| DeepSeek 1.5B | oracle | 1 | 0.0000 | 0.000 |
| DeepSeek 1.5B | verifier_first_correct | 2 | 0.0470 | 0.335 |
| DeepSeek 1.5B | first_answer | 3 | 0.2764 | 0.000 |
| DeepSeek 7B | oracle | 1 | 0.0000 | 0.000 |
| DeepSeek 7B | verifier_first_correct | 2 | 0.0021 | 0.142 |
| DeepSeek 7B | hazard_drift | 3 | 0.2976 | 0.382 |
| Qwen 0.5B | oracle | 1 | 0.0000 | 0.000 |
| Qwen 0.5B | first_answer | 2 | 0.0274 | 0.000 |
| Qwen 0.5B | e_process | 3 | 0.0774 | 0.000 |
| Qwen 3B | verifier_first_correct | 1 | -0.0115 | 0.195 |
| Qwen 3B | oracle | 2 | 0.0000 | 0.000 |
| Qwen 3B | first_answer | 3 | 0.0054 | 0.000 |
| Qwen 7B | verifier_first_correct | 1 | -0.0362 | 0.073 |
| Qwen 7B | first_answer | 2 | -0.0101 | 0.000 |
| Qwen 7B | oracle | 3 | 0.0000 | 0.000 |
| Qwen 14B | first_answer | 1 | -0.0364 | 0.000 |
| Qwen 14B | verifier_first_correct | 2 | -0.0347 | 0.049 |
| Qwen 14B | oracle | 3 | 0.0000 | 0.000 |
| Qwen 32B | verifier_first_correct | 1 | -0.0382 | 0.035 |
| Qwen 32B | oracle | 2 | 0.0000 | 0.000 |
| Qwen 32B | first_answer | 3 | 0.0052 | 0.000 |
| Mistral instruct 7B | verifier_first_correct | 1 | -0.0194 | 0.184 |
| Mistral instruct 7B | oracle | 2 | 0.0000 | 0.000 |
| Mistral instruct 7B | first_answer | 3 | 0.0002 | 0.000 |
| Phi 4 4B | verifier_first_correct | 1 | -0.0403 | 0.074 |
| Phi 4 4B | oracle | 2 | 0.0000 | 0.000 |
| Phi 4 4B | hazard_drift | 3 | 0.1530 | 0.229 |
| InternLM3 8B | first_answer | 1 | -0.0332 | 0.000 |
| InternLM3 8B | verifier_first_correct | 2 | -0.0304 | 0.087 |
| InternLM3 8B | oracle | 3 | 0.0000 | 0.000 |
| Yi 1.5 9B | verifier_first_correct | 1 | -0.0269 | 0.106 |
| Yi 1.5 9B | oracle | 2 | 0.0000 | 0.000 |
| Yi 1.5 9B | first_answer | 3 | 0.0063 | 0.000 |
| Mistral Small 22B | verifier_first_correct | 1 | -0.0116 | 0.071 |
| Mistral Small 22B | oracle | 2 | 0.0000 | 0.000 |
| Mistral Small 22B | answer_stability | 3 | 0.0987 | 0.381 |
| Llama 3.1 8B | verifier_first_correct | 1 | -0.0246 | 0.067 |
| Llama 3.1 8B | oracle | 2 | 0.0000 | 0.000 |
| Llama 3.1 8B | answer_stability | 3 | 0.1238 | 0.397 |

## Signal Comparison
| Run | Strongest correctness signal | Strongest corruption signal |
| --- | --- | --- |
| DeepSeek 1.5B | verbosity-confidence proxy (verbose_confidence_proxy, coeff=-0.162) | answer revision flag (answer_changed, coeff=0.520) |
| DeepSeek 7B | self-reported confidence (confidence, coeff=0.835) | answer revision flag (answer_changed, coeff=0.364) |
| Qwen 0.5B | verbosity-confidence proxy (verbose_confidence_proxy, coeff=-0.165) | hidden-state L2 drift (hidden_l2_shift, coeff=0.848) |
| Qwen 3B | self-reported confidence (confidence, coeff=0.762) | entropy volatility (entropy_std, coeff=0.624) |
| Qwen 7B | token entropy (entropy_mean, coeff=-0.758) | verbosity-confidence proxy (verbose_confidence_proxy, coeff=1.086) |
| Qwen 14B | self-reported confidence (confidence, coeff=0.589) | reasoning length (thought_token_count, coeff=1.055) |
| Qwen 32B | self-reported confidence (confidence, coeff=1.789) | self-reported confidence (confidence, coeff=4.455) |
| Mistral instruct 7B | verbosity-confidence proxy (verbose_confidence_proxy, coeff=-0.447) | hidden-state L2 drift (hidden_l2_shift, coeff=0.865) |
| Phi 4 4B | self-reported confidence (confidence, coeff=0.761) | verbosity-confidence proxy (verbose_confidence_proxy, coeff=0.468) |
| InternLM3 8B | reasoning length (thought_token_count, coeff=-0.288) | hidden-state L2 drift (hidden_l2_shift, coeff=1.354) |
| Yi 1.5 9B | self-reported confidence (confidence, coeff=0.610) | answer revision flag (answer_changed, coeff=0.564) |
| Mistral Small 22B | self-reported confidence (confidence, coeff=1.850) | answer revision flag (answer_changed, coeff=0.645) |
| Llama 3.1 8B | self-reported confidence (confidence, coeff=0.691) | reasoning length (thought_token_count, coeff=0.559) |

## Figures
![Cross-family boundary comparison](outputs/cross_family/cross_family_boundary_comparison.png)

![Cross-family detector gaps](outputs/cross_family/cross_family_detector_gaps.png)
