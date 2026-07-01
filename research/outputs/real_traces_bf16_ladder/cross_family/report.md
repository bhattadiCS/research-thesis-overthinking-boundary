# Cross-Family Report

## Executive Summary
A clearly late corrected boundary replicates in 3 capable run(s) across 1 family/families: Qwen 7B (step 5); Qwen 14B (step 5); Qwen 32B (step 5). Late-boundary evidence remains confined to a single capable family, so cross-family robustness is not yet established.

Task IDs align across all 3 runs under the shared GSM8K train split and shuffle seed 17 protocol.

## Run Summary
| Run | Family | Params | Backend | Quant | Step-1 acc | Peak acc | Peak step | Corrected boundary | Repair | Corruption | Hazard gap | E-process gap | Never-stop gap | Probe Brier | Probe AUC | Assessment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen 7B | Qwen2.5 instruct | 7B | transformers+torch(cuda) | none | 0.2740 | 0.7053 | 10 | 5 | 0.1215 | 0.0635 | 0.1672 | 0.1978 | 0.3878 | 0.1960 | 0.7525 | Late-boundary replication |
| Qwen 14B | Qwen2.5 instruct | 14B | transformers+torch(cuda) | none | 0.0300 | 0.4733 | 9 | 5 | 0.1439 | 0.2570 | 0.3535 | 0.3990 | 0.5676 | 0.1026 | 0.9122 | Late-boundary replication |
| Qwen 32B | Qwen2.5 instruct | 32B | transformers+torch(cuda) | none | 0.0413 | 0.8787 | 10 | 5 | 0.2274 | 0.0611 | 0.1835 | 0.1889 | 0.3529 | 0.1500 | 0.8346 | Late-boundary replication |

## Drift Audit
| Run | Empirical boundary | Corrected boundary | Fitted boundary | Legacy pooled proxy | Mismatch |
| --- | --- | --- | --- | --- | --- |
| Qwen 7B | 5 | 5 | 5 | 5 | no |
| Qwen 14B | 5 | 5 | 6 | 5 | no |
| Qwen 32B | 5 | 5 | 5 | 5 | no |

## Detector Rankings
| Run | Detector | Rank | Mean oracle gap | False-late rate |
| --- | --- | --- | --- | --- |
| Qwen 7B | oracle | 1 | 0.0000 | 0.000 |
| Qwen 7B | verifier_first_correct | 2 | 0.0508 | 0.239 |
| Qwen 7B | answer_stability | 3 | 0.1543 | 0.892 |
| Qwen 14B | oracle | 1 | 0.0000 | 0.000 |
| Qwen 14B | verifier_first_correct | 2 | 0.0956 | 0.277 |
| Qwen 14B | hazard_drift | 3 | 0.3535 | 0.438 |
| Qwen 32B | oracle | 1 | 0.0000 | 0.000 |
| Qwen 32B | verifier_first_correct | 2 | 0.0376 | 0.102 |
| Qwen 32B | hazard_drift | 3 | 0.1835 | 0.783 |

## Signal Comparison
| Run | Strongest correctness signal | Strongest corruption signal |
| --- | --- | --- |
| Qwen 7B | self-reported confidence (confidence, coeff=0.740) | token entropy (entropy_mean, coeff=1.431) |
| Qwen 14B | hidden-state L2 drift (hidden_l2_shift, coeff=1.755) | answer revision flag (answer_changed, coeff=0.942) |
| Qwen 32B | hidden-state L2 drift (hidden_l2_shift, coeff=1.152) | hidden-state L2 drift (hidden_l2_shift, coeff=1.537) |

## Figures
![Cross-family boundary comparison](outputs/cross_family/cross_family_boundary_comparison.png)

![Cross-family detector gaps](outputs/cross_family/cross_family_detector_gaps.png)
