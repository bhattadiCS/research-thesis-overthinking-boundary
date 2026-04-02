# L4 Overthinking Results

## Executive Summary
The L4 execution loop completed the environment check, parser repair, GSM8K scaling refactor, and real-trace collection for DeepSeek-R1 distill 1.5B on 900 runs. The model entered a competent regime immediately, with step-1 accuracy $q_1=0.237$, and reached peak correctness $q_t=0.320$ at step 10. This run clears the current capability gate for a cross-family boundary claim.

## Mathematical Validation
The hazard decomposition exhibits repair rate 0.189 and corruption rate 0.461. The corrected conditional hazard drift crosses zero at step 1, while the raw empirical utility drift crosses at step 1, and the fitted hazard drift estimate crosses at step 1. The never-stop policy loses 0.7463 utility on average relative to the oracle, which is direct evidence that extra reasoning past the boundary is harmful. The new mixture e-process closes part of the gap to the hazard rule with mean oracle gap 0.4441. A previous report cited step 7 from a pooled proxy drift built from unconditional transition frequencies; that proxy is retained only as an audit trail and is no longer used as the boundary witness.

## Drift Audit
| Drift Curve | First zero crossing | Role |
| --- | ---: | --- |
| empirical utility drift | 1 | raw mean $\Delta U_t$ from realized utilities |
| conditional hazard drift | 1 | theorem-facing $((1-q_t)\alpha_t - q_t\beta_t - c)$ witness |
| fitted hazard drift | 1 | model-based estimate from learned probes |
| pooled proxy drift | 7 | legacy unconditional proxy kept for auditability only |

## Observables Evaluation
The strongest correctness proxy in the fitted models was answer revision flag (answer_changed, coeff=-0.618). The strongest corruption-side signal was answer revision flag (answer_changed, coeff=0.396). Those coefficients identify the dominant correctness and corruption observables for this run without assuming they transfer unchanged across model families.

## Stopping Comparison
| Policy | Mean stop step | Mean utility | Mean oracle gap |
| --- | ---: | ---: | ---: |
| oracle | 2.47 | 0.6163 | 0.0000 |
| hazard_drift | 3.60 | 0.2042 | 0.4121 |
| e_process | 3.00 | 0.1722 | 0.4441 |
| empirical_bernstein | 9.00 | -0.0978 | 0.7141 |
| never_stop | 10.00 | -0.1300 | 0.7463 |

## Graphs
### Drift Crossing Proof
![Drift crossing proof](../research/outputs/real_traces_l4_deepseek_1p5b/drift_crossing_proof.png)

### Detector Gap Comparison
![Detector gap comparison](../research/outputs/real_traces_l4_deepseek_1p5b/real_trace_detector_gaps.png)

### Feature Weight Summary
![Feature weight summary](../research/outputs/real_traces_l4_deepseek_1p5b/real_trace_feature_weights.png)
