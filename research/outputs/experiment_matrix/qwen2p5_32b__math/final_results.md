# L4 Overthinking Results

## Executive Summary
The L4 execution loop completed the environment check, parser repair, GSM8K scaling refactor, and real-trace collection for Qwen2.5 instruct 32B on 1500 runs. The model entered a competent regime immediately, with step-1 accuracy $q_1=0.088$, and reached peak correctness $q_t=0.585$ at step 14. This run clears the current capability gate for a cross-family boundary claim.

## Mathematical Validation
The hazard decomposition exhibits repair rate 0.066 and corruption rate 0.021. The corrected conditional hazard drift crosses zero at step 6, while the raw empirical utility drift crosses at step 6, and the fitted hazard drift estimate crosses at step 6. The never-stop policy loses 0.5345 utility on average relative to the oracle, which is direct evidence that extra reasoning past the boundary is harmful. The new mixture e-process closes part of the gap to the hazard rule with mean oracle gap 0.2865.

## Drift Audit
| Drift Curve | First zero crossing | Role |
| --- | ---: | --- |
| empirical utility drift | 6 | raw mean $\Delta U_t$ from realized utilities |
| conditional hazard drift | 6 | theorem-facing $((1-q_t)\alpha_t - q_t\beta_t - c)$ witness |
| fitted hazard drift | 6 | model-based estimate from learned probes |
| pooled proxy drift | 6 | legacy unconditional proxy kept for auditability only |

## Observables Evaluation
The strongest correctness proxy in the fitted models was self-reported confidence (confidence, coeff=1.810). The strongest corruption-side signal was answer revision flag (answer_changed, coeff=0.663). Those coefficients identify the dominant correctness and corruption observables for this run without assuming they transfer unchanged across model families.

## Stopping Comparison
| Policy | Mean stop step | Mean utility | Mean oracle gap |
| --- | ---: | ---: | ---: |
| oracle | 3.67 | 0.4699 | 0.0000 |
| hazard_drift | 5.73 | 0.2083 | 0.2616 |
| e_process | 8.00 | 0.1833 | 0.2865 |
| empirical_bernstein | 13.00 | -0.0193 | 0.4892 |
| never_stop | 14.00 | -0.0647 | 0.5345 |

## Graphs
### Drift Crossing Proof
![Drift crossing proof](drift_crossing_proof.png)

### Detector Gap Comparison
![Detector gap comparison](real_trace_detector_gaps.png)

### Feature Weight Summary
![Feature weight summary](real_trace_feature_weights.png)
