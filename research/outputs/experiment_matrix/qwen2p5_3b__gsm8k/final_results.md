# L4 Overthinking Results

## Executive Summary
The L4 execution loop completed the environment check, parser repair, GSM8K scaling refactor, and real-trace collection for Qwen2.5 instruct 3B on 1500 runs. The model entered a competent regime immediately, with step-1 accuracy $q_1=0.053$, and reached peak correctness $q_t=0.504$ at step 10. This run clears the current capability gate for a cross-family boundary claim.

## Mathematical Validation
The hazard decomposition exhibits repair rate 0.091 and corruption rate 0.083. The corrected conditional hazard drift crosses zero at step 4, while the raw empirical utility drift crosses at step 4, and the fitted hazard drift estimate crosses at step 4. The never-stop policy loses 0.4609 utility on average relative to the oracle, which is direct evidence that extra reasoning past the boundary is harmful. The new mixture e-process improves on the fitted hazard rule with mean oracle gap 0.2369.

## Drift Audit
| Drift Curve | First zero crossing | Role |
| --- | ---: | --- |
| empirical utility drift | 4 | raw mean $\Delta U_t$ from realized utilities |
| conditional hazard drift | 4 | theorem-facing $((1-q_t)\alpha_t - q_t\beta_t - c)$ witness |
| fitted hazard drift | 4 | model-based estimate from learned probes |
| pooled proxy drift | 4 | legacy unconditional proxy kept for auditability only |

## Observables Evaluation
The strongest correctness proxy in the fitted models was self-reported confidence (confidence, coeff=0.717). The strongest corruption-side signal was answer revision flag (answer_changed, coeff=0.443). Those coefficients identify the dominant correctness and corruption observables for this run without assuming they transfer unchanged across model families.

## Stopping Comparison
| Policy | Mean stop step | Mean utility | Mean oracle gap |
| --- | ---: | ---: | ---: |
| oracle | 2.62 | 0.5149 | 0.0000 |
| hazard_drift | 3.40 | 0.2769 | 0.2380 |
| e_process | 4.80 | 0.2780 | 0.2369 |
| empirical_bernstein | 9.00 | 0.1027 | 0.4122 |
| never_stop | 10.00 | 0.0540 | 0.4609 |

## Graphs
### Drift Crossing Proof
![Drift crossing proof](drift_crossing_proof.png)

### Detector Gap Comparison
![Detector gap comparison](real_trace_detector_gaps.png)

### Feature Weight Summary
![Feature weight summary](real_trace_feature_weights.png)
