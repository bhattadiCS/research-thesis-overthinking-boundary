# L4 Overthinking Results

## Executive Summary
The L4 execution loop completed the environment check, parser repair, GSM8K scaling refactor, and real-trace collection for Mistral instruct 7B on 1500 runs. The model entered a competent regime immediately, with step-1 accuracy $q_1=0.295$, and reached peak correctness $q_t=0.320$ at step 10. This run clears the current capability gate for a cross-family boundary claim.

## Mathematical Validation
The hazard decomposition exhibits repair rate 0.053 and corruption rate 0.134. The corrected conditional hazard drift crosses zero at step 3, while the raw empirical utility drift crosses at step 3, and the fitted hazard drift estimate crosses at step 5. The never-stop policy loses 0.5533 utility on average relative to the oracle, which is direct evidence that extra reasoning past the boundary is harmful. The new mixture e-process improves on the fitted hazard rule with mean oracle gap 0.2506.

## Drift Audit
| Drift Curve | First zero crossing | Role |
| --- | ---: | --- |
| empirical utility drift | 3 | raw mean $\Delta U_t$ from realized utilities |
| conditional hazard drift | 3 | theorem-facing $((1-q_t)\alpha_t - q_t\beta_t - c)$ witness |
| fitted hazard drift | 5 | model-based estimate from learned probes |
| pooled proxy drift | 3 | legacy unconditional proxy kept for auditability only |

## Observables Evaluation
The strongest correctness proxy in the fitted models was answer revision flag (answer_changed, coeff=-0.552). The strongest corruption-side signal was answer revision flag (answer_changed, coeff=0.581). Those coefficients identify the dominant correctness and corruption observables for this run without assuming they transfer unchanged across model families.

## Stopping Comparison
| Policy | Mean stop step | Mean utility | Mean oracle gap |
| --- | ---: | ---: | ---: |
| oracle | 1.32 | 0.4233 | 0.0000 |
| hazard_drift | 4.20 | 0.1586 | 0.2647 |
| e_process | 3.00 | 0.1727 | 0.2506 |
| empirical_bernstein | 7.00 | 0.0160 | 0.4073 |
| never_stop | 10.00 | -0.1300 | 0.5533 |

## Graphs
### Drift Crossing Proof
![Drift crossing proof](drift_crossing_proof.png)

### Detector Gap Comparison
![Detector gap comparison](real_trace_detector_gaps.png)

### Feature Weight Summary
![Feature weight summary](real_trace_feature_weights.png)
