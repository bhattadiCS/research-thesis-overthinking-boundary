# L4 Overthinking Results

## Executive Summary
The L4 execution loop completed the environment check, parser repair, GSM8K scaling refactor, and real-trace collection for Phi 4 4B on 1344 runs. The model entered a competent regime immediately, with step-1 accuracy $q_1=0.264$, and reached peak correctness $q_t=0.286$ at step 7. This run clears the current capability gate for a cross-family boundary claim.

## Mathematical Validation
The hazard decomposition exhibits repair rate 0.089 and corruption rate 0.226. The corrected conditional hazard drift crosses zero at step 2, while the raw empirical utility drift crosses at step 2, and the fitted hazard drift estimate crosses at step 2. The never-stop policy loses 0.5881 utility on average relative to the oracle, which is direct evidence that extra reasoning past the boundary is harmful. The new mixture e-process closes part of the gap to the hazard rule with mean oracle gap 0.2629.

## Drift Audit
| Drift Curve | First zero crossing | Role |
| --- | ---: | --- |
| empirical utility drift | 2 | raw mean $\Delta U_t$ from realized utilities |
| conditional hazard drift | 2 | theorem-facing $((1-q_t)\alpha_t - q_t\beta_t - c)$ witness |
| fitted hazard drift | 2 | model-based estimate from learned probes |
| pooled proxy drift | 2 | legacy unconditional proxy kept for auditability only |

## Observables Evaluation
The strongest correctness proxy in the fitted models was self-reported confidence (confidence, coeff=0.132). The strongest corruption-side signal was answer revision flag (answer_changed, coeff=0.515). Those coefficients identify the dominant correctness and corruption observables for this run without assuming they transfer unchanged across model families.

## Stopping Comparison
| Policy | Mean stop step | Mean utility | Mean oracle gap |
| --- | ---: | ---: | ---: |
| oracle | 2.73 | 0.4246 | 0.0000 |
| hazard_drift | 2.55 | 0.2036 | 0.2210 |
| e_process | 3.20 | 0.1616 | 0.2629 |
| empirical_bernstein | 9.00 | -0.1180 | 0.5426 |
| never_stop | 10.00 | -0.1635 | 0.5881 |

## Graphs
### Drift Crossing Proof
![Drift crossing proof](drift_crossing_proof.png)

### Detector Gap Comparison
![Detector gap comparison](real_trace_detector_gaps.png)

### Feature Weight Summary
![Feature weight summary](real_trace_feature_weights.png)
