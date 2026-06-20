# L4 Overthinking Results

## Executive Summary
The L4 execution loop completed the environment check, parser repair, GSM8K scaling refactor, and real-trace collection for Yi 1.5 9B on 1500 runs. The model entered a competent regime immediately, with step-1 accuracy $q_1=0.095$, and reached peak correctness $q_t=0.381$ at step 10. This run clears the current capability gate for a cross-family boundary claim.

## Mathematical Validation
The hazard decomposition exhibits repair rate 0.100 and corruption rate 0.155. The corrected conditional hazard drift crosses zero at step 3, while the raw empirical utility drift crosses at step 3, and the fitted hazard drift estimate crosses at step 4. The never-stop policy loses 0.4968 utility on average relative to the oracle, which is direct evidence that extra reasoning past the boundary is harmful. The new mixture e-process closes part of the gap to the hazard rule with mean oracle gap 0.2341.

## Drift Audit
| Drift Curve | First zero crossing | Role |
| --- | ---: | --- |
| empirical utility drift | 3 | raw mean $\Delta U_t$ from realized utilities |
| conditional hazard drift | 3 | theorem-facing $((1-q_t)\alpha_t - q_t\beta_t - c)$ witness |
| fitted hazard drift | 4 | model-based estimate from learned probes |
| pooled proxy drift | 3 | legacy unconditional proxy kept for auditability only |

## Observables Evaluation
The strongest correctness proxy in the fitted models was self-reported confidence (confidence, coeff=0.994). The strongest corruption-side signal was token entropy (entropy_mean, coeff=0.718). Those coefficients identify the dominant correctness and corruption observables for this run without assuming they transfer unchanged across model families.

## Stopping Comparison
| Policy | Mean stop step | Mean utility | Mean oracle gap |
| --- | ---: | ---: | ---: |
| oracle | 2.58 | 0.4274 | 0.0000 |
| hazard_drift | 3.91 | 0.2197 | 0.2077 |
| e_process | 3.60 | 0.1933 | 0.2341 |
| empirical_bernstein | 9.00 | -0.0333 | 0.4608 |
| never_stop | 10.00 | -0.0693 | 0.4968 |

## Graphs
### Drift Crossing Proof
![Drift crossing proof](drift_crossing_proof.png)

### Detector Gap Comparison
![Detector gap comparison](real_trace_detector_gaps.png)

### Feature Weight Summary
![Feature weight summary](real_trace_feature_weights.png)
