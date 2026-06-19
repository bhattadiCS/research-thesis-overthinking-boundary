# L4 Overthinking Results

## Executive Summary
The L4 execution loop completed the environment check, parser repair, GSM8K scaling refactor, and real-trace collection for Phi 4 4B on 1500 runs. The model entered a competent regime immediately, with step-1 accuracy $q_1=0.324$, and reached peak correctness $q_t=0.341$ at step 10. This run clears the current capability gate for a cross-family boundary claim.

## Mathematical Validation
The hazard decomposition exhibits repair rate 0.159 and corruption rate 0.343. The corrected conditional hazard drift crosses zero at step 1, while the raw empirical utility drift crosses at step 1, and the fitted hazard drift estimate crosses at step 3. The never-stop policy loses 0.7795 utility on average relative to the oracle, which is direct evidence that extra reasoning past the boundary is harmful. The new mixture e-process closes part of the gap to the hazard rule with mean oracle gap 0.4609. A previous report cited step 3 from a pooled proxy drift built from unconditional transition frequencies; that proxy is retained only as an audit trail and is no longer used as the boundary witness.

## Drift Audit
| Drift Curve | First zero crossing | Role |
| --- | ---: | --- |
| empirical utility drift | 1 | raw mean $\Delta U_t$ from realized utilities |
| conditional hazard drift | 1 | theorem-facing $((1-q_t)\alpha_t - q_t\beta_t - c)$ witness |
| fitted hazard drift | 3 | model-based estimate from learned probes |
| pooled proxy drift | 3 | legacy unconditional proxy kept for auditability only |

## Observables Evaluation
The strongest correctness proxy in the fitted models was answer revision flag (answer_changed, coeff=-0.567). The strongest corruption-side signal was answer revision flag (answer_changed, coeff=0.325). Those coefficients identify the dominant correctness and corruption observables for this run without assuming they transfer unchanged across model families.

## Stopping Comparison
| Policy | Mean stop step | Mean utility | Mean oracle gap |
| --- | ---: | ---: | ---: |
| oracle | 2.01 | 0.6702 | 0.0000 |
| hazard_drift | 3.76 | 0.2498 | 0.4204 |
| e_process | 3.00 | 0.2093 | 0.4609 |
| empirical_bernstein | 9.00 | -0.0700 | 0.7402 |
| never_stop | 10.00 | -0.1093 | 0.7795 |

## Graphs
### Drift Crossing Proof
![Drift crossing proof](drift_crossing_proof.png)

### Detector Gap Comparison
![Detector gap comparison](real_trace_detector_gaps.png)

### Feature Weight Summary
![Feature weight summary](real_trace_feature_weights.png)
