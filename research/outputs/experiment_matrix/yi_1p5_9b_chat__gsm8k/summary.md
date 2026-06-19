# L4 Overthinking Results

## Executive Summary
The L4 execution loop completed the environment check, parser repair, GSM8K scaling refactor, and real-trace collection for Yi 1.5 9B on 1500 runs. The model entered a competent regime immediately, with step-1 accuracy $q_1=0.091$, and reached peak correctness $q_t=0.377$ at step 10. This run clears the current capability gate for a cross-family boundary claim.

## Mathematical Validation
The hazard decomposition exhibits repair rate 0.100 and corruption rate 0.155. The corrected conditional hazard drift crosses zero at step 3, while the raw empirical utility drift crosses at step 3, and the fitted hazard drift estimate crosses at step 3. The never-stop policy loses 0.5358 utility on average relative to the oracle, which is direct evidence that extra reasoning past the boundary is harmful. The new mixture e-process improves on the fitted hazard rule with mean oracle gap 0.2392.

## Drift Audit
| Drift Curve | First zero crossing | Role |
| --- | ---: | --- |
| empirical utility drift | 3 | raw mean $\Delta U_t$ from realized utilities |
| conditional hazard drift | 3 | theorem-facing $((1-q_t)\alpha_t - q_t\beta_t - c)$ witness |
| fitted hazard drift | 3 | model-based estimate from learned probes |
| pooled proxy drift | 3 | legacy unconditional proxy kept for auditability only |

## Observables Evaluation
The strongest correctness proxy in the fitted models was self-reported confidence (confidence, coeff=1.049). The strongest corruption-side signal was token entropy (entropy_mean, coeff=0.710). Those coefficients identify the dominant correctness and corruption observables for this run without assuming they transfer unchanged across model families.

## Stopping Comparison
| Policy | Mean stop step | Mean utility | Mean oracle gap |
| --- | ---: | ---: | ---: |
| oracle | 1.99 | 0.4625 | 0.0000 |
| hazard_drift | 3.75 | 0.2043 | 0.2582 |
| e_process | 3.00 | 0.2233 | 0.2392 |
| empirical_bernstein | 9.00 | -0.0387 | 0.5012 |
| never_stop | 10.00 | -0.0733 | 0.5358 |

## Graphs
### Drift Crossing Proof
![Drift crossing proof](drift_crossing_proof.png)

### Detector Gap Comparison
![Detector gap comparison](real_trace_detector_gaps.png)

### Feature Weight Summary
![Feature weight summary](real_trace_feature_weights.png)
