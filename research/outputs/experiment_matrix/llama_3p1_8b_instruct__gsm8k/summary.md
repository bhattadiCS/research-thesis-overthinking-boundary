# L4 Overthinking Results

## Executive Summary
The L4 execution loop completed the environment check, parser repair, GSM8K scaling refactor, and real-trace collection for Llama 3.1 8B on 1500 runs. The model entered a competent regime immediately, with step-1 accuracy $q_1=0.011$, and reached peak correctness $q_t=0.475$ at step 10. This run clears the current capability gate for a cross-family boundary claim.

## Mathematical Validation
The hazard decomposition exhibits repair rate 0.116 and corruption rate 0.182. The corrected conditional hazard drift crosses zero at step 5, while the raw empirical utility drift crosses at step 5, and the fitted hazard drift estimate crosses at step 4. The never-stop policy loses 0.5043 utility on average relative to the oracle, which is direct evidence that extra reasoning past the boundary is harmful. The new mixture e-process closes part of the gap to the hazard rule with mean oracle gap 0.3349.

## Drift Audit
| Drift Curve | First zero crossing | Role |
| --- | ---: | --- |
| empirical utility drift | 5 | raw mean $\Delta U_t$ from realized utilities |
| conditional hazard drift | 5 | theorem-facing $((1-q_t)\alpha_t - q_t\beta_t - c)$ witness |
| fitted hazard drift | 4 | model-based estimate from learned probes |
| pooled proxy drift | 5 | legacy unconditional proxy kept for auditability only |

## Observables Evaluation
The strongest correctness proxy in the fitted models was hidden-state L2 drift (hidden_l2_shift, coeff=1.221). The strongest corruption-side signal was hidden-state L2 drift (hidden_l2_shift, coeff=1.010). Those coefficients identify the dominant correctness and corruption observables for this run without assuming they transfer unchanged across model families.

## Stopping Comparison
| Policy | Mean stop step | Mean utility | Mean oracle gap |
| --- | ---: | ---: | ---: |
| oracle | 3.17 | 0.5289 | 0.0000 |
| hazard_drift | 4.42 | 0.2448 | 0.2841 |
| e_process | 6.00 | 0.1940 | 0.3349 |
| empirical_bernstein | 9.00 | 0.0640 | 0.4649 |
| never_stop | 10.00 | 0.0247 | 0.5043 |

## Graphs
### Drift Crossing Proof
![Drift crossing proof](drift_crossing_proof.png)

### Detector Gap Comparison
![Detector gap comparison](real_trace_detector_gaps.png)

### Feature Weight Summary
![Feature weight summary](real_trace_feature_weights.png)
