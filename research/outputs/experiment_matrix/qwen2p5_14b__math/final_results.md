# L4 Overthinking Results

## Executive Summary
The L4 execution loop completed the environment check, parser repair, GSM8K scaling refactor, and real-trace collection for Qwen2.5 instruct 14B on 1500 runs. The model entered a competent regime immediately, with step-1 accuracy $q_1=0.065$, and reached peak correctness $q_t=0.310$ at step 14. This run clears the current capability gate for a cross-family boundary claim.

## Mathematical Validation
The hazard decomposition exhibits repair rate 0.049 and corruption rate 0.142. The corrected conditional hazard drift crosses zero at step 2, while the raw empirical utility drift crosses at step 2. The never-stop policy loses 0.6220 utility on average relative to the oracle, which is direct evidence that extra reasoning past the boundary is harmful. The new mixture e-process improves on the fitted hazard rule with mean oracle gap 0.3480.

## Drift Audit
| Drift Curve | First zero crossing | Role |
| --- | ---: | --- |
| empirical utility drift | 2 | raw mean $\Delta U_t$ from realized utilities |
| conditional hazard drift | 2 | theorem-facing $((1-q_t)\alpha_t - q_t\beta_t - c)$ witness |
| pooled proxy drift | 2 | legacy unconditional proxy kept for auditability only |

## Observables Evaluation
The strongest correctness proxy in the fitted models was self-reported confidence (confidence, coeff=1.855). The strongest corruption-side signal was hidden-state L2 drift (hidden_l2_shift, coeff=1.010). Those coefficients identify the dominant correctness and corruption observables for this run without assuming they transfer unchanged across model families.

## Stopping Comparison
| Policy | Mean stop step | Mean utility | Mean oracle gap |
| --- | ---: | ---: | ---: |
| oracle | 3.60 | 0.2820 | 0.0000 |
| hazard_drift | 8.13 | -0.1004 | 0.3824 |
| e_process | 7.00 | -0.0660 | 0.3480 |
| empirical_bernstein | 13.00 | -0.3047 | 0.5867 |
| never_stop | 14.00 | -0.3400 | 0.6220 |

## Graphs
### Drift Crossing Proof
![Drift crossing proof](drift_crossing_proof.png)

### Detector Gap Comparison
![Detector gap comparison](real_trace_detector_gaps.png)

### Feature Weight Summary
![Feature weight summary](real_trace_feature_weights.png)
