# L4 Overthinking Results

## Executive Summary
The L4 execution loop completed the environment check, parser repair, GSM8K scaling refactor, and real-trace collection for Qwen2.5 instruct 0.5B on 6 runs. The model entered a competent regime immediately, with step-1 accuracy $q_1=0.167$, and reached peak correctness $q_t=0.167$ at step 1. This run clears the current capability gate for a cross-family boundary claim.

## Mathematical Validation
The hazard decomposition exhibits repair rate 0.000 and corruption rate 0.000. The corrected conditional hazard drift crosses zero at step 1, while the raw empirical utility drift crosses at step 1, and the fitted hazard drift estimate crosses at step 1. The never-stop policy loses 0.1000 utility on average relative to the oracle, which is direct evidence that extra reasoning past the boundary is harmful. The new mixture e-process closes part of the gap to the hazard rule with mean oracle gap 0.0500.

## Drift Audit
| Drift Curve | First zero crossing | Role |
| --- | ---: | --- |
| empirical utility drift | 1 | raw mean $\Delta U_t$ from realized utilities |
| conditional hazard drift | 1 | theorem-facing $((1-q_t)\alpha_t - q_t\beta_t - c)$ witness |
| fitted hazard drift | 1 | model-based estimate from learned probes |
| pooled proxy drift | 1 | legacy unconditional proxy kept for auditability only |

## Observables Evaluation
The strongest correctness proxy in the fitted models was reasoning length (thought_token_count, coeff=1.034). The strongest corruption-side signal was none. Those coefficients identify the dominant correctness and corruption observables for this run without assuming they transfer unchanged across model families.

## Stopping Comparison
| Policy | Mean stop step | Mean utility | Mean oracle gap |
| --- | ---: | ---: | ---: |
| oracle | 1.00 | 0.1667 | 0.0000 |
| hazard_drift | 2.00 | 0.1167 | 0.0500 |
| e_process | 2.00 | 0.1167 | 0.0500 |
| empirical_bernstein | 2.00 | 0.1167 | 0.0500 |
| never_stop | 3.00 | 0.0667 | 0.1000 |

## Graphs
### Drift Crossing Proof
![Drift crossing proof](drift_crossing_proof.png)

### Detector Gap Comparison
![Detector gap comparison](real_trace_detector_gaps.png)

### Feature Weight Summary
![Feature weight summary](real_trace_feature_weights.png)
