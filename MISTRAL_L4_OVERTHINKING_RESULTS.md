# Mistral 7B L4 Overthinking Results

## Executive Summary
The L4 execution loop completed the environment check, parser repair, GSM8K scaling refactor, and real-trace collection for Mistral instruct 7B on 900 runs. The model entered a competent regime immediately, with step-1 accuracy $q_1=0.302$, and reached peak correctness $q_t=0.319$ at step 10. This run clears the current capability gate for a cross-family boundary claim.

## Mathematical Validation
The hazard decomposition exhibits repair rate 0.055 and corruption rate 0.138. The corrected conditional hazard drift crosses zero at step 3, while the raw empirical utility drift crosses at step 3, and the fitted hazard drift estimate crosses at step 5. The never-stop policy loses 0.5696 utility on average relative to the oracle, which is direct evidence that extra reasoning past the boundary is harmful. The new mixture e-process closes part of the gap to the hazard rule with mean oracle gap 0.3351.

## Drift Audit
| Drift Curve | First zero crossing | Role |
| --- | ---: | --- |
| empirical utility drift | 3 | raw mean $\Delta U_t$ from realized utilities |
| conditional hazard drift | 3 | theorem-facing $((1-q_t)\alpha_t - q_t\beta_t - c)$ witness |
| fitted hazard drift | 5 | model-based estimate from learned probes |
| pooled proxy drift | 3 | legacy unconditional proxy kept for auditability only |

## Observables Evaluation
The strongest correctness proxy in the fitted models was answer revision flag (answer_changed, coeff=-0.528). The strongest corruption-side signal was token entropy (entropy_mean, coeff=0.563). Those coefficients identify the dominant correctness and corruption observables for this run without assuming they transfer unchanged across model families.

## Stopping Comparison
| Policy | Mean stop step | Mean utility | Mean oracle gap |
| --- | ---: | ---: | ---: |
| oracle | 1.34 | 0.4384 | 0.0000 |
| hazard_drift | 4.21 | 0.1604 | 0.2781 |
| e_process | 5.00 | 0.1033 | 0.3351 |
| empirical_bernstein | 9.00 | -0.0878 | 0.5262 |
| never_stop | 10.00 | -0.1311 | 0.5696 |

## Graphs
### Drift Crossing Proof
![Drift crossing proof](research/outputs/real_traces_l4_mistral_7b/drift_crossing_proof.png)

### Detector Gap Comparison
![Detector gap comparison](research/outputs/real_traces_l4_mistral_7b/real_trace_detector_gaps.png)

### Feature Weight Summary
![Feature weight summary](research/outputs/real_traces_l4_mistral_7b/real_trace_feature_weights.png)
