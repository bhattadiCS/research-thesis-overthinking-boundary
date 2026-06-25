# L4 Overthinking Results

## Executive Summary
The L4 execution loop completed the environment check, parser repair, GSM8K scaling refactor, and real-trace collection for Mistral instruct 7B on 1500 runs. The model entered a competent regime immediately, with step-1 accuracy $q_1=0.723$, and reached peak correctness $q_t=0.723$ at step 1. This run clears the current capability gate for a cross-family boundary claim.

## Mathematical Validation
The hazard decomposition exhibits repair rate 0.097 and corruption rate 0.046. The corrected conditional hazard drift crosses zero at step 2, while the raw empirical utility drift crosses at step 2, and the fitted hazard drift estimate crosses at step 2. The never-stop policy loses 0.3762 utility on average relative to the oracle, which is direct evidence that extra reasoning past the boundary is harmful. The new mixture e-process improves on the fitted hazard rule with mean oracle gap 0.0709.

## Drift Audit
| Drift Curve | First zero crossing | Role |
| --- | ---: | --- |
| empirical utility drift | 2 | raw mean $\Delta U_t$ from realized utilities |
| conditional hazard drift | 2 | theorem-facing $((1-q_t)\alpha_t - q_t\beta_t - c)$ witness |
| fitted hazard drift | 2 | model-based estimate from learned probes |
| pooled proxy drift | 2 | legacy unconditional proxy kept for auditability only |

## Observables Evaluation
The strongest correctness proxy in the fitted models was verbosity-confidence proxy (verbose_confidence_proxy, coeff=-0.447). The strongest corruption-side signal was hidden-state L2 drift (hidden_l2_shift, coeff=0.865). Those coefficients identify the dominant correctness and corruption observables for this run without assuming they transfer unchanged across model families.

## Stopping Comparison
| Policy | Mean stop step | Mean utility | Mean oracle gap |
| --- | ---: | ---: | ---: |
| oracle | 2.17 | 0.7235 | 0.0000 |
| hazard_drift | 2.35 | 0.6483 | 0.0753 |
| e_process | 2.00 | 0.6527 | 0.0709 |
| empirical_bernstein | 2.00 | 0.6527 | 0.0709 |
| never_stop | 8.00 | 0.3473 | 0.3762 |

## Graphs
### Drift Crossing Proof
![Drift crossing proof](drift_crossing_proof.png)

### Detector Gap Comparison
![Detector gap comparison](real_trace_detector_gaps.png)

### Feature Weight Summary
![Feature weight summary](real_trace_feature_weights.png)
