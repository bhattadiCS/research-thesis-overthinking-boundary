# L4 Overthinking Results

## Executive Summary
The L4 execution loop completed the environment check, parser repair, GSM8K scaling refactor, and real-trace collection for Qwen2.5 instruct 0.5B on 1500 runs. The model entered a competent regime immediately, with step-1 accuracy $q_1=0.074$, and reached peak correctness $q_t=0.081$ at step 5. This run remains below the current capability gate, so it should be treated as a weak-regime control rather than a decisive family-level witness.

## Mathematical Validation
The hazard decomposition exhibits repair rate 0.004 and corruption rate 0.040. The corrected conditional hazard drift crosses zero at step 2, while the raw empirical utility drift crosses at step 2, and the fitted hazard drift estimate crosses at step 2. The never-stop policy loses 0.4067 utility on average relative to the oracle, which is direct evidence that extra reasoning past the boundary is harmful. The new mixture e-process improves on the fitted hazard rule with mean oracle gap 0.0121.

## Drift Audit
| Drift Curve | First zero crossing | Role |
| --- | ---: | --- |
| empirical utility drift | 2 | raw mean $\Delta U_t$ from realized utilities |
| conditional hazard drift | 2 | theorem-facing $((1-q_t)\alpha_t - q_t\beta_t - c)$ witness |
| fitted hazard drift | 2 | model-based estimate from learned probes |
| pooled proxy drift | 2 | legacy unconditional proxy kept for auditability only |

## Observables Evaluation
The strongest correctness proxy in the fitted models was token entropy (entropy_mean, coeff=-0.368). The strongest corruption-side signal was verbosity-confidence proxy (verbose_confidence_proxy, coeff=0.625). Those coefficients identify the dominant correctness and corruption observables for this run without assuming they transfer unchanged across model families.

## Stopping Comparison
| Policy | Mean stop step | Mean utility | Mean oracle gap |
| --- | ---: | ---: | ---: |
| oracle | 2.03 | 0.0381 | 0.0000 |
| hazard_drift | 2.80 | -0.0112 | 0.0493 |
| e_process | 2.00 | 0.0260 | 0.0121 |
| empirical_bernstein | 2.00 | 0.0260 | 0.0121 |
| never_stop | 10.00 | -0.3687 | 0.4067 |

## Graphs
### Drift Crossing Proof
![Drift crossing proof](drift_crossing_proof.png)

### Detector Gap Comparison
![Detector gap comparison](real_trace_detector_gaps.png)

### Feature Weight Summary
![Feature weight summary](real_trace_feature_weights.png)
