# Qwen 0.5B L4 Overthinking Results

## Executive Summary
The L4 execution loop completed the environment check, parser repair, GSM8K scaling refactor, and real-trace collection for Qwen2.5 instruct 0.5B on 900 runs. The model entered a competent regime immediately, with step-1 accuracy $q_1=0.071$, and reached peak correctness $q_t=0.082$ at step 3. This run remains below the current capability gate, so it should be treated as a weak-regime control rather than a decisive family-level witness.

## Mathematical Validation
The hazard decomposition exhibits repair rate 0.003 and corruption rate 0.024. The corrected conditional hazard drift crosses zero at step 1, while the raw empirical utility drift crosses at step 1, and the fitted hazard drift estimate crosses at step 4. The never-stop policy loses 0.4595 utility on average relative to the oracle, which is direct evidence that extra reasoning past the boundary is harmful. The new mixture e-process improves on the fitted hazard rule with mean oracle gap 0.0595.

## Drift Audit
| Drift Curve | First zero crossing | Role |
| --- | ---: | --- |
| empirical utility drift | 1 | raw mean $\Delta U_t$ from realized utilities |
| conditional hazard drift | 1 | theorem-facing $((1-q_t)\alpha_t - q_t\beta_t - c)$ witness |
| fitted hazard drift | 4 | model-based estimate from learned probes |
| pooled proxy drift | 1 | legacy unconditional proxy kept for auditability only |

## Observables Evaluation
The strongest correctness proxy in the fitted models was verbosity-confidence proxy (verbose_confidence_proxy, coeff=0.448). The strongest corruption-side signal was token entropy (entropy_mean, coeff=0.847). Those coefficients identify the dominant correctness and corruption observables for this run without assuming they transfer unchanged across model families.

## Stopping Comparison
| Policy | Mean stop step | Mean utility | Mean oracle gap |
| --- | ---: | ---: | ---: |
| oracle | 1.03 | 0.0884 | 0.0000 |
| hazard_drift | 3.89 | -0.0647 | 0.1531 |
| e_process | 2.00 | 0.0289 | 0.0595 |
| empirical_bernstein | 9.00 | -0.3222 | 0.4106 |
| never_stop | 10.00 | -0.3711 | 0.4595 |

## Graphs
### Drift Crossing Proof
![Drift crossing proof](../research/outputs/real_traces_l4_qwen_0p5b/drift_crossing_proof.png)

### Detector Gap Comparison
![Detector gap comparison](../research/outputs/real_traces_l4_qwen_0p5b/real_trace_detector_gaps.png)

### Feature Weight Summary
![Feature weight summary](../research/outputs/real_traces_l4_qwen_0p5b/real_trace_feature_weights.png)
