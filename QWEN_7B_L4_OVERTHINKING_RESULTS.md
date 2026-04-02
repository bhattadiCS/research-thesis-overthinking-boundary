# Qwen 7B L4 Overthinking Results

## Executive Summary
The L4 execution loop completed the environment check, parser repair, GSM8K scaling refactor, and real-trace collection for Qwen2.5 instruct 7B on 900 runs. The model entered a competent regime immediately, with step-1 accuracy $q_1=0.364$, and reached peak correctness $q_t=0.779$ at step 9. This run clears the current capability gate for a cross-family boundary claim.

## Mathematical Validation
The hazard decomposition exhibits repair rate 0.179 and corruption rate 0.168. The corrected conditional hazard drift crosses zero at step 6, while the raw empirical utility drift crosses at step 6, and the fitted hazard drift estimate crosses at step 7. The never-stop policy loses 0.4317 utility on average relative to the oracle, which is direct evidence that extra reasoning past the boundary is harmful. The new mixture e-process closes part of the gap to the hazard rule with mean oracle gap 0.3139. A previous report cited step 5 from a pooled proxy drift built from unconditional transition frequencies; that proxy is retained only as an audit trail and is no longer used as the boundary witness.

## Drift Audit
| Drift Curve | First zero crossing | Role |
| --- | ---: | --- |
| empirical utility drift | 6 | raw mean $\Delta U_t$ from realized utilities |
| conditional hazard drift | 6 | theorem-facing $((1-q_t)\alpha_t - q_t\beta_t - c)$ witness |
| fitted hazard drift | 7 | model-based estimate from learned probes |
| pooled proxy drift | 5 | legacy unconditional proxy kept for auditability only |

## Observables Evaluation
The strongest correctness proxy in the fitted models was self-reported confidence (confidence, coeff=0.714). The strongest corruption-side signal was verbosity-confidence proxy (verbose_confidence_proxy, coeff=0.622). Those coefficients identify the dominant correctness and corruption observables for this run without assuming they transfer unchanged across model families.

## Stopping Comparison
| Policy | Mean stop step | Mean utility | Mean oracle gap |
| --- | ---: | ---: | ---: |
| oracle | 2.48 | 0.7606 | 0.0000 |
| hazard_drift | 4.93 | 0.5413 | 0.2193 |
| e_process | 7.00 | 0.4467 | 0.3139 |
| empirical_bernstein | 9.00 | 0.3789 | 0.3817 |
| never_stop | 10.00 | 0.3289 | 0.4317 |

## Graphs
### Drift Crossing Proof
![Drift crossing proof](research/outputs/real_traces_l4_qwen_7b_4bit/drift_crossing_proof.png)

### Detector Gap Comparison
![Detector gap comparison](research/outputs/real_traces_l4_qwen_7b_4bit/real_trace_detector_gaps.png)

### Feature Weight Summary
![Feature weight summary](research/outputs/real_traces_l4_qwen_7b_4bit/real_trace_feature_weights.png)
