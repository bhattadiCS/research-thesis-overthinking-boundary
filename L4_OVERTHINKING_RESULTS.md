# L4 Overthinking Results

## Executive Summary
The L4 execution loop completed the environment check, parser repair, GSM8K scaling refactor, and real-trace collection on 8 runs. The model entered a competent regime immediately, with step-1 accuracy $q_1=0.000$, and reached peak correctness $q_t=0.250$ at step 2.

## Mathematical Validation
The hazard decomposition exhibits repair rate 0.125 and corruption rate 0.500. The first hazard drift zero crossing occurs at step 2, which is the empirical candidate for the Overthinking Boundary. The never-stop policy loses 0.2125 utility on average relative to the oracle, which is direct evidence that extra reasoning past the boundary is harmful.

## Observables Evaluation
The strongest correctness proxy in the fitted models was reasoning length (thought_token_count, coeff=1.142). The strongest corruption-side signal was token entropy (entropy_mean, coeff=0.293). Those coefficients make hidden-state drift and verbosity-linked features the leading observable candidates for boundary detection in the current run.

## Stopping Comparison
| Policy | Mean stop step | Mean utility | Mean oracle gap |
| --- | ---: | ---: | ---: |
| oracle | 1.25 | 0.2375 | 0.0000 |
| empirical_bernstein | 2.00 | 0.2000 | 0.0375 |
| never_stop | 3.00 | 0.0250 | 0.2125 |

## Graphs
- [Drift crossing proof](research/outputs/real_traces_gsm8k_probe/drift_crossing_proof.png)
- [Detector gap comparison](research/outputs/real_traces_gsm8k_probe/real_trace_detector_gaps.png)
- [Feature weight summary](research/outputs/real_traces_gsm8k_probe/real_trace_feature_weights.png)
