# Matched Peer-Dynamics Committee Ablation

Every pair uses identical task-held-out folds, LightGBM seed/configuration, corpus hashes, and preprocessing code. Only the peer-dynamics feature block differs.

| Contract | OOF ROC-AUC | Peer features |
| :--- | ---: | ---: |
| anonymous_minimal | 0.954664 | 110 |
| anonymous_minimal_baseline | 0.945336 | 0 |
| roster_no_timing | 0.956236 | 110 |
| roster_no_timing_baseline | 0.947305 | 0 |

| Matched treatment | Delta AUC | 95% paired task-bootstrap CI | P(Delta > 0) |
| :--- | ---: | :--- | ---: |
| anonymous_minimal_minus_anonymous_minimal_baseline | +0.009328 | [+0.008028, +0.010639] | 1.0000 |
| roster_no_timing_minus_roster_no_timing_baseline | +0.008931 | [+0.007670, +0.010169] | 1.0000 |

This is a retrospective closed-barrier analysis. It remains non-prospective until a fixed roster is synchronously collected with timestamped peer completion before scoring.
