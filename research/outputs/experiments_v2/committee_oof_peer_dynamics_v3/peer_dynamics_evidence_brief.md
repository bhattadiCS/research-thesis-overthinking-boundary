# Peer-Dynamics Evidence Decision Brief

**Historical candidate only:** `anonymous_minimal` - 0.954664 OOF ROC-AUC, above the 0.95 gate without learner-visible model identity or timing metadata.

| Evidence set | Treatment AUC | Baseline AUC | Delta AUC (95% task-bootstrap CI) | Decision |
| :--- | ---: | ---: | :--- | :--- |
| Full corpus, anonymous 110-feature profile | 0.954664 | 0.945336 | +0.009328 [+0.008028, +0.010639] | Selected historical candidate |
| Fixed 13-member panels | 0.940009 | 0.931266 | +0.008743 [+0.006653, +0.010882] | Gain survives; absolute gate not met |
| Compact topology-only profile | 0.948138 | 0.945336 | +0.002802 [+0.002046, +0.003537] | Rejected for < 0.95 AUC |

The roster-aware full profile reaches 0.956236, but it is corroborating evidence rather than the selected candidate because the selected candidate avoids model-identity learner inputs.

This is not a prospective stopping claim. The stopping threshold is intentionally unassigned until it is chosen on a preregistered development split and tested once on a fresh, synchronized fixed-roster collection.

## Fresh confirmation gate

- Pre-register the anonymous full peer-telemetry feature contract, model-training split, calibration method, and stopping threshold before accessing fresh evaluation labels.
- Use a frozen roster with pinned model/tokenizer and dataset revisions on a fresh task set disjoint from the historical corpus.
- Close and timestamp every full peer barrier before calculating a candidate score; record immutable event, barrier, decision, and token ledgers.
- Run the hardware preflight on the actual target GPU and retain its device, VRAM, temperature, power, and software-version provenance.
- Report the fresh task-held-out ROC-AUC, calibration, token utility, and audit status separately from this retrospective evidence.

## Legacy manifest disclosure

Uniformly absent legacy fields prove no cross-arm mismatch, but do not retroactively prove their values for already-completed v3 runs.
