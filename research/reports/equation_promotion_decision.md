# Equation Promotion Decision

## Decision

- Deployed default: keep `quadratic_top4` unchanged for now.
- Hazard recommendation: promote `hazard_quadratic_combo_entropymean_entropystd_confidence_thoughttokencount` only as the documented next hazard candidate, not as live metadata.
- Empirical comparator: keep `direct_drift_ridge_top4` experimental only.

## Comparison

| variant | role | mean_test_auc | boundary_within_one | mean_oracle_gap | theorem_preserving | deployed_locally | decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `quadratic_top4` / `hazard_quadratic_top4` | current deployed hazard baseline | 0.6479 | 0.3986 | 0.3287 | yes | yes | keep as deployed baseline until promotion prerequisites are satisfied |
| `hazard_quadratic_combo_entropymean_entropystd_confidence_thoughttokencount` | best hazard-form boundary detector | 0.6151 | 0.5111 | 0.2512 | yes | no | document as recommended successor, but do not switch metadata yet |
| `direct_drift_ridge_top4` | best overall empirical stop rule | 0.5890 | 0.5358 | 0.3216 | no | no | keep as experimental comparator only |

## Rationale

The local repository currently has two different selection objectives:

1. `research/universal_feature_analysis.py` selects the deployed model from a small hazard-only candidate set by hazard-model generalization metrics.
2. `research/equation_analysis.py` ranks a much larger family of stop rules by boundary accuracy and oracle-gap behavior.

Those objectives do not currently agree. The CPU rerun of `research/universal_feature_analysis.py` still selected `quadratic_top4`, while the broader equation sweep found a better hazard-form boundary detector and an even stronger non-hazard empirical comparator.

Promoting the new hazard equation immediately would therefore be premature for three reasons:

- The deployed metadata writer and frontier validator still consume the smaller `universal_feature_analysis.py` model set.
- The recommended replacement has better stop-step behavior, but lower mean test AUC than the current deployed hazard baseline, so the promotion would change the optimization target.
- Full frontier validation is still missing, and parse-quality issues remain unresolved for some families.

Promoting `direct_drift_ridge_top4` would be an even larger change because it abandons the original q/alpha/beta hazard decomposition that the thesis currently treats as the interpretable algorithmic core.

## Promotion Gate

Only promote a new default after all of the following are done together:

1. Add the chosen variant to the live metadata/training pipeline.
2. Regenerate `universal_hazard_model_metadata.json` and any downstream reports from that new default.
3. Re-run frontier validation on real full frontier traces, not smoke traces.
4. State explicitly whether the thesis is still presenting Algorithm X as a hazard decomposition or is switching to a direct-drift rule.

## Immediate Outcome

- Mathematical recommendation changed at the analysis layer.
- Deployed local equation did not change.
- Deployed local algorithm did not change.