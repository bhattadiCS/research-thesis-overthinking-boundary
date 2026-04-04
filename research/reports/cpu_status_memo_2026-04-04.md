# CPU Status Memo — 2026-04-04

## Executive Summary

The CPU continuation resolved the truth gap between reports, code, and metadata. The local workspace now tells a coherent story: legacy CPU analysis reruns are stable, the equation recommendation changed only at the analysis layer, the deployed Algorithm X baseline did not change, and full frontier validation is still pending because only smoke frontier traces exist locally.

## Did the math equation change?

Yes at the recommendation layer, no at the deployed layer.

- The equation sweep still recommends `hazard_quadratic_combo_entropymean_entropystd_confidence_thoughttokencount` as the best theorem-preserving hazard equation.
- The local deployed metadata still points to `quadratic_top4`.
- The CPU rerun of `research/universal_feature_analysis.py --random-state 7` still selected `quadratic_top4`.

Short answer: the recommended math changed, but the live local baseline did not.

## Did the algorithm change?

No.

- The deployed local algorithm remains the original hazard-style Algorithm X using the `quadratic_top4` intake.
- `direct_drift_ridge_top4` outperformed on boundary accuracy, but it remains an experimental comparator because it is not the original q/alpha/beta hazard decomposition.

Short answer: the deployed algorithm did not change.

## Did we parse more data?

Yes in the sense of auditing and classifying more local traces, but no in the sense of finding additional completed full frontier runs.

- We confirmed the Gemma 4 Edge 4B and Qwen 3.5 9B smoke frontier traces and diagnosed their parse behavior.
- We found additional auxiliary smoke/debug directories and recovered mirrors.
- We did not find any additional completed full `real_traces_colab_<MODEL_KEY>` frontier directories for the corrected frontier set.

Short answer: more local data were inspected and classified, but no new full frontier-complete dataset was added.

## How much did we progress toward the research goal?

Progress was meaningful on analysis clarity and low on frontier completion.

- High progress: the repository now has a defensible truth audit, a promotion decision, and a parse-success diagnosis.
- Moderate progress: smoke frontier evidence for Gemma 4 Edge 4B and Qwen 3.5 9B is now clearly framed as systems validation only.
- No frontier completion progress: the required full frontier trace directories are still missing, so universal frontier generalization remains unproven locally.

Short answer: we moved from ambiguous partial completion to coherent partial completion.

## Current Thesis-Grade Status

- Legacy zero-shot hazard baseline: rerun and confirmed.
- Equation comparison: rerun and confirmed.
- Best hazard candidate: documented, not promoted.
- Best empirical rule: documented, not deployed.
- Frontier smoke validation: confirmed for two runs only.
- Full frontier claim: still pending.

## Net Conclusion

Algorithm X is in a stronger analytical state than before this CPU continuation, but not in a stronger frontier-validation state. The main result of this session is epistemic cleanup: we now know exactly what changed, what did not, what data exist locally, and why the parse metrics behave the way they do.