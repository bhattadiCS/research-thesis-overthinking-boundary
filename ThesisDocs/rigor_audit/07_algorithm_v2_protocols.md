# Algorithm v2 — Pre-registered Protocols for Better Boundary Detection

**Date:** 2026-07-10 · **Status:** designed, not yet run · **Format:** as `04_next_experiment_protocols.md` (one IV; held-constant list; falsifiable hypothesis; n / power per `July_1_Checkin.md` Part 7; cost; success criterion fixed in advance).

## Why these and not more guards

The offline campaign ([05](05_offline_experiment_results.md)) established: every deterministic guard/selector built on the **existing per-run signals** fails at census (P1, P3a, P3b, P3c) — the per-run information set is exhausted, and on late-boundary cells the policy already sits at ≈ the online oracle-capture bound *for its current observables*. The two directions with evidence or theory behind them:

1. **Context-conditioning** — cell identity alone out-predicts all 16 per-run signals (AUC 0.760 vs 0.643, [02](02_failure_taxonomy.md) §8), and per-cell threshold calibration is the one lever that passed (+1,545.85 OOF, [06](06_calibrated_thresholds.md)). Open question: does it *generalize* (new cells, per-task granularity), or is it a lookup table?
2. **Belief-state quality** — μ_t = (1−q̂)α̂ − q̂β̂ − λ is only as good as q̂ and the hazards. Improving the *filtration* (better probes on existing features; genuinely new observables) is the only legitimate way to raise the online bound itself. The F-category "structural" verdict is explicitly linear-probe-scoped ([02] Blocker 3) — nonlinear and sequence models are untested.

**Standard metrics for every protocol below** (all out-of-fold): (i) net Δutility vs the frozen baseline policy (primary); (ii) oracle-gap capture; (iii) boundary-detection error — distribution of (stop_step − oracle_step), reported as median signed error + P(|err| ≤ 1); (iv) W/L/T with the tie caveat. Utility and win-rate always reported together.

---

## Tier 1 — CPU-only, frozen traces (run first; ~1 day each; extend the validated harnesses)

### N1 — Meta-calibration: predict the cell threshold from observable cell statistics
- **IV:** threshold source — per-cell fitted δ (P3d lookup) vs δ̂ = g(cell features) predicted by a small regressor on observable cell statistics (step-1/step-2 accuracy proxy from a small labeled probe set, mean answer-churn rate, mean entropy, mean trace length, dataset one-hot).
- **Held constant:** frozen traces; same recovery harness as `research/calibrate_cell_thresholds.py` (validated 0 stop mismatches); same λ/T_MIN/oracle.
- **Design:** leave-one-cell-out across 52 cells (fit g on 51 cells' fitted δ + features, predict the held-out cell's δ̂, score OOF utility there). Secondary: leave-one-**model**-out (all 4 of a model's cells held out) — the deployment-realistic variant.
- **Hypothesis:** LOCO meta-calibration recovers ≥ 60% of the P3d lookup gain (≥ +927 of +1,546). Falsified below that.
- **n / power:** census, 75,965 runs; paired enumeration (P1–P3 style) — no sampling.
- **Cost:** CPU, ~half a day (52 recovery passes are already cached patterns; the regressor is trivial).
- **Success:** ≥ +927 OOF ⇒ calibration is a *transferable mechanism* (thesis claim upgrades from "lookup helps" to "the boundary is predictable from cheap cell statistics"). **Failure is informative:** calibration is memorization, and new deployments need a labeled warm-up set — quantify how many labeled runs a warm-up needs (learning curve: δ fit on 50/100/200/400 runs vs full-cell δ).

### N2 — Probe upgrade suite (belief-state quality on existing features)
Three single-IV arms vs the production logistic probe, each swapped into the same fold structure (GroupKFold(5) by run_id, exactly as `trace_analysis.out_of_sample_eval`) and scored by census re-simulation of the μ-crossing policy:
- **N2a:** gradient-boosted trees (or small MLP) instead of logistic — same 10 features.
- **N2b:** isotonic-calibrated probe outputs (miscalibrated q̂ distorts μ directly; calibration fit on train folds only).
- **N2c:** trace-history window — features from steps t−2..t (deltas/rolling stats), same model class, so the probe sees short-run dynamics instead of a single step.
- **Held constant:** feature *inputs* limited to recorded columns; hazards refit the same way per arm; stop rule unchanged (μ ≤ 0, T_MIN=2).
- **Hypothesis (per arm):** OOF net Δutility > 0 and boundary-error P(|err|≤1) improves by ≥ 2 pp. Report per-arm with a 3-way multiple-comparison note.
- **Cost:** CPU, ~1 day total. **Success:** any arm clearing its bar becomes the production probe candidate; all-fail ⇒ the 10 recorded features are exhausted at any model class, sharpening the case for N7/N8 (new observables).

### N3 — Hazard estimation: partial pooling across cells
- **IV:** hazard estimator — current per-fold logistic α̂/β̂ vs empirical-Bayes/hierarchical per-step-index hazards (cell-level estimates shrunk toward cross-cell means; optional monotone constraint over t).
- **Motivation:** per-cell transition data is thin at large t (few runs survive that deep); shrinkage should stabilize exactly where the late boundary lives.
- **Hypothesis:** OOF net Δutility > 0 with the largest gains concentrated in the 7 negative-δ (late-boundary) cells. **Cost:** CPU, ~1 day. **Success:** Δutility > 0 AND late-cell oracle capture improves ≥ 2 pp.

### N4 — Online task-difficulty modulation (two-parameter threshold)
- **IV:** stop rule μ ≤ δ_cell (P3d) vs μ ≤ δ_cell + γ·s_early, where s_early is an online early-window signal (answer churn over steps 1–2, or q̂₂), γ fit jointly with δ on train folds (GroupKFold by task_id).
- **Hypothesis:** the two-parameter rule beats P3d's one-parameter rule OOF (Δutility > +150 over P3d). Falsified otherwise — meaning within-cell difficulty variation is not actionable from early steps.
- **Cost:** CPU, ~half a day (extends `calibrate_cell_thresholds.py`).

## Tier 2 — GPU, already pre-registered (run as one box-day; blocks unchanged from [04])

### N5 (= P4b) — Token-budget causal test: `max_new_tokens` 256→512, `mistral_small_24b_2409__gsm8k`
~6 GPU-h; success = cell loss rate drops ≥ 5 pp from 30.27%. Decides whether truncation *causes* the E/F churn or merely correlates. If causal, a budget increase is an algorithm-external fix that shrinks the loss pool before any detector change.

### N6 (= P8) — Clean bf16 vs 4-bit pair, Qwen2.5-7B, same code/box/tasks
~6–8 GPU-h; success = step-2 correctness gap ≥ 0.10 at Z ≥ 1.96. Upgrades Axis 2 to a clean CONFIRMED (rigor, not performance).

## Tier 3 — GPU + engineering: enrich the filtration (the real ceiling-raisers)

### N7 — Self-consistency belief pilot (k=2 agreement as an observable)
- **IV:** presence of a per-step agreement feature — at each step, a second independent short continuation is sampled and agreement of extracted answers is recorded. Policy arms then test q̂ (and μ) with vs without the agreement feature, offline, on the pilot traces.
- **Held constant (7-axis):** 2 cells (`qwen2p5_7b__gsm8k` + `qwen2p5_7b__math` — one late, one weakened boundary), same 500 tasks/seeds/temps/prompts/grader as the recorded cells; collection code extended only to sample k=2 and record agreement.
- **Honest cost accounting (mandatory):** the extra sample's tokens are charged inside utility — λ applies to total generated tokens, so the agreement feature must pay for itself; pre-register utility at both λ-accounting variants (charged vs uncharged) but the CHARGED variant is the success metric.
- **Hypothesis:** the agreement-augmented probe improves OOF boundary detection on the pilot cells: P(|stop−oracle| ≤ 1) +5 pp AND charged net Δutility > 0. Prior support: agreement/self-consistency is the classic strong correctness signal; nothing in the current 10 features measures cross-sample stability.
- **n / power:** 2 cells × 1,500 runs = 3,000 runs (each with doubled generation) — powered for the Part 7 bar at Δ ≥ 1.6 pp on binary metrics.
- **Cost:** ~1–2 days engineering in `real_trace_experiments.py` (new collection mode; keep it flag-gated) + **~12 GPU-h**.
- **Success:** both bars met ⇒ plan the 13-model agreement sweep; failure ⇒ multi-sample belief is not worth its token cost at k=2 — document and stop this line.

### N8 — Extended signal recorder pilot (answer-span logprobs + pooled mid-layer probe features)
- **IV:** recorder version — current signals vs extended: (a) mean/min token logprob restricted to the extracted **answer span** (current `mean_token_logprob` averages the whole step); (b) pooled mid-layer hidden-state features (mean-pooled activations at 2–3 fixed layers, projected to ≤ 64 dims at record time — KB per step, avoiding the old un-storable `.npz` problem); (c) answer-span entropy.
- **Held constant:** 2 pilot cells as N7; generation itself unchanged (recorder is read-only w.r.t. sampling), so traces are comparable to the recorded cells modulo seed-identical regeneration.
- **Hypothesis:** at least one extended feature raises out-of-fold probe AUC by ≥ 0.03 over the 10-feature baseline on pilot cells AND census re-simulation with the enriched probe gives net Δutility > 0.
- **Cost:** ~1–2 days engineering + **~6 GPU-h**. **Success:** promote the winning features into `FEATURE_COLUMNS` for the next full sweep; failure with tight CIs ⇒ strong evidence the useful signal is not in cheap observables, elevating N7's multi-sample line.

### N9 (= P11) — Coding-domain pilot (HumanEval)
Parser/executor wiring in `real_trace_experiments.py` (+ sandboxed execution grader), then a 1-cell pilot (~1–2 GPU-h). Success = a measurable boundary + win rate > 70%. Generalization evidence, orthogonal to detector quality.

## Sequencing and decision points

1. **Week 1 (CPU):** N1 → N4 → N2 → N3 (N1 first: if meta-calibration transfers, N4's parameterization changes; N2/N3 inform what N7/N8 must record).
2. **One GPU day:** N5 + N6 (independent, pre-registered, no engineering).
3. **After N2 results:** build N7/N8 recorders (1–2 days eng) → **second GPU day:** N7 + N8 pilots.
4. **Parallel track:** N9 wiring whenever engineering time allows.
5. **Full-sweep decision:** only if N7 or N8 passes does a 13-model re-collection sweep get designed (it would be the first re-collection since June; cost ≈ the original 52-cell sweep × arms).

Expectation-setting (from [03] §3 and [05]): late-cell capture is already at ≈ the online bound for the current observables; Tier-1 gains come from context, Tier-3 gains from *changing what the policy can see*. Anything promising loss ≪ 5% on the existing traces without new observables remains a red flag.
