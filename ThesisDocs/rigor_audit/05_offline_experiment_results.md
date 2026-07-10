# Offline Experiment Results — Pre-registered Protocols P1/P2/P3a/P13 (+P6/P9)

**Date:** 2026-07-10 · **Corpus:** post-repair canonical matrix (75,965 runs; 68,095 W / 2,135 T / 5,735 L = 7.5495% loss) · **Harness:** `research/offline_policy_experiments.py` (census, per-cell cached, deterministic)
**Counterfactual semantics** (as pre-registered in [04_next_experiment_protocols.md](04_next_experiment_protocols.md)): stop steps come from the frozen `hazard_drift` decisions in `detector_comparison_by_run.csv`; a guard veto defers the stop to the first later step where the predicate clears (drift-persistence assumption), else the run's last step; `utility = correct − 0.05·(step−1)`; `never_stop` is never modified. **Validation:** recomputed stop-step utilities matched the pipeline's `stop_utility` for **75,965/75,965 runs (0 mismatches)**.

## P1 — Empty-answer guard: **FAILED (guard not shipped)** ❌

| Pre-registered criterion | Result |
|---|---|
| net ΔU > 0 | **−116.55** ❌ |
| ≥100 B-losses dissolved | 140 ✅ |
| ≤50 win→loss conversions | **118** ❌ |

Affected runs: 196 losses, **1,056 wins**, 14 ties. The decisive fact the census surfaced: five times more *wins* than losses stop on an empty extracted answer — on hard datasets the model's final answer is wrong anyway, so an early empty-answer stop (utility −0.05·(s−1)) beats paying the horizon cost, and deferring lands on wrong answers later. Per-dataset ΔU: ARC +12.0 (only positive), GSM8K −0.25, MATH −1.45, GPQA −126.85. **Interpretation:** Module 2's AUC 0.991 separated B-losses from other *losses*, not from empty-stopping *wins* at decision time. A blanket guard is net-negative; only an ARC-scoped variant is even neutral. Publishable negative, per the protocol's own clause.

## P2 — T_MIN=2 vs T_MIN=3: **FLOOR DEFENDED** ✅

Moving the floor to 3 costs **net ΔU = −1,912.35**: it dissolves 1,169 losses but converts 880 wins to losses and charges +0.05 delay to the (previously unquantified) **36,527 step-2-stopping runs, 33,574 of which are wins**. The T_MIN=2 structural choice now has a quantified defense: the modal loss (quit at the floor, repair later) is the tail of a bet that is overwhelmingly correct — step-2 stops are 91.9% wins.

## P3a — Blanket token-cap guard: **FAILED (guard not shipped)** ❌

| Pre-registered criterion | Result |
|---|---|
| net ΔU > 0 | **−8,033.65** ❌ |
| ≥200 of the capped losses dissolve | 1,730 ✅ |

Affected: 1,771 losses, **20,410 wins**, 492 ties — capped-step stopping is endemic and usually *right* (2,023 win→loss under the guard). Truncation is heavily involved in losses (Module 2's E/F sub-slices) but "never stop on a capped step" is far too blunt; 27% of the whole corpus stops on capped steps. Any viable truncation lever must be conditional (e.g., capped AND churning AND low q̂) — that is P3b/P3c territory, not a deterministic guard. The *causal* question (would longer budgets fix the underlying instability?) remains P4b (GPU).

## P13 — Temperature→loss monotonicity: **CONFIRMED 4/4** ✅

Cochran–Armitage trend on loss vs temperature {0.1, 0.6, 1.0}, per dataset: ARC Z=+2.77 (4.35/4.38/5.38%), GPQA Z=+4.43 (6.15/7.68/8.30%), GSM8K Z=+2.33 (10.66/11.06/11.95%), MATH Z=+2.08 (6.45/6.80/7.37%). Criterion (Z>1.96 increasing in ≥3/4) met in **4/4** — with the pre-registered honesty clause that the direction was observed before this confirmatory pass, so this is a within-corpus consistency check, not fresh-data confirmation.

## P6 — Raw-text re-extraction audit: **PASS on rate / FAIL on zero-flips, scoped** ⚠️

Census of all loss stop/final rows + 1.1% seeded sample of remaining rows. **Label delta rate 0.188% (bar: ≤0.5% — PASSED).** Verdict-relevant deltas: **30 (bar: 0 — FAILED)** — but every flagged row sits in a **DeepSeek** cell, and the mechanism was identified by direct inspection (`deepseek_r1_distill_1p5b__math …00243`, step 14): reasoning-mode outputs carry the answer inside the `<think>` block followed by an *incomplete* JSON block; collection-time extraction credited the think-block answer via its fallback path, which a from-scratch re-parse of `raw_text` alone does not reproduce. Storage truncation was ruled out (stored text ≥ recorded length). **Scoped verdict:** extraction is NOT a live confound for the 11 non-reasoning models (0 verdict-relevant deltas outside DeepSeek); for the 2 DeepSeek models, a follow-up that replicates the collection-time fallback/carry-forward semantics is required before treating any of the 30 rows as label errors. Worst-case bound if all 30 were genuine: ≤0.04 pp of verdicts, mixed direction.

## P9 — λ-sensitivity sweep: **PASSED** ✅

Method note: the original plan (full out-of-place `trace_analysis.py` re-runs per (cell, λ)) was abandoned at 19/208 cells for runtime (~14 min/cell); replaced by the **exact analytic form** — since the fitted models are λ-independent, `mu_λ ≤ 0 ⇔ (1−q̂)α̂ − q̂β̂ ≤ λ`, so recovering each run's validated out-of-fold base series once yields the identical sweep (λ in both the rule and the utility) in minutes. **Validation anchor: λ=0.05 reproduces 68,095/2,135/5,735 with 0 stop mismatches against the recorded stops across all 75,965 runs.**

| λ | win % | loss % |
|---|---|---|
| 0.01 | 87.65 | 6.60 |
| 0.02 | 88.84 | 6.85 |
| **0.05 (published)** | **89.64** | **7.55** |
| 0.10 | 90.86 | 7.06 |
| 0.20 | 98.78 | 0.27 |

**Criterion (win within ±5pp of 89.6% for λ ∈ [0.02, 0.10]): PASSED** — max deviation 1.26pp. The headline is not knife-edged on the λ=0.05 convention. Caveat: λ=0.20 is a degenerate regime (the rule stops almost immediately everywhere while `never_stop` pays a heavy horizon cost — win rate is trivially high there and should not be quoted as policy quality).

## P3b/P3c/P3d — conditional policy arms (run 2026-07-10, same census harness family: `p3bcd.py`)

**Replication gate:** the harness rebuilt the production scoring exactly (same sanitizer, temporal features, GroupKFold(5)-by-run_id fold structure, per-fold model fits) and recovered each run's out-of-fold per-step drift series; the recomputed first-crossing stop equaled the recorded `hazard_drift` stop for **75,965/75,965 runs (0 mismatches, 0 invalid cells)** — the arm deltas below are measured against a perfectly reproduced baseline.

| Arm (single IV) | Pre-registered criteria | Result | Verdict |
|---|---|---|---|
| **P3b** churn-gated hysteresis | net ΔU>0 AND ≥150 E-losses dissolve | ΔU **−1,003.75**; 1,081 losses dissolved (683 E ✅); 191 win→loss; 15,922 stops moved | **FAILED** ❌ |
| **P3c** best-so-far selection (eligible ≥2) | ≥180 C-losses dissolve AND ≤90 win→loss | **16** C-dissolved; **132** win→loss; ΔU −85.0 | **FAILED** ❌ |
| P3c2 sub-arm (incl. step 1; answer-selection framing only) | — | ΔU −171.0; 375 dissolved; 1,006 win→loss | negative |
| **P3d** per-cell threshold calibration (OOF by task_id) | OOF net ΔU > P3a/P3b/P3c | **ΔU +1,545.85** — 631 losses dissolved, 982 win→loss, 20,678 stops re-timed; fitted δ mean +0.080 (range −0.15…+0.15) | **PASSED** ✅ |

**Interpretation.** P3c is the campaign's most instructive failure: the q̂ ranking that separates C-losses *diagnostically* (AUC 0.824) cannot *select* the earlier-correct candidate reliably at stop time — best-so-far selection is dead in both framings. P3b's hysteresis dissolves plenty of photo-finish losses but pays more in delayed wins. **P3d is the one lever that survives its pre-registration**: one scalar drift-threshold offset per cell, fit out-of-fold, is worth **+1,546 utility** — more than 40% of the entire 3,543-utility loss mass — confirming Module 2's core diagnostic that *cell identity* (slice AUC 0.760) carries more actionable signal than any per-run feature. The fitted offsets are mostly positive (stop *sooner*), i.e., the uncalibrated policy systematically over-thinks on most cells relative to each cell's own cost structure.

**Framing caveat (must accompany any advisor-facing use):** P3d optimizes *utility*, not win rate — it trades 982 new small losses for large step-cost savings plus 631 dissolved losses, so the **loss count rises** (5,735 → ~6,086; loss rate ≈8.0%) while **net utility improves by +1,546**. Under the thesis's own objective (expected utility) it is a strict improvement; under the "% of runs beaten" headline it is not. Report both numbers together, always.

## What this changes in the improvement picture

The two deterministic guards ranked #2 and #4 in [03_improvement_levers.md](03_improvement_levers.md) are **empirically dead as blanket policies** — the census showed their trigger conditions are dominated by wins. The floor (52.2% of losses) is **defended with numbers**. What survives after the full P3 campaign: **only P3d (per-cell threshold calibration, +1,546 OOF utility)** among the policy levers, plus the token-budget causal experiment P4b (GPU). P3b and P3c joined P1/P3a as pre-registered negatives. Realistic recovery is now *measured*, not estimated: ~+1,546 utility (~44% of the loss mass) via calibration — taken mostly as step-cost savings rather than loss-count reduction — and essentially nothing else from recorded per-run signals. The structural-core diagnosis strengthens: per-run observables support no deployable guard or selector beyond calibration; Module 3's ~800–1,400-loss recovery estimate is superseded by these census results.
