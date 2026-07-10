# Executive Summary — Scientific Rigor Audit & Failure Autopsy

**Date:** 2026-07-09 · **Repo state audited:** HEAD `e049cc7` + one uncommitted hunk in `research/real_trace_experiments.py` (characterized in [00_repo_state_and_staleness.md](00_repo_state_and_staleness.md) §C, left untouched for human review)
**Orchestration:** Module 0 (Sonnet 5) → Modules 1–2 (Fable 5, adversarial verification + full-population taxonomy) → Modules 3–4 (Fable 5, levers + pre-registered protocols) → this synthesis. Every number below traces to a module report that cites the exact file/line/command; nothing here is quoted from the pre-audit reports.

> **Update 2026-07-10 — P5 executed and verified.** With the user's approval, the defective `qwen2p5_7b__math` cell was re-analyzed in place (committed `trace_analysis.py`, same traces) and its reports regenerated. All four pre-registered predictions matched exactly (cell 1,374 W / 35 T / 91 L; global; MATH slice; capture). **The canonical headline is now 75,965 runs: 68,095 W / 2,135 T / 5,735 L = 7.5495% loss** (MATH slice 6.8721%; 12-late-cell capture 50.41%). §3 item 1 is closed; the 74,540/7.5745% figures below describe the pre-repair tree and remain correct for what they measured. Residual: the cell's `metadata.json` nested reconciliation block and the `_aggregate/math` cross-family artifacts still predate the repair (regenerating aggregates was not pre-registered in P5 — flagged, not done).

---

## 1. The verdict — direct answers

**Is the scientific-method claim genuinely verified now?**
**Partially — with named exceptions, none of which threatens the headline result.** Three axes are cleanly CONFIRMED (temperature, grading identity, stakes sweep [scope-limited]); three are CONFIRMED-WITH-CAVEAT (scale ladder, precision/quantization, trace-set identity); one is honestly NOT CONTROLLABLE (prompted-vs-distilled — it is a suggestive case comparison, not an isolated-factor experiment, and should be presented as such). The quantization axis deserves emphasis: **the isolation claim as previously written is not supported** (the bf16 and 4-bit runs differ in task count, code vintage, hardware, and batch size — not just the quantization flag), but the *conclusion* survives three independent robustness checks. Full table: [01_scientific_method_adversarial_verification.md](01_scientific_method_adversarial_verification.md) §Verdict table.

**Is the 7.57% loss figure real?**
**Yes — it reproduces exactly and is label-robust.** Fresh recomputation via an independent code path: 74,540 runs → win 66,784 (89.5948%) / tie 2,110 (2.8307%) / loss 5,646 (7.5745%), run-for-run identical to the July 1 figure ([02_failure_taxonomy.md](02_failure_taxonomy.md) §0–1). The feared label-integrity exposure (commit `074bc70`'s "~1.62% of ~799K rows would flip on regrade") is **>99% an artifact of bugs in the regrade tool itself**, which grades ARC/GPQA as free-form instead of multiple-choice. Under corrected semantics the genuine label drift across all 52 cells is **1 row in 798,770 (0.0001%)**, and **0 of 74,540 win/loss/tie verdicts move** under every known label correction ([01] §4/§Label-trust; [02] §Regrade sensitivity). ⚠️ Corollary: running `research/regrade_traces.py` for real, as-is, would *corrupt* ~12,284 correct labels — it must be patched first ([03] lever 5).

**What is actually causing the losses, and how much is fixable?**
The losses are 100% "missed late corrections" (every loss stopped on a wrong answer whose trace later became correct; the July 1 Part 6 "stopped too late" clause is mechanically impossible and should be deleted). Realistic recovery: **7.57% → ~5.7–6.5%**; roughly **39% of losses (~2,211 runs) are a genuine online-decision limit** that no recorded signal predicts (linear-probe evidence). On late-boundary cells the policy already sits at ≈ the ~50% online oracle-capture bound — anything promising loss ≪5% from these traces is a red flag, not a goal ([03] §Ceiling).

## 2. The true failure rate and its cause breakdown

Loss = the stopping policy's stop yielded strictly lower utility than never stopping. Decision quality, **not** model accuracy.

| Category (mutually exclusive, sums exactly) | Count | Share | Tag | Realistic recovery |
|---|---|---|---|---|
| B — stopped on an **empty extracted answer** | 196 | 3.5% | (i) pipeline bug (probe AUC 0.991) | ~120–175 (guard: never stop without a candidate answer) |
| C — sailed past an earlier correct answer | 459 | 8.1% | (iii) feature-fixable (AUC 0.824) | ~40–60% via best-so-far answer cache |
| D — correct only at forced step 1 (T_MIN floor) | 599 | 10.6% | (iii)-qualified — floor is a theory choice | ~25–50%, but only as answer-*selection*, never folded into the stopping headline |
| E — repair arrived exactly 1 step after stop | 1,652 | 29.3% | (iii) partial (AUC 0.669; 29% token-capped) | ~10–25%; blanket +1-step patience is provably net-negative |
| F — repair ≥2 steps after stop | 2,740 | 48.5% | (ii) fundamental at linear evidence (529 truncation-suspected) | truncation sub-slice only |
| A — grader artifacts | 0 | 0% | — | the loss set is not a grading artifact |
| **Total** | **5,646** | **7.5745%** | | **→ ~5.7–6.5% plausible floor** |

Key slice facts ([02] §2): GSM8K is the heaviest loss dataset (11.23%), ARC lightest (4.71%); **the published "losses concentrate in GSM8K/MATH" narrative is REFUTED as stated** — GPQA (7.37%) outranks MATH (6.92%); the old story fell for the win-rate-complement trap (MATH's complement is mostly ties). Losses rise monotonically with temperature (6.95% → 7.51% → 8.27%). Worst model: `mistral_small_24b_2409` at 16.43% (its GSM8K cell: 30.27%, token-cap implicated). 52.2% of all losses stop at the T_MIN=2 floor.

## 3. Defects found that were not previously flagged anywhere

1. **`qwen2p5_7b__math` is defective**: detector comparison covers 75/1,500 collected runs (first 25 shuffled tasks × 3 temps) — cross-model MATH comparisons are silently unbalanced. Out-of-place recompute of the full cell: 6.07% cell loss; global headline 7.5495% (spliced) vs 7.5745% (as-is) vs 7.5794% (excluded). In-place repair is a 5-min CPU job awaiting your approval (P5). ([00] §A; [02] §qwen7b_math; [04] P5)
2. **The published 50.9% oracle-gap capture** reproduces only under a hardcoded 12-cell selection (`_thesis_audit_latecells.py`, committed inside the outputs tree) that full-weights the defective 75-run cell. Honest variants: 48.84–50.41% on late cells; all-cells run-pooled figure is 63.66%. "About half the gap on late cells" survives; the bare number does not. ([02] §Oracle capture)
3. **GPQA count anomaly resolved both directions**: 4 cells exceed 1,344 runs at collection (CSV-corruption fragments) and fall below at analysis (sanitizer drops real runs); mechanism proven run-by-run; effect bounded and small. ([01] Axis 5)
4. **Undocumented split asymmetry**: GSM8K/GPQA use *train* splits; MATH/ARC use *test*. Needs a disclosure line in every affected table. ([01] §Split)
5. **Hardware misattribution**: the ladder runs were not on an "NVIDIA L4" as one report claims (61 GB allocation trace). ([01] S1 corrections)
6. **Roster mislabel**: `mistral_small_24b_2409` is a 22B model presented as 24B; also the failed `mistral_small_3p1_24b` → 2409 substitution was never documented in prose. ([03]/[04] P12)
7. The two July 1 Scientific Method reports: **CONFIRMED-WITH-NAMED-CORRECTIONS** (not superseded) — most tables verify against current data, but each contains specific overclaims/hand-patches enumerated in [01] §Supersession. Neither carries a staleness banner; both predate two relevant fixes.

## 4. Top next actions (ranked; each traceable to [03] levers / [04] protocols)

| # | Action | Cost | Why first |
|---|---|---|---|
| 1 | ~~**P5**: in-place repair of `qwen2p5_7b__math`~~ **DONE 2026-07-10** — prediction matched exactly (1,374W/35T/91L; global 7.5495%) | CPU, 5 min | Executed with user approval; see Update note above |
| 2 | Patch `regrade_traces.py` (MCQ answer-type, `keep_default_na=False`, fragment skip) + the `'N/A'`→`'A'` regex | hours | Gates all future label work; prevents a 12,284-label corruption |
| 3 | **P1**: offline census of the empty-answer guard (B) on existing traces | CPU ~1 h | Cheapest real utility gain (~0.2pp), AUC 0.991 |
| 4 | **P2/P3**: offline policy-variant census — T_MIN 2v3, token-cap guard, best-so-far cache, churn hysteresis, per-cell calibration | CPU 1–2 days | Tests every (iii) lever with zero GPU; pre-registered criteria fixed |
| 5 | Advisor-facing prose corrections (this document §3: concentration narrative, oracle-capture caption, "stopped too late" clause, split asymmetry, 22B, L4) | hours | Rigor debt with zero compute |
| 6 | **P9/P13**: λ-sensitivity sweep + temperature-trend confirmatory test | CPU hours | Long-standing backlog, closes two known gaps |
| 7 | **P6/P7**: raw-text re-extraction audit + sampled label adjudication (MATH-weighted) | CPU/API | Closes the two residual label-quality blockers |
| 8 | **P8**: clean bf16-vs-4-bit rerun (same code, same box, 500 tasks) | ~6–8 GPU-h | Converts Axis 2's caveat into a clean CONFIRMED |
| 9 | **P4b/P11**: token-cap 256→512 on the worst cell; HumanEval wiring + pilot | GPU | First new-data experiments; sequenced last deliberately |

Nothing before item 8 needs a GPU. Full pre-registration blocks (single IV, 7-axis constancy list, falsifiable hypothesis, n via the Part 7 SE/Z method, cost, in-advance success criteria): [04_next_experiment_protocols.md](04_next_experiment_protocols.md).

## 5. For the next advisor check-in

**Say with full confidence:** the 74,540-run headline (89.59/2.83/7.57) reproduces exactly from raw per-run data via two independent code paths and is invariant to every known label correction (drift: 1 row in ~799K); all three previously-flagged numeric contradictions are closed with named causes ([00] §B); every boundary value in current data respects the t≥2 floor (208 checked); the loss population is now exhaustively classified by a re-runnable committed script (`research/classify_losses.py`), not anecdotes; the policy captures ≈half the achievable oracle gap on late-boundary cells, which is at the online bound — and ~39% of remaining losses are provably unpredictable from recorded signals at linear-probe evidence.

**Proactively flag as corrected/open:** the six §3 items above; the defective-cell repair decision (P5); prompted-vs-distilled reframed as suggestive; the e-process union-bound reframing and λ-sensitivity sweep remain undone (now P9/P10); the uncommitted `reconcile_existing_outputs` hunk awaits your accept/revert decision ([00] §C — accepting it is reasonable *if* hidden-state caches are truly gone for good, but that is your call).

## 6. Audit provenance

Module reports: [00_repo_state_and_staleness.md](00_repo_state_and_staleness.md) · [01_scientific_method_adversarial_verification.md](01_scientific_method_adversarial_verification.md) · [02_failure_taxonomy.md](02_failure_taxonomy.md) · [03_improvement_levers.md](03_improvement_levers.md) · [04_next_experiment_protocols.md](04_next_experiment_protocols.md). Classification pipeline: `research/classify_losses.py` (deterministic; run 3×; commands + runtimes in [02] §T7). Module 1's first draft was interrupted by an API limit and subsequently re-verified line-by-line; one fabricated figure in that draft (a claimed 656-row flip set) was caught and corrected by the verification pass — the report's own Post-interruption section documents exactly what was re-executed. Scratchpad analysis scripts and intermediate CSVs are session-local; everything needed to reproduce any number in this summary is either committed or cited by command in the module reports.
