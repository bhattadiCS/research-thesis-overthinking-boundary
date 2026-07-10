# Module 3: Root-Cause → Improvement Levers

**Auditor:** Fable 5 (Modules 3+4 merged pass) · **Date:** 2026-07-09 · **HEAD at audit:** `e049cc7`
**Inputs (binding, not re-derived):** `00_repo_state_and_staleness.md` (Module 0), `01_scientific_method_adversarial_verification.md` (Module 1, FINALIZED), `02_failure_taxonomy.md` (Module 2, incl. its §10 HANDOFF). Spec: `ThesisDocs/Scientific_Rigor_Research_Prompt.md` §Module 3.
**Method note:** this is a synthesis/design module. Every number below is cited to its source report; no upstream number was recomputed. Category counts are **upper bounds** on recovery — removing a failure mode does not guarantee the policy then makes the right call. Where a realistic fraction is estimated, the estimate is labeled and the offline experiment that measures it exactly is named (→ `04_next_experiment_protocols.md`, "P#").

**Base facts (Module 2 §§0-1,7):** 74,540 runs; loss 5,646 (7.5745%); losses cost 3,543.0 utility vs 27,815.1 gained by wins (7.85:1); mean loss margin 0.6275 utility (range 0.40–0.95); 52.2% of losses (2,946) stop at the T_MIN=2 floor; 72,430 runs stop before the horizon (all wins + all losses have `hd_step < ns_step`; only the 2,110 ties run out).

---

## 1. Per-category levers

| Cat | Count (UB) | Tag | Probe AUC (M2 §8) | Lever(s) | Realistic recovery (est.) | Measured by |
|---|---:|---|---|---|---|---|
| B stopped_on_empty_answer | 196 | (i) | 0.991 | Guard: never stop while `answer_normalized` is empty | **~60–90% (~120–175)** — near-full-recovery case | P1 (CPU) |
| C passed_earlier_correct | 459 | (iii) | 0.824 | Best-so-far answer cache; answer-stability features | ~40–60% (~185–275) | P3 (CPU) |
| D step1_only_correct | 599 | (iii)-qual. | 0.747 | Step-1 revert target; T_MIN structural question | ~25–50% (~150–300) | P2, P3 (CPU) |
| E next_step_repair | 1,652 | (iii) | 0.669 | Gated hysteresis; token-cap guard (479 free subset) | ~10–25% (~165–415) | P3, P4 |
| F late_repair | 2,740 | (ii) linear | 0.681 | Truncation guard on the 529 sub-slice; cross-cell calibration; rest structural | ~5–15% (~140–400), almost all from the 529 + calibration | P3, P4; §3 |
| Cross-cutting | all | — | slice-identity alone 0.760 | Per-cell threshold/λ calibration | unbounded by category counts; largest *model* lever | P3 |

### 1.B — pipeline guard: never stop on an empty extracted answer (196, tag i)

Mechanism (M2 §7.B): the policy stopped on a step where extraction produced no candidate (160 `json_field` / 36 `fallback` at the stop step); GPQA 128 / ARC 58; concentrated in `llama_3p1_8b_instruct` (59) and `mistral_small_24b_2409` (54).

**Why near-full recovery is the right prior.** An empty-answer stop at step *s* scores exactly `−0.05(s−1) ≤ 0` (utility = correct − 0.05·(step−1), correct=0 guaranteed). Every loss is a missed late correction (final-step answer correct, 5,646/5,646 — M2 §1), so for all 196 B-losses even the worst-case fallback of running to the horizon beats the empty stop (`ns` utility ≥ 0.35 at max horizon 14). The guard is deterministic, one line, and B-runs are near-perfectly identifiable at stop time (AUC 0.991 [0.982, 1.000]).

**Why not 196/196:** (a) after the guard fires, the policy stops on the *next non-empty* candidate, which may be wrong-and-pre-repair — the run then re-enters C/E/F; (b) win-side cost: stopped-wrong **wins** that stopped on an empty answer get delayed at 0.05/step — count currently unquantified [UNVERIFIED — P1 computes it as part of the census]. Both are exactly measurable offline on frozen traces (P1). Max headline movement: 196/74,540 = **−0.26 pp** loss rate; realistic ~−0.16 to −0.23 pp.

### 1.C — best-so-far answer cache + stability features (459, tag iii)

Mechanism (M2 §7.C): a decision-eligible correct answer existed strictly before the stop; the policy sailed past it and stopped on a wrong answer. GSM8K 286; `mistral_small_24b_2409` 128. The winning state was *observed and abandoned* — this is the strongest per-category signal in the data (AUC 0.824 [0.800, 0.847]).

Levers, grounded in the mechanism:
1. **Best-so-far cache (protocol change):** at stop, return the candidate with the highest predicted-correctness q̂ seen at any eligible step, not the current one. Converts a C-loss iff the selector ranks the earlier-correct candidate on top; AUC 0.824 supports ~half-to-most, hence the 40–60% estimate. Risk: the cache also acts on **wins** that stopped on a correct current answer — a mis-ranked earlier wrong answer converts a win to a loss. Net effect must be measured (P3), not assumed.
2. **Stability features in the hazard model:** `answer_streak` is already recorded (M2 §8 feature list); the deployed policy did not use per-run answer-stability information in its stopping decision. Feeds P3's feature-augmented arm.

Honest caveat: this lever changes *answer selection*, not just *when to stop* — advisor-facing framing must present it as a policy-class extension (stop-and-select), not a tweak to the same policy.

### 1.D — the T_MIN=2 floor tension (599, tag iii-qualified) — structural choice discussion

The tension, stated plainly: **the floor exists for theory reasons** — step 1 is the forced-commit init, not a decision point (`trace_analysis.py:20`; the floored oracle cannot take step 1 either — Module 0 §B3, Module 1 Axis 6), and the 06-20 remediation introduced it precisely because unfloored step-1 boundaries were theory-forbidden artifacts. Yet **52.2% of all losses stop at that floor** (M2 §1) and category D (correct at step 1 only, corrupted by step 2, never eligible-correct again before the stop) is *created* by the floor: the ideal stop is protocol-unreachable.

Resolution: the floor should **stay** (removing it re-opens the exact defect the remediation closed), but two offline-testable variants bound what it costs:
1. **T_MIN=3 variant (P2):** if the modal loss is "quit at the first admissible step; repair arrived later," raising the floor by one step directly tests whether step-2 stops are systematically premature. Cost side is mechanical: every stopped run pays +0.05 per delayed step; the win/loss ledger is exactly computable on frozen traces. This is a *diagnostic*, not a proposal to change the theory — pre-registered either way.
2. **Step-1 revert target (P3 arm):** keep the step-1 candidate; when the eligible-step answer churns at the stop (66% of D had `answer_changed=1` at stop — M2 §7.D), return the step-1 candidate instead. Upper bound 599 and every D-run's step-1 answer is correct *by category definition*, so precision of the trigger (AUC 0.747) is the only loss term on the D side; the risk term is reverting to a wrong step-1 answer on non-D runs. Same stop-and-select framing caveat as C.

### 1.E — gated patience + token-cap guard (1,652, tag iii)

Mechanism (M2 §7.E): first eligible correct answer at stop+1 — one step of patience from a win on 29.3% of all losses. At the stop: 41% still churning (`answer_changed=1`), **29% (479) token-capped** (`hit_max_new_tokens=1`, a recorded live flag).

**Blanket patience is roughly net-negative — do not propose it.** Arithmetic on M2 §1 counts: +1 mandatory step on all 72,430 stopped runs costs 0.05 × 72,430 ≈ **3,622 utility — more than the entire 3,543-utility loss mass**, against an E-side upper-bound gain of 1,652 × 0.6275 ≈ 1,037. Any patience must be **gated**:
1. **Token-cap guard:** never stop on a step whose generation hit `max_new_tokens` (deterministic flag; 479 E-losses + 529 F-losses qualify). The stopped-on answer on a capped step is an artifact of truncation, not a settled candidate. Win-side cost bounded by the (unquantified [UNVERIFIED — P1/P3 census]) number of wins that stopped on capped steps.
2. **Churn-gated hysteresis:** require the drift signal negative on 2 consecutive steps *only when* `answer_changed=1` at the candidate stop. Targets the 680 churning E-stops at a cost proportional to churning wins only.
Realistic recovery ~10–25% of E; exact number is P3's job. The truncation *cause* (would un-truncated generations at 512 tokens change stop behavior?) is a GPU question — P4.

### 1.F — late_repair: the honest online-decision ceiling (2,740, tag ii at linear evidence)

The policy's exact bet is that a currently-wrong trace won't repair, and it wins that bet 86.5% of the time (M2 §7.F). F is where the bet fails ≥2 steps out. Stop-step separability is the weakest of all categories (AUC 0.681 [0.670, 0.692]); no single recorded feature separates F-losses from stopped-wrong wins (all |d| ≲ 0.2). Recoverable edge: the 529 truncation-suspected stops (19% — arguably tag iii; covered by the token-cap guard above) and cross-cell calibration (below). **Treat the remaining ~2,200 as structural.** Full ceiling quantification in §3.

### 1.X — cross-cell risk calibration (all categories; the highest-leverage *model* change)

Slice identity alone predicts losses at AUC 0.760 — better than all 16 recorded per-run signals combined (0.643) (M2 §8). Per-cell loss rates span 0.13%–30.27%; the worst single cell, `mistral_small_24b_2409__gsm8k`, is 30.27% (M2 §5) — one cell carrying an outsized share of the loss mass while the policy applies per-cell-trained hazards with **no cross-cell risk calibration**. Lever: per-cell stopping-threshold (or λ/drift-threshold) calibration on held-out data. Not bounded by any single category count; the probe evidence says this is where the most predictive information sits. Measured by P3 (calibration arm). Risk: per-cell tuning must be GroupKFold-held-out or it manufactures leakage — the exact sin the June audit removed.

---

## 2. Pipeline / hygiene levers (Modules 0/1/2 blockers)

Each with cost, risk, and **the number it changes**. None of these is optional if the thesis is to cite the audited figures.

| # | Lever | Source | Cost | Risk | Number it changes |
|---|---|---|---|---|---|
| H1 | Patch `regrade_traces.py`: `_answer_type` mcq for arc/gpqa; `keep_default_na=False`; skip corruption-fragment rows | M1 §4b, HANDOFF 2 | ~1 h + tests | Low (tool is dry-run-only so far) | None today; **prevents corrupting ~12,284 correct ARC/GPQA labels** if ever run for real |
| H2 | `normalize_mcq_answer` regex: reject `N/A` → `'A'` match | M1 §4a, HANDOFF 5 | ~15 min + test | Low | 10 falsely-credited rows → 0 verdict flips (M2 §3); absolute-quality fix only |
| H3 | **Re-run `trace_analysis.py` in place for `qwen2p5_7b__math`** + regenerate its stale `summary.md`/`final_results.md` | M0 Blocker 1, M1 §5c, M2 §4 (human-approved production step) | ~5 min CPU + commit | Low (expected values pre-known from the OOP recompute) | Headline 7.5745% → **7.5495% on 75,965 runs**; MATH 6.92→6.87%; Part 5 capture 50.9→**50.41%** |
| H4 | CSV control-character escaping at write time (corruption root cause) + `trace_counts()` well-formed-run_id filter | M1 Axis 5a, HANDOFF 4 | ~2 h + test | Low; prospective only | Future cells stop corrupting; Part 3 collected total 75,996 → 75,972 (drops 24 phantom run_ids) |
| H5 | Harmonize `run_stakes_sweep.py` λ·step vs pipeline λ·(step−1) | M1 Axis 6, HANDOFF 7 | ~30 min | Low; boundaries unaffected | Stakes-report utility *levels* become cross-readable with matrix `stop_utility`; T* values unchanged |
| H6 | Relocate `research/outputs/experiment_matrix/_thesis_audit_latecells.py` → `research/` | M2 Blocker 4 | ~10 min | None | No number; removes analysis code from a write-protected outputs tree |
| H7 | SUPERSEDED banners / README on `research/outputs/cross_family/` raw CSVs (stale third tree) | M0 Task F.3 | ~15 min | None | Prevents re-citation of unfloored pre-remediation boundaries (3 floor violations live in that CSV) |
| H8 | Document the train/test split asymmetry (GSM8K/GPQA train; MATH/ARC test; which were forced vs chosen) | M1 headline asterisk item 5 | ~30 min prose | None | No number; blocks a real external-validity objection at the defense |
| H9 | Fix `Startup_Research_Prompt.md:72` "prefer step 1" | M0 §B3 | 1 line | None | Prevents a future agent acting on inverted guidance |
| H10 | Correct `July_1_Checkin.md` Part 6's impossible "stopped too late" clause; re-present Part 5 oracle capture with the 12-cell selection disclosed + honest variants (48.84–50.41% late-cell, 63.66% all-cells) + slice/aggregation caption | M2 §1 (mechanical impossibility proof), §6 | ~1 h prose | None | Advisor-facing claims: "~half the oracle gap on late cells; ~64% matrix-wide" replaces the un-captioned 50.9% |
| H11 | Math-grader sympy guard (`0x^2` → `TokenError`); pin/record sympy version in metadata | M1 HANDOFF 9 | ~1 h + test | Low | The 1 genuine flip row grades correctly; collection-box vs analysis-box grading becomes provably identical |
| H12 | Fix `run_experiment_matrix.analysis_complete()` file-existence check (root cause of the stale 75-run cell) | M1 §5c, M2 §10.7 | ~1 h | Low | No current number; prevents recurrence of H3-class staleness |
| H13 | Human ruling on the uncommitted `real_trace_experiments.py` hunk (hidden-state gate drop) — commit or revert, with rationale | M0 Task C | decision + commit | Medium if left ambient | No on-disk number yet; determines resume behavior of every future collection |
| H14 | Roster label: `mistral_small_24b_2409` is 22B, not 24B; document the 3p1 substitution in prose | M1 Axis 1.3, M0 Task A2 | 2 lines | None | Part 3 roster table correctness |

---

## 3. The honest ceiling (spec item 3)

**What the capture numbers actually say** (M2 §6; brief's cited ~50% online bound and 1/e ≈ 36.8% worst case, `Scientific_Rigor_Research_Prompt.md` §Module 3.3):

- Published 50.9% reproduces only under the hardcoded 12-late-cell selection that full-weights the defective 75-run cell. Honest late-cell variants: **48.84%** (11 cells), **50.41%** (12 cells with the fresh recompute), 48.96% (run-pooled). **All-cells run-pooled: 63.66%** (per-dataset 52.4 / 64.6 / 70.1 / 69.2).
- On the late-boundary cells — the regime the thesis narrative is about — the policy already sits **at ≈ the ~50% online bound** the brief cites. Claiming large further gains *there* would amount to claiming to beat the online-decision bound; the honest statement is that late-cell headroom is small and mostly structural.
- Matrix-wide, 63.66% > 50% is not a contradiction: the folk ~50%/1/e bounds are adversarial-instance results, and the all-cells denominator is inflated by deeply-negative `never_stop` on MATH/GPQA (M2 §6). Any advisor-facing use must state slice + aggregation (lever H10).

**Loss-mass decomposition of the remaining gap:**

| Slice of the 5,646 losses | Count | % of losses | % of runs | Verdict |
|---|---:|---:|---:|---|
| Tag (i) B + tool-fixable | 196 | 3.5% | 0.26 pp | Mostly closeable (P1) |
| Tag (iii) C+D+E | 2,710 | 48.0% | 3.64 pp | Partially closeable; realistic ~20–35% of it (§1) |
| F truncation-suspected sub-slice | 529 | 9.4% | 0.71 pp | Arguably (iii); token-cap/truncation levers (P3/P4) |
| F structural core | 2,211 | 39.2% | 2.97 pp | **Structural online-decision cost — not plausibly closeable** with recorded features (AUC 0.681, no feature |d| > 0.2; linear-probe caveat M2 Blocker 3) |

**Bottom line estimate [estimate, measured by P1–P4]:** realistic aggregate recovery is ~800–1,400 losses (≈1.1–1.9 pp of runs), i.e. loss rate 7.57% → **≈5.7–6.5%**, worth roughly 500–900 utility of the 3,543 lost — *before* subtracting win-side delay costs, which the offline census computes exactly. The structural core (~3 pp of runs, ~39% of losses) plus most of F is the price of deciding online, and the thesis should present it as such rather than as unfinished engineering. Anything promising loss ≪ 5% from these traces should be treated as a red flag, not a goal.

---

## 4. Master ranking — (estimated recovery or rigor value) × (implementation cost/risk)

Direct input to Module 4. "Recovery" is bounded by §1; "rigor value" = what advisor-facing claim it repairs.

| Rank | Lever | Type | Est. recovery / rigor value | Cost | Risk | M4 protocol |
|---|---|---|---|---|---|---|
| 1 | H3 qwen2p5_7b__math in-place re-analysis + report regen | hygiene | Fixes the known-defective published headline basis (7.5745→7.5495% on 75,965; capture →50.41%) | ~5 min CPU + commit | Low (expected values pre-known) | P5 |
| 2 | L-B empty-answer guard | policy (i) | ≤196 losses; realistic ~120–175 (−0.16 to −0.23 pp); can also only help utility on the loss side | 1-line policy predicate; offline sim first | Near-zero | P1 |
| 3 | H10 + H8 + H9 + H14 advisor-facing prose corrections | hygiene | Removes every known overstated/impossible claim from the check-in narrative | ~2 h prose | None | — (direct edit) |
| 4 | L-token-cap/truncation guard (E 479 + F 529) | policy (iii) | ≤1,008; realistic ~200–400 | 1-line predicate; offline sim | Low (win-side delay, measurable) | P1/P3 |
| 5 | H1 regrade-tool patches (+H2 regex, +H11 sympy guard) | hygiene | Prevents a ~12,284-label corruption event; makes the label-trust story durable | ~2–3 h | Low | P7 gate |
| 6 | L-X cross-cell threshold calibration | model | Largest evidence-backed model lever (slice AUC 0.76); unbounded by one category | ~1–2 days offline | Medium (leakage discipline) | P3 |
| 7 | L-C best-so-far cache + L-D step-1 revert | policy (iii) | ≤1,058 combined; realistic ~330–575 | ~2 days offline; reframes policy class | Medium (changes win-side too; framing) | P3 |
| 8 | L-E churn-gated hysteresis | policy (iii) | ≤1,652 but realistically ~165–415; blanket version provably ≈net-negative (§1.E) | ~1 day offline | Medium | P3 |
| 9 | H12 analysis_complete() fix + H4 CSV escaping + H13 hunk ruling + H6/H7 | hygiene | Prevents recurrence of the two worst data defects found by the audit | ~1 day | Low | — |
| 10 | Axis-2 clean bf16-vs-4bit rerun | rigor | Converts Module 1 Axis 2 CONFIRMED-WITH-CAVEAT → clean CONFIRMED | ~1 GPU-day-class | Low | P8 |
| 11 | Raw-text re-extraction audit | rigor | Bounds the extraction-stage blind spot (M2 Blocker 2) — the one loss-taxonomy residual | CPU hours | Low | P6 |
| 12 | Sampled label adjudication (MATH-heavy) | rigor | Absolute grader quality vs semantic truth (M1 Blocker 2) | API/human, ~3K rows | Low | P7 |
| 13 | λ-sensitivity sweep | rigor | Shows headline isn't knife-edged at λ=0.05 (backlog) | CPU re-scoring | Low | P9 |
| 14 | e-process honest reframing (union bound) | rigor | Removes an overclaimed math label from the writeup (deep_code_audit.md:55) | prose + rename | None | P10 (writeup) |
| 15 | Coding-domain wiring + pilot | scope | New-domain generalization (backlog; roster declared saturated, domain isn't) | days eng + GPU pilot | Medium | P11 |
| 16 | H5 stakes-λ harmonization | hygiene | Cross-readability of stakes utility levels | ~30 min | Low | — |
| 17 | mistral_small_3p1_24b retry | scope | Low: substitution already served the roster; 3p1 needs multimodal-aware loading | days eng + GPU | Medium | P12 (documented-decision only) |

**Explicitly deprioritized:** blanket +1-step patience (provably ≈net-negative, §1.E); any attempt to "fix" F's structural core with more features before a nonlinear-probe check; adding more LLMs to the matrix (July_1_Checkin Part 9 note: model dimension saturated).
