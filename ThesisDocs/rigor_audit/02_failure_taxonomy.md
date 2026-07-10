# Module 2: Failure Taxonomy at Scale

**Auditor:** Fable 5 (Module 2 pass) · **Date:** 2026-07-09 · **HEAD at audit:** `e049cc7` (working tree unchanged from Modules 0/1)
**Scope:** `ThesisDocs/Scientific_Rigor_Research_Prompt.md` §Module 2. Module 0 (`00_repo_state_and_staleness.md`) and Module 1 (`01_scientific_method_adversarial_verification.md`, FINALIZED) are binding inputs; their established facts are cited, not re-derived.
**Deliverable script:** `research/classify_losses.py` (finalized this pass from an interrupted predecessor's draft; deterministic; read-only w.r.t. `research/outputs/`; refuses `--out` paths inside it).
**Provenance discipline:** a predecessor session (killed mid-flight 2026-07-09 ~19:01) left a draft report and partial outputs in the session scratchpad. Per the Module 1 precedent (its draft contained a fabricated section), **every number below was re-executed by this pass** via the commands in §8; the predecessor's text was used only as a map. Deviations from the predecessor's draft are footnoted where found.

**Vocabulary (brief §0.4):** a "loss" is a *decision-quality* event — the `hazard_drift` policy's stop yielded strictly lower `stop_utility` than `never_stop` on the same run. It is not a model-accuracy number.

---

## 0. Verdict summary (headline first)

| Question | Answer |
|---|---|
| Does the July 1 headline reproduce from a fresh, independent (merge-based) code path? | **Yes, exactly**: 74,540 runs; win 66,784 (89.5948%), tie 2,110 (2.8307%), **loss 5,646 (7.5745%)**. Zero duplicate `(run_id, detector)` pairs, zero NaN utilities, zero unmatched hazard_drift/never_stop pairs, in all 51 populated cells (`m2_final/join_audit.txt`) |
| Is the tie boundary float-fragile? | **No.** Verdict counts are identical for tie tolerance ε ∈ {0, 1e-12, 1e-9, 1e-6, 1e-3, 0.01}; the smallest nonzero \|hd−ns\| in all 74,540 runs is **0.05** — exactly one step-cost quantum, so exact float equality is structurally safe (`slice_tables.txt` lines 6-13) |
| Is the headline label-robust? | **Yes — 0 of 74,540 verdicts move** under every label correction known. (a) Module 1's single genuine grader disagreement (`mistral_small_24b_2409__math …00257_20ddae87__temp1.00__seed7`, step 6) sits at neither that run's stop step (7) nor its final step (14) — verdict "win" unchanged (§3). (b) The 10 falsely-credited `'N/A'`→`'A'` rows (Module 1 §4a) live in 3 runs and none sits at a stop/final step — 0 flips (§3). (c) Loss-scoped regrade of all 5,646 losses' stop+final answers under corrected MCQ-aware semantics dissolves **0 losses** (§3, `m2_final/regrade_sensitivity.csv`) |
| `qwen2p5_7b__math` (75/1,500-run defective cell) three ways | Global loss rate: **7.5745%** as-is · **7.5794%** cell-excluded (n=74,465) · **7.5495%** with the full-cell out-of-place recompute spliced in (n=75,965). MATH-slice loss rate: 6.9215% / 6.9393% / **6.8721%**. The headline moves at most ±0.03 pp in any direction — the defect biases no conclusion, but the published run count is short 1,425 runs (§4) |
| Standing prediction "losses concentrate in GSM8K/MATH, thin in ARC/GPQA" | **Half right — REFUTED as stated.** GSM8K is decisively heaviest (11.23%) and ARC lightest (4.71%), but **MATH (6.92%) sits *below* GPQA (7.37%)** (§5) |
| Published 50.9% oracle-gap capture | **Reproduces exactly (50.90%)** under its own aggregation (mean of 12 late-boundary cell means), and the selection was reconstructed (= cells with `corrected_boundary_step > 2`); but it silently includes the defective 75-run cell at full cell weight. Honest variants: **48.84%** (11 cells excl. defect), 48.96% (12 cells run-pooled), **50.41%** (12 cells with the fresh full-cell recompute), **63.66%** (all 51 cells, run-pooled) (§6) |
| Taxonomy | 5 non-empty programmatic categories (+1 empty), mutually exclusive, first-match precedence, counts sum to **5,646 exactly**; the three legacy narrative modes do not partition the data (§7) |

---

## 1. Fresh global breakdown (T1)

**Definition** (verified against `July_1_Checkin.md` Part 6, `trace_analysis.py`, brief §0.4): per `run_id`, compare `stop_utility` of detector `hazard_drift` vs `never_stop` in `{cell}/detector_comparison_by_run.csv`; strict `>` win, exact `==` tie, strict `<` loss.

**Code path:** `classify_losses.fresh_verdicts` — outer-merge of the two detector frames per cell with an indicator column (deliberately *not* the `pivot()` of `scratch_analyze.py`/`analyze_runs.py`), plus explicit duplicate/NaN/unmatched accounting. Result reconciles **run-for-run exactly** with the reference breakdown (Module 0 Task E / Module 1 verdict H): no discrepant runs to explain.

| verdict | runs | share |
|---|---:|---:|
| win | 66,784 | 89.5948% |
| tie | 2,110 | 2.8307% |
| loss | 5,646 | 7.5745% |
| total | 74,540 | |

**Epsilon sensitivity:** identical counts at ε = 0 through 0.01; smallest nonzero \|diff\| = 0.05 (one step-cost quantum) — ties are exact by construction, not by floating-point luck.

**Mechanical structure (each checked exhaustively this pass, `m2_checks.py`):**
- `utility = correct − 0.05·(step−1)` holds with 0 violations across all sanitized step rows (`join_audit.txt`: no mismatch lines); `never_stop` stops at the run's last step.
- **Every loss is a missed late correction**: stop-step answer wrong AND final-step answer correct — **5,646/5,646** (script line `mechanical signature`). The margin identity `ns_util − hd_util = 1 − 0.05·(ns_step − hd_step)` holds 5,646/5,646 (margins 0.40–0.95, mean 0.6275). Since max horizon is 14 steps, `1 − 0.05·k > 0` always: **"stopped too late" mechanically cannot produce a loss** against never_stop — `July_1_Checkin.md` Part 6's "or too late (wasting tokens)" clause is impossible for this comparison and should be dropped.
- Ties: all 2,110 have `hd_step == ns_step` (policy ran to the horizon); 1,865 final-wrong + 245 final-correct.
- Win decomposition (from dcbr utilities): 36,070 stopped-wrong-and-stayed-wrong (pure step-cost saving), 26,589 stopped-correct-and-final-correct (cost saving), **4,125 stopped-correct-while-final-wrong (corruption actually prevented)**.
- Aggregate utility: the 5,646 losses cost 3,543.0 utility; the wins gained 27,815.1 — a 7.85:1 gain:loss ratio.
- **52.2% of all losses (2,946) stop at step 2**, the T_MIN floor — the modal loss is "quit at the first admissible step; repair arrived later."

*(Predecessor-draft deviations, for the record: its win decomposition 36,067/26,590 and tie split 1,864/246 were computed off current-trace labels rather than dcbr utilities; the dcbr-native numbers above are 36,070/26,589 and 1,865/245.)*

---

## 2. Known integrity notes inherited by this recompute

From Module 0/1, visible in my own `join_audit.txt`: the sanitizer (`trace_analysis._sanitize_step_frame`, imported — not re-implemented — for pipeline-identical drop semantics) removes the corruption fragments in exactly the 5 known cells (`llama_…gpqa` 24 rows/1348→1342, `qwen2p5_0p5b__gpqa` 41/1354→1342, `qwen2p5_3b__gpqa` 20/1349→1343, `yi_…gpqa` 25/1349→1343, `yi_…math` 14/1500→1499); `mistral_small_3p1_24b_instruct__gsm8k` has no comparison file (failed collect); `qwen2p5_7b__math` shows **6 dcbr-vs-current-trace stop-utility mismatches** — the only cell where the comparison file disagrees with its own current trace labels (both artifacts are on-disk pipeline outputs of different vintages; Module 1 §5c has the mechanism).

---

## 3. Regrade sensitivity — is 7.57% a labeling artifact? (T3, lean per Module 1 handoff)

Module 1 (binding, finalized): the real behavioral flip set under corrected per-dataset regrade semantics is **1 row in 798,770** (`01_…md` §4b/4c; scratchpad artifacts `flip_summary_by_cell.csv`, `m1v_flips_math*.py`; the 12,284-row `flip_rows.csv` is the regrade-*tool* artifact set, not behavioral flips). This pass did **not** re-derive that; it computed the verdict-level consequences directly:

1. **The 1 real flip row** (`mistral_small_24b_2409__math_test_00257_20ddae87__temp1.00__seed7`, step 6): that run's `hd_step=7`, `ns_step=14` — the row is at neither scoring step. Verdict ("win") unchanged. Moreover the stored label (correct=1) is the *right* label (the answer is mathematically correct; the flip is a grader parse-edge), so no correction is even warranted. **Headline movement: exactly 0.**
2. **The 10 `'N/A'`→`'A'` falsely-credited rows** (Module 1 §4a; independently re-located this pass with NA-safe reads): they live in 3 runs (`mistral_7b_instruct_v0p3__arc …00234…temp1.00` steps 4-6, `qwen2p5_14b__gpqa …00327…temp1.00` steps 1-2, `yi_1p5_9b_chat__gpqa …00405…temp1.00` steps 4-8). None sits at its run's stop or final step; de-crediting all 10 flips **0** verdicts (`m2_checks.py` output).
3. **Loss-scoped regrade (the script's `--with-regrade`, scope=losses):** all 5,646 losses' stop and final answers re-graded with the current fixed graders under both the tool's verbatim semantics (v1) and corrected MCQ-aware semantics (v2). Result (`m2_final/regrade_sensitivity.csv`):

| label variant | win | tie | loss | loss rate |
|---|---:|---:|---:|---:|
| dcbr as-is (published) | 66,784 | 2,110 | 5,646 | 7.5745% |
| v0: current trace labels, no regrade | 66,782 | 2,110 | 5,648 | 7.5771% |
| v1: tool regrade verbatim (incl. its MCQ bug) | 66,955 | 2,110 | 5,475 | 7.3451% |
| **v2: corrected regrade (arc/gpqa as mcq)** | **66,782** | **2,110** | **5,648** | **7.5771%** |
| v3: conservative (gsm8k/math regrade only) | 66,782 | 2,110 | 5,648 | 7.5771% |

   Reading: **v2 dissolves 0 of the 5,646 losses** (its only movement vs the published verdicts is 2 `qwen2p5_7b__math` runs flipping win→loss — the §2 stale-dcbr-vs-current-trace disagreement, a cell-vintage artifact, not a grader-semantics one; v0 shows the same 2 without any regrade). v1 — the regrade tool's verbatim buggy semantics — spuriously *dissolves* 173 ARC/GPQA losses (175 runs change: 69 arc, 104 gpqa, 2 math) by de-crediting their correct final MCQ answers: the Module 1 answer-type bug, reproduced at verdict level. loss_rate is stable at 7.57–7.58% under every defensible variant.

4. **Machinery validation (cheap, two cells, scope=all)** — `--cells qwen2p5_0p5b__arc,mistral_small_24b_2409__math --with-regrade --regrade-scope all` (3,000 runs, every stop/final row regraded, 67 s): v2 = v0 = dcbr **exactly (0 verdict changes)** — including the cell that contains the single real Module-1 flip row — while v1 changes 48 ARC verdicts (326→288 losses), again reproducing the tool-bug mechanism. The machinery is validated without a 52-cell sweep.

**Conclusion: the 7.5745% loss rate is label-robust to every known label defect — 0 of 74,540 verdicts move.** Do **not** run `research/regrade_traces.py` for real without the Module 1 handoff patches (`_answer_type` mcq fix + `keep_default_na=False`); as-is it would corrupt ~12,284 correct ARC/GPQA labels.

*(Scope note, disclosed: loss-scoped regrade detects losses that dissolve; the win→loss direction is bounded by items 1-2 — the only in-population rows where stored labels disagree with corrected grading — hence also 0.)*

---

## 4. `qwen2p5_7b__math` three ways (T4)

Per the Module 1 orchestrator ruling, the cell's complete 1,500-run traces were analyzed **out-of-place**: `trace_steps.csv`/`trace_runs.csv` copied to scratchpad `qwen7b_math_oop/` and the committed production script run there (`python research/trace_analysis.py --input-dir <scratchpad>/qwen7b_math_oop`, ~3 min CPU; wrote all 9-detector artifacts + PNGs into the scratchpad copy only; `git status research/outputs/` clean throughout). Correctness probe on the full cell: OOF Brier 0.1918, AUC 0.7809.

**Fresh full-cell result (1,500 runs):** win 1,374 (91.60%), tie 35 (2.33%), loss 91 (**6.07%**) — the cell is *less* lossy than the MATH average, and much less than the stale 75-run snapshot implied.

| slice | as-is (published basis) | cell excluded | full-cell out-of-place spliced |
|---|---:|---:|---:|
| GLOBAL n | 74,540 | 74,465 | 75,965 |
| GLOBAL loss | 5,646 = **7.5745%** | 5,644 = 7.5794% | 5,735 = **7.5495%** |
| GLOBAL win/tie | 89.5948 / 2.8307% | 89.6005 / 2.8201% | 89.6400 / 2.8105% |
| MATH n | 18,074 | 17,999 | 19,499 |
| MATH loss | 6.9215% | 6.9393% | **6.8721%** |

**Disclosed caveat:** `trace_analysis.py` trains the hazard models per cell (GroupKFold out-of-sample), so the fresh run is "what the pipeline would have published had its analyze stage re-run after collection completed" — not a re-scoring of the frozen 75-run-era policy. On the 75 overlapping run_ids, 13/75 verdicts differ (10 tie→win, 1 tie→loss, 2 win→loss; the old model, trained on 75 runs, frequently never fired and ran to step 14). The 2 win→loss runs (`…00022_8fba220b__temp0.10/0.60`) are the same runs whose stale dcbr rows contradict current trace labels (§2).

---

## 5. Per-slice loss rates (T2)

All computed directly from per-run verdicts (never from published win-rate complements). Full tables incl. model×dataset and the excluding-defective-cell variants: `m2_final/slice_tables.txt`. SE = binomial.

### By dataset

| dataset | n | loss | loss % | ±SE pp | excl. `qwen2p5_7b__math` | spliced fresh cell |
|---|---:|---:|---:|---:|---:|---:|
| gsm8k | 19,500 | 2,189 | **11.2256** | 0.226 | 11.2256 | 11.2256 |
| gpqa | 17,466 | 1,288 | 7.3743 | 0.198 | 7.3743 | 7.3743 |
| math | 18,074 | 1,251 | 6.9215 | 0.189 | 6.9393 | **6.8721** |
| arc | 19,500 | 918 | 4.7077 | 0.152 | 4.7077 | 4.7077 |

### By temperature (monotone)

| temp | n | loss % | ±SE pp |
|---|---:|---:|---:|
| 0.1 | 24,847 | 6.9465 | 0.161 |
| 0.6 | 24,846 | 7.5103 | 0.167 |
| 1.0 | 24,847 | 8.2666 | 0.175 |

Δ(1.0 − 0.1) = 1.32 pp, z ≈ 5.6 — hotter sampling produces more late repairs for the policy to miss. Excluding the defective cell: 6.9495/7.5138/8.2749 (unchanged conclusion).

### By model (extremes; full 13-row table in outputs)

Worst: `mistral_small_24b_2409` **16.43%** (its GSM8K cell alone: **30.27%**), `llama_3p1_8b_instruct` 11.18%. Best: `qwen2p5_0p5b` 2.02% (it almost never repairs late — nothing to miss), `mistral_7b_instruct_v0p3` 4.02%, `qwen2p5_7b` 4.44% (4.47% excluding its defective MATH cell). Side observation for Module 3: `qwen2p5_14b` has an outlier **tie** mass (824 ties = 14.10%, next highest 4.16%) — not loss-relevant, flagged.

### Concentration verdict (standing prediction: "losses concentrate in GSM8K/MATH, thin in ARC/GPQA")

- Loss shares of the 5,646: GSM8K 38.8%, GPQA 22.8%, MATH 22.2%, ARC 16.3% (GSM8K+MATH = 60.9% of losses on 50.4% of runs).
- Loss rates: **GSM8K 11.23% ≫ GPQA 7.37% ≥ MATH 6.92% > ARC 4.71%** (GSM8K vs GPQA: z ≈ 12.8; GPQA vs MATH: z ≈ 1.7, and MATH is certainly not *above* GPQA; MATH spliced-complete = 6.87% makes the ordering robust to the defective cell).

**VERDICT: REFUTED as stated.** GSM8K-heavy: confirmed decisively. ARC-light: confirmed. But MATH sits at/below GPQA, so the clean "math-repair-rich vs MCQ-floor" story fails — GPQA carries the second-highest loss rate. The published per-dataset *win* rates (MATH 86.40% < GPQA 92.05%) suggest the opposite ordering only because MATH's complement is mostly **ties** (6.68% vs GPQA's 0.57%) — exactly the win-rate-complement trap the brief warned about.

---

## 6. Oracle-gap capture: the 50.9% (T2, Module 1 handoff item 6)

**Selection reconstructed.** The "12 late-boundary cells" are hardcoded in the git-tracked helper `research/outputs/experiment_matrix/_thesis_audit_latecells.py` (committed in `3a5dde7`, the July-1 check-in commit — it generated Part 5's table). This pass re-derived the selection criterion: the 12 cells are **exactly** the cells with `corrected_boundary_step > 2` (i.e., above the T_MIN floor) in the four `_aggregate/{ds}/cross_family/cross_family_summary.csv` files — 9 GSM8K (qwen 3b/7b/14b/32b, mistral_7b, internlm3, yi, mistral_small, llama) + 2 MATH (qwen 7b/32b) + 1 ARC (mistral_small). GPQA contributes none.

capture = (mean hd − mean ns) / (mean oracle − mean ns), from dcbr oracle rows:

| slice | aggregation | capture |
|---|---|---:|
| 12 late cells | mean of per-cell means (published aggregation) | **50.90%** — reproduces `July_1_Checkin.md` Part 5 digit-for-digit (oracle 0.5563 / hd 0.3360 / ns 0.1077) |
| 11 late cells (excl. defective `qwen2p5_7b__math`) | cell-mean | **48.84%** |
| 12 late cells, defective cell replaced by fresh full-cell recompute | cell-mean | **50.41%** |
| 12 late cells | run-pooled | 48.96% |
| all 51 cells | run-pooled | **63.66%** |
| gsm8k / math / arc / gpqa | run-pooled | 52.43 / 64.62 / 70.09 / 69.20% |

Findings: (1) the published 50.9% counts the 75-run defective cell as a full-weight cell; under the same aggregation the honest figure is 48.8% (cell dropped) or 50.4% (cell completed) — "about half the oracle gap" survives, the third digit does not. (2) The late-boundary restriction *lowers* capture: the all-matrix figure is ~64% (never_stop is deeply negative on MATH/GPQA, inflating the denominator the policy clears easily). Any advisor-facing use should state the slice and the aggregation.

---

## 7. The taxonomy (T5)

**Population:** all 5,646 losses. **Join:** every loss joined to its full per-step history from the cell's `trace_steps.csv` after the production sanitizer (`_sanitize_step_frame` imported from `trace_analysis.py` — identical drop semantics to the pipeline that produced the comparison files). Coverage: 100% of losses, 0 missing stop-step rows, 0 duplicate `(run_id, step)` pairs.

**Design:** programmatic predicates, mutually exclusive by first-match precedence A→F, exhaustive by construction (F is the residual); every category validated non-overlapping and the counts **sum to 5,646 exactly** (`m2_final/taxonomy_summary.csv`).

| # | category | predicate (one line) | count | share | tag (§7.8) |
|---|---|---|---:|---:|---|
| A | grader_artifact_dissolves | re-scored verdict under corrected labels ≠ loss | **0** | 0.0% | (i) — empty |
| B | stopped_on_empty_answer | `answer_normalized` at stop step empty/NaN | **196** | 3.5% | (i) |
| C | passed_earlier_correct | correct answer existed at eligible step ∈ [2, stop) | **459** | 8.1% | (iii) |
| D | step1_only_correct | correct at forced-init step 1 only, never at eligible step ≤ stop | **599** | 10.6% | (iii)-qualified |
| E | next_step_repair | first eligible correct step = stop+1 | **1,652** | 29.3% | (iii) |
| F | late_repair | first eligible correct step ≥ stop+2 | **2,740** | 48.5% | (ii) |
| | **total** | | **5,646** | 100% | |

Category × dataset (rows sum to the counts above):

| category | arc | gpqa | gsm8k | math |
|---|---:|---:|---:|---:|
| B | 58 | 128 | 6 | 4 |
| C | 73 | 54 | 286 | 46 |
| D | 185 | 197 | 150 | 67 |
| E | 259 | 326 | 734 | 333 |
| F | 343 | 583 | 1,013 | 801 |

Category × model: `m2_final/slice_tables.txt`; largest single cells: F×`mistral_small_24b_2409` 547, F×`llama_3p1_8b` 337, E×`mistral_small` 220, D×`deepseek_r1_distill_1p5b` 148.

### 7.A grader_artifact_dissolves — 0 — tag (i), empty
Under corrected MCQ-aware regrade of every loss's stop/final answers, **no loss dissolves** (§3). The loss set is not a grading artifact — the strongest counter-finding to "the 7.57% is label noise." Residual disclosed: the regrade grades the *recorded* answer string; answers the extractor never captured are invisible to it (see B and Blocker 3).

### 7.B stopped_on_empty_answer — 196 (3.5%) — tag (i) pipeline/policy-guard bug
The policy stopped on a step where extraction produced **no candidate answer** (160 `json_field`, 36 `fallback` extraction at the stop step). Stopping with an empty candidate is a guaranteed-wrong decision that a one-line deterministic guard ("never stop while the current candidate is empty") removes. Concentrated in MCQ cells (GPQA 128, ARC 58) and in `llama_3p1_8b_instruct` (59) / `mistral_small_24b_2409` (54).
Examples: `deepseek_r1_distill_1p5b__arc_test_00249_d6f66ee0__temp0.10__seed7`, `…arc_test_00467_824e4f06__temp0.60__seed7`, `…temp1.00__seed7`.

### 7.C passed_earlier_correct — 459 (8.1%) — tag (iii)
A decision-eligible (step ≥ 2) correct answer existed **strictly before the stop**; the policy sailed past it, stopped on a wrong answer, and the trace repaired again later. The failure is not future-unpredictability — the winning state was already observed and abandoned. Heaviest: GSM8K 286; `mistral_small_24b_2409` 128. Probe: category-C losses separate from stopped-wrong wins at **AUC 0.824 [0.800, 0.847]** — the strongest category signal.
Examples: `deepseek_r1_distill_1p5b__arc_test_00092_25cc90e9__temp0.10__seed7`, `…gpqa_main_00257_3b6aa64c__temp0.10__seed7`, `…gsm8k_train_00004_3ee48c01__temp0.60__seed7`.

### 7.D step1_only_correct — 599 (10.6%) — tag (iii)-qualified
Correct at step 1 — the forced-commit init, not a decision point (T_MIN=2, `trace_analysis.py:20`; the floored oracle cannot take step 1 either) — corrupted by step 2, never correct again at any eligible step up to the stop. The *ideal* stop is protocol-unreachable (structural), but avoiding the *loss* only required continuing to the late repair, and signal existed: **66% (395/599) had `answer_changed=1` at the stop step** (the corruption itself was visible); probe AUC 0.747 [0.724, 0.770]. ARC 185 / GPQA 197 / GSM8K 150 / MATH 67; `deepseek_r1_distill_1p5b` alone 148.
Examples: `deepseek_r1_distill_1p5b__arc_test_00004_89613a4b__temp0.60__seed7`, `…00010_38aa3c95__temp0.60__seed7`, `…00016_75d83e3f__temp0.60__seed7`.

### 7.E next_step_repair — 1,652 (29.3%) — tag (iii)
The first eligible correct answer arrived **exactly one step after the stop** — one step of patience away from a win on 29% of all losses. At the stop step: 41% (680) answer still churning (`answer_changed=1`), **29% (479) the stopped-on generation had hit the token cap** (`hit_max_new_tokens=1` — a recorded, actionable live flag). GSM8K 734 / GPQA 326 / MATH 333 / ARC 259. Probe AUC 0.669 [0.654, 0.684].
Examples: `deepseek_r1_distill_1p5b__arc_test_00000_398537ec__temp0.10__seed7`, `…00010_38aa3c95__temp0.10__seed7`, `…00015_495befc7__temp0.60__seed7`.

### 7.F late_repair — 2,740 (48.5%) — tag (ii), with quantified caveat
Repair arrived ≥2 steps after the stop (gap median 3, mean 3.89, max 12). The policy's exact bet is that a currently-wrong trace won't repair — and it wins that bet 86.5% of the time (36,070 stopped-wrong runs never repair vs the 5,646 that do). Stop-step signal separability for F is the weakest of the categories (probe AUC 0.681 [0.670, 0.692]), and no single recorded feature separates losses from stopped-wrong wins (all standardized differences \|d\| ≲ 0.2) — this is the closest thing in the data to a genuine online-decision limit. Recoverable edge, disclosed: 529 F-losses (19%) stopped on a truncation-suspected step (`truncated_output_suspected=1`) — that sub-slice is arguably (iii) and is the first place to look for recoverable F mass.
Examples: `deepseek_r1_distill_1p5b__arc_test_00003_c16c54fb__temp0.10__seed7`, `…00022_51e7e2c1__temp1.00__seed7`, `…00023_56ce0db0__temp0.60__seed7`.

*(Predecessor-draft deviation: it claimed 767/28% truncated in F; recomputed value is 529/19% on `stop_truncated_output_suspected`.)*

### 7.8 Fate of the three legacy narrative modes (from the July 1 Scientific Method reports)
They do **not** map onto a partition of the losses:
- **"Arithmetic slip then late fix"** ≈ D∪E∪F *restricted to the math-style datasets* (GSM8K+MATH: 217+1,067+1,814 = 3,098 runs), but the identical step-shape carries 1,893 more losses on ARC/GPQA where "arithmetic slip" is not a meaningful description. It names a trajectory shape, not a mechanism.
- **"Sub-problem/intermediate-answer confusion"** is not programmatically detectable from recorded fields (no sub-answer annotation exists); whatever it is, it lives *inside* C/E/F. Testing it would require re-parsing `raw_text` (~800K rows with MB-scale text fields) — not done, flagged.
- **"Parser lag/misextraction"** splits into B (196 runs, real, now bounded) plus a hypothesis the data refutes at the whole-loss level: in **0 of 5,646** losses does the stop-step normalized answer equal the final-step normalized answer (the regrade also dissolves 0 losses) — the recorded stop answers are genuinely different and genuinely wrong, not lagged copies of the right answer.
- None of the legacy modes names the actual modal loss: **"quit at the T_MIN floor (step 2), repair arrived later"** — 52.2% of all losses.

---

## 8. Probe methodology — honesty section (T6)

**Question:** at the moment the policy stopped on a wrong answer, did the *recorded* signals carry information about whether the trace would later repair (→ loss) vs stay wrong (→ win)? This is exactly the α-hazard bet the policy makes.

**Setup** (`classify_losses.run_probe`): population = 41,715 stopped-wrong non-tie runs (5,646 positives = 13.5% base rate; ties excluded — they stop at the horizon, no future exists). 16 features recorded at/before the stop (entropy mean/std, confidence, answer_changed, answer_streak, thought tokens, hidden-state shifts, lexical echo, verbosity proxy, stop step, parse_success, hit_max_new_tokens, truncation flag, model_stop_flag, temperature) — **no post-stop information**. StandardScaler + class-balanced logistic regression; **GroupKFold(5) grouped by `task_id`** (stricter than the pipeline's run_id grouping: a problem seen at temp 0.1 in training can never appear at any temperature in test). Pooled out-of-fold AUC, Hanley–McNeil 95% CI. sklearn 1.8.0; full population, no subsampling.

| probe | OOF AUC | 95% CI |
|---|---:|---|
| global, signals only | **0.6428** | [0.635, 0.651] |
| global, signals + dataset/model one-hots | 0.7540 | [0.746, 0.762] |
| per-(dataset,model) base rate alone (no per-run signal) | **0.7597** | — |
| gsm8k / math / arc / gpqa, signals only | 0.7045 / 0.7173 / 0.6774 / 0.6497 | all lower-CI ≥ 0.63 |
| per category vs stopped-wrong wins | B 0.991 / C 0.824 / D 0.747 / E 0.669 / F 0.681 | B [0.982, 1.000], C [0.800, 0.847], D [0.724, 0.770], E [0.654, 0.684], F [0.670, 0.692] |

**Honest reading.** (1) Recorded per-run signals carry real out-of-sample repair information the deployed policy did not use (0.64 global, 0.65–0.72 within datasets) — but AUC ~0.64 is *weak*; nobody should promise large loss recovery from these features alone. (2) The single most informative "feature" is **slice identity**: knowing only (dataset, model) scores AUC 0.76 — per-cell loss rates span 0.13%–30.27% and the policy applies per-cell-trained hazards with no cross-cell risk calibration. (3) Category-level AUCs (C strongest → F weakest) are what justify the (iii) vs (ii) tags; they are separability-vs-wins measurements, not deployable-recovery estimates. (4) All probe claims are logistic-linear; nonlinear models could find more — absence of linear signal is evidence, not proof, of unpredictability.

---

## 9. Script, commands, runtimes (T7)

`research/classify_losses.py` — finalized: deterministic (fixed fold construction, sorted iteration), documented taxonomy predicates, refuses `--out` inside `research/outputs/`, reuses audited helpers (`analyze_runs.parse_temp`, `trace_analysis._sanitize_step_frame`) instead of duplicating them. New this pass: `--regrade-scope {losses,all}` (default `losses` — cheap category-A support; `all` for full sweeps), `--cells` filter (cheap validation runs), per-category probes, NA-handling note in `regrade_stop_rows`.

Commands executed this pass (all outputs under the session scratchpad; runtimes on the Windows box, CPU only):

```
python research/trace_analysis.py --input-dir <scratchpad>/qwen7b_math_oop        # T4, ~3 min
python research/classify_losses.py --out <scratchpad>/m2_out --with-probe         # run 1, ~5 min
python research/classify_losses.py --out <scratchpad>/m2_final \
       --with-regrade --regrade-scope losses --with-probe                         # run 2 (definitive), ~1.5 min; re-run once after a display-string edit — outputs identical (determinism check)
python research/classify_losses.py --out <scratchpad>/m2_val \
       --cells qwen2p5_0p5b__arc,mistral_small_24b_2409__math \
       --with-regrade --regrade-scope all                                         # run 3 (validation), 1 min 07 s
python <scratchpad>/m2_checks.py                                                  # T3/T4 splice + mechanical checks, ~2 min
```

Not committed (no git per remit); the script file is the deliverable, ready for the orchestrator's end-of-audit commit.

---

## 10. HANDOFF TO MODULE 3 (ranked by loss count at stake)

1. **F_late_repair — 2,740 losses (48.5%), tag (ii).** Realistic ceiling argument: the policy already wins the no-repair bet 86.5% of the time; stop-step features separate F only at AUC 0.681 [0.670, 0.692]. Treat most of F as structural online-decision cost. Recoverable edge: the 529 truncation-suspected F-stops (+479 in E) — a "never stop on a token-capped/truncated step" guard is deterministic and cheap; net utility must be checked against win-side costs (delaying stops on truncated steps that would have been good).
2. **E_next_step_repair — 1,652 losses (29.3%), tag (iii).** One-step patience/hysteresis (e.g., require the drift signal negative on 2 consecutive steps, or answer_changed-gated stopping) directly targets E (upper bound 1,652 recovered) — but every added step costs 0.05 × (all 74,540 runs that stop) on the win side; Module 3 must net it. The 479 token-capped E-stops are the free subset.
3. **D_step1_only_correct — 599 losses (10.6%), tag (iii)-qualified.** Init-answer protection: keep the step-1 candidate as a revert target when eligible-step answers churn (66% had answer_changed at stop). Also feeds the "is T_MIN=2 right?" protocol question (Module 4).
4. **C_passed_earlier_correct — 459 losses (8.1%), tag (iii).** The policy observed a correct state and left it: stop-on-stability features (answer_streak already recorded; probe AUC 0.824 [0.800, 0.847] for this category) or a best-so-far answer cache (stop returns the highest-q̂ candidate seen, not the current one — a protocol change worth a pre-registered experiment).
5. **B_stopped_on_empty_answer — 196 losses (3.5%), tag (i).** One-line guard: never stop while `answer_normalized` is empty. Upper bound 196 (recovery not guaranteed — the run must still repair in time; but the guard also can't hurt: an empty answer scores 0 wherever it stops... net-positive analysis is one line of algebra on the same data).
6. **Cross-cell risk calibration (affects all categories).** Slice identity alone predicts losses at AUC 0.76 (per-cell loss rates 0.13%–30.27%): a per-cell stopping-threshold calibration (e.g., λ or drift-threshold per cell) is the highest-leverage *model* change the probe evidence supports.
7. **Bookkeeping levers:** re-run the analyze stage for `qwen2p5_7b__math` in place (production step; refreshes the headline to ~7.55% on 75,965 runs and Part 5's 50.9%→50.4%); fix `run_experiment_matrix.analysis_complete()` file-existence check that let the stale cell survive (Module 1 §5c); drop the impossible "or too late" clause from `July_1_Checkin.md` Part 6; add the slice/aggregation caption to every future use of "~51% oracle capture".

## Blockers

1. **`qwen2p5_7b__math` in-place repair is a production pipeline run** (out of audit scope). Out-of-place numbers above are complete and can be quoted with the retraining caveat of §4; the published tree remains internally stale until a human-approved re-run of `research/trace_analysis.py` on the cell.
2. **Extraction-stage misses are invisible to any regrade.** `parse_success=0` at a nontrivial share of stops means recorded answers sometimes came from fallback heuristics; a raw_text-level re-extraction audit (~800K rows, MB text fields) was not performed. B's 196 bound the *detected* empty-answer mass; the undetected wrong-extraction mass is unbounded by this report (Module 1's 4-bit re-extraction spot-check — 0.42% label delta on 9K rows — is the only calibration point).
3. **Probe is logistic-linear only** (by design, cheap and honest); tags (ii)/(iii) could shift at the margin under nonlinear probes. The C/D/E ordering is robust in this data; treat F's (ii) as "(ii) at linear evidence."
4. `research/outputs/experiment_matrix/_thesis_audit_latecells.py` is analysis code living inside an outputs tree (committed `3a5dde7`) — harmless but misplaced; move to `research/` in a housekeeping commit (not done here: outputs tree is write-forbidden this pass).
