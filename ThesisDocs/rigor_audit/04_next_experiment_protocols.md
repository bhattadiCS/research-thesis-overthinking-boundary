# Module 4: Next-Experiment Protocols (Pre-Registered)

**Auditor:** Fable 5 (Modules 3+4 merged pass) · **Date:** 2026-07-09 · **HEAD at audit:** `e049cc7`
**Inputs:** Modules 0/1/2 reports (binding), `03_improvement_levers.md` (this pass), spec `Scientific_Rigor_Research_Prompt.md` §Module 4, sample-size method from `ThesisDocs/July_1_Checkin.md` Part 7 (SE = √(p(1−p)/N), Z = δ/SE — reused, not reinvented).

**Shared statistical facts used below (Part 7 method applied to the frozen corpus):**
- Census SE of the 7.5745% loss rate at N=74,540: SE = √(0.0757·0.9243/74,540) = **0.097 pp**. A loss-rate change is detectable at Z≥1.96 if it exceeds 1.96·√2·SE ≈ **0.27 pp** under the conservative unpaired treatment. All offline re-simulations below are **paired censuses** (same runs, only guard-affected runs can change verdict), so effects far below 0.27 pp are still exactly countable — significance is then reported as an exact discordant-run count, with the 0.27 pp bar quoted only for "would this generalize to a fresh sample" framing.
- GPU cost anchors (recorded, `matrix_manifest.json` collect elapsed_min, Blackwell 96GB box): qwen2p5_7b__gsm8k 176.8 min; mistral_small_24b_2409__gsm8k 217.1 min; qwen2p5_32b__gsm8k 233.5 min; qwen2p5_7b__math 438.5 min. Per-cell ≈ **3–7.5 GPU-h**.
- 7-axis held-constant checklist (Module 1): 1 model scale · 2 precision/quantization · 3 temperature · 4 dataset/grading · 5 question/trace-set identity · 6 stakes c · 7 reasoning paradigm. For offline re-simulations on frozen traces, **all 7 axes are held constant by construction** (no new generation occurs); the block below then only names the analysis-level constants.

---

## 1. Backlog reconciliation (FIRST — status of every named item)

| Backlog item | Status | Evidence | Action |
|---|---|---|---|
| MATH GPU-verification completeness across the roster | **DONE at collection**: all 13 MATH cells collected+graded on GPU in the 06-27 sweep (52/52 cells, M0 Task A1). **One residual analyze-stage defect**: `qwen2p5_7b__math` detector stage covers 75/1,500 runs (M0 A2, M1 §5c); full-cell out-of-place recompute already done (M2 §4: cell loss 6.07%) | Module 0/1/2 reports | P5 (in-place repair, human-approved) |
| `llama_3p1_8b_instruct` | **CONFIRMED — real data in all 4 datasets**: 52 populated cells = 13 models × 4 datasets (M0 A1); llama cells appear individually in M0's tables (gsm8k collected 06-26 per M1 Axis 1; gpqa 1,342 dcbr runs per M0 A2) and in M2's per-model loss table (11.18% across its 4 cells) | M0 A1/A2, M1 Axis 1, M2 §5 | None — closed |
| `mistral_small_3p1_24b` | **Still failed / substituted.** Collect failed in 0.13 min, return_code 1 (M0 A2) — vision-language checkpoint unloadable by the text-only harness (M1 Axis 1.3). Substituted by `mistral_small_24b_2409` (a **22B** model, listed as "24B" in Part 3). Substitution documented **only** in `matrix_manifest.json` — no prose doc narrates it (M0 A2) | M0 A2, M1 Axis 1.3 | P12: document the substitution decision (lever H14); **do not retry** by default — rationale in P12 |
| Coding domain (HumanEval/MBPP) | **Never wired.** `real_trace_experiments.py:2339` `--task-source` choices = {builtin, gsm8k, math, arc, gpqa}; `run_experiment_matrix.py:92` carries only a commented-out humaneval stub: `# "humaneval": … # needs parser (code execution)` | direct file check this pass | P11 (wire + pilot) |
| e-process honest reframing | **Pending.** `research/reports/deep_code_audit.md:55,95`: the "e-process" detector is a per-step batch e-value **union bound**, not a time-uniform supermartingale (no Ville optional-stopping validity); writeup still uses the martingale-flavored name; `mathematical_foundations_proofs.md:188,222` already derives the union-bound form | direct file check this pass | P10 (writeup + rename) |
| λ-sensitivity sweep | **Pending.** Flagged in deep_code_audit remediation leftovers (project memory, 06-20) and never run; no λ-sweep artifact exists in the repo [checked: no `lambda` sweep output found this pass] | grep this pass | P9 |

---

## 2. Protocols

Format per spec: exactly one IV; 7-axis held-constant list; falsifiable hypothesis fixed in advance; minimum n via Part 7 SE/Z (arithmetic shown); cost (GPU-h or CPU-ONLY/API-ONLY); success/failure criterion fixed in advance. Experiments with multiple candidate policies (P3) are pre-registered as separate single-IV arms against one shared baseline.

### P1 — Offline re-simulation of the empty-answer guard (Module 3 lever L-B) — CPU-ONLY

- **IV (one):** presence of the guard "policy may not stop on a step whose `answer_normalized` is empty/NaN" in the stopping rule. Everything else in the policy is byte-identical.
- **Held constant (7-axis):** all 7 axes by construction (frozen traces, no generation). Analysis constants: same 52-cell corpus, same sanitizer (`_sanitize_step_frame`), same utility λ=0.05, same T_MIN=2, same hazard model outputs — only the stop-step selection changes.
- **Hypothesis (falsifiable):** the guard strictly increases total corpus utility, dissolving ≥100 of the 196 B-losses while converting <50 wins to losses. Falsified if net Δutility ≤ 0 or win→loss conversions ≥ loss dissolutions.
- **Minimum n:** census, N=74,540 (fixed). Detectability arithmetic: max effect 196/74,540 = 0.26 pp vs the 0.27 pp unpaired bar — marginal unpaired, but the paired design makes it exact: every changed verdict is enumerated; binomial SE on the dissolved fraction of 196 candidates = √(0.5·0.5/196) = 3.6 pp, so a 100/196 vs 98/196 split is resolvable to ±7 pp — adequate.
- **Cost:** CPU-ONLY, ~30–60 min (one pass over dcbr + trace_steps, reusing `classify_losses.py` joins).
- **Success:** net Δutility > 0 AND ≥100 B-losses dissolved AND ≤50 win→loss. **Failure:** anything else → guard is not shipped; result still publishable as a negative.

### P2 — Offline T_MIN=2 vs T_MIN=3 policy variant (Module 3 §1.D) — CPU-ONLY

- **IV:** T_MIN ∈ {2, 3} in the stopping rule (floor only; hazard fits unchanged).
- **Held constant:** all 7 axes by construction; same corpus/sanitizer/λ/utility; the **oracle stays floored at 2** in both arms (comparator fixed — otherwise the denominator moves too).
- **Hypothesis:** raising the floor to 3 does NOT increase net utility — i.e., the 2,946 step-2 loss-stops (52.2% of losses, M2 §1) are outweighed by the +0.05 delay charged to every step-2-stopping win. (Directional pre-registration: we expect the floor to survive; the experiment is a quantified defense of a structural choice, per Module 3 §1.D.)
- **Minimum n:** census, N=74,540. The number of step-2 stops among wins is currently unquantified [UNVERIFIED — computed as the first output of this protocol]; the effect is a deterministic paired count, no sampling.
- **Cost:** CPU-ONLY, ~30 min.
- **Success (for the thesis):** T_MIN=2 arm has ≥ the net utility of T_MIN=3 → the floor is defended with numbers. If T_MIN=3 wins by >200 net utility, the floor choice must be re-argued in the writeup — either outcome is usable; the criterion is fixed now.

### P3 — Offline feature-augmented / calibrated policy family (Module 3 levers L-C, L-D, L-E, L-X) — CPU-ONLY

Four pre-registered arms, each a single IV vs the shared frozen baseline policy; evaluated with GroupKFold(5) by `task_id` (stricter grouping, matching M2 §8); reported jointly with a 4-way multiple-comparison note.

| Arm | IV (one per arm) | Targets | Hypothesis (falsifiable) |
|---|---|---|---|
| P3a | token-cap guard: may not stop on `hit_max_new_tokens=1` step | E 479 + F 529 | net Δutility > 0; ≥200 of the 1,008 dissolve |
| P3b | churn-gated hysteresis: if `answer_changed=1` at candidate stop, require 2 consecutive negative-drift steps | E 680 churning | net Δutility > 0; ≥150 E-losses dissolve |
| P3c | best-so-far selection: return argmax-q̂ eligible candidate at stop (stop step unchanged) | C 459 (+D 599 via step-1 excluded/included sub-variants — step-1 inclusion is a *separate pre-registered sub-arm* since it crosses the T_MIN theory line) | ≥180 C-losses dissolve with ≤90 win→loss |
| P3d | per-cell threshold calibration: one scalar drift-threshold offset per cell, fit on GroupKFold train folds only | all (slice AUC 0.76, M2 §8) | out-of-fold net Δutility > P3a–c individually |
- **Held constant:** all 7 axes by construction; same probe feature set as M2 §8 (16 recorded stop-time features, no post-stop information — hard rule); same λ/T_MIN/oracle.
- **Minimum n:** census N=74,540; OOF evaluation → the Part 7 bar applies per arm: detectable loss-rate change ≥0.27 pp unpaired-conservative; arms whose upper bound is below that (none — smallest UB is P3c's 459 = 0.62 pp) are adequately powered.
- **Cost:** CPU-ONLY, ~1–2 days total (mostly harness extension of `classify_losses.py`).
- **Success per arm:** fixed in table above; overall success = at least one arm clears its bar out-of-fold. **Failure:** no arm clears → publish as "recorded features do not support deployable recovery beyond the deterministic guards," consistent with M2 §8's weak-AUC warning.

### P4 — Token-cap/truncation causal study (Module 3 §1.E/F) — GPU

P4a (offline half) is P3a above. P4b tests the *cause*:
- **IV:** `--max-new-tokens` 256 → 512, one cell: `mistral_small_24b_2409__gsm8k` (worst loss cell 30.27%, heavy truncation involvement — M2 §5, §7).
- **Held constant (7-axis):** 1 same model; 2 bf16; 3 temps {0.1,0.6,1.0}; 4 same grader (number); 5 same 500 tasks, shuffle-seed 17, seed 7; 6 c=0, λ=0.05; 7 same paradigm. Same July code, same Blackwell box, same batch size as the recorded cell. Control arm = the existing cell (same code vintage — verified collected post-`122358a`, M1 Axis 4 timeline).
- **Hypothesis:** doubling the token budget reduces the cell's loss rate by ≥5 pp (from 30.27%) by removing truncation-forced answer instability. Falsified if Δ < 5 pp.
- **Minimum n (Part 7):** p₁=0.3027, p₂=0.2527, δ=0.05: n ≥ (p₁q₁+p₂q₂)·(1.96/δ)² = (0.2111+0.1888)·(39.2)² /… = 0.3999·1,536.6 ≈ **615 runs/arm**. The standard cell (1,500 runs) is 2.4× that — powered.
- **Cost:** 1 new arm ≈ existing cell's 217 min × ~1.6 (longer generations) ≈ **6 GPU-h**.
- **Success:** loss rate at 512 ≤ 25.3% (δ≥5 pp, Z≥1.96). **Failure:** smaller drop → truncation is correlate not cause; the offline guard (P3a) remains the only truncation lever.

### P5 — `qwen2p5_7b__math` in-place re-analysis + report regeneration (lever H3) — CPU-ONLY, human-approved production step

- **IV:** none in the experimental sense — a pipeline repair with pre-registered expected values (the audit's out-of-place recompute, M2 §4). Pre-registration here = prediction of the outcome before the production run.
- **Held constant:** everything — same committed `trace_analysis.py`, same cell traces.
- **Prediction (falsifiable):** in-place run reproduces the OOP numbers: cell win/tie/loss = 1,374/35/91 (6.07% loss); global headline becomes 75,965 runs / **7.5495%** loss; MATH slice 6.8721%; 12-late-cell capture 50.41% (M2 §§4,6). Any deviation ⇒ nondeterminism or environment drift ⇒ stop and investigate before trusting either number.
- **Minimum n:** N/A (deterministic recompute; determinism already demonstrated by M2 §9's repeat run).
- **Cost:** CPU-ONLY ~5 min + regeneration of the cell's `summary.md`/`final_results.md` + commit. Requires the human approval Module 1's orchestrator ruling reserved.
- **Success:** exact match to prediction. This is the **first** thing to run: every advisor-facing headline citation should switch to 75,965/7.5495% afterward.

### P6 — Raw-text re-extraction audit (M2 Blocker 2) — CPU-ONLY

- **IV:** extraction vintage — stored `answer_normalized` vs current-parser re-extraction from stored `raw_text`, same rows.
- **Held constant:** all 7 axes by construction; same grader (current, H1/H2-patched first); same rows.
- **Hypothesis:** re-extraction changes ≤0.5% of labels and flips 0 verdicts — i.e., the loss taxonomy is not an extraction artifact. Calibration prior: the 4-bit tree spot-check found 0.42% label delta on 9,000 rows (M1 Axis 2).
- **Minimum n (Part 7):** to estimate a delta rate p≈0.005 with half-width 0.1 pp: n ≥ p(1−p)·(1.96/0.001)² = 0.004975·3.84×10⁶ ≈ **19,110 rows** → a 20K-row stratified sample (over-weighting stop/final steps of the 5,646 losses, which is where verdicts can move) suffices; full 798,770-row census preferred if the parser pass is fast enough (budget: hours; MB-scale text fields — M2 §7.8 warning).
- **Cost:** CPU-ONLY, ~2–8 h depending on sample vs census.
- **Success:** label delta ≤0.5% AND 0 loss verdicts dissolve. **Failure:** more → extraction is a live confound; B's 196 was a lower bound and the taxonomy needs a re-pass.

### P7 — Sampled label adjudication, MATH-weighted (M1 Blocker 2) — API-ONLY (+human spot check)

- **IV:** grading source — pipeline `correct` vs independent adjudication (frontier-LLM judge with human audit of disagreements).
- **Held constant:** rows fixed; adjudicator never sees the pipeline label (blind); prompt/rubric frozen before the first row.
- **Hypothesis:** pipeline grader accuracy ≥99% on MATH stop/final-step answers (the sympy-fallback dataset, where the only genuine flip lived — M1 §4c); ≥99.5% on the other three.
- **Minimum n (Part 7):** error rate p≈0.02 measured to half-width 0.5 pp: n ≥ 0.02·0.98·(1.96/0.005)² = 0.0196·153,664 ≈ **3,012 rows** → 3,000 stratified (≥1,500 MATH; stop/final steps of losses over-sampled).
- **Cost:** API-ONLY (~3,000 judge calls) + ~2 h human review of disagreements. No GPU.
- **Success:** both accuracy bars met → "labels adjudicated at scale" becomes an advisor-facing sentence. **Failure:** MATH < 99% → apply H11 grader fixes and re-run the M2 regrade sensitivity before quoting the taxonomy again.

### P8 — bf16 vs 4-bit clean rerun, same code, same box (M1 HANDOFF 8; converts Axis 2 caveat → clean CONFIRMED) — GPU

- **IV:** quantization flag: `--quantization none` vs `4bit`, Qwen2.5-7B.
- **Held constant (7-axis):** 1 same model; 3 temps {0.1,0.6,1.0}, seed 7; 4 GSM8K-train, current grader; 5 same 500 tasks, shuffle-seed 17; 6 c=0; 7 same paradigm. Same July code vintage, same Blackwell box, **same batch size both arms** (fixes all four Axis-2 mismatches: task count, code vintage, hardware, batch — M1 Axis 2 table).
- **Hypothesis:** the step-2 correctness crash reproduces under clean isolation: q₂(4bit) ≤ q₂(bf16) − 0.10 (April data: 0.1444 vs 0.2978 on matched tasks — M1 Axis 2). Falsified if the gap < 0.10.
- **Minimum n (Part 7):** p₁=0.30, p₂=0.14, δ=0.10 (conservative): n ≥ (0.21+0.12)·(19.6)² ≈ 0.33·384 ≈ **127 runs/arm**; at δ=0.22 (point estimate) only ≈28. The standard 500×3=1,500/arm is ≥11× overpowered — chosen anyway for protocol identity with the ladder tree.
- **Cost:** bf16 arm ≈ 3 GPU-h (recorded 176.8 min for this exact cell); 4-bit arm similar or slower ≈ 3–4 GPU-h → **≈6–8 GPU-h total**.
- **Success:** gap ≥0.10 with Z≥1.96 (SE at n=1,500/arm: √(0.33/1500)=1.48 pp ⇒ Z≈6.7 at δ=0.10) → Axis 2 verdict upgraded to CONFIRMED and the quantization-confound paragraph becomes clean. **Failure:** gap <0.10 → the April crash was partly code/hardware vintage; Scientific-Method reports need a further correction — still a win for rigor.

### P9 — λ-sensitivity sweep (backlog) — CPU-ONLY

- **IV:** step cost λ ∈ {0.01, 0.02, 0.05 (published), 0.10, 0.20}, entering both the drift rule and the utility, re-scored on frozen traces (same re-simulation machinery as the stakes sweep, which is the c-axis analogue — M1 Axis 6).
- **Held constant:** all 7 axes by construction; c=0; T_MIN=2; same corpus. Use the pipeline convention λ·(step−1) (lever H5), not the stakes script's λ·step.
- **Hypothesis:** the headline win rate stays within ±5 pp of 89.6% for λ ∈ [0.02, 0.10] (factor-2.5 band around the published choice) — i.e., the result is not knife-edged on the λ=0.05 convention.
- **Minimum n:** census N=74,540 per λ; Part 7 SE on a win rate of 0.896 = √(0.896·0.104/74,540) = 0.11 pp — any drift ≥0.31 pp between λ values is unpaired-detectable; the ±5 pp criterion is 16× that.
- **Cost:** CPU-ONLY, ~2–4 h (5 full re-scores).
- **Success:** criterion met → one advisor-facing robustness sentence + a small table. **Failure:** win rate leaves the band → λ becomes a headline caveat and the stakes-sweep narrative needs a λ column.

### P10 — e-process honest reframing (backlog) — writing task, CPU-ONLY

Not an experiment; pre-registered edit. Replace the martingale/e-process framing with the union-bound statement already derived in `mathematical_foundations_proofs.md:188,222`, per `deep_code_audit.md:55,95`; rename the detector label in future writeups ("union-bound e-value grid," keeping the on-disk `e_process` column name with a glossary note to avoid touching outputs). Success: no thesis document claims Ville/optional-stopping validity for it. Cost: ~2 h prose. Failure mode: none (pure honesty repair).

### P11 — Coding-domain wiring + pilot (backlog) — engineering + GPU

- **Phase 1 (engineering, no experiment):** wire `humaneval` into `real_trace_experiments.py` (loader + `answer_type="code"` + sandboxed execution grader) mirroring the existing dataset loaders; unit tests analogous to `test_graders.py`. ~2–4 days. The grader is the hard part (code execution ≠ string match; `run_experiment_matrix.py:92` already warns this).
- **Phase 2 (pilot experiment):**
  - **IV:** dataset = humaneval (vs the 4 wired datasets as fixed reference points).
  - **Held constant (7-axis):** 1 one model, `qwen2p5_7b` (mid-ladder, well-characterized); 2 bf16; 3 temps {0.1,0.6,1.0}, seed 7; 5 all 164 HumanEval tasks, shuffle-seed 17; 6 c=0, λ=0.05; 7 same paradigm. Axis 4 necessarily varies — that IS the IV (M1's Factor-4 warning applies: dataset bundles horizon/split/grader; disclose).
  - **Hypothesis:** the overthinking-boundary structure generalizes: a late (step ≥3) corrected boundary exists and hazard_drift beats never_stop on >70% of runs. Falsified if boundary = floor or win rate ≤70%.
  - **Minimum n (Part 7):** win rate measured at p≈0.85 on n=164×3=492 runs: SE = √(0.85·0.15/492) = 1.61 pp → the 70% criterion sits ≈9 SE below 85% — adequately powered for the direction call.
  - **Cost:** ≈ **1–2 GPU-h** (492 runs ≈ ⅓ of a 3-h cell) + sandbox infra.
- **Success:** hypothesis holds → coding domain joins the matrix roadmap (MBPP next). **Failure:** publishable boundary-scope finding (mirrors the ARC/GPQA "no boundary" result — M2 §5 context).

### P12 — `mistral_small_3p1_24b` decision (backlog) — documented decision, no run by default

Pre-registered recommendation: **do not retry.** Basis: the failure is architectural (vision-language checkpoint, text-only harness — M1 Axis 1.3), a retry requires multimodal-aware loading work with no hypothesis riding on that specific checkpoint; the roster is declared saturated (July_1_Checkin Part 9 note); the substitute `mistral_small_24b_2409` (22B) has complete data in all 4 datasets (M0 A2). Required instead (lever H14, ~2 lines of prose): document the substitution + the 22B-not-24B correction in the check-in roster table. Revisit only if a multimodal harness lands for unrelated reasons.

### P13 — Temperature-monotonicity confirmatory analysis (candidate k) — CPU-ONLY

- **IV:** temperature ∈ {0.1, 0.6, 1.0} (existing data; Axis 3 isolation already CONFIRMED — M1).
- **Held constant:** all other axes per M1 Axis 3 (same tasks, seed, config within cells).
- **Honesty clause (pre-registered):** the direction (loss rate 6.95→7.51→8.27%, z≈5.6 on the extremes — M2 §5) has **already been observed**, so this cannot be sold as independent confirmation. Value = fixing the test statistic in advance of the *formal* analysis: one-sided Cochran–Armitage trend test across the three temperature groups, α=0.05, plus per-dataset replication (pre-registered secondary: trend holds within ≥3 of 4 datasets).
- **Minimum n (Part 7):** census; SE per temperature arm at p≈0.075, n≈24,847: √(0.0694/24,847) = 0.167 pp; extreme-arm δ=1.32 pp ⇒ Z≈5.6 — the already-known power; the sub-hypothesis (within-dataset replication) is the genuinely new content.
- **Cost:** CPU-ONLY, ~1 h.
- **Success:** trend significant globally and in ≥3/4 datasets → "hotter sampling produces more late repairs the policy misses" becomes a defensible thesis claim with a named test. **Failure of the secondary** → claim is stated matrix-global only.

---

## 3. Sequencing (cheapest-highest-value first) and what each unblocks

| Order | Protocol | Cost | Unblocks / delivers | Advisor-facing claim changes? |
|---|---|---|---|---|
| 1 | P5 qwen-cell repair | CPU 5 min (+approval) | The corrected headline (75,965 / 7.5495%) and capture (50.41%); every later experiment then runs on a non-defective corpus | **YES** — headline + Part 5 |
| 2 | H1/H2/H11 tool patches (Module 3 §2) | ~3 h | Gates P6/P7 (any regrade-adjacent work must use the patched tool) | No (protective) |
| 3 | P1 empty-answer guard sim | CPU ~1 h | First lever verdict; template harness for P2/P3 | YES if shipped (−0.16 to −0.23 pp loss) |
| 4 | P2 T_MIN 2v3 | CPU ~30 min | Quantified defense of the floor (52.2%-of-losses talking point) | YES — turns a vulnerability into a defended choice |
| 5 | P3 policy-family arms (incl. P4a) | CPU 1–2 days | The recovery-vs-ceiling table in Module 3 §3 gets measured numbers; decides which levers ship | YES — the "realistic recovery" claim |
| 6 | P9 λ-sweep | CPU 2–4 h | Robustness sentence; needed before any stakes-adjacent advisor discussion | YES (robustness) |
| 7 | P13 temperature confirmatory | CPU 1 h | Named-test upgrade of an existing observation | Minor |
| 8 | P6 re-extraction audit | CPU 2–8 h | Closes the last taxonomy blind spot (M2 Blocker 2); prerequisite for calling B's 196 a *bound* | YES if it fails; hygiene if it passes |
| 9 | P7 label adjudication | API + 2 h human | Absolute grader-quality claim (M1 Blocker 2) | YES — "labels adjudicated" |
| 10 | P8 bf16-vs-4bit clean pair | ≈6–8 GPU-h | Axis 2 → clean CONFIRMED; removes the last isolation caveat with an experiment | YES — methods section |
| 11 | P4b token-budget arm | ≈6 GPU-h | Causal status of truncation in E/F; informs whether 512 tokens becomes the default | Maybe (protocol default) |
| 12 | P11 coding wiring + pilot | days eng + 1–2 GPU-h | Domain-generalization chapter material | YES (scope) |
| 13 | P10 e-process reframe; P12 substitution note; remaining Module 3 §2 hygiene (H4–H10, H12–H14) | prose/hours | Writeup honesty; recurrence prevention | YES (wording only) |

**Advisor-facing vs internal:** items 1, 3–5, 8–12 change or defend claims the advisor sees; items 2, 6–7, 13 are robustness/hygiene that make those claims durable. Nothing in this list requires the GPU before item 10, and items 1–9 deliver the bulk of the rigor value for ≈zero hardware cost.

## Blockers

1. **P5 requires explicit human approval** — it is the one production-tree write (Module 1 orchestrator ruling reserved in-place repair for a human-approved step). Everything else sequenced after it can technically run before it, but would then need re-running on the corrected corpus for headline-consistency.
2. **P6/P7 must not run with the unpatched `regrade_traces.py` semantics** (would re-introduce the ~12,284-label artifact — M1 HANDOFF 2); H1/H2 gate them.
3. **P8/P4b/P11 GPU arms assume the Blackwell box is re-leasable**; cost anchors are recorded cell runtimes, but current box availability was not verified this pass [UNVERIFIED — operational, not empirical].
4. **P3c's step-1-inclusion sub-arm crosses the T_MIN theory line** (step 1 is a forced commit, not a decision point) — it must be reported as an answer-selection extension, never silently folded into the stopping-policy headline.
