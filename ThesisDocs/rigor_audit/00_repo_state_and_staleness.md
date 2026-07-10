# Module 0: Repo State & Staleness Audit

**Auditor:** Sonnet 5 (mechanical fact-finding pass) · **Date:** 2026-07-09 · **HEAD at start:** `e049cc7`
**Scope:** `ThesisDocs/Scientific_Rigor_Research_Prompt.md` §0.2, §0.3, §0.5 and Section 2 "Module 0" mandate. This report is read-only fact-finding — no isolation verdicts (Module 1), no loss classification (Module 2).

**Working tree at time of audit:** exactly one modified file, `research/real_trace_experiments.py` (one hunk, `reconcile_existing_outputs`), plus untracked `.agents/` (not read, per instructions) and `ThesisDocs/Scientific_Rigor_Research_Prompt.md`. Verified via `git status` — see Task C.

---

## Rulings summary

| Question | Ruling |
|---|---|
| **Which tree is authoritative for the 74,540/7.57% headline?** | **`research/outputs/experiment_matrix/`** (the 52-cell matrix). Summing distinct `run_id`s in `detector_comparison_by_run.csv` across all 52 cells = **74,540**, exact match. `research/outputs/real_traces_bf16_ladder/` sums to 4,500 (3 Qwen models × GSM8K only) and is not close to 74,540. See Task A1. |
| **Does 74,540 reproduce?** | Yes, exactly, by direct recount (Task A1) **and independently** by running `scratch_analyze.py` fresh (Task E), which reproduces win=66,784 (89.5948%), tie=2,110 (2.8307%), loss=5,646 (7.5745%) against the authoritative tree — matching `July_1_Checkin.md` Part 6 to the exact run count. |
| **Item 1 — Qwen-7B boundary (step 5 vs 1)** | **CLOSED.** `stakes_sweep_report.md` (regenerated in `8ce9b9f`) shows `qwen2p5_7b, c=0.0 → T*=5`. |
| **Item 2 — stakes_sweep prose vs table** | **CLOSED.** Current "Key Insights" says boundary shifts *later* as `c` rises, matching the table (c=0→5, c=10→7, c=100→10 for qwen2p5_7b). |
| **Item 3 — DeepSeek-R1-Distill-7B boundary (step 2 vs 1)** | **CLOSED**, with a named cause: the "step 1" claim traces to a specific stale artifact, `research/outputs/cross_family/cross_family_summary.csv` (2026-06-12, pre-audit), whose `corrected_boundary_step` for `deepseek_r1_distill_7b` (and `deepseek_r1_distill_1p5b`, `phi_4_mini_instruct`) on GSM8K is literally `1.0`. Its report `research/CROSS_FAMILY_REPORT.md` carries a SUPERSEDED banner as of `9819ee2`. Every current artifact (cell `summary.md`, `_aggregate/gsm8k/CROSS_FAMILY_REPORT.md`, `prompted_vs_distilled_report.md`) shows step 2. `ThesisDocs/Startup_Research_Prompt.md:72` still tells a reader to "prefer step 1" — that guidance is itself now stale and should be corrected (flagged, not fixed, per remit). |
| **Item 4 — stragglers (generalize the boundary check)** | Checked ~90 boundary-bearing statements across 8 files/CSV families (Section B4 table). **1 confirmed violation of the t≥2 floor** in currently-live (non-superseded) data: none found — the only floor violations found are in the already-SUPERSEDED `research/outputs/cross_family/` tree and its linked report, which are correctly banner-marked. **1 contradiction worth flagging**: `Startup_Research_Prompt.md:72`'s "prefer step 1" instruction is itself wrong under current data. |
| **Uncommitted change (`research/real_trace_experiments.py`)** | Drops a `run_id_str in hidden_run_ids` (cached-hidden-state-file) gate from a completeness check. **No evidence it has ever executed** — the entire `research/outputs/` tree is byte-identical to git HEAD (`git status --porcelain research/outputs/` = empty). It is consequential prospectively: `hidden_states*/` subdirectories are absent from every sampled cell (6/6), so *re-running* the committed code today would silently zero out `hidden_run_ids` and mark **every** run incomplete on any resume. See Task C. |
| **Join feasibility (Module 2)** | **Feasible in all 51 populated cells**, 100% match rate of `detector_comparison_by_run.csv` `run_id`s into `trace_steps.csv` `run_id`s (checked exhaustively, all 52 cells, cheap). One cell (`mistral_small_3p1_24b_instruct__gsm8k`) has no `detector_comparison_by_run.csv` at all (failed collect, 0 runs). One cell (`qwen2p5_7b__math`) has a severe **completeness gap**: `trace_steps.csv`/`trace_runs.csv` show 1,500/1,500 complete runs, but `detector_comparison_by_run.csv` has only **75** — the detector-comparison stage was never re-run after the trace collection finished. Module 2's fresh recompute inherits this gap as-is (it's already inside the reproduced 74,540). |
| **Staleness of the two July 1 Scientific Method reports** | Confirmed stale by the "One Fact" claim: neither `Scientific_Method_Verification_Report.md` (`27a8b42`, 2026-07-01) nor `Scientific_Method_Deep_Dive_Report.md` (`d4fb095`, 2026-07-01) was touched after `8ce9b9f`/`074bc70`, and neither carries a SUPERSEDED banner (only `research/ALPHA_BETA_PREDICTION_REPORT.md`, `research/CROSS_FAMILY_REPORT.md`, `research/overthinking_boundary.md` got banners in `9819ee2`). **However**, spot-checking their boundary-bearing tables shows most cited *numbers* happen to already be correct (Qwen capability-ladder table, Qwen-32B stakes table, DeepSeek T*=2) — the staleness is a **provenance/reproducibility** problem (values not regenerated from current audited code), not, on the specific cross-checks performed here, a **numeric-accuracy** problem. Module 1 should still make the final supersession call; this is exposure facts only. |

---

## Task A — Output-tree authority ruling

### A1. Per-cell recount, both trees, vs 74,540

Method: one script (`task_a1.py`, reproduced below in essence) read every `{cell}/detector_comparison_by_run.csv` under both `research/outputs/experiment_matrix/` and `research/outputs/real_traces_bf16_ladder/`, and for each cell recorded row count, distinct `run_id` count, and the per-detector row breakdown (`value_counts()` on the `detector` column).

```python
import pandas as pd, os
def scan_tree(root):
    cells = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    total_rows = total_distinct = 0
    for cell in cells:
        p = os.path.join(root, cell, "detector_comparison_by_run.csv")
        if not os.path.exists(p): continue
        df = pd.read_csv(p)
        total_rows += len(df)
        total_distinct += df["run_id"].nunique()
    return total_rows, total_distinct
```

**`research/outputs/experiment_matrix/` (52 model×dataset cell dirs + `_aggregate/` + 1 failed cell dir):**
- Total rows across all cells' `detector_comparison_by_run.csv`: **670,860**
- Sum of per-cell distinct `run_id` counts: **74,540** — exact match to `July_1_Checkin.md` Part 6's "Out of 74,540 Completed Runs."
- Every populated cell has exactly **9 detectors**: `oracle, first_answer, verifier_first_correct, answer_stability, entropy_plateau, hazard_drift, empirical_bernstein, e_process, never_stop`, each contributing exactly `distinct_run_ids` rows (9 × distinct = total rows per cell, confirmed for all 51 populated cells).
- `mistral_small_3p1_24b_instruct__gsm8k` has **no** `detector_comparison_by_run.csv` (directory exists, 0 files — see A4/anomaly note).

**`research/outputs/real_traces_bf16_ladder/` (`qwen2p5_7b`, `qwen2p5_14b`, `qwen2p5_32b` only):**
- Total rows: 40,500. Sum of distinct `run_id`s: **4,500** (3 × 1,500, GSM8K only, 9 detectors each). Not close to 74,540.

**Ruling: `research/outputs/experiment_matrix/` is the authoritative tree for the 74,540/7.57% headline.** This is not just "the bigger tree" — the total is an *exact* digit-for-digit match, which given 52 independently-varying cell sizes (1,342–1,500 runs each) is not plausible by coincidence.

### Per-cell mismatches vs `July_1_Checkin.md` Part 3

Part 3's table (`ThesisDocs/July_1_Checkin.md:77-92`) reports **collected** run counts (matches `matrix_manifest.json`'s `n_runs` per cell, the "collect" stage), totaling **75,996**. Part 6's 74,540 is the **completed** count from `detector_comparison_by_run.csv` (post detector-comparison stage). These are two different pipeline stages, not two inconsistent claims about the same thing, and the gap between them (75,996 − 74,540 = **1,456**) is fully and exactly accounted for by six cells where `detector_comparison_by_run.csv` has fewer distinct `run_id`s than the manifest's collected count:

| Cell | Manifest `n_runs` (Part 3 / collect stage) | `detector_comparison_by_run.csv` distinct `run_id` | Gap |
|---|---:|---:|---:|
| `llama_3p1_8b_instruct__gpqa` | 1,348 | 1,342 | −6 |
| `qwen2p5_0p5b__gpqa` | 1,354 | 1,342 | −12 |
| `qwen2p5_3b__gpqa` | 1,349 | 1,343 | −6 |
| `yi_1p5_9b_chat__gpqa` | 1,349 | 1,343 | −6 |
| `yi_1p5_9b_chat__math` | 1,500 | 1,499 | −1 |
| `qwen2p5_7b__math` | 1,500 | **75** | **−1,425** |
| **Sum of gaps** | | | **−1,456** (exact match to 75,996 − 74,540) |

All other 45 populated cells match Part 3 exactly. **No cell in `experiment_matrix` shows a distinct-`run_id` count *above* its Part 3 figure** — i.e., the four GPQA cells flagged in the brief as "above the clean 1,344" (Task F, item 2 below) are above at the **manifest/collect** stage but *below* even 1,344 at the **detector-comparison** stage. Full numbers under Task F.

### A2. Per-cell data-availability matrix (Module 2 join feasibility)

Checked **exhaustively, all 52 cell directories** (cheap — max cell CSV is ~17MB, total scan under 2 minutes) via a second script that, per cell, loaded `detector_comparison_by_run.csv` and `trace_steps.csv`, computed `set(dcbr.run_id) & set(trace_steps.run_id)`, and reported the match count/rate.

**Result: every populated cell has 100% join match rate** — every `run_id` present in `detector_comparison_by_run.csv` is also present in `trace_steps.csv`. The join is never *broken* (no orphaned rows). It is, however, **lossy** in the same six cells as above: `trace_steps.csv` (and `trace_runs.csv`) contain *more* distinct `run_id`s than `detector_comparison_by_run.csv` does, meaning those extra trace-complete runs have no win/loss/tie verdict at all and are invisible to any recompute keyed off `detector_comparison_by_run.csv`.

The standout is **`qwen2p5_7b__math`**: `trace_runs.csv` = 1,500/1,500 distinct run_ids, `trace_steps.csv` = 1,500 distinct run_ids (21,000 rows = 14 steps × 1,500), but `detector_comparison_by_run.csv` = only **75** distinct run_ids (675 rows = 9 detectors × 75). This cell's `metadata.json` shows the mechanism directly: the top-level field `"completed_run_count": 1500` disagrees with the nested `"checkpoint_reconciliation": {"completed_run_count": 75, "hidden_state_file_count": 75, ...}` **inside the same file**. File mtimes corroborate a two-stage history: `answers.md`/`detector_comparison_by_run.csv`/`final_results.md`/etc. are dated 2026-06-20 11:39, while `trace_runs.csv`/`trace_steps.csv`/`metadata.json`/`batch_metrics.csv`/`hazard_by_step.csv`/`pilot_summary.csv` are dated 2026-06-20 23:03 — i.e., a later collection/reconcile pass grew the cell from 75 to 1,500 complete runs, but `research/trace_analysis.py` (the script that writes `detector_comparison_by_run.csv`, confirmed via its docstring/manifest command string) was **never re-run** for this cell afterward. **Flag for Module 2: this cell's fresh win/loss/tie recompute is silently based on 75/1,500 (5%) of its actual trace data, not disclosed anywhere in current reports.** (`research/outputs/experiment_matrix/qwen2p5_7b__math/final_results.md`/`summary.md` still narrate the 75-run-era numbers; not checked line-by-line here — Module 1/2's remit.)

The four GPQA cells (`llama_3p1_8b_instruct`, `qwen2p5_0p5b`, `qwen2p5_3b`, `yi_1p5_9b_chat`) show a three-way split worth recording precisely (raw counts only — dedup analysis is explicitly Module 1's, not mine, per the brief):

| Cell | `trace_runs.csv` distinct | `trace_steps.csv` distinct | `detector_comparison_by_run.csv` distinct |
|---|---:|---:|---:|
| `llama_3p1_8b_instruct__gpqa` | 1,344 | 1,348 | 1,342 |
| `qwen2p5_0p5b__gpqa` | 1,344 | 1,354 | 1,342 |
| `qwen2p5_3b__gpqa` | 1,344 | 1,349 | 1,343 |
| `yi_1p5_9b_chat__gpqa` | 1,344 | 1,349 | 1,343 |

`trace_runs.csv` is deduplicated to the clean 1,344 in all four; `trace_steps.csv` is inflated (matches the manifest); `detector_comparison_by_run.csv` is *below* 1,344. All `detector_comparison_by_run.csv` `run_id`s were still found with 100% match rate inside the (larger) `trace_steps.csv`, so the join itself is not broken, just built on a smaller-than-1,344 base for these four cells specifically.

`mistral_small_3p1_24b_instruct__gsm8k`: directory exists (dated 2026-06-18) but is **completely empty** (0 files) — `matrix_manifest.json`'s `cells["mistral_small_3p1_24b_instruct__gsm8k"]` shows `"collect": {"status": "failed", "return_code": 1, "elapsed_min": 0.13}, "n_runs": 0`. This is the fast-fail referenced in the pre-verified facts. `mistral_small_24b_2409` (a different model — Mistral Small **2409**, not **3.1**) has full, real data in all 4 datasets (`arc/gpqa/gsm8k/math`, all `ok`, 1,500/1,500/1,500/1,344 in the manifest) and is the model actually counted in the 13-model roster and in the 74,540 total. **`mistral_small_3p1_24b_instruct` was substituted by `mistral_small_24b_2409` after the former's collect-stage failure; this substitution is documented only in `matrix_manifest.json`'s per-cell status field — no prose report or `metadata.json` narrates it.** (`grep`: `mistral_small_3p1_24b` appears only in `matrix_manifest.json`, `research/reports/phase2_final_report.md`, and `ThesisDocs/Scientific_Rigor_Research_Prompt.md` — no ThesisDocs narrative doc mentions the substitution.)

### A3. Schema of `trace_steps.csv` / `trace_runs.csv`

Checked `qwen2p5_7b__gsm8k`, `qwen2p5_7b__gpqa` (both `experiment_matrix`), `deepseek_r1_distill_7b__gsm8k` (`experiment_matrix`), and `qwen2p5_7b` (ladder tree, GSM8K). **No schema drift** — identical column lists in all four.

`trace_steps.csv` (46 columns): `run_id, model_alias, model_name, task_id, domain, difficulty, task_source, task_source_index, expected_answer, step, thought, answer, answer_normalized, correct, confidence, model_stop_flag, answer_changed, thought_token_count, raw_generation_tokens, mean_token_logprob, entropy_mean, entropy_std, hidden_norm, hidden_l2_shift, hidden_cosine_shift, lexical_echo, verbose_confidence_proxy, utility, elapsed_seconds, tokens_per_second, gpu_memory_allocated_gb, seed, temperature, device, prompt_mode, system_prompt_mode, is_baseline, parse_success, output_format_type, answer_extraction_source, stop_extraction_source, confidence_extraction_source, hit_max_new_tokens, truncated_output_suspected, raw_text_length_chars, raw_text_length_tokens, raw_text`.

- **Step index:** `step`.
- **Extracted answer:** `answer` (raw) / `answer_normalized` (normalized for grading).
- **Self-reported confidence:** `confidence`.
- **Entropy:** `entropy_mean` / `entropy_std` (per-step average over generated tokens).
- **There is no per-step `q_t` column in `trace_steps.csv`** — `q_t` (the fitted correctness probability) is a *population-level*, per-step-index aggregate, materialized separately in `hazard_by_step.csv` (columns: `step, q_t, n_repairs, n_corruptions, repair_rate, corruption_rate, hazard_mu, entropy_mean, confidence_mean, answer_changed_rate, hidden_shift_mean, n_transitions`), not a per-run per-step field. Module 2 needing per-run correctness should use `trace_steps.csv`'s `correct` (binary, ground truth per step) rather than expect a per-run `q_t`.

`trace_runs.csv` (21 columns, one row per run): `run_id, model_alias, model_name, task_id, domain, difficulty, task_source, task_source_index, temperature, seed, prompt_mode, system_prompt_mode, is_baseline, ever_correct, correct_at_step_1, oracle_stop, first_correct_step, first_model_stop_step, revision_count, best_utility, final_correct, device`.

### A4. Which script writes which tree; where the two trees' numbers currently circulate

`grep`-verified output-dir constants (`research/*.py`):

| Script | Writes to |
|---|---|
| `research/run_experiment_matrix.py` | `research/outputs/experiment_matrix` (`DEFAULT_OUTPUT_ROOT`) — the 52-cell matrix launcher, invokes `real_trace_experiments.py` per cell (commands visible in `matrix_manifest.json`). **Authoritative tree.** |
| `research/run_qwen_bf16_ladder.py` | `research/outputs/real_traces_bf16_ladder` (`--output-root` default) — 3-model (Qwen 7B/14B/32B), GSM8K-only capability-ladder sweep. Confirmed via each cell's `metadata.json`: `task_source: "gsm8k"` for all three, `quantization: "none"`. |
| `research/run_stakes_sweep.py`, `research/run_active_stopping.py`, `research/compare_prompted_vs_distilled.py`, `research/extract_scientific_stats.py` | Ad-hoc scripts. All four **read** from the ladder tree's `qwen2p5_7b` cell (`extract_scientific_stats.py`, `run_active_stopping.py`) or from `experiment_matrix` directly (`compare_prompted_vs_distilled.py`'s `MATRIX_ROOT`, used for both its Qwen and DeepSeek data) and **write** their three md reports (`active_stopping_simulation_report.md`, `stakes_sweep_report.md`, `prompted_vs_distilled_report.md`) plus 2 PNGs into the **ladder tree's root** (not a per-cell dir) — confirming the brief's note that these ad-hoc reports live in the ladder tree even though `compare_prompted_vs_distilled.py` actually sources from the authoritative tree for its DeepSeek half. |
| `research/analyze_runs.py`, `scratch_analyze.py` | Read from `experiment_matrix` only. |
| `research/regrade_traces.py` | Generic `--root`-parameterized; its docstring example and the actual `074bc70` dry run both point at `experiment_matrix`. |

**Numbers in circulation, by tree:**
- `ThesisDocs/July_1_Checkin.md` — no literal `bf16_ladder`/`experiment_matrix` path strings in the prose, but its headline (74,540, per-dataset/per-temperature win rates) is **fully reproducible from `experiment_matrix`** — see Task A1/Task E.
- `ThesisDocs/cross_benchmark_results.md` — explicitly cites `research/outputs/experiment_matrix/_aggregate/{gsm8k,math,arc,gpqa}/CROSS_FAMILY_REPORT.md` (line 146). **Authoritative-tree sourced.**
- `ThesisDocs/Scientific_Method_Verification_Report.md` and `ThesisDocs/Scientific_Method_Deep_Dive_Report.md` — cite `research/outputs/real_traces_bf16_ladder/qwen2p5_7b/trace_steps.csv` directly for their two named concrete failure-mode trace examples (lines 68/72 and 146/150 respectively). **These two reports are the only current ThesisDocs files that cite the ladder tree for anything beyond the (correctly-scoped) capability-ladder/stakes/prompted-vs-distilled ad-hoc studies**, and they cite it for GSM8K-only, Qwen-7B-only trace examples used to illustrate the (allegedly general) failure taxonomy — a scope mismatch Module 1 should weigh in on.
- A third, older tree, **`research/outputs/cross_family/`** (2026-06-12, top-level, distinct from both named trees and from the four `_aggregate/{dataset}/cross_family/` dirs), is written by `research/cross_family_analysis.py` and reported in `research/CROSS_FAMILY_REPORT.md` — one of the three files banner-marked SUPERSEDED in `9819ee2`. This tree is not one of the two the brief names, but it is directly relevant to Item 3 (see Rulings summary and B3 below) and is correctly flagged stale via its report's banner, even though the raw CSVs underneath carry no banner of their own (see Task F).

---

## Task B — Reconciliation items

### B1/B2. Qwen-7B boundary + stakes_sweep prose — CLOSED

`research/outputs/real_traces_bf16_ladder/stakes_sweep_report.md` (current file on disk, read directly):

```
| qwen2p5_7b | 0.0 | 5 | 0.6498 | 0.4153 | 0.2053 | 0.2344 | 0.4444 |
...
| qwen2p5_7b | 10.0 | 7 | -1.7369 | -2.6793 | -2.7413 | 0.9424 | 1.0044 |
...
| qwen2p5_7b | 100.0 | 10 | -23.2169 | -29.2613 | -29.2613 | 6.0444 | 6.0444 |
```
`qwen2p5_7b, c=0.0 → T*=5`. `c=10.0 → T*=7`. `c=100.0 → T*=10`. Matches the brief's expected values exactly.

"Key Insights" (same file): *"**Boundary Shifts Later:** As error penalty (c) scales, the boundary step shifts to the right (later), not earlier... the model is incentivized to keep reasoning longer to avoid an even costlier wrong answer."* — direction matches the table.

**Regeneration provenance:** `git log --oneline -- research/outputs/real_traces_bf16_ladder/stakes_sweep_report.md` → `8ce9b9f` (post-fix), `5955600` (original). `git show 8ce9b9f -- .../stakes_sweep_report.md` confirms the diff: `qwen2p5_7b, c=0.0` row changed from `T*=1` to `T*=5`, and `c=0.5` from `T*=1` to `T*=5`; every other row (including all `qwen2p5_32b` rows) is **unchanged by that commit**, because the unfloored bug only manifested for `qwen2p5_7b` at `c ∈ {0, 0.5}` — the other cells' raw drift already crossed at `t ≥ 2` even without the floor. The "Key Insights" #1 bullet text was also rewritten in the same diff (was: *"Boundary Shifts Earlier... from step 5 to step 2 or 3"* — itself internally contradictory with the pre-fix table, which is exactly reconciliation item 2). **Both items CLOSED**, confirmed from the file as it exists now, not from prior-session narration.

### B3. DeepSeek-R1-Distill-7B boundary — CLOSED, named cause

**(a) `research/outputs/real_traces_bf16_ladder/prompted_vs_distilled_report.md` (current):** `"Corrected Stopping Boundary (T*) | Step 5 | Step 2"` for Qwen2.5-7B / DeepSeek-R1-Distill-7B respectively. `compare_prompted_vs_distilled.py` (the generator) sources DeepSeek data from `MATRIX_ROOT / "deepseek_r1_distill_7b__gsm8k"` — the **authoritative** tree, not the ladder tree (the ladder tree has no DeepSeek cell at all) — and imports `T_MIN, first_zero_crossing` from `trace_analysis.py` rather than reimplementing boundary detection locally (comment at line 26-29: *"T_MIN-floored first_zero_crossing (commit ef45dda) instead of a local unfloored ... scan, which silently reintroduced ..."*). Regenerated in `8ce9b9f`; commit message states *"DeepSeek T*=2 (was 1)"*.

**(b) Current authoritative-tree cell data, independently:**
- `research/outputs/experiment_matrix/deepseek_r1_distill_7b__gsm8k/summary.md` and `final_results.md` (both last touched by `6f832ea`, 2026-06-20, "fix: correct canonical boundary operator + regenerate all results out-of-sample"): *"The corrected conditional hazard drift crosses zero at step 2, while the raw empirical utility drift crosses at step 2..."*
- `research/outputs/experiment_matrix/_aggregate/gsm8k/CROSS_FAMILY_REPORT.md` "Run Summary" table: DeepSeek 7B `Corrected boundary = 2`. "Drift Audit" table: DeepSeek 7B `Empirical=2, Corrected=2, Fitted=4, Legacy pooled proxy=2` — **all four boundary variants satisfy the t≥2 floor**, for every one of the 13 models in this file (checked programmatically, zero violations).
- `research/trace_analysis.py` — the canonical pipeline script — hard-codes `T_MIN = 2` (line 20: *"earliest admissible stop; step 1 is the forced-commit init, not a decision point"*) and applies it inside `oracle_stop`, `first_zero_crossing`, and the hazard-training/-stopping filters (lines 107, 208-211, 321, 444, 479-480, 672-674). This floor predates `8ce9b9f` — it was introduced by the 2026-06-20 remediation (`ef45dda`/`6f832ea`), i.e. it was already the canonical behavior when the DeepSeek-7B cell's `summary.md` was last written.

**(c) Named cause of the "step 1" claim:** traced to `research/outputs/cross_family/cross_family_summary.csv` (2026-06-12, mtime and `git log -1` both confirm; **predates** the `ef45dda`/`6f832ea` remediation entirely). Direct read:

```
model_alias                task_source  corrected_boundary_step  empirical_boundary_step  pooled_proxy_boundary_step  fitted_boundary_step
qwen2p5_0p5b                gsm8k        1.0                      1.0                      1.0                        4.0
deepseek_r1_distill_1p5b    gsm8k        1.0                      1.0                      7.0                        1.0
qwen2p5_7b                  gsm8k        6.0                      6.0                      5.0                        7.0
mistral_7b_instruct_v0p3    gsm8k        3.0                      3.0                      3.0                        5.0
deepseek_r1_distill_7b      gsm8k        1.0                      1.0                      1.0                        1.0
phi_4_mini_instruct         gsm8k        1.0                      1.0                      4.0                        4.0
```

`deepseek_r1_distill_1p5b`, `deepseek_r1_distill_7b`, and `phi_4_mini_instruct` all show `corrected_boundary_step = 1.0` — this is **exactly** the set the prior-session note describes ("Reasoning-distill (DeepSeek-1.5B/7B) and Phi-4-mini stay early (step 1)"), and it is a pre-remediation, unfloored artifact, not a second unfloored code path in current code. Its report, `research/CROSS_FAMILY_REPORT.md`, carries the SUPERSEDED banner added in `9819ee2` (*"This file predates the deep_code_audit.md findings and the ef45dda ... remediation commit — its boundary and AUC numbers were computed under the pre-audit pipeline (unfixed graders, in-sample leakage, unfloored boundary)"*).

**Ruling: CLOSED. Named cause = stale pre-remediation artifact** (`research/outputs/cross_family/cross_family_summary.csv` + `research/CROSS_FAMILY_REPORT.md`, both dated 2026-06-12, both predating `ef45dda`), **already correctly marked SUPERSEDED at the report level**. No genuine second unfloored code path was found in current code — `trace_analysis.py`'s `T_MIN=2` floor is applied uniformly, and `git log -S "run_id_str in hidden_run_ids"` / manual review found no other boundary-detection reimplementation outside the three files `8ce9b9f` already fixed. **One residual defect: `ThesisDocs/Startup_Research_Prompt.md:72` still instructs a reader to "Prefer step 1 unless you find a source that justifies step 2"** — this is now backwards relative to every current artifact and should be corrected by a human (flagged, not edited, per this module's remit — `ThesisDocs/Startup_Research_Prompt.md` is explicitly out of scope for editing here).

### B4. Generalizing — every file stating a boundary value

Files matching `T\*|T_star|boundary|step 1|stops at step|crossing` under `ThesisDocs/`, `research/reports/`, and boundary columns in every `cross_family_summary.csv`: `ThesisDocs/July_1_Checkin.md`, `Scientific_Method_Deep_Dive_Report.md`, `Scientific_Method_Verification_Report.md`, `Startup_Research_Prompt.md`, `acm_thesis_proposal_draft.md`, `cross_benchmark_results.md`, `dual_boundary_appendix.md`, `literature_review_novelty.md`, `mathematical_foundations_proofs.md`, `project_plan_advisor_meeting.md`, `thesis_stopping_rule_algorithm.md`, plus several `research/reports/*.md` (older 2026-04 memos, not model-boundary-bearing on inspection — theoretical/CPU-status content). Model-boundary-bearing files, cross-checked directly:

| File | Model(s)/scope | Stated boundary | t≥2 floor? | Current (post-`8ce9b9f`/`6f832ea`) code? | Contradicted by a sibling? |
|---|---|---|---|---|---|
| `cross_benchmark_results.md` (§5 table, `3a5dde7`, 2026-06-30) | Qwen 0.5B/3B/7B/14B/32B, GSM8K | 2(floor)/4/5/5/5 | Yes | Yes — sourced from `experiment_matrix/_aggregate`, post-`6f832ea` | No — matches `_aggregate/gsm8k/cross_family_summary.csv` exactly |
| `_aggregate/gsm8k/CROSS_FAMILY_REPORT.md` (`f89ff38`-era) | All 13 models, GSM8K | see B3(b) table | Yes, all 13 | Yes | No |
| `_aggregate/{arc,gpqa,math}/cross_family/cross_family_summary.csv` | All 13 models × 3 datasets | — (programmatic check) | Yes, zero violations found (39 rows checked) | Yes | No |
| `Scientific_Method_Verification_Report.md:31` | Qwen 7B/14B/32B, GSM8K | Step 5 / Step 5 / Step 5 | Yes | **No — predates `8ce9b9f`/`074bc70`, not regenerated** | No — happens to match current `_aggregate/gsm8k` data (see Task D exposure map) |
| `Scientific_Method_Deep_Dive_Report.md:113-118` (Factor 5) | Qwen-32B stakes sweep | c=0→5, c=1→6, c=10→8, c=100→10 | Yes | **No — predates fix**, but the underlying `qwen2p5_32b` rows in `stakes_sweep_report.md` were *unaffected* by the `8ce9b9f` diff (see B1/B2) — values match current data exactly | No |
| `Scientific_Method_Deep_Dive_Report.md:132` (Factor 6) | Qwen-7B / DeepSeek-7B | Step 5 / "Step 2 (Floor)" | Yes | **No — predates fix**, and the *underlying* `prompted_vs_distilled_report.md` at that time (commit `5955600`, verified via `git show`) actually read **Step 1 / Step 1** for both models — the Deep Dive report's author hand-entered "Step 5" and "Step 2 (Floor)" into the narrative rather than transcribing the (buggy) source table. **Numerically correct today, but not reproducible from its own cited source at time of writing** — a provenance defect distinct from a numeric one. | No (values match current) but **yes relative to its own un-regenerated source file at authoring time** |
| `Startup_Research_Prompt.md:61-64,72` | DeepSeek-7B | §0.3 items 1-3 narrated as open questions (superseded by this report); **line 72 explicitly says "Prefer step 1"** | N/A (a meta-instruction, not a data point) | N/A | **Yes — contradicts every current artifact (B3 above)** |
| `research/CROSS_FAMILY_REPORT.md` / `research/outputs/cross_family/cross_family_summary.csv` | 6 models, GSM8K (2026-06-12) | 3 of 6 models show `corrected_boundary_step=1.0` (violates floor) | **No — 3 violations** | No — pre-`ef45dda` | Report carries SUPERSEDED banner (`9819ee2`); **raw CSV does not** |
| `research/ALPHA_BETA_PREDICTION_REPORT.md`, `research/overthinking_boundary.md` | (not deep-read; out of scope beyond banner check per Task D) | — | Not verified | No | SUPERSEDED banner present (`9819ee2`) |

**Straggler count:** ~90 individual boundary-value cells checked across the tables above (13-model × 4-dataset `_aggregate` CSVs = 52 rows × 4 boundary-variant columns = 208 individual values scanned programmatically, plus ~15 prose-stated values). **Violations of the t≥2 floor found: 3** (`deepseek_r1_distill_1p5b`, `deepseek_r1_distill_7b`, `phi_4_mini_instruct`, all in the single pre-remediation, already-SUPERSEDED `research/outputs/cross_family/cross_family_summary.csv`). **Contradictions between a live (non-superseded) document and current data: 1** (`Startup_Research_Prompt.md:72`'s "prefer step 1" instruction). **Provenance-without-numeric-error findings: 1** (`Scientific_Method_Deep_Dive_Report.md` Factor 6, hand-patched against a source it doesn't actually reproduce).

---

## Task C — Uncommitted-change characterization (flag, not judge)

**The diff** (`git diff research/real_trace_experiments.py`), inside `reconcile_existing_outputs` (function spans `research/real_trace_experiments.py:1251-1363`):

```diff
-            is_complete = observed_steps == expected_step_sequence and run_id_str in hidden_run_ids
+            is_complete = observed_steps == expected_step_sequence
```

**What `hidden_run_ids` is:** `hidden_run_ids = {path.stem for path in hidden_dir.glob("*.npz")} if hidden_dir.exists() else set()` (line 1272). `hidden_dir` is passed in by the caller as `output_dir / ("hidden_states_baseline" if args.run_baseline else "hidden_states")` (line 2358) — a per-cell subdirectory of cached model hidden-state arrays (one `.npz` per run, used for the correctness probe / hidden-state feature extraction). So `hidden_run_ids` is the set of run_ids that currently have an on-disk hidden-state cache file.

**What the committed (HEAD) code requires:** a run counts as complete during reconciliation only if (a) its rows in `trace_steps.csv` form the exact expected step sequence `[1..expected_steps]`, **and** (b) it also has a matching `.npz` file in `hidden_states(_baseline)/`. Condition (b) gates `completed_from_steps`, which in turn gates `sanitized_steps`/`sanitized_runs` (the CSVs actually written back to disk) and the summary counters `completed_run_count`/`hidden_state_file_count` in `metadata.json`/`checkpoint_reconciliation.json`.

**What the edit changes:** drops condition (b). **The class of run newly counted complete** = any run with a fully-stepped `trace_steps.csv` sequence but **no** corresponding hidden-state `.npz` file — e.g. runs whose `hidden_states/` cache was cleaned up after initial collection (to save disk), or where hidden-state saving was skipped/failed independently of step generation.

**Could this change any published number if the reconcile path were re-run?** Yes, plausibly and materially, under a specific condition that I confirmed holds today: I sampled 6 cells' directory listings (`qwen2p5_7b` ladder cell, `qwen2p5_7b__gsm8k`, `deepseek_r1_distill_7b__gsm8k`, `qwen2p5_32b__math`, `yi_1p5_9b_chat__gpqa`, `phi_4_mini_instruct__arc`) and **none currently have a `hidden_states/` or `hidden_states_baseline/` subdirectory on disk**. Under the **committed** code, if `reconcile_existing_outputs` were invoked again for any of these cells (e.g. on a resume), `hidden_dir.exists()` would be `False`, so `hidden_run_ids` would be the empty set, so `run_id_str in hidden_run_ids` would be `False` for *every* run, so `is_complete` would be `False` for *every already-complete run* — a full-cell false-negative that would drop all rows from `sanitized_steps`/`sanitized_runs` and could trigger unnecessary re-collection. Under the **edited** code, the same resume would correctly recognize all 1,500 (or however many) runs as complete from step data alone. **This is a real, currently-live divergence in behavior, not a hypothetical one** — it depends only on whether a cell's hidden-state cache directory still exists, which is empirically false everywhere sampled.

**Is there evidence anything currently on disk was already produced under the edited semantics?** **No.** `git status --porcelain research/outputs/` returns **empty** — the entire `research/outputs/` tree, including every `metadata.json` and `checkpoint_reconciliation.json`, is byte-identical to git HEAD. I specifically checked the three ladder-tree qwen cells' `metadata.json` (their filesystem mtimes, 2026-07-01 17:19–17:29, are minutes *after* the working file's edit mtime of 2026-07-01 17:16:52, which looked suspicious at first) — `git diff --quiet HEAD` on each confirms **zero content difference from HEAD** (last real content change: commit `554b43b`, long before this edit). The newer mtimes are a filesystem touch unrelated to content (most likely a git checkout/branch operation), **not** evidence of a re-run. The one standalone `checkpoint_reconciliation.json` found anywhere in the repo (`research/outputs/experiment_matrix/yi_1p5_9b_chat__math/checkpoint_reconciliation.json`) is dated 2026-06-27 (commit `b41705c`, well before the edit) and shows `hidden_state_file_count: 400` matching `completed_run_count: 400` — i.e., under **that** historical run, `hidden_states/` did exist and did gate correctly; it is unrelated to the current edit. **Conclusion: the uncommitted edit has not yet executed against real data; every number currently on disk was produced under the committed (HEAD) semantics.** Flagged for explicit human review per the brief's instruction — not characterized as bug or fix.

---

## Task D — Report staleness map

**SUPERSEDED-banner commit (`9819ee2`, "docs: add SUPERSEDED banners to pre-audit reports"):** exactly 3 files got banners: `research/ALPHA_BETA_PREDICTION_REPORT.md`, `research/CROSS_FAMILY_REPORT.md`, `research/overthinking_boundary.md`. All three point to `research/reports/deep_code_audit.md` and `ThesisDocs/cross_benchmark_results.md` as authoritative. Repo-wide `grep -rl SUPERSEDED` (excluding `.agents/`) additionally finds the banner text quoted inside `Scientific_Rigor_Research_Prompt.md` and `Startup_Research_Prompt.md` (both discuss the banners, don't carry them) — **no other file carries the banner itself.**

**Last-commit dates for every `ThesisDocs/*.md`:**

| File | Last commit | Post-dates `8ce9b9f`/`074bc70` (2026-07-02)? |
|---|---|---|
| `July_1_Checkin.md` | `8ce9b9f` (2026-07-02) | Touched *by* `8ce9b9f` itself (one cosmetic row fix, see below); predates `074bc70` by ~1 second in the same session |
| `Scientific_Method_Deep_Dive_Report.md` | `d4fb095` (2026-07-01) | **No** |
| `Scientific_Method_Verification_Report.md` | `27a8b42` (2026-07-01) | **No** |
| `Startup_Research_Prompt.md` | `e049cc7` (2026-07-02) | Yes, but as a docs/planning doc, not a data report |
| `cross_benchmark_results.md` | `3a5dde7` (2026-06-30) | No, but sources from the already-`6f832ea`-fixed canonical pipeline (not the ad-hoc scripts `8ce9b9f` touched), so not practically exposed |
| `literature_review_novelty.md`, `mathematical_foundations_proofs.md` | `3a5dde7` (2026-06-30) | No — theory-only content, not data-exposed |
| `project_plan_advisor_meeting.md` | `daa98bc` (2026-06-04) | No — explicitly noted in the brief as an outdated model-roster plan |
| `acm_thesis_proposal_draft.md`, `dual_boundary_appendix.md`, `acm_625803_804_email_drafts.md` | `2a883cc` (2026-04-03) | No — early proposal-stage docs |
| `thesis_stopping_rule_algorithm.md` | `33482ea` (2026-04-02) | No — algorithm spec, not results |

**Per-cell reports (52-cell tree), sampled directly rather than exhaustively (52 × 4 report types = 208 files; sampled 3):** `deepseek_r1_distill_7b__gsm8k/summary.md` and `final_results.md` (`6f832ea`, post-remediation, floor-consistent — confirmed above); `qwen2p5_7b__math/final_results.md`/`summary.md` (`2026-06-20 11:39` mtime, i.e. from the 75-run-era snapshot of that cell — **narrates the incomplete 75-run state, not the eventual 1,500-run trace data**; not read line-by-line, flagged for Module 1/2). Given the mtime pattern (`_aggregate` regenerated `2026-06-27`/`2026-06-28`, matching `f89ff38`'s "full 52-cell 4-dataset sweep completion"), per-cell reports as a class are treated as one row per dataset here per the brief's allowance; a full 208-file per-cell audit was not performed (time-boxed).

**`ThesisDocs/July_1_Checkin.md` exposure to post-dating fixes:** `git show 8ce9b9f -- ThesisDocs/July_1_Checkin.md` shows exactly **one** line changed — the "First Answer" row's Mean Stop Step, `2.00 → 1.00` (Part 5 table). This is a narrow, disclosed correction (the commit message calls it out by name: *"July_1_Checkin.md's First Answer row"*), **not** a change to the 74,540/89.59%/2.83%/7.57% headline figures. `July_1_Checkin.md` was **not** touched by `074bc70` (the regrade NaN-handling fix). Since `074bc70` ran `regrade_traces.py` in **dry-run mode only** (commit message: *"running the tool for the first time (50/52 cells, dry-run)"*) and no `trace_steps.csv` anywhere postdates that commit (`0` files found by directly checking mtimes), **the `correct` labels underlying every number in `July_1_Checkin.md` are unchanged from before `074bc70`** — but `074bc70`'s own commit message discloses that a *real* (non-dry) regrade would flip **~1.62% of ~799,000 rows** (vs. a previously-circulating false claim of "0.33% of 165,000"). **This is a known, quantified, not-yet-applied correction that would touch every downstream `correct`-derived number, including the 7.57% loss figure, if actually applied** — flagged as a blocker below, not resolved (applying it is out of this module's remit and would be a new pipeline run).

**Exposure map, the two Scientific Method reports:**

| Report | Table/claim | Exposed to `8ce9b9f` (T_MIN floor, ad-hoc scripts)? | Exposed to `074bc70` (regrade NaN)? | Source artifact regenerated after either fix? |
|---|---|---|---|---|
| `Scientific_Method_Verification_Report.md:31` | Qwen 7/14/32B capability-ladder boundary table | No numeric exposure found (values match current `_aggregate/gsm8k` data, itself post-`6f832ea` not post-`8ce9b9f`) | Not checked directly (no regrade applied to anything yet) | No — report file itself never regenerated |
| `Scientific_Method_Verification_Report.md:67-72` | Two named GSM8K "arithmetic slip / late correction" trace examples, cited via `file:///.../real_traces_bf16_ladder/qwen2p5_7b/trace_steps.csv` | The cited trace file's `correct`/`utility` columns were not touched by `8ce9b9f` (that fix only touched two `.md` report files + `.py` scripts + one `.png`, confirmed via `git show 8ce9b9f --stat`) | **Yes, in principle** — `trace_steps.csv`'s `correct` labels are exactly what `regrade_traces.py` would rewrite, and this file has not been regraded (real run) | No — `qwen2p5_7b`'s `trace_steps.csv` mtime predates the audit window entirely (part of the original `554b43b`-era ladder collection) |
| `Scientific_Method_Deep_Dive_Report.md` Factor 5 (stakes, Qwen-32B) | Numerically matches current `stakes_sweep_report.md` exactly (Task B4) | **No practical exposure** — the `8ce9b9f` diff did not touch any `qwen2p5_32b` row | Same trace-label exposure as above, in principle | No |
| `Scientific_Method_Deep_Dive_Report.md` Factor 6 (prompted vs. distilled) | Numerically matches current `prompted_vs_distilled_report.md` (Step 5 / Step 2) | **Provenance-exposed but not numerically wrong** — see B4 table: the report's own source at authoring time read Step 1/Step 1 | Same trace-label exposure, in principle | No |
| Both reports' "3 named failure modes for the 7.57%" narrative | Not independently re-derived by this module (Module 1's remit) — flagged only | N/A | The 7.57% itself is confirmed reproducible from current, unregraded data (Task A1/E) | N/A |

**Bottom line for Module 1's supersession call:** neither report has been regenerated post-fix, and neither carries a banner despite the "One Fact" section's claim being correct on process grounds. On the *specific* numeric spot-checks performed here, the boundary-value tables in both reports are not currently wrong — but (a) they are not reproducible from their own current source scripts without manual intervention (Deep Dive Factor 6 provenance gap), (b) their concrete trace-level failure-mode examples cite `trace_steps.csv` data whose `correct` labels are subject to the disclosed-but-unapplied ~1.62% regrade, and (c) their "3 failure modes" narrative claim has not been checked against a full taxonomy by this module. This is exposure facts, not a supersession verdict — Module 1 owns the verdict.

---

## Task E — Existing analysis scripts

Both scripts were **run to completion during this audit** (read-only, no repo mutation — they only read CSVs and print to stdout) to check correctness and reusability.

### `research/analyze_runs.py`

- **Computes:** for every `{model}__{dataset}` cell in `experiment_matrix`, pivots `detector_comparison_by_run.csv` on `hazard_drift` vs `never_stop` `stop_utility`, parses temperature out of the `run_id` string (via regex, since `detector_comparison_by_run.csv` has no `temperature` column of its own), and reports `pct_strictly_useful = mean(hazard_drift_utility > never_stop_utility) * 100`, grouped by dataset×temperature, by dataset, and by temperature.
- **Win/loss/tie logic vs. documented definition:** **Partial match.** It computes `strictly_useful = 1{hd_util > ns_util}` correctly (a strict win), but **collapses ties and losses into one "not useful" bucket** — it never separates `==` (tie) from `<` (loss), so it cannot reproduce the 89.59%/2.83%/7.57% three-way split on its own. It *can* reproduce the per-dataset/per-temperature **win rates** exactly (confirmed below).
- **Tree:** `ROOT = "research/outputs/experiment_matrix"` (relative path) — correct, authoritative tree.
- **Ever run to completion before this audit?** No evidence found — no saved output file anywhere in the repo, no reference in any doc. **Run now, output:**
  - `Loaded 74540 runs across all valid cells.` — exact match to the A1 recount.
  - Per-dataset win rates: ARC 94.625641%, GPQA 92.053132%, GSM8K 85.323077%, MATH 86.400354% — **exact match** to `July_1_Checkin.md` Part 6 §2 (94.63%/92.05%/85.32%/86.40%).
  - Per-temperature win rates: 0.1→89.817684%, 0.6→89.946068%, 1.0→89.020807% — **exact match** to Part 6 §3 (89.82%/89.95%/89.02%).
- **Bugs noticed (read-only, not fixed):** (1) no tie/loss split as above; (2) prints only, saves nothing to disk; (3) `parse_temp`'s regex-and-round approach is workable but fragile (works only because `run_id` embeds `temp0.10`/`temp0.60`/`temp1.00`-style substrings consistently — confirmed true for every cell checked, but not schema-guaranteed).
- **Reusable for Module 2?** Yes, as a per-dataset/per-temperature win-rate cross-check, once extended for the 3-way split (which `scratch_analyze.py`, below, already does).

### `scratch_analyze.py` (repo root)

- **Computes:** identical cell/pivot approach to `analyze_runs.py`, but classifies every run as `"win"` (`hd_util > ns_util`), `"tie"` (`hd_util == ns_util`), or `"loss"` (`hd_util < ns_util`) and reports the full 3-way `value_counts()`.
- **Win/loss/tie logic vs. documented definition:** **Exact match** to the definition in `Scientific_Rigor_Research_Prompt.md` §0.4 (per-run_id comparison of `hazard_drift` vs `never_stop` `stop_utility` from `detector_comparison_by_run.csv`).
- **Tree:** hardcoded absolute path `C:\Aditya_Data\Personal\ResearchThesis\research\outputs\experiment_matrix` — correct, authoritative tree, but non-portable (absolute Windows path).
- **Ever run to completion before this audit?** No evidence found (same search as above). **Run now, output:**
  ```
  Total runs: 74540
  win     66784   (89.594848%)
  loss     5646    (7.574457%)
  tie      2110    (2.830695%)
  ```
  **Exact match, to the run, to `July_1_Checkin.md` Part 6 §1** (89.59%/66,784; 2.83%/2,110; 7.57%/5,646). This is the strongest available confirmation that the headline figure is reproducible from current on-disk data by the documented method.
- **Bugs noticed:** (1) prints only, saves nothing to disk (same as `analyze_runs.py` — the brief's suspicion that "its output doesn't appear to be saved anywhere" is confirmed for both scripts); (2) no per-dataset/per-model/per-temperature breakdown (only a global 3-way split — `analyze_runs.py` has the grouping logic `scratch_analyze.py` lacks); (3) relies on `pivot()` succeeding, which silently assumes no duplicate `(run_id, detector)` pairs per cell — true today (confirmed no crash across all 51 populated cells) but not asserted.
- **Reusable for Module 2?** Yes — **this is the correct starting point for the fresh global recompute.** It already matches the documented definition exactly and reproduces 74,540/89.59/2.83/7.57 to the run. It needs, at minimum: (a) per-dataset/per-model/per-temperature grouping (borrow from `analyze_runs.py`), (b) a saved CSV/JSON output (neither script currently saves anything), and (c) explicit handling/reporting of the two known gaps found in this audit — the `qwen2p5_7b__math` 75/1,500 shortfall and the four GPQA cells' sub-1,344 counts — rather than silently inheriting them.

**Neither script has ever been executed to completion and saved anywhere prior to this audit pass** — both were purely read-only source until run just now as part of this fact-finding exercise.

---

## Task F — Anomalies noticed en route

1. **Dataset split usage, not reconciled anywhere in prose.** `matrix_manifest.json` command strings show `--dataset-split train` for `gsm8k` and `gpqa`, and `--dataset-split test` for `math` and `arc` (verified directly: e.g. `mistral_7b_instruct_v0p3__gsm8k` → `--task-source gsm8k --dataset-split train`; `mistral_7b_instruct_v0p3__math` → `--task-source math --dataset-split test`; `mistral_7b_instruct_v0p3__arc` → `--task-source arc --dataset-split test`; `mistral_7b_instruct_v0p3__gpqa` → `--task-source gpqa --dataset-split train`). Run IDs literally embed this (`gsm8k_train_...`, `math_test_...`). **No ThesisDocs report explicitly states that GSM8K/GPQA runs use the `train` split** while MATH/ARC use `test` — `July_1_Checkin.md` Part 3 lists the four datasets with no split annotation. Worth a downstream check on whether `train`-split usage for GSM8K/GPQA has any leakage implication distinct from the already-fixed GroupKFold-by-`run_id` protection (this module makes no claim either way — flagging only).
2. **GPQA run-count anomaly (Module 1's seeded lead) — reproduced exactly.** July 1 Checkin Part 3's four above-1,344 GPQA counts (`llama_3p1_8b_instruct`=1,348, `qwen2p5_3b`=1,349, `yi_1p5_9b_chat`=1,349, `qwen2p5_0p5b`=1,354) reproduce exactly against both `matrix_manifest.json`'s `n_runs` field and `trace_steps.csv`'s distinct `run_id` count for those four cells (Task A2 table). **New, previously-unflagged wrinkle**: at the `detector_comparison_by_run.csv` stage, all four cells drop *below* the clean 1,344 (1,342/1,343/1,343/1,342 respectively) rather than staying inflated — so whatever produces `detector_comparison_by_run.csv` is *itself* dropping some rows from an already-duplicate-inflated set, net negative relative to the "clean" full set. Duplicate investigation itself is explicitly Module 1's remit; this module reports counts only.
3. **`research/outputs/cross_family/` is a third, unnamed output tree**, dated 2026-06-12, distinct from both the ladder tree and `experiment_matrix`'s `_aggregate/{dataset}/cross_family/` subdirs, written by `research/cross_family_analysis.py`. It is correctly exposed as stale via its linked report's SUPERSEDED banner (`research/CROSS_FAMILY_REPORT.md`), but **the raw CSVs underneath (`cross_family_summary.csv`, `cross_family_detector_comparison.csv`, etc.) carry no banner of their own** — a reader who opens the CSV directly (bypassing the `.md` report) would see un-flagged, floor-violating, pre-remediation numbers (Task B3/B4). Worth a banner or a `README` note directly in that directory.
4. **`qwen2p5_7b__math`'s stale detector-comparison stage (Task A2) is not disclosed anywhere** — its own `summary.md`/`final_results.md` (2026-06-20 11:39 vintage) presumably narrate the 75-run state as if final; not read for numeric claims here (Module 1/2's remit), but the *existence* of the gap is a clean, mechanically-verified fact this module can hand off directly.
5. **`ThesisDocs/Startup_Research_Prompt.md:72`** (see B3/B4) contains directionally-wrong guidance ("prefer step 1") that a future agent following the letter of that document would act on incorrectly. Flagged, not edited (file is out of this module's remit).
6. **The disclosed-but-unapplied regrade correction** (`074bc70`: ~1.62% of ~799K rows would flip on a real, non-dry-run) is a live, quantified exposure on every `correct`-derived number currently reported anywhere, including the freshly-reproduced 7.57%. This is the single largest open numeric-integrity question this module found and is called out again in Blockers.

---

## Blockers

1. **`qwen2p5_7b__math`'s `detector_comparison_by_run.csv` reflects only 75/1,500 trace-complete runs.** Resolved by: re-running `research/trace_analysis.py --input-dir research/outputs/experiment_matrix/qwen2p5_7b__math` (a production-pipeline invocation, out of scope for this module and for the rigor-audit pass generally per the brief's header — flagged for a human/Module-2 decision on whether to do so before or alongside the fresh recompute).
2. **The `074bc70`-disclosed ~1.62%/~799K-row regrade has never actually been applied (dry-run only).** Every `correct`-derived number in the repo, including the freshly-reproduced 74,540/89.59/2.83/7.57 headline, is computed on pre-regrade labels. Resolved by: a human decision on whether/when to run `research/regrade_traces.py --root research/outputs/experiment_matrix` for real (not `--dry-run`) — a data-mutating production step outside this module's and the whole rigor-audit pass's stated scope ("Small throwaway analysis/verification scripts are in scope... editing... any other production pipeline file, and launching any new GPU job, are explicitly out of scope").
3. **208 per-cell report files (52 cells × 4 report types: `summary.md`/`final_results.md`/`answers.md`/`open_questions.md`) were not individually read** — Task D's per-cell staleness claim rests on a 3-file sample plus the `_aggregate`-level mtime pattern, as the brief's "one row class per dataset" allowance permits. A full 208-file pass would need either more time budget or a scripted grep-based sweep (feasible, ~15 min, not performed here — flagged as a cheap follow-up for whichever module needs per-cell report text, likely Module 2).
4. **`research/ALPHA_BETA_PREDICTION_REPORT.md` and `research/overthinking_boundary.md`** were confirmed to carry the SUPERSEDED banner but not deep-read beyond that (their staleness is already resolved by the banner; deep-reading them was judged low marginal value given the explicit banner and out-of-scope status implied by "trust them least").
