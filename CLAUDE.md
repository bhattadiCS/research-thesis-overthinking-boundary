# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

M.S. thesis research on the **overthinking boundary** in reasoning LLMs: when should chain-of-thought stop? The theory (repair/corruption hazards α_t/β_t, belief q_t, drift μ_t = [(1−q_t)α_t − q_t·β_t](v+c) − λ, boundary T* = inf{t ≥ T_MIN : μ_t ≤ 0}) is in `README.md` — that math is canonical, but the README's "Key Findings" section is April/L4-era and superseded by the 52-cell experiment matrix and the rigor audit. **Current source of truth on what is verified: `ThesisDocs/rigor_audit/00_EXECUTIVE_SUMMARY.md`** (and 01–06 beneath it). Many historical reports in `research/` and the repo root carry ⚠️ SUPERSEDED or correction banners — check for one before citing any report.

## Commands

- **Grader tests**: `python research/tests/test_graders.py` — a direct-run suite that prints `N/N passed`. It is NOT pytest-style; `pytest` collects zero tests here.
- **Quick global win/loss/tie recompute**: `python scratch_analyze.py` (repo root).
- **Loss taxonomy / verdict recompute**: `python research/classify_losses.py --matrix-root research/outputs/experiment_matrix --out <dir> [--with-regrade] [--with-probe]`
- **Re-analyze one cell (CPU)**: `python research/trace_analysis.py --input-dir <cell-dir>` — ⚠️ writes outputs **in place next to the inputs**; for experiments, copy the cell's inputs to a scratch dir first.
- **Regenerate a cell's prose reports**: `python research/generate_thesis_artifacts.py --input-dir <cell-dir> --answers-output ... --open-questions-output ... --research-report-output ... --root-report-output ...` (canonical arg set: `report_cmd()` in `research/run_experiment_matrix.py`).
- **Per-dataset aggregate**: `python research/cross_family_analysis.py --run-dirs <13 cell dirs> --output-dir _aggregate/<ds>/cross_family --report-output _aggregate/<ds>/CROSS_FAMILY_REPORT.md --open-questions-output _aggregate/<ds>/CROSS_FAMILY_OPEN_QUESTIONS.md`
- **Full matrix orchestration (GPU collection)**: `research/run_experiment_matrix.py` drives collect → analyze → report → aggregate per cell; GPU sessions bootstrap via `bash tools/runai/bootstrap_session.sh` (RunAI box; gated models need `.hf_token`). Analysis/report/aggregate stages are CPU-only and safe locally; do not launch collection (GPU) casually.

Local environment: Windows, plain `python` with pandas/numpy/sklearn/sympy. Long analyses: write per-cell intermediate caches so interrupted runs resume.

## Architecture

**Pipeline (one "cell" = `{model}__{dataset}`)**
1. `research/real_trace_experiments.py` (collect, GPU): generates step-by-step reasoning traces, extracts/normalizes answers (`parse_generation`), grades them (`verify_answer`, dispatching on dataset `answer_type`: gsm8k=number, math=math with sympy fallback, arc/gpqa=mcq), writes `trace_steps.csv` / `trace_runs.csv` append-only.
2. `research/trace_analysis.py` (analyze, CPU): fits a correctness probe + repair/corruption hazard logistic models (features: `FEATURE_COLUMNS`), evaluates 9 stopping detectors, writes `detector_comparison_by_run.csv` (one row per run × detector) and summary CSVs. Everything benchmark-facing is **out-of-fold** via GroupKFold(5) by `run_id` (`out_of_sample_eval`); the production stop rule is `hazard_stop_for_group`: first step ≥ 2 with `mu_hat ≤ 0`, else last step.
3. `research/generate_thesis_artifacts.py` (report): the cell's `answers.md` / `open_questions.md` / `final_results.md` / `summary.md`.
4. `research/cross_family_analysis.py` (aggregate): `_aggregate/{dataset}/CROSS_FAMILY_REPORT.md` + `cross_family_summary.csv` (boundary values per model).

**Output trees**
- `research/outputs/experiment_matrix/` — **authoritative**: 52 cells (13 models × gsm8k/math/arc/gpqa), `matrix_manifest.json` is the lab notebook (exact command strings, per-cell statuses). Canonical corpus after the 2026-07 repair: **75,965 runs; wins/ties/losses = 68,095/2,135/5,735 (7.5495% loss)**.
- `research/outputs/real_traces_bf16_ladder/` — legacy 3-model Qwen ladder plus ad-hoc stakes-sweep and prompted-vs-distilled reports.
- `research/outputs/cross_family/` — SUPERSEDED pre-remediation tree (see its `README_SUPERSEDED.md`); never cite.

**Core definitions (used everywhere; do not re-derive variants)**
- `utility = correct − 0.05·(step−1)`; λ convention is `λ·(step−1)` in every script.
- Win/loss/tie = per-`run_id` strict comparison of `stop_utility` for detector `hazard_drift` vs `never_stop`. A "loss" is a decision-quality event, **not** model accuracy; mechanically every loss is a missed late correction (stop-step wrong, final-step correct).
- `T_MIN = 2` floor: step 1 is the forced-commit init, never a decision point — the floor is theory-motivated and was defended quantitatively (protocol P2). Boundary values below 2 anywhere indicate stale pre-remediation data.
- Dataset splits are asymmetric: gsm8k/gpqa use **train**, math/arc use **test**.

## Conventions and footguns

- **`research/regrade_traces.py` mutates `trace_steps.csv` in place** when run without `--dry-run`, and any regrade requires re-running `trace_analysis.py` per cell plus aggregates afterward. Default to `--dry-run`.
- **Joining traces to detector outputs** requires the production corruption filter: `from trace_analysis import _sanitize_step_frame` (interrupted writes historically field-shifted rows; the sanitizer's drop semantics must match the pipeline's).
- **Reproducing policy decisions offline**: the GroupKFold fold assignment depends on exact frame row order, so replication must mirror the chain read_csv → `_sanitize_step_frame` → `add_temporal_features` → add repair/corruption cols → `reset_index(drop=True)`. `research/offline_policy_arms_p3.py` and `research/calibrate_cell_thresholds.py` are validated templates (they reproduced all 75,965 recorded stops with 0 mismatches) — extend them rather than approximating, and always report the stop-mismatch validation count.
- **Leakage discipline**: anything advisor-facing is out-of-fold (GroupKFold by `run_id` for model fits, by `task_id` for policy calibration). In `_aggregate/calibrated_cell_thresholds.csv`: quote `oof_dU`, deploy `delta_full_DEPLOY`, never cite `cal_full_*` (in-sample) as gains.
- **Metric framing**: report utility and win-rate together (they diverge — per-cell calibration improves utility while raising loss count), and never present win-rate complements as loss rates (complements include ties; that trap produced a wrong published narrative once).
- **Staleness**: when collection grows a cell, its analysis artifacts go stale (`analysis_complete()` in `run_experiment_matrix.py` now checks output-vs-trace mtimes). Verify `detector_comparison_by_run.csv` freshness before computing on a cell.
- **Pre-registration pattern**: new experiments follow `ThesisDocs/rigor_audit/04_next_experiment_protocols.md` — one IV, held-constant checklist, falsifiable hypothesis, n via the SE/Z method in `ThesisDocs/July_1_Checkin.md` Part 7, success criterion fixed in advance; results recorded pass/fail either way (05/06 show the format).
- **Git**: direct-to-main, small purposeful commits; regenerated data artifacts are committed together with the code/fix that produced them. Never `git add -A` — the root holds untracked scratch and prompt documents. `.gitignore` keeps `ThesisDocs/*` out except top-level `*.md` and `ThesisDocs/rigor_audit/`; under `research/outputs/` the `.npz`/logs/`hidden_states/` are ignored (hidden-state caches are expected to be absent locally).
