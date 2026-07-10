#!/usr/bin/env python
"""Classify every hazard_drift-vs-never_stop LOSS run into a programmatic taxonomy.

Purpose (rigor-audit Module 2, 2026-07-09)
------------------------------------------
The July 1 headline says: over 74,540 completed runs, the hazard_drift stopping
policy beat the never_stop baseline in 89.59%, tied in 2.83%, and LOST in 7.57%
(5,646 runs).  A "loss" is a DECISION-QUALITY event (the policy's stop yielded
strictly lower stop_utility than never stopping) -- it is NOT a model-accuracy
number.  This script:

  1. Recomputes the global win/loss/tie breakdown fresh from
     ``{matrix-root}/{model}__{dataset}/detector_comparison_by_run.csv``
     (merge-based, explicit duplicate/NaN accounting, epsilon sensitivity)
     rather than trusting any prior report.
  2. Computes per-dataset / per-model / per-temperature loss rates directly,
     with and without the known-defective ``qwen2p5_7b__math`` cell (whose
     detector_comparison stage covers only 75/1,500 collected runs).
  3. Joins every loss run to its per-step history in ``trace_steps.csv``
     (after the production ``_sanitize_step_frame`` corruption filter,
     imported from ``trace_analysis.py`` so drop semantics are identical to
     the pipeline that produced the comparison files) and assigns each loss
     to exactly one category (see TAXONOMY below).
  4. Optionally (``--with-regrade``) re-derives stop/final-step labels with the
     fixed graders (imported from ``real_trace_experiments.py``, the same
     functions ``regrade_traces.py`` uses) and re-scores verdicts under
     label variants, quantifying whether the 7.57% survives the
     never-applied label correction disclosed in commit 074bc70.
     ``--regrade-scope losses`` (default) regrades only loss runs' stop/final
     rows -- cheap, and sufficient to test whether any loss DISSOLVES under
     corrected labels (category A).  It cannot detect wins/ties that would
     BECOME losses; that direction is bounded by the Module 1 audit
     (rigor_audit/01_...md section 4b/4c: 1 behavioral flip in 798,770 rows,
     not at any stop/final step).  ``--regrade-scope all`` regrades every
     run's stop/final rows (slow on MATH cells: sympy fallback).
  5. Optionally (``--with-probe``) fits an out-of-sample logistic probe:
     among runs that stopped on a wrong answer, do the signals recorded AT the
     stop step predict which traces later repair (= become losses)?
     GroupKFold(5) grouped by task_id, so the same problem is never in train
     and test.

Definitions (verified against July_1_Checkin.md Part 6 + trace_analysis.py)
---------------------------------------------------------------------------
Per run_id: compare ``stop_utility`` of detector ``hazard_drift`` vs detector
``never_stop`` in detector_comparison_by_run.csv.  Strict ``>`` is a win,
equality a tie, strict ``<`` a loss.  ``stop_utility`` equals the trace_steps
``utility`` at the detector's stop step, and ``utility = correct - 0.05*(step-1)``
(real_trace_experiments.py:159,726-727; trace_analysis.py:103-104).  never_stop
stops at the run's last step (trace_analysis.py:417).  Because of that formula,
every loss is mechanically a run whose stop-step answer was WRONG and whose
final-step answer was CORRECT (a missed late correction) -- verified, not assumed.

TAXONOMY (mutually exclusive, first matching predicate wins; see functions
``classify_*`` below).  Applied to the loss population:
  A  grader_artifact_dissolves  (--with-regrade only) under corrected labels
       (MCQ-aware regrade variant) the run is no longer a loss.
  B  stopped_on_empty_answer    normalized answer at the stop step is empty:
       the policy stopped while the extractor had NO candidate answer.
  C  passed_earlier_correct     a decision-eligible (step>=2) correct answer
       existed strictly BEFORE the stop step; the policy sailed past it,
       stopped on a wrong answer, and the trace repaired again later.
  D  step1_only_correct         correct at step 1 (the forced-commit init,
       not a decision point; T_MIN=2 floor) and never correct at any
       eligible step <= stop; repaired after the stop.
  E  next_step_repair           first eligible correct step == stop+1
       (photo-finish miss).
  F  late_repair                first eligible correct step >= stop+2.

Re-run instructions
-------------------
    python research/classify_losses.py \
        --matrix-root research/outputs/experiment_matrix \
        --out <SOME_DIR_OUTSIDE_research/outputs> \
        [--with-regrade] [--regrade-scope losses|all] [--with-probe] \
        [--cells CELL1,CELL2]   # optional cell filter, e.g. for cheap validation

Never point --out inside research/outputs/ (this script must stay read-only
with respect to pipeline artifacts; it refuses such paths).  Outputs:
per_run_verdicts.csv, loss_classification.csv, taxonomy_summary.csv,
slice_tables.txt, regrade_sensitivity.csv (+ .txt), probe_results.txt,
join_audit.txt.  Deterministic: fixed fold construction and sorted iteration.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# Reuse the audited helpers rather than duplicating them (Module 0 Task E):
#  - analyze_runs.parse_temp: temperature from run_id (dcbr has no temp column)
#  - trace_analysis._sanitize_step_frame: the production corrupt-row filter that
#    determined which runs are inside detector_comparison_by_run.csv at all.
# scratch_analyze.py is intentionally NOT imported: it is a side-effect-only
# script (runs its whole analysis at import time) with no importable functions.
def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


analyze_runs = _load_module("analyze_runs", HERE / "analyze_runs.py")
trace_analysis = _load_module("trace_analysis", HERE / "trace_analysis.py")

STEP_COST = 0.05  # == real_trace_experiments.STEP_COST / trace_analysis.STEP_COST

TRACE_USECOLS = [
    "run_id", "task_id", "step", "correct", "utility", "answer", "answer_normalized",
    "expected_answer", "domain", "task_source", "confidence", "entropy_mean", "entropy_std",
    "answer_changed", "thought_token_count", "hidden_l2_shift", "hidden_cosine_shift",
    "hidden_norm", "elapsed_seconds", "lexical_echo", "verbose_confidence_proxy",
    "parse_success", "answer_extraction_source", "model_stop_flag", "hit_max_new_tokens",
    "truncated_output_suspected", "temperature", "seed",
]

PROBE_FEATURES = [
    "stop_entropy_mean", "stop_entropy_std", "stop_confidence", "stop_answer_changed",
    "stop_answer_streak", "stop_thought_token_count", "stop_hidden_l2_shift",
    "stop_hidden_cosine_shift", "stop_lexical_echo", "stop_verbose_confidence_proxy",
    "hd_step", "stop_parse_success", "stop_hit_max_new_tokens",
    "stop_truncated_output_suspected", "stop_model_stop_flag", "temperature",
]

CATEGORY_TAGS = {
    "A_grader_artifact_dissolves": "(i) pipeline/grader bug",
    "B_stopped_on_empty_answer": "(i) pipeline/policy-guard bug (empty-candidate stop state; probe AUC~0.99)",
    "C_passed_earlier_correct": "(iii) fixable-with-better-feature (visible earlier correct state; probe AUC~0.82)",
    "D_step1_only_correct": "(iii) qualified: T_MIN=2 floor made the ideal stop unreachable, but repair signal exists (probe AUC~0.75)",
    "E_next_step_repair": "(iii) fixable-with-better-feature (repair one step away; probe AUC~0.67)",
    "F_late_repair": "(ii) hard online-decision limit at the required foresight horizon; residual signal weak (probe AUC~0.68)",
}


# --------------------------------------------------------------------------- #
# stage 1: fresh win/loss/tie from detector_comparison_by_run.csv
# --------------------------------------------------------------------------- #
def cell_names(matrix_root: Path, only: set[str] | None = None) -> list[str]:
    names = sorted(d.name for d in matrix_root.iterdir()
                   if d.is_dir() and "__" in d.name and not d.name.startswith("_"))
    if only:
        missing = only - set(names)
        if missing:
            raise SystemExit(f"--cells not found under {matrix_root}: {sorted(missing)}")
        names = [n for n in names if n in only]
    return names


def fresh_verdicts(matrix_root: Path, audit: list[str], only: set[str] | None = None) -> pd.DataFrame:
    parts = []
    for cell in cell_names(matrix_root, only):
        f = matrix_root / cell / "detector_comparison_by_run.csv"
        if not f.exists():
            audit.append(f"{cell}: NO detector_comparison_by_run.csv (skipped)")
            continue
        df = pd.read_csv(f)
        model, dataset = cell.split("__", 1)
        ndup = int(df.duplicated(subset=["run_id", "detector"]).sum())
        if ndup:
            audit.append(f"{cell}: {ndup} duplicate (run_id,detector) rows in dcbr")
        frames = {}
        for det, tag in (("hazard_drift", "hd"), ("never_stop", "ns"), ("oracle", "oracle")):
            frames[tag] = (df[df.detector == det][["run_id", "stop_step", "stop_utility"]]
                           .rename(columns={"stop_step": f"{tag}_step", "stop_utility": f"{tag}_util"}))
        m = frames["hd"].merge(frames["ns"], on="run_id", how="outer", indicator=True)
        n_unmatched = int((m["_merge"] != "both").sum())
        if n_unmatched:
            audit.append(f"{cell}: {n_unmatched} run_ids missing one of hazard_drift/never_stop")
        m = m[m["_merge"] == "both"].drop(columns="_merge").merge(frames["oracle"], on="run_id", how="left")
        n_nan = int(m["hd_util"].isna().sum() + m["ns_util"].isna().sum())
        if n_nan:
            audit.append(f"{cell}: {n_nan} NaN stop_utility values")
        m["cell"], m["model"], m["dataset"] = cell, model, dataset
        m["temperature"] = m["run_id"].map(analyze_runs.parse_temp)
        if m["temperature"].isna().any():
            audit.append(f"{cell}: {int(m['temperature'].isna().sum())} unparseable temperatures")
        parts.append(m)
    per_run = pd.concat(parts, ignore_index=True)
    diff = per_run["hd_util"] - per_run["ns_util"]
    per_run["util_diff"] = diff
    per_run["verdict"] = np.where(diff > 0, "win", np.where(diff < 0, "loss", "tie"))
    return per_run


def epsilon_sensitivity(per_run: pd.DataFrame) -> str:
    lines = ["tie-detection sensitivity (exact vs |diff|<=eps):"]
    d = per_run["util_diff"]
    for eps in (0.0, 1e-12, 1e-9, 1e-6, 1e-3, 1e-2):
        lines.append(f"  eps={eps:<8g} win={int((d > eps).sum()):7d} tie={int((d.abs() <= eps).sum()):6d} "
                     f"loss={int((d < -eps).sum()):6d}")
    nz = d[d != 0].abs()
    lines.append(f"  smallest nonzero |diff| = {nz.min():g} (one step-cost quantum = {STEP_COST})")
    return "\n".join(lines)


def slice_table(frame: pd.DataFrame, keys) -> pd.DataFrame:
    g = frame.groupby(keys)["verdict"].value_counts().unstack(fill_value=0)
    for c in ("win", "tie", "loss"):
        if c not in g.columns:
            g[c] = 0
    g["n"] = g[["win", "tie", "loss"]].sum(axis=1)
    for c in ("win", "tie", "loss"):
        g[f"{c}_rate_pct"] = 100.0 * g[c] / g["n"]
    p = g["loss"] / g["n"]
    g["loss_se_pp"] = 100.0 * np.sqrt(p * (1 - p) / g["n"])
    return g[["n", "win", "tie", "loss", "win_rate_pct", "tie_rate_pct", "loss_rate_pct", "loss_se_pp"]].round(4)


# --------------------------------------------------------------------------- #
# stage 2: trace join (per-step histories)
# --------------------------------------------------------------------------- #
def _answer_streak(values: list) -> list[int]:
    out, cur, prev = [], 0, object()
    for v in values:
        key = v if isinstance(v, str) else ("" if pd.isna(v) else v)
        cur = cur + 1 if (key != "" and key == prev) else 1
        out.append(cur)
        prev = key
    return out


def join_traces(matrix_root: Path, per_run: pd.DataFrame, audit: list[str]):
    stop_parts, loss_step_parts = [], []
    for cell, cell_runs in sorted(per_run.groupby("cell"), key=lambda kv: kv[0]):
        ts = pd.read_csv(matrix_root / cell / "trace_steps.csv",
                         usecols=lambda c: c in TRACE_USECOLS, low_memory=False)
        n_runs_pre = ts["run_id"].nunique()
        ts, n_dropped = trace_analysis._sanitize_step_frame(ts)
        if n_dropped:
            audit.append(f"{cell}: sanitizer dropped {n_dropped} corrupt rows "
                         f"[{n_runs_pre} -> {ts['run_id'].nunique()} distinct run_ids]")
        n_dup = int(ts.duplicated(subset=["run_id", "step"]).sum())
        if n_dup:
            audit.append(f"{cell}: {n_dup} duplicated (run_id,step) rows -> keep first")
            ts = ts.drop_duplicates(subset=["run_id", "step"], keep="first")
        bad_util = (ts["utility"] - (ts["correct"].astype(float) - STEP_COST * (ts["step"] - 1.0))).abs() > 1e-9
        if bad_util.any():
            audit.append(f"{cell}: {int(bad_util.sum())} rows where utility != correct-0.05*(step-1)")

        ts = ts.sort_values(["run_id", "step"])
        ts["answer_streak"] = ts.groupby("run_id")["answer_normalized"].transform(
            lambda s: _answer_streak(s.tolist()))
        sub = ts[ts["run_id"].isin(set(cell_runs["run_id"]))]
        g = sub.groupby("run_id")
        corr = sub[sub["correct"] == 1].groupby("run_id")["step"]
        traj = pd.DataFrame({
            "n_steps": g["step"].count(),
            "last_step": g["step"].max(),
            "n_label_flips": g["correct"].apply(lambda s: int((s.diff().abs() > 0).sum())),
            "correct_at_1": g.apply(
                lambda x: int(x.loc[x["step"] == 1, "correct"].iloc[0]) if (x["step"] == 1).any() else np.nan,
                include_groups=False),
        }).join(corr.min().rename("first_correct_step")).join(corr.max().rename("last_correct_step"))
        cr = cell_runs.set_index("run_id")
        traj = traj.join(cr[["hd_step", "ns_step", "hd_util", "ns_util", "oracle_step", "oracle_util",
                             "verdict", "model", "dataset", "temperature", "cell"]], how="right")

        idx = sub.set_index(["run_id", "step"])
        for prefix, stepcol in (("stop", "hd_step"), ("final", "ns_step")):
            rows = idx.reindex(list(zip(traj.index, traj[stepcol].astype(int)))).reset_index(drop=True)
            n_missing = int(rows["correct"].isna().sum())
            if n_missing:
                audit.append(f"{cell}: {n_missing} missing trace rows at {prefix} steps")
            for col in ("correct", "utility", "answer", "answer_normalized", "expected_answer",
                        "confidence", "entropy_mean", "entropy_std", "answer_changed",
                        "thought_token_count", "hidden_l2_shift", "hidden_cosine_shift",
                        "lexical_echo", "verbose_confidence_proxy", "parse_success",
                        "answer_extraction_source", "model_stop_flag", "hit_max_new_tokens",
                        "truncated_output_suspected", "answer_streak", "task_id", "domain", "task_source"):
                if col in rows.columns:
                    traj[f"{prefix}_{col}"] = rows[col].to_numpy()

        n_mm = int(((traj["hd_util"] - traj["stop_utility"]).abs() > 1e-9).sum() +
                   ((traj["ns_util"] - traj["final_utility"]).abs() > 1e-9).sum())
        if n_mm:
            audit.append(f"{cell}: {n_mm} dcbr-vs-current-trace stop_utility mismatches (stale labels)")

        pre = sub[(sub["correct"] == 1) & (sub["step"] >= 2)]
        traj = traj.join(pre.groupby("run_id")["step"].min().rename("first_correct_step_ge2"))
        ec = pre.merge(traj["hd_step"].rename("hs"), left_on="run_id", right_index=True)
        traj = traj.join(ec[ec["step"] < ec["hs"]].groupby("run_id").size().rename("n_correct_ge2_before_stop"))
        traj["n_correct_ge2_before_stop"] = traj["n_correct_ge2_before_stop"].fillna(0).astype(int)
        stop_parts.append(traj.reset_index().rename(columns={"index": "run_id"}))

        loss_ids = set(cell_runs.loc[cell_runs["verdict"] == "loss", "run_id"])
        if loss_ids:
            ls = sub[sub["run_id"].isin(loss_ids)].copy()
            ls["cell"] = cell
            loss_step_parts.append(ls)
    return (pd.concat(stop_parts, ignore_index=True),
            pd.concat(loss_step_parts, ignore_index=True) if loss_step_parts else pd.DataFrame())


# --------------------------------------------------------------------------- #
# stage 3 (--with-regrade): corrected labels + verdict re-scoring
# --------------------------------------------------------------------------- #
def regrade_stop_rows(stop_rows: pd.DataFrame, scope: str = "losses"):
    """Re-grade the answers at the stop and final step with the fixed
    graders (same functions regrade_traces.py uses).  Two variants:
      tool   -- regrade_traces.py semantics verbatim: answer_type='math' for
                task_source=='math' else 'number' (NOTE: this grades the two
                MCQ datasets, arc/gpqa, with the number/text grader -- a
                mismatch vs collection-time answer_type='mcq';
                real_trace_experiments.py:1084,1120).
      mcqfix -- arc/gpqa graded as 'mcq' (collection-time semantics);
                gsm8k/math identical to `tool`.
    scope='losses' regrades only loss runs (cheap; category-A support);
    scope='all' regrades every run.  Non-regraded rows get NaN, which
    rescore() fills with the stored labels.
    Returns stop_rows with new columns {stop,final}_correct_{tool,mcqfix}.
    NA note: pd.read_csv already swallowed literal 'N/A' answer strings to
    NaN, which grade as "" (wrong).  For MCQ that is semantically correct
    behavior (a decline-to-answer is not the gold letter) and it sidesteps
    the collection-time normalize_mcq_answer('N/A')->'A' false-credit defect
    (Module 1 section 4a); it is NOT label drift.
    """
    rte = _load_module("rte_for_classify", HERE / "real_trace_experiments.py")
    memo: dict = {}

    def grade(atype, ans, exp) -> int:
        a = "" if pd.isna(ans) else str(ans)
        e = "" if pd.isna(exp) else str(exp)
        key = (atype, a, e)
        if key not in memo:
            task = rte.TaskSpec(task_id="", domain="", difficulty="", prompt="",
                                answer_type=atype, expected_answer=e, notes="",
                                source="", source_index=-1)
            memo[key] = int(rte.verify_answer(task, a))
        return memo[key]

    mask = (stop_rows["verdict"] == "loss") if scope == "losses" else pd.Series(True, index=stop_rows.index)
    sub = stop_rows[mask]
    for prefix in ("stop", "final"):
        src = sub[f"{prefix}_task_source"].astype(str).str.strip().str.lower()
        atype_tool = np.where(src == "math", "math", "number")
        atype_fix = np.where(src.isin(["arc", "gpqa"]), "mcq", atype_tool)
        ans = sub[f"{prefix}_answer"].to_numpy(dtype=object)
        exp = sub[f"{prefix}_expected_answer"].to_numpy(dtype=object)
        for label, atypes in ((f"{prefix}_correct_tool", atype_tool),
                              (f"{prefix}_correct_mcqfix", atype_fix)):
            stop_rows.loc[mask, label] = [grade(atypes[i], ans[i], exp[i]) for i in range(len(sub))]
    return stop_rows


def rescore(stop_rows: pd.DataFrame) -> pd.DataFrame:
    """Verdicts under label variants.  The hazard_drift STOP STEP is held fixed:
    its stopping decision is a deterministic function of the recorded generation
    signals (trace_analysis.FEATURE_COLUMNS -- `correct` is not among them), so
    correcting labels changes the SCORING of the already-made decision, not the
    decision itself.  (Second-order caveat: the probe/hazard models were TRAINED
    on labels; a full retrain-and-resimulate is a pipeline run, out of scope,
    and would answer a different question -- "what would a policy trained on
    clean labels do" -- rather than "was the reported scoreboard right".)"""
    out = []
    variants = {
        "v0_current_trace_labels": ("stop_correct", "final_correct"),
        "v1_regrade_tool": ("stop_correct_tool", "final_correct_tool"),
        "v2_regrade_mcqfix": ("stop_correct_mcqfix", "final_correct_mcqfix"),
    }
    sr = stop_rows.copy()
    # rows outside the regrade scope keep their stored labels
    for prefix in ("stop", "final"):
        for suffix in ("tool", "mcqfix"):
            col = f"{prefix}_correct_{suffix}"
            sr[col] = sr[col].fillna(sr[f"{prefix}_correct"])
    # v3: conservative -- apply tool regrade only where its answer_type matches
    # collection semantics (gsm8k/math); keep current labels on arc/gpqa.
    is_mcq_ds = sr["dataset"].isin(["arc", "gpqa"])
    sr["stop_correct_v3"] = np.where(is_mcq_ds, sr["stop_correct"], sr["stop_correct_tool"])
    sr["final_correct_v3"] = np.where(is_mcq_ds, sr["final_correct"], sr["final_correct_tool"])
    variants["v3_regrade_gsm8k_math_only"] = ("stop_correct_v3", "final_correct_v3")

    for name, (sc, fc) in variants.items():
        hd = sr[sc].astype(float) - STEP_COST * (sr["hd_step"] - 1.0)
        ns = sr[fc].astype(float) - STEP_COST * (sr["ns_step"] - 1.0)
        d = hd - ns
        sr[f"verdict_{name}"] = np.where(d > 0, "win", np.where(d < 0, "loss", "tie"))
        row = {"variant": name}
        for v in ("win", "tie", "loss"):
            row[v] = int((sr[f"verdict_{name}"] == v).sum())
        row["loss_rate_pct"] = 100.0 * row["loss"] / len(sr)
        out.append(row)
    base = {"variant": "dcbr_asis", "win": int((sr.verdict == "win").sum()),
            "tie": int((sr.verdict == "tie").sum()), "loss": int((sr.verdict == "loss").sum())}
    base["loss_rate_pct"] = 100.0 * base["loss"] / len(sr)
    return pd.DataFrame([base] + out), sr


# --------------------------------------------------------------------------- #
# stage 4: taxonomy predicates (documented, mutually exclusive, precedence order)
# --------------------------------------------------------------------------- #
def classify_A_grader_artifact(L: pd.DataFrame) -> pd.Series:
    """Loss dissolves under corrected (mcq-aware) labels: re-scored verdict != loss."""
    if "verdict_v2_regrade_mcqfix" not in L.columns:
        return pd.Series(False, index=L.index)
    return L["verdict_v2_regrade_mcqfix"] != "loss"


def classify_B_empty_answer(L: pd.DataFrame) -> pd.Series:
    """The stop-step normalized answer is empty/NaN: policy stopped with NO candidate."""
    return L["stop_answer_normalized"].isna() | (L["stop_answer_normalized"].astype(str).str.strip() == "")


def classify_C_passed_earlier_correct(L: pd.DataFrame) -> pd.Series:
    """A correct answer existed at some decision-eligible step (>=2) strictly before the stop."""
    return L["n_correct_ge2_before_stop"] > 0


def classify_D_step1_only(L: pd.DataFrame) -> pd.Series:
    """Correct at the forced-init step 1 only; never correct at eligible steps <= stop."""
    return (L["correct_at_1"] == 1) & (L["first_correct_step_ge2"].isna() |
                                       (L["first_correct_step_ge2"] > L["hd_step"]))


def classify_E_next_step(L: pd.DataFrame) -> pd.Series:
    """First eligible correct step is exactly stop+1."""
    return L["first_correct_step_ge2"] == L["hd_step"] + 1


def classify(L: pd.DataFrame) -> pd.Series:
    cats = pd.Series("F_late_repair", index=L.index)
    order = [
        ("A_grader_artifact_dissolves", classify_A_grader_artifact),
        ("B_stopped_on_empty_answer", classify_B_empty_answer),
        ("C_passed_earlier_correct", classify_C_passed_earlier_correct),
        ("D_step1_only_correct", classify_D_step1_only),
        ("E_next_step_repair", classify_E_next_step),
    ]
    assigned = pd.Series(False, index=L.index)
    for name, fn in order:
        hit = fn(L) & ~assigned
        cats[hit] = name
        assigned |= hit
    return cats


# --------------------------------------------------------------------------- #
# stage 5 (--with-probe): out-of-sample repair-prediction probe
# --------------------------------------------------------------------------- #
def run_probe(stop_rows: pd.DataFrame, out_lines: list[str]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # ties stopped at the final step (no future left to predict) -> excluded
    sw = stop_rows[(stop_rows.stop_correct == 0) & (stop_rows.verdict != "tie")].copy()
    sw["is_loss"] = (sw.verdict == "loss").astype(int)

    def one(df, feats, name):
        df = df.dropna(subset=feats).reset_index(drop=True)
        y = df["is_loss"].astype(int)
        if y.nunique() < 2 or len(df) < 500:
            out_lines.append(f"{name}: skipped (n={len(df)})")
            return
        X = df[feats].astype(float)
        oof = np.full(len(df), np.nan)
        for tr, te in GroupKFold(n_splits=5).split(X, y, df["stop_task_id"]):
            pipe = Pipeline([("sc", StandardScaler()),
                             ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])
            pipe.fit(X.iloc[tr], y.iloc[tr])
            oof[te] = pipe.predict_proba(X.iloc[te])[:, 1]
        auc = roc_auc_score(y, oof)
        n1, n0 = int(y.sum()), int((1 - y).sum())
        q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
        se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0))
        out_lines.append(f"{name}: n={len(df)} pos={n1} ({100.0*n1/len(df):.1f}%) "
                         f"OOF AUC={auc:.4f} [95% ~ {auc-1.96*se:.4f},{auc+1.96*se:.4f}]")

    one(sw, PROBE_FEATURES, "probe/global/signals-only")
    sw_oh = pd.get_dummies(sw, columns=["dataset", "model"], dtype=float)
    oh = [c for c in sw_oh.columns if c.startswith(("dataset_", "model_"))]
    sw_oh["is_loss"] = sw["is_loss"].to_numpy()
    one(sw_oh, PROBE_FEATURES + oh, "probe/global/signals+cell-onehots")
    for ds in sorted(sw.dataset.unique()):
        one(sw[sw.dataset == ds], PROBE_FEATURES, f"probe/{ds}/signals-only")
    if "category" in sw.columns:
        # per-category separability: category's losses vs ALL stopped-wrong wins
        for cat in sorted(sw.loc[sw.is_loss == 1, "category"].dropna().unique()):
            one(sw[(sw.is_loss == 0) | (sw.category == cat)], PROBE_FEATURES,
                f"probe/category/{cat}-vs-stopped-wrong-wins")
    sw["cell_rate"] = sw.groupby(["dataset", "model"])["is_loss"].transform("mean")
    out_lines.append(f"probe/baseline: AUC of per-(dataset,model) loss base rate alone = "
                     f"{roc_auc_score(sw['is_loss'], sw['cell_rate']):.4f}")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix-root", default=str(HERE / "outputs" / "experiment_matrix"))
    ap.add_argument("--out", required=True, help="output dir (must NOT be inside research/outputs)")
    ap.add_argument("--with-regrade", action="store_true",
                    help="re-grade stop/final answers with the fixed graders and re-score verdicts")
    ap.add_argument("--regrade-scope", choices=("losses", "all"), default="losses",
                    help="'losses' (default, cheap: can only detect losses that dissolve) or 'all'")
    ap.add_argument("--with-probe", action="store_true", help="fit the out-of-sample repair probe")
    ap.add_argument("--cells", default=None,
                    help="comma-separated cell-name filter (default: all cells)")
    args = ap.parse_args()

    matrix_root = Path(args.matrix_root).resolve()
    out = Path(args.out).resolve()
    forbidden = (HERE / "outputs").resolve()
    if str(out).lower().startswith(str(forbidden).lower()):
        raise SystemExit(f"refusing to write into {forbidden}")
    out.mkdir(parents=True, exist_ok=True)
    audit: list[str] = []

    # -- stage 1
    only = set(args.cells.split(",")) if args.cells else None
    per_run = fresh_verdicts(matrix_root, audit, only)
    vc = per_run["verdict"].value_counts()
    headline = [
        f"runs={len(per_run)} (distinct run_ids={per_run['run_id'].nunique()})",
        *(f"{k}: {int(vc.get(k, 0))} ({100.0 * vc.get(k, 0) / len(per_run):.4f}%)" for k in ("win", "tie", "loss")),
        epsilon_sensitivity(per_run),
    ]
    per_run.to_csv(out / "per_run_verdicts.csv", index=False)

    lines = ["== FRESH GLOBAL BREAKDOWN =="] + headline
    defect = "qwen2p5_7b__math"
    for label, frame in (("ALL CELLS", per_run), (f"EXCLUDING {defect}", per_run[per_run.cell != defect])):
        lines += [f"\n== SLICES: {label} (n={len(frame)}) ==",
                  "\n-- by dataset --", slice_table(frame, "dataset").to_string(),
                  "\n-- by temperature --", slice_table(frame, "temperature").to_string(),
                  "\n-- by model --", slice_table(frame, "model").to_string(),
                  "\n-- by model x dataset (loss rate %) --",
                  frame.pivot_table(index="model", columns="dataset", values="verdict",
                                    aggfunc=lambda s: round(100.0 * (s == "loss").mean(), 2)).to_string()]
    # oracle-gap capture
    lines.append("\n== ORACLE-GAP CAPTURE (hd-ns)/(oracle-ns), run-pooled ==")

    def cap(frame):
        o, h, n = frame["oracle_util"].mean(), frame["hd_util"].mean(), frame["ns_util"].mean()
        return f"oracle={o:.4f} hd={h:.4f} ns={n:.4f} capture={100 * (h - n) / (o - n):.2f}%"
    lines.append(f"all cells: {cap(per_run)}")
    for ds in sorted(per_run.dataset.unique()):
        lines.append(f"{ds}: {cap(per_run[per_run.dataset == ds])}")
    late = ["qwen2p5_3b__gsm8k", "qwen2p5_7b__gsm8k", "qwen2p5_7b__math", "qwen2p5_14b__gsm8k",
            "qwen2p5_32b__gsm8k", "qwen2p5_32b__math", "mistral_7b_instruct_v0p3__gsm8k",
            "internlm3_8b_instruct__gsm8k", "yi_1p5_9b_chat__gsm8k", "mistral_small_24b_2409__gsm8k",
            "mistral_small_24b_2409__arc", "llama_3p1_8b_instruct__gsm8k"]
    lr = per_run[per_run.cell.isin(late)]
    lines.append(f"12 late-boundary cells (run-pooled): {cap(lr)}")
    lines.append(f"11 late-boundary cells excl {defect} (run-pooled): {cap(lr[lr.cell != defect])}")
    cellm = lr.groupby("cell")[["hd_util", "ns_util", "oracle_util"]].mean()
    o, h, n = cellm["oracle_util"].mean(), cellm["hd_util"].mean(), cellm["ns_util"].mean()
    lines.append(f"12 late cells, published cell-mean aggregation: oracle={o:.4f} hd={h:.4f} ns={n:.4f} "
                 f"capture={100 * (h - n) / (o - n):.2f}%")

    # -- stage 2
    stop_rows, loss_steps = join_traces(matrix_root, per_run, audit)
    sig = ((stop_rows.verdict == "loss") &
           (stop_rows.stop_correct == 0) & (stop_rows.final_correct == 1))
    lines.append(f"\nmechanical signature (loss => stopped-wrong & final-correct): "
                 f"{int(sig.sum())}/{int((stop_rows.verdict == 'loss').sum())}")

    # -- stage 3
    if args.with_regrade:
        stop_rows = regrade_stop_rows(stop_rows, args.regrade_scope)
        table, stop_rows = rescore(stop_rows)
        table.to_csv(out / "regrade_sensitivity.csv", index=False)
        lines.append(f"\n== REGRADE SENSITIVITY (scope={args.regrade_scope}; verdicts re-scored "
                     "under corrected labels) ==")
        if args.regrade_scope == "losses":
            lines.append("(scope=losses: only loss runs regraded -- detects losses that DISSOLVE; "
                         "wins/ties becoming losses is bounded separately by the Module 1 flip audit)")
        lines.append(table.to_string(index=False))
        for name in ("v1_regrade_tool", "v2_regrade_mcqfix"):
            moved = stop_rows[stop_rows["verdict"] != stop_rows[f"verdict_{name}"]]
            lines.append(f"\n{name}: {len(moved)} runs change verdict; by dataset: "
                         f"{moved.groupby('dataset').size().to_dict()}")

    # -- stage 4
    L = stop_rows[stop_rows.verdict == "loss"].copy()
    L["category"] = classify(L)
    L["category_tag"] = L["category"].map(CATEGORY_TAGS)
    L["gap_steps_to_first_correct"] = L["first_correct_step_ge2"] - L["hd_step"]
    L["util_diff"] = L["hd_util"] - L["ns_util"]
    keep_cols = ["run_id", "cell", "model", "dataset", "temperature", "category", "category_tag",
                 "hd_step", "ns_step", "hd_util", "ns_util", "util_diff", "oracle_step",
                 "first_correct_step_ge2", "gap_steps_to_first_correct", "n_correct_ge2_before_stop",
                 "correct_at_1", "n_label_flips", "stop_answer", "stop_answer_normalized",
                 "final_answer_normalized", "stop_expected_answer", "stop_parse_success",
                 "stop_hit_max_new_tokens", "stop_truncated_output_suspected",
                 "stop_answer_changed", "stop_answer_streak", "stop_confidence", "stop_entropy_mean"]
    if args.with_regrade:
        keep_cols += ["stop_correct_tool", "final_correct_tool", "stop_correct_mcqfix",
                      "final_correct_mcqfix", "verdict_v1_regrade_tool", "verdict_v2_regrade_mcqfix"]
    L[keep_cols].sort_values(["category", "cell", "run_id"]).to_csv(out / "loss_classification.csv", index=False)

    summ = (L.groupby("category").agg(count=("run_id", "size"))
            .assign(share_pct=lambda d: (100.0 * d["count"] / len(L)).round(2)))
    summ["tag"] = summ.index.map(CATEGORY_TAGS)
    summ.to_csv(out / "taxonomy_summary.csv")
    lines.append(f"\n== TAXONOMY (loss population n={len(L)}; counts sum={int(summ['count'].sum())}) ==")
    lines.append(summ.to_string())
    lines.append("\n-- category x dataset --")
    lines.append(pd.crosstab(L.category, L.dataset).to_string())
    lines.append("\n-- category x model --")
    lines.append(pd.crosstab(L.category, L.model).to_string())
    lines.append("\n-- examples (first 3 run_ids per category, sorted) --")
    for cat, grp in L.sort_values("run_id").groupby("category"):
        lines.append(f"  {cat}: " + ", ".join(grp["run_id"].head(3)))

    # -- stage 5
    if args.with_probe:
        probe_lines: list[str] = []
        stop_rows = stop_rows.merge(L[["run_id", "category"]], on="run_id", how="left")
        run_probe(stop_rows, probe_lines)
        (out / "probe_results.txt").write_text("\n".join(probe_lines), encoding="utf-8")
        lines.append("\n== PROBE ==")
        lines += probe_lines

    (out / "join_audit.txt").write_text("\n".join(audit) if audit else "(no anomalies)", encoding="utf-8")
    (out / "slice_tables.txt").write_text("\n".join(lines), encoding="utf-8")
    loss_steps.to_csv(out / "loss_step_histories.csv", index=False)
    print("\n".join(lines))
    print(f"\n[classify_losses] wrote outputs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
