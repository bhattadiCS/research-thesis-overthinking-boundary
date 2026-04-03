"""Phase A+B refined: compute TWO boundary definitions and produce reports.

Boundary T_c^first = first step where μ_t ≤ 0
Boundary T_c^late  = LAST step where μ_t crosses from positive to ≤ 0
                     (i.e., the last usable positive-drift window)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=FutureWarning)

STEP_COST = 0.05
MAX_STEPS = 10
OUTPUT_BASE = Path(__file__).resolve().parent / "outputs"

RUNS = {
    "Qwen 0.5B": "real_traces_l4_qwen_0p5b",
    "DeepSeek 1.5B": "real_traces_l4_deepseek_1p5b",
    "Mistral 7B": "real_traces_l4_mistral_7b",
    "Qwen 7B": "real_traces_l4_qwen_7b_4bit",
}


def load_data(run_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = OUTPUT_BASE / run_dir
    return pd.read_csv(base / "trace_steps.csv"), pd.read_csv(base / "trace_runs.csv")


def compute_step_hazards(steps_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for step in sorted(steps_df["step"].unique()):
        if step < 1:
            continue
        at_step = steps_df[steps_df["step"] == step]
        n = len(at_step)
        if n == 0:
            continue
        q_t = at_step["correct"].astype(bool).mean()

        next_step = step + 1
        at_next = steps_df[steps_df["step"] == next_step]
        if len(at_next) == 0:
            alpha_t = beta_t = 0.0
        else:
            merged = at_step[["run_id", "correct"]].merge(
                at_next[["run_id", "correct"]], on="run_id", suffixes=("_now", "_next")
            )
            merged["correct_now"] = merged["correct_now"].astype(bool)
            merged["correct_next"] = merged["correct_next"].astype(bool)
            wrong_now = merged[~merged["correct_now"]]
            right_now = merged[merged["correct_now"]]
            alpha_t = wrong_now["correct_next"].mean() if len(wrong_now) > 0 else 0.0
            beta_t = (~right_now["correct_next"]).mean() if len(right_now) > 0 else 0.0

        mu_t = (1 - q_t) * alpha_t - q_t * beta_t - STEP_COST
        records.append({"step": step, "q_t": q_t, "alpha_t": alpha_t, "beta_t": beta_t, "mu_t": mu_t, "n_runs": n})
    return pd.DataFrame(records)


def find_boundaries(haz: pd.DataFrame) -> tuple[int, int]:
    """Return (T_c_first, T_c_late).

    T_c_first: first step where μ_t ≤ 0
    T_c_late:  last step BEFORE which the drift was still positive
               (= step where the last positive→non-positive crossing happens)
    """
    if haz.empty:
        return 1, 1
    mu_vals = haz["mu_t"].values
    steps = haz["step"].values.astype(int)

    # First crossing
    neg = np.where(mu_vals <= 0)[0]
    first_tc = int(steps[neg[0]]) if len(neg) > 0 else MAX_STEPS + 1

    # Last crossing: find the LAST step where μ transitions from >0 to ≤0
    last_tc = first_tc
    for i in range(1, len(mu_vals)):
        if mu_vals[i - 1] > 0 and mu_vals[i] <= 0:
            last_tc = int(steps[i])

    return first_tc, last_tc


def classify_problems(all_runs, all_steps):
    records = []
    for model, runs_df in all_runs.items():
        steps_df = all_steps[model]
        for task_idx in sorted(runs_df["task_source_index"].unique()):
            task_runs = runs_df[runs_df["task_source_index"] == task_idx]
            task_steps = steps_df[steps_df["task_source_index"] == task_idx]
            step1_steps = task_steps[task_steps["step"] == 1]
            step1_sr = step1_steps["correct"].astype(bool).mean() if len(step1_steps) > 0 else 0.0
            ever_sr = task_runs["ever_correct"].mean()
            diff = "easy" if step1_sr >= 0.5 else ("medium" if ever_sr > 0 else "hard")
            records.append({"model": model, "task_source_index": task_idx, "step1_solve_rate": step1_sr,
                            "ever_solved_rate": ever_sr, "difficulty": diff})
    return pd.DataFrame(records)


# ── Phase C: trajectory type classification ─────────────────────────────────

def classify_trajectories(steps_df: pd.DataFrame, runs_df: pd.DataFrame) -> pd.DataFrame:
    """Classify each run into repair/corruption/persistent correct/persistent wrong."""
    records = []
    for _, run in runs_df.iterrows():
        rid = run["run_id"]
        run_steps = steps_df[steps_df["run_id"] == rid].sort_values("step")
        if len(run_steps) == 0:
            continue

        correct_seq = run_steps["correct"].astype(bool).values
        step1_correct = correct_seq[0]
        ever_correct = any(correct_seq)

        if step1_correct and all(correct_seq):
            traj_type = "persistent_correct"
        elif step1_correct and not all(correct_seq):
            traj_type = "corruption"
        elif not step1_correct and ever_correct:
            traj_type = "repair"
        else:
            traj_type = "persistent_wrong"

        records.append({"run_id": rid, "trajectory_type": traj_type})
    return pd.DataFrame(records)


def compute_trajectory_features(
    steps_df: pd.DataFrame,
    traj_df: pd.DataFrame,
) -> pd.DataFrame:
    """Mean feature profiles per step × trajectory type."""
    merged = steps_df.merge(traj_df, on="run_id")
    feature_cols = [
        "entropy_mean", "confidence", "hidden_l2_shift", "hidden_cosine_shift",
        "answer_changed", "verbose_confidence_proxy", "correct",
    ]
    avail = [c for c in feature_cols if c in merged.columns]
    grouped = merged.groupby(["trajectory_type", "step"])[avail].mean().reset_index()
    return grouped


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("REFINED ANALYSIS: Phases A, B, C, D")
    print("=" * 70)

    # Load data
    print("\n[1/7] Loading data...")
    all_steps, all_runs = {}, {}
    for label, d in RUNS.items():
        s, r = load_data(d)
        all_steps[label], all_runs[label] = s, r
        print(f"  {label}: {len(s)} steps, {len(r)} runs")

    # Phase A: classify + stratum hazards
    print("\n[2/7] Phase A: Classifying problems...")
    prob_cls = classify_problems(all_runs, all_steps)

    print("\n[3/7] Phase A: Computing stratum hazards with dual-boundary...")
    stratum_rows = []
    stratum_hazards = {}  # model → diff → DataFrame
    for model in RUNS:
        stratum_hazards[model] = {}
        mc = prob_cls[prob_cls["model"] == model]
        for diff in ["easy", "medium", "hard"]:
            tids = mc[mc["difficulty"] == diff]["task_source_index"].values
            if len(tids) == 0:
                stratum_hazards[model][diff] = pd.DataFrame()
                stratum_rows.append({"model": model, "difficulty": diff, "n_problems": 0,
                                     "T_c_first": 1, "T_c_late": 1, "alpha_mean": 0, "beta_mean": 0,
                                     "alpha_beta_ratio": 0, "step1_acc": 0, "peak_acc": 0,
                                     "peak_step": 0, "delta_peak_step1": 0})
                continue
            subset = all_steps[model][all_steps[model]["task_source_index"].isin(tids)]
            haz = compute_step_hazards(subset)
            haz["difficulty"] = diff
            haz["n_problems"] = len(tids)
            stratum_hazards[model][diff] = haz

            tc_first, tc_late = find_boundaries(haz)
            am = haz["alpha_t"].mean()
            bm = haz["beta_t"].mean()
            ratio = am / bm if bm > 0 else (float("inf") if am > 0 else 0)
            s1 = float(haz[haz["step"] == 1]["q_t"].iloc[0]) if len(haz[haz["step"] == 1]) > 0 else 0
            pk = float(haz["q_t"].max())
            pk_step = int(haz.loc[haz["q_t"].idxmax(), "step"]) if len(haz) > 0 else 0
            stratum_rows.append({
                "model": model, "difficulty": diff, "n_problems": len(tids),
                "T_c_first": tc_first, "T_c_late": tc_late,
                "alpha_mean": am, "beta_mean": bm, "alpha_beta_ratio": ratio,
                "step1_acc": s1, "peak_acc": pk, "peak_step": pk_step,
                "delta_peak_step1": pk - s1,
            })

    stratum_summary = pd.DataFrame(stratum_rows)
    print("\n  Stratum boundary summary (dual boundary):")
    cols = ["model", "difficulty", "n_problems", "step1_acc", "peak_acc", "peak_step",
            "delta_peak_step1", "alpha_mean", "beta_mean", "alpha_beta_ratio", "T_c_first", "T_c_late"]
    print(stratum_summary[cols].to_string(index=False, float_format="{:.4f}".format))

    # Phase C: trajectory classification
    print("\n[4/7] Phase C: Classifying trajectories...")
    all_traj = {}
    all_traj_features = {}
    for model in RUNS:
        traj = classify_trajectories(all_steps[model], all_runs[model])
        all_traj[model] = traj
        print(f"  {model}:")
        print(f"    {traj['trajectory_type'].value_counts().to_dict()}")

        feat_prof = compute_trajectory_features(all_steps[model], traj)
        all_traj_features[model] = feat_prof

    # ── Save everything ──────────────────────────────────────────────────
    print("\n[5/7] Saving outputs...")

    # Phase A
    out_a = OUTPUT_BASE / "difficulty_stratified_analysis"
    out_a.mkdir(parents=True, exist_ok=True)
    prob_cls.to_csv(out_a / "problem_classification.csv", index=False)
    stratum_summary.to_csv(out_a / "stratum_boundary_summary.csv", index=False)

    for model in RUNS:
        safe = model.lower().replace(" ", "_")
        for diff, haz in stratum_hazards[model].items():
            if not haz.empty:
                haz.to_csv(out_a / f"hazard_{safe}_{diff}.csv", index=False)

    # Phase C
    out_c = OUTPUT_BASE / "trajectory_type_analysis"
    out_c.mkdir(parents=True, exist_ok=True)
    for model in RUNS:
        safe = model.lower().replace(" ", "_")
        all_traj[model].to_csv(out_c / f"trajectories_{safe}.csv", index=False)
        all_traj_features[model].to_csv(out_c / f"feature_profiles_{safe}.csv", index=False)

    # ── Figures ──────────────────────────────────────────────────────────
    print("\n[6/7] Generating figures...")

    # Figure 1: 4x3 drift curves (Phase A)
    fig, axes = plt.subplots(4, 3, figsize=(16, 18), sharex=True)
    model_order = list(RUNS.keys())
    diff_order = ["easy", "medium", "hard"]
    colors_diff = {"easy": "#2ecc71", "medium": "#3498db", "hard": "#e74c3c"}

    for row, model in enumerate(model_order):
        for col, diff in enumerate(diff_order):
            ax = axes[row, col]
            haz = stratum_hazards[model].get(diff, pd.DataFrame())
            if haz.empty:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=12, color="gray")
            else:
                steps = haz["step"].values
                mu = haz["mu_t"].values
                ax.plot(steps, mu, "o-", color=colors_diff[diff], linewidth=2, markersize=5)
                ax.fill_between(steps, mu, 0, alpha=0.15, color=colors_diff[diff])
                ax.axhline(0, color="red", linestyle="--", alpha=0.7, linewidth=1)

                _, tc_late = find_boundaries(haz)
                if 1 < tc_late <= MAX_STEPS:
                    ax.axvline(tc_late, color="orange", linestyle=":", linewidth=2, label=f"T_c={tc_late}")
                    ax.legend(fontsize=9, loc="upper right")

                # Annotate n_problems
                n_p = int(haz["n_problems"].iloc[0]) if "n_problems" in haz.columns else "?"
                ax.text(0.98, 0.02, f"n={n_p}", transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            ax.set_xlim(0.5, 10.5)
            if row == 0:
                ax.set_title(f"{diff.capitalize()}", fontsize=13, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{model}\nμ_t", fontsize=11)
            if row == len(model_order) - 1:
                ax.set_xlabel("Step", fontsize=11)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Drift μ_t by Model × Problem Difficulty\nLast Positive→Negative Crossing = T_c",
                 fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_a / "stratum_drift_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved stratum drift grid")

    # Figure 2: trajectory type feature profiles (Phase C)
    feature_to_plot = ["entropy_mean", "confidence", "hidden_l2_shift", "answer_changed"]
    avail_features = [f for f in feature_to_plot if f in all_traj_features[model_order[0]].columns]

    fig, axes = plt.subplots(len(model_order), len(avail_features), figsize=(5 * len(avail_features), 4 * len(model_order)))
    if len(model_order) == 1:
        axes = axes.reshape(1, -1)
    traj_colors = {"repair": "#2ecc71", "corruption": "#e74c3c", "persistent_correct": "#3498db", "persistent_wrong": "#95a5a6"}

    for row, model in enumerate(model_order):
        feat_df = all_traj_features[model]
        for col, feat in enumerate(avail_features):
            ax = axes[row, col]
            for ttype in ["repair", "corruption", "persistent_correct", "persistent_wrong"]:
                sub = feat_df[feat_df["trajectory_type"] == ttype]
                if len(sub) > 0 and feat in sub.columns:
                    ax.plot(sub["step"], sub[feat], "o-", color=traj_colors[ttype], label=ttype, markersize=3, linewidth=1.5)
            if row == 0:
                ax.set_title(feat.replace("_", " ").title(), fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(model, fontsize=10)
            if row == len(model_order) - 1:
                ax.set_xlabel("Step", fontsize=10)
            if row == 0 and col == len(avail_features) - 1:
                ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

    fig.suptitle("Feature Profiles by Trajectory Type", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_c / "trajectory_feature_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved trajectory feature profiles")

    # Figure 3: α/β scatter (Phase D)
    out_d = OUTPUT_BASE / "alpha_beta_predictive_analysis"
    out_d.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors_model = {"Qwen 0.5B": "#e74c3c", "DeepSeek 1.5B": "#e67e22", "Mistral 7B": "#3498db", "Qwen 7B": "#2ecc71"}
    markers_diff = {"easy": "o", "medium": "s", "hard": "^"}

    plot_x, plot_y = [], []
    for _, row in stratum_summary.iterrows():
        ratio = row["alpha_beta_ratio"]
        if ratio <= 0 or np.isinf(ratio) or row["n_problems"] == 0:
            continue
        x = np.log10(ratio)
        y = row["T_c_late"]
        plot_x.append(x)
        plot_y.append(y)
        ax.scatter(x, y, color=colors_model.get(row["model"], "gray"),
                   marker=markers_diff.get(row["difficulty"], "o"),
                   s=max(30, row["n_problems"]), zorder=5, edgecolors="black", linewidth=0.5)
        ax.annotate(f"{row['model']}\n{row['difficulty']} (n={row['n_problems']})",
                    (x, y), fontsize=7, ha="left", va="bottom",
                    xytext=(6, 4), textcoords="offset points")

    # Add aggregate model-level points from cross_family_summary
    agg_points = [
        ("Qwen 0.5B", 0.0029/0.0236, 1),
        ("DeepSeek 1.5B", 0.1887/0.4612, 1),
        ("Mistral 7B", 0.0545/0.1381, 3),
        ("Qwen 7B", 0.1794/0.1678, 6),
    ]
    for label, ratio, tc in agg_points:
        x = np.log10(ratio)
        ax.scatter(x, tc, color=colors_model[label], marker="D", s=200, zorder=6,
                   edgecolors="black", linewidth=1.5, alpha=0.8)
        ax.annotate(f"{label} (aggregate)", (x, tc), fontsize=8, fontweight="bold",
                    ha="left", va="bottom", xytext=(6, 6), textcoords="offset points")
        plot_x.append(x)
        plot_y.append(tc)

    # Regression on all points
    px, py = np.array(plot_x), np.array(plot_y)
    finite = np.isfinite(px) & np.isfinite(py)
    if finite.sum() > 2:
        sl, ic, rv, pv, _ = scipy_stats.linregress(px[finite], py[finite])
        r2 = rv**2
        rho, rho_p = scipy_stats.spearmanr(px[finite], py[finite])
        xl = np.linspace(px[finite].min(), px[finite].max(), 100)
        ax.plot(xl, sl * xl + ic, "k--", linewidth=2, label=f"OLS: R²={r2:.3f}, ρ={rho:.3f} (p={rho_p:.3f})")
        print(f"\n  Phase D regression: R2={r2:.4f}, Spearman rho={rho:.4f} (p={rho_p:.2e})")
    else:
        r2, rho, rho_p = float("nan"), float("nan"), float("nan")

    ax.axvline(0, color="red", linestyle="--", alpha=0.4, label="α/β = 1 (balanced)")
    ax.set_xlabel("log₁₀(α/β)  ← corruption-dominant | repair-dominant →", fontsize=12)
    ax.set_ylabel("Boundary T_c  (last positive→negative μ_t crossing)", fontsize=12)
    ax.set_title("α/β Ratio Predicts Boundary Location\n(◆ = aggregate model-level, ○/□/△ = difficulty strata)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_d / "alpha_beta_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved α/β scatter")

    pd.DataFrame([{"R_squared": r2, "Spearman_rho": rho, "Spearman_p": rho_p}]).to_csv(
        out_d / "regression_summary.csv", index=False
    )

    # ── Trajectory type summary ──────────────────────────────────────────
    print("\n[7/7] Generating trajectory type summary...")
    traj_summary_rows = []
    for model in RUNS:
        traj = all_traj[model]
        total = len(traj)
        for tt in ["repair", "corruption", "persistent_correct", "persistent_wrong"]:
            count = (traj["trajectory_type"] == tt).sum()
            traj_summary_rows.append({
                "model": model, "trajectory_type": tt,
                "count": count, "pct": count / total * 100 if total > 0 else 0,
            })
    traj_summary = pd.DataFrame(traj_summary_rows)
    traj_summary.to_csv(out_c / "trajectory_type_summary.csv", index=False)

    print("\n  Trajectory type distribution:")
    pivot = traj_summary.pivot(index="model", columns="trajectory_type", values="pct").round(1)
    print(pivot.to_string())

    print("\n" + "=" * 70)
    print("ALL PHASES COMPLETE (A, B/refined, C, D)")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  Phase A: {out_a}")
    print(f"  Phase C: {out_c}")
    print(f"  Phase D: {out_d}")

    # Print the KEY finding
    print("\n" + "=" * 70)
    print("KEY FINDING: Qwen 7B medium stratum")
    print("=" * 70)
    qwen_med = stratum_summary[(stratum_summary["model"] == "Qwen 7B") & (stratum_summary["difficulty"] == "medium")]
    if len(qwen_med) > 0:
        row = qwen_med.iloc[0]
        print(f"  194 problems, step-1 acc = {row['step1_acc']:.3f}, peak = {row['peak_acc']:.3f} at step {row['peak_step']}")
        print(f"  α/β = {row['alpha_beta_ratio']:.3f}, T_c_first = {row['T_c_first']}, T_c_late = {row['T_c_late']}")
        print(f"  → Improvement of {row['delta_peak_step1']*100:.1f}pp confirms substantial repair window")

    print("\n  Mistral 7B medium stratum:")
    mis_med = stratum_summary[(stratum_summary["model"] == "Mistral 7B") & (stratum_summary["difficulty"] == "medium")]
    if len(mis_med) > 0:
        row = mis_med.iloc[0]
        print(f"  136 problems, step-1 acc = {row['step1_acc']:.3f}, peak = {row['peak_acc']:.3f} at step {row['peak_step']}")
        print(f"  α/β = {row['alpha_beta_ratio']:.3f}, T_c_first = {row['T_c_first']}, T_c_late = {row['T_c_late']}")
        print(f"  → Improvement of {row['delta_peak_step1']*100:.1f}pp — much smaller repair window")


if __name__ == "__main__":
    main()
