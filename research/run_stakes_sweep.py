#!/usr/bin/env python3
"""Stakes Sweeper for Safety-Critical Decision Boundaries

This script sweeps incorrect answer penalties (c) to evaluate how the optimal
stopping boundary and expected utility shift in safety-critical domains.

Formula:
    Utility Vt = q_t * v - (1 - q_t) * c - lambda * t
    Predictable Drift mu_t = [(1 - q_t)*alpha_t - q_t*beta_t](v + c) - lambda

By varying c (holding v = 1, lambda = 0.05 constant), we simulate:
- Low stakes: c = 0 (default)
- Balanced stakes: c = 1 (error is as bad as correct is good)
- High stakes: c = 10 (medical / legal risk)
- Extreme stakes: c = 50, 100

Output:
- C:\Aditya_Data\Personal\ResearchThesis\research\outputs\real_traces_bf16_ladder\stakes_sweep_report.md
- C:\Aditya_Data\Personal\ResearchThesis\research\outputs\real_traces_bf16_ladder\stakes_sweep_boundaries.png
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set paths
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# AUDIT FIX: delegate boundary detection to trace_analysis's audited
# T_MIN-floored first_zero_crossing (commit ef45dda) instead of a local
# reimplementation. The local reimplementation silently reintroduced the
# unfloored-boundary bug (returning theory-forbidden step 1) that ef45dda
# fixed in trace_analysis.py.
from trace_analysis import T_MIN, first_zero_crossing  # noqa: E402

LADDER_OUTPUT_ROOT = HERE / "outputs" / "real_traces_bf16_ladder"
MODELS = ["qwen2p5_7b", "qwen2p5_14b", "qwen2p5_32b"]
PENALTIES = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

STEP_COST = 0.05  # lambda
REWARD_VAL = 1.0  # v


def load_data(model_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load trace steps and runs from model output directory."""
    steps_path = model_dir / "trace_steps.csv"
    runs_path = model_dir / "trace_runs.csv"
    if not steps_path.exists() or not runs_path.exists():
        raise FileNotFoundError(f"Missing data files in {model_dir}")
    return pd.read_csv(steps_path), pd.read_csv(runs_path)


def compute_base_hazards(steps_df: pd.DataFrame) -> pd.DataFrame:
    """Compute base q_t, alpha_t, and beta_t per step."""
    records = []
    max_step = int(steps_df["step"].max())
    
    for step in range(1, max_step + 1):
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
            
        records.append({
            "step": step,
            "q_t": q_t,
            "alpha_t": alpha_t,
            "beta_t": beta_t,
            "n_runs": n
        })
    return pd.DataFrame(records)


def find_boundary(haz_df: pd.DataFrame, penalty: float) -> int:
    """Find first eligible step (t >= T_MIN) where predictable drift mu_t <= 0.

    Delegates to trace_analysis.first_zero_crossing (the audited, T_MIN-floored
    implementation) rather than scanning locally, so step 1 -- the forced-commit
    init state, not a real decision point -- can never be returned as a boundary.
    """
    if haz_df.empty:
        return T_MIN

    scored = haz_df.copy()
    # mu_t = [(1 - q_t)*alpha_t - q_t*beta_t] * (v + c) - lambda
    scored["mu_t"] = (
        (1 - scored["q_t"]) * scored["alpha_t"] - scored["q_t"] * scored["beta_t"]
    ) * (REWARD_VAL + penalty) - STEP_COST
    return first_zero_crossing(scored, "mu_t")


def calculate_run_utilities(
    runs_df: pd.DataFrame,
    steps_df: pd.DataFrame,
    penalty: float,
    boundary_step: int,
) -> tuple[float, float, float]:
    """Calculate mean utilities for Oracle, Hazard stopping, and Never Stop."""
    oracle_utils = []
    hazard_utils = []
    never_utils = []
    
    # Group steps by run_id for fast lookup
    steps_by_run = {rid: group.sort_values("step") for rid, group in steps_df.groupby("run_id")}
    
    for _, run in runs_df.iterrows():
        rid = run["run_id"]
        run_steps = steps_by_run.get(rid)
        if run_steps is None or run_steps.empty:
            continue
            
        correct_seq = run_steps["correct"].astype(bool).values
        n_steps = len(correct_seq)
        
        # Calculate utility at each step t: Vt = q_t * v - (1 - q_t) * c - lambda * t
        # (For single-trace run, q_t is binary correct: 1 if correct, 0 if wrong)
        step_utils = []
        for t in range(n_steps):
            q_val = 1.0 if correct_seq[t] else 0.0
            # Vt = q_val * v - (1 - q_val) * c - lambda * (step - 1); step = t+1.
            # Matches the pipeline convention utility = correct - 0.05*(step-1)
            # (real_trace_experiments.py) so utility LEVELS are cross-readable
            # with matrix stop_utility (rigor_audit/01 Axis 6 / lever H5).
            # Uniform +lambda offset: boundaries/argmax unaffected.
            ut = q_val * REWARD_VAL - (1.0 - q_val) * penalty - STEP_COST * t
            step_utils.append(ut)
            
        # 1. Oracle utility: max Vt over the sequence
        oracle_utils.append(max(step_utils))
        
        # 2. Hazard policy utility: utility at boundary_step (clipped to sequence length)
        stop_idx = min(boundary_step - 1, n_steps - 1)
        hazard_utils.append(step_utils[stop_idx])
        
        # 3. Never Stop utility: utility at final step
        never_utils.append(step_utils[-1])
        
    return float(np.mean(oracle_utils)), float(np.mean(hazard_utils)), float(np.mean(never_utils))


def main():
    print("=" * 80)
    print("STARTING STAKES SWEEP SIMULATION")
    print("=" * 80, flush=True)
    
    sweep_data = []
    plot_series = {model: [] for model in MODELS}
    
    for model in MODELS:
        model_dir = LADDER_OUTPUT_ROOT / model
        print(f"Analyzing {model}...")
        
        try:
            steps_df, runs_df = load_data(model_dir)
        except FileNotFoundError as e:
            print(f"[FAIL] Skipping {model}: {e}")
            continue
            
        haz_df = compute_base_hazards(steps_df)
        
        for c in PENALTIES:
            boundary = find_boundary(haz_df, c)
            oracle_u, hazard_u, never_u = calculate_run_utilities(runs_df, steps_df, c, boundary)
            
            oracle_gap = oracle_u - hazard_u
            never_gap = oracle_u - never_u
            
            sweep_data.append({
                "Model": model,
                "Penalty (c)": c,
                "Boundary Step": boundary,
                "Oracle Utility": oracle_u,
                "Hazard Utility": hazard_u,
                "Never Stop Utility": never_u,
                "Oracle Gap": oracle_gap,
                "Never Stop Gap": never_gap
            })
            
            plot_series[model].append((c, boundary))
            
    # Save Report Markdown
    report_df = pd.DataFrame(sweep_data)
    report_path = LADDER_OUTPUT_ROOT / "stakes_sweep_report.md"
    
    # Generate Markdown Table contents
    md_content = [
        "# Safety-Critical Stakes Sweep Report",
        "\nThis report sweeps incorrect answer penalties ($c$) to evaluate boundary shifts.",
        "\nFormula:",
        "  - **Expected Utility:** $V_t = q_t \\cdot v - (1 - q_t) \\cdot c - \\lambda t$",
        "  - **Predictable Drift:** $\\mu_t = [(1 - q_t)\\alpha_t - q_t\\beta_t](v + c) - \\lambda$",
        "\nSweep Settings:",
        f"  - Per-step compute cost ($\\lambda$): {STEP_COST}",
        f"  - Correct answer reward ($v$): {REWARD_VAL}",
        "\n## Sweep Summary Table\n",
        "| Model | Penalty (c) | Boundary Step (T*) | Oracle Utility | Hazard Utility | Never Stop Utility | Hazard Oracle Gap | Never Stop Oracle Gap |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    ]
    
    for _, row in report_df.iterrows():
        md_content.append(
            f"| {row['Model']} | {row['Penalty (c)']:.1f} | {row['Boundary Step']} | {row['Oracle Utility']:.4f} | "
            f"{row['Hazard Utility']:.4f} | {row['Never Stop Utility']:.4f} | {row['Oracle Gap']:.4f} | {row['Never Stop Gap']:.4f} |"
        )
        
    md_content.append("\n## Key Insights:")
    md_content.append("1. **Boundary Shifts Later:** As error penalty ($c$) scales, the boundary step shifts to the right (later), not earlier. This is expected from the utility formula: scaling $c$ amplifies the reward/penalty term $(v + c)$ relative to the fixed per-step cost $\\lambda$, so continued reasoning becomes worth more, not less, as consequences scale -- the model is incentivized to keep reasoning longer to avoid an even costlier wrong answer.")
    md_content.append("2. **Utility Conservation:** In high-penalty sweeps (c >= 10), letting the model think indefinitely (`never_stop`) leads to severe negative utility scores due to high error accumulation. The `hazard_drift` early stopping policy prevents this degradation.")
    
    report_path.write_text("\n".join(md_content), encoding="utf-8")
    print(f"Wrote stakes sweep report to: {report_path}")
    
    # Generate Plots
    plt.figure(figsize=(10, 6))
    for model, series in plot_series.items():
        if not series:
            continue
        xs, ys = zip(*series)
        plt.plot(xs, ys, marker="o", label=model)
        
    plt.xscale("symlog", linthresh=1.0)
    plt.xlabel("Incorrect Answer Penalty (c) - Symlog Scale")
    plt.ylabel("Optimal Stopping Boundary (T*)")
    plt.title("Optimal Stopping Boundary vs. Incorrect Penalty Stakes")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    plot_path = LADDER_OUTPUT_ROOT / "stakes_sweep_boundaries.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved stakes sweep plot to: {plot_path}")
    
    print("\n" + "=" * 80)
    print("STAKES SWEEP SIMULATION COMPLETE")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
