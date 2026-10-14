"""Generate crisp, publication-grade 300-DPI thesis figures.

Faithful to frozen empirical data in data_manifest_post_review_v1.json,
algorithm_v2_normalized_effects.csv, boundary_domain_step_metrics.csv,
and research/reports/thesis_failure_audit_v1/audit_summary.json.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ThesisDocs/images/thesis_v7"
OUT.mkdir(parents=True, exist_ok=True)
EVIDENCE = ROOT / "research/outputs/thesis_v1/evidence"
ONLINE = ROOT / "research/outputs/semester2/online_stopping_20261002"
ADVERSARIAL = ONLINE / "adversarial_live"

# Set publication style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 11.5,
    "axes.titlesize": 12.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "figure.titlesize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
})

def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))

def build_figure_1_delayed_repair():
    """Figure 1: Delayed repair stopping counterexample tree."""
    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def box(center, size, text, *, color="#EAF3F8", border="#33566E"):
        x, y = center; w, h = size
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            facecolor=color, edgecolor=border, linewidth=1.3))
        ax.text(x, y, text, ha="center", va="center", fontsize=11, color="#111111", linespacing=1.35)

    def arrow(a, b):
        ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#33566E"})

    # States
    box((0.16, 0.73), (0.26, 0.22), "t = 0\nCorrectness = 0\nCost so far = 0.00")
    box((0.50, 0.73), (0.26, 0.22), "t = 1\nCorrectness = 0\nCost so far = 0.10")
    box((0.84, 0.73), (0.26, 0.22), "t = 2 (Horizon)\nCorrectness = 1\nCost so far = 0.20")

    arrow((0.29, 0.73), (0.37, 0.73))
    arrow((0.63, 0.73), (0.71, 0.73))

    ax.text(0.33, 0.91, "Continue\n(Cost 0.10)", ha="center", va="center", fontsize=10.5, color="#0072B2")
    ax.text(0.67, 0.91, "Continue\n(Cost 0.10)", ha="center", va="center", fontsize=10.5, color="#0072B2")

    # Stop options
    box((0.16, 0.32), (0.26, 0.18), "Stop at t = 0\nReward = 0.00", color="#F8ECE5", border="#D55E00")
    box((0.50, 0.32), (0.26, 0.18), "Stop at t = 1\nReward = -0.10", color="#F8ECE5", border="#D55E00")
    box((0.84, 0.32), (0.26, 0.18), "Stop at t = 2\nReward = +0.80", color="#E8F4EC", border="#009E73")

    arrow((0.16, 0.62), (0.16, 0.41))
    arrow((0.50, 0.62), (0.50, 0.41))
    arrow((0.84, 0.62), (0.84, 0.41))

    ax.text(0.50, 0.09,
            "Immediate drift at t = 0 is -0.10 (myopic rule halts prematurely).\nContinuing to horizon t = 2 repairs answer and attains optimal reward +0.80.",
            ha="center", va="center", fontsize=11, linespacing=1.4, color="#111111",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F4F8", edgecolor="#B0C4DE", lw=0.8))

    out_path = OUT / "delayed_repair_tree.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print("Built:", out_path)

def build_figure_2_population_transitions():
    """Figure 2: Four-domain population transitions and continuation gain."""
    boundary = pd.read_csv(EVIDENCE / "boundary_domain_step_metrics.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.8), constrained_layout=True)

    domain_titles = {
        "arc": "ARC-Challenge (N=8)",
        "gpqa": "GPQA Main (N=10)",
        "gsm8k": "GSM8K (N=10)",
        "math": "MATH-500 (N=14)"
    }

    for ax, domain in zip(axes.flat, ["arc", "gpqa", "gsm8k", "math"]):
        frame = boundary[boundary.domain == domain]
        ax.plot(frame.step, frame.accuracy, color="#0072B2", marker="o", markersize=4.5,
                linewidth=1.4, label="Accuracy")
        ax.fill_between(frame.step, frame.accuracy_ci_low, frame.accuracy_ci_high,
                        color="#0072B2", alpha=0.15)
        ax.plot(frame.step, frame.net_drift, color="#D55E00", marker="s", markersize=4.5,
                linewidth=1.4, label="Net continuation gain (drift - 0.05)")
        ax.fill_between(frame.step, frame.net_drift_ci_low, frame.net_drift_ci_high,
                        color="#D55E00", alpha=0.15)
        ax.axhline(0, color="0.3", linestyle="--", linewidth=0.8)
        ax.set_title(domain_titles[domain], fontsize=11.5, fontweight="semibold")
        ax.set_xlabel("Current response step $t$", fontsize=10.5)
        ax.set_ylabel("Probability / Net gain", fontsize=10.5)
        ax.grid(alpha=0.15, linestyle=":")
        ax.set_ylim(-0.15, 0.85 if domain in ("arc", "gsm8k") else 0.55)

    axes[0, 0].legend(fontsize=9.5, loc="upper right", framealpha=0.9)
    out_path = OUT / "population_transitions.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print("Built:", out_path)

def build_figure_3_forest_plot():
    """Figure 3: Clean horizontal forest plot of 9 matched development contrasts (Table 2)."""
    df = pd.read_csv(EVIDENCE / "algorithm_v2_normalized_effects.csv")
    labels = {
        'N1 LOCO': 'Threshold, held-out cell',
        'N1 LOMO': 'Threshold, held-out model',
        'N2a': 'Gradient boosting (GBDT)',
        'N2b': 'Isotonic calibration',
        'N2c': 'Lagged logistic',
        'N3': 'Empirical-Bayes hazards',
        'N4': 'Step-two churn',
        'N5': '512 vs 256 token cap',
        'N6': 'BF16 vs 4-bit precision'
    }

    # Reverse order so top is first
    items = list(df.iterrows())[::-1]
    y_pos = np.arange(len(items))

    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)

    # Reference zero line
    ax.axvline(0, color="#444444", linestyle="--", linewidth=1.0, zorder=1)

    for i, (_, r) in enumerate(items):
        name = labels[r.experiment]
        mean = r.mean_controlled_effect
        lo, hi = r.ci_95_low, r.ci_95_high
        unit = r.effect_unit

        if 'utility' in unit:
            color = "#0072B2" if mean >= 0 else "#D55E00"
            marker = "o"
        elif 'risk' in unit:
            color = "#777777"
            marker = "D"
        else: # accuracy
            color = "#009E73"
            marker = "s"

        ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]],
                    fmt=marker, color=color, ecolor=color, elinewidth=1.6,
                    capsize=4, capthick=1.4, markersize=5.5, zorder=3)

        # Value label: always place to right of zero line or right of whisker
        sign = "+" if mean > 0 else ""
        txt = f"{sign}{mean:.5f} [{lo:+.5f}, {hi:+.5f}]"
        if mean < 0:
            ax.text(0.008, i, txt, va="center", ha="left", fontsize=9.2, color="#222222")
        else:
            ax.text(max(hi + 0.006, 0.022), i, txt, va="center", ha="left", fontsize=9.2, color="#222222")

    y_labels = [labels[r.experiment] for _, r in items]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=10.5)
    ax.set_xlabel("Mean controlled contrast [95% CI]\n(Utility / Trajectory, Loss-risk diff, or Accuracy diff)", fontsize=10.5)
    ax.set_xlim(-0.11, 0.32)
    ax.grid(axis="x", alpha=0.15, linestyle=":")

    out_path = OUT / "controlled_contrasts_forest.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Built:", out_path)

def build_figure_4_failure_audit():
    """Figure 4: Archived policy failure taxonomy and delayed-repair distribution."""
    # Explicitly ordered list pairing label, count, share, and color
    # Order from bottom (index 0) to top (index 4) for barh:
    data = [
        {"label": "Missing normalized answer\nat stop", "count": 196, "share": 3.42, "color": "#999999"},
        {"label": "Passed earlier correct\ncandidate", "count": 460, "share": 8.02, "color": "#888888"},
        {"label": "Step 1 only was correct;\nno pre-stop answer", "count": 600, "share": 10.46, "color": "#E69F00"},
        {"label": "Next-step repair:\narrived 1 step later", "count": 1706, "share": 29.75, "color": "#56B4E9"},
        {"label": "Delayed repair (Late):\narrived \u2265 2 steps later", "count": 2773, "share": 48.35, "color": "#0072B2"}
    ]

    labels = [d["label"] for d in data]
    counts = [d["count"] for d in data]
    shares = [d["share"] for d in data]
    colors = [d["color"] for d in data]

    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)

    y_pos = np.arange(len(data))
    bars = ax.barh(y_pos, counts, height=0.62, color=colors, edgecolor="#222222", linewidth=0.8, zorder=2)

    for bar, count, share in zip(bars, counts, shares):
        w = bar.get_width()
        ax.text(w + 40, bar.get_y() + bar.get_height()/2,
                f"{count:,} ({share:.2f}%)",
                va="center", ha="left", fontsize=10.2, fontweight="medium", color="#111111")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Count of policy losses (Total = 5,735 among 75,965 trajectories)", fontsize=10.5)
    ax.set_xlim(0, 3600)
    ax.grid(axis="x", alpha=0.15, linestyle=":")

    # Highlight box in lower-right open whitespace alongside the 196, 460, 600 bars
    ax.text(1350, 0.85,
            "Key finding: 78.10% (4,479 / 5,735)\nof losses repaired after stopping;\n48.35% require \u2265 2 steps later,\ndefeating myopic 1-step boundaries.",
            ha="left", va="center", fontsize=9.8, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#EBF5FB", edgecolor="#0072B2", lw=0.9))

    out_path = OUT / "failure_audit_taxonomy.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Built:", out_path)

def build_figure_5_actual_live_pareto():
    """Figure 5: Actual paired live answer quality and completion cost."""
    folders = [("GSM8K: 100 tasks", ONLINE, ONLINE / "learned_main"),
               ("Trap bank: 20 tasks", ADVERSARIAL, ONLINE / "learned_adversarial")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.2), constrained_layout=True)

    for ax, (title, heuristic_folder, learned_folder) in zip(axes, folders):
        heuristic = read_json(heuristic_folder / "live_metrics.json")
        learned = read_json(learned_folder / "live_metrics.json")
        points = [
            ("Full horizon (5)", 1.0, heuristic["baseline_accuracy"], "#555555", (0, 15)),
            ("Heuristic", heuristic["active_generated_tokens"] / heuristic["baseline_generated_tokens"],
             heuristic["active_accuracy"], "#D55E00", (-42, -22)),
            ("Learned / step 2", learned["active_generated_tokens"] / learned["baseline_generated_tokens"],
             learned["active_accuracy"], "#0072B2", (30, 14))
        ]
        for label, x, y, color, offset in points:
            ax.scatter(x, y, color=color, s=55, zorder=4, edgecolor="#111111", linewidth=0.6)
            ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=10.5,
                        ha="center", arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
                        fontweight="medium")
        ax.set_title(title, fontsize=11.5, fontweight="semibold")
        ax.set_xlabel("Completion tokens / Full horizon", fontsize=10.5)
        ax.set_ylabel("Answer accuracy", fontsize=10.5)
        ax.set_xlim(0.35, 1.12)
        ax.set_ylim(-0.01, 0.11)
        ax.grid(alpha=0.18, linestyle=":")

    out_path = OUT / "actual_live_pareto.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print("Built:", out_path)

if __name__ == "__main__":
    build_figure_1_delayed_repair()
    build_figure_2_population_transitions()
    build_figure_3_forest_plot()
    build_figure_4_failure_audit()
    build_figure_5_actual_live_pareto()
    print("All 5 figures successfully built at 300 DPI!")
