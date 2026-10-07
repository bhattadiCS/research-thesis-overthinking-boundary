"""Build the formal thesis with checked math and frozen evidence.

Install rendering dependencies with
  npm install --prefix tmp/thesis_pdf_runtime --save-exact katex@0.19.0
and run with a Python environment containing markdown, pandas, matplotlib,
PyMuPDF, and reportlab. Chrome is used only for local HTML-to-PDF printing.
Every TeX expression must parse; incomplete live results fail the default build.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
from pathlib import Path

import fitz
import markdown
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "ThesisDocs"
EVIDENCE = ROOT / "research/outputs/thesis_v1/evidence"
ONLINE = ROOT / "research/outputs/semester2/online_stopping_20261002"
ADVERSARIAL = ONLINE / "adversarial_live"
PREFIX_MODEL = ROOT / "research/outputs/semester2/prefix_model_v1"
IMAGES = DOCS / "images/thesis_v2"
WORK = ROOT / "tmp/pdfs/formal_thesis/digital"
OUTPUT = WORK / "source.pdf"
LEFT = 72
RIGHT = 540
SUBMISSION_DATE = "October 2026"
LIST_ENTRY_GAP = 8  # Front lists may be single-spaced; keep all entries legible.
CHAPTERS = ["chapter1_intro.md", "chapter2_theory.md", "chapter3_methodology.md",
            "chapter4_empirical.md", "chapter5_online.md", "chapter6_discussion.md"]
FONT = Path("C:/Windows/Fonts/arial.ttf")
TITLE = "Cost aware stopping boundaries in reasoning language models"
ABSTRACT = (
    "Additional reasoning can repair an incorrect answer, replace a correct answer, or consume computation without sufficient improvement. "
    "This thesis formulates response-level stopping as a finite-horizon decision based on observable prefixes and hidden correctness. "
    "It derives the binary repair-corruption drift identity, applies the standard Bellman/Snell optimal stopping construction, "
    "gives a sufficient persistence condition for a drift-sign rule, and constructs exact counterexamples to unconditional myopic optimality. "
    "The experimental record separates a variable-horizon matrix from a standardized corpus of 144,440 saved rows, 28,888 trajectories, and 2,948 tasks. "
    "Recomputed cluster-based tables show model- and domain-dependent continuation value and matched effects of estimator, token-cap, and precision changes. "
    "The historical stacked ROC-AUC of 0.955156 is retained as a retrospective non-nested diagnostic rather than an online performance guarantee. "
    "A fitted prefix controller enforces a two-step floor and prevents future generation after stopping. "
    "In a newly executed evaluation on 100 GSM8K questions, its actual arm saves 56.51 percent of completion tokens with 7 correct answers versus 6 at the full horizon. "
    "On 20 traps it saves 52.11 percent, with one correct answer in each arm. "
    "Every learned run stops at step two; adaptive benefit and accuracy noninferiority remain unestablished. "
    "The results support protocol-specific, cost-sensitive stopping experiments while identifying limits of myopic rules, probability calibration, and accuracy preservation."
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers: list[str], rows: list[list], caption: str) -> str:
    def cell(value) -> str:
        return str(value).replace("|", " / ").replace("\n", " ")
    return (f"**{caption}**\n\n| " + " | ".join(headers) + " |\n| " +
            " | ".join(["---"] * len(headers)) + " |\n" +
            "\n".join("| " + " | ".join(map(cell, row)) + " |" for row in rows))


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def build_figures(boundary: pd.DataFrame, detector: pd.DataFrame, replay: pd.DataFrame | None) -> None:
    # At the narrower six-inch print width, 13pt figure labels retain an
    # effective size above the library's 10pt minimum (13 * 6 / 7.6).
    plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans", "axes.spines.top": False,
                        "axes.spines.right": False, "savefig.dpi": 240})
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.3), constrained_layout=True)
    for ax, domain in zip(axes.flat, ["arc", "gpqa", "gsm8k", "math"]):
        frame = boundary[boundary.domain == domain]
        ax.plot(frame.step, frame.accuracy, color="#0072B2", marker="o", label="Accuracy")
        ax.fill_between(frame.step, frame.accuracy_ci_low, frame.accuracy_ci_high,
                        color="#0072B2", alpha=.16)
        ax.plot(frame.step, frame.net_drift, color="#D55E00", marker="s", label="Net gain")
        ax.fill_between(frame.step, frame.net_drift_ci_low, frame.net_drift_ci_high,
                        color="#D55E00", alpha=.16)
        ax.axhline(0, color="0.3", linewidth=.7)
        ax.set(title=domain.upper(), xlabel="Current response step", ylabel="Accuracy or net gain")
    axes[0, 0].legend(fontsize=13)
    fig.savefig(IMAGES / "population_transitions.png")
    plt.close(fig)
    chosen = detector.iloc[:7].copy()
    names = [x.replace("Causal ", "").replace("Fourier neural operator", "Fourier operator") for x in chosen.configuration]
    fig, ax = plt.subplots(figsize=(7.6, 4.7), constrained_layout=True)
    ax.plot(chosen.micro_oof_auc, range(len(chosen)), "o", color="#0072B2", label="Micro")
    ax.plot(chosen.domain_macro_auc, range(len(chosen)), "s", color="#009E73", label="Domain macro")
    ax.plot(chosen.worst_domain_auc, range(len(chosen)), "^", color="#D55E00", label="Worst domain")
    ax.set(yticks=range(len(names)), yticklabels=names, xlabel="Stored task-grouped ROC-AUC", xlim=(.5, 1))
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=3, frameon=False)
    fig.savefig(IMAGES / "causal_detector_domains.png")
    plt.close(fig)
    if replay is not None:
        fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
        for _, row in replay.iterrows():
            x = row.active_generated_tokens / row.baseline_generated_tokens
            ax.scatter(x, row.active_accuracy, color="#0072B2")
            offset = {"confidence_80": (-36, 32), "confidence_90": (28, 18),
                      "confidence_95": (-15, -30), "never": (-6, 12)}.get(row.policy, (8, 9))
            ax.annotate(row.policy.replace("confidence_", "conf ").replace("fixed_", "step "),
                        (x, row.active_accuracy), xytext=offset, textcoords="offset points", fontsize=13,
                        ha="right" if row.policy in ("confidence_80", "confidence_95", "never") else "left",
                        arrowprops={"arrowstyle": "-", "color": "#0072B2", "lw": .7})
        ax.margins(x=.07, y=.16)
        ax.set(xlabel="Completion tokens / full-horizon completion tokens", ylabel="Replay answer accuracy")
        ax.grid(alpha=.2)
        fig.savefig(IMAGES / "replay_pareto.png")
        plt.close(fig)
    folders = [("GSM8K: 100 tasks", ONLINE, ONLINE / "learned_main"),
               ("Trap bank: 20 tasks", ADVERSARIAL, ONLINE / "learned_adversarial")]
    if all((folder / "live_metrics.json").is_file() for _, h, l in folders for folder in (h, l)):
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.7), constrained_layout=True)
        for ax, (title, heuristic_folder, learned_folder) in zip(axes, folders):
            heuristic = read_json(heuristic_folder / "live_metrics.json")
            learned = read_json(learned_folder / "live_metrics.json")
            points = [("Full horizon", 1.0, heuristic["baseline_accuracy"], "#666666", (0, 19)),
                      ("Heuristic", heuristic["active_generated_tokens"] / heuristic["baseline_generated_tokens"],
                       heuristic["active_accuracy"], "#D55E00", (-44, -25)),
                      ("Learned / step 2", learned["active_generated_tokens"] / learned["baseline_generated_tokens"],
                       learned["active_accuracy"], "#0072B2", (28, 15))]
            for label, x, y, color, offset in points:
                ax.scatter(x, y, color=color, s=45)
                ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=13,
                            ha="center", arrowprops={"arrowstyle": "-", "color": color, "lw": .7})
            ax.set(title=title, xlabel="Completion tokens / full horizon", ylabel="Answer accuracy",
                   xlim=(.35, 1.12), ylim=(.0, .11))
            ax.grid(alpha=.2)
        fig.savefig(IMAGES / "actual_live_pareto.png")
        plt.close(fig)


def evidence_inserts(allow_pending: bool, *, preserve_figures: bool = False) -> dict[str, str]:
    boundary = pd.read_csv(EVIDENCE / "boundary_domain_step_metrics.csv")
    controlled = pd.read_csv(EVIDENCE / "algorithm_v2_normalized_effects.csv")
    detector = pd.read_csv(EVIDENCE / "tournament_balanced_summary.csv")
    failures = read_json(ROOT / "research/reports/thesis_failure_audit_v1/audit_summary.json")
    prefix_evaluation = read_json(PREFIX_MODEL / "evaluation.json")
    replay = pd.read_csv(ONLINE / "replay_pareto.csv") if (ONLINE / "replay_pareto.csv").exists() else None
    if preserve_figures:
        # An editorial revision reuses the reviewed scientific figures. It must
        # not regenerate the historical assets or silently change their bytes.
        for entry in read_json(DOCS / "formal/source_integrity_v4.json")["figure_files"]:
            if sha(ROOT / entry["path"]) != entry["sha256"]:
                raise ValueError(f"Reviewed figure changed: {entry['path']}")
    else:
        build_figures(boundary, detector, replay)
    counts = [["GSM8K", "train", "1,000", "8,064", "40,320"],
              ["MATH-500", "test", "500", "6,500", "32,500"],
              ["ARC-Challenge", "test", "1,000", "8,500", "42,500"],
              ["GPQA main", "train (inferred; request test)", "448", "5,824", "29,120"],
              ["Total", "four domains", "2,948", "28,888", "144,440"]]
    roster = {}
    for path in sorted((ROOT / "research/outputs/experiment_matrix").glob("*/metadata.json")):
        if not (path.parent / "detector_comparison_by_run.csv").exists():
            continue
        model = read_json(path)["model"]
        roster[model["alias"]] = [model["hf_name"], model["parameter_count"], "Matrix + detector"]
    roster["internlm3_8b_instruct"][2] = "Matrix only"
    roster["qwen_3p5_9b"] = ["Qwen/Qwen3.5-9B", "9B", "Detector only"]
    chosen = boundary[(boundary.domain == "gsm8k") & boundary.step.isin([2, 4])]
    rows = [[int(r.step), f"{r.accuracy:.4f}", f"{int(r.repair_events):,}/{int(r.repair_denominator):,}",
             f"{r.repair_hazard:.4f}", f"{int(r.corruption_events):,}/{int(r.corruption_denominator):,}",
             f"{r.corruption_hazard:.4f}", f"{r.net_drift:+.4f}",
             f"[{r.net_drift_ci_low:+.4f}, {r.net_drift_ci_high:+.4f}]"] for _, r in chosen.iterrows()]
    results = {
        "CORPUS_TABLE": table(["Domain", "Effective split", "Tasks", "Trajectories", "Rows"], counts, "Table 1. Standardized five-step corpus."),
        "MODEL_TABLE": table(["Model ID", "Recorded scale", "Membership"], list(roster.values()), "Table 2. Model configurations. Source: cell metadata."),
        "BOUNDARY_TABLE": table(["Step", "Accuracy", "Repair events / at risk<br>Probability", "Corruption events / at risk<br>Probability", "Net gain<br>95% interval"],
                                [[r[0], r[1], f"{r[2]}<br>{r[3]}", f"{r[4]}<br>{r[5]}", f"{r[6]}<br>{r[7]}"] for r in rows],
                                "Table 3. GSM8K transition panel; 19,500 trajectories and 500 task clusters per row. Event cells give the count and at-risk denominator above the conditional probability; the final column gives net gain above its 95% interval."),
        "CONTROLLED_TABLE": table(["Arm", "Units", "Matched effect", "95% interval", "Endpoint"],
            [[r.experiment, f"{int(r.n_units):,}", f"{r.mean_controlled_effect:+.5f}",
              f"[{r.ci_95_low:+.5f}, {r.ci_95_high:+.5f}]", r.effect_unit] for _, r in controlled.iterrows()],
            "Table 4. Matched effects. Estimator intervals resample 52 cells; systems intervals resample 500 tasks."),
        "DETECTOR_TABLE": table(["Configuration", "Micro AUC", "Task macro", "Domain macro", "Worst AUC", "Step utility"],
            [[r.configuration, f"{r.micro_oof_auc:.4f}", f"{r.task_macro_auc:.4f}", f"{r.domain_macro_auc:.4f}",
              f"{r.worst_domain_auc:.4f}", f"{r.micro_step_utility:.4f}"] for _, r in detector.iterrows()],
            "Table 5. Stored causal detectors; common task-held-out development evaluation."),
        "BOUNDARY_FIGURE": "![Four-domain population curves](images/thesis_v1/population_transitions.png)\n\n**Figure 1.** Population accuracy and net gain. Bands are task-cluster bootstrap intervals. The horizontal line marks zero gain, not zero accuracy.",
        "DETECTOR_FIGURE": "![Causal detector domain scores](images/thesis_v1/causal_detector_domains.png)\n\n**Figure 2.** Micro, domain-macro, and worst-domain ranking differ substantially. Stored grouped outputs, not live-run results.",
        "EVIDENCE_TABLE": table(["Claim or output", "Authoritative repository source"], [
            ["Tournament corpus", "data_manifest_v1.json and ultimate_tournament_manifest.json"],
            ["Boundary and matched effects", "research/outputs/thesis_v1/evidence/*.csv"],
            ["Proofs and exact finite checks", "research/mathematical_foundations.md; test_mathematical_foundations.py"],
            ["Controller execution", "online_stopping_controller.py; online_generation.py"],
            ["Latency and replay", "research/outputs/semester2/online_stopping_20261002/"],
            ["Adversarial bank", "research/adversarial_tasks_v1.jsonl; adversarial_gold_v1.jsonl"],
            ["Archived policy failures", "research/reports/thesis_failure_audit_v1/audit_summary.json"],
            ["Deployable prefix predictor", "research/outputs/semester2/prefix_model_v1/"],
            ["PDF authoring", "tools/build_master_thesis.py; tools/render_thesis_math.mjs"]], "Table 14. Claim evidence sources."),
        "FAILURE_TABLE": table(["Observed loss pattern", "Count", "Share of losses"],
            [[row["description"], f"{row['count']:,}", pct(row["share_of_losses"])] for row in failures["categories"]],
            "Table 6. Mutually exclusive archived-policy loss patterns; 5,735 utility losses among 75,965 trajectories."),
        "PREDICTOR_TABLE": table(["Target", "Held-out rows", "AUC", "Brier", "ECE (10 bins)"],
            [[name, item["calibrated"]["rows"], f"{item['calibrated']['auc']:.4f}",
              f"{item['calibrated']['brier']:.4f}", f"{item['calibrated']['ece_10_equal_width']:.4f}"]
             for name, item in prefix_evaluation["probabilities"].items()],
            "Table 10. Deployable probability heads on 276 archived held-out tasks; repeated rows are not independent trials."),
    }
    pending = "This experiment is still in progress. No completed result is claimed in this interim build."
    latency_path = ONLINE / "latency_summary.json"
    if latency_path.exists():
        lat = read_json(latency_path)
        groups = [("All measured decisions", lat.get("all", lat))]
        groups.extend(lat.get("policies", {}).items())
        results["LATENCY_TABLE"] = table(["Scope", "Decisions", "Median ms", "p99 ms", "Maximum ms"],
            [[name, item["decisions"], f"{item['median_ms']:.5f}", f"{item['p99_ms']:.5f}", f"{item['max_ms']:.5f}"]
             for name, item in groups], "Table 7. Controller-only latency on 100 saved math-question prefixes.")
    else:
        results["LATENCY_TABLE"] = pending
    for token, folder, label in [("LIVE_TABLE", ONLINE, "Table 8. Actual paired local generation on 100 GSM8K questions."),
                                 ("ADVERSARIAL_TABLE", ADVERSARIAL, "Table 9. Actual paired local generation on the 20-question trap bank."),
                                 ("LEARNED_TABLE", ONLINE / "learned_main", "Table 11. Actual trained-policy generation paired with the existing 100-question live baseline."),
                                 ("LEARNED_ADVERSARIAL_TABLE", ONLINE / "learned_adversarial", "Table 12. Actual trained-policy generation paired with the existing trap-bank live baseline.")]:
        path = folder / "live_metrics.json"
        if not path.exists():
            if not allow_pending:
                raise FileNotFoundError(f"Completed live result required: {path}")
            results[token] = f"**{label}**\n\n{pending}"
            continue
        live = read_json(path)
        metrics = [["Questions", live["problems_or_trajectories"]],
                   ["Full-horizon accuracy", pct(live["baseline_accuracy"])],
                   ["Stopped accuracy", pct(live["active_accuracy"])],
                   ["Accuracy difference", f"{100 * live['accuracy_delta']:+.2f} percentage points"],
                   ["Full-horizon completion tokens", f"{live['baseline_generated_tokens']:,}"],
                   ["Stopped completion tokens", f"{live['active_generated_tokens']:,}"],
                   ["Full-horizon prompt tokens", f"{live['baseline_prompt_tokens']:,}"],
                   ["Stopped prompt tokens", f"{live['active_prompt_tokens']:,}"],
                   ["Completion-token saving", pct(live["measured_completion_token_savings"])],
                   ["Mean stopped step", f"{live['mean_active_stop_step']:.2f}"],
                   ["Identical shared prefixes", live["shared_prefix_identical_problems"]]]
        uncertainty_path = folder / "live_uncertainty.json"
        if uncertainty_path.exists():
            uncertainty = read_json(uncertainty_path)
            lo, hi = uncertainty["accuracy_delta_conservative_exact_95ci"]
            metrics.append(["Conservative paired 95% accuracy interval", f"[{100*lo:+.2f}, {100*hi:+.2f}] percentage points"])
            lo, hi = uncertainty["completion_token_savings_cluster_bootstrap_95ci"]
            metrics.append(["Task-bootstrap saving interval", f"[{100*lo:.2f}%, {100*hi:.2f}%]"])
        results[token] = table(["Endpoint", "Measured value"], metrics, label)
    if replay is not None:
        results["PARETO_TABLE"] = table(["Policy", "Accuracy", "Completion tokens", "Saving", "Nondominated"],
            [[r.policy, pct(r.active_accuracy), f"{int(r.active_generated_tokens):,}",
              pct(r.measured_completion_token_savings), str(r.pareto_nondominated)] for _, r in replay.iterrows()],
            "Table 13. Frozen variants replayed on 1,500 saved trajectories. These are development comparisons.")
        results["PARETO_FIGURE"] = "![Replay accuracy cost comparison](images/thesis_v1/replay_pareto.png)\n\n**Figure 3.** Development replay accuracy against completion-token cost. Labels identify fixed-step and confidence variants; observed nondominance is sample-specific."
    else:
        results["PARETO_TABLE"] = results["PARETO_FIGURE"] = pending
    actual_figure = IMAGES / "actual_live_pareto.png"
    results["ACTUAL_PARETO_FIGURE"] = (
        "![Actual live accuracy and completion-token comparison](images/thesis_v1/actual_live_pareto.png)\n\n"
        "**Figure 4.** Actual generated arms on two development panels. Each learned arm reuses its panel's "
        "actually generated full-horizon baseline and stops at step two on every task. The plotted point "
        "therefore supplies no evidence of adaptation beyond a fixed-two-step budget. Point estimates "
        "omit uncertainty, which is reported in Tables 9, 10, 12 and 13.") if actual_figure.exists() else pending
    # Reserve Table 1 for the mathematical scope table, which precedes the
    # experimental evidence in the final manuscript.
    for key, value in results.items():
        results[key] = re.sub(r"Table (\d+)\.",
                             lambda match: f"Table {int(match[1]) + 1}.", value)
    return results


def sync_theory() -> None:
    source = (ROOT / "research/mathematical_foundations.md").read_text(encoding="utf-8")
    start = source.index("## 1.")
    end = source.index("## 9.")
    theory = source[start:end]
    theory = theory.replace(
        "All observations, executed peer calls, verifier outputs, and controller",
        "For empirical evaluation, correctness is the versioned domain grader "
        "$C_t=g_d(A_t,Y^*)\\in\\{0,1\\}$. Exact answer equality is the special "
        "case displayed above. The binary-reward arguments remain unchanged "
        "for this grading predicate. This notation identifies the measured "
        "endpoint; it does not certify semantic correctness of every stored label. "
        "Appendix E illustrates the information restrictions and delayed-repair counterexample.\n\n"
        "All observations, executed peer calls, verifier outputs, and controller")
    theory = theory.replace(
        "$\\mathcal B_t$ is the candidate set available at $t$.",
        "$\\mathcal B_t$ is the candidate set available at $t$. With the empirical "
        "grading predicate, the analogous immediate reward is "
        "$\\max_{a\\in\\mathcal B_t}\\mathbb E[g_d(a,Y^*)\\mid\\mathcal F_t]$.")
    # Reflow this wide display only in the manuscript. The canonical frozen
    # mathematical source and its historical manifest remain byte-identical.
    theory = theory.replace(
        "\\widetilde q_t=\\mathbb P(C_t=1\\mid Z_t),\\quad\n"
        "\\widetilde\\alpha_t=\\mathbb P(C_{t+1}=1\\mid C_t=0,Z_t),\\quad\n"
        "\\widetilde\\beta_t=\\mathbb P(C_{t+1}=0\\mid C_t=1,Z_t).",
        "\\begin{aligned}\n"
        "\\widetilde q_t&=\\mathbb P(C_t=1\\mid Z_t),\\\\\n"
        "\\widetilde\\alpha_t&=\\mathbb P(C_{t+1}=1\\mid C_t=0,Z_t),\\\\\n"
        "\\widetilde\\beta_t&=\\mathbb P(C_{t+1}=0\\mid C_t=1,Z_t).\n"
        "\\end{aligned}")
    theory = theory.replace(
        "| Mathematical object | Required information or assumptions | Defensible interpretation |",
        "**Table 1. Mathematical objects and assumptions.**\n\n"
        "| Mathematical object | Required information or assumptions | Defensible interpretation |")
    theory = re.sub(r"^## (\d+)\.\s*", lambda m: f"## 2.{m[1]} ", theory, flags=re.M)
    (DOCS / "chapters/chapter2_theory.md").write_text(
        "# Chapter 2 Mathematical formulation and stopping theory\n\n" + theory,
        encoding="utf-8", newline="\n")


def html_math(source: str, runtime: Path, key: str) -> str:
    expressions = []
    # Proof sources wrap inline TeX across physical lines. Those line breaks
    # are whitespace inside the same expression, not unmatched delimiters.
    pattern = re.compile(r"\$\$(.+?)\$\$|(?<!\\)\$([^$]+?)(?<!\\)\$", re.S)
    def protect(match: re.Match) -> str:
        index = len(expressions)
        expressions.append({"tex": (match[1] if match[1] is not None else match[2]).strip(),
                            "display": match[1] is not None})
        return f"MATHPLACEHOLDER{index}END"
    code_blocks = []
    def protect_code(match: re.Match) -> str:
        index = len(code_blocks)
        code_blocks.append(match[0])
        return f"CODEBLOCKPLACEHOLDER{index}END"
    prose = re.sub(r"```[^\n]*\n.*?```|`[^`\n]+`", protect_code, source, flags=re.S)
    protected = pattern.sub(protect, prose)
    if re.search(r"(?<!\\)\$", protected):
        raise ValueError(f"Unmatched math delimiter in {key}")
    input_path, rendered_path = WORK / f"{key}_math.json", WORK / f"{key}_math_rendered.json"
    input_path.write_text(json.dumps(expressions), encoding="utf-8")
    subprocess.run(["node", str(ROOT / "tools/render_thesis_math.mjs"), str(runtime),
                    str(input_path), str(rendered_path)], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    rendered = read_json(rendered_path)
    for index, value in enumerate(code_blocks):
        protected = protected.replace(f"CODEBLOCKPLACEHOLDER{index}END", value)
    content = markdown.markdown(protected, extensions=["tables", "fenced_code"])
    if len(re.findall(r"<table>", content)) != len(re.findall(r"<p><strong>Table \d+\.", content)):
        raise ValueError(f"Every manuscript table must have a numbered caption in {key}")
    # Keep each numbered caption with its table and keep these short evidence
    # tables intact. A repeated header alone does not prevent orphan captions.
    content = re.sub(r'(<p><strong>Table \d+\..*?</strong></p>\s*<table>.*?</table>)',
                     r'<div class="table-block">\1</div>', content, flags=re.S)
    content = re.sub(r'(<p><img\b[^>]*></p>\s*<p><strong>Figure \d+\..*?</p>)',
                     r'<div class="figure-block">\1</div>', content, flags=re.S)
    # Keep short proof conclusions together; equations can otherwise defeat
    # the browser's usual widow/orphan count at a page boundary.
    content = re.sub(r'<p>((?:(?!</?p>).)*?□)</p>',
                     lambda match: '<p class="proof-ending">' + match[1] + '</p>'
                     if len(re.sub(r'<[^>]+>', '', match[1]).split()) <= 80 else match[0],
                     content, flags=re.S)
    for index, value in enumerate(rendered):
        token = f"MATHPLACEHOLDER{index}END"
        if expressions[index]["display"]:
            pattern = rf"(<p>(?:(?!</?p>).)*?</p>)(\s*<p>{token}</p>)"
            content = re.sub(pattern, r'<div class="math-intro">\1\2</div>', content, flags=re.S)
            content = content.replace(token, value)
        else:
            # KaTeX permits breaks between its base spans. Bind punctuation
            # inside the final base so it cannot start the next line. Short
            # expressions also stay intact across lines and pages.
            def inline(match: re.Match) -> str:
                punctuation = match[1] or ""
                result = value
                if punctuation:
                    fragments = result.rsplit("</span>", 3)
                    if len(fragments) != 4:
                        raise ValueError(f"Unexpected inline KaTeX structure in {key}")
                    fragments[0] += '<span class="math-punctuation">' + punctuation + '</span>'
                    result = "</span>".join(fragments)
                if len(expressions[index]["tex"]) <= 80:
                    result = '<span class="math-inline-short">' + result + '</span>'
                return result
            content = re.sub(rf"{token}([.,;:!?])?", inline, content)
    return content


def print_chapter(source: str, name: str, chrome: Path, runtime: Path) -> Path:
    content = html_math(source, runtime, name)
    css_path = runtime / "node_modules/katex/dist/katex.min.css"
    css = """
    @page {size:letter; margin:1in 1in 1.35in LEFTMARGIN;}
    body {font-family:Arial,sans-serif; font-size:12pt; line-height:2; color:#111; margin:0;}
    p {margin:0 0 12pt;orphans:2;widows:2;} h1 {font-size:18pt;line-height:1.35;margin:0 0 25pt;break-after:avoid;break-before:page;}
    h1:first-child {break-before:auto;}
    h2 {font-size:14pt;line-height:1.4;margin:21pt 0 11pt;break-after:avoid;}
    h3 {font-size:12pt;line-height:1.4;margin:15pt 0 9pt;break-after:avoid;}
    table {font-size:10.1pt;line-height:1.35;border-collapse:collapse;width:100%;margin:12pt 0 18pt;break-inside:avoid;}
    .table-block {break-inside:avoid;}
    .figure-block {break-inside:avoid;}
    .math-intro {break-inside:avoid;}
    .proof-ending {break-inside:avoid;}
    .math-inline-short {white-space:nowrap;}
    .math-punctuation {font-family:Arial,sans-serif;font-size:12pt;}
    th {text-align:left;border-bottom:1pt solid #333;} td,th {padding:6pt 4pt;vertical-align:top;overflow-wrap:anywhere;}
    tr {break-inside:avoid;} td {border-bottom:.4pt solid #bbb;} thead {display:table-header-group;}
    code {font-size:10.1pt;overflow-wrap:anywhere;} pre {font-size:10.1pt;line-height:1.4;white-space:pre;}
    pre code {overflow-wrap:normal;}
    img {width:100%;height:auto;break-inside:avoid;} a {color:#222;overflow-wrap:anywhere;text-decoration:none;}
    .katex {font-size:1.03em;} .katex-display {margin:14pt 0;line-height:1.2;break-inside:avoid;}
    blockquote {margin:12pt 15pt;font-size:11pt;}
    """
    css = css.replace("LEFTMARGIN", f"{LEFT / 72:g}in")
    page = (f'<!doctype html><html><head><meta charset="utf-8"><base href="{DOCS.as_uri()}/">'
            f'<link rel="stylesheet" href="{css_path.as_uri()}"><style>{css}</style></head><body>{content}</body></html>')
    input_path, output_path = WORK / f"{name}.html", WORK / f"{name}.pdf"
    input_path.write_text(page, encoding="utf-8")
    subprocess.run([str(chrome), "--headless", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                    f"--user-data-dir={WORK / 'chrome_profile'}", f"--print-to-pdf={output_path}",
                    "--run-all-compositor-stages-before-draw", "--virtual-time-budget=2000", input_path.as_uri()],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                   creationflags=subprocess.CREATE_NO_WINDOW, timeout=90)
    if not output_path.exists():
        raise RuntimeError(f"Browser did not produce {output_path}")
    return output_path


def roman(value: int) -> str:
    parts = []
    for number, symbol in [(1000,"m"),(900,"cm"),(500,"d"),(400,"cd"),(100,"c"),(90,"xc"),
                           (50,"l"),(40,"xl"),(10,"x"),(9,"ix"),(5,"v"),(4,"iv"),(1,"i")]:
        while value >= number:
            parts.append(symbol)
            value -= number
    return "".join(parts)


def canvas_lines(text: str, size: float, width: float) -> list[str]:
    lines = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and pdfmetrics.stringWidth(candidate, "ThesisArial", size) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def wrap_canvas(c: canvas.Canvas, text: str, y: float, *, size: float = 12,
                leading: float = 24, width: float | None = None) -> float:
    c.setFont("ThesisArial", size)
    for line in canvas_lines(text, size, width or RIGHT - LEFT):
        c.drawString(LEFT, y, line)
        y -= leading
    return y


def caption_titles(sources: list[str], kind: str) -> list[str]:
    # A list repeats the caption's title sentence; subsequent sentences explain it.
    found = {}
    pattern = rf"\*\*{kind} (\d+)\.(?:\*\*)?\s*([^\n]+)"
    for source in sources:
        for match in re.finditer(pattern, source):
            text = match[2].replace("**", "").strip()
            title = re.split(r"(?<=\.)\s+", text, maxsplit=1)[0]
            number = int(match[1])
            if number in found:
                raise ValueError(f"Duplicate {kind} caption: {number}")
            found[number] = title
    if sorted(found) != list(range(1, len(found) + 1)):
        raise ValueError(f"Nonconsecutive {kind} captions: {sorted(found)}")
    return [found[i] for i in sorted(found)]


def list_layout(kind: str, labels: list[str], pages: list[int]) -> list[list[tuple[list[str], int]]]:
    chunks = [[]]
    y = 670
    for index, (label, page) in enumerate(zip(labels, pages), 1):
        lines = canvas_lines(f"{kind} {index}. {label}", 11, RIGHT - LEFT - 38)
        height = len(lines) * 16 + LIST_ENTRY_GAP
        if y - height < 108:
            chunks.append([])
            y = 670
        chunks[-1].append((lines, page))
        y -= height
    return chunks


def front_matter(chapter_lengths: list[int], chapter_titles: list[str],
                 figure_pages: list[int], table_pages: list[int],
                 appendix_entries: list[tuple[str, int]],
                 figures: list[str], tables: list[str],
                 body_entries: list[tuple[int, str, int]]) -> tuple[Path, list[tuple[str, int]]]:
    pdfmetrics.registerFont(TTFont("ThesisArial", str(FONT)))
    if len(ABSTRACT.split()) > 350:
        raise ValueError("Abstract exceeds the JHU 350-word limit")
    table_chunks = list_layout("Table", tables, table_pages)
    figure_chunks = list_layout("Figure", figures, figure_pages)
    def contents_layout(front_entries):
        entries = [(1, title, roman(page)) for title, page in front_entries
                   if title != "Table of contents"]
        entries.extend((level, title, str(page)) for level, title, page in body_entries)
        chunks = [[]]
        y = 670
        for index, (level, title, label) in enumerate(entries):
            indent = (level - 1) * 14
            lines = canvas_lines(title, 11, RIGHT - LEFT - 38 - indent)
            height = len(lines) * 16 + (10 if level == 1 else 4)
            reserve = 0
            if index + 1 < len(entries) and entries[index + 1][0] > level:
                next_level, next_title, _ = entries[index + 1]
                reserve = len(canvas_lines(next_title, 11,
                              RIGHT - LEFT - 38 - (next_level - 1) * 14)) * 16 + 4
            if y - height - reserve < 108:
                chunks.append([])
                y = 670
            chunks[-1].append((indent, lines, label, height))
            y -= height
        return chunks

    # The linked JHU front-matter example includes chapter and section titles.
    # Compute list positions from the expanded, paginated contents itself.
    front_entries = [("Abstract", 2), ("Table of contents", 3),
                     ("List of tables", 4), ("List of figures", 4 + len(table_chunks))]
    contents_chunks = contents_layout(front_entries)
    front_entries = [("Abstract", 2), ("Table of contents", 3),
                     ("List of tables", 3 + len(contents_chunks)),
                     ("List of figures", 3 + len(contents_chunks) + len(table_chunks))]
    contents_chunks = contents_layout(front_entries)
    assert front_entries[2][1] == 3 + len(contents_chunks)
    path = WORK / "front.pdf"
    c = canvas.Canvas(str(path), pagesize=(612, 792), initialFontName="ThesisArial")
    center = (LEFT + RIGHT) / 2
    c.setFont("ThesisArial", 16)
    first_y = 792 - 108 - pdfmetrics.getAscent("ThesisArial", 16)
    title_lines = ["COST AWARE STOPPING BOUNDARIES", "IN REASONING LANGUAGE MODELS"]
    for index, text in enumerate(title_lines):
        c.drawCentredString(center, first_y - index * 24, text)
    by_y = first_y - 24 - 72
    author_y = by_y - 12
    c.setFont("ThesisArial", 12)
    c.drawCentredString(center, by_y, "by")
    c.drawCentredString(center, author_y, "Aditya Bhatt")
    statement = "A thesis submitted to Johns Hopkins University in conformity with the requirements for the degree of Master of Science"
    y = author_y - 108
    statement_lines = canvas_lines(statement, 12, RIGHT - LEFT)
    for line in statement_lines:
        c.drawCentredString(center, y, line)
        y -= 18
    location_y = y + 18 - 36
    c.drawCentredString(center, location_y, "Baltimore, Maryland")
    c.drawCentredString(center, location_y - 12, SUBMISSION_DATE)
    c.showPage()
    c.setFont("ThesisArial", 16)
    c.drawString(LEFT, 700, "Abstract")
    end = wrap_canvas(c, ABSTRACT, 675)
    if end - 44 < 108:
        raise ValueError("Abstract and reader names do not fit within the page margins")
    end = wrap_canvas(c, "Research adviser: Zerotti Woods", end - 12, size=11, leading=16)
    wrap_canvas(c, "Second reader: Moustapha Pemy", end, size=11, leading=16)
    c.showPage()
    for chunk_index, chunk in enumerate(contents_chunks):
        c.setFont("ThesisArial", 16)
        c.drawString(LEFT, 700, "Table of contents" + (" (continued)" if chunk_index else ""))
        y = 670
        for indent, lines, label, height in chunk:
            c.setFont("ThesisArial", 11)
            for line_index, line in enumerate(lines):
                c.drawString(LEFT + indent, y - line_index * 16, line)
            c.drawRightString(RIGHT, y, label)
            y -= height
        c.showPage()
    for title, chunks in [("List of tables", table_chunks), ("List of figures", figure_chunks)]:
        for chunk_index, chunk in enumerate(chunks):
            c.setFont("ThesisArial", 16)
            c.drawString(LEFT, 700, title + (" (continued)" if chunk_index else ""))
            y = 670
            for lines, page in chunk:
                c.setFont("ThesisArial", 11)
                for line_index, line in enumerate(lines):
                    c.drawString(LEFT, y - line_index * 16, line)
                c.drawRightString(RIGHT, y, str(page))
                y -= len(lines) * 16 + LIST_ENTRY_GAP
            c.showPage()
    c.save()
    return path, front_entries


def inspect_pdf(document: fitz.Document) -> dict:
    defects=[]
    for page_index,page in enumerate(document):
        if not page.get_text().strip():
            defects.append({"page":page_index+1,"error":"blank page"})
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    x0,y0,x1,y1=span["bbox"]
                    if x0 < LEFT - 2 or x1 > RIGHT + 2 or y0 < 70 or y1 > 723:
                        defects.append({"page":page_index+1,"error":"text outside edition margins","text":span["text"],"bbox":span["bbox"]})
    if defects:
        (WORK/"layout_defects.json").write_text(json.dumps(defects,indent=2),encoding="utf-8")
        raise RuntimeError(f"PDF layout has {len(defects)} defects; see {WORK/'layout_defects.json'}")
    fonts=[]
    for page in document:
        for entry in page.get_fonts(full=True):
            if entry[0] and entry[0] not in fonts:
                fonts.append(entry[0])
    unembedded=[xref for xref in fonts if not document.extract_font(xref)[3]]
    if unembedded:
        raise RuntimeError(f"Unembedded fonts: {unembedded}")
    return {"pages":len(document),"text_margin_defects":0,"blank_pages":0,"embedded_font_objects":len(fonts),
            "pdfa_conformance":"not asserted; dedicated final validator required"}


def main() -> None:
    global WORK, OUTPUT, IMAGES, LEFT, SUBMISSION_DATE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-pending-live",action="store_true")
    parser.add_argument("--edition", choices=["digital", "print"], default="digital")
    parser.add_argument("--document-version", choices=["v5"], default="v5",
                        help="Historical editions require their archived source and builder versions.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--submission-date", default="October 2026")
    parser.add_argument("--chrome",type=Path,default=Path("C:/Program Files/Google/Chrome/Application/chrome.exe"))
    args=parser.parse_args()
    import datetime
    datetime.datetime.strptime(args.submission_date, "%B %Y")
    SUBMISSION_DATE = args.submission_date
    LEFT = 108 if args.edition == "print" else 72
    WORK = ROOT / "tmp/pdfs/formal_thesis" / args.document_version / args.edition
    OUTPUT = args.output.resolve() if args.output else WORK / "source.pdf"
    IMAGES = DOCS / "images/thesis_v2"
    formal_docs = DOCS / "formal"
    formal_docs.mkdir(parents=True, exist_ok=True)
    for path in (WORK, IMAGES, OUTPUT.parent):
        path.mkdir(parents=True,exist_ok=True)
    runtime=ROOT/"tmp/thesis_pdf_runtime"
    if not (runtime/"node_modules/katex/dist/katex.min.css").is_file():
        raise FileNotFoundError("Install the pinned KaTeX renderer as documented in this script.")
    sync_theory()
    inserts=evidence_inserts(args.allow_pending_live, preserve_figures=True)
    sources=[DOCS/"chapters"/name for name in CHAPTERS]+[DOCS/"references.md",DOCS/"appendices.md"]
    reference_source=(DOCS/"references.md").read_text(encoding="utf-8")
    ref_keys=re.findall(r"^\[([A-Za-z0-9]+)\]",reference_source,re.M)
    citations={key:str(index+1) for index,key in enumerate(ref_keys)}
    if len(citations) != len(ref_keys):
        raise ValueError("Duplicate bibliography keys")
    rendered_sources=[]
    for path in sources:
        source=path.read_text(encoding="utf-8")
        for key,value in inserts.items():
            source=source.replace(f"[[{key}]]",value)
        if re.search(r"\[\[[A-Z_]+\]\]",source):
            raise ValueError(f"Unresolved manuscript insert in {path}")
        citation_text = re.sub(r"```[^\n]*\n.*?```|`[^`\n]+`", "", source, flags=re.S)
        citation_text = re.sub(r"\$\$(.+?)\$\$|(?<!\\)\$([^$]+?)(?<!\\)\$", "", citation_text, flags=re.S)
        named_citations = set(re.findall(r"\[([A-Za-z][A-Za-z0-9]+)\](?!\()", citation_text))
        unknown = named_citations - citations.keys()
        if unknown:
            raise ValueError(f"Unresolved bibliography keys in {path}: {sorted(unknown)}")
        for key,value in citations.items():
            source=source.replace(f"[{key}]",f"[{value}]")
        source=source.replace("images/thesis_v1/", "images/thesis_v2/")
        source=source.translate(str.maketrans({c:"-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015"}))
        rendered_sources.append(source)
    compiled_front = (f"# {TITLE}\n\nby\n\nAditya Bhatt\n\n"
        "A thesis submitted to Johns Hopkins University in conformity with the requirements for the degree of Master of Science\n\n"
        f"Baltimore, Maryland\n\n{SUBMISSION_DATE}\n\n# Abstract\n\n{ABSTRACT}\n\n"
        "Research adviser: Zerotti Woods\n\nSecond reader: Moustapha Pemy")
    compiled="\n\n".join([compiled_front]+rendered_sources)
    (DOCS/f"Masters_Thesis_Formal_{args.document_version}.md").write_text(compiled,encoding="utf-8",newline="\n")
    parts=[]
    for index,source in enumerate(rendered_sources):
        parts.append(print_chapter(source,f"part_{index+1}",args.chrome,runtime))
    body=fitz.open()
    starts=[]
    lengths=[]
    for part in parts:
        starts.append(len(body)+1)
        with fitz.open(part) as section:
            lengths.append(len(section))
            body.insert_pdf(section)
    def locate(label: str) -> int:
        for index,page in enumerate(body):
            # Chrome may wrap immediately after a hyphen inside a heading.
            if re.sub(r"\s+", "", label) in re.sub(r"\s+", "", page.get_text()):
                return index+1
        raise ValueError(f"Caption not found in PDF: {label}")
    figures = caption_titles(rendered_sources, "Figure")
    tables = caption_titles(rendered_sources, "Table")
    figure_pages=[locate(f"Figure {i}.") for i in range(1,len(figures)+1)]
    table_pages=[locate(f"Table {i}.") for i in range(1,len(tables)+1)]
    titles=[source.splitlines()[0].lstrip("# ") for source in rendered_sources]
    appendix_entries=[(title, locate(title)) for title in re.findall(r"^# (Appendix [B-Z][^\n]*)", rendered_sources[-1], re.M)]
    body_entries = [(len(m[1]), m[2], locate(m[2])) for source in rendered_sources
                    for m in re.finditer(r"^(#{1,3}) ([^\n]+)$", source, re.M)]
    front, front_entries = front_matter(lengths,titles,figure_pages,table_pages,appendix_entries,figures,tables,body_entries)
    final=fitz.open(front)
    front_count=len(final)
    final.insert_pdf(body)
    # ReportLab subsets the folio font to valid mappings. The earlier full-font
    # insertion included six unused malformed supplementary Unicode ranges.
    folio_path = WORK / "folios.pdf"
    folios = canvas.Canvas(str(folio_path),pagesize=(612,792),initialFontName="ThesisArial")
    for index in range(len(final)):
        if index:
            label=roman(index+1) if index<front_count else str(index-front_count+1)
            folios.setFont("ThesisArial",10)
            folios.drawCentredString(306,75,label)
        folios.showPage()
    folios.save()
    with fitz.open(folio_path) as overlay:
        for index in range(1,len(final)):
            final[index].show_pdf_page(final[index].rect,overlay,index)
    # ReportLab creates an unused base-font resource. Purge unused resources,
    # then require every retained font object to have embedded font data.
    for page in final:
        page.clean_contents(sanitize=True)
    final.set_toc([[1,title,page] for title,page in front_entries] +
                  [[level,title,front_count+start] for level,title,start in body_entries])
    final.set_metadata({"title":TITLE,"author":"Aditya Bhatt","subject":"Master of Science thesis, Applied and Computational Mathematics"})
    final.xref_set_key(final.pdf_catalog(), "Lang", "(en-US)")
    final.set_page_labels([{"startpage":0,"prefix":"","style":"r","firstpagenum":1},
                          {"startpage":front_count,"prefix":"","style":"D","firstpagenum":1}])
    audit=inspect_pdf(final)
    final.save(OUTPUT,garbage=4,deflate=True)
    # Garbage collection deduplicates retained font objects. Audit the saved
    # file so the receipt reports the actual final count rather than memory.
    with fitz.open(OUTPUT) as saved:
        audit=inspect_pdf(saved)
    manifest={"output":OUTPUT.relative_to(ROOT).as_posix(),"sha256":sha(OUTPUT),"audit":audit,
              "schema":f"formal-thesis-build-{args.document_version}","edition":args.edition,
              "main_builder_sha256":sha(Path(__file__)),
              "contents_body_entries":[{"level":level,"title":title,"body_page":page}
                                      for level,title,page in body_entries],
              "submission_month_year":SUBMISSION_DATE,"front_matter_pages":front_count,
              "left_margin_inches":LEFT/72,"other_minimum_margins_inches":1,
              "abstract_words":len(ABSTRACT.split()),"figure_titles":figures,"table_titles":tables,
              "figure_body_pages":figure_pages,"table_body_pages":table_pages,
              "source_files":{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
              "canonical_math_source_sha256":sha(ROOT/"research/mathematical_foundations.md"),
              "evidence_manifest_sha256":sha(EVIDENCE/"manifest.json"),"katex":"0.19.0",
              "data_manifest_sha256":sha(ROOT/"data_manifest_v1.json"),
              "chapter_page_counts":dict(zip(titles,lengths)),
              "word_count":len(re.findall(r"\b[\w'-]+\b",compiled)),
              "interim_live_results_permitted":args.allow_pending_live,
              "visual_review":"required after this build; machine checks alone do not prove visual quality"}
    (formal_docs/f"build_manifest_{args.edition}_{args.document_version}.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8",newline="\n")
    review=WORK/"rendered"
    review.mkdir(exist_ok=True)
    for index,page in enumerate(final):
        page.get_pixmap(matrix=fitz.Matrix(.8,.8),alpha=False).save(review/f"page_{index+1:03}.png")
    print(json.dumps({"output":str(OUTPUT),"pages":len(final),"words":manifest["word_count"],"audit":audit}))


if __name__=="__main__":
    main()
