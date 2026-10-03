"""Build the six-chapter thesis draft with checked math and frozen evidence.

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
IMAGES = DOCS / "images/thesis_v1"
WORK = ROOT / "tmp/pdfs/master_thesis"
OUTPUT = ROOT / "output/pdf/Masters_Thesis_Draft_v1_Aditya_Bhatt.pdf"
CHAPTERS = ["chapter1_intro.md", "chapter2_theory.md", "chapter3_methodology.md",
            "chapter4_empirical.md", "chapter5_online.md", "chapter6_discussion.md"]
FONT = Path("C:/Windows/Fonts/arial.ttf")
TITLE = "Cost aware stopping boundaries in reasoning language models"
FIGURES = ["Population accuracy and one-step net gain", "Causal detector ranking across domains",
           "Development accuracy and completion-token Pareto comparison",
           "Actual accuracy and completion-token comparisons"]
TABLES = ["Standardized five-step corpus", "Model configurations and corpus membership",
          "Selected population transition estimates", "Matched estimator and systems contrasts",
          "Causal detector metrics", "Archived policy failure taxonomy", "Controller latency", "Paired live generation",
          "Adversarial live generation", "Deployable predictor evaluation", "Trained policy live generation",
          "Trained policy adversarial generation", "Development replay policies", "Claim evidence sources"]


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
    plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans", "axes.spines.top": False,
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
    axes[0, 0].legend(fontsize=12)
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
                        (x, row.active_accuracy), xytext=offset, textcoords="offset points", fontsize=12,
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
                ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=12,
                            ha="center", arrowprops={"arrowstyle": "-", "color": color, "lw": .7})
            ax.set(title=title, xlabel="Completion tokens / full horizon", ylabel="Answer accuracy",
                   xlim=(.35, 1.12), ylim=(.0, .11))
            ax.grid(alpha=.2)
        fig.savefig(IMAGES / "actual_live_pareto.png")
        plt.close(fig)


def evidence_inserts(allow_pending: bool) -> dict[str, str]:
    boundary = pd.read_csv(EVIDENCE / "boundary_domain_step_metrics.csv")
    controlled = pd.read_csv(EVIDENCE / "algorithm_v2_normalized_effects.csv")
    detector = pd.read_csv(EVIDENCE / "tournament_balanced_summary.csv")
    failures = read_json(ROOT / "research/reports/thesis_failure_audit_v1/audit_summary.json")
    prefix_evaluation = read_json(PREFIX_MODEL / "evaluation.json")
    replay = pd.read_csv(ONLINE / "replay_pareto.csv") if (ONLINE / "replay_pareto.csv").exists() else None
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
        "BOUNDARY_TABLE": table(["Step", "Accuracy", "Repair count", "Repair prob.", "Corruption count", "Corruption prob.", "Net gain", "95% interval"], rows,
                                "Table 3. GSM8K transition panel; 19,500 trajectories and 500 task clusters per row."),
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
        "omit uncertainty, which is reported in Tables 8, 9, 11 and 12.") if actual_figure.exists() else pending
    return results


def sync_theory() -> None:
    source = (ROOT / "research/mathematical_foundations.md").read_text(encoding="utf-8")
    start = source.index("## 1.")
    end = source.index("## 9.")
    theory = source[start:end]
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
    protected = pattern.sub(protect, source)
    if re.search(r"(?<!\\)\$", protected):
        raise ValueError(f"Unmatched math delimiter in {key}")
    input_path, rendered_path = WORK / f"{key}_math.json", WORK / f"{key}_math_rendered.json"
    input_path.write_text(json.dumps(expressions), encoding="utf-8")
    subprocess.run(["node", str(ROOT / "tools/render_thesis_math.mjs"), str(runtime),
                    str(input_path), str(rendered_path)], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    rendered = read_json(rendered_path)
    content = markdown.markdown(protected, extensions=["tables", "fenced_code"])
    # Keep each numbered caption with its table and keep these short evidence
    # tables intact. A repeated header alone does not prevent orphan captions.
    content = re.sub(r'(<p><strong>Table \d+\..*?</strong></p>\s*<table>.*?</table>)',
                     r'<div class="table-block">\1</div>', content, flags=re.S)
    content = re.sub(r'(<p><img\b[^>]*></p>\s*<p><strong>Figure \d+\.</strong>.*?</p>)',
                     r'<div class="figure-block">\1</div>', content, flags=re.S)
    for index, value in enumerate(rendered):
        content = content.replace(f"MATHPLACEHOLDER{index}END", value)
    return content


def print_chapter(source: str, name: str, chrome: Path, runtime: Path) -> Path:
    content = html_math(source, runtime, name)
    css_path = runtime / "node_modules/katex/dist/katex.min.css"
    css = """
    @page {size:letter; margin:1in 1in 1.35in 1in;}
    body {font-family:Arial,sans-serif; font-size:12pt; line-height:2; color:#111; margin:0;}
    p {margin:0 0 12pt;} h1 {font-size:18pt;line-height:1.35;margin:0 0 25pt;break-after:avoid;}
    h2 {font-size:14pt;line-height:1.4;margin:21pt 0 11pt;break-after:avoid;}
    h3 {font-size:12pt;line-height:1.4;margin:15pt 0 9pt;break-after:avoid;}
    table {font-size:10pt;line-height:1.35;border-collapse:collapse;width:100%;margin:12pt 0 18pt;break-inside:avoid;}
    .table-block {break-inside:avoid;}
    .figure-block {break-inside:avoid;}
    th {text-align:left;border-bottom:1pt solid #333;} td,th {padding:6pt 4pt;vertical-align:top;}
    tr {break-inside:avoid;} td {border-bottom:.4pt solid #bbb;} thead {display:table-header-group;}
    code {font-size:10pt;overflow-wrap:anywhere;} pre {font-size:10pt;line-height:1.4;white-space:pre-wrap;}
    img {width:100%;height:auto;break-inside:avoid;} a {color:#222;overflow-wrap:anywhere;text-decoration:none;}
    .katex {font-size:1.03em;} .katex-display {margin:14pt 0;line-height:1.2;break-inside:avoid;}
    blockquote {margin:12pt 15pt;font-size:11pt;}
    """
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


def wrap_canvas(c: canvas.Canvas, text: str, y: float, *, size: int = 12, leading: int = 24) -> float:
    c.setFont("ThesisArial", size)
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if pdfmetrics.stringWidth(candidate, "ThesisArial", size) > 468:
            c.drawString(72, y, line)
            y -= leading
            line = word
        else:
            line = candidate
    if line:
        c.drawString(72, y, line)
        y -= leading
    return y


def front_matter(chapter_lengths: list[int], chapter_titles: list[str], figure_pages: list[int], table_pages: list[int], appendix_entries: list[tuple[str, int]]) -> Path:
    pdfmetrics.registerFont(TTFont("ThesisArial", str(FONT)))
    path = WORK / "front.pdf"
    c = canvas.Canvas(str(path), pagesize=(612,792), initialFontName="ThesisArial")
    c.setFont("ThesisArial", 16)
    for y, text in [(684,"COST AWARE STOPPING BOUNDARIES"),(660,"IN REASONING LANGUAGE MODELS")]:
        c.drawCentredString(306, y, text)
    c.setFont("ThesisArial",12)
    c.drawCentredString(306, 580, "by")
    c.drawCentredString(306, 556, "Aditya Bhatt")
    for y, text in [(445,"A thesis submitted to Johns Hopkins University"),(421,"in conformity with the requirements for the degree of"),
                    (397,"Master of Science in Applied and Computational Mathematics"),(349,"Baltimore, Maryland"),(325,"October 2026")]:
        c.drawCentredString(306,y,text)
    c.showPage()
    c.setFont("ThesisArial",16)
    c.drawString(72,700,"Abstract")
    abstract = ("Additional reasoning can repair an incorrect answer, replace a correct answer, or consume computation without sufficient improvement. "
        "This thesis formulates response-level stopping as a finite-horizon decision based on observable prefixes and hidden correctness. "
        "It proves the binary repair-corruption drift identity and the Bellman/Snell optimal stopping result, gives a sufficient persistence condition for a drift-sign rule, "
        "and constructs exact counterexamples to unconditional myopic optimality. "
        "The experimental record separates a variable-horizon matrix from a standardized corpus of 144,440 saved rows, 28,888 trajectories, and 2,948 tasks. "
        "Recomputed cluster-based tables show model- and domain-dependent continuation value and matched effects of estimator, token-cap, and precision changes. "
        "The historical stacked ROC-AUC of 0.955156 is retained as a retrospective non-nested diagnostic, rather than an online performance guarantee. "
        "A fitted prefix controller enforces a two-step floor and prevents future generation after stopping. "
        "On 100 new GSM8K questions, its actual arm saves 56.51 percent of completion tokens with 7 correct answers versus 6 at the full horizon. "
        "On 20 traps it saves 52.11 percent, with one correct answer in each arm. "
        "Every learned run stops at step two; adaptive benefit and accuracy noninferiority remain unestablished. "
        "The results support causal, cost-sensitive stopping experiments while leaving stronger calibration, universal optimality, and external noninferiority claims unestablished.")
    end = wrap_canvas(c,abstract,675)
    wrap_canvas(c,"Research draft v1.0 for committee review. Defense, approval, final PDF/A validation, and institutional acceptance are pending.",end-18,size=10,leading=18)
    c.showPage()
    c.setFont("ThesisArial",16)
    c.drawString(72,700,"Table of contents")
    y, start = 670, 1
    for title, length in zip(chapter_titles, chapter_lengths):
        c.setFont("ThesisArial",11)
        c.drawString(72,y,title)
        c.drawRightString(540,y,str(start))
        start += length
        y -= 36
    for title, page in appendix_entries:
        c.setFont("ThesisArial",11)
        c.drawString(72,y,title)
        c.drawRightString(540,y,str(page))
        y -= 36
    c.showPage()
    for title, labels, pages in [("List of figures", FIGURES, figure_pages),("List of tables", TABLES, table_pages)]:
        c.setFont("ThesisArial",16)
        c.drawString(72,700,title)
        y=670
        for index,(label,page) in enumerate(zip(labels,pages),1):
            y=wrap_canvas(c,f"{index}. {label}",y,size=11,leading=18)
            c.setFont("ThesisArial",11)
            c.drawRightString(540,y+18,str(page))
            y-=18
        c.showPage()
    c.save()
    return path


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
                    if x0 < 70 or x1 > 542 or y0 < 68 or y1 > 725:
                        defects.append({"page":page_index+1,"error":"text outside digital margins","text":span["text"],"bbox":span["bbox"]})
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-pending-live",action="store_true")
    parser.add_argument("--chrome",type=Path,default=Path("C:/Program Files/Google/Chrome/Application/chrome.exe"))
    args=parser.parse_args()
    for path in (WORK, IMAGES, OUTPUT.parent):
        path.mkdir(parents=True,exist_ok=True)
    runtime=ROOT/"tmp/thesis_pdf_runtime"
    if not (runtime/"node_modules/katex/dist/katex.min.css").is_file():
        raise FileNotFoundError("Install the pinned KaTeX renderer as documented in this script.")
    sync_theory()
    inserts=evidence_inserts(args.allow_pending_live)
    sources=[DOCS/"chapters"/name for name in CHAPTERS]+[DOCS/"references.md",DOCS/"appendices.md"]
    reference_source=(DOCS/"references.md").read_text(encoding="utf-8")
    ref_keys=re.findall(r"^\[([A-Za-z0-9]+)\]",reference_source,re.M)
    citations={key:str(index+1) for index,key in enumerate(ref_keys)}
    rendered_sources=[]
    for path in sources:
        source=path.read_text(encoding="utf-8")
        for key,value in inserts.items():
            source=source.replace(f"[[{key}]]",value)
        if re.search(r"\[\[[A-Z_]+\]\]",source):
            raise ValueError(f"Unresolved manuscript insert in {path}")
        for key,value in citations.items():
            source=source.replace(f"[{key}]",f"[{value}]")
        source=source.translate(str.maketrans({c:"-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015"}))
        rendered_sources.append(source)
    compiled="\n\n".join([f"# {TITLE}\n\nAditya Bhatt. Research draft v1.0. October 2026."]+rendered_sources)
    (DOCS/"Masters_Thesis_Draft_v1.md").write_text(compiled,encoding="utf-8",newline="\n")
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
            if label in page.get_text():
                return index+1
        raise ValueError(f"Caption not found in PDF: {label}")
    figure_pages=[locate(f"Figure {i}.") for i in range(1,len(FIGURES)+1)]
    table_pages=[locate(f"Table {i}.") for i in range(1,len(TABLES)+1)]
    titles=[source.splitlines()[0].lstrip("# ") for source in rendered_sources]
    appendix_entries=[(title, locate(title)) for title in re.findall(r"^# (Appendix [B-Z][^\n]*)", rendered_sources[-1], re.M)]
    front=front_matter(lengths,titles,figure_pages,table_pages,appendix_entries)
    final=fitz.open(front)
    front_count=len(final)
    final.insert_pdf(body)
    for index,page in enumerate(final):
        if index==0:
            continue
        label=roman(index+1) if index<front_count else str(index-front_count+1)
        page.insert_font(fontname="ThesisPageArial",fontfile=str(FONT))
        width=fitz.Font(fontfile=str(FONT)).text_length(label,fontsize=10)
        page.insert_text((306-width/2,717),label,fontsize=10,fontname="ThesisPageArial")
    # ReportLab creates an unused base-font resource. Purge unused resources,
    # then require every retained font object to have embedded font data.
    for page in final:
        page.clean_contents(sanitize=True)
    final.set_toc([[1,title,front_count+start] for title,start in zip(titles,starts)] +
                  [[1,title,front_count+start] for title,start in appendix_entries])
    final.set_metadata({"title":TITLE,"author":"Aditya Bhatt","subject":"Research thesis draft v1.0; committee approval pending"})
    audit=inspect_pdf(final)
    final.save(OUTPUT,garbage=4,deflate=True)
    manifest={"output":OUTPUT.relative_to(ROOT).as_posix(),"sha256":sha(OUTPUT),"audit":audit,
              "source_files":{p.relative_to(ROOT).as_posix():sha(p) for p in sources},
              "canonical_math_source_sha256":sha(ROOT/"research/mathematical_foundations.md"),
              "evidence_manifest_sha256":sha(EVIDENCE/"manifest.json"),"katex":"0.19.0",
              "data_manifest_sha256":sha(ROOT/"data_manifest_v1.json"),
              "chapter_page_counts":dict(zip(titles,lengths)),
              "word_count":len(re.findall(r"\b[\w'-]+\b",compiled)),
              "interim_live_results_permitted":args.allow_pending_live,
              "visual_review":"required after this build; machine checks alone do not prove visual quality"}
    (DOCS/"thesis_build_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8",newline="\n")
    review=WORK/"rendered"
    review.mkdir(exist_ok=True)
    for index,page in enumerate(final):
        page.get_pixmap(matrix=fitz.Matrix(.8,.8),alpha=False).save(review/f"page_{index+1:03}.png")
    print(json.dumps({"output":str(OUTPUT),"pages":len(final),"words":manifest["word_count"],"audit":audit}))


if __name__=="__main__":
    main()
