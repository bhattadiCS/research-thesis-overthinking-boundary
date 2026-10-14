#!/usr/bin/env python3
"""Build the six-page Semester 1 mathematical research report PDF."""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import time
from pathlib import Path

import fitz
import markdown
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "ThesisDocs" / "Thesis_Semester1_Research_Report_Fall_2026.md"
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "Thesis_Semester1_Research_Report_Aditya_Bhatt_Fall_2026.pdf"
DEFAULT_WORK_DIR = ROOT / "tmp" / "pdfs" / "semester1_research_report_build"
BOUNDARY_CSV = ROOT / "research" / "outputs" / "progress_report_review_metrics" / "boundary_domain_step_metrics.csv"
ALGORITHM_CSV = ROOT / "research" / "outputs" / "progress_report_review_metrics" / "algorithm_v2_normalized_effects.csv"

DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--chrome", type=Path)
    parser.add_argument("--keep-intermediates", action="store_true")
    return parser.parse_args()


def find_chrome(explicit: Path | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(
        [
            Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
            Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
            Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No supported Chrome or Edge executable was found.")


def split_pages(source: str) -> list[str]:
    source = source.translate(DASH_TRANSLATION)
    pages = [part.strip() for part in source.split("<!-- PAGE BREAK -->")]
    if len(pages) != 6:
        raise ValueError(f"Expected exactly six report pages, found {len(pages)}.")
    return pages


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.7,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "black",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )


def make_pipeline_figure(work_dir: Path) -> Path:
    output = work_dir / "scientific_pipeline.svg"
    figure, axis = plt.subplots(figsize=(7.25, 1.28))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 2)
    axis.axis("off")
    labels = [
        "Benchmark\nquestion",
        "Model generates\nresponse at step t",
        "Trace store\nanswer + signals",
        "Reference grader\ncorrect / incorrect",
        "Hazards, plots,\nand stop boundary",
    ]
    x_positions = [0.1, 2.15, 4.2, 6.25, 8.3]
    widths = [1.55, 1.65, 1.65, 1.65, 1.6]
    stage_colors = ["#E8EEF5", "#D7EAF7", "#DDF2EE", "#FFF0C9", "#DDEDDC"]
    for x, width, label, stage_color in zip(x_positions, widths, labels, stage_colors):
        box = patches.FancyBboxPatch(
            (x, 0.55),
            width,
            0.9,
            boxstyle="round,pad=0.025,rounding_size=0.03",
            linewidth=0.9,
            edgecolor="#263746",
            facecolor=stage_color,
        )
        axis.add_patch(box)
        axis.text(x + width / 2, 1.0, label, ha="center", va="center", fontsize=7.5)
    for index in range(4):
        start = x_positions[index] + widths[index] + 0.06
        end = x_positions[index + 1] - 0.06
        axis.annotate(
            "",
            xy=(end, 1.0),
            xytext=(start, 1.0),
            arrowprops={"arrowstyle": "->", "color": "#38556A", "linewidth": 1.0},
        )
    axis.text(5.0, 0.18, "Controlled factors: model, domain, temperature, horizon, precision, and stopping features", ha="center", fontsize=7.2)
    figure.savefig(output, format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return output


def make_results_figure(work_dir: Path) -> Path:
    output = work_dir / "semester1_results.svg"
    frame = pd.read_csv(BOUNDARY_CSV)
    gsm = frame[(frame["domain"] == "gsm8k") & (frame["step"] >= 2)].sort_values("step")
    x = gsm["step"].to_numpy(dtype=float)
    effects = pd.read_csv(ALGORITHM_CSV)
    effects = effects[effects["experiment"].isin(["N2a", "N2b", "N2c", "N3", "N4"])].copy()
    effects["experiment"] = pd.Categorical(
        effects["experiment"], categories=["N2a", "N2b", "N2c", "N3", "N4"], ordered=True
    )
    effects = effects.sort_values("experiment")

    figure = plt.figure(figsize=(7.25, 3.05))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.12, 1.0], hspace=0.56, wspace=0.36)
    top = figure.add_subplot(grid[0, :])
    left = figure.add_subplot(grid[1, 0])
    right = figure.add_subplot(grid[1, 1])

    accuracy_color = "#0072B2"
    repair_color = "#009E73"
    corruption_color = "#D55E00"
    drift_color = "#6A3D9A"
    negative_color = "#D55E00"
    positive_color = "#009E73"

    curve_specs = [
        ("accuracy", "accuracy_ci_low", "accuracy_ci_high", "Panel accuracy", accuracy_color, "o"),
        ("repair_hazard", "repair_ci_low", "repair_ci_high", "Repair hazard", repair_color, "s"),
        ("corruption_hazard", "corruption_ci_low", "corruption_ci_high", "Corruption hazard", corruption_color, "^"),
    ]
    for value_col, low_col, high_col, label, color, marker in curve_specs:
        values = gsm[value_col].to_numpy(dtype=float)
        bounds_low = gsm[low_col].to_numpy(dtype=float)
        bounds_high = gsm[high_col].to_numpy(dtype=float)
        top.fill_between(x, bounds_low, bounds_high, color=color, alpha=0.14, linewidth=0)
        top.plot(
            x,
            values,
            color=color,
            marker=marker,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=3.7,
            linewidth=1.25,
            label=label,
        )
    top.set_xlim(1.7, 9.3)
    top.set_ylim(0.05, 0.53)
    top.set_xticks(np.arange(2, 10))
    top.set_ylabel("Probability")
    top.set_title("(a) Stepwise correctness and transition hazards")
    top.grid(axis="y", color="0.88", linewidth=0.45)
    top.legend(frameon=False, fontsize=6.8, ncol=3, loc="upper left")

    y = gsm["net_drift"].to_numpy(dtype=float)
    low = gsm["net_drift_ci_low"].to_numpy(dtype=float)
    high = gsm["net_drift_ci_high"].to_numpy(dtype=float)
    left.fill_between(x, low, high, color=drift_color, linewidth=0, alpha=0.16, label="95% task bootstrap")
    left.plot(
        x,
        y,
        color=drift_color,
        marker="o",
        markerfacecolor=drift_color,
        markeredgecolor="white",
        markeredgewidth=0.4,
        markersize=3.7,
        linewidth=1.25,
        label="Estimated gain",
    )
    left.axhline(0.0, color="#4D4D4D", linewidth=0.8, linestyle="--")
    left.set_xlim(1.7, 9.3)
    left.set_xticks(np.arange(2, 10))
    left.set_xlabel("Current reasoning step t")
    left.set_ylabel("One-step net gain")
    left.set_title("(b) One-step net gain, cost = 0.05")
    left.grid(axis="y", color="0.88", linewidth=0.45)
    value_by_step = dict(zip(x.astype(int), y))
    left.annotate(
        "+0.051",
        xy=(2, value_by_step[2]),
        xytext=(2.50, 0.055),
        fontsize=6.8,
        color=drift_color,
        arrowprops={"arrowstyle": "-", "linewidth": 0.65, "color": drift_color},
    )
    left.annotate(
        "-0.013",
        xy=(4, value_by_step[4]),
        xytext=(4.25, -0.002),
        fontsize=6.8,
        color=negative_color,
        arrowprops={"arrowstyle": "-", "linewidth": 0.65, "color": negative_color},
    )
    left.legend(frameon=False, fontsize=6.3, loc="lower left")

    effect_labels = {
        "N2a": "Gradient-boosted trees",
        "N2b": "Probability recalibration",
        "N2c": "Prior-step features",
        "N3": "Pooled hazard estimates",
        "N4": "Step-2 answer changes",
    }
    effect_y = np.arange(len(effects), 0, -1)
    effect_mean = effects["mean_controlled_effect"].to_numpy(dtype=float)
    effect_low = effects["ci_95_low"].to_numpy(dtype=float)
    effect_high = effects["ci_95_high"].to_numpy(dtype=float)
    effect_err = np.vstack([effect_mean - effect_low, effect_high - effect_mean])
    right.axvline(0.0, color="#4D4D4D", linewidth=0.8, linestyle="--")
    for mean, y_pos, err_low, err_high in zip(effect_mean, effect_y, effect_err[0], effect_err[1]):
        effect_color = positive_color if mean > 0 else negative_color
        right.errorbar(
            [mean],
            [y_pos],
            xerr=np.array([[err_low], [err_high]]),
            fmt="o",
            color=effect_color,
            ecolor=effect_color,
            markerfacecolor=effect_color,
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=4.5,
            capsize=2.2,
            elinewidth=1.1,
        )
    right.set_yticks(effect_y)
    right.set_yticklabels([effect_labels[str(value)] for value in effects["experiment"]])
    right.set_xlim(-0.102, 0.025)
    right.set_ylim(0.35, len(effects) + 0.65)
    right.set_xlabel("Matched utility effect per trajectory")
    right.set_title("(c) Controlled estimator changes")
    right.grid(axis="x", color="0.88", linewidth=0.45)
    right.scatter([], [], s=22, color=positive_color, label="Higher utility")
    right.scatter([], [], s=22, color=negative_color, label="Lower utility")
    right.legend(frameon=False, fontsize=6.5, loc="lower left")

    for axis in (top, left, right):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.savefig(output, format="svg", bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)
    return output


def render_math(source: str, work_dir: Path, namespace: str, equation_start: int) -> tuple[str, int]:
    formula_dir = work_dir / "formulae" / namespace
    formula_dir.mkdir(parents=True, exist_ok=True)
    block_counter = equation_start
    inline_counter = 0
    font = FontProperties(family="STIXGeneral", size=12)

    def normalize_mathtext(expression: str) -> str:
        """Translate a small LaTeX subset that Matplotlib mathtext does not parse."""
        replacements = {
            r"\bigl": "",
            r"\bigr": "",
            r"\Bigl": "",
            r"\Bigr": "",
            r"\biggl": "",
            r"\biggr": "",
        }
        for old, new in replacements.items():
            expression = expression.replace(old, new)
        expression = re.sub(r"\\le(?![A-Za-z])", r"\\leq", expression)
        expression = re.sub(r"\\ge(?![A-Za-z])", r"\\geq", expression)
        return expression

    def block_replacement(match: re.Match[str]) -> str:
        nonlocal block_counter
        block_counter += 1
        expression = normalize_mathtext(" ".join(match.group(1).split()))
        output = formula_dir / f"equation_{block_counter:02d}.svg"
        math_to_image(f"${expression}$", output, prop=font, dpi=220, format="svg", color="black")
        return (
            f'<div class="equation"><img src="{output.as_uri()}" '
            f'alt="{html.escape(expression, quote=True)}"><span class="eqno">({block_counter})</span></div>'
        )

    source = re.sub(r"\$\$(.*?)\$\$", block_replacement, source, flags=re.DOTALL)

    def inline_replacement(match: re.Match[str]) -> str:
        nonlocal inline_counter
        inline_counter += 1
        expression = normalize_mathtext(" ".join(match.group(1).split()))
        output = formula_dir / f"inline_{inline_counter:03d}.svg"
        math_to_image(f"${expression}$", output, prop=FontProperties(family="STIXGeneral", size=10), dpi=180, format="svg", color="black")
        return f'<img class="math-inline" src="{output.as_uri()}" alt="{html.escape(expression, quote=True)}">'

    source = re.sub(r"\\\((.*?)\\\)", inline_replacement, source, flags=re.DOTALL)
    return source, block_counter


def markdown_html(text: str) -> str:
    converter = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists"], output_format="html5")
    return converter.convert(text)


def build_html(input_path: Path, work_dir: Path) -> Path:
    configure_matplotlib()
    pipeline_figure = make_pipeline_figure(work_dir)
    results_figure = make_results_figure(work_dir)
    pages = split_pages(input_path.read_text(encoding="utf-8"))
    rendered_pages = []
    equation_counter = 0
    for page_index, page in enumerate(pages, start=1):
        page, equation_counter = render_math(page, work_dir, f"page_{page_index}", equation_counter)
        page = page.replace(
            "[[PIPELINE_FIGURE]]",
            (
                f'<figure class="pipeline"><img src="{pipeline_figure.as_uri()}" alt="Scientific workflow from benchmark question through model generation, trace storage, grading, and boundary analysis.">'
                '<figcaption><strong>Figure 1.</strong> Semester 1 scientific workflow. Complete responses are saved and graded before boundary statistics are computed.</figcaption></figure>'
            ),
        )
        page = page.replace(
            "[[RESULTS_FIGURE]]",
            (
                f'<figure class="results"><img src="{results_figure.as_uri()}" alt="Three colorblind-safe plots showing GSM8K accuracy, repair and corruption hazards, one-step net gain, and matched utility effects with bootstrap intervals.">'
                '<figcaption><strong>Figure 2.</strong> Stepwise evidence and controlled effects. Panels (a)-(b) use 19,500 GSM8K traces grouped into 500 task clusters at every plotted step; colored bands are 95% task-cluster bootstrap intervals. Panel (b) subtracts the fixed 0.05 continuation cost. Panel (c) reports matched utility changes on 75,965 trajectories; teal/rust marks indicate higher/lower utility and bars are 10,000-draw bootstraps over 52 model-domain cells.</figcaption></figure>'
            ),
        )
        rendered_pages.append(markdown_html(page))

    identity = """
    <div class="identity">
      <div class="institution">JOHNS HOPKINS UNIVERSITY | APPLIED AND COMPUTATIONAL MATHEMATICS</div>
      <div class="report-title">Overthinking Boundaries in Reasoning Language Models</div>
      <div class="report-subtitle">Semester 1 Mathematical Research Report | MATH ACM 625.803-804 | Date: August 6, 2026</div>
      <table class="identity-table">
        <tr><th>Student</th><td>Aditya Bhatt</td><th>Research adviser</th><td>Dr. Zerotti Woods</td></tr>
        <tr><th>Second reader</th><td>Dr. Moustapha Pemy</td><th>Completion target</th><td>Late November / early December 2026</td></tr>
      </table>
    </div>
    """

    page_html = []
    for index, body in enumerate(rendered_pages):
        prefix = identity if index == 0 else ""
        page_html.append(f'<section class="page">{prefix}{body}</section>')

    css = r"""
@page { size: Letter; margin: 0; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: white; }
body { color: #000; font-family: "Times New Roman", Times, serif; }
.page {
  position: relative;
  width: 8.5in;
  height: 11in;
  padding: 0.48in 0.55in 0.46in;
  overflow: visible;
  page-break-after: always;
}
.page:last-child { page-break-after: auto; }
.identity { margin-bottom: 0.11in; }
.institution { font: 700 7.4pt Arial, Helvetica, sans-serif; margin-bottom: 0.045in; }
.report-title { font: 700 17pt "Times New Roman", Times, serif; line-height: 1.04; padding-bottom: 0.045in; border-bottom: 1.1px solid #000; }
.report-subtitle { font: 8.3pt Arial, Helvetica, sans-serif; margin: 0.045in 0 0.075in; }
.identity-table { margin: 0; width: 100%; font: 7.5pt Arial, Helvetica, sans-serif; border-top: 0.7px solid #000; border-bottom: 0.7px solid #000; }
.identity-table th, .identity-table td { border: 0; border-bottom: 0.35px solid #777; padding: 2.5px 4px; }
.identity-table th { width: 14%; font-weight: 700; }
.identity-table td { width: 36%; }
h1, h2, h3 { color: #000; font-family: "Times New Roman", Times, serif; page-break-after: avoid; }
h2 { font-size: 12.2pt; line-height: 1.08; margin: 0.08in 0 0.055in; padding-bottom: 0.025in; border-bottom: 0.7px solid #000; }
h3 { font-size: 9.7pt; line-height: 1.08; margin: 0.075in 0 0.035in; }
p { font-size: 9pt; line-height: 1.19; margin: 0 0 0.055in; orphans: 2; widows: 2; }
strong { font-weight: 700; }
ol, ul { margin: 0.02in 0 0.052in; padding-left: 0.21in; }
li { font-size: 8.9pt; line-height: 1.18; margin-bottom: 0.025in; }
table { width: 100%; border-collapse: collapse; margin: 0.035in 0 0.06in; font: 7.5pt Arial, Helvetica, sans-serif; line-height: 1.13; border-top: 0.85px solid #000; border-bottom: 0.85px solid #000; }
thead { display: table-header-group; }
tr { break-inside: avoid; page-break-inside: avoid; }
th { padding: 2.5px 3.5px; border-bottom: 0.7px solid #000; text-align: left; vertical-align: bottom; font-weight: 700; }
td { padding: 2.5px 3.5px; border-bottom: 0.3px solid #888; vertical-align: top; overflow-wrap: anywhere; }
.table-caption { margin: 0.04in 0 0.012in; font: 7.5pt/1.14 Arial, Helvetica, sans-serif; text-align: left; break-after: avoid; page-break-after: avoid; }
code { font: 0.86em Consolas, "Cascadia Mono", monospace; }
.equation { position: relative; min-height: 0.27in; margin: 0.035in 0 0.048in; padding: 0 0.34in; text-align: center; break-inside: avoid; }
.equation img { max-width: 100%; max-height: 0.39in; vertical-align: middle; }
.eqno { position: absolute; right: 0.03in; top: 50%; transform: translateY(-50%); font-size: 8pt; }
.math-inline { height: 0.95em; max-width: 1.55in; vertical-align: -0.16em; margin: 0 0.03em; }
figure { margin: 0.04in 0 0.06in; break-inside: avoid; page-break-inside: avoid; text-align: center; }
figure img { width: 100%; display: block; }
figure.pipeline img { height: 1.08in; object-fit: contain; }
figure.results img { height: 2.72in; object-fit: contain; }
figcaption { margin-top: 0.02in; font-size: 7.45pt; line-height: 1.14; text-align: left; }
.references { font-size: 7.3pt; line-height: 1.15; margin-top: 0.01in; }
"""
    document = f"<!doctype html><html><head><meta charset=\"utf-8\"><style>{css}</style></head><body>{''.join(page_html)}</body></html>"
    work_dir.mkdir(parents=True, exist_ok=True)
    html_path = work_dir / "semester1_research_report.html"
    html_path.write_text(document, encoding="utf-8")
    return html_path


def print_html(chrome: Path, html_path: Path, raw_pdf: Path, work_dir: Path) -> None:
    profile = work_dir / "chrome_profile"
    profile.mkdir(parents=True, exist_ok=True)
    command = [
        str(chrome),
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        f"--user-data-dir={profile}",
        f"--print-to-pdf={raw_pdf}",
        html_path.as_uri(),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=90)
    if completed.returncode != 0:
        raise RuntimeError(f"Chrome PDF generation failed: {completed.stderr}")
    deadline = time.time() + 20
    while time.time() < deadline:
        if raw_pdf.exists() and raw_pdf.stat().st_size > 15_000:
            return
        time.sleep(0.25)
    raise RuntimeError("Chrome did not create a usable PDF.")


def add_headers_and_metadata(raw_pdf: Path, output_pdf: Path) -> None:
    document = fitz.open(raw_pdf)
    total_pages = document.page_count
    for index, page in enumerate(document):
        width, height = page.rect.width, page.rect.height
        if index > 0:
            page.insert_text((38, 22), "SEMESTER 1 MATHEMATICAL RESEARCH REPORT", fontname="helv", fontsize=6.7, color=(0, 0, 0))
            right = "ADITYA BHATT  |  MATH ACM 625.803-804"
            right_width = fitz.get_text_length(right, fontname="helv", fontsize=6.7)
            page.insert_text((width - 38 - right_width, 22), right, fontname="helv", fontsize=6.7, color=(0, 0, 0))
            page.draw_line((38, 28), (width - 38, 28), color=(0, 0, 0), width=0.4)
        label = f"Page {index + 1} of {total_pages}"
        label_width = fitz.get_text_length(label, fontname="helv", fontsize=6.7)
        page.draw_line((38, height - 27), (width - 38, height - 27), color=(0, 0, 0), width=0.4)
        page.insert_text(((width - label_width) / 2, height - 15), label, fontname="helv", fontsize=6.7, color=(0, 0, 0))

    metadata = document.metadata or {}
    metadata.update(
        {
            "title": "Overthinking Boundaries in Reasoning Language Models: Semester 1 Mathematical Research Report",
            "author": "Aditya Bhatt",
            "subject": "MATH ACM 625.803-804 Semester 1 research report",
            "keywords": "optimal stopping, reasoning models, overthinking, empirical Bayes, thesis",
            "creator": "Markdown, Matplotlib mathtext, Chrome, and PyMuPDF",
            "producer": "PyMuPDF",
        }
    )
    document.set_metadata(metadata)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_pdf, garbage=4, deflate=True, clean=True)
    document.close()


def verify_pdf(output_pdf: Path) -> tuple[int, int]:
    document = fitz.open(output_pdf)
    pages = document.page_count
    if pages != 6:
        raise RuntimeError(f"Research report must be exactly six pages; generated {pages} pages.")
    text = "\n".join(page.get_text("text") for page in document)
    searchable_text = re.sub(r"\s+", " ", text)
    required = [
        "Abstract",
        "Date: August 6, 2026",
        "Mathematical formulation and claim status",
        "Proposition 1",
        "Proposition 2",
        "Theorem 1",
        "Scientific method and experimental design",
        "75,965",
        "144,440",
        "225.09",
        "oracle one-step-look-ahead",
        "one-step net gain",
        "Results, metrics, and evaluation of conclusions",
        "3,145/14,818",
        "2,679/2,948",
        "250 completion tokens",
        "non-nested",
        "0.00781",
        "0.1427",
        "0.955156",
        "0.011933",
        "0.700305",
        "54.34%",
        "30/30",
        "Table 5",
        "Table 6",
        "legacy reduced-feature LightGBM control",
        "fixed temperature 0.6 and seed 7",
        "aggregate CUDA collection job-hours",
        "Savitzky-Golay",
        "References",
        "Unique challenges or problems: none at present",
        "September",
        "October",
        "November and early December",
    ]
    missing = [value for value in required if value not in searchable_text]
    if missing:
        raise RuntimeError(f"Required report content is missing: {missing}")
    forbidden = [
        "469 commits",
        "7400d3a6",
        "97.17%",
        "no-peer control",
        "one current-grader disagreement",
        "aggregate GPU-hours",
        "live overthinking errors",
        "universal optimum",
        "singular most optimal",
        "peer information improves",
        "A task-held-out stacked hybrid",
        "separate analysis view",
        "fully reproducible system",
        "Continuation value",
    ]
    present = [value for value in forbidden if value in searchable_text]
    if present:
        raise RuntimeError(f"Unsupported or stale report content is present: {present}")

    footer_limit = 748.0
    for page_index, page in enumerate(document, start=1):
        for block in page.get_text("blocks"):
            block_text = " ".join(str(block[4]).split())
            if not block_text:
                continue
            if block_text.startswith("SEMESTER 1 MATHEMATICAL RESEARCH REPORT"):
                continue
            if block_text.startswith("ADITYA BHATT"):
                continue
            if block_text.startswith("Page "):
                continue
            if float(block[3]) > footer_limit:
                raise RuntimeError(
                    f"Page {page_index} content crosses the footer safety limit: "
                    f"y={float(block[3]):.1f}, text={block_text[:80]!r}"
                )
    document.close()
    return pages, len(text)


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    work_dir = args.work_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    chrome = find_chrome(args.chrome)
    html_path = build_html(input_path, work_dir)
    raw_pdf = work_dir / "semester1_research_report_raw.pdf"
    print_html(chrome, html_path, raw_pdf, work_dir)
    add_headers_and_metadata(raw_pdf, output_path)
    pages, characters = verify_pdf(output_path)
    print(f"Chrome: {chrome}")
    print(f"HTML: {html_path}")
    print(f"PDF: {output_path}")
    print(f"Pages: {pages}")
    print(f"Extracted text characters: {characters}")
    print(f"Bytes: {output_path.stat().st_size}")
    if not args.keep_intermediates:
        shutil.rmtree(work_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
