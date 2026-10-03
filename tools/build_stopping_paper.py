"""Build the standalone research paper with figures and parsed TeX equations.

This builder uses ReportLab for searchable PDF text, embedded Windows fonts,
matplotlib for scientific figures/equations, and PyMuPDF for layout checks and
page renders. A final build refuses incomplete live evidence. Use
--allow-pending-live only for an explicitly identified interim research draft.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
from pathlib import Path
import re
import subprocess
import sys
from xml.sax.saxutils import escape

from bs4 import BeautifulSoup
import fitz
import markdown
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.mathtext import math_to_image
from matplotlib.font_manager import FontProperties
from PIL import Image as PillowImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.platypus import Image, KeepTogether, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "ThesisDocs/paper/overthinking_stopping_paper_v1.md"
FIGURES = PAPER.parent / "figures"
WORK = ROOT / "tmp/pdfs/stopping_paper"
OUTPUT = ROOT / "output/pdf/Overthinking_Stopping_Research_Paper_v1_Aditya_Bhatt.pdf"
EVIDENCE = ROOT / "research/outputs/thesis_v1/evidence"
ONLINE = ROOT / "research/outputs/semester2/online_stopping_20261002"
ADVERSARIAL = ONLINE / "adversarial_live"
SOURCE_PATHS = [
    PAPER, ROOT / "tools/build_stopping_paper.py",
    ROOT / "research/mathematical_foundations.md",
    ROOT / "research/tests/test_mathematical_foundations.py",
    ROOT / "tools/recompute_thesis_evidence.py",
    ROOT / "tools/compute_progress_report_review_metrics.py",
    ROOT / "research/outputs/experiments_v2/blackwell_5day_tournament_v1/blackwell_tournament_report.json",
    ROOT / "data_manifest_v1.json", ROOT / "software_provenance_v1.json", ROOT / "requirements.lock.txt",
    ROOT / "research/adversarial_tasks_v1.jsonl", ROOT / "research/adversarial_gold_v1.jsonl",
]
WIDTH = 468


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_figures() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.right": False,
                        "axes.spines.top": False, "savefig.dpi": 260})
    rows = read_rows(EVIDENCE / "boundary_domain_step_metrics.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), constrained_layout=True)
    for ax, domain in zip(axes.flat, ["arc", "gpqa", "gsm8k", "math"]):
        group = [row for row in rows if row["domain"] == domain]
        values = lambda key: [float(row[key]) for row in group]
        steps = values("step")
        ax.plot(steps, values("accuracy"), "o-", color="#0072B2", markersize=3, label="Accuracy")
        ax.fill_between(steps, values("accuracy_ci_low"), values("accuracy_ci_high"), color="#0072B2", alpha=.16)
        ax.plot(steps, values("net_drift"), "s-", color="#D55E00", markersize=3, label="Net gain")
        ax.fill_between(steps, values("net_drift_ci_low"), values("net_drift_ci_high"), color="#D55E00", alpha=.16)
        ax.axhline(0, color=".4", linewidth=.7)
        ax.set(title=domain.upper(), xlabel="Current response step", ylabel="Accuracy or net gain")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(FIGURES / "population_continuation.png")
    plt.close(fig)
    effects = read_rows(EVIDENCE / "algorithm_v2_normalized_effects.csv")
    labels = {"N1 LOCO": "Threshold / held-out cell", "N1 LOMO": "Threshold / held-out model",
              "N2a": "Gradient-boosted probe", "N2b": "Isotonic calibration",
              "N2c": "Lagged logistic features", "N3": "Empirical-Bayes hazards", "N4": "Churn threshold"}
    chosen = [row for row in effects if row["experiment"] in labels]
    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    for index, row in enumerate(chosen):
        low, high, value = (float(row[key]) for key in ["ci_95_low", "ci_95_high", "mean_controlled_effect"])
        ax.plot([low, high], [index, index], color="#0072B2", linewidth=2)
        ax.scatter([value], [index], color="#0072B2", s=30, zorder=3)
    ax.axvline(0, color=".35", linewidth=.8)
    ax.set(yticks=range(len(chosen)), yticklabels=[labels[row["experiment"]] for row in chosen],
           xlabel="Matched mean step-utility difference per trajectory")
    ax.invert_yaxis()
    fig.savefig(FIGURES / "controlled_estimator_effects.png")
    plt.close(fig)


def register_fonts() -> None:
    fonts = {"PaperTimes": "times.ttf", "PaperTimesBold": "timesbd.ttf", "PaperTimesItalic": "timesi.ttf",
             "PaperTimesBoldItalic": "timesbi.ttf", "PaperMono": "cour.ttf"}
    for name, filename in fonts.items():
        path = Path("C:/Windows/Fonts") / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required embedded font unavailable: {path}")
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily("PaperTimes", normal="PaperTimes", bold="PaperTimesBold",
                                 italic="PaperTimesItalic", boldItalic="PaperTimesBoldItalic")


def equation_image(tex: str, index: int) -> Path:
    # mathtext accepts a braced bold numeral; standard TeX also permits \mathbf1.
    tex = re.sub(r"\\mathbf([0-9])", r"\\mathbf{\1}", tex)
    tex = re.sub(r"\\(mathbb|mathcal)\s+([A-Za-z])", r"\\\1{\2}", tex)
    tex = re.sub(r"\\le\b", r"\\leq", tex)
    tex = " ".join(tex.split())
    output = WORK / f"equation_{index:02}.png"
    math_to_image(f"${tex}$", output, dpi=300, format="png", color="black", prop=FontProperties(size=13))
    return output


def paragraph_markup(element) -> str:
    value = element.decode_contents()
    # ReportLab supports these semantic text tags and embedded hyperlinks.
    value = re.sub(r"<a([^>]*)>", r"<link\1>", value).replace("</a>", "</link>")
    value = re.sub(r"<code>", '<font name="PaperMono" size="9">', value).replace("</code>", "</font>")
    value = value.replace("<br/>", "<br />").replace("<br>", "<br />")
    return value


def image_flowable(path: Path, *, maximum_height: float = 390, equation: bool = False) -> Image:
    with PillowImage.open(path) as image:
        pixel_width, pixel_height = image.size
    ratio = min(WIDTH / pixel_width, maximum_height / pixel_height)
    if equation:
        # Preserve the rendered math font size rather than enlarging short
        # equations to the full text width. The PNG is rendered at 300 dpi.
        ratio = min(ratio, 72 / 300)
    flowable = Image(str(path), width=pixel_width * ratio, height=pixel_height * ratio)
    flowable.hAlign = "CENTER"
    return flowable


def build_story(source: str, *, leading: float) -> tuple[list, int]:
    equations = []
    def replace_equation(match):
        equations.append(match.group(1).strip())
        return f"\n\nPAPER_DISPLAY_EQUATION_{len(equations)}\n\n"
    prose = re.sub(r"\$\$\s*([\s\S]*?)\s*\$\$", replace_equation, source)
    document = BeautifulSoup(markdown.markdown(prose, extensions=["tables", "fenced_code", "nl2br"]), "html.parser")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="PaperBody", fontName="PaperTimes", fontSize=11.5, leading=leading,
                              spaceAfter=8, allowWidows=0, allowOrphans=0))
    styles.add(ParagraphStyle(name="PaperTitle", parent=styles["PaperBody"], fontName="PaperTimesBold",
                              fontSize=17, leading=21, spaceAfter=13, keepWithNext=True))
    styles.add(ParagraphStyle(name="PaperSection", parent=styles["PaperBody"], fontName="PaperTimesBold",
                              fontSize=13, leading=16, spaceBefore=12, spaceAfter=7, keepWithNext=True))
    styles.add(ParagraphStyle(name="PaperSubsection", parent=styles["PaperSection"], fontSize=11.5,
                              leading=14, spaceBefore=9, spaceAfter=6))
    styles.add(ParagraphStyle(name="PaperTable", fontName="PaperTimes", fontSize=9.5, leading=12.3, spaceAfter=0))
    styles.add(ParagraphStyle(name="PaperReference", parent=styles["PaperBody"], fontSize=10, leading=12.5, spaceAfter=4))
    story = []
    def append_kept_block(blocks):
        # ReportLab's keepWithNext does not propagate through a KeepTogether
        # wrapper. Include preceding headings in the same group so a heading
        # cannot be stranded on the page before its table or figure.
        while story and isinstance(story[-1], Paragraph) and getattr(story[-1].style, "keepWithNext", False):
            blocks.insert(0, story.pop())
        story.append(KeepTogether(blocks))
    reference_section = False
    children = [element for element in document.children if getattr(element, "name", None)]
    used_captions = set()
    for element_index, element in enumerate(children):
        if element_index in used_captions:
            continue
        if not getattr(element, "name", None):
            continue
        tag = element.name
        if tag in {"h1", "h2", "h3"}:
            if element.get_text().strip() == "References":
                reference_section = True
            style = styles[{"h1": "PaperTitle", "h2": "PaperSection", "h3": "PaperSubsection"}[tag]]
            story.append(Paragraph(paragraph_markup(element), style))
        elif tag == "p":
            text = element.get_text().strip()
            equation = re.fullmatch(r"PAPER_DISPLAY_EQUATION_(\d+)", text)
            if equation:
                index = int(equation.group(1))
                path = equation_image(equations[index - 1], index)
                story.extend([Spacer(1, 4), image_flowable(path, maximum_height=85, equation=True), Spacer(1, 10)])
            elif element.find("img"):
                img = element.find("img")
                path = (PAPER.parent / img["src"]).resolve()
                if not path.is_relative_to(ROOT):
                    raise ValueError("Figure path escapes repository")
                figure_blocks = [Spacer(1, 5), image_flowable(path), Spacer(1, 8)]
                if element_index + 1 < len(children):
                    caption = children[element_index + 1]
                    if caption.name == "p" and caption.get_text().startswith("Figure "):
                        figure_blocks.append(Paragraph(paragraph_markup(caption), styles["PaperBody"]))
                        used_captions.add(element_index + 1)
                append_kept_block(figure_blocks)
            else:
                story.append(Paragraph(paragraph_markup(element), styles["PaperReference" if reference_section else "PaperBody"]))
        elif tag == "table":
            rows = []
            for row in element.find_all("tr"):
                rows.append([Paragraph(paragraph_markup(cell), styles["PaperTable"]) for cell in row.find_all(["th", "td"])])
            if not rows or any(len(row) != len(rows[0]) for row in rows):
                raise ValueError("Malformed paper table")
            columns = len(rows[0])
            first = {3: 255, 4: 190, 5: 160}.get(columns, WIDTH / columns)
            widths = [first] + [(WIDTH - first) / (columns - 1)] * (columns - 1)
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, -1), "PaperTimes"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f3")),
                ("LINEBELOW", (0, 0), (-1, 0), .8, colors.black),
                ("LINEBELOW", (0, 1), (-1, -1), .3, colors.HexColor("#cccccc")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            append_kept_block([Spacer(1, 5), table, Spacer(1, 10)])
        elif tag == "pre":
            story.append(Preformatted(element.get_text(), ParagraphStyle(name="Code", fontName="PaperMono",
                                     fontSize=8.5, leading=11, spaceBefore=5, spaceAfter=9)))
        else:
            raise ValueError(f"Unsupported paper block: {tag}")
    return story, len(equations)


def page_decoration(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("PaperTimes", 8.5)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(72, 752, "Causal and Cost Aware Stopping")
    canvas.drawRightString(540, 752, "Research draft")
    canvas.drawCentredString(306, 39, str(document.page))
    canvas.restoreState()


def audit_pdf(path: Path) -> dict:
    defects = []
    with fitz.open(path) as document:
        for index, page in enumerate(document):
            if not page.get_text().strip():
                defects.append({"page": index + 1, "kind": "blank"})
            for block in page.get_text("dict")["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        x0, y0, x1, y1 = span["bbox"]
                        if x0 < 70 or x1 > 542 or y0 < 27 or y1 > 768:
                            defects.append({"page": index + 1, "kind": "out_of_bounds", "text": span["text"], "bbox": span["bbox"]})
            # Fail on an unrendered math marker or replacement glyph.
            text = page.get_text()
            if "PAPER_DISPLAY_EQUATION_" in text or "\ufffd" in text:
                defects.append({"page": index + 1, "kind": "unrendered_content"})
        fonts = {font[0] for page in document for font in page.get_fonts(full=True) if font[0]}
        unembedded = [xref for xref in fonts if not document.extract_font(xref)[3]]
        if unembedded:
            defects.append({"kind": "unembedded_fonts", "xrefs": unembedded})
        if defects:
            (WORK / "layout_defects.json").write_text(json.dumps(defects, indent=2), encoding="utf-8")
            raise ValueError(f"Paper PDF has {len(defects)} layout defects")
        rendered = WORK / "rendered"
        rendered.mkdir(exist_ok=True)
        for index, page in enumerate(document):
            page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False).save(rendered / f"page_{index+1:03}.png")
        return {"pages": len(document), "blank_pages": 0, "text_margin_defects": 0,
                "unembedded_fonts": 0, "rendered_pages": len(document),
                "visual_review": "PNG inspection required before delivery", "pdfa_conformance": "not asserted"}


def dependency_map(*, allow_pending: bool) -> dict:
    files = set(SOURCE_PATHS)
    files.update(EVIDENCE.glob("*.csv"))
    files.add(EVIDENCE / "manifest.json")
    files.update(FIGURES.glob("*.png"))
    for directory in [ONLINE, ADVERSARIAL]:
        files.update(path for path in directory.rglob("*") if path.is_file())
    files.update(path for path in (ROOT / "research/outputs/semester2/prefix_model_v1").rglob("*") if path.is_file())
    files.update(path for path in (ROOT / "research/reports/thesis_failure_audit_v1").rglob("*") if path.is_file())
    files.update(ROOT / name for name in ["research/prefix_stopping_model.py", "research/train_prefix_stopping_model.py",
                 "research/learned_online_stopping_controller.py", "research/run_learned_online_stopping.py",
                 "research/analyze_online_stopping_results.py", "tools/summarize_failure_audit.py",
                 "tools/analyze_live_stopping_uncertainty.py"])
    identity = {path.relative_to(ROOT).as_posix(): {"sha256": sha(path), "bytes": path.stat().st_size}
                for path in sorted(files) if path.is_file()}
    master = json.loads((ROOT / "data_manifest_v1.json").read_text(encoding="utf-8"))
    return {"schema_version": "stopping-paper-dependencies-v1",
            "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "interpretation": "Authoring/evidence dependencies; upstream experimental bytes are selected by the separately verified master data freeze.",
            "master_data_content_fingerprint": master["content_fingerprint"],
            "live_evidence_complete_required": not allow_pending,
            "files": identity,
            "claim_sources": {"theorems": "research/mathematical_foundations.md",
                              "boundary_curves": "research/outputs/thesis_v1/evidence/boundary_domain_step_metrics.csv",
                              "controlled_effects": "research/outputs/thesis_v1/evidence/algorithm_v2_normalized_effects.csv",
                              "causal_detector_scores": "research/outputs/thesis_v1/evidence/tournament_balanced_summary.csv",
                              "historical_replay": "research/outputs/thesis_v1/evidence/offline_replay_metrics.csv",
                              "historical_stacked_diagnostic": "research/outputs/experiments_v2/blackwell_5day_tournament_v1/blackwell_tournament_report.json",
                              "runtime_and_adversarial": [ONLINE.relative_to(ROOT).as_posix(), ADVERSARIAL.relative_to(ROOT).as_posix()]}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-pending-live", action="store_true")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--leading", type=float, default=15.0)
    args = parser.parse_args()
    if OUTPUT.exists() and not args.replace:
        raise FileExistsError(f"Paper output exists; use --replace explicitly: {OUTPUT}")
    source = PAPER.read_text(encoding="utf-8")
    if not args.allow_pending_live:
        required = [ONLINE / "live_metrics.json", ADVERSARIAL / "live_metrics.json",
                    ONLINE / "latency_summary.json", ONLINE / "replay_pareto.csv",
                    ONLINE / "live_uncertainty.json", ADVERSARIAL / "live_uncertainty.json",
                    ONLINE / "learned_main/live_metrics.json", ONLINE / "learned_adversarial/live_metrics.json",
                    ONLINE / "learned_main/live_uncertainty.json", ONLINE / "learned_adversarial/live_uncertainty.json",
                    ROOT / "research/outputs/semester2/prefix_model_v1/evaluation.json"]
        missing = [path.relative_to(ROOT).as_posix() for path in required if not path.is_file()]
        if missing or "still executing at this manuscript revision" in source or "pending at this manuscript revision" in source:
            raise ValueError(f"Final paper refuses incomplete live evidence or stale pending text: {missing}")
        subprocess.run([sys.executable, str(ROOT / "tools/freeze_research_data.py"), "verify"], check=True, cwd=ROOT)
    for path in [WORK, OUTPUT.parent]:
        path.mkdir(parents=True, exist_ok=True)
    register_fonts()
    build_figures()
    story, equation_count = build_story(source, leading=args.leading)
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=(612, 792), rightMargin=72, leftMargin=72,
                            topMargin=64, bottomMargin=58, title="Causal and Cost Aware Stopping for Iterative Language Model Revision",
                            author="Aditya Bhatt", subject="Research manuscript draft; no venue acceptance or deployment guarantee asserted")
    def embedded_canvas(*values, **options):
        options["initialFontName"] = "PaperTimes"
        return pdf_canvas.Canvas(*values, **options)
    doc.build(story, onFirstPage=page_decoration, onLaterPages=page_decoration, canvasmaker=embedded_canvas)
    # Remove ReportLab's unused Helvetica resource before embedded-font checks.
    with fitz.open(OUTPUT) as document:
        for page in document:
            page.clean_contents(sanitize=True)
        cleaned = WORK / "embedded_paper.pdf"
        document.save(cleaned, garbage=4, deflate=True)
    OUTPUT.write_bytes(cleaned.read_bytes())
    audit = audit_pdf(OUTPUT)
    dependencies = dependency_map(allow_pending=args.allow_pending_live)
    dependency_path = PAPER.parent / "dependency_map_v1.json"
    dependency_path.write_text(json.dumps(dependencies, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    manifest = {"schema_version": "stopping-paper-build-v1", "output": OUTPUT.relative_to(ROOT).as_posix(),
                "sha256": sha(OUTPUT), "source_sha256": sha(PAPER), "dependency_map_sha256": sha(dependency_path),
                "word_count": len(source.split()), "display_equations": equation_count,
                "body_leading_pt": args.leading, "audit": audit, "interim_live_results_permitted": args.allow_pending_live}
    (PAPER.parent / "paper_build_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                                           encoding="utf-8", newline="\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
