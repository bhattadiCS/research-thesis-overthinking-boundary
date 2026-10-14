#!/usr/bin/env python3
"""Build the thesis progress report as a submission-ready PDF.

The pipeline is intentionally self-contained:

1. Convert the Markdown report to print-oriented HTML.
2. Render every inline and display LaTeX expression to vector SVG with
   Matplotlib MathText.
3. Print the HTML with a local Chromium/Chrome executable.
4. Add academic running headers, page numbers, and PDF metadata with PyMuPDF.

Example:
    python tools/build_progress_report_pdf.py
"""

from __future__ import annotations

import argparse
import hashlib
import html
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import fitz
import markdown
import matplotlib

matplotlib.use("Agg")

from matplotlib import mathtext  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "ThesisDocs" / "Thesis_Progress_Report_2026-07-30.md"
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "Thesis_Progress_Report_Aditya_Bhatt_2026-08-02.pdf"
DEFAULT_WORK_DIR = ROOT / "tmp" / "pdfs" / "progress_report_build"

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
    candidates: list[Path] = []
    if explicit:
        candidates.append(explicit)

    for name in ("chrome.exe", "chrome", "msedge.exe", "msedge"):
        resolved = shutil.which(name)
        if resolved:
            candidates.append(Path(resolved))

    candidates.extend(
        [
            Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
            Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        ]
    )

    playwright_root = Path.home() / "AppData" / "Local" / "ms-playwright"
    if playwright_root.exists():
        candidates.extend(
            sorted(
                playwright_root.glob("chromium-*/chrome-win64/chrome.exe"),
                reverse=True,
            )
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("No Chrome, Chromium, or Edge executable was found.")


def normalize_math(expression: str) -> str:
    expression = " ".join(expression.split())
    expression = expression.replace(r"\Pr", r"\mathrm{Pr}")
    expression = re.sub(r"\\ge(?![A-Za-z])", r"\\geq", expression)
    expression = re.sub(r"\\le(?![A-Za-z])", r"\\leq", expression)
    expression = expression.replace(r"\arg\max", r"\mathrm{arg\,max}")
    expression = expression.replace(r"\arg\min", r"\mathrm{arg\,min}")
    expression = expression.replace(r"\overline H", r"\overline{H}")
    expression = expression.replace(r"\overline U", r"\overline{U}")
    expression = expression.replace(r"\overline G", r"\overline{G}")
    expression = re.sub(r"\\mathbf\s+([A-Za-z])", r"\\mathbf{\1}", expression)
    expression = re.sub(
        r"\\frac\s*([0-9A-Za-z])\s*([0-9A-Za-z])",
        r"\\frac{\1}{\2}",
        expression,
    )
    return expression


def render_math_svg(expression: str, destination: Path, display: bool) -> None:
    normalized = normalize_math(expression)
    font_size = 12.2 if display else 10.5
    mathtext.math_to_image(
        f"${normalized}$",
        destination,
        prop=FontProperties(family="DejaVu Serif", size=font_size),
        dpi=180,
        format="svg",
        color="#111111",
    )


def protect_math(markdown_text: str, math_dir: Path) -> tuple[str, dict[str, str]]:
    math_dir.mkdir(parents=True, exist_ok=True)
    replacements: dict[str, str] = {}
    sequence = 0

    def make_replacement(expression: str, display: bool) -> str:
        nonlocal sequence
        sequence += 1
        kind = "block" if display else "inline"
        digest = hashlib.sha256(
            f"{kind}\0{expression}".encode("utf-8")
        ).hexdigest()[:12]
        filename = f"math_{sequence:03d}_{digest}.svg"
        destination = math_dir / filename
        render_math_svg(expression, destination, display)

        token = f"MATH{kind.upper()}TOKEN{sequence:04d}END"
        alt = html.escape(expression, quote=True)
        if display:
            replacement = (
                '<div class="display-math">'
                f'<img src="math/{filename}" alt="{alt}">'
                "</div>"
            )
        else:
            replacement = (
                f'<img class="inline-math" src="math/{filename}" alt="{alt}">'
            )
        replacements[token] = replacement
        return f"\n\n{token}\n\n" if display else token

    protected = re.sub(
        r"\$\$(.*?)\$\$",
        lambda match: make_replacement(match.group(1).strip(), True),
        markdown_text,
        flags=re.DOTALL,
    )
    protected = re.sub(
        r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)",
        lambda match: make_replacement(match.group(1).strip(), False),
        protected,
        flags=re.DOTALL,
    )
    return protected, replacements


def extract_front_matter(markdown_text: str) -> tuple[dict[str, str], str]:
    lines = markdown_text.splitlines()
    metadata: dict[str, str] = {}

    if lines and lines[0].startswith("# "):
        metadata["Document"] = lines[0][2:].strip()

    body_start = None
    for index, line in enumerate(lines):
        if line.startswith("## Scope, evidence policy"):
            body_start = index
            break
        match = re.match(r"\*\*(.+?):\*\*\s*(.+?)\s*$", line)
        if match:
            metadata[match.group(1)] = match.group(2).rstrip("\\").strip()

    if body_start is None:
        raise ValueError("Could not locate the report body.")

    body_lines = lines[body_start:]
    body_lines[0] = "# Scope, Evidence Policy, and Executive Findings"
    return metadata, "\n".join(body_lines)


def replace_tokens(rendered_html: str, replacements: dict[str, str]) -> str:
    result = rendered_html
    for token, replacement in replacements.items():
        result = result.replace(f"<p>{token}</p>", replacement)
        result = result.replace(token, replacement)
    return result


def build_html(markdown_path: Path, work_dir: Path) -> Path:
    source = markdown_path.read_text(encoding="utf-8").translate(DASH_TRANSLATION)
    metadata, body = extract_front_matter(source)
    math_dir = work_dir / "math"
    protected, replacements = protect_math(body, math_dir)

    converter = markdown.Markdown(
        extensions=["tables", "fenced_code", "sane_lists", "toc"],
        extension_configs={
            "toc": {
                "permalink": False,
                "toc_depth": "1-2",
                "title": "Contents",
            }
        },
        output_format="html5",
    )
    article_html = replace_tokens(converter.convert(protected), replacements)
    toc_html = converter.toc

    project = metadata.get("Project", "Overthinking Boundary in Reasoning Models")
    student = metadata.get("Student", "Aditya Bhatt")
    program = metadata.get(
        "Program",
        "M.S. in Applied and Computational Mathematics, Johns Hopkins University",
    )
    advisor = metadata.get("Research Advisor", "Dr. Zerotti Woods")
    reader = metadata.get("Proposed Second Reader", "Dr. Moustapha Pemy")
    report_date = metadata.get("Report date", "August 2, 2026")
    completion = metadata.get("Target completion", "December 2026")
    snapshot = metadata.get("Repository snapshot", "469 commits")

    css = r"""
@page {
  size: Letter;
  margin: 0.66in 0.58in 0.66in 0.58in;
}

* {
  box-sizing: border-box;
}

html {
  font-size: 9.4pt;
}

body {
  margin: 0;
  color: #151515;
  font-family: Georgia, "Times New Roman", serif;
  line-height: 1.34;
  text-rendering: optimizeLegibility;
}

.cover {
  min-height: 9.55in;
  display: flex;
  flex-direction: column;
  justify-content: center;
  text-align: center;
  break-after: page;
}

.cover .institution {
  margin-bottom: 0.18in;
  color: #17365d;
  font-family: Arial, Helvetica, sans-serif;
  font-size: 10pt;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.cover .rule {
  width: 1.1in;
  margin: 0.14in auto 0.28in;
  border-top: 2px solid #17365d;
}

.cover h1 {
  margin: 0;
  color: #111111;
  font-family: Arial, Helvetica, sans-serif;
  font-size: 25pt;
  font-weight: 700;
  line-height: 1.14;
}

.cover h2 {
  max-width: 6.7in;
  margin: 0.22in auto 0.45in;
  color: #303030;
  font-size: 14pt;
  font-weight: 400;
  line-height: 1.38;
}

.cover .submission {
  margin-bottom: 0.36in;
  color: #17365d;
  font-family: Arial, Helvetica, sans-serif;
  font-size: 9pt;
  font-weight: 700;
  letter-spacing: 0.09em;
  text-transform: uppercase;
}

.cover table {
  width: 5.8in;
  margin: 0 auto;
  border: 0;
  font-size: 10pt;
  table-layout: auto;
}

.cover td {
  border: 0;
  padding: 0.055in 0.08in;
  vertical-align: top;
}

.cover td:first-child {
  width: 1.72in;
  text-align: right;
  font-family: Arial, Helvetica, sans-serif;
  font-size: 8.4pt;
  font-weight: 700;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}

.cover td:last-child {
  text-align: left;
}

.toc-page {
  break-after: page;
}

.toc-page > h1 {
  margin-top: 0;
}

.toc {
  columns: 2;
  column-gap: 0.34in;
  font-family: Arial, Helvetica, sans-serif;
  font-size: 8.1pt;
  line-height: 1.26;
}

.toc ul {
  margin: 0;
  padding-left: 0;
  list-style: none;
}

.toc li {
  margin: 0 0 0.055in;
  break-inside: avoid;
}

.toc li li {
  margin-left: 0.15in;
  color: #404040;
  font-size: 7.7pt;
}

.toc a {
  color: #151515;
  text-decoration: none;
}

article > h1 {
  break-before: page;
}

article > h1:first-child {
  break-before: auto;
}

h1, h2, h3 {
  color: #17365d;
  font-family: Arial, Helvetica, sans-serif;
  page-break-after: avoid;
}

h1 {
  margin: 0 0 0.17in;
  padding-bottom: 0.065in;
  border-bottom: 1.25px solid #17365d;
  font-size: 16.2pt;
  line-height: 1.18;
}

.cover h1 {
  padding-bottom: 0;
  border-bottom: 0;
}

h2 {
  margin: 0.19in 0 0.075in;
  font-size: 12.2pt;
  line-height: 1.2;
}

h3 {
  margin: 0.15in 0 0.055in;
  color: #2a2a2a;
  font-size: 10.3pt;
  line-height: 1.2;
}

h3#61-freeze-and-repair-the-evidence-base,
h3#65-multi-objective-search {
  break-before: page;
}

p {
  margin: 0 0 0.078in;
  orphans: 3;
  widows: 3;
}

strong {
  color: #111111;
}

ul, ol {
  margin: 0.04in 0 0.09in;
  padding-left: 0.24in;
}

li {
  margin: 0 0 0.035in;
}

li > ul, li > ol {
  margin-top: 0.025in;
}

blockquote {
  margin: 0.1in 0.18in;
  padding: 0.08in 0.14in;
  border-left: 3px solid #6a7c91;
  background: #f5f7f9;
}

code {
  overflow-wrap: anywhere;
  color: #17283a;
  font-family: "Cascadia Mono", Consolas, monospace;
  font-size: 0.88em;
}

pre {
  margin: 0.08in 0 0.12in;
  padding: 0.09in 0.11in;
  border: 1px solid #c8ced5;
  border-radius: 2px;
  background: #f6f7f8;
  font-family: "Cascadia Mono", Consolas, monospace;
  font-size: 7.7pt;
  line-height: 1.26;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  break-inside: avoid;
}

table {
  width: 100%;
  margin: 0.07in 0 0.14in;
  border-collapse: collapse;
  border-top: 1.25px solid #38495b;
  border-bottom: 1.25px solid #38495b;
  font-family: Arial, Helvetica, sans-serif;
  font-size: 7.45pt;
  line-height: 1.19;
  table-layout: auto;
}

thead {
  display: table-header-group;
}

tr {
  break-inside: avoid;
  page-break-inside: avoid;
}

th {
  padding: 0.045in 0.04in;
  border-bottom: 1px solid #6a7887;
  background: #e8edf2;
  color: #17283a;
  font-weight: 700;
  text-align: left;
  vertical-align: bottom;
}

td {
  padding: 0.04in;
  border-bottom: 0.45px solid #c8ced5;
  vertical-align: top;
  overflow-wrap: anywhere;
  hyphens: auto;
}

tbody tr:nth-child(even) td {
  background: #f7f8fa;
}

table.wide {
  font-size: 6.65pt;
  line-height: 1.13;
}

table.ultra-wide {
  font-size: 6.1pt;
  line-height: 1.08;
}

table.wide th, table.wide td {
  padding: 0.032in 0.027in;
}

.display-math {
  max-width: 100%;
  margin: 0.09in 0 0.13in;
  padding: 0.055in 0.07in;
  overflow: hidden;
  text-align: center;
  break-inside: avoid;
}

.display-math img {
  display: block;
  max-width: 96%;
  max-height: 0.62in;
  width: auto;
  height: auto;
  margin: 0 auto;
}

.inline-math {
  display: inline-block;
  max-width: 100%;
  height: 1.18em;
  width: auto;
  margin: 0 0.055em;
  vertical-align: -0.24em;
}

a {
  color: #17365d;
  text-decoration: underline;
  text-decoration-thickness: 0.5px;
}

.appendix-note {
  color: #505050;
  font-size: 8.5pt;
}
"""

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Thesis Research Progress Report - Aditya Bhatt</title>
  <style>{css}</style>
</head>
<body>
  <section class="cover">
    <div class="institution">Johns Hopkins University</div>
    <h1>Thesis Research<br>Progress Report</h1>
    <div class="rule"></div>
    <h2>{html.escape(project)}</h2>
    <div class="submission">Advisor Submission Copy</div>
    <table aria-label="Report metadata">
      <tr><td>Student</td><td>{html.escape(student)}</td></tr>
      <tr><td>Program</td><td>{html.escape(program)}</td></tr>
      <tr><td>Research Advisor</td><td>{html.escape(advisor)}</td></tr>
      <tr><td>Second Reader</td><td>{html.escape(reader)}</td></tr>
      <tr><td>Report Date</td><td>{html.escape(report_date)}</td></tr>
      <tr><td>Target Completion</td><td>{html.escape(completion)}</td></tr>
      <tr><td>Repository Snapshot</td><td>{html.escape(snapshot)}</td></tr>
    </table>
  </section>
  <section class="toc-page">
    <h1>Contents</h1>
    {toc_html}
  </section>
  <article>
    {article_html}
  </article>
  <script>
    document.querySelectorAll("table").forEach((table) => {{
      const columns = table.querySelectorAll("thead th").length;
      if (columns >= 8) table.classList.add("ultra-wide");
      else if (columns >= 5) table.classList.add("wide");
    }});
  </script>
</body>
</html>
"""

    html_path = work_dir / "progress_report.html"
    html_path.write_text(document, encoding="utf-8")
    return html_path


def print_html(chrome: Path, html_path: Path, raw_pdf: Path, work_dir: Path) -> None:
    profile = work_dir / "chrome_profile"
    profile.mkdir(parents=True, exist_ok=True)
    command = [
        str(chrome),
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-extensions",
        "--disable-background-networking",
        "--disable-component-update",
        "--disable-default-apps",
        "--disable-sync",
        "--hide-scrollbars",
        "--no-pdf-header-footer",
        f"--user-data-dir={profile}",
        f"--print-to-pdf={raw_pdf}",
        html_path.resolve().as_uri(),
    ]
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=180,
        creationflags=creation_flags,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Chrome PDF generation failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    deadline = time.time() + 20
    while time.time() < deadline:
        if raw_pdf.exists() and raw_pdf.stat().st_size > 10_000:
            return
        time.sleep(0.25)
    raise RuntimeError(f"Chrome did not create a usable PDF at {raw_pdf}.")


def add_headers_and_metadata(
    raw_pdf: Path, output_pdf: Path, report_date: str = "August 2, 2026"
) -> None:
    source = fitz.open(raw_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    total_pages = source.page_count
    header_color = (0.09, 0.21, 0.36)
    rule_color = (0.42, 0.49, 0.57)
    text_color = (0.24, 0.24, 0.24)

    for page_index, page in enumerate(source):
        page_number = page_index + 1
        width = page.rect.width
        height = page.rect.height

        if page_number > 1:
            page.insert_text(
                (42, 24),
                "THESIS RESEARCH PROGRESS REPORT",
                fontname="helv",
                fontsize=7.2,
                color=header_color,
            )
            right_header = f"ADITYA BHATT  |  {report_date.upper()}"
            text_width = fitz.get_text_length(
                right_header, fontname="helv", fontsize=7.2
            )
            page.insert_text(
                (width - 42 - text_width, 24),
                right_header,
                fontname="helv",
                fontsize=7.2,
                color=header_color,
            )
            page.draw_line(
                (42, 31),
                (width - 42, 31),
                color=rule_color,
                width=0.45,
            )

            page_label = f"Page {page_number} of {total_pages}"
            label_width = fitz.get_text_length(
                page_label, fontname="helv", fontsize=7.2
            )
            page.draw_line(
                (42, height - 31),
                (width - 42, height - 31),
                color=rule_color,
                width=0.45,
            )
            page.insert_text(
                ((width - label_width) / 2, height - 18),
                page_label,
                fontname="helv",
                fontsize=7.2,
                color=text_color,
            )

    metadata = source.metadata or {}
    metadata.update(
        {
            "title": "Thesis Research Progress Report",
            "author": "Aditya Bhatt",
            "subject": (
                "Overthinking Boundary in Reasoning Large Language Models: "
                "Dynamic Stop Policies via Optimal Stopping"
            ),
            "keywords": (
                "optimal stopping, reasoning models, overthinking, "
                "token efficiency, computational mathematics"
            ),
            "creator": "Codex PDF workflow using Markdown, MathText, Chrome, and PyMuPDF",
            "producer": "PyMuPDF",
        }
    )
    source.set_metadata(metadata)
    source.save(output_pdf, garbage=4, deflate=True, clean=True)
    source.close()


def verify_pdf(output_pdf: Path, expected_title: str) -> tuple[int, int]:
    document = fitz.open(output_pdf)
    page_count = document.page_count
    if page_count < 10:
        raise RuntimeError(f"Unexpectedly short PDF: {page_count} pages.")
    if document.metadata.get("title") != expected_title:
        raise RuntimeError("PDF title metadata is missing or incorrect.")

    extracted = "\n".join(page.get_text("text") for page in document)
    extracted_lower = extracted.lower()
    required = [
        "Thesis Research",
        "Research Objectives",
        "Comprehensive Model",
        "Data Engineering",
        "Overthinking Detection",
        "Detailed Commit",
        "Actionable Fall",
        "Reproducibility and Reporting Corrections",
    ]
    missing = [value for value in required if value.lower() not in extracted_lower]
    if missing:
        raise RuntimeError(f"Required PDF text is missing: {missing}")

    document.close()
    return page_count, len(extracted)


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    work_dir = args.work_dir.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    source_metadata, _ = extract_front_matter(
        input_path.read_text(encoding="utf-8")
    )

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chrome = find_chrome(args.chrome)
    html_path = build_html(input_path, work_dir)
    raw_pdf = work_dir / "progress_report_raw.pdf"
    print_html(chrome, html_path, raw_pdf, work_dir)
    add_headers_and_metadata(
        raw_pdf,
        output_path,
        source_metadata.get("Report date", "August 2, 2026"),
    )
    page_count, extracted_characters = verify_pdf(
        output_path, "Thesis Research Progress Report"
    )

    print(f"Chrome: {chrome}")
    print(f"HTML: {html_path}")
    print(f"PDF: {output_path}")
    print(f"Pages: {page_count}")
    print(f"Extracted text characters: {extracted_characters}")
    print(f"Bytes: {output_path.stat().st_size}")

    if not args.keep_intermediates:
        shutil.rmtree(work_dir)
        parent = work_dir.parent
        while parent != ROOT and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
