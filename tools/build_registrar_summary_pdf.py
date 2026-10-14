#!/usr/bin/env python3
"""Build the concise Registrar-facing thesis status summary PDF."""

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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "ThesisDocs" / "Thesis_Progress_Report_Registrar_Summary_2026-08-02.md"
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "Thesis_Progress_Report_Aditya_Bhatt_Fall_2026.pdf"
DEFAULT_WORK_DIR = ROOT / "tmp" / "pdfs" / "registrar_summary_build"

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
    raise FileNotFoundError("No supported Chrome/Edge executable was found.")


def split_summary(source: str) -> list[str]:
    source = source.translate(DASH_TRANSLATION)
    parts = [part.strip() for part in source.split("<!-- PAGE BREAK -->")]
    if len(parts) != 3:
        raise ValueError(f"Expected exactly three summary pages, found {len(parts)}.")
    return parts


def markdown_html(text: str) -> str:
    converter = markdown.Markdown(
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )
    return converter.convert(text)


def build_html(input_path: Path, work_dir: Path) -> Path:
    pages = split_summary(input_path.read_text(encoding="utf-8"))
    rendered = [markdown_html(page) for page in pages]
    page_one_identity = """
    <div class="identity">
      <div class="institution">JOHNS HOPKINS UNIVERSITY</div>
      <div class="title">MATH ACM 625.804 THESIS STATUS REPORT</div>
      <div class="subtitle">Registrar Attachment Summary | Fall 2026 | Completion target: late November / early December 2026</div>
      <table class="metadata">
        <tr><th>Student</th><td>Aditya Bhatt</td><th>Report period</th><td>Semester 1</td></tr>
        <tr><th>Advisor</th><td>Dr. Zerotti Woods</td><th>Second reader</th><td>Dr. Moustapha Pemy</td></tr>
        <tr><th>Report date</th><td colspan="3">August 2026</td></tr>
      </table>
    </div>
    """

    pages_html = []
    for index, body in enumerate(rendered):
        prefix = page_one_identity if index == 0 else ""
        pages_html.append(f'<section class="page">{prefix}{body}</section>')

    css = r"""
@page { size: Letter; margin: 0; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: white; }
body { color: #000; font-family: "Times New Roman", Times, serif; }
.page {
  position: relative;
  width: 8.5in;
  height: 11in;
  padding: 0.55in 0.58in 0.48in;
  overflow: hidden;
  page-break-after: always;
}
.page:last-child { page-break-after: auto; }
.identity { margin-bottom: 0.13in; }
.institution {
  color: #000;
  font: 700 8.2pt Arial, Helvetica, sans-serif;
  letter-spacing: 0;
  text-align: left;
  margin-bottom: 0.06in;
}
.title {
  color: #000;
  font: 700 16pt "Times New Roman", Times, serif;
  text-align: left;
  margin-bottom: 0.035in;
  padding-bottom: 0.04in;
  border-bottom: 1px solid #000;
}
.subtitle {
  color: #000;
  font: 9pt Arial, Helvetica, sans-serif;
  text-align: left;
  margin-bottom: 0.11in;
}
.metadata {
  width: 100%;
  border-collapse: collapse;
  font: 8.5pt Arial, Helvetica, sans-serif;
  line-height: 1.18;
  margin-bottom: 0.12in;
}
.metadata th, .metadata td { padding: 3.5px 5px; border-bottom: 0.5px solid #000; text-align: left; }
.metadata th { color: #000; background: #fff; font-weight: 700; width: 14%; }
.metadata td { width: 36%; }
h1, h2, h3 { color: #000; font-family: "Times New Roman", Times, serif; page-break-after: avoid; }
h1 { font-size: 16pt; line-height: 1.15; margin: 0 0 0.13in; padding-bottom: 0.05in; border-bottom: 1.1px solid #000; }
h2 { font-size: 12.4pt; line-height: 1.18; margin: 0.14in 0 0.07in; }
h3 { color: #000; font-size: 10.8pt; line-height: 1.18; margin: 0.11in 0 0.05in; }
p { font-size: 10.1pt; line-height: 1.3; margin: 0 0 0.09in; orphans: 2; widows: 2; }
strong { color: #000; }
ol, ul { margin: 0.03in 0 0.07in; padding-left: 0.22in; }
li { font-size: 10pt; line-height: 1.28; margin-bottom: 0.045in; }
table { width: 100%; border-collapse: collapse; margin: 0.045in 0 0.09in; font: 7.15pt Arial, Helvetica, sans-serif; line-height: 1.13; table-layout: auto; border-top: 1.1px solid #000; border-bottom: 1.1px solid #000; }
thead { display: table-header-group; }
tr { break-inside: avoid; page-break-inside: avoid; }
th { padding: 3px 4px; background: #fff; color: #000; border-bottom: 0.8px solid #000; text-align: left; vertical-align: bottom; }
td { padding: 3px 4px; background: #fff; border-bottom: 0.4px solid #000; vertical-align: top; overflow-wrap: anywhere; }
tbody tr:nth-child(even) td { background: #fff; }
code { color: #000; font: 0.92em Consolas, "Cascadia Mono", monospace; }
blockquote { margin: 0.06in 0.12in; padding: 0.05in 0.1in; border-left: 2px solid #000; background: #fff; }
"""
    document = f"""<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body>{''.join(pages_html)}</body></html>"""
    work_dir.mkdir(parents=True, exist_ok=True)
    html_path = work_dir / "registrar_summary.html"
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
        if raw_pdf.exists() and raw_pdf.stat().st_size > 10_000:
            return
        time.sleep(0.25)
    raise RuntimeError("Chrome did not create a usable PDF.")


def add_headers_and_metadata(raw_pdf: Path, output_pdf: Path) -> None:
    document = fitz.open(raw_pdf)
    total_pages = document.page_count
    header_color = (0, 0, 0)
    rule_color = (0, 0, 0)
    text_color = (0, 0, 0)
    for index, page in enumerate(document):
        width, height = page.rect.width, page.rect.height
        if index > 0:
            page.insert_text((42, 24), "THESIS STATUS SUMMARY", fontname="helv", fontsize=7.2, color=header_color)
            right = "ADITYA BHATT  |  FALL 2026"
            right_width = fitz.get_text_length(right, fontname="helv", fontsize=7.2)
            page.insert_text((width - 42 - right_width, 24), right, fontname="helv", fontsize=7.2, color=header_color)
            page.draw_line((42, 31), (width - 42, 31), color=rule_color, width=0.45)
        page_label = f"Page {index + 1} of {total_pages}"
        label_width = fitz.get_text_length(page_label, fontname="helv", fontsize=7.2)
        page.draw_line((42, height - 31), (width - 42, height - 31), color=rule_color, width=0.45)
        page.insert_text(((width - label_width) / 2, height - 18), page_label, fontname="helv", fontsize=7.2, color=text_color)

    metadata = document.metadata or {}
    metadata.update(
        {
            "title": "MATH ACM 625.804 Thesis Status Summary",
            "author": "Aditya Bhatt",
            "subject": "Registrar attachment summary for Fall 2026 thesis completion",
            "keywords": "thesis status, Registrar, optimal stopping, reasoning models",
            "creator": "Codex concise PDF workflow using Markdown, Chrome, and PyMuPDF",
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
    if pages < 2 or pages > 3:
        raise RuntimeError(f"Summary must be 2-3 pages; generated {pages} pages.")
    text = "\n".join(page.get_text("text") for page in document)
    required = [
        "Semester 1: progress to date",
        "Semester 1: what I completed and learned",
        "Semester 2: plan to completion",
        "late November / early December 2026",
        "Research question and terms",
        "How the scientific method was applied",
        "Correctness",
        "GPQA",
        "0.955",
        "0.943",
        "Registrar",
    ]
    missing = [value for value in required if value not in text]
    if missing:
        raise RuntimeError(f"Required summary content is missing: {missing}")
    forbidden = [
        "469 commits",
        "7400d3a6",
        "Evidence snapshot",
        "Problems that must be resolved",
        "model-domain-temperature cells",
        "scheduled for November 23-30",
        "December 7, 2026",
        "August 3-28",
        "August 31-September 25",
        "September 28-October 23",
        "October 26-November 30",
        "December 1-7",
    ]
    present = [value for value in forbidden if value in text]
    if present:
        raise RuntimeError(f"Forbidden stale or ambiguous summary content is present: {present}")
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
    raw_pdf = work_dir / "registrar_summary_raw.pdf"
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
