"""Compile and audit an anonymous NeurIPS main-track draft in official style.

The portable Tectonic engine and untouched official style are fetched by
fetch_neurips_build_dependencies.py. The next conference cycle's rules must
be rechecked before actual submission. This command does not submit anything.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

import fitz

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ThesisDocs/neurips"
WORK = ROOT / "tmp/neurips_build/compiled"
OUTPUT = ROOT / "output/pdf/NeurIPS_Anonymous_Draft_v1.pdf"
ENGINE = ROOT / "tmp/neurips_build/tectonic/tectonic.exe"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((SOURCE / "template_provenance.json").read_text(encoding="utf-8"))
    official_style_sha = provenance["retained_template_files"]["ThesisDocs/neurips/template/neurips_2026.sty"]
    if sha(SOURCE / "neurips_2026.sty") != official_style_sha:
        raise ValueError("The official style was modified")
    source_files = sorted(p for p in SOURCE.glob("*.tex") if p.is_file())
    identity_pattern = re.compile(r"\baditya\b|\bbhatt\b|johns\s+hopkins|zerotti|moustapha|C:[/\\]|/Users/", re.I)
    for path in source_files:
        if identity_pattern.search(path.read_text(encoding="utf-8")):
            raise ValueError(f"Identifying text in anonymous source: {path.name}")
    command = [str(ENGINE), "--untrusted", "--keep-logs", "--keep-intermediates",
               "--outdir", str(WORK), str(SOURCE / "paper.tex")]
    process = subprocess.run(command, cwd=SOURCE, text=True, capture_output=True)
    (WORK / "compile_output.txt").write_text(process.stdout + process.stderr, encoding="utf-8")
    print(process.stdout)
    print(process.stderr)
    if process.returncode:
        raise RuntimeError(f"Tectonic failed with exit code {process.returncode}")
    aux = (WORK / "paper.aux").read_text(encoding="utf-8")
    label = re.search(r"\\newlabel\{sec:main-text-end\}\{\{[^}]*\}\{(\d+)\}", aux)
    if label is None:
        raise ValueError("Main-text end label missing from auxiliary output")
    main_end_page = int(label[1])
    if main_end_page > 9:
        raise ValueError(f"Main content occupies {main_end_page} pages, exceeding nine")
    log = (WORK / "paper.log").read_text(encoding="utf-8", errors="replace")
    overfull = re.findall(r"Overfull \\[hv]box \(([^)]*)\)", log)
    unresolved = [line for line in log.splitlines() if
                  "undefined" in line.lower() and ("reference" in line.lower() or "citation" in line.lower())]
    if overfull or unresolved:
        raise ValueError(f"TeX layout/reference findings: overfull={overfull}, unresolved={unresolved}")
    document = fitz.open(WORK / "paper.pdf")
    extracted = "\n".join(page.get_text() for page in document)
    if identity_pattern.search(extracted):
        raise ValueError("Identifying text in anonymous PDF")
    blank_pages = [i + 1 for i, page in enumerate(document) if not page.get_text().strip()]
    if blank_pages:
        raise ValueError(f"Blank pages: {blank_pages}")
    fonts = {font[0] for page in document for font in page.get_fonts(full=True)}
    missing_fonts = [xref for xref in fonts if not document.extract_font(xref)[3]]
    if missing_fonts:
        raise ValueError(f"Fonts without embedded data: {missing_fonts}")
    document.set_metadata({"title": "Cost aware stopping boundaries in reasoning language models",
                           "author": "", "subject": "Anonymous research draft; not submitted"})
    document.save(OUTPUT, garbage=4, deflate=True)
    render_dir = WORK / "rendered"
    render_dir.mkdir(exist_ok=True)
    for index, page in enumerate(document):
        page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False).save(render_dir / f"page_{index+1:03}.png")
    manifest = {
        "output": OUTPUT.relative_to(ROOT).as_posix(), "sha256": sha(OUTPUT),
        "main_content_end_page": main_end_page, "total_pages": len(document),
        "anonymous_text_scan": "pass", "blank_pages": blank_pages,
        "embedded_fonts": len(fonts), "overfull_boxes": overfull,
        "unresolved_references": unresolved,
        "official_style_sha256": official_style_sha,
        "engine_version": "0.17.0", "engine_executable_sha256": sha(ENGINE),
        "source_files": {p.relative_to(ROOT).as_posix(): sha(p) for p in source_files},
        "data_manifest_sha256": sha(ROOT / "data_manifest_v1.json"),
        "status": "Future-cycle preparation in published 2026 format; not submitted",
        "visual_review": "Required for these exact bytes before delivery",
    }
    (SOURCE / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
