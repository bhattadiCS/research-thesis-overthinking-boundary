"""Read-only, separately rendered Poppler checks of recorded v5 editions.

Writes only a fresh directory beneath this script. Never changes PDFs, source
manifests, prior receipts, or publication files. Contact sheets require human
review; successful extraction/rendering alone is not a visual approval.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

from PIL import Image, ImageDraw


OWNED = Path(__file__).resolve().parent
ROOT = OWNED.parents[4]
BINARY_DIR = ROOT / "tmp/pdfs/recheck_v2/poppler/poppler-26.09.0/Library/bin"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normal(value):
    return re.sub(r"\s+", "", value)


def roman(n):
    result = ""
    for value, token in [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"),
                         (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
                         (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]:
        while n >= value:
            result += token
            n -= value
    return result


def caption_list_entries(text, kind):
    """Parse the title column separately from its right-aligned folio.

    Poppler -layout emits the folio beside the first line, before continuation
    lines. It must not be treated as the last token of the full caption.
    Require an explicit two-space column gap and a single integer per entry.
    """
    entries = []
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or re.fullmatch(r"[ivxlcdm]+", line):
            continue
        if line in ("List of tables", "List of tables (continued)",
                    "List of figures", "List of figures (continued)"):
            continue
        start = re.match(rf"^{kind.title()}\s+(\d+)\.", line)
        if start:
            if current:
                entries.append(current)
            current = {"number": int(start[1]), "title_lines": [], "folios": []}
        if current is None:
            raise ValueError(f"Unexpected text before first {kind} list entry: {raw!r}")
        folio = re.search(r"\s{2,}(\d+)\s*$", raw)
        if folio:
            current["folios"].append(int(folio[1]))
            line = raw[:folio.start()].strip()
        current["title_lines"].append(line)
    if current:
        entries.append(current)
    for entry in entries:
        entry["caption"] = " ".join(entry.pop("title_lines"))
        if len(entry["folios"]) != 1:
            raise ValueError(f"Expected exactly one folio in {kind} list entry {entry}")
        entry["body_page"] = entry.pop("folios")[0]
    return entries


def run(command, log):
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8", errors="replace")
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"Poppler exit {completed.returncode}; inspect {log}")


def inspect(edition, output, dpi):
    source_manifest = ROOT / f"ThesisDocs/formal/build_manifest_{edition}_v5.json"
    conversion_manifest = ROOT / f"ThesisDocs/archival/formal_v5_{edition}_build_manifest.json"
    source = read(source_manifest)
    conversion = read(conversion_manifest)
    pdf = ROOT / conversion["output"]
    front = source["front_matter_pages"]
    expected_pages = conversion["pages"]
    before = {str(path.relative_to(ROOT).as_posix()): sha(path)
              for path in (pdf, source_manifest, conversion_manifest)}
    if before[conversion["output"]] != conversion["sha256"]:
        raise ValueError("PDF bytes do not match conversion manifest")
    folder = output / edition
    folder.mkdir()
    text_path = folder / "text_layout.txt"
    run([str(BINARY_DIR / "pdftotext.exe"), "-layout", str(pdf), str(text_path)],
        folder / "text_stderr.txt")
    run([str(BINARY_DIR / "pdftoppm.exe"), "-r", str(dpi), "-png", str(pdf), str(folder / "page")],
        folder / "render_stderr.txt")
    pages = text_path.read_text(encoding="utf-8").split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    files = sorted(folder.glob("page-*.png"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))
    checks = []

    def check(name, passed, details=None):
        checks.append({"name": name, "passed": bool(passed), "details": details})

    check("extracted_page_count", len(pages) == expected_pages, len(pages))
    check("rendered_page_count", len(files) == expected_pages, len(files))
    check("recorded_front_count", isinstance(front, int) and 2 <= front < len(pages), front)
    folios = []
    for physical, text in enumerate(pages, 1):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        expected = None if physical == 1 else roman(physical) if physical <= front else str(physical-front)
        actual = lines[-1] if lines else None
        folios.append({"physical_page": physical, "expected": expected, "last_nonempty_line": actual,
                       "matches": expected is None or actual == expected})
    check("printed_folios", all(row["matches"] for row in folios))
    front_pages = pages[:front]

    def starts(title):
        return [i+1 for i, text in enumerate(front_pages)
                if [line.strip() for line in text.splitlines() if line.strip()]
                and [line.strip() for line in text.splitlines() if line.strip()][0] == title]

    positions = {title: starts(title) for title in ("Abstract", "Table of contents", "List of tables", "List of figures")}
    check("unique_front_section_starts", all(len(v) == 1 for v in positions.values()), positions)
    heading_checks = []
    caption_checks = []
    if all(len(v) == 1 for v in positions.values()):
        abstract, toc, tables, figures = [positions[name][0] for name in positions]
        check("front_section_order", abstract == 2 and abstract < toc < tables < figures <= front)
        table_text = "".join(pages[tables-1:figures-1])
        figure_text = "".join(pages[figures-1:front])
        table_entries = caption_list_entries(table_text,"table")
        figure_entries = caption_list_entries(figure_text,"figure")
        table_numbers = [e["number"] for e in table_entries]
        figure_numbers = [e["number"] for e in figure_entries]
        check("listed_17_scientific_tables", table_numbers == list(range(1,18)), table_numbers)
        check("listed_6_figures", figure_numbers == list(range(1,7)), figure_numbers)
        toc_text = "".join(pages[toc-1:tables-1])
        for entry in source["contents_body_entries"]:
            physical = entry["body_page"] + front
            present = physical <= len(pages) and normal(entry["title"]) in normal(pages[physical-1])
            listed = normal(entry["title"]+str(entry["body_page"])) in normal(toc_text)
            heading_checks.append({**entry, "physical_page": physical, "present_on_target": present, "listed": listed})
        check("all_toc_targets", all(row["present_on_target"] and row["listed"] for row in heading_checks))
        for kind, listed_entries in (("table", table_entries), ("figure", figure_entries)):
            for number, (title, body_page) in enumerate(zip(source[kind+"_titles"], source[kind+"_body_pages"]),1):
                caption = f"{kind.title()} {number}. {title}"
                physical = body_page + front
                matches = [i+1 for i, text in enumerate(pages[front:],front) if normal(caption) in normal(text)]
                listed = [e for e in listed_entries if e["number"] == number]
                exact_listed = len(listed) == 1 and normal(listed[0]["caption"]) == normal(caption) and listed[0]["body_page"] == body_page
                caption_checks.append({"kind": kind, "number": number, "physical_page": physical,
                                       "matching_body_pages": matches, "listed": exact_listed, "actual_list_entry": listed,
                                       "matches": matches == [physical]})
        check("caption_targets", all(row["matches"] and row["listed"] for row in caption_checks))
    sheets = []
    for first in range(0,len(files),6):
        sheet = Image.new("RGB",(1260,1650),"#d5d5d5")
        draw = ImageDraw.Draw(sheet)
        for slot, file in enumerate(files[first:first+6]):
            with Image.open(file) as original:
                tile = original.convert("RGB")
            tile.thumbnail((400,790))
            x, y = 12+(slot%3)*416, 28+(slot//3)*820
            sheet.paste(tile,(x,y))
            draw.text((x,y-18),f"{edition} physical {first+slot+1}",fill="black")
        target = folder / f"contact_{first//6+1:02}.png"
        sheet.save(target)
        sheets.append({"path": str(target.relative_to(ROOT).as_posix()), "sha256": sha(target),
                       "physical_pages": list(range(first+1,min(first+7,len(files)+1)))})
    changed = [path for path, expected in before.items() if sha(ROOT/path) != expected]
    check("inputs_unchanged_during_audit", not changed, changed)
    result = {"edition": edition, "pdf": conversion["output"], "sha256": before[conversion["output"]],
              "pages": len(pages), "front_matter_pages": front, "dpi": dpi, "checks": checks,
              "printed_folios": folios, "toc_targets": heading_checks, "caption_targets": caption_checks,
              "text_path": str(text_path.relative_to(ROOT).as_posix()), "text_sha256": sha(text_path),
              "page_renders": [{"physical_page": int(p.stem.rsplit("-",1)[1]), "path": str(p.relative_to(ROOT).as_posix()),
                                "sha256": sha(p)} for p in files], "contact_sheets": sheets,
              "input_bindings": before, "all_extraction_and_render_checks_pass": all(c["passed"] for c in checks),
              "human_visual_review_required": True}
    print(json.dumps({"edition":edition,"pages":len(pages),"pass":result["all_extraction_and_render_checks_pass"],
                      "defects":[c for c in checks if not c["passed"]]}),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory",required=True)
    parser.add_argument("--edition",choices=["digital","print"])
    parser.add_argument("--dpi",type=int,default=144)
    args = parser.parse_args()
    output = Path(args.directory).resolve()
    if OWNED not in output.parents:
        raise ValueError("Use a fresh subdirectory beneath the verifier directory")
    if output.exists():
        raise FileExistsError("Choose a new receipt directory; preliminary receipts are immutable")
    if not 72 <= args.dpi <= 300:
        raise ValueError("DPI must be between 72 and 300")
    output.mkdir(parents=True)
    editions = (args.edition,) if args.edition else ("digital","print")
    result = {"schema":"independent-v5-poppler-checks-v1", "created_utc":datetime.now(timezone.utc).isoformat(),
              "script":str(Path(__file__).resolve().relative_to(ROOT).as_posix()), "script_sha256":sha(__file__),
              "binaries":{str(p.relative_to(ROOT).as_posix()):sha(p) for p in (BINARY_DIR/"pdftotext.exe",BINARY_DIR/"pdftoppm.exe")},
              "scope":"Independent Poppler extraction and rendering, dynamic front matter and caption targets; no PDF writes.",
              "editions":[inspect(edition,output,args.dpi) for edition in editions]}
    result["all_extraction_and_render_checks_pass"] = all(e["all_extraction_and_render_checks_pass"] for e in result["editions"])
    target = output / "poppler_checks.json"
    target.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"receipt":str(target.relative_to(ROOT).as_posix()),"sha256":sha(target),
                      "pass":result["all_extraction_and_render_checks_pass"]}),flush=True)
    if not result["all_extraction_and_render_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
