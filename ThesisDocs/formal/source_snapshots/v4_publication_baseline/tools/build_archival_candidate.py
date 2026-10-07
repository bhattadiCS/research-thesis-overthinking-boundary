"""Convert a reviewed formal thesis to PDF/A-2b without editing its content.

Use the pinned isolated authoring environment and veraPDF executable. The main
thesis builder owns all formatting, including the front matter. This converter
accepts the current source dynamically, preserves every page's text and word
coordinates, and verifies the exact saved output. Historical v1 artifacts are
protected. Archival conformance is separate from institutional acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

import fitz
import numpy as np
import pikepdf
from pikepdf import pdfa


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tmp/pdfs/formal_thesis/digital/source.pdf"
OUTPUT = ROOT / "output/pdf/Masters_Thesis_Formal_v2_Aditya_Bhatt.pdf"
RECEIPTS = ROOT / "ThesisDocs/archival"
PINNED_PIKEPDF_VERSION = "10.16.0"
VERAPDF_JAR_SHA256 = "889075253fb9df4db5482efb8f8208fb3b4f2e00f5f7e1b1e31edf6fb4b69bb6"
HISTORICAL_FILES = {
    ROOT / "output/pdf/Masters_Thesis_Draft_v1_Aditya_Bhatt.pdf",
    ROOT / "output/pdf/Masters_Thesis_Archival_Candidate_v1_Aditya_Bhatt.pdf",
    *(RECEIPTS / name for name in (
        "candidate_build_manifest.json", "candidate_verapdf_2b.xml",
        "visual_review.json", "tool_provenance.json", "formatting_audit.md", "README.md",
    )),
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_label(path: Path) -> str:
    path = path.resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def roman(value: int) -> str:
    result = ""
    for number, text in ((1000, "m"), (900, "cm"), (500, "d"), (400, "cd"),
                         (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
                         (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")):
        while value >= number:
            result += text
            value -= number
    return result


def check_targets(source: Path, targets: list[Path], *, replace: bool) -> None:
    """Validate write targets before conversion; preserve source and v1 files."""
    resolved = [target.resolve() for target in targets]
    if len(resolved) != len(set(resolved)):
        raise ValueError("Output and receipt targets must be distinct")
    for target in resolved:
        if target in HISTORICAL_FILES or any(
                target.exists() and protected.exists() and target.samefile(protected)
                for protected in HISTORICAL_FILES):
            raise ValueError(f"Historical v1 artifact cannot be overwritten: {target}")
        if target == source or (target.exists() and target.samefile(source)):
            raise ValueError("Conversion output or receipt would overwrite the source PDF")
        if target.exists() and not replace:
            raise FileExistsError(f"Output exists; use --replace for a deliberate v2 rebuild: {target}")
    existing = [target for target in resolved if target.exists()]
    for index, target in enumerate(existing):
        if any(target.samefile(other) for other in existing[index + 1:]):
            raise ValueError("Output and receipt targets cannot alias the same file")


def document_structure(document: fitz.Document) -> dict:
    return {"outline": document.get_toc(), "page_labels": document.get_page_labels(),
            "language": document.xref_get_key(document.pdf_catalog(), "Lang")[1],
            "tagged_structure_present": document.xref_get_key(document.pdf_catalog(), "StructTreeRoot")[0] != "null",
            "document_information": {key: document.metadata[key]
                                     for key in ("title", "author", "subject", "keywords", "creator")}}


def font_audit(document: fitz.Document) -> dict:
    fonts = {font[0]: font for page in document for font in page.get_fonts(full=True)}
    missing = [font[3] for xref, font in fonts.items()
               if not xref or not document.extract_font(xref)[3]]
    if missing:
        raise ValueError(f"Font resources without embedded font data: {missing}")
    sizes = []
    undersized = []
    mathematical_spans = 0
    for number, page in enumerate(document, 1):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if not span["text"].strip():
                        continue
                    # KaTeX includes naturally smaller mathematical scripts and
                    # invisible struts; neither is ordinary body/table/code type.
                    if "KaTeX_" in span["font"]:
                        mathematical_spans += 1
                        continue
                    size = float(span["size"])
                    sizes.append(size)
                    if size < 10 - .005:
                        undersized.append({"physical_page": number, "font": span["font"],
                                           "size_points": size, "text": span["text"][:120]})
    if not sizes:
        raise ValueError("No ordinary text was available for the type-size audit")
    if undersized:
        raise ValueError(f"Ordinary text below the formal thesis's 10pt minimum: {undersized[:12]}")
    return {"embedded_font_objects": len(fonts),
            "font_names": sorted({font[3] for font in fonts.values()}),
            "unembedded_fonts": [],
            "ordinary_text_minimum_size_points": min(sizes),
            "ordinary_text_size_requirement_points": 10,
            "type_size_rounding_tolerance_points": .005,
            "ordinary_text_undersized_spans": [],
            "mathematical_spans_excluded_from_ordinary_type_minimum": mathematical_spans,
            "type_size_scope": "Extractable ordinary text includes body, tables, labels and code. Formal v2 has no separate footnotes. Mathematical scripts/KaTeX struts are excluded; lettering within raster figures is assessed by visual review, not this text-extractor measurement."}


def printed_pagination(document: fitz.Document, *, left_margin: float) -> dict:
    """Find and check the Roman-to-Arabic transition without a fixed page count."""
    entries = []
    body_start = None
    for index, page in enumerate(document):
        centers = (page.rect.width / 2, (left_margin + page.rect.width - 72) / 2)
        words = [word for word in page.get_text("words")
                 if word[1] >= page.rect.height - 92
                 and min(abs((word[0] + word[2]) / 2 - center) for center in centers) <= 24]
        texts = [word[4] for word in words]
        if index == 0:
            if texts:
                raise ValueError("The cover must not have a printed page number")
            entries.append({"physical_page": 1, "printed_number": None})
            continue
        if len(texts) != 1:
            raise ValueError(f"Expected one centered bottom folio on physical page {index + 1}: {texts}")
        label = texts[0]
        center = (words[0][0] + words[0][2]) / 2
        if min(abs(center - reference) for reference in centers) > .75:
            raise ValueError(f"Folio is not centered on page or text rectangle: {index + 1}")
        if label == "1" and body_start is None:
            body_start = index
            heading = " ".join(page.get_text().split())
            if not re.match(r"Chapter\s+1\s+Introduction\b", heading, flags=re.I):
                raise ValueError("Arabic page 1 must begin the Introduction")
        expected = roman(index + 1) if body_start is None else str(index - body_start + 1)
        if label != expected:
            raise ValueError(f"Incorrect folio on physical page {index + 1}: {label!r}, expected {expected!r}")
        entries.append({"physical_page": index + 1, "printed_number": label,
                        "horizontal_center_points": center,
                        "center_reference": "page" if abs(center - centers[0]) <= .75 else "text rectangle"})
    if body_start is None:
        raise ValueError("No Introduction with Arabic page 1 was found")
    return {"front_matter_pages_including_cover": body_start,
            "introduction_physical_page": body_start + 1,
            "last_arabic_page": len(document) - body_start, "pages": entries}


def ink_bounds(pixmap: fitz.Pixmap) -> list[float]:
    rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
    ink = np.any(rgb < 245, axis=2)
    columns = np.flatnonzero(np.any(ink, axis=0))
    rows = np.flatnonzero(np.any(ink, axis=1))
    if not len(columns):
        raise ValueError("Blank rendered page")
    return [float(columns[0]) / 2, float(rows[0]) / 2,
            float(columns[-1] + 1) / 2, float(rows[-1] + 1) / 2]


def check_ink_margin(bounds: list[float], rect: fitz.Rect, left: float, page: int) -> None:
    # One raster pixel at 144 dpi is half a point. Allow that rounding only.
    x0, y0, x1, y1 = bounds
    if x0 < left - .5 or y0 < 71.5 or x1 > rect.width - 71.5 or y1 > rect.height - 71.5:
        raise ValueError(f"Visible ink outside the declared margins on page {page}: {bounds}")


def render_one_document(source: Path, render_dir: Path) -> None:
    """Render one PDF per fresh process, avoiding cross-document image caches."""
    identity = sha(source)
    render_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    with fitz.open(source) as document:
        for number, page in enumerate(document, 1):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False, colorspace=fitz.csRGB)
            target = render_dir / f"page_{number:03d}.png"
            pixmap.save(target)
            pages.append({"page": number, "png_sha256": sha(target),
                          "rgb_sha256": hashlib.sha256(pixmap.samples).hexdigest()})
    if sha(source) != identity:
        raise ValueError("PDF changed during isolated rendering")
    record = {"pdf": relative_label(source), "pdf_sha256": identity,
              "engine": f"PyMuPDF {fitz.VersionBind}", "dpi": 144,
              "process_scope": "one PDF", "pages": pages}
    (render_dir / "render_manifest.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")


def isolated_render(source: Path, render_dir: Path) -> dict:
    result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                             "--_render-document", str(source), str(render_dir)],
                            capture_output=True, check=False)
    if result.returncode:
        raise ValueError(f"Isolated rendering failed: {result.stderr.decode(errors='replace')[:1500]}")
    record = json.loads((render_dir / "render_manifest.json").read_text(encoding="utf-8"))
    if record["pdf_sha256"] != sha(source):
        raise ValueError("Isolated rendering does not match the current PDF identity")
    return record


def compare_documents(source: Path, output: Path, *, margin_profile: str, render_dir: Path) -> dict:
    left = 108.0 if margin_profile == "print" else 72.0
    source_render_dir = render_dir / "source_comparison"
    source_render = isolated_render(source, source_render_dir)
    output_render = isolated_render(output, render_dir)
    with fitz.open(source) as original, fitz.open(output) as converted:
        if not len(original) or len(original) != len(converted):
            raise ValueError("Conversion changed the source page count")
        if len(source_render["pages"]) != len(original) or len(output_render["pages"]) != len(converted):
            raise ValueError("Isolated rendering has an incorrect page count")
        before_structure, after_structure = document_structure(original), document_structure(converted)
        if before_structure != after_structure:
            raise ValueError("Conversion changed outline, logical page labels, language, metadata or tagging presence")
        if before_structure["language"] in {"null", ""}:
            raise ValueError("The formal source must declare its document language")
        if not before_structure["page_labels"]:
            raise ValueError("The formal source must have logical page labels")
        fonts = font_audit(converted)
        pagination = printed_pagination(converted, left_margin=left)
        comparisons = []
        render_dir.mkdir(parents=True, exist_ok=True)
        for number, (before, after) in enumerate(zip(original, converted), 1):
            if before.rect != after.rect or tuple(before.rect) != (0.0, 0.0, 612.0, 792.0):
                raise ValueError(f"Changed or unsupported non-Letter page size: {number}")
            if not before.get_text().strip():
                raise ValueError(f"Blank source page: {number}")
            if before.get_text() != after.get_text() or before.get_text("words") != after.get_text("words"):
                raise ValueError(f"Conversion changed source text or word coordinates on page {number}")
            source_png = source_render_dir / f"page_{number:03d}.png"
            output_png = render_dir / f"page_{number:03d}.png"
            if (sha(source_png) != source_render["pages"][number - 1]["png_sha256"]
                    or sha(output_png) != output_render["pages"][number - 1]["png_sha256"]):
                raise ValueError(f"Isolated page render changed before comparison: {number}")
            a, b = fitz.Pixmap(source_png), fitz.Pixmap(output_png)
            if (a.width, a.height, a.n) != (b.width, b.height, b.n):
                raise ValueError(f"Conversion changed rendered dimensions: {number}")
            if a.n != 3:
                raise ValueError(f"Isolated page renders must have exactly three RGB channels: {number}")
            delta = np.abs(np.frombuffer(a.samples, dtype=np.uint8).astype(np.int16)
                           - np.frombuffer(b.samples, dtype=np.uint8).astype(np.int16))
            changed = int(np.count_nonzero(np.any(delta.reshape(a.height, a.width, 3), axis=2)))
            maximum, fraction = int(delta.max()), changed / (a.width * a.height)
            if maximum > 1 or fraction >= .02:
                raise ValueError(f"Conversion changed page appearance beyond ICC rounding: {number}; max channel delta {maximum}, changed fraction {fraction}")
            before_bounds, after_bounds = ink_bounds(a), ink_bounds(b)
            check_ink_margin(before_bounds, before.rect, left, number)
            check_ink_margin(after_bounds, after.rect, left, number)
            comparisons.append({"page": number, "width": b.width, "height": b.height,
                                "source_rgb_pixels_sha256": hashlib.sha256(a.samples).hexdigest(),
                                "rgb_pixels_sha256": hashlib.sha256(b.samples).hexdigest(),
                                "text_identical": True, "word_geometry_identical": True,
                                "pixels_identical": changed == 0, "changed_pixels": changed,
                                "changed_pixel_fraction": fraction, "max_channel_difference": maximum,
                                "source_ink_bounds_points": before_bounds,
                                "output_ink_bounds_points": after_bounds})
        return {"pages": len(converted), **fonts, "document_structure": after_structure,
                "printed_pagination": pagination, "margin_profile": margin_profile,
                "required_margins_inches": {"left": left / 72, "right": 1, "top": 1, "bottom": 1},
                "margin_measurement": {"dpi": 144, "nonwhite_channel_threshold": 245,
                                       "raster_rounding_tolerance_points": .5, "defects": 0},
                "unchanged_text_and_word_geometry_pages": len(converted),
                "intentional_content_edit_pages": [],
                "pixel_identical_pages": sum(page["pixels_identical"] for page in comparisons),
                "pixel_max_channel_difference": max(page["max_channel_difference"] for page in comparisons),
                "pixel_max_changed_fraction": max(page["changed_pixel_fraction"] for page in comparisons),
                "render_isolation": {"method": "one fresh process per PDF; no shared document/image caches",
                                     "source_render_manifest": relative_label(source_render_dir / "render_manifest.json"),
                                     "source_render_manifest_sha256": sha(source_render_dir / "render_manifest.json"),
                                     "output_render_manifest": relative_label(render_dir / "render_manifest.json"),
                                     "output_render_manifest_sha256": sha(render_dir / "render_manifest.json")},
                "page_comparisons": comparisons}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--receipts-dir", type=Path, default=RECEIPTS)
    parser.add_argument("--receipt-prefix", default="formal_v2_digital")
    parser.add_argument("--render-dir", type=Path, help="Render base directory; each exact output hash gets a separate child")
    parser.add_argument("--margin-profile", choices=("digital", "print"), default="digital")
    parser.add_argument("--expected-source-sha256", help="Optional explicit current-source identity from the main build")
    parser.add_argument("--verapdf-jar", required=True, type=Path)
    parser.add_argument("--replace", action="store_true", help="Rebuild the named v2 outputs; historical v1 files remain protected")
    args = parser.parse_args(argv)
    builder_sha = sha(Path(__file__))
    if pikepdf.__version__ != PINNED_PIKEPDF_VERSION:
        raise ValueError(f"Use pikepdf {PINNED_PIKEPDF_VERSION} from the recorded isolated authoring environment")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.receipt_prefix):
        raise ValueError("Receipt prefix must be a simple filename component")
    source, output, receipts = args.source.resolve(), args.output.resolve(), args.receipts_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Reviewed formal source does not exist: {source}")
    source_sha = sha(source)
    if args.expected_source_sha256 and source_sha != args.expected_source_sha256:
        raise ValueError("Source differs from the explicit current main-build identity")
    jar = args.verapdf_jar.resolve()
    if not jar.is_file() or sha(jar) != VERAPDF_JAR_SHA256:
        raise ValueError("veraPDF jar differs from the pinned, publisher-verified 1.30.2 executable")
    xml_path = receipts / f"{args.receipt_prefix}_verapdf_2b.xml"
    manifest_path = receipts / f"{args.receipt_prefix}_build_manifest.json"
    check_targets(source, [output, xml_path, manifest_path], replace=args.replace)
    render_base = (args.render_dir or ROOT / "tmp/pdfs/formal_thesis" / args.receipt_prefix / "archival_review").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)
    repairs = []
    with pikepdf.open(source) as pdf:
        for number, page in enumerate(pdf.pages, 1):
            if "/Trans" in page.obj:
                if len(page.obj["/Trans"]):
                    raise ValueError("Nonempty page transitions require separate review")
                del page.obj["/Trans"]
                repairs.append({"page": number, "repair": "remove empty /Trans dictionary"})
        for obj in pdf.objects:
            if (isinstance(obj, pikepdf.Dictionary)
                    and obj.get("/Subtype") == pikepdf.Name("/CIDFontType2")
                    and "/CIDToGIDMap" not in obj):
                if str(obj.get("/BaseFont")) != "/ArialMT" or "/FontFile2" not in obj["/FontDescriptor"]:
                    raise ValueError("Unrecognized CID font without a map requires separate review")
                obj["/CIDToGIDMap"] = pikepdf.Name("/Identity")
                repairs.append({"object": list(obj.objgen), "font": str(obj["/BaseFont"]),
                                "repair": "add explicit /Identity CIDToGIDMap; all-page geometry and pixels verified"})
        report = pdfa.save(pdf, output, "2b")
        if not report.passed:
            raise ValueError(f"PDF/A preparation failed: {report.summary()}")
        pike_summary, preparations = report.summary(), list(report.prepared.describe())
    output_sha = sha(output)
    render_dir = render_base / output_sha
    result = subprocess.run(["java", "-jar", str(jar), "--flavour", "2b", "--format", "xml", str(output)],
                            capture_output=True, check=False)
    tree = ET.fromstring(result.stdout)
    reports, details = list(tree.iter("validationReport")), list(tree.iter("details"))
    if (result.returncode != 0 or len(reports) != 1 or len(details) != 1
            or reports[0].get("isCompliant") != "true"
            or details[0].get("failedRules") != "0" or details[0].get("failedChecks") != "0"):
        raise ValueError(f"veraPDF rejected the exact saved output: {result.stderr.decode(errors='replace')[:1000]}")
    comparison = compare_documents(source, output, margin_profile=args.margin_profile, render_dir=render_dir)
    if sha(source) != source_sha or sha(output) != output_sha or sha(Path(__file__)) != builder_sha:
        raise ValueError("Source, saved output or converter changed during conversion/verification")
    xml_path.write_bytes(result.stdout)
    record = {"schema": "thesis-archival-conversion-v2", "profile": "PDF/A-2b",
              "status": "validated_conversion", "source": relative_label(source), "source_sha256": source_sha,
              "output": relative_label(output), "sha256": output_sha, "bytes": output.stat().st_size,
              "builder": relative_label(Path(__file__)), "builder_sha256": builder_sha,
              "pikepdf_version": pikepdf.__version__, "numpy_version": np.__version__, "pikepdf_report": pike_summary,
              "manual_structural_repairs": repairs, "pikepdf_preparation": preparations,
              "authoritative_validator": "veraPDF Greenfield 1.30.2", "validator_jar_sha256": sha(jar),
              "validator_exit_code": result.returncode, "validator_report": relative_label(xml_path),
              "validator_report_sha256": sha(xml_path), "validator_details": details[0].attrib,
              "validator_report_attributes": reports[0].attrib, "render_engine": f"PyMuPDF {fitz.VersionBind}",
              "comparison_dpi": 144, "render_dir": relative_label(render_dir),
              "approval_scope": {"semester_approvals": "user-confirmed complete",
                                 "institutional_format_acceptance": "not asserted by technical conversion validation"},
              "limitations": ["PDF/A-2b conformance is separate from institutional acceptance.",
                              "No PDF/UA accessibility or semantic-tagging conformance is asserted.",
                              "October 2026 is the title-page month; no actual submission event or institutional format acceptance is asserted."],
              "visual_review": "required for the exact converted PDF before delivery", **comparison}
    manifest_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: record[key] for key in ("status", "source_sha256", "output", "sha256", "pages",
                                                 "margin_profile", "validator_details", "unchanged_text_and_word_geometry_pages",
                                                 "pixel_identical_pages", "pixel_max_channel_difference")}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--_render-document":
        render_one_document(Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve())
    else:
        main()
