"""Read-only v5 PDF audit; writes only new audit receipts under the requested directory."""
from __future__ import annotations

import argparse
from collections import Counter
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pikepdf
import pymupdf


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def norm(value):
    return re.sub(r"\s+", "", value)


def roman(n):
    result = ""
    for value, token in [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"), (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]:
        while n >= value:
            result += token
            n -= value
    return result


def inherited(obj, key):
    while obj is not None:
        if key in obj:
            return obj[key]
        obj = obj.get("/Parent")
    return None


def audit_cmap(path):
    """Track actual text operators, including nested forms, for the generic folio font."""
    result = []
    with pikepdf.open(path) as pdf:
        seen_fonts = {}
        for page in pdf.pages:
            resources = inherited(page.obj, "/Resources")
            for key, font in resources.get("/Font", {}).items():
                if str(font.get("/Subtype", "")) == "/TrueType" and "Arial" in str(font.get("/BaseFont", "")):
                    seen_fonts[font.objgen] = (str(key), font)
        for font_id, (resource_name, target) in seen_fonts.items():
            cmap = target["/ToUnicode"]
            data = bytes(cmap.read_bytes()).decode("ascii")
            spaces = re.findall(r"begincodespacerange(.*?)endcodespacerange", data, re.S)
            code_width = max(len(source)//2 for source in re.findall(r"<([0-9a-fA-F]+)>", "".join(spaces)))
            used = Counter()
            used_pages = []
            odd_strings = []

            def walk(stream, resources, page_number, initial_font=None, depth=0):
                if depth > 16:
                    raise RuntimeError("Unexpectedly deep Form XObject nesting")
                active = initial_font
                stack = []
                for instruction in pikepdf.parse_content_stream(stream):
                    op = str(instruction.operator)
                    operands = instruction.operands
                    if op == "q":
                        stack.append(active)
                    elif op == "Q":
                        active = stack.pop() if stack else None
                    elif op == "Tf":
                        active = resources.get("/Font", {}).get(operands[0])
                    elif op == "Do":
                        form = resources.get("/XObject", {}).get(operands[0])
                        if form is not None and str(form.get("/Subtype", "")) == "/Form":
                            walk(form, form.get("/Resources", resources), page_number, active, depth + 1)
                    elif op in ("Tj", "TJ", "'", '"') and active is not None and active.objgen == font_id:
                        values = operands[0] if op == "TJ" else [operands[-1]]
                        for value in values:
                            if not isinstance(value, pikepdf.String):
                                continue
                            raw = bytes(value)
                            if len(raw) % code_width:
                                odd_strings.append({"page": page_number, "bytes_hex": raw.hex()})
                            for j in range(0, len(raw) - code_width + 1, code_width):
                                used[int.from_bytes(raw[j:j+code_width], "big")] += 1
                            if raw and page_number not in used_pages:
                                used_pages.append(page_number)

            for page_number, page in enumerate(pdf.pages, 1):
                walk(page, inherited(page.obj, "/Resources"), page_number)
            cmap = target["/ToUnicode"]
            data = bytes(cmap.read_bytes()).decode("ascii")
            malformed = []
            mapped = {}
            for section in re.findall(r"beginbfrange(.*?)endbfrange", data, re.S):
                for first, last, destination in re.findall(r"<([0-9a-fA-F]+)>\s*<([0-9a-fA-F]+)>\s*<([0-9a-fA-F]+)>", section):
                    lo, hi = int(first, 16), int(last, 16)
                    used_codes = [f"{code:04x}" for code in used if lo <= code <= hi]
                    if len(destination) % 4:
                        malformed.append({"source_first_hex": first, "source_last_hex": last, "destination_hex": destination, "used_source_codes": used_codes})
                    else:
                        for code in used:
                            if lo <= code <= hi:
                                integer = int(destination, 16) + code - lo
                                mapped[f"{code:04x}"] = integer.to_bytes(len(destination)//2, "big").decode("utf-16-be")
            for section in re.findall(r"beginbfchar(.*?)endbfchar", data, re.S):
                for source, destination in re.findall(r"<([0-9a-fA-F]+)>\s*<([0-9a-fA-F]+)>", section):
                    code = int(source, 16)
                    if code in used:
                        mapped[f"{code:04x}"] = bytes.fromhex(destination).decode("utf-16-be")
            result.append({"font_object": list(font_id), "cmap_object": list(cmap.objgen), "resource": resource_name, "basefont": str(target["/BaseFont"]), "source_code_width_bytes": code_width, "encoding": str(target.get("/Encoding", "source code width from ToUnicode codespace")), "used_cid_codes_hex": [f"{c:04x}" for c in sorted(used)], "total_used_glyph_instances": sum(used.values()), "used_pages": used_pages, "odd_encoded_text_strings": odd_strings, "malformed_range_entries": malformed, "valid_mapped_used_characters": mapped, "all_used_codes_mapped": set(mapped) == {f"{c:04x}" for c in used}, "any_malformed_entry_used": any(entry["used_source_codes"] for entry in malformed)})
    return result



def audit_all_cmaps(paths):
    audits = []
    for path in paths:
        fonts = []
        malformed = []
        with pikepdf.open(path) as pdf:
            for obj in pdf.objects:
                if not isinstance(obj, pikepdf.Dictionary) or obj.get("/Type") != pikepdf.Name("/Font") or obj.get("/ToUnicode") is None:
                    continue
                raw = obj["/ToUnicode"].read_bytes().decode("ascii")
                fonts.append({"font_object": list(obj.objgen), "basefont": str(obj.get("/BaseFont")), "cmap_sha256": hashlib.sha256(obj["/ToUnicode"].read_bytes()).hexdigest()})
                destinations = []
                for section in re.findall(r"beginbfchar(.*?)endbfchar", raw, re.S):
                    destinations += [destination for source, destination in re.findall(r"<([0-9a-fA-F]+)>\s*<([0-9a-fA-F]+)>", section)]
                for section in re.findall(r"beginbfrange(.*?)endbfrange", raw, re.S):
                    for first, last, destination in re.findall(r"<([0-9a-fA-F]+)>\s*<([0-9a-fA-F]+)>\s*<([0-9a-fA-F]+)>", section):
                        destinations.append(destination)
                    for array in re.findall(r"\[(.*?)\]", section, re.S):
                        destinations += re.findall(r"<([0-9a-fA-F]+)>", array)
                for destination in destinations:
                    try:
                        if len(destination) % 4:
                            raise ValueError("destination is not complete UTF-16 code units")
                        bytes.fromhex(destination).decode("utf-16-be")
                    except (ValueError, UnicodeDecodeError) as error:
                        malformed.append({"font_object": list(obj.objgen), "destination_hex": destination, "error": str(error)})
        audits.append({"path": path, "fonts_with_ToUnicode": len(fonts), "font_maps": fonts, "malformed_mappings": malformed})
    return audits

def main(folder, selected_editions=("digital", "print")):
    folder = Path(folder)
    bindings = []

    def bind(label, path, expected):
        actual = sha(path)
        entry = {"label": label, "path": str(path).replace("\\", "/"), "expected_sha256": expected, "actual_sha256": actual, "matches": actual == expected}
        bindings.append(entry)
        return entry

    folder = folder.resolve()
    owned = Path(__file__).resolve().parent
    if folder != owned and owned not in folder.parents:
        raise ValueError("Audit outputs must stay under tmp/pdfs/formal_thesis/v5/independent")
    folder.mkdir(parents=True, exist_ok=True)
    snapshots = read("ThesisDocs/formal/source_snapshots/v4_publication_baseline/manifest.json")["files"]
    for snapshot in snapshots:
        bind("preserved exact v4 source snapshot", snapshot["snapshot_path"], snapshot["sha256"])
    baseline_aliases = {s["original_path"]: s["snapshot_path"] for s in snapshots}
    baseline_hashes = {s["original_path"]: s["sha256"] for s in snapshots}
    protected = read("tmp/pdfs/formal_thesis/v5/protected_baseline.json")
    for item in protected:
        bind("protected v4 publication/scientific file", item["path"], item["sha256"])
    current_builder = sha("tools/build_master_thesis.py")
    for edition in selected_editions:
        recorded_main = read(f"ThesisDocs/formal/build_manifest_{edition}_v5.json")["main_builder_sha256"]
        bind(edition + " recorded execution builder", "tools/build_master_thesis.py", recorded_main)
    bind("preserved mathematical renderer", "tools/render_thesis_math.mjs", baseline_hashes["tools/render_thesis_math.mjs"])
    prior = read("ThesisDocs/archival/formal_v4_independent_technical_audit.json")
    historical = []
    for old in prior["hash_bindings"]:
        path = baseline_aliases.get(old["path"], old["path"])
        actual = sha(path)
        expected = old["expected_sha256"]
        historical.append({"path": path, "historical_sha256": expected, "actual_sha256": actual, "matches": actual == expected})
    for old in prior.get("historical_v3_and_v2_preservation", []):
        path = baseline_aliases.get(old["path"], old["path"])
        actual = sha(path)
        historical.append({"path": path, "historical_sha256": old["historical_sha256"], "actual_sha256": actual, "matches": actual == old["historical_sha256"]})

    editions = []
    for edition in selected_editions:
        conversion_path = Path(f"ThesisDocs/archival/formal_v5_{edition}_build_manifest.json")
        source_manifest_path = Path(f"ThesisDocs/formal/build_manifest_{edition}_v5.json")
        c = read(conversion_path)
        m = read(source_manifest_path)
        bind(edition + " final PDF", c["output"], c["sha256"])
        bind(edition + " raw source PDF / conversion receipt", c["source"], c["source_sha256"])
        bind(edition + " raw source PDF / source build receipt", m["output"], m["sha256"])
        bind(edition + " converter", c["builder"], c["builder_sha256"])
        bind(edition + " original validator XML", c["validator_report"], c["validator_report_sha256"])
        bind(edition + " validator binary", "tmp/pdfs/pdfa_tools/verapdf-1.30.2/bin/cli-1.30.2.jar", c["validator_jar_sha256"])
        for path, expected in m["source_files"].items():
            bind(edition + " manuscript source", path, expected)
        for path, key in [("research/mathematical_foundations.md", "canonical_math_source_sha256"), ("research/outputs/thesis_v1/evidence/manifest.json", "evidence_manifest_sha256"), ("data_manifest_v1.json", "data_manifest_sha256")]:
            bind(edition + " frozen dependency", path, m[key])
        for side in ("source", "output"):
            key = side + "_render_manifest"
            render_path = Path(c["render_isolation"][key])
            bind(edition + " " + side + " sealed render receipt", render_path, c["render_isolation"][key + "_sha256"])
            render = read(render_path)
            expected_pdf = c["source_sha256"] if side == "source" else c["sha256"]
            assert render["pdf_sha256"] == expected_pdf
            assert len(render["pages"]) == c["pages"]
            for entry in render["pages"]:
                png = render_path.parent / f"page_{entry['page']:03}.png"
                bind(edition + " " + side + " sealed page PNG", png, entry["png_sha256"])

        raw = pymupdf.open(c["source"])
        pdf = pymupdf.open(c["output"])
        front = m["front_matter_pages"]
        text_equal = [raw[i].get_text() == pdf[i].get_text() for i in range(len(pdf))]
        words_equal = [raw[i].get_text("words") == pdf[i].get_text("words") for i in range(len(pdf))]
        fonts = {font[0]: font for page in pdf for font in page.get_fonts(full=True)}
        unembedded = [font for xref, font in fonts.items() if not pdf.extract_font(xref)[3]]
        ordinary = []
        footer_rows = []
        all_page_sizes = []
        for number, page in enumerate(pdf, 1):
            all_page_sizes.append(tuple(page.rect))
            lines = [line for block in page.get_text("dict")["blocks"] for line in block.get("lines", [])]
            expected_folio = None if number == 1 else roman(number) if number <= front else str(number - front)
            footers = []
            for line in lines:
                text = "".join(span["text"] for span in line["spans"])
                if line["bbox"][1] > 700:
                    footers.append((text, line["bbox"]))
                for span in line["spans"]:
                    if span["text"].strip() and "KaTeX" not in span["font"]:
                        ordinary.append({"page": number, "size": span["size"], "text": span["text"], "font": span["font"]})
            matches = [] if expected_folio is None else [bbox for text, bbox in footers if text == expected_folio]
            footer_rows.append({"physical_page": number, "expected_folio": expected_folio, "actual_footer_text": [text for text, bbox in footers], "matches": (not footers) if expected_folio is None else len(matches) == 1 and len(footers) == 1, "center_points": None if not matches else (matches[0][0] + matches[0][2])/2})
        abstract = pdf[1].get_text().split("Abstract\n", 1)[1].split("Research adviser:", 1)[0]
        outline = pdf.get_toc()
        toc_start = next(row[2] for row in outline if row[1] == "Table of contents")
        table_list_start = next(row[2] for row in outline if row[1] == "List of tables")
        figure_list_start = next(row[2] for row in outline if row[1] == "List of figures")
        toc_text = "".join(page.get_text() for page in pdf[toc_start-1:table_list_start-1])
        toc_checks = []
        for entry in m["contents_body_entries"]:
            page_number = entry["body_page"] + front
            heading_lines = []
            for block in pdf[page_number - 1].get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    if line["spans"] and all("Bold" in s["font"] and s["size"] >= 10 for s in line["spans"] if s["text"].strip()):
                        heading_lines.append("".join(s["text"] for s in line["spans"]))
            toc_checks.append({**entry, "physical_page": page_number, "visible_toc_entry": norm(entry["title"] + str(entry["body_page"])) in norm(toc_text), "actual_bold_heading_on_target": norm(entry["title"]) in norm("".join(heading_lines)), "outline_destination": [entry["level"], entry["title"], page_number] in outline})
        front_entries = {title: norm(title + folio) in norm(toc_text) for title, folio in [("Abstract", "ii"), ("List of tables", roman(table_list_start)), ("List of figures", roman(figure_list_start))]}
        caption_checks = {}
        for kind in ("table", "figure"):
            actual = []
            list_start = table_list_start if kind == "table" else figure_list_start
            list_end = figure_list_start if kind == "table" else front+1
            list_text = "".join(page.get_text() for page in pdf[list_start-1:list_end-1])
            for number, (title, body_page) in enumerate(zip(m[kind + "_titles"], m[kind + "_body_pages"]), 1):
                caption = f"{kind.title()} {number}. {title}"
                expected_physical = body_page + front
                locations = [i+1 for i in range(front, len(pdf)) if norm(caption) in norm(pdf[i].get_text())]
                actual.append({"number": number, "title": title, "body_page": body_page, "physical_page": expected_physical, "actual_physical_pages": locations, "caption_destination_matches": locations == [expected_physical], "visible_list_entry_matches": norm(caption + str(body_page)) in norm(list_text)})
            caption_checks[kind] = actual
        figure_measurements = []
        for body_page in m["figure_body_pages"]:
            page = pdf[body_page + front-1]
            widths = [image["bbox"][2]-image["bbox"][0] for image in page.get_image_info() if image["bbox"][2]-image["bbox"][0] > 100]
            figure_measurements.append({"physical_page": body_page + front, "display_widths_points": widths})
        html_parts = sorted(Path(m["output"]).parent.glob("part_*.html"), key=lambda p: int(p.stem.split("_")[-1]))
        if not html_parts:
            raise FileNotFoundError("No source HTML parts available for table enumeration")
        body_html = "".join(p.read_text(encoding="utf-8") for p in html_parts)
        actual_html_tables = len(re.findall(r"<table(?:\s|>)", body_html))
        html_table_captions = re.findall(r"Table (\d+)\. ", body_html)
        html_table_caption_numbers = sorted({int(x) for x in html_table_captions})

        output_dir = Path(c["render_dir"])
        output_render = read(output_dir/"render_manifest.json")
        source_render = read(output_dir/"source_comparison/render_manifest.json")
        margins = []
        pixels = []
        left = 108.0 if edition == "print" else 72.0
        for number in range(1, len(pdf)+1):
            image = pymupdf.Pixmap(str(output_dir/f"page_{number:03}.png"))
            source_image = pymupdf.Pixmap(str(output_dir/f"source_comparison/page_{number:03}.png"))
            rgb = np.frombuffer(image.samples, dtype=np.uint8).reshape(image.height, image.width, image.n)
            source_rgb = np.frombuffer(source_image.samples, dtype=np.uint8).reshape(source_image.height, source_image.width, source_image.n)
            assert image.n == source_image.n == 3
            assert hashlib.sha256(image.samples).hexdigest() == output_render["pages"][number-1]["rgb_sha256"]
            assert hashlib.sha256(source_image.samples).hexdigest() == source_render["pages"][number-1]["rgb_sha256"]
            ink = np.any(rgb < 245, axis=2)
            ys, xs = np.nonzero(ink)
            bounds = [float(xs.min())/2, float(ys.min())/2, float(xs.max()+1)/2, float(ys.max()+1)/2]
            defect = bounds[0] < left-.5 or bounds[1] < 72-.5 or bounds[2] > 540+.5 or bounds[3] > 720+.5
            margins.append({"physical_page": number, "visible_ink_bounds_points": bounds, "defect": defect})
            diff = np.abs(rgb.astype(np.int16) - source_rgb.astype(np.int16))
            fraction = float(np.any(diff != 0, axis=2).mean())
            maximum = int(diff.max())
            pixels.append({"physical_page": number, "maximum_channel_difference": maximum, "changed_pixel_fraction": fraction, "within_sealed_conversion_tolerance": maximum <= 1 and fraction < .02})
        xml = ET.parse(c["validator_report"]).getroot()
        reports = [e for e in xml.iter() if e.tag.rsplit("}", 1)[-1] == "validationReport"]
        if not reports or any(e.attrib.get("isCompliant") != "true" for e in reports):
            raise RuntimeError("Recorded authoritative veraPDF report is missing or not compliant")
        cmap = audit_cmap(c["output"])
        min_size = min(span["size"] for span in ordinary)
        record = {"edition": edition, "output": c["output"], "sha256": sha(c["output"]), "source": c["source"], "source_sha256": sha(c["source"]), "pages": len(pdf), "front_matter_pages": front, "source_output_text_equal_all_pages": all(text_equal), "source_output_words_equal_all_pages": all(words_equal), "outline_preserved": raw.get_toc() == pdf.get_toc(), "page_labels_preserved": raw.get_page_labels() == pdf.get_page_labels(), "descriptive_metadata_preserved": all(raw.metadata.get(key) == pdf.metadata.get(key) for key in ("title", "author", "subject", "creator")), "metadata": pdf.metadata, "language": pdf.xref_get_key(pdf.pdf_catalog(), "Lang")[1], "tagged_structure_present": pdf.xref_get_key(pdf.pdf_catalog(), "StructTreeRoot")[0] != "null", "embedded_font_objects": len(fonts) - len(unembedded), "font_names": sorted({f[3] for f in fonts.values()}), "unembedded_fonts": unembedded, "minimum_ordinary_type_points": min_size, "undersized_ordinary_spans": [s for s in ordinary if s["size"] < 9.995], "abstract_words": len(abstract.split()), "logical_page_labels": pdf.get_page_labels(), "all_pages_letter": all(rect == (0, 0, 612, 792) for rect in all_page_sizes), "printed_folios": footer_rows, "printed_folios_exact": all(row["matches"] for row in footer_rows), "maximum_footer_center_deviation_points": max(abs(row["center_points"] - 306) for row in footer_rows if row["center_points"] is not None), "toc_entries": toc_checks, "toc_heading_count": len(toc_checks), "toc_heading_level_counts": dict(Counter(e["level"] for e in toc_checks)), "toc_front_entries": front_entries, "front_section_starts": {"toc": toc_start, "list_tables": table_list_start, "list_figures": figure_list_start}, "front_section_page_counts": {"toc": table_list_start-toc_start, "list_tables": figure_list_start-table_list_start, "list_figures": front+1-figure_list_start}, "caption_destinations": caption_checks, "actual_html_tables": actual_html_tables, "html_table_caption_numbers": html_table_caption_numbers, "figure_measurements": figure_measurements, "visible_ink_margin_measurement": {"method": "Read every immutable converter PNG; independently verify PNG and RGB hashes; RGB threshold245,144dpi,0.5pt rounding tolerance.", "required_points": {"left": left, "top": 72, "right": 72, "bottom": 72}, "defect_count": sum(row["defect"] for row in margins), "pages": margins}, "source_output_pixels": {"pages": pixels, "identical_page_count": sum(p["maximum_channel_difference"] == 0 for p in pixels), "maximum_channel_difference": max(p["maximum_channel_difference"] for p in pixels), "maximum_changed_pixel_fraction": max(p["changed_pixel_fraction"] for p in pixels), "all_within_original_conversion_tolerance": all(p["within_sealed_conversion_tolerance"] for p in pixels)}, "generic_folio_cmap": cmap, "original_converter_manifest_sha256": sha(conversion_path), "source_build_manifest_sha256": sha(source_manifest_path)}
        record["all_substantive_checks_pass"] = all([record["source_output_text_equal_all_pages"], record["source_output_words_equal_all_pages"], record["outline_preserved"], record["page_labels_preserved"], record["descriptive_metadata_preserved"], not unembedded, min_size >= 9.995, record["abstract_words"] == m["abstract_words"], record["all_pages_letter"], record["printed_folios_exact"], record["maximum_footer_center_deviation_points"] < .01, all(all(e[k] for k in ("visible_toc_entry", "actual_bold_heading_on_target", "outline_destination")) for e in toc_checks), all(front_entries.values()), all(e["caption_destination_matches"] and e["visible_list_entry_matches"] for rows in caption_checks.values() for e in rows), actual_html_tables == len(m["table_titles"]) == 17, html_table_caption_numbers == list(range(1,18)), len(m["figure_titles"]) == 6, not any(row["defect"] for row in margins), record["source_output_pixels"]["all_within_original_conversion_tolerance"], bool(cmap) and all(not f["any_malformed_entry_used"] and not f["odd_encoded_text_strings"] and f["all_used_codes_mapped"] for f in cmap)])
        editions.append(record)
        print(json.dumps({"edition": edition, "pages": len(pdf), "pass": record["all_substantive_checks_pass"], "toc_headings": len(toc_checks), "margin_defects": record["visible_ink_margin_measurement"]["defect_count"], "font_objects": len(fonts), "min_ordinary_points": min_size, "caption_entries": sum(map(len, caption_checks.values()))}), flush=True)

    cmap_audit = audit_all_cmaps([e["output"] for e in editions])
    result = {"schema": "independent-formal-thesis-technical-audit-v5", "created_utc": datetime.now(timezone.utc).isoformat(), "scope": "Read-only source/current-file, all-page text/word/metadata, sealed PNG/RGB/pixel/margin and live-CID checks of exact final v5 editions; no PDF or prior receipt writes.", "audit_script": "tmp/pdfs/formal_thesis/v5/independent/check_v5_final.py", "audit_script_sha256": sha("tmp/pdfs/formal_thesis/v5/independent/check_v5_final.py"), "hash_bindings_checked": len(bindings), "hash_mismatches": [row for row in bindings if not row["matches"]], "hash_bindings": bindings, "historical_v4_and_earlier_hash_bindings_checked": len(historical), "historical_v4_and_earlier_hash_mismatches": [row for row in historical if not row["matches"]], "historical_v4_and_earlier_preservation": historical, "compiled_manuscript": {"path": "ThesisDocs/Masters_Thesis_Formal_v5.md", "sha256": sha("ThesisDocs/Masters_Thesis_Formal_v5.md")}, "builder_provenance": {"current_path": "tools/build_master_thesis.py", "current_sha256": current_builder, "executed_digital_sha256": (read("ThesisDocs/formal/build_manifest_digital_v5.json")["main_builder_sha256"] if "digital" in selected_editions else None), "executed_print_sha256": (read("ThesisDocs/formal/build_manifest_print_v5.json")["main_builder_sha256"] if "print" in selected_editions else None), "execution_identity_source": "Each v5 source build manifest records execution-time main-builder SHA; current file bytes independently compared to both."}, "editions": editions, "all_document_cmaps": cmap_audit, "folio_map_scope": "Actual ReportLab subset TrueType Arial glyph bytes and Unicode mappings inspected across page text streams, including nested forms. All document ToUnicode streams independently checked for malformed destination strings.", "all_substantive_checks_pass": all(e["all_substantive_checks_pass"] for e in editions) and all(b["matches"] for b in bindings) and all(b["matches"] for b in historical) and all(x["fonts_with_ToUnicode"] > 0 and not x["malformed_mappings"] for x in cmap_audit)}
    target = folder / ("technical_audit.json" if len(selected_editions) == 2 else "technical_audit_" + selected_editions[0] + ".json")
    result["needs_manual_visual_and_scientific_semantic_review"] = True
    result["reuse_provenance"] = {"source": "tmp/pdfs/recheck_v4/data_independent_technical_check.py", "source_sha256": "140ebef344a2d82fa3c7eca3e390aa0ff03c50674fecb9a61a2a48d5d6eddecd", "scope": "Copied read-only v4 technical/font-map checks; v5 paths, protected v4 snapshots, 17 tables, 6 figures and dynamic front matter."}
    if target.exists():
        raise FileExistsError("Immutable independent receipt exists; choose a fresh owned subdirectory")
    changed_after_read = [b["path"] for b in bindings if sha(b["path"]) != b["actual_sha256"]]
    result["changed_during_audit"] = changed_after_read
    result["all_substantive_checks_pass"] &= not changed_after_read
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(target).replace("\\", "/"), "sha256": sha(target), "all_substantive_checks_pass": result["all_substantive_checks_pass"], "hash_bindings": len(bindings), "historical_v2_bindings": len(historical), "hash_mismatches": result["hash_mismatches"], "historical_v2_mismatches": result["historical_v4_and_earlier_hash_mismatches"]}), flush=True)

    if not result["all_substantive_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--edition", choices=["digital", "print"])
    args = parser.parse_args()
    main(args.directory, (args.edition,) if args.edition else ("digital", "print"))
