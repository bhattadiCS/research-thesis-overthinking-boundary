"""Audit exact v6 PDFs and render them with Poppler; never edit a PDF.

Run from the repository root. A fresh output directory is required for each run.
The saved v5 font-map audit supplies read-only parsing functions, not its main.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
POPPLER = ROOT / "tmp/pdfs/recheck_v2/poppler/poppler-26.09.0/Library/bin"
JAR = ROOT / "tmp/pdfs/pdfa_tools/verapdf-1.30.2/bin/cli-1.30.2.jar"
FONT_AUDIT = ROOT / "ThesisDocs/formal/independent_checks_v5/executed_verifier_sources/check_v5_final.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    def encode_path(value):
        if isinstance(value, Path):
            return value.as_posix()
        raise TypeError(f"Unsupported receipt value: {type(value).__name__}")
    Path(path).write_text(json.dumps(value, indent=2, default=encode_path) + "\n", encoding="utf-8", newline="\n")


def normal(text):
    return re.sub(r"\s+", "", text)


def run(command, stdout, stderr):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    stdout.write_bytes(result.stdout)
    stderr.write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {stderr}")


def main(directory):
    directory = directory.resolve()
    if ROOT / "ThesisDocs/formal/independent_checks_v6" not in directory.parents:
        raise ValueError("Use a fresh directory under ThesisDocs/formal/independent_checks_v6")
    if directory.exists():
        raise FileExistsError("Choose a fresh audit directory")
    directory.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location("preserved_font_audit", FONT_AUDIT)
    fonts_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fonts_module)
    checks = []

    def check(name, passed, details=None):
        checks.append({"name": name, "passed": bool(passed), "details": details})

    baseline_path = ROOT / "ThesisDocs/verification/concise_v6_2026-10-07/protected_baseline.json"
    baseline = read(baseline_path)
    protected = [{**item, "actual_sha256": sha(ROOT / item["path"])} for item in baseline["files"]]
    check("all historical and frozen bytes preserved", all(x["sha256"] == x["actual_sha256"] for x in protected), len(protected))
    compiled_path = ROOT / "ThesisDocs/Masters_Thesis_Formal_v6.md"
    compiled = compiled_path.read_text(encoding="utf-8")
    chapters = "\n".join(p.read_text(encoding="utf-8") for p in sorted((ROOT / "ThesisDocs/concise_v6").glob("chapter*.md")))
    references = (ROOT / "ThesisDocs/references.md").read_text(encoding="utf-8")
    defined = set(re.findall(r"^\[([A-Za-z0-9]+)\]", references, re.M))
    cited = set(re.findall(r"\[([A-Za-z][A-Za-z0-9]+)\](?!\()", re.sub(r"\$\$(.*?)\$\$|(?<!\\)\$(.*?)(?<!\\)\$|`[^`]+`", "", chapters, flags=re.S)))
    check("all 25 references defined and cited", defined == cited and len(defined) == 25, sorted(defined - cited))
    check("no unexpanded insertion or citation markers", "[[" not in compiled and not any(f"[{key}]" in compiled for key in defined))
    old = (ROOT / "ThesisDocs/Masters_Thesis_Formal_v5.md").read_text(encoding="utf-8")
    def corpus_table(text, number):
        return re.search(rf"\*\*Table {number}\. .*?\*\*\s*(\|.*?)(?=\n\n)", text, re.S)[1]
    check("corpus table equals preserved v5 data", corpus_table(compiled, 1) == corpus_table(old, 2))
    controls_path = ROOT / "research/outputs/thesis_v1/evidence/algorithm_v2_normalized_effects.csv"
    with controls_path.open(encoding="utf-8", newline="") as stream:
        controls = list(csv.DictReader(stream))
    control_checks = []
    for row in controls:
        values = [f"{float(row[key]):+.5f}" for key in ("mean_controlled_effect", "ci_95_low", "ci_95_high")]
        control_checks.append({"experiment": row["experiment"], "effect_unit": row["effect_unit"], "n_units": row["n_units"], "values": values, "all_values_in_manuscript": all(value in compiled for value in values)})
    check("nine normalized control contrasts match source CSV", len(controls) == 9 and all(x["all_values_in_manuscript"] for x in control_checks), control_checks)
    # Import only to obtain established evidence locations, not table-rendering code.
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import build_master_thesis as locations
    live_checks = []
    for folder in (locations.ONLINE, locations.ADVERSARIAL, locations.ONLINE / "learned_main", locations.ONLINE / "learned_adversarial"):
        metrics = read(folder / "live_metrics.json")
        interval = read(folder / "live_uncertainty.json")
        values = [f"{metrics[key]:,}" for key in ("baseline_generated_tokens", "active_generated_tokens")]
        values += [f"{100*metrics['measured_completion_token_savings']:.2f}%"]
        values += [f"{100*v:+.2f}" for v in interval["accuracy_delta_conservative_exact_95ci"]]
        values += [f"{100*v:.2f}" for v in interval["completion_token_savings_cluster_bootstrap_95ci"]]
        live_checks.append({"source": str(folder.relative_to(ROOT)).replace("\\", "/"), "metrics_sha256": sha(folder / "live_metrics.json"), "uncertainty_sha256": sha(folder / "live_uncertainty.json"), "n": metrics["problems_or_trajectories"], "values": values, "all_values_in_manuscript": all(value in compiled for value in values)})
    check("live table cost and intervals match saved paired evidence", all(x["all_values_in_manuscript"] for x in live_checks), live_checks)
    editions = []
    for edition in ("digital", "print"):
        folder = directory / edition
        folder.mkdir()
        source_path = ROOT / f"ThesisDocs/formal/build_manifest_{edition}_v6.json"
        conversion_path = ROOT / f"ThesisDocs/archival/formal_v6_{edition}_build_manifest.json"
        m, c = read(source_path), read(conversion_path)
        final_path, raw_path = ROOT / c["output"], ROOT / c["source"]
        check(edition + " PDF/source hash bindings", sha(final_path) == c["sha256"] and sha(raw_path) == c["source_sha256"] == m["sha256"])
        check(edition + " build and source provenance", sha(ROOT / "tools/build_concise_thesis.py") == m["main_builder_sha256"] and sha(ROOT / "tools/build_master_thesis.py") == m["shared_renderer_sha256"] and all(sha(ROOT / p) == s for p, s in m["source_files"].items()))
        raw, pdf = fitz.open(raw_path), fitz.open(final_path)
        front, n = m["front_matter_pages"], len(pdf)
        check(edition + " total pages in 25-35 range", 25 <= n <= 35, n)
        check(edition + " all page text/word geometry unchanged", n == len(raw) and all(p.get_text() == q.get_text() and p.get_text("words") == q.get_text("words") for p, q in zip(pdf, raw)))
        check(edition + " outline and logical labels unchanged", pdf.get_toc() == raw.get_toc() and pdf.get_page_labels() == raw.get_page_labels())
        check(edition + " abstract length", m["abstract_words"] <= 350, m["abstract_words"])
        fonts = {item[0]: item for page in pdf for item in page.get_fonts(full=True)}
        unembedded = [item for xref, item in fonts.items() if not pdf.extract_font(xref)[3]]
        ordinary = [span for page in pdf for block in page.get_text("dict")["blocks"] for line in block.get("lines", []) for span in line["spans"] if span["text"].strip() and "KaTeX" not in span["font"]]
        check(edition + " embedded fonts and ordinary type >=10pt", not unembedded and min(s["size"] for s in ordinary) >= 9.995, {"fonts": len(fonts), "minimum_points": min(s["size"] for s in ordinary)})
        toc = pdf.get_toc()
        front_starts = {name: next(row[2] for row in toc if row[1] == name) for name in ("Table of contents", "List of tables", "List of figures")}
        toc_text = "".join(pdf[i].get_text() for i in range(front_starts["Table of contents"]-1, front_starts["List of tables"]-1))
        heading_checks = []
        for entry in m["contents_body_entries"]:
            target = entry["body_page"] + front
            heading_checks.append({**entry, "destination_matches": [entry["level"], entry["title"], target] in toc and normal(entry["title"]) in normal(pdf[target-1].get_text()), "printed_toc_matches": normal(entry["title"] + str(entry["body_page"])) in normal(toc_text)})
        check(edition + " every contents heading and destination", all(x["destination_matches"] and x["printed_toc_matches"] for x in heading_checks), heading_checks)
        captions = []
        for kind in ("table", "figure"):
            start = front_starts[f"List of {kind}s"]
            end = front_starts["List of figures"] if kind == "table" else front+1
            list_text = "".join(pdf[i].get_text() for i in range(start-1, end-1))
            for number, (title, body_page) in enumerate(zip(m[f"{kind}_titles"], m[f"{kind}_body_pages"]), 1):
                caption = f"{kind.title()} {number}. {title}"
                targets = [i+1 for i in range(front, n) if normal(caption) in normal(pdf[i].get_text())]
                captions.append({"kind": kind, "number": number, "physical_page": body_page+front, "passed": targets == [body_page+front] and normal(caption+str(body_page)) in normal(list_text)})
        check(edition + " three figures and three tables with list destinations", len(captions) == 6 and all(x["passed"] for x in captions), captions)
        output_dir = ROOT / c["render_dir"]
        margins, pixels = [], []
        for i in range(1, n+1):
            rgb = np.asarray(Image.open(output_dir / f"page_{i:03}.png").convert("RGB"))
            original = np.asarray(Image.open(output_dir / f"source_comparison/page_{i:03}.png").convert("RGB"))
            ys, xs = np.nonzero(np.any(rgb < 245, axis=2))
            bounds = [float(xs.min())/2, float(ys.min())/2, float(xs.max()+1)/2, float(ys.max()+1)/2]
            left = 108 if edition == "print" else 72
            margins.append({"page": i, "ink_bounds_points": bounds, "passed": bounds[0] >= left-.5 and bounds[1] >= 71.5 and bounds[2] <= 540.5 and bounds[3] <= 720.5})
            difference = np.abs(rgb.astype(np.int16) - original.astype(np.int16))
            pixels.append({"page": i, "maximum_channel_difference": int(difference.max()), "passed": int(difference.max()) <= 1})
        check(edition + " all rendered ink respects margins", all(x["passed"] for x in margins), margins)
        check(edition + " all converted pixels agree within one channel level", all(x["passed"] for x in pixels), pixels)
        cmap = fonts_module.audit_cmap(final_path)
        all_cmaps = fonts_module.audit_all_cmaps([final_path])
        check(edition + " used folio codes and every Unicode map valid", bool(cmap) and all(not x["any_malformed_entry_used"] and not x["odd_encoded_text_strings"] and x["all_used_codes_mapped"] for x in cmap) and all(not x["malformed_mappings"] for x in all_cmaps), {"folio": cmap, "all": all_cmaps})
        check(edition + " validator binary identity", sha(JAR) == c["validator_jar_sha256"])
        run(["java", "-Xmx1g", "-jar", str(JAR), "--flavour", "2b", "--format", "xml", str(final_path)], folder / "verapdf.xml", folder / "verapdf_stderr.txt")
        xml = ET.parse(folder / "verapdf.xml")
        reports = [x for x in xml.iter() if x.tag.rsplit("}", 1)[-1] == "validationReport"]
        check(edition + " fresh PDF/A-2b validation", bool(reports) and all(x.attrib.get("isCompliant") == "true" for x in reports), [x.attrib for x in reports])
        run([str(POPPLER / "pdftotext.exe"), "-layout", str(final_path), str(folder / "text_layout.txt")], folder / "pdftotext_stdout.txt", folder / "pdftotext_stderr.txt")
        run([str(POPPLER / "pdftoppm.exe"), "-r", "120", "-png", str(final_path), str(folder / "page")], folder / "pdftoppm_stdout.txt", folder / "pdftoppm_stderr.txt")
        texts = (folder / "text_layout.txt").read_text(encoding="utf-8").split("\f")
        if texts and not texts[-1].strip():
            texts.pop()
        files = sorted(folder.glob("page-*.png"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))
        folios = []
        for physical, text in enumerate(texts, 1):
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            expected = None if physical == 1 else fonts_module.roman(physical) if physical <= front else str(physical-front)
            folios.append({"physical_page": physical, "expected": expected, "last_line": lines[-1], "passed": expected is None or lines[-1] == expected})
        check(edition + " Poppler page count and printed folios", len(texts) == len(files) == n and all(x["passed"] for x in folios), folios)
        # Six pages per contact sheet. Full-resolution pages remain for inspection.
        contacts = []
        for start in range(0, n, 6):
            sheet = Image.new("RGB", (1500, 2100), "#d9d9d9")
            draw = ImageDraw.Draw(sheet)
            for k, path in enumerate(files[start:start+6]):
                thumb = Image.open(path).convert("RGB")
                thumb.thumbnail((730, 995))
                x, y = (k % 2)*750+10, (k // 2)*700+30
                thumb.thumbnail((730, 650))
                sheet.paste(thumb, (x, y))
                draw.text((x, y-22), f"{edition}: physical page {start+k+1}", fill="black")
            contact = folder / f"contact_{start+1:02}_{min(start+6,n):02}.png"
            sheet.save(contact)
            contacts.append({"path": str(contact.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(contact), "pages": [start+1, min(start+6,n)]})
        editions.append({"edition": edition, "output": c["output"], "sha256": sha(final_path), "pages": n, "front_pages": front, "word_count": m["word_count"], "abstract_words": m["abstract_words"], "contacts": contacts, "page_pngs": [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(p)} for p in files], "fresh_verapdf_xml": str((folder / "verapdf.xml").relative_to(ROOT)).replace("\\", "/"), "fresh_verapdf_xml_sha256": sha(folder / "verapdf.xml")})
        print(json.dumps({"edition": edition, "pages": n, "checks_pass_so_far": all(x["passed"] for x in checks)}), flush=True)
    result = {"schema": "concise-thesis-technical-audit-v6", "created_utc": datetime.now(timezone.utc).isoformat(), "audit_script": "tools/verify_concise_thesis.py", "audit_script_sha256": sha(__file__), "font_audit_source": str(FONT_AUDIT.relative_to(ROOT)).replace("\\", "/"), "font_audit_sha256": sha(FONT_AUDIT), "protected_baseline_commit": baseline["commit"], "protected_files": protected, "compiled_manuscript_sha256": sha(compiled_path), "checks": checks, "editions": editions, "all_machine_checks_pass": all(x["passed"] for x in checks), "manual_visual_and_semantic_review_required": True}
    save(directory / "technical_audit.json", result)
    print(json.dumps({"receipt": str(directory / "technical_audit.json"), "passed": result["all_machine_checks_pass"], "failed": [x["name"] for x in checks if not x["passed"]]}))
    if not result["all_machine_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    main(parser.parse_args().directory)
