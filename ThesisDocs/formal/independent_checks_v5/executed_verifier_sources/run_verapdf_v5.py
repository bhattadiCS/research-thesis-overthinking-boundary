"""Fresh read-only veraPDF validation of existing v5 PDFs; no PDF export."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

OWNED = Path(__file__).resolve().parent
ROOT = OWNED.parents[4]
JAR = ROOT / "tmp/pdfs/pdfa_tools/verapdf-1.30.2/bin/cli-1.30.2.jar"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(folder, editions):
    folder = folder.resolve()
    if folder != OWNED and OWNED not in folder.parents:
        raise ValueError("Output must remain within the independent v5 directory")
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError("Use a new owned output directory for validation")
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for edition in editions:
        receipt_path = ROOT / f"ThesisDocs/archival/formal_v5_{edition}_build_manifest.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        pdf = ROOT / receipt["output"]
        before = sha(pdf)
        if before != receipt["sha256"] or sha(JAR) != receipt["validator_jar_sha256"]:
            raise ValueError("PDF or validator binary differs from the recorded execution")
        command = ["java", "-Xmx1g", "-jar", str(JAR), "--flavour", "2b", "--format", "xml", str(pdf)]
        run = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
        report = folder / f"independent_verapdf_{edition}.xml"
        errors = folder / f"independent_verapdf_{edition}_stderr.txt"
        report.write_bytes(run.stdout)
        errors.write_bytes(run.stderr)
        xml = ET.fromstring(run.stdout)
        reports = [n for n in xml.iter() if n.tag.rsplit("}", 1)[-1] == "validationReport"]
        after = sha(pdf)
        passed = run.returncode == 0 and before == after and bool(reports) and all(n.attrib.get("isCompliant") == "true" for n in reports)
        results.append({"edition": edition, "pdf": receipt["output"], "pdf_sha256_before": before, "pdf_sha256_after": after, "validator_jar_sha256": sha(JAR), "conversion_receipt_sha256": sha(receipt_path), "command_arguments": command, "exit_code": run.returncode, "report": str(report), "report_sha256": sha(report), "validation_report_attributes": [n.attrib for n in reports], "pass": passed})
    summary = {"schema": "independent-verapdf-v5-v1", "created_utc": datetime.now(timezone.utc).isoformat(), "scope": "Fresh validation of existing exact v5 PDF bytes; no document authoring, PDF export, marker or prior-receipt mutation.", "editions": results, "all_pass": all(r["pass"] for r in results)}
    path = folder / "independent_verapdf_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(path), "sha256": sha(path), "all_pass": summary["all_pass"]}))
    if not summary["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--edition", choices=["digital", "print"])
    args = parser.parse_args()
    main(args.directory, [args.edition] if args.edition else ["digital", "print"])
