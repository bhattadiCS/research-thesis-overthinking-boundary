"""Read-only v5 scientific-content comparison; writes a new owned receipt only.

Run after the v5 source and compiled Markdown are declared complete. Numeric
inventory differences are review aids, not a proof of scientific equivalence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

OWNED = Path(__file__).resolve().parent
ROOT = OWNED.parents[4]
BASELINE = ROOT / "ThesisDocs/formal/source_snapshots/v4_publication_baseline/manifest.json"
PROTECTED = ROOT / "tmp/pdfs/formal_thesis/v5/protected_baseline.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_code(text):
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    return re.sub(r"`[^`\n]*`", "", text)


def citations(text):
    text = clean_code(text)
    # [[NAME]] is the builder's insertion grammar, not a bibliography key.
    text = re.sub(r"\[\[[^\[\]\n]+\]\]", "", text)
    # E[G_m] and similar square brackets inside TeX are mathematical operators.
    text = re.sub(r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<![\\$])\$(?!\$).*?(?<![\\$])\$(?!\$)", "", text, flags=re.S)
    return set(re.findall(r"(?<!\\)\[([A-Za-z][A-Za-z0-9_-]{2,})\]", text))


def math_blocks(text):
    return re.findall(r"(?<!\\)\$\$.*?(?<!\\)\$\$|(?<![\\$])\$(?!\$).*?(?<![\\$])\$(?!\$)", text, re.S)


def tables(text):
    lines = text.splitlines()
    result = {}
    number = None
    i = 0
    while i < len(lines):
        cap = re.match(r"^\s*(?:\*\*)?Table\s+(\d+)\.", lines[i])
        if cap:
            number = int(cap.group(1))
        if lines[i].lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            if number is not None:
                if number in result:
                    raise ValueError(f"Multiple table bodies for caption {number}")
                result[number] = re.sub(r"\s+", "", re.sub(r"<br\s*/?>", "", "\n".join(block), flags=re.I))
            continue
        i += 1
    return result


def main(output):
    output = output.resolve()
    if output != OWNED and OWNED not in output.parents:
        raise ValueError("Output must remain within the independent v5 directory")
    target = output / "source_comparison.json"
    if target.exists():
        raise FileExistsError("Choose a fresh owned output directory")
    snapshot_manifest = read(BASELINE)
    snapshots = snapshot_manifest["files"]
    snapshot_checks = [dict(path=r["snapshot_path"], expected_sha256=r["sha256"], actual_sha256=sha(ROOT / r["snapshot_path"])) for r in snapshots]
    protected_checks = [dict(path=r["path"], expected_sha256=r["sha256"], actual_sha256=sha(ROOT / r["path"])) for r in read(PROTECTED)]
    md_paths = [r for r in snapshots if r["original_path"].endswith(".md")]
    numeric = []
    math_checks = []
    current_texts = []
    all_used = set()
    current_hashes = {}
    for r in md_paths:
        current = ROOT / r["original_path"]
        before = (ROOT / r["snapshot_path"]).read_text(encoding="utf-8")
        after = current.read_text(encoding="utf-8")
        current_hashes[r["original_path"]] = sha(current)
        if current.name != "references.md":
            current_texts.append(after)
            all_used |= citations(after)
        pattern = r"(?<![A-Za-z_])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:%|ms|B)?"
        old_numbers, new_numbers = Counter(re.findall(pattern, clean_code(before))), Counter(re.findall(pattern, clean_code(after)))
        numeric.append({"path": r["original_path"], "removed_numeric_tokens": dict(old_numbers - new_numbers), "added_numeric_tokens": dict(new_numbers - old_numbers), "interpretation": "Context must be reviewed; concision and relocated material can legitimately change token counts."})
        # Code-fenced PowerShell variables in old Appendix A are not TeX.
        old_math = math_blocks(clean_code(before))
        new_math = math_blocks(clean_code(after))
        math_checks.append({"path": r["original_path"], "v4_expression_count": len(old_math),
                            "v5_expression_count": len(new_math), "ordered_tex_equal": old_math == new_math})
    reference_text = (ROOT / "ThesisDocs/references.md").read_text(encoding="utf-8")
    reference_record = next(r for r in snapshots if r["original_path"] == "ThesisDocs/references.md")
    references_equal = current_hashes[reference_record["original_path"]] == reference_record["sha256"]
    definitions = citations(reference_text)
    theory_record = next(r for r in snapshots if r["original_path"] == "ThesisDocs/chapters/chapter2_theory.md")
    old_theory = (ROOT / theory_record["snapshot_path"]).read_text(encoding="utf-8")
    new_theory = (ROOT / theory_record["original_path"]).read_text(encoding="utf-8")
    old_md = (ROOT / "ThesisDocs/Masters_Thesis_Formal_v4.md").read_text(encoding="utf-8")
    new_md_path = ROOT / "ThesisDocs/Masters_Thesis_Formal_v5.md"
    compiled_hash_before = sha(new_md_path)
    new_md = new_md_path.read_text(encoding="utf-8")
    old_tables, new_tables = tables(old_md), tables(new_md)
    expected_map = {**{i: i for i in range(1, 17)}, 18: 17}
    table_checks = [{"v4_number": old, "v5_number": new, "both_present": old in old_tables and new in new_tables, "table_body_equal_whitespace_normalized": old in old_tables and new in new_tables and old_tables[old] == new_tables[new]} for old, new in expected_map.items()]
    figure_checks = [{"path": r["path"], "expected_sha256": r["sha256"], "actual_sha256": sha(ROOT / r["path"])} for r in read(ROOT / "ThesisDocs/formal/source_integrity_v4.json")["figure_files"]]
    theory_ok = old_theory.replace("Appendix F", "Appendix E") == new_theory
    result = {
        "schema": "independent-v5-source-comparison-v1",
        "verifier_source": {"path": Path(__file__).resolve().relative_to(ROOT).as_posix(), "sha256": sha(Path(__file__).resolve())},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_commit": snapshot_manifest["baseline_commit"],
        "snapshot_manifest_sha256": sha(BASELINE),
        "protected_baseline_sha256": sha(PROTECTED),
        "snapshot_checks": snapshot_checks,
        "protected_checks": protected_checks,
        "current_source_hashes": current_hashes,
        "compiled_v5_sha256": compiled_hash_before,
        "chapter2_only_approved_appendix_cross_reference_change": theory_ok,
        "chapter2_math_blocks_unchanged": math_blocks(old_theory) == math_blocks(new_theory),
        "ordered_true_tex_by_source": math_checks,
        "all_ordered_true_tex_unchanged": all(r["ordered_tex_equal"] for r in math_checks),
        "references_byte_identical_to_v4": references_equal,
        "citation_integrity": {"used_keys": sorted(all_used), "defined_keys": sorted(definitions), "undefined_keys": sorted(all_used - definitions), "unused_definitions": sorted(definitions - all_used)},
        "citation_parser_exclusions": ["Fenced and inline code", "Double-bracket renderer insertion tokens", "Inline/display TeX, including expectation brackets"],
        "scientific_table_mapping": table_checks,
        "removed_v4_table17_status_table": 17 not in expected_map,
        "v5_table_numbers": sorted(new_tables),
        "preserved_six_image_bytes": figure_checks,
        "numeric_inventory_review_aids": numeric,
        "scope": "No experiments, model fitting, document rendering or old-receipt mutation. Tables1-16 exact; old prediction table18 maps to17; old status table17 stays in the supplement.",
        "manual_semantic_review_required": True,
    }
    changed = [path for path, expected in current_hashes.items() if sha(ROOT / path) != expected]
    if sha(new_md_path) != compiled_hash_before:
        changed.append(new_md_path.relative_to(ROOT).as_posix())
    result["source_inputs_changed_during_audit"] = changed
    result["structural_preservation_checks_pass"] = (all(r["expected_sha256"] == r["actual_sha256"] for r in snapshot_checks + protected_checks + figure_checks) and theory_ok and result["chapter2_math_blocks_unchanged"] and result["all_ordered_true_tex_unchanged"] and references_equal and not changed and not result["citation_integrity"]["undefined_keys"] and all(r["table_body_equal_whitespace_normalized"] for r in table_checks) and sorted(new_tables) == list(range(1, 18)))
    output.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(target), "sha256": sha(target), "structural_preservation_checks_pass": result["structural_preservation_checks_pass"], "manual_semantic_review_required": True}))
    if not result["structural_preservation_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    main(args.directory)
