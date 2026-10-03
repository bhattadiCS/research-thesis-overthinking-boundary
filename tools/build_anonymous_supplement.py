"""Build an anonymous, integrity-audited review capsule from the final freeze.

Full raw historic corpora, model weights, authoring documents, Git metadata and
provisional model fits are omitted. Bound artifacts with no identifying text
remain byte-identical. Any sanitized review copy has both hashes recorded.
No materials are uploaded or submitted by this command.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/neurips/anonymous_stopping_supplement_v1.zip"
FIXED = [
    "data_manifest_v1.json", "software_provenance_v1.json", "requirements.lock.txt",
    "software_repository_venv_observed_v1.json", "requirements.repository-venv.lock.txt",
    "requirements.historical-tournament.partial.txt",
    "research/outputs/experiments_v2/ultimate_tournament_manifest.json",
    "research/mathematical_foundations.md", "research/adversarial_tasks_v1.jsonl",
    "research/adversarial_gold_v1.jsonl", "research/real_trace_experiments.py",
    "research/online_stopping_controller.py", "research/online_generation.py",
    "research/run_online_stopping_evaluation.py", "research/learned_online_stopping_controller.py",
    "research/run_learned_online_stopping.py", "research/prefix_stopping_model.py",
    "research/train_prefix_stopping_model.py", "research/analyze_online_stopping_results.py",
    "research/tests/test_online_controller.py", "research/tests/test_learned_online_controller.py",
    "research/tests/test_online_replay_accounting.py",
    "research/tests/test_prefix_stopping_model.py", "research/tests/test_mathematical_foundations.py",
    "research/tests/test_data_freeze.py", "research/tests/test_live_uncertainty.py",
    "research/tests/test_graders.py", "tools/freeze_research_data.py",
    "tools/analyze_live_stopping_uncertainty.py",
]
GLOBS = [
    "research/outputs/semester2/online_stopping_20261002/**/*",
    "research/outputs/semester2/prefix_model_v1/*",
    "research/outputs/thesis_v1/evidence/*.csv",
    "research/outputs/thesis_v1/evidence/*.json",
    "research/reports/thesis_failure_audit_v1/audit_summary.json",
    "research/reports/thesis_failure_audit_v1/taxonomy_summary.csv",
    "research/_vendor/torchvision_stub/**/*.py",
]
TEXT = {".py", ".json", ".jsonl", ".csv", ".md", ".txt", ".svg", ".log", ".html", ".mjs", ".yaml", ".yml", ".toml"}
DEPLOYED_SHA = "92fe0af86ac0f204d514a6938d0a29dace2b3cdffa24800cc4e2c85a2d51879f"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def substitutions() -> list[tuple[str, str, str]]:
    rules = []
    for prefix, replacement, label in [
        (str(ROOT), "<WORKSPACE>", "workspace_path"),
        (str(Path.home()), "<USER_HOME>", "user_home_path"),
    ]:
        variants = {prefix, prefix.replace("\\", "/"), prefix.replace("\\", "\\\\")}
        rules.extend((variant, replacement, label) for variant in sorted(variants, key=len, reverse=True))
    # The remote can contain credentials; never print or store its value.
    result = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, text=True, capture_output=True)
    if result.returncode == 0:
        remote = result.stdout.strip()
        rules.append((remote, "<REPOSITORY_URL_WITHHELD_FOR_REVIEW>", "repository_url"))
        if remote.endswith(".git"):
            rules.append((remote[:-4], "<REPOSITORY_URL_WITHHELD_FOR_REVIEW>", "repository_url"))
    rules.append(("Aditya Bhatt", "Anonymous Author", "author_name"))
    return rules


def sanitize(raw: bytes, suffix: str, rules) -> tuple[bytes, dict[str, int]]:
    if suffix.lower() not in TEXT:
        return raw, {}
    encoding = "utf-16-le" if raw.startswith(b"\xff\xfe") else "utf-16-be" if raw.startswith(b"\xfe\xff") else "utf-8"
    text = raw.decode(encoding)
    counts = {}
    for value, replacement, label in rules:
        number = text.count(value)
        if number:
            text = text.replace(value, replacement)
            counts[label] = counts.get(label, 0) + number
    # Bibliographic Aaditya Ramdas remains correctly attributed. Only an
    # explicitly separated researcher name/username is an anonymity defect.
    if re.search(r"\bAditya\b|\bBhatt\b|Aditya_Data|ResearchThesis", text, re.IGNORECASE):
        raise ValueError("Unresolved possible author identifier in a selected review copy")
    return (text.encode(encoding) if counts else raw), counts


def selected_sources() -> list[Path]:
    paths = {ROOT / relative for relative in FIXED}
    for pattern in GLOBS:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    paths = {path for path in paths if "__pycache__" not in path.parts and path.suffix.lower() != ".pyc"}
    absent = [path.relative_to(ROOT).as_posix() for path in paths if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Required supplement sources absent: {absent}")
    return sorted(paths)


README = """# Anonymous stopping research supplement

This is a review capsule for a research draft. It is not a submitted or accepted
NeurIPS paper. The target-year deadline and venue rules must be verified separately.

The capsule contains the qualified proofs and exact checks, deployable runtime,
the exact frozen prefix predictor, paired actual generation ledgers for 100 main
questions and 20 handpicked traps, task-disjoint archive training/calibration/
evaluation evidence, reconstructed labels, and derived historical evidence tables.
Gold ledgers are separate from public questions; controller inputs have no labels.

All four actual run ledgers are development experiments. The probability policy
stops every main and trap question at response two, so its realized behavior is a
fixed-two budget. Main accuracy is 7/100 versus 6/100 at full horizon, and trap
accuracy is 1/20 in each arm. Measured token savings do not establish a registered
noninferiority margin, conditional calibration, general adversarial robustness or
an adaptive advantage over a fixed-two policy.

Source copies with explicit local user paths or a researcher repository link have
those strings replaced. capsule_manifest.json records both original and review-
copy SHA256, sizes and redaction categories. Embedded historical source hashes
continue to describe original bytes; they are not rewritten to create a false
original identity. The deployed predictor stays byte-identical with SHA256
92fe0af86ac0f204d514a6938d0a29dace2b3cdffa24800cc4e2c85a2d51879f.

The 52-file historic trace corpus, larger matrix, OOF arrays and large LLM weights are
omitted. Their identities and coverage remain in data_manifest_v1.json. Complete
historical training environments are unknown. The two observed lock files are
inventories, not clean minimal install recipes. This capsule cannot reproduce all
historic tables without those omitted raw sources; it does support checking the
runtime, proofs, frozen artifact, task partition and actual paired evidence.

In an environment with the recorded Python/scientific dependencies, run:

```sh
python -m unittest discover -s research/tests -p 'test_*.py'
```

These tests do not generate new model responses. A live rerun needs a locally
available model snapshot and compatible GPU/software and produces a new identity;
existing model weights and original GPU timing are not reconstructed by this ZIP.
Do not run a full-corpus freeze verification on the reduced review capsule.
The original master freeze was verified before packaging; the capsule has its
own file inventory because its scope and sanitized copies differ.
"""


def write_member(archive, relative: str, raw: bytes) -> None:
    info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--validate", action="store_true", help="execute included tests on the extracted review copies")
    args = parser.parse_args()
    if OUTPUT.exists() and not args.replace:
        raise FileExistsError("Supplement exists; use --replace explicitly")
    subprocess.run([sys.executable, str(ROOT / "tools/freeze_research_data.py"), "verify"], cwd=ROOT, check=True)
    master = json.loads((ROOT / "data_manifest_v1.json").read_text(encoding="utf-8"))
    rules = substitutions()
    entries, contents = [], {}
    for path in selected_sources():
        relative = path.relative_to(ROOT).as_posix()
        original = path.read_bytes()
        try:
            review_copy, redactions = sanitize(original, path.suffix, rules)
        except ValueError as error:
            raise ValueError(f"Anonymity review failed for {relative}") from error
        if path.name in {"prefix_model.json", "frozen_prefix_predictor.json"}:
            if sha(original) != DEPLOYED_SHA or review_copy != original:
                raise ValueError("Deployed frozen predictor must remain unchanged")
        contents[relative] = review_copy
        entries.append({"path": relative, "original_sha256": sha(original), "review_copy_sha256": sha(review_copy),
                        "original_bytes": len(original), "review_copy_bytes": len(review_copy),
                        "sanitized": bool(redactions), "redaction_counts": redactions})
    contents["README.md"] = README.encode("utf-8")
    manifest = {"schema_version": "anonymous-stopping-review-capsule-v1",
                "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "master_data_content_fingerprint": master["content_fingerprint"],
                "master_freeze_original_sha256": sha((ROOT / "data_manifest_v1.json").read_bytes()),
                "interpretation": "Original-to-anonymous-copy provenance; original raw corpus and model weights are omitted.",
                "files": entries, "readme_sha256": sha(contents["README.md"]),
                "submitted": False, "accepted": False}
    if args.validate:
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="anonymous_supplement_check_", dir=ROOT / "tmp") as directory:
            location = Path(directory)
            if not location.resolve().is_relative_to((ROOT / "tmp").resolve()):
                raise ValueError("Temporary validation workspace escaped its intended directory")
            for relative, raw in contents.items():
                output = location / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(raw)
            result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "research/tests", "-p", "test_*.py"],
                                    cwd=location, text=True, capture_output=True)
            if result.returncode:
                raise ValueError("Extracted review-copy tests failed:\n" + result.stdout + result.stderr)
            print(result.stderr.strip())
            manifest["review_copy_tests"] = {"command": "python -m unittest discover -s research/tests -p test_*.py", "exit_code": result.returncode}
    contents["capsule_manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative, raw in sorted(contents.items()):
            write_member(archive, relative, raw)
    with zipfile.ZipFile(OUTPUT) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
        for entry in entries:
            if sha(archive.read(entry["path"])) != entry["review_copy_sha256"]:
                raise ValueError("ZIP member identity mismatch")
    if OUTPUT.stat().st_size > 100_000_000:
        raise ValueError("Supplement exceeds the 100 MB target limit")
    report = {"output": OUTPUT.relative_to(ROOT).as_posix(), "sha256": sha(OUTPUT.read_bytes()),
              "bytes": OUTPUT.stat().st_size, "source_files": len(entries),
              "sanitized_files": sum(entry["sanitized"] for entry in entries),
              "master_data_content_fingerprint": master["content_fingerprint"],
              "zip_integrity": "passed", "review_copy_tests": manifest.get("review_copy_tests"),
              "submitted": False, "accepted": False}
    (OUTPUT.parent / "anonymous_supplement_build_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
