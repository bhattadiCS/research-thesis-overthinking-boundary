"""Recompute the thesis tables from stored experiments, without editing raw data.

The review-metrics implementation is retained as a separately versioned source;
this entry point redirects its outputs to the thesis evidence directory and
records the exact implementation and current repository revision.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "research/outputs/thesis_v1/evidence")
    args = parser.parse_args()
    source = ROOT / "tools/compute_progress_report_review_metrics.py"
    spec = importlib.util.spec_from_file_location("thesis_review_metrics", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUTPUT = args.output_dir.resolve()
    result = module.main()
    manifest_path = module.OUTPUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["implementation"] = {
        "path": source.relative_to(ROOT).as_posix(),
        "sha256_lf": hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "entry_point": Path(__file__).relative_to(ROOT).as_posix(),
        "git_revision_before_recomputation": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": sys.version,
    }
    manifest["interpretation"] = {
        "stored_labels": "Uses frozen stored correct labels; does not regrade raw answers.",
        "uncertainty": "Development-corpus cluster bootstrap, not external replication.",
        "replay": "In-sample stored-trace simulation; not live generation.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8", newline="\n")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
