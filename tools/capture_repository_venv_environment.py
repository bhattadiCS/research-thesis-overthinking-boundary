"""Capture the separate repository venv used for the failure-classification audit.

Run with .venv/Scripts/python.exe. This records installed distributions and
interpreter identity only; it does not infer the original experiment software.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    expected = (ROOT / ".venv").resolve()
    if Path(sys.prefix).resolve() != expected:
        parser.error("Use the repository .venv/Scripts/python.exe interpreter explicitly")
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
        version = distribution.version
        if name in packages and packages[name] != version:
            raise ValueError(f"Conflicting installed versions for {name}")
        packages[name] = version
    packages = dict(sorted(packages.items()))
    record = {
        "schema_version": "repository-venv-observed-environment-v1",
        "captured_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "interpretation": "Current observed repository venv, used for the new failure-classification command; distinct from the base audit/live environment and from historical generation software.",
        "associated_command": ".venv/Scripts/python.exe research/classify_losses.py --matrix-root research/outputs/experiment_matrix --out research/reports/thesis_failure_audit_v1",
        "interpreter": {"executable": sys.executable, "prefix": sys.prefix,
                        "python_version": platform.python_version(), "python_build": sys.version,
                        "platform": platform.platform()},
        "distribution_count": len(packages), "installed_distributions": packages,
        "not_reconstructed": ["package indexes", "wheel hashes", "direct-install source identities",
                              "system libraries", "historical training packages", "original model and dataset revisions"],
    }
    header = "\n".join([
        "# Observed repository .venv inventory; not a minimal regeneration recipe.",
        "# Used for the new failure classification; base audit/live uses a separate lock.",
        "# Provenance: software_repository_venv_observed_v1.json.",
        "# No original experiment environment or wheel/source hashes are reconstructed.",
        "",
    ])
    outputs = {
        "requirements.repository-venv.lock.txt": header + "\n".join(f"{name}=={version}" for name, version in packages.items()) + "\n",
        "software_repository_venv_observed_v1.json": json.dumps(record, indent=2, sort_keys=True) + "\n",
    }
    if not args.replace:
        for name in outputs:
            if (ROOT / name).exists():
                raise FileExistsError(f"Output exists; use --replace: {name}")
    for name, value in outputs.items():
        (ROOT / name).write_text(value, encoding="utf-8", newline="\n")
    print(json.dumps({"distribution_count": len(packages), "executable": sys.executable,
                      "output_files": list(outputs)}))


if __name__ == "__main__":
    main()
