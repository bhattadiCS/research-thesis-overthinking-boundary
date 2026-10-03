"""Fetch the official style and a pinned, checksum-verified portable TeX engine."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "tmp/neurips_build"
DOCS = ROOT / "ThesisDocs/neurips"
STYLE_URL = "https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip"
ENGINE_URL = "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/tectonic-0.17.0-x86_64-pc-windows-msvc.zip"
ENGINE_SHA = "f61ce51f0b0ade1015b7de7ef368541c5424e9756ecbd0d7af97d6d48030845f"


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "thesis-reproducible-build"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    style = download(STYLE_URL)
    (WORK / "official_template.zip").write_bytes(style)
    members = {}
    with zipfile.ZipFile(io.BytesIO(style)) as archive:
        for item in archive.infolist():
            path = Path(item.filename)
            if "__MACOSX" in path.parts or path.name.startswith("."):
                continue
            if path.suffix not in (".sty", ".tex", ".bib"):
                continue
            content = archive.read(item)
            output = DOCS / "template" / path.name
            output.parent.mkdir(exist_ok=True)
            if output.exists() and output.read_bytes() != content:
                raise ValueError(f"Official template differs from retained copy: {output.name}")
            output.write_bytes(content)
            members[output.relative_to(ROOT).as_posix()] = hashlib.sha256(content).hexdigest()
            if path.name == "neurips_2026.sty":
                (DOCS / path.name).write_bytes(content)
    engine_path = WORK / "tectonic.zip"
    engine = engine_path.read_bytes() if engine_path.exists() else download(ENGINE_URL)
    if hashlib.sha256(engine).hexdigest() != ENGINE_SHA:
        raise ValueError("Portable TeX engine checksum differs from official GitHub release digest")
    engine_path.write_bytes(engine)
    extracted = WORK / "tectonic"
    extracted.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(engine)) as archive:
        for item in archive.infolist():
            # Flat extraction into the intended directory avoids trusting ZIP
            # traversal components. The pinned binary digest covers all bytes.
            if not item.is_dir():
                (extracted / Path(item.filename).name).write_bytes(archive.read(item))
    provenance = {"official_template_url": STYLE_URL,
                  "official_template_archive_sha256": hashlib.sha256(style).hexdigest(),
                  "retained_template_files": members,
                  "portable_engine": {"version": "0.17.0", "url": ENGINE_URL, "archive_sha256": ENGINE_SHA},
                  "target": "Future NeurIPS main-track preparation using the published 2026 style; next-cycle rules must be rechecked",
                  "sources": ["https://neurips.cc/Conferences/2026/CallForPapers",
                              "https://neurips.cc/Conferences/2026/MainTrackHandbook",
                              "https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0"]}
    (DOCS / "template_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"template_files": list(members), "engine": str(extracted / "tectonic.exe")}))


if __name__ == "__main__":
    main()
