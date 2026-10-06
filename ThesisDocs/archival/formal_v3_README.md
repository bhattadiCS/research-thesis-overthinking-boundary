# Corrected formal thesis v3

The current digital edition is [Masters_Thesis_Formal_v3_Aditya_Bhatt.pdf](../../output/pdf/Masters_Thesis_Formal_v3_Aditya_Bhatt.pdf), 96 pages. The separate [print edition](../../output/pdf/Masters_Thesis_Formal_v3_Print_Aditya_Bhatt.pdf) has 99 pages and a 1.5-inch left binding margin. Edit the [compiled Markdown manuscript](../Masters_Thesis_Formal_v3.md) for reading/review; persistent manuscript changes belong in the [chapter sources](../chapters/), [references](../references.md), [appendices](../appendices.md), or the builder's cover/abstract fields. A rebuild regenerates the compiled manuscript.

Version 3 corrects the contents' missing section headings and the historical test command's omitted grader module. It preserves the reported scientific measurements. All 62 body headings, 16 tables and four figures have verified contents/list destinations. See the [requirements audit](formal_v3_formatting_audit.md) and [visual review](../formal/visual_review_v3.json).

Use the installed authoring environment and pinned renderer to rebuild the source PDFs. The cover month must match the actual ETD submission month.

```powershell
& 'tmp/pdfs/pdfa_tools/venv/Scripts/python.exe' tools/build_master_thesis.py `
  --edition digital --document-version v3 `
  --submission-date 'October 2026'
& 'tmp/pdfs/pdfa_tools/venv/Scripts/python.exe' tools/build_master_thesis.py `
  --edition print --document-version v3 `
  --submission-date 'October 2026'
```

The builder uses KaTeX 0.19.0 and installed Chrome. Rendering and document dependencies include markdown, pandas, matplotlib, PyMuPDF and reportlab. The conversion environment pins pikepdf 10.16.0; the veraPDF 1.30.2 jar is checked against its recorded publisher-verified SHA256. Conversion verifies text, word coordinates, page labels, bookmarks, language, embedded fonts, margins, pagination and 144-dpi appearance. Each input/output document is rendered in a separate fresh process.

```powershell
& 'tmp/pdfs/pdfa_tools/venv/Scripts/python.exe' tools/build_archival_candidate.py `
  --source tmp/pdfs/formal_thesis/v3/digital/source.pdf `
  --output output/pdf/Masters_Thesis_Formal_v3_Aditya_Bhatt.pdf `
  --receipt-prefix formal_v3_digital --margin-profile digital `
  --verapdf-jar tmp/pdfs/pdfa_tools/verapdf-1.30.2/bin/cli-1.30.2.jar
& 'tmp/pdfs/pdfa_tools/venv/Scripts/python.exe' tools/build_archival_candidate.py `
  --source tmp/pdfs/formal_thesis/v3/print/source.pdf `
  --output output/pdf/Masters_Thesis_Formal_v3_Print_Aditya_Bhatt.pdf `
  --receipt-prefix formal_v3_print --margin-profile print `
  --verapdf-jar tmp/pdfs/pdfa_tools/verapdf-1.30.2/bin/cli-1.30.2.jar
```

Existing targets require `--replace` for a deliberate rebuild. Rebuilding invalidates earlier visual-review identities; inspect and validate the resulting exact files again before using them. The [digital](formal_v3_digital_build_manifest.json) and [print](formal_v3_print_build_manifest.json) manifests contain their authoritative validator reports. A separate [independent technical audit](formal_v3_independent_technical_audit.json) records fresh validator reruns on the exact delivered bytes. A second renderer, Poppler 26.09.0, rendered every final page and independently checked printed pagination and front-matter order.

The [source-integrity report](../formal/source_integrity_v3.json) distinguishes the two executed builder snapshots. The print locator also handles headings that wrap after a hyphen; this has no scientific-content effect. Exact historical v2 source bytes and the executed v3 builder bytes are preserved in [source snapshots](../formal/source_snapshots/manifest.json). These copies are archival references: restore a builder to its original repository path in an isolated checkout before executing it, because its relative paths depend on that location. The unchanged current builder reproduces both edition layouts.

V1 and v2 PDFs, raw sources and immutable receipts remain separate. This preparation records no submission, academic signoff or library acceptance. Semester approvals are complete as confirmed by the user.
