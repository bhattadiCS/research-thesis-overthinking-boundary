# Formal thesis v2 archival builds

The formal v2 digital and print editions are separate from the historical v1 review PDF, archival candidate, and receipts. Semester approvals are complete as confirmed by the user. PDF/A conversion establishes technical conformance for the saved bytes; institutional format acceptance is not asserted.

The main thesis builder owns the title page, abstract, contents, lists, typography, margins, logical page labels, and document language. The archival converter does not overlay or rewrite those pages. It reads the current main-build PDF, records its SHA256 and page count, and verifies that conversion preserves every page's text and word coordinates, outline, labels, language, and descriptive metadata. Each PDF is rendered in its own fresh process so cross-document resource/color caches cannot distort the comparison. Full-page 144-dpi comparisons permit only the recorded one-level ICC rounding, with less than two percent of each page affected; the limit was not relaxed to accommodate the demonstrated interleaved-render artifact.

Use the existing isolated authoring environment and publisher-verified veraPDF executable:

```powershell
& 'tmp/pdfs/pdfa_tools/venv/Scripts/python.exe' -B tools/build_archival_candidate.py `
  --source tmp/pdfs/formal_thesis/digital/source.pdf `
  --output output/pdf/Masters_Thesis_Formal_v2_Aditya_Bhatt.pdf `
  --receipt-prefix formal_v2_digital --margin-profile digital `
  --verapdf-jar tmp/pdfs/pdfa_tools/verapdf-1.30.2/bin/cli-1.30.2.jar

& 'tmp/pdfs/pdfa_tools/venv/Scripts/python.exe' -B tools/build_archival_candidate.py `
  --source tmp/pdfs/formal_thesis/print/source.pdf `
  --output output/pdf/Masters_Thesis_Formal_v2_Print_Aditya_Bhatt.pdf `
  --receipt-prefix formal_v2_print --margin-profile print `
  --verapdf-jar tmp/pdfs/pdfa_tools/verapdf-1.30.2/bin/cli-1.30.2.jar
```

For an explicitly reviewed source, add `--expected-source-sha256` with the current main-build identity. Existing v2 outputs require `--replace` for a deliberate rebuild. Historical v1 targets and source-file aliases are rejected even with that flag. The converter checks the pinned pikepdf version and veraPDF jar SHA256 before writing.

Each prefix produces a separate build manifest and validation XML in this directory. The manifest records the exact source/output hashes, actual page count, every page comparison, embedded fonts, measured ordinary-text sizes, printed pagination, margins, document structure, and authoritative validator result. Extractable ordinary body, table, label and code text must meet the 10-point minimum after any rendering scale; mathematical scripts and KaTeX struts are excluded from that ordinary-text measurement. Lettering within raster figures requires visual review. The formal v2 source has no separately styled footnotes. A successful machine audit still requires visual inspection of the exact resulting PDF before delivery. Its temporary page renders are identified in the manifest.

The [Sheridan formatting requirements](https://www.library.jhu.edu/library-services/electronic-theses-dissertations/formatting-requirements/) specify one-inch margins for digital-only documents and a 1.5-inch left binding margin for print copies. The print edition therefore comes from its own layout, with its own pagination, contents, and lists; it is not a cropped or translated digital PDF. The converted bytes are independently validated as PDF/A-2b. October 2026 is the title-page month; these receipts do not record an actual submission event or assert institutional format acceptance. No PDF/UA or semantic-tagging conformance is claimed.

The earlier `candidate_build_manifest.json`, validation XML, `visual_review.json`, `formatting_audit.md`, and `README.md` describe the preserved v1 preparation. Their historical findings and approval wording do not describe the formal v2 edition. Consult the new formal v2 audit and receipts for its measured status. To reproduce v1, use the historical source revision whose builder SHA256 is recorded in its manifest; the current converter owns v2.

The completed [requirements audit](formal_v2_formatting_audit.md), [source-integrity receipt](../formal/source_integrity_v2.json), and [consolidated visual review](../formal/visual_review_v2.json) describe the final editions. [Direct converted-page review](formal_v2_assigned_visual_review.json) records additional inspection. All 16 tables and four figures are captioned/listed; no material layout defects remain.

| Edition | Pages | PDF/A-2b validation |
| --- | ---: | --- |
| Digital | 94 | 144 rules and 247,687 checks passed; zero failures |
| Print | 97 | 144 rules and 249,171 checks passed; zero failures |

Digital SHA256: `5f1845de7653f02245a62016e8ffe1634b77a160dda367e4a3979d88fbcec240`.

Print SHA256: `b35b02bf6d8de33e62d74e1876c2a7fa48450876d99e4de7d1c04ddeaa6569a1`.
