# Archival candidate and remaining deposit gates

The separate archival candidate preserves the reviewed 86-page thesis's scientific content and corrects three front-matter details. It passes **PDF/A-2b** validation by veraPDF Greenfield 1.30.2: 144 passed rules, 232,678 passed checks, zero failed rules or checks. [The validation XML](candidate_verapdf_2b.xml) and [build manifest](candidate_build_manifest.json) bind this result to its exact bytes. Academic approval and deposit have not occurred.

The candidate single-spaces the cover author line, adds adviser/reader names after the abstract, and begins the contents with Abstract ii. All other 83 pages retain exactly the source's extracted text and word coordinates; the scientific abstract is also unchanged. All page sizes and the outline remain the same. Rendering at 144 dpi gives identical RGB pixels on 64 unedited pages; the other 19 have color-profile rounding of at most one level out of 255. The builder rejects larger differences outside the declared front-matter edits. This is not a claim that every pixel is identical. Every font object is embedded. The original review PDF and frozen science remain unchanged.

Preparation adds XMP, an sRGB output intent, explicit archival annotation flags and one explicit embedded Arial CID map, and removes five empty page-transition dictionaries. It does not rasterize the mathematics or rewrite the scientific argument. PDF/A-2b conformance establishes neither tagged accessibility nor institutional acceptance. Sheridan demonstrates PDF/A-1b conversion but does not identify it as the sole allowed profile on the checked page; confirm acceptance of 2b with the program/library before deposit.

## Reproduction

Use an isolated Python authoring environment with `pikepdf[pdfa]==10.16.0`, NumPy and PyMuPDF; do not alter the frozen research environment. Download the stable veraPDF installer from its [official installation page](https://docs.verapdf.org/install/), verify the publisher signature and install into a fresh temporary directory. The retained [tool provenance](tool_provenance.json) records the verified key fingerprint, package/installer checksums and actual tool versions.

```powershell
& 'tmp/pdfs/pdfa_tools/venv/Scripts/python.exe' tools/build_archival_candidate.py --verapdf-jar tmp/pdfs/pdfa_tools/verapdf-1.30.2/bin/cli-1.30.2.jar
```

The builder intentionally accepts only the exact previously reviewed draft. A revised final thesis requires a new reviewed source, an updated build contract, validation of the final written bytes and renewed visual checks. Conversion is not final approval.

## Actual final-deposit requirements

The [EP course description](https://ep.jhu.edu/courses/625804-applied-and-computational-mathematics-masters-thesis/) directs this thesis to [Sheridan ETD standards](https://www.library.jhu.edu/library-services/electronic-theses-dissertations/formatting-requirements/). The [formatting audit](formatting_audit.md) checks title wording/case, abstract/names, separate lists, spacing, margins and numbering. The candidate retains October 2026 as the draft's cover date; the final cover must use the actual ETD submission month. The shortened list labels can be aligned with final captions during approved revision. It must not be submitted as a committee-approved final thesis.

After a successful public defense and final edits, obtain adviser/reader certification and program-chair approval through the [current July 2026 ACM process](https://ep.jhu.edu/wp-content/uploads/2024/10/EP-ACM-Research-or-Thesis-Option_guidelines.pdf). Confirm the EP internal approval/deposit cutoff, announcement lead time and whether the course webpage's bound-copy requirement remains applicable. The residential WSE deadline must not be substituted for the EP deadline without confirmation.

Sheridan requires deposit at least two full working days before the applicable approval deadline, excluding university holidays; corrections may take further time. Actual submission requires the student's agreement, metadata/access decisions and fee, followed by library acceptance. Keep the exact deposited file, validation report and institutional receipts together. Use the [administrative plan](../completion_and_submission_plan_2026.md) for outstanding records and calendar corrections.
