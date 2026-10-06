# Current formal thesis

Use the audited **v4** editions for the current formal document:

- [Editable compiled manuscript](Masters_Thesis_Formal_v4.md)
- [Digital PDF, 104 pages](../output/pdf/Masters_Thesis_Formal_v4_Aditya_Bhatt.pdf)
- [Print PDF, 109 pages, with binding margin](../output/pdf/Masters_Thesis_Formal_v4_Print_Aditya_Bhatt.pdf)
- [Research completion and repository history review](verification/repository_review_2026-10-05/research_review.html)
- [Independent formatting and PDF technical audit](archival/formal_v4_independent_technical_audit.json)
- [Source integrity](formal/source_integrity_v4.json) and [complete visual review](formal/visual_review_v4.json)
- [Current research completion status](verification/publication_completion_2026-10-06.json)
- [Publication package and exact-byte manifest](../output/research/Thesis_v4_Publication_Package_2026-10-06_manifest.json)

The editable manuscript is Markdown. The [chapter sources](chapters/), [references](references.md), [appendices](appendices.md), and [builder](../tools/build_master_thesis.py) generate the formal PDFs; the compiled Markdown is regenerated during a build. Cover and abstract fields are in the builder. October 2026 is the current cover month and must match the actual ETD submission month.

Version 4 incorporates the completed strict baselines and selected-answer analyses, qualifies the legacy peer results and reporting calibration, clarifies the grader notation and historical evaluation folds, and adds information-flow and delayed-repair diagrams. Appendix E records completed work and the evidence needed only for stronger prospective claims. The existing corpus supports the documented conclusions; wholesale recollection is unnecessary. No new model generation or retraining was performed.

The v1-v3 PDFs, receipts and exact source snapshots are preserved historical versions. The frozen repository README and dated advisor materials retain their historical links. The repository-history review records the baseline at `6e4378b`; the Git commit containing this index identifies the current v4 publication package. The dated administrative ledgers describe earlier evidence and are superseded for technical preparation and user-confirmed semester approvals by the current completion status above.

Rebuild from the repository root with the installed authoring environment:

```powershell
& 'tmp/pdfs/pdfa_tools/venv/Scripts/python.exe' tools/build_master_thesis.py `
  --edition digital --document-version v4 --submission-date 'October 2026'
& 'tmp/pdfs/pdfa_tools/venv/Scripts/python.exe' tools/build_master_thesis.py `
  --edition print --document-version v4 --submission-date 'October 2026'
```

The raw files are under `tmp/pdfs/formal_thesis/v4/`. Archival conversion uses `tools/build_archival_candidate.py` with the respective raw source, explicit v4 output path, `formal_v4_digital` or `formal_v4_print` receipt prefix, edition margin profile, and the pinned veraPDF jar. A rebuild requires fresh validation and visual review before its new bytes are used. Historical editions require their archived source versions.
