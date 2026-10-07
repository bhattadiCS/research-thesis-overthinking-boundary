# Current formal thesis

Use the audited **v5** editions for the current formal document:

- [Editable compiled manuscript](Masters_Thesis_Formal_v5.md)
- [Digital PDF, 83 pages](../output/pdf/Masters_Thesis_Formal_v5_Aditya_Bhatt.pdf)
- [Print PDF, 88 pages, with binding margin](../output/pdf/Masters_Thesis_Formal_v5_Print_Aditya_Bhatt.pdf)
- [Editorial revision and preservation review](verification/editorial_revision_v5_2026-10-06/revision_review.html)
- [Independent technical audit](archival/formal_v5_independent_technical_audit.json)
- [Source integrity](formal/source_integrity_v5.json) and [complete visual review](formal/visual_review_v5.json)
- [Electronic reproduction and historical evidence guide](formal/supplements/v5/reproduction_and_history.txt)
- [Current completion record](verification/publication_completion_v5_2026-10-06.json)
- [Publication package manifest](../output/research/Thesis_v5_Publication_Package_2026-10-06_manifest.json)

Version 5 removes repeated explanations and numerical recitals, moves workflow and project-status detail to the electronic supplement, and improves chapter endings, figure-caption grouping and front lists. The manuscript is **15,866 words**, down from 20,064 (20.9%). All seventeen scientific tables retain their v4 data; the completed-work status table is preserved in the supplement. All 257 mathematical expressions, six figures and twenty-five references are retained. The paired-interval prose is shorter with its assumptions and argument intact.

The [chapter sources](chapters/), [references](references.md), [appendices](appendices.md), and [builder](../tools/build_master_thesis.py) generate the PDFs and compiled Markdown. Cover and abstract fields are in the builder. October 2026 is the current cover month and must match the actual ETD submission month.

The v1-v4 editions and original receipts remain historical records. Exact [v4 authoring sources](formal/source_snapshots/v4_publication_baseline/manifest.json) identify the published baseline at `c8ba3bc`. The repository-history review retains its earlier baseline at `6e4378b`. New technical verification establishes preparation and file conformance; academic defense, faculty certification and institutional acceptance are separately evidenced events. Semester approvals are user-confirmed complete.

Rebuild from the repository root with the installed authoring environment:

```powershell
& 'tmp/pdfs/pdfa_tools/venv/Scripts/python.exe' tools/build_master_thesis.py `
  --edition digital --document-version v5 --submission-date 'October 2026'
& 'tmp/pdfs/pdfa_tools/venv/Scripts/python.exe' tools/build_master_thesis.py `
  --edition print --document-version v5 --submission-date 'October 2026'
```

Raw files are under `tmp/pdfs/formal_thesis/v5/`; [reviewed raw copies](formal/raw_sources/v5/) are retained with their hashes. Archival conversion uses `tools/build_archival_candidate.py`, the respective raw source, explicit v5 output and receipt prefix, edition margin profile, and pinned veraPDF jar. A rebuild produces new bytes requiring renewed validation and visual review. Historical editions require their archived source versions.

The [revision verifier](../tools/verify_formal_thesis_revision.py) compares the current manuscript with v4. Exact executed independent verification scripts and reproduction recipes are retained under [independent checks](formal/independent_checks_v5/executed_verifier_sources/README.txt).
