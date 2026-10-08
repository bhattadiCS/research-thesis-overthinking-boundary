Concise formal thesis v7: content-restored revision

Primary sources: chapter1_introduction.md through chapter5_discussion.md,
../references.md, and ../../tools/build_concise_thesis_v7.py.
V5 and v6 sources, PDFs, builders and receipts remain unchanged.

From the repository root in the existing authoring environment:
  & tmp/pdfs/pdfa_tools/venv/Scripts/python.exe tools/build_concise_thesis_v7.py --edition digital
  & tmp/pdfs/pdfa_tools/venv/Scripts/python.exe tools/build_concise_thesis_v7.py --edition print

Raw output: tmp/pdfs/compact_v7/{digital,print}/source.pdf
Reviewed raw copies: ../formal/raw_sources/v7/{digital,print}.pdf
Compiled text: ../Masters_Thesis_Formal_v7.md
Build receipts: ../formal/build_manifest_{digital,print}_v7.json

The authoring runtime uses Chrome, KaTeX, PyMuPDF and ReportLab. The
unchanged v5 renderer supplies shared formatting and evidence extraction.
Runtime and source identities are recorded in the build receipts.

Archival conversion uses ../../tools/build_archival_candidate.py with the
raw source, explicit output/pdf/Masters_Thesis_Formal_v7_Aditya_Bhatt.pdf
(or Masters_Thesis_Formal_v7_Print_Aditya_Bhatt.pdf), formal_v7_digital
(or formal_v7_print) receipt prefix, edition margin profile, raw SHA256,
and the pinned veraPDF jar. A deliberate uncommitted candidate rebuild may
need --replace; historical published editions must never be overwritten.

Read-only final validation, choosing a fresh directory:
  & tmp/pdfs/pdfa_tools/venv/Scripts/python.exe tools/verify_concise_thesis_v7.py --directory ThesisDocs/formal/independent_checks_v7/new_run

The verifier requires local Poppler and veraPDF paths recorded in the
source/conversion receipts. Raw paths can be restored from the reviewed
copies after verifying their hashes. Every rebuilt PDF requires fresh
archival and visual review. Machine success is not visual/scientific approval.

The content audit, full v5 section ledger and preservation checks are under
../verification/content_audit_v7_2026-10-08/. The extended v5 report retains
supporting derivations and secondary analyses; the compact PDF is not a
lossless reproduction of every original table or formula.

No new research observations or training are required for this editorial
revision. The title-page month must match actual ETD deposit.
