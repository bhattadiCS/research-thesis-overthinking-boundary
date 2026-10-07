Concise formal thesis v6

Primary authoring sources are chapter1_introduction.md through
chapter5_discussion.md, with ../references.md and ../../tools/build_concise_thesis.py.
The shared renderer ../../tools/build_master_thesis.py remains unchanged from v5.

Run from the repository root in the existing authoring environment:
  & tmp/pdfs/pdfa_tools/venv/Scripts/python.exe tools/build_concise_thesis.py --edition digital
  & tmp/pdfs/pdfa_tools/venv/Scripts/python.exe tools/build_concise_thesis.py --edition print

Raw output: tmp/pdfs/compact_v6/{digital,print}/source.pdf
Durable reviewed raw copies: ../formal/raw_sources/v6/{digital,print}.pdf
Compilation: ../Masters_Thesis_Formal_v6.md
Build receipts: ../formal/build_manifest_{digital,print}_v6.json

The runtime uses Chrome, KaTeX, PyMuPDF and ReportLab. The converter uses
pikepdf and the pinned veraPDF 1.30.2 jar. Runtime paths and source hashes
are recorded in build/conversion receipts. Historical sources and results
are available in the published Git repository; v5 is the extended report.

Archival conversion, for each edition (substitute print for digital):
  & tmp/pdfs/pdfa_tools/venv/Scripts/python.exe tools/build_archival_candidate.py --source tmp/pdfs/compact_v6/digital/source.pdf --output output/pdf/Masters_Thesis_Formal_v6_Aditya_Bhatt.pdf --receipt-prefix formal_v6_digital --margin-profile digital --expected-source-sha256 <raw-source-sha256> --verapdf-jar tmp/pdfs/pdfa_tools/verapdf-1.30.2/bin/cli-1.30.2.jar

For print the output name is Masters_Thesis_Formal_v6_Print_Aditya_Bhatt.pdf.
An intentional rebuild of an uncommitted candidate may require --replace;
never use that option on a historical published edition. Rebuilt bytes
require renewed validation and visual review.

Read-only verification of exact PDFs (choose a fresh directory):
  & tmp/pdfs/pdfa_tools/venv/Scripts/python.exe tools/verify_concise_thesis.py --directory ThesisDocs/formal/independent_checks_v6/new_run

The verifier requires the preserved baseline and local validator/Poppler
runtime. It reads the source-build raw paths, which can be restored from the
durable raw copies after verifying their recorded hashes. It renders every
page and produces contact sheets for manual inspection. Machine success is
not a substitute for the visual or scientific review.

No new research observations, model execution or fitting are needed to
reproduce this editorial revision. The title-page month must match the
actual ETD submission month.
