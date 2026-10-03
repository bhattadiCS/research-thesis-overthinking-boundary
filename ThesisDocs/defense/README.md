# Master's thesis defense package

The content source has exactly 25 slides and 1,800 seconds of speaking time. The speaker notes contain approximately 3,750 words, suitable for a 30-minute talk at 125 words per minute with the allocated timing. The separate defense question bank contains 46 questions and rigorous answers.

- defense_slides.json: editable content specification, table cells, numerical chart series, speaker notes and source citations.
- defense_slide_source.md: human-readable slide text and numerical evidence.
- speaker_notes_30_minutes.md: timed notes with primary proof references.
- timing_plan.md: cumulative start times.
- defense_question_bank.md: mathematical, statistical, deployment and institutional questions.
- generate_defense_source.py: deterministic content regeneration from the current final evidence files.

Generate content from the repository root:

    python ThesisDocs/defense/generate_defense_source.py

Build and export the editable PowerPoint using the user-approved installed-PowerPoint fallback:

    powershell -NoProfile -ExecutionPolicy Bypass -File tools/build_defense_deck.ps1

The original presentation skill's bundled artifact runtime was unavailable; the user explicitly approved installed PowerPoint. The operation-start marker was executed exactly once before authoring. Evidence tables and charts must remain native editable PowerPoint objects with embedded chart data. The builder, exported slide images, finalization and visual inspection are separate from content generation. Presence of these source files alone does not certify a finished or visually checked PPTX.

Slides 23 and 24 refresh from the final heuristic/learned main and trap ledgers. Until a ledger exists, the outcome slot reads Pending; no visual value is invented. All four final ledgers now exist. The original 100-task heuristic result is 2.94% actual completion-token savings and 6% accuracy in both arms. The separately generated learned arm saves 56.51% with 7% accuracy versus 6%, and every task stops at two. Its accuracy-change interval includes harm and no noninferiority margin was registered. The learned archive replay and historical retrospective detector remain separate evidence. Before a final build, regenerate the content and recheck every rendered slide.

All evidence visuals in this package use original project numerical data; primary theorem references and local artifact paths are included in the notes. No stock or generated decorative image is required.

The completed editable deck is `output/presentation/Thesis_Defense_v1_Aditya_Bhatt.pptx`; its PDF preview is `output/pdf/Thesis_Defense_Preview_v1_Aditya_Bhatt.pdf`. All 25 slides were visually inspected after actual PowerPoint export. It contains 13 native tables and 3 native charts with embedded Excel workbooks. `final_build_manifest.json` records the exact reviewed bytes, native save/reopen and edit checks, and the two conservative table-fit warnings resolved by inspecting the native renders. `native_content_audit.json` verifies every chart/workbook value and every primary or secondary table cell against the editable source and original scientific tables. Rebuilding creates a new candidate that requires fresh review; it does not automatically update the reviewed final file.
