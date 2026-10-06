# Formal thesis v2 requirements audit

Checked against the [EP/ACM research and thesis guidelines](https://ep.jhu.edu/wp-content/uploads/2024/10/EP-ACM-Research-or-Thesis-Option_guidelines.pdf), revised July 13, 2026, and the [Sheridan formatting requirements](https://www.library.jhu.edu/library-services/electronic-theses-dissertations/formatting-requirements/) and linked examples. Semester approvals are user-confirmed complete and were excluded from this document task.

| Requirement | Final document evidence |
| --- | --- |
| Significant mathematics | Chapter 2 covers the filtration, repair/corruption identity, finite-horizon stopping, counterexamples and uncertainty conditions. |
| Contributions beyond direct application | Sections 1.4 and 1.7 distinguish established theory from protocol-specific derivations, controlled contrasts, implementation audit and negative live findings in Chapters 4-5. Readers assess academic significance. |
| Literature and contribution distinctions | Section 1.7 compares closely related approaches; 25 scholarly references identify antecedents. No claim of the first stopping formulation or comparative superiority. |
| Cover | Centered uppercase title; author beneath by; Master of Science statement without department/concentration; Baltimore, Maryland; hidden folio. |
| Abstract | 209 English words, immediately after cover, double spacing, adviser and second reader named afterward. |
| Contents and lists | Abstract first in contents; separate table and figure lists with matching titles/destinations; all 16 actual tables and four figures listed. |
| Typography | Nominal 12-point Arial body with double spacing; measured ordinary type at least 10 points. Tables/code use 10.1 points; print figure labels remain at least 10.263 points. Mathematical scripts are measured separately. |
| Margins | Digital: one inch. Print: 1.5-inch left binding margin and one inch elsewhere. All-page raster checks found zero defects. |
| Pagination | Cover folio hidden; abstract ii; lowercase Roman front matter; introduction restarts at Arabic 1; every subsequent folio consecutive and centered inside the bottom margin. |
| Back matter | References after six chapters; Appendices A-D start separately; PowerShell continuations and tokens intact. |
| Archival format | Exact final bytes pass PDF/A-2b validation. All fonts embedded; language, bookmarks, page labels and metadata preserved. |
| Appearance and integrity | Every page has visual coverage. Changed pages freshly reviewed; transferred reviews require exact mapped body pixels/word coordinates. Conversion preserves all page text/geometry with only recorded one-level ICC rounding. Frozen research data/code/math source verify unchanged. |

The [source-integrity receipt](../formal/source_integrity_v2.json) records source hashes, table/figure enumeration, measured figure sizes and successful frozen-manifest verification: 52 data files, 688 auxiliary files, 144,440 rows and 28,888 trajectories. The [consolidated visual review](../formal/visual_review_v2.json) binds coverage to the final source/output identities; [direct converted-page review](formal_v2_assigned_visual_review.json) records additional full-page inspections. Scientific measurements were preserved; Brier prose now correctly distinguishes raw and calibrated scores.

| Edition | Pages | veraPDF rules/checks passed | Failed rules/checks |
| --- | ---: | ---: | ---: |
| Digital | 94 | 144 / 247,687 | 0 / 0 |
| Print | 97 | 144 / 249,171 | 0 / 0 |

Digital SHA256: `5f1845de7653f02245a62016e8ffe1634b77a160dda367e4a3979d88fbcec240`.

Print SHA256: `b35b02bf6d8de33e62d74e1876c2a7fa48450876d99e4de7d1c04ddeaa6569a1`.

The immutable [digital receipt](formal_v2_digital_build_manifest.json) and [print receipt](formal_v2_print_build_manifest.json) bind those PDFs to their separate authoritative veraPDF 1.30.2 XML reports. Historical v1 artifacts remain separate.

The cover says **October 2026** and must match the actual ETD submission month. Rebuild that field if submission occurs in another month. This audit verifies document preparation and technical conformance; it records no deposit or institutional acceptance.
