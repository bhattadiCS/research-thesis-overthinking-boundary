# Formal thesis v3 requirements audit

The independent recheck found and corrected two omissions: the chapter-only contents omitted the section titles shown in JHU's linked front-matter example, and Appendix A's historical pytest command omitted the grader module included in the 107-test receipt. The three-page contents now includes all 62 body headings, with the actual destinations for each edition. The reproduction command includes the thirty grader tests. Scientific results were preserved.

Checked against the [ACM research and thesis guidelines](https://ep.jhu.edu/wp-content/uploads/2024/10/EP-ACM-Research-or-Thesis-Option_guidelines.pdf), revised July 13, 2026, the [Sheridan formatting requirements](https://www.library.jhu.edu/library-services/electronic-theses-dissertations/formatting-requirements/), and its linked [title-page](https://www.library.jhu.edu/wp-content/uploads/sites/3/2019/06/annotated-title-page-1.pdf) and [front-matter](https://www.library.jhu.edu/wp-content/uploads/sites/3/2021/01/annotated-front-matter.pdf) examples. Semester approvals are user-confirmed complete and excluded from this document task.

| Requirement | Evidence in the corrected document |
| --- | --- |
| Significant mathematics | Chapter 2 gives the probability model, exact correctness drift, finite-horizon stopping construction, persistence condition, counterexamples, perturbation bounds and uncertainty conditions. Independent mathematical review found no new defect. |
| Contribution and literature | Sections 1.4 and 1.7 distinguish established theory from protocol-specific derivations, controlled empirical contrasts and negative live findings. All 25 bibliography entries resolve. Scholarly significance remains an academic judgment for the readers. |
| Cover | Uppercase title centered inside edition margins; author below "by"; Master of Science statement without department or concentration; Baltimore, Maryland; hidden cover folio. |
| Abstract | 209 English words, immediately after the cover, double spacing, adviser and second reader named after the text. |
| Contents and lists | Three contents pages cover all 62 chapter, section and appendix headings, matching actual printed pages and bookmarks. Separate lists include all 16 tables and four figures with accurate destinations. |
| Typography and spacing | Nominal 12-point Arial body and double spacing; ordinary text measures at least 10 points. Tables and code use 10.1 points; print figure labels are at least 10.263 points. Mathematical scripts are treated separately. |
| Margins | Digital edition has one-inch margins; print edition has the required 1.5-inch left binding margin. Text and rendered ink bounds were checked on every page. |
| Pagination | Hidden cover folio; abstract ii; Roman front matter through vii; introduction restarts at Arabic 1; all subsequent folios consecutive and centered inside the bottom margin. |
| Back matter and reproduction | References follow six chapters; Appendices A-D start separately. All 46 nonempty command lines retain the source's paths, tokens and PowerShell continuations. |
| Archival format | Exact final files pass independent PDF/A-2b validation, with 37 embedded font objects per edition and no failed rules or checks. |
| Appearance and evidence integrity | All pages have visual coverage, with fresh inspection of changed pages. Unchanged reviews transfer only after exact mapped pixels/text/word coordinates. Conversion preserves all text and word coordinates; permitted ICC rounding is at most one RGB level. Frozen data, research code and mathematical source verify unchanged. |

| Edition | Pages | veraPDF rules passed | Checks passed | Failures |
| --- | ---: | ---: | ---: | ---: |
| Digital | 96 | 144 | 250,386 | 0 |
| Print | 99 | 144 | 251,870 | 0 |

Digital SHA256: `d17638789394790d55bf02968e89f196f9ef85b06e3918fc9fb81406b8e9bf9f`.

Print SHA256: `84ddf73e86f92169aab885d3a2552a4c9734bc629772a1d9d7b248d6d3798ea1`.

The [source-integrity report](../formal/source_integrity_v3.json), [consolidated visual review](../formal/visual_review_v3.json), [mathematics and contents review](../formal/visual_math_v3.json), [supplemental visual review](../formal/visual_supplement_v3.json), and [independent technical audit](formal_v3_independent_technical_audit.json) record verification evidence. The [digital](formal_v3_digital_build_manifest.json) and [print](formal_v3_print_build_manifest.json) conversion receipts bind the final bytes to their separate validator reports. Previous v1/v2 artifacts and receipts remain historical; exact v2 builder and appendix bytes are retained in the [source snapshots](../formal/source_snapshots/manifest.json).

The cover date is **October 2026** and must match the actual ETD submission month. This audit verifies document preparation and technical conformance; it does not record a deposit or institutional acceptance. Defense feedback must be incorporated when received; no unprovided committee comments are assumed resolved.
