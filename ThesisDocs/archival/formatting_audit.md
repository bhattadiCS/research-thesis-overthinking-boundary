# Draft archival formatting audit

Checked against [Sheridan formatting requirements](https://www.library.jhu.edu/library-services/electronic-theses-dissertations/formatting-requirements/) on October 2, 2026. This is internal preparation, not library acceptance. The exact candidate and review receipt are identified in [the manifest](candidate_build_manifest.json).

| Requirement | Candidate evidence | Status |
| --- | --- | --- |
| Cover | Capitalized centered title; prescribed thesis statement, degree, Baltimore/Maryland; no printed cover number | Checked. Cover author line now single-spaced. |
| Submission month | October 2026 retained from the reviewed draft | Pending actual deposit month. It must be updated before submission. |
| Abstract | Physical page 2; English, 204 whitespace-delimited words; 12-point Arial with 24-point leading | Checked. Adviser and reader names now follow its unchanged scientific text. |
| Contents | Physical page 3; Abstract ii first, followed by existing chapters/references/appendix page numbers | Corrected and checked. |
| Separate lists | Physical pages 4 and 5; all four figures and fourteen tables, correct destination pages | Checked. Short descriptive labels should be reconciled with final captions during revision; no institutional rejection is inferred. |
| Body | 12-point Arial prose with double spacing; consistent headings; tables/code use separate spacing | Checked in source and representative page measurements. |
| Digital margins | All 86 pages' visible ink inside the one-inch rectangle; figure boxes within margins | Independently checked on the prior candidate. Unedited-page renders are identical to that candidate; revised front matter is visually checked. |
| Pagination | Hidden cover number; ii–v front matter; Introduction 1 through final page 81; centered bottom numbers inside margins | Checked. |
| Blank pages/fonts | No blank pages; all retained font objects embedded | Checked. |
| Archival conformance | veraPDF Greenfield 1.30.2, explicit PDF/A-2b profile | Pass for exact candidate bytes: 144 rules and 232,678 checks; zero failures. Accepted program profile remains to be confirmed. |
| Print/binding | Current file uses one-inch digital margins | Program must resolve whether binding is required. A print edition requires a separate 1½-inch left margin and renewed review. |
| Final approval/deposit | No academic approval or deposit receipt supplied | Pending actual defense, approved edits, student decisions and institutional records. |

The original review draft had three concrete front-matter gaps: double-spaced author lines, missing faculty names after the abstract, and contents beginning with Chapter 1. The separate candidate corrects these without altering the reviewed original. Its 83 other pages retain the same text and word coordinates. The abstract's scientific text and word coordinates are also unchanged.

The original main-thesis builder still reproduces the original review edition. `tools/build_archival_candidate.py` owns the candidate's additional front-matter preparation. Any final revision must carry these corrections into its own build, use the actual submission month and recheck all pagination, lists, margins and exact written PDF/A bytes. The institutional formatting review occurs through the submission process; conversion or agent review does not replace it.
