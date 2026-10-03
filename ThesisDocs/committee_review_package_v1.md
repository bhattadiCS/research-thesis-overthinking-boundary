# Committee review package for thesis draft v1.0

Prepared October 2, 2026. This is a delivery draft and review procedure. No email has been sent, committee feedback received, meeting booked, or approval recorded by this document.

## Draft delivery message

**To:** Dr. Woods and the thesis committee, including Dr. Pemy; addresses to be supplied by Aditya.

**Subject:** Thesis draft v1.0: cost aware stopping boundaries in reasoning language models

Dear Dr. Woods and committee members,

I am sharing the first complete draft of my Master's thesis, *Cost aware stopping boundaries in reasoning language models*, for your review. The manuscript includes six chapters, formal finite-horizon stopping proofs and counterexamples, a frozen evidence inventory, recomputed empirical comparisons, and actual online stopping experiments with an adversarial question bank. The accompanying repository contains the implementations, task-level outcomes, tests, and reproducible build commands.

The draft also corrects several claims in the earlier progress material. The historical 0.955156 ROC-AUC is a retrospective, non-nested diagnostic. It does not establish online performance. The empirical repair and corruption curves support model- and domain-specific cost trade-offs. The theorem distinguishing a one-step drift rule from general optimal stopping makes its sufficient assumptions explicit.

The learned live arm saves 56.51 percent of completion tokens on the hundred-question panel, with seven correct answers rather than six at the full horizon. Every question stops at step two. The result is a measured budget reduction on a weak model, with no established adaptive benefit over fixed two or accuracy noninferiority. The thesis retains the adversarial failures, uncertainty and prompt-processing costs rather than treating observed equality as an accuracy guarantee.

For the first revision cycle, I would appreciate comments on the research question, the mathematical assumptions and proofs in Chapter 2, the separation of the two archived corpora in Chapter 3, and the strength of the conclusions drawn from the live experiment. For the second cycle, I propose reviewing the revised argument, notation, figures, citations, and final presentation in detail. I will maintain a comment-by-comment resolution ledger and circulate a revised draft after each cycle.

The review PDF and editable source are attached or linked, together with the evidence and milestone audit. Defense scheduling, committee approval, archival submission, and degree clearance remain pending.

Thank you for your guidance,

Aditya Bhatt

## Delivery contents

| Item | Repository path | Purpose |
| --- | --- | --- |
| Full review PDF | `output/pdf/Masters_Thesis_Draft_v1_Aditya_Bhatt.pdf` | Paginated six-chapter thesis with figures, tables and bibliography |
| Editable manuscript | `ThesisDocs/Masters_Thesis_Draft_v1.md` and `ThesisDocs/chapters/` | Complete source and chapter-level revision |
| Formal proof source | `research/mathematical_foundations.md` | Definitions, assumptions, proofs and counterexamples |
| Data and software provenance | `data_manifest_v1.json`, `software_provenance_v1.json`, `requirements.lock.txt` | Evidence identity and current environment disclosure |
| Empirical evidence | `research/outputs/thesis_v1/evidence/` | Recomputed task-cluster and matched contrasts |
| Online experiment record | `research/outputs/semester2/online_stopping_20261002/` | Frozen manifests, actual generation events and paired results |
| Failure analysis | `research/reports/thesis_failure_audit_v1/audit_summary.json` | Complete descriptive partition of archived policy losses |
| Research paper | `output/pdf/Overthinking_Stopping_Research_Paper_v1_Aditya_Bhatt.pdf` | Separate substantive 25-page paper |
| NeurIPS preparation | `ThesisDocs/neurips/`, `output/neurips/anonymous_stopping_supplement_v1.zip` | Anonymous official-style main-track draft and tested review code/evidence capsule |
| Editable defense | `output/presentation/Thesis_Defense_v1_Aditya_Bhatt.pptx`, `ThesisDocs/defense/` | 25 slides, timed 30-minute notes and 46-question bank |
| Revision ledger | `ThesisDocs/committee_revision_ledger_v1.csv` | Reviewer comments, resolutions and verification |
| Completion audit | `ThesisDocs/milestone_completion_audit_v1.md` | Explicit status against every accelerated milestone |

The build manifests identify the exact reviewed artifact bytes. The completion audit records final verification and the remaining academic events. Any later changes require rebuilding and reviewing the affected artifacts before this delivery message is sent.

The chosen paper venue is NeurIPS. The published 2026 main-track format allows nine main-content pages, with references, appendices and the mandatory checklist outside that limit. The anonymous version is separate from the 25-page long paper. The 2026 submission deadline has passed, so this is future-cycle preparation using that official style; the next cycle's rules must be rechecked before actual submission. See the [2026 call for papers](https://neurips.cc/Conferences/2026/CallForPapers) and [main-track handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook).

## Two actual revision cycles

Cycle 1 addresses the argument and mathematical validity. Preserve the original comment, identify the affected definition or claim, document the revision, and cite the proof or empirical artifact supporting it. A reviewer must be able to distinguish a resolved issue from a disagreement awaiting a decision. Produce v1.1 after this work has occurred.

Cycle 2 addresses the revised manuscript at sentence, equation, figure, and reference level. Recheck dependent results whenever a label, cost, denominator, feature, or split changes. Produce v2.0 after this work has occurred. An assistant's internal review is recorded separately and does not count as either committee cycle.

## Mock defense and logistics record

Use the 25-slide deck and its timed speaker notes for a 30-minute talk. Record the first rehearsal with consenting peers, retain the actual start and end times for each slide, and record questions verbatim. For each answer, identify the supporting proof, data or measurement. Revise the notes where the answer is unclear; revise the manuscript where the answer exposes a substantive problem.

The second rehearsal should include the adviser and the hardest questions from the first. Record the actual discussion and decisions. Confirm the defense date, committee availability, room or remote link, accessibility arrangements, and required announcement through the program. No planned rehearsal should be described as completed without its recording or dated record.

## Final archival and clearance gates

Following a successful defense, incorporate the committee's final edits and obtain the required approvals. Check the current university formatting and submission requirements, export the final PDF/A, and retain a dedicated validator report for the exact deposited bytes. Complete the library deposit and any required corrections before recording repository acceptance. Submit the signed completion paperwork through the program's actual process, and retain the registrar's confirmation of the final grade and degree clearance.

The current review PDF is an ordinary embedded-font PDF. It is not asserted to be PDF/A or institutionally accepted. Relevant university sources are the [Sheridan Libraries formatting requirements](https://www.library.jhu.edu/library-services/electronic-theses-dissertations/formatting-requirements/) and [submission checklist](https://www.library.jhu.edu/library-services/electronic-theses-dissertations/submission-checklist/), checked October 2, 2026.
