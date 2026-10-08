# Thesis content audit and targeted restoration

The 31-page v6 thesis was a focused rewrite of the 83-page v5 report. It was not a lossless rewrite: supporting derivations, full numerical tables and secondary analyses were moved to the preserved extended report. No research files were deleted. This audit checks whether the shorter primary document carries the assumptions, methods, evidence and qualifications needed for its own claims.

Version 7 restores about 500 words of useful detail and is 33 total pages digitally and 34 in print. It retains the five-chapter structure, three scientific tables, three figures and twenty-five cited references. The unchanged extended report remains supporting material, with a pinned repository link in Section 5.2.

## Findings and changes

The research questions, information model, observable-reward lemma, binary repair-corruption identity, finite-horizon optimality proof, conditional persistence theorem, delayed-repair counterexample and empirical conclusions were retained in v6. No missing premise was found in the core mathematical argument. Chapter 2 is byte-for-byte unchanged in v7.

V6 relied too heavily on the extended report for experimental configuration detail. V7 restores the complete model-name roster, matrix seeds/temperatures/token cap/domain horizons, the distinct standardized collection settings, and the development-selected two-response floor. It defines trajectory index and completion-token notation, clarifies the common first-response convention, and explains the ratio-of-totals savings denominator.

The runtime account now specifies frozen heuristic thresholds and answer-retention behavior, the learned policy's terminal-first decision order, its twenty-one-feature serialized heads, and the actual greedy generation contract with batch size, token cap, seed and pinned model revision. Head-specific archive AUC, Brier and ten-bin ECE results are restored from the saved evaluation, with row dependence and archive-to-live transport limits intact. These are restored observations, not new experiments.

All seventy mathematical expressions previously printed in v6 remain in their original order. All three scientific table bodies and all three figure paths/captions are identical. Chapters 1, 2 and 5 are byte-for-byte unchanged. Every original methods/results paragraph is retained except two methods paragraphs expanded to add definitions and roster detail. The numerical and policy checks compare the text to frozen manifests and saved evaluation values; their exact paths and hashes are recorded in `content_preservation.json`.

The document QA also detected an incorrect table-list destination caused by a new prose mention of Table 3. The v7 builder now locates full caption titles, and the final audit checks the printed list against the actual caption page. This changes navigation, not a result or table value.

## What remains in the extended report

The longer document retains full calibration/perturbation derivations, sequential-certificate assumptions, a second adaptive-information counterexample, complete per-cell and architecture summaries, selected-answer/peer results, full failure taxonomies, additional latency/Pareto comparisons and detailed historical reproduction records. These items were deliberately excluded from the compact PDF. V7 does not claim that its fitted controller is a Bellman policy, that its probabilities have certified live conditional calibration, that it establishes accuracy noninferiority, or that a substantial prospective thirteen-peer fleet was executed.

The section ledger maps every scientific section and appendix of v5 to its concise location or preserved supporting location. Thus the claim is complete preservation of the research and its supporting files, plus a self-contained core argument; it is not that every old page appears in the primary PDF.

## JHU requirements and peer comparison

The current [ACM guidelines](https://ep.jhu.edu/wp-content/uploads/2024/10/EP-ACM-Research-or-Thesis-Option_guidelines.pdf) require applied or theoretical work beyond straightforward implementation and significant mathematical content. The primary thesis carries the defined stopping model, proofs and counterexample, matched positive/negative findings and measured runtime evaluation. The contribution is a specialization and empirical study, with prior halting and optimal-stopping work acknowledged; the department's assessment of significance remains separate.

The thesis option uses [JHU ETD requirements](https://www.library.jhu.edu/library-services/electronic-theses-dissertations/formatting-requirements/). The source specifies a standard title page, double-spaced abstract/main text, a 166-word abstract, twelve-point Arial body, ordinary type no smaller than ten points, separate contents/table/figure lists, Roman frontmatter and Arabic main pagination, digital/print margins, embedded fonts and PDF/A-2b validation. The cover month must match actual deposit. No thesis-wide page minimum or maximum was found in these sources; the approximately twenty-five-page technical-paper rule concerns the research option.

Four actual MSc theses in the [EP ACM catalog](https://ep.jhu.edu/programs/applied-and-computational-mathematics/acm-student-thesis-and-research-papers-projects/) were checked beyond page counts. Their contents show introductory motivation/prior work, mathematical or methodological development, empirical evaluation and concluding discussion/endmatter. V7 follows these functions across its five chapters. Byerly, Galinkin, Baeder and Columbus have 44, 52, 61 and 88 total pages, respectively. This is a convenience sample, not a census or a mandatory length template. Historical layouts differ, and page count cannot certify research quality.

## Verification and limits

The content audit passes nineteen checks, including exact core preservation, restored head metrics, live settings and policy parameters. It verifies 286 previously published/frozen file hashes, including v5/v6 documents, authoring tools, figures and prior receipts. The twenty-one mathematical-foundation/live-uncertainty tests already passed during v6; research code and those tests have not changed, so no new test run is claimed here. Final v7 PDF/A, rendering, font, margin, folio and caption checks bind the new exact PDF bytes separately.

Semester approvals remain user-confirmed complete. This document audit does not fabricate a public defense, faculty certification, deposit or institutional acceptance. No data collection, regrading, fitting or model generation was needed for the restored, qualified claims.
