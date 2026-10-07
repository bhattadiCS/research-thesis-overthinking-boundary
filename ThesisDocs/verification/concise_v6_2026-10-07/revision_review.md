# Concise thesis revision v6

The user requested a 25-35-page formal thesis, counting the entire PDF. Version 6 reorganizes the argument into five chapters. The preserved v5 document remains the extended technical report. This is a focused rewrite, not a claim that every item in the longer manuscript fits in the shorter PDF.

## Content retained in the primary thesis

| Research component | Location in v6 | Review finding |
| --- | --- | --- |
| Research questions, endpoint and prior work | Chapter 1 | Distinguishes successive complete-response revision from independent sampling, early layer exit and within-chain truncation; no untested comparative-superiority claim. |
| Information and reward model | Section 2.1 | Finite integer horizon, admissible stopping times, measurable causal answer selection and grading, adapted integrable cumulative cost, and conditional correctness are defined. The deterministic-reference information issue is explicit. |
| Observable reward lemma | Section 2.1 | Finite sum and conditional expectation justify replacing unobserved correctness by its conditional expectation for every admissible stop. |
| Repair-corruption identity | Section 2.2 | Conditional hazards, zero-probability versions, tower property, incremental cost and the difference between hazards and joint event frequencies remain explicit. |
| Finite-horizon optimum | Section 2.3 | Bellman/Snell recursion and attainment proof retain measurability, integrability, the permitted floor and finite termination. This specializes established theory. |
| Myopic persistence theorem | Section 2.4 | True conditional drift must stay nonpositive after its first crossing. Positive drift before crossing and finite optional sampling after it justify optimality. Fitted heads and population curves do not establish the condition. |
| Counterexample and diagram | Section 2.4, Figure 1 | The horizon-two delayed-repair calculation gives -0.10 immediate drift and +0.80 horizon reward; its floor of zero differs from the live floor of two. |
| Corpora and evaluation contracts | Chapter 3, Table 1 | Both overlapping collections, task dependence, roster differences, inferred GPQA split, single generation seed, grader versions and upstream fold limitations are disclosed. |
| Controller and learned fitting | Section 3.3 | Decision-time inputs, prevention of future generation, training/calibration/evaluation partition, candidate reconstruction and archive/live JSON mismatch are retained. |
| Paired uncertainty | Section 3.4 | Defines iid-reference pairs and both discordance probabilities, uses a union bound on exact marginal intervals, and retains nonzero uncertainty with zero discordances. No prespecified noninferiority margin is invented. |
| Population and controlled findings | Sections 4.1-4.2, Figure 2, Table 2 | Positive and negative effects, normalized units, uncertainty clusters, cap/precision controls and descriptive crossings retain their development scope. |
| Predictor and replay limitations | Section 4.2 | Strict grouped results are distinguished from the future-dependent, non-nested stacked diagnostic. The 7B replay's accuracy loss and delayed-repair taxonomy remain explicit. |
| Physically executed stopping | Section 4.3, Table 3, Figure 3 | Paired counts, accuracy, completion-token costs and intervals are retained. Learned stopping at the floor on all 120 tasks is disclosed; adaptation and noninferiority remain unestablished. |
| Other costs and latency | Section 4.3 | Prompt totals, prompt-plus-completion savings, model time, decision latency and omitted energy measurement distinguish physical endpoints. |
| Interpretation, limitations and reproducibility | Chapter 5 | Labels, transport, holdout reuse, peer acquisition costs, incomplete historical provenance and the immutable extended report link qualify the conclusion. |

The mathematics was checked as an argument, not only as rendered notation. Assumptions are sufficient for the finite-horizon results stated; the persistence condition is additional structure, and the delayed-repair example demonstrates its necessity for the proposed sufficient-rule argument. There is no claim of an unconditional online safety guarantee, useful live adaptation, or accuracy noninferiority.

## Supporting material in the preserved v5 report

Full perturbation and calibration derivations, sequential certificate assumptions, the adaptive-information counterexample, complete model/cell rosters, additional selected-answer and peer analyses, full failure taxonomies, all seventeen scientific tables, all six figures and historical reproduction details remain in v5. The new primary thesis includes three scientific tables and three figures. All twenty-five references remain and are cited. V5 and all older artifacts remain byte-for-byte unchanged.

## JHU requirements checked

Primary sources: [ACM research/thesis guidelines, revised July 13, 2026](https://ep.jhu.edu/wp-content/uploads/2024/10/EP-ACM-Research-or-Thesis-Option_guidelines.pdf), [JHU ETD formatting requirements](https://www.library.jhu.edu/library-services/electronic-theses-dissertations/formatting-requirements/), and the [EP ACM thesis and research-paper catalog](https://ep.jhu.edu/programs/applied-and-computational-mathematics/acm-student-thesis-and-research-papers-projects/).

| Document requirement | Implementation and verification |
| --- | --- |
| Significant applied or theoretical mathematics beyond straight implementation | Chapter 2 develops a conditional mathematical stopping argument and Chapter 4 tests its empirical implications; faculty assessment remains separate. |
| Standard title-page fields | University, Master of Science, author, Baltimore and month/year appear on the cover. October 2026 must match the actual deposit month. |
| Abstract no longer than 350 words | 166 words, double spaced, with research adviser and second reader identified. |
| Double-spaced main text and abstract | Main text uses Arial 12 pt with line-height 2; abstract is double spaced. Bibliography and tables use their separate endmatter/table spacing. |
| Font size at least 10 pt; sans serif preferred | Ordinary extractable type is at least 10 pt. Mathematical scripts are measured separately and figures are visually inspected. |
| Digital and print margins | Digital: at least one inch on all sides. Print: at least 1.5 inches left, one inch elsewhere. Visible ink is checked on every rendered page. |
| Front and main pagination | Hidden title folio i; front ii-v; Arabic body begins at 1. Visible folios, logical labels, contents and outline destinations are checked. |
| Separate contents/table/figure lists | Each list begins on a separate frontmatter page; captions and their printed destinations are checked. |
| References and diagrams | Three numbered scientific tables, three figures/diagrams and twenty-five cited references. |
| Archival PDF and embedded fonts | Both exact final files pass veraPDF PDF/A-2b validation and embedded-font/Unicode-map checks. |
| Overall length | No thesis-wide maximum or minimum was found in these sources. The 25-page technical-paper rule concerns the research option. The user's 25-35-page target includes all frontmatter and references. |

The user confirmed semester approvals are complete. This revision verifies the document and research claims; it does not record an unobserved defense, faculty certification, ETD deposit or institutional acceptance.

## Length comparison

Four actual EP ACM Master of Science thesis covers and entire PDFs were inspected: Byerly (2021), 44 pages; Galinkin (2020), 52; Baeder (2022), 61; Columbus (2022), 88. URLs, degree confirmation, counts and downloaded-file hashes are in `jhu_thesis_lengths.json`. These are a convenience sample, not a census. Layouts and historical requirements differ. V5's 83-page digital edition was longer than three of four; its 88-page print edition tied the largest sampled file. The compact version is shorter than all four, without establishing a program-wide rank.

## Research preservation

The protected baseline records the previously published commit `df03265cfe3b9bf4b1ee66061d81ef5796375be0`. Hash checks cover the frozen sources, older manuscripts and PDFs, old audit receipts, figures, root README and shared authoring tools. No new data collection, regrading, fitting or model generation was performed. The mathematical-foundation and live-uncertainty suites already executed for this revision passed 21 tests; layout changes do not require repeating them.

Machine receipts and manual visual review bind the exact final file hashes. Passing software checks supports technical readiness, not academic acceptance.
