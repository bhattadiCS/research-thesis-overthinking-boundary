# Appendix A Reproducibility and evidence sources

The original research profile is fixed at revision `09225c95c676ab3cae1c4b9586d74f7f803f519f` by `data_manifest_v1.json`. The post-review profile at `6e4378bef9a98037c20d1381eb9c7b61462a6578` uses `data_manifest_post_review_v1.json` for revised grading, controller validation and behavior checks. Both preserve the collected corpus, fitted predictor and recorded generation outcomes. Each live manifest binds the sources actually executed in its `locked_code` directory.

[[EVIDENCE_TABLE]]

The electronic supplement, `ThesisDocs/formal/supplements/v5/reproduction_and_history.txt`, gives commands for detached checkouts, data verification and isolated reanalysis. Verification uses `tools/freeze_research_data.py`; `tools/recompute_thesis_evidence.py` reconstructs tables, and `tools/verify_post_review_behavior.py` checks saved controller behavior. Analysis outputs go to new scratch directories, and uncertainty analysis operates on copied live panels.

The recorded receipts distinguish thirty historical grader cases from ninety-eight revised tests, and 107 passing historical-suite tests from 243 post-review repository tests. Saved-prefix verification reproduces 480 decision histories across four live collections; reused baseline rows are not additional generations. These counts identify their source versions, rather than measuring corpus-wide label validity.

The fifty-two standardized trace files are mandatory freeze inputs. Supporting evidence includes the variable-horizon matrix, paired study arms, stored predictions and live event ledgers. Exact-byte and canonical-LF hashes identify selected evidence; they do not establish bit-identical regeneration of the partially recorded historical GPU environment. The current authoring manifests separately identify the rendering environment and exact PDF bytes.

# Appendix B Claim scope and mathematical review

**Table 16. Claim scope and required evidence.**

| Claim | Evidence required | Supported interpretation |
| --- | --- | --- |
| Binary repair-corruption identity | Common transition panel or conditional probability proof | Exact decomposition in Chapter 2 |
| General optimal stopping | Full conditional continuation law and all costs | Finite-horizon theorem; not a calibrated learned deployment |
| First drift crossing is optimal | Pathwise persistence after the crossing | Conditional theorem; not established universally by the empirical curves |
| 0.955156 AUC | Stored historical tournament result | Retrospective non-nested diagnostic |
| Causal detector ranking | Task-grouped causal sequence outputs | Internal development evaluation |
| Replay completion-token saving | Frozen traces and selected prefixes | Counterfactual development quantity |
| Live completion-token saving | Actual generation events from both arms | Model-, task-, and protocol-specific measurement |
| No meaningful accuracy loss | Prespecified tolerance and adequate paired confidence interval | Not inferred merely from observed equality |
| Grader regression coverage | Versioned test inputs and execution receipts | Thirty historical cases and ninety-eight revised tests cover specified parser behaviors; corpus-wide validity requires separate adjudication |

# Appendix C Paired accuracy uncertainty

For iid task pairs, let $I$ indicate an incorrect baseline answer paired with a correct active answer, and $W$ a correct baseline answer paired with an incorrect active answer. The accuracy difference is $\delta=\pi_I-\pi_W$, where $\pi_I=\Pr(I=1)$ and $\pi_W=\Pr(W=1)$. The indicators are mutually exclusive within a task; their independence is not assumed.

Across $n$ independent pairs, each marginal discordance count is binomial. Exact two-sided 97.5% Clopper-Pearson intervals $[L_I,U_I]$ for $\pi_I$ and $[L_W,U_W]$ for $\pi_W$ each have noncoverage probability at most 0.025 [Clopper1934]. A union bound gives simultaneous coverage at least 0.95 regardless of dependence between the counts. Subtraction yields

$$\delta\in[L_I-U_W,\ U_I-L_W].$$

This is the conservative 95% paired interval in the live tables. Exact multinomial enumeration supplements the binomial coverage argument. With no discordances, both lower bounds are zero and both upper bounds equal $1-0.0125^{1/n}$. Observed equality therefore leaves nonzero uncertainty; no noninferiority tolerance or hypothesis was prespecified.

The twenty handpicked traps form a fixed challenge bank, so the iid-reference interval gives no randomized adversarial-population coverage. Completion-token intervals bootstrap paired tasks and the ratio of total token differences to baseline tokens; these are descriptive resampling intervals.

# Appendix D Retrospective prediction analyses

Table 17 distinguishes row-level correctness from correctness of one causally selected answer per closed barrier. The endpoints and eligible populations differ, so the scores do not form a common architecture ranking.

**Table 17. Additional retrospective prediction analyses.**

| Analysis | Units and tasks | Saved OOF AUC | Endpoint |
| --- | --- | --- | --- |
| Strict tabular | 144,440 rows; 2,948 tasks | 0.849510 | Current-candidate correctness |
| Strict text | 144,440 rows; 2,948 tasks | 0.808976 | Current-candidate correctness |
| Anonymous peer features | 144,440 rows; 2,948 tasks | 0.954664 | Current-candidate correctness |
| Matched anonymous baseline | Same rows and tasks | 0.945336 | Current-candidate correctness |
| Fixed thirteen-member peers | 98,280 rows; 1,512 tasks | 0.940009 | Current-candidate correctness |
| Matched fixed-roster baseline | Same rows and tasks | 0.931266 | Current-candidate correctness |
| Selected answer without timing | 14,740 decisions; 2,948 tasks | 0.934350 | Selected-candidate correctness |
| Selected causal dynamics | 14,740 decisions; 2,948 tasks | 0.937037 | Selected-candidate correctness |

The strict baseline and paired peer scores were recomputed from saved predictions, with source hashes and task-fold memberships checked. The anonymous peer lift is 0.009328, with paired task-bootstrap interval [0.008028, 0.010639]; the fixed-roster lift is 0.008743, with interval [0.006653, 0.010882]. These intervals describe the archived panels. Selected-answer source and five-fold checkpoint coverage were also checked.

Strict baseline probabilities include the original task-disjoint calibration. Peer and selected-answer scores use original uncalibrated outputs; this check applies no additional reporting calibration. The later reporting calibrators use previously computed out-of-fold scores across folds and are not fully nested, so their Brier and ECE summaries are diagnostic.

Strict-baseline and selected-answer source hashes match retained code. The legacy peer runner and feature-module hashes do not match any recovered Git version, including line-ending reconstructions, although their common base script resolves to the recorded July commit. The saved arrays and paired folds are auditable; exact reproduction of the executed peer-feature pipeline remains incomplete. This qualification covers both peer populations in Table 17.

The historical peer candidate has no assigned stopping threshold. Its full-horizon collector and small smoke ledgers establish neither calibrated prospective fleet stopping nor avoided peer generation. The electronic supplement records the development history and distinguishes reused predictions from independently generated trajectories.

# Appendix E Information flow and delayed repair diagrams

![Offline fitting and runtime information flow](images/thesis_v4/information_flow.png)

**Figure 5. Offline fitting and runtime information flow.** Reference labels enter fitting and evaluation offline; runtime decisions receive observed prefixes and frozen parameters. The continuation loop incurs additional generation cost. This schematic describes the information contract and does not assert a Bellman-optimal fitted controller or an executed peer fleet.

The fitted single-model controller estimates current and next selected-answer correctness and uses a one-step drift rule, with a declared response floor and horizon. The general theorem instead uses conditional multi-step continuation value.

![Decision tree for the delayed-repair counterexample](images/thesis_v4/delayed_repair_tree.png)

**Figure 6. Delayed repair defeats a myopic stopping rule.** The exact counterexample has horizon two, earliest permitted stop zero and incremental cost 0.10. Correctness follows zero, zero, one. Immediate drift at zero is -0.10, whereas continuation to the horizon yields reward 0.80. This abstract floor differs from the live experiments' floor of two; the diagram illustrates the Chapter 2 counterexample rather than new generated data.
