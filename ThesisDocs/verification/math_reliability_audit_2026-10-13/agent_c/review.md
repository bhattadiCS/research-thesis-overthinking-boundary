# Agent C independent initial review

Recorded 2026-10-13. Read PROTOCOL.md and baseline_hashes.json before substantive review. Scope: empirical arithmetic, estimator targets, runtime and statistical implementation for primary concise v7 and supporting v5. Repository CLAUDE.md was read; no AGENTS.md was found by `rg --files -g AGENTS.md`. CURRENT_FORMAL_THESIS.md and formal/source_integrity_v5.json supply the source hierarchy. No prior audit verdict is used as proof.

This initial review precedes exchange of other reviewers' mathematical findings. Numerical outcomes remain NOT ESTABLISHED until the independent source-row checks below finish.

## Initial code findings

- The actual learned-controller decision branch records the completed observation through its base controller, applies the terminal branch first, then at eligible intermediate steps computes `(p_next-q_current)*(reward_value+wrong_penalty)-step_cost`. `research/learned_online_stopping_controller.py:91` onward and `research/online_stopping_controller.py:253` onward agree with v7 Section 3.3's fitted myopic policy, two-response floor, latest-nonempty selector and horizon. This is code correspondence, not establishment that estimated probabilities equal conditional truth or that the rule solves Bellman's problem.
- `research/prefix_stopping_model.py` exposes only Observation prefixes and public domain; feature calculations use current/prior rows, and scoring implements separately standardized logistic logits followed by separate Platt mappings. `research/train_prefix_stopping_model.py` assigns public task hashes before fitting, limits the scaler/logistic to train rows and calibrators to calibration rows, and excludes terminal next-step labels. Serialized artifact and actual source-row correspondence remain to be independently checked.
- `research/trace_analysis.py:611` sanitizes all trajectories touched by malformed numeric fields, not only individual bad rows. Corpus and transition arithmetic must reproduce this selection exactly. Reading only correctness columns would miss eligible-population changes.
- The live uncertainty implementation uses marginal Clopper-Pearson intervals at error .025 each, and subtracts opposite endpoints. Its task bootstrap recomputes integer token totals. The manuscript states the iid-pair and handpicked-bank limitations rather than treating zero discordance as exact zero uncertainty. Source-row numerical reconstruction remains pending.

## Tests completed

Existing Python 3.12.4 environment (NumPy 2.3.5, pandas 2.2.2, SciPy 1.17.1, sklearn 1.8.0):

- `python research/tests/test_mathematical_foundations.py`: exit 0, 12 tests, 1.544 seconds. Log: test_mathematical_foundations.log.
- `python research/tests/test_live_uncertainty.py`: exit 0, 9 tests, .208 seconds. Log: test_live_uncertainty.log.

Tests supplement mathematical derivations; these results do not establish all manuscript mathematics or population validity. PowerShell's initial combined stderr redirection adds NativeCommandError formatting around unittest's stderr, despite both actual native exit codes being zero; the test counts and OK are preserved.

## Independent checks in progress

Raw-source corpus census and population transition panels; frozen coefficient/scaler/Platt scoring and probability metrics; archive current/next target and split correspondence; task-resampling arithmetic; actual event-ledger/policy histories and reused-baseline identities; matched-estimator cache arithmetic; standardized tournament and strict baseline metrics; historical stacked temporal/nesting limitations; source-qualified failure taxonomy and replay limitations. Any inaccessible rows, unresolved source provenance or unrefitted training history will remain explicitly NOT ESTABLISHED.

# Completed independent empirical review

The accessible source-row arithmetic and prefix implementation largely reproduce. The audit does not establish all manuscript mathematics, original-generation authenticity, corpus-wide label semantics, untouched-population inference, live calibration, adaptive benefit or accuracy noninferiority. Exact historical N1 and stacked/replay numerical gaps remain unresolved rather than silently passing.

`ledger.json` contains C-001 through C-021, exact source line anchors, dependencies and outcomes. `coverage_map.json` links 785 empirical inventory units across both compiled manuscripts to those checks. It preserves 238 unresolved paragraph/cell units rather than treating a partly reviewed paragraph as wholly verified. The 546 favorable unit outcomes concern the linked arithmetic/code scopes; central second-review reconciliation is explicitly left to the coordinator. One wording unit requires correction. Numeric duplicates in both editions remain separate coverage entries.

## Reproduced quantities

| Check | Independent result | Evidence |
| --- | --- | --- |
| Matrix | 798,770 raw rows; 798,646 sanitized rows; 75,965 trajectories; 52 cells; 1,948 task IDs | matrix_independent.json |
| Standardized corpus | 144,440 rows; 28,888 five-response trajectories; 2,948 task IDs; 52 files; thirteen models | tournament_independent.json |
| Population curves | All 55 domain/selected-cell strata: accuracy, repair/corruption counts and hazards, net drift, and all four ten-thousand-draw task-bootstrap intervals | matrix_independent.json |
| GSM8K step two | 4,682/19,500 accuracy; repair 3,145/14,818; corruption 1,170/4,682; net drift 1,975/19,500 - .05 = .05128205128 | matrix_independent.json |
| GSM8K step four | Accuracy 7,839/19,500; repair 1,830/11,661; corruption 1,102/7,839; net drift 728/19,500 - .05 = -.01266666667 | matrix_independent.json |
| N2a/N2b/N2c/N3/N4 | Per-trajectory controlled effects -.05693543/-.06164615/+.00331271/+.00781347/+.00209965, all recorded cell-bootstrap intervals reproduced | controls_independent.json |
| N5/N6 | N5 454/1,500 losses per arm, zero discordance; N6 418 versus 204 correct, contrast .14266667, task interval [.11133333,.17533333] | controls_independent.json |
| Causal detector table | All nine configurations' micro/task/domain/worst AUC and step utility reproduce; raw completion and invoked-K2 token utilities agree with recorded float32 accounting | tournament_independent.json |
| Strict OOF | Tabular .849510, text .808976, each 144,440 rows; original labels joined one-to-one | tournament_independent.json |
| GRU | Micro AUC .87432642294, task macro .82139231349, domain macro .81024152301, worst GPQA .63125321260; 2,679 of 2,948 tasks have both labels | tournament_independent.json |
| Archived verdicts | 68,095 wins, 2,135 ties, 5,735 losses; all endpoint utilities match source labels with .05(step-1) cost; every loss is stopped-wrong/final-correct | matrix_independent.json |
| Failure priority partition | 196/460/600/1,706/2,773, late-repair share 48.35222319% | matrix_independent.json |
| Current probability head | 1,380 rows; AUC .71009431387; Brier .09759902461; ten-bin ECE .02222728103 | prefix_independent.json |
| Next probability head | 1,104 rows; AUC .6923, Brier .0997, ECE .0209 at manuscript precision; terminal next targets excluded | prefix_independent.json |
| Prefix training evidence | 902/322/276 task roles; 7,500 label-version rows; 4,269 candidate and 43 label changes; zero strict JSON archive rows; all 1,380 heldout source feature vectors and both train scalers reproduce exactly | prefix_independent.json; prefix_source_independent.json |
| Learned archive replay | 275 stops at two, one at three; 33/276 correct vs 29/276; token saving 51.29%, task interval [49.31,53.09]%; accuracy-change interval [-1.45,+4.35] pp | prefix_independent.json |
| Four live panels | Heuristic main 22,244→21,591 completions; heuristic traps 4,736→4,578; learned main 22,244→9,674; learned traps 4,736→2,268; all paired CP/token intervals reproduced | live_independent.json |
| Actual cost ledgers | Prompt/completion totals, padded slots and recorded model seconds reproduce; main learned total-token saving 67.50%, traps 67.93% | live_independent.json |
| Saved runtime actions | 480 histories, including reused baselines, and 2,020 decisions replay with no action/reason/selection mismatch; closed controllers reject later observations | runtime_compare.json |
| Controller latency | All 18,880 heuristic recorded durations and 4,000 learned durations reproduce; learned median .059 ms, p99 .314315 ms, max .9369 ms | ancillary_independent.json; live_independent.json |
| Frozen Pareto variants | Seven 1,500-trajectory paired files reproduce accuracy/token means and strict dominance flags | ancillary_independent.json |
| Appendix D | All eight saved AUCs reproduce. Anonymous lift .00932828466, 2,000-task-bootstrap interval [.00802751545,.01063899084]; fixed-roster lift .00874303419, interval [.00665303785,.01088174682] | ancillary_independent.json |

Bootstrap draws resample already saved observations; they are not generation repetitions. Source labels are accepted as versioned endpoints. Reproducing percentile endpoints establishes a deterministic development calculation, not nominal 95% population coverage without a suitable sampling/dependence model. In particular, cells share models/domains/questions; a recorded 52-cell resampling scheme is not itself evidence that those cells are independent population draws. Task intervals condition on the fixed model/temperature panel. No changed generation, fitting or regrading was performed.

## Implementation and provenance findings

Current and next heads target the same latest-nonempty answer selector; current/next labels independently reconstructed from the frozen label-version rows agree with prediction targets. Source prefixes determine all features. Original training source parsing at recorded freeze revision `09225c95c676ab3cae1c4b9586d74f7f803f519f` matches all 7,500 reconstructed candidate strings. Current corrected parsing differs on three train/calibration strings; it does not differ on the heldout rows checked here. This version difference is recorded in historical_parser_comparison.json and prefix_source_independent.json, rather than treated as evidence of an invalid frozen artifact. No semantic labels were independently regraded.

The portable artifact has twenty-one serialized features, separate scaler/coefficient/Platt components and no future-reference input. Direct standardization means/scales match 4,510 current-head and 3,608 next-head fitting rows exactly. Coefficients were inspected and applied, never refit. Both heads plus drift reproduce saved learned decisions when computed independently from live public-domain prefixes. The terminal branch precedes inference; the two-response floor and five-response horizon are implemented. Reward one, wrong penalty zero and continuation cost .05 match the manifest. Heuristic prior-answer retention branches exist but do not fire in these panels.

The learned generator's actual batching path removes stopped tasks from the survivors before the next `generate_batch` call. Current-controller replay separately checks historical timestamp ordering and rebases monotonic clocks for replay, because clocks from another machine boot cannot be compared to the current boot. All retained locked-code files match recorded raw or canonical-LF hashes. All original baseline record objects equal the copied learned-panel baseline objects, so those rows are not independent baseline generations.

The historical hazard estimator splits upstream rows by `run_id`, then calibrates thresholds by task. Other-temperature versions of a task can enter upstream training. Strict tabular/text outer fitting and calibration instead use task-disjoint partitions, as does the portable predictor. These distinct experimental contracts should not be combined into an unqualified claim of untouched-question evaluation.

The historical Blackwell stack applies five-point Savitzky-Golay smoothing to complete trajectories and uses bidirectional sequence models with a final-trajectory score copied to earlier rows. Its upstream OOF predictions used for meta-training are not recomputed exclusively inside each meta-fold's outer training set. Reduced-feature controls still retain vote/count/independent-agreement inputs. These source dependencies support the manuscript's retrospective/non-nested qualifications; they do not establish any causal peer value. The fitted one-step controller is not a Bellman continuation policy.

## Correction and exact unresolved evidence

C-017 is a low-severity presentation correction: v5 Table 7 labels 196 normalized-missing cases as “No extracted candidate at the stop.” Direct frozen source joins retaining literal strings find 108 nonempty raw answer strings in these cases, including numeric or chemical answers whose normalized MCQ field is absent. The category predicate is pandas missing/empty `answer_normalized`, not literal empty `answer`. Change the label to “Recorded normalized candidate missing at the stop.” Counts, category priority, utility-loss endpoints and the 48.35% delayed-repair conclusion remain unchanged. empty_candidate_check.json gives exact cells, run IDs, steps, raw strings and hashes for independent checking.

C-018: N1 LOCO/LOMO caches preserve features, targets and utility profiles but do not preserve the per-heldout Ridge predictions needed to reproduce which profile cell was selected. Only the normalization of stored totals 919.20 and 976.80 by 75,965 is independently checked. No Ridge refit was authorized/performed. Exact effects and their intervals remain NOT ESTABLISHED.

C-014: the recorded historical stack output directory has an aggregate JSON/Markdown report. A final expanded search of twenty workspace archives, 3,970 NPZ header dictionaries, five checkpoint paths and thirteen prepared-pickle metadata files found no exact Blackwell per-row stack/control score arrays. The advanced checkpoint contains distinct 202,440-row linear/dynamic/BiGRU/TCN/BetaLikelihood predictions; the ultimate checkpoint contains another experiment's linear/MoE heads. Neither is the 144,440-row Blackwell stack. The arithmetic difference .9551560211009974 - .9432225590934173 = .011933462007580053 is checkable; absolute AUCs, Brier and ten-thousand bootstrap lifts remain unestablished. Code-based temporal/nesting findings are separate.

C-015: expanded search found the original 7B report's three fitted heads rounded to four decimals. Historical source fully specifies four unscaled raw features, so the missing evidence is coefficient precision or original stopped rows. The baseline is independently reproduced from all 15,000 source rows: 827,804 tokens and 1,058/1,500 correct. Deterministic interval evaluation across rounding-compatible coefficients certifies 1,403 stops and leaves 97 unclassified by this conservative calculation. The reported rounded weights yield 377,774 tokens and 963 correct; the historical headline is 377,960 tokens and 963 correct. Conservative outer bounds are 374,891–380,891 tokens and 958–964 correct. Shared coefficient vectors couple the trajectories, so these outer endpoints are not claimed jointly attainable. Exact stopped endpoints therefore remain NOT ESTABLISHED without contradicting the historical full-precision result. No refit occurred. gap_search.json/md and rounded_replay_bounds.json record exact search scope, source anchors and assumptions.

C-016: original generation authenticity, original GPQA executed loader version, independent semantic label correctness, complete historical peer executable-source reconstruction, exact historical GPU regeneration and old grader/suite receipt totals were not established. Accessible saved scores and retained byte identities do not establish those separate claims.

## Tests and preservation

The final clean logs in tests.json record 87 existing cases, all passing: twelve mathematical-foundations, nine live-uncertainty, twenty-five online-controller, ten learned-controller, fourteen prefix-model, three replay-accounting and fourteen ledger cases. Python 3.12.4 with existing dependencies was used. No dependency installation or environment change occurred; final checks disabled bytecode and constrained test scratch to agent_c/test_tmp. Finite checks supplement, never replace, general proofs.

commands.json records final reproducible commands. hashes_*.json record actual input bytes; baseline_integrity_after.json confirms all thirty-six protocol-baseline files remain byte-identical. The audit's scripts, JSON, logs and review are confined to agent_c. No manuscript, data, receipt or production output was edited, and no new training, generation, collection or regrading occurred. Source/PDF correspondence, full theorem review and institutional acceptance are owned separately by the coordinator/theory reviewers.

## Advisor assessment and verdict boundaries

The directly accessible empirical arithmetic supports the qualified protocol-specific narrative: population repair/corruption trade-offs, both positive and negative development estimator contrasts, and real termination with large learned completion savings at weak absolute accuracy. Uniform live stopping and nearly uniform archive replay provide little evidence of useful adaptation beyond the floor. The frozen artifact is inspectable and causal at the response boundary, while archive marginal probability metrics do not establish live conditional calibration or noninferiority.

Primary v7 empirical arithmetic/code verdict: the accessible checked components reproduce under recorded-label and development-panel assumptions, with the exact N1/stacked/historical-replay gaps retained. Supporting v5 empirical verdict: the same checked source results reproduce, additional table/latency/Pareto/Appendix D outputs are reconstructible from saved arrays, and the normalized-candidate wording needs correction. Neither verdict is a blanket “all mathematics verified”; central quantities still need the protocol's coordinator second-review reconciliation, and theorem/PDF/institutional conclusions lie outside this subreview.
