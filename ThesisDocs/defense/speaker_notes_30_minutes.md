# Thirty-minute defense speaker notes

## Slide 1 • 00:00–00:45 • Cost aware stopping boundaries in reasoning language models

My thesis studies a decision that a reasoning system must make repeatedly: after the current answer, is another response worth its computation? The study concerns complete response-and-revision increments. It does not observe every internal thought or locate an optimal token inside a response. I will first establish the mathematical decision problem, then examine the two frozen experimental corpora, and finally show the implementable causal stopper and its measured limits. The key separation throughout is between information available at a decision and information visible only after a full trajectory. These are research findings and a defense package; institutional approval and degree completion require the committee's actual review.

Sources: ThesisDocs/Masters_Thesis_Draft_v1.md; ThesisDocs/ADVISOR_MEETING_SEMESTER_2_ROADMAP.md


## Slide 2 • 00:45–01:40 • When is another reasoning increment worth its cost?

A final-answer score hides important changes along a revision trajectory. A model can repair an initially wrong answer, and it can replace a correct answer with a wrong revision. I use repair and corruption as changes in externally graded candidates, not as claims about hidden cognition. There is also a separate economic outcome: continuation can be unproductive even if average accuracy still increases, because the increase may be smaller than the cost. The first research question is where this happens in the recorded panels. The second is what a policy can infer from a permitted prefix. The third is whether actual generation can be avoided while retaining an acceptable accuracy trade-off. The thesis evaluates these separately. Neither a high classifier AUC nor a fast controller call establishes that the system preserves accuracy.

Sources: ThesisDocs/chapters/chapter1_intro.md; research/mathematical_foundations.md


## Slide 3 • 01:40–02:45 • Two corpora answer different questions

The canonical revision matrix and the standardized detector corpus have different observation rules and denominators. The first supports the boundary and matched estimator analyses. Its 75,965 observations here are trajectories, not reasoning steps; the corresponding raw step table contains 798,770 rows. The second supports five-step detector evaluation and contains 28,888 trajectories, or 144,440 rows. Both have 52 model-domain cells, but the standardized selection replaces InternLM with Qwen3.5-9B. Their task sets overlap, so summing the task or trajectory counts would double count evidence. A table or claim must identify its own corpus. GPQA also has a recorded metadata-versus-loader split discrepancy; the manuscript retains that provenance limitation. No SVAMP result is claimed in this selected corpus.

Sources: data_manifest_v1.json; ThesisDocs/chapters/chapter3_methodology.md


## Slide 4 • 02:45–04:00 • The observation unit is a complete revision response

The repeated-response experiment supplies a concrete point at which to observe a candidate and decide whether to schedule another response. A revision is therefore not merely another token in one uninterrupted internal chain. The archive records candidate answers, parse outcomes, token counts and additional instrumentation. In the live adapter, the required JSON response boundary is shared by active and full-horizon arms. Both enforce a floor of two and maximum of five. The floor protects the declared task protocol; it is not a theorem that two increments are universally necessary or sufficient. The historical panel uses seed seven at three temperatures. Those are three temperature settings, not independent generation seeds. A controller can stop future calls only after the currently observed increment completes. Within-call cancellation is a separate engineering control.

Sources: ThesisDocs/chapters/chapter3_methodology.md; research/online_generation.py; research/online_stopping_controller.py


## Slide 5 • 04:00–05:10 • Correctness is hidden from the stopping policy

The filtration formalizes the declared decision information. It grows when a response or a probe finishes. Offline correctness labels evaluate execution and are not supplied to the runtime. The candidate selector is part of the policy contract. The default carries forward the latest nonempty available answer; baseline and learned arms use it. The heuristic can retain the previous answer on a confidence drop or the best observed valid confidence on wobble. Neither branch triggered in the reported runs. C denotes correctness of the selected candidate, and q is its theoretical conditional expectation; reported confidence is not automatically q. A stopping time is admissible when deciding to stop by time t uses information present by t. A full-trajectory score or unseen future answer violates that declared interface. Mathematical measurability is a separate assumption: a known deterministic reference can make q equal C even without a gold field.

Sources: research/mathematical_foundations.md; research/online_stopping_controller.py


## Slide 6 • 05:10–06:15 • Accuracy, utility and computation are separate outcomes

The cost-sensitive reward is expected correctness minus cumulative computation cost. The cost process is adapted, integrable and nondecreasing in the natural computation setting. A unit response penalty of point zero five is a declared preference: it is not a measured price of every response. The token-weighted analyses use their separately declared scaling, and actual completion-token savings do not inherit that scaling as a dollar or energy saving. For a correct answer worth v and an incorrect answer penalized by p, the coefficient of correctness becomes v plus p. Under a fixed potential process, unchanged prompts, selector and nondecreasing costs, higher stakes produce nested earliest optimal stopping sets. That corollary does not apply when changing stakes also changes generation. The empirical report therefore presents accuracy, utility, generated tokens, prompt work and time side by side rather than converting one into all the others.

Sources: research/mathematical_foundations.md; ThesisDocs/chapters/chapter5_online.md


## Slide 7 • 06:15–07:35 • Conditional repair and corruption yield the exact drift

The drift identity follows by separating the only two transitions that change a binary correctness label. The increment C next minus C current equals the repair indicator minus the corruption indicator. Conditional expectation given the current observed information gives the repair joint mass minus the corruption joint mass. Factoring these masses gives one minus q times alpha and q times beta. Subtract the expected next cost to obtain mu. The conditional probabilities are defined through these joint masses, so an event with zero conditional probability contributes zero rather than an undefined ratio. This resolves a common denominator error: repair rate divides by currently incorrect candidates, whereas corruption rate divides by currently correct candidates. Subtracting those two conditional rates directly would not equal population accuracy drift. With affine stakes, the accuracy part is multiplied by v plus p, then the same incremental cost is subtracted. This identity is exact; estimating its terms is a separate statistical problem.

Sources: research/mathematical_foundations.md

- [Ferguson, Optimal Stopping and Applications, finite horizon](https://www.math.ucla.edu/~tom/Stopping/sr3.pdf)
- [Peskir and Shiryaev, Optimal Stopping and Free-Boundary Problems (2006)](https://doi.org/10.1007/978-3-7643-7390-0)

## Slide 8 • 07:35–09:20 • Bellman continuation value defines the optimal stop

The finite-horizon theorem is the main optimality result. Start at the terminal reward, then recursively compare stopping now with the conditional value of continuing one step and following an optimal later policy. Backward induction makes S adapted and integrable, shows it dominates G, and makes it a supermartingale. Any integrable supermartingale dominating G must dominate S by the same backward induction, so S is the smallest such process. Optional stopping in the finite bounded horizon gives an upper bound S at the current time for every admissible stopping reward. Before the first contact with G, the recursion uses conditional continuation with equality. The Snell process stopped at that first contact is therefore a martingale from the chosen starting time. At contact its value equals the reward, so that stopping rule attains the upper bound. The displayed continuation advantage subtracts G now from expected S next. It splits into immediate drift plus a nonnegative future option value. A positive drift forces continuation in this reward model. A nonpositive drift can still be outweighed by that future value.

Sources: research/mathematical_foundations.md; research/tests/test_mathematical_foundations.py

- [Ferguson, Optimal Stopping and Applications, finite horizon](https://www.math.ucla.edu/~tom/Stopping/sr3.pdf)
- [Peskir and Shiryaev, Optimal Stopping and Free-Boundary Problems (2006)](https://doi.org/10.1007/978-3-7643-7390-0)

## Slide 9 • 09:20–10:55 • A first drift crossing is valid under persistent signs

The boundary result is deliberately narrower than the Bellman theorem. Suppose the conditional drift is positive before its first nonpositive value and remains nonpositive afterward along every admissible realized path. For any competing stop, decompose the expected reward difference using the Doob decomposition and the bounded stopping times. If the competitor stops before the crossing, it omits positive conditional drifts. If it stops after the crossing, it adds only nonpositive conditional drifts. Both differences are nonnegative in favor of the first crossing. This proves optimality under the persistent-sign condition. One sufficient structural pattern is increasing q, decreasing repair hazard, increasing corruption hazard, and constant or nondecreasing predictable cost, all pathwise. These are strong hypotheses, not claims established for every language model. Average negative drift at every time still permits an adaptive policy to exploit subgroups. The source includes a second exact counterexample where all population drifts are negative yet observation-dependent continuation improves value. The empirical crossing curves are descriptive unless the required conditional structure is established.

Sources: research/mathematical_foundations.md; research/tests/test_mathematical_foundations.py


## Slide 10 • 10:55–12:30 • Delayed repair defeats a myopic stopping rule

This small deterministic example is enough to disprove a universal first-negative-drift rule. The first extra response remains wrong. The second extra response produces the correct answer. Each response costs one tenth. A one-step policy at the first time sees no correctness improvement and a positive cost, so it stops and receives zero. A policy that pays for both responses earns one minus two tenths, or point eight. The Bellman recursion also gives point eight at the first time because it carries forward the future repair opportunity. This example is realizable with candidates judged against one fixed reference answer; it is not an impossible collection of unrelated marginal probabilities. The mathematical source supplies that realization and independently enumerated verification. The conclusion is about scope: a learned estimate of immediate drift can be causally valid and still fail to be globally optimal. Empirical success of a drift heuristic does not remove this counterexample. Approximating Bellman value would require a transition model or direct continuation-value learning with appropriate validation.

Sources: research/mathematical_foundations.md; research/tests/test_mathematical_foundations.py


## Slide 11 • 12:30–13:40 • A ranking score is not a calibrated probability

The theory requires conditional probabilities under the runtime information filtration. A fitted score is an estimator, and a feature-restricted model generally estimates a coarser target. High AUC shows ranking discrimination; it does not show that a score of point nine corresponds to ninety percent correctness. Weighted logistic loss changes the population optimum: with class weights a and b, its probability is a times q divided by a times q plus b times one minus q. The original posterior can be recovered algebraically only under that loss-and-population model or calibrated with independent representative labels. The new portable predictor avoids class weighting and uses independent task groups for Platt calibration. Even then, Brier score and reliability bins are finite-sample marginal diagnostics. They do not prove conditional calibration on every prefix or transfer to a changed prompt. Treating the public domain and a few current and past response statistics as a Markov state would require an additional transition-sufficiency assumption; the thesis does not assert it.

Sources: research/mathematical_foundations.md; research/prefix_stopping_model.py; research/outputs/semester2/prefix_model_v1/evaluation.json


## Slide 12 • 13:40–14:45 • Repeated checking needs a valid sequential target

A sign certificate has a precise form. If upper bounds cover the true conditional drift simultaneously at all admissible decision times, stopping when the upper bound is nonpositive controls stopping on a positive true drift by the coverage error. It does not certify Bellman optimality or zero accuracy loss. A Hoeffding construction requires a named pre-probe sigma-field, bounded independent probes given that field, and a conditional mean target. Once probes are observed, the policy's information changes. A bound on the pre-probe mean must not be relabeled as a bound on the post-probe drift without a comparison argument. Similarly, a nonnegative e-process needs conditional supermartingale increments under its declared null, not just favorable unconditional averages. The source proves the elementary claim and supplies a shared-sign dependence counterexample. The repository's across-task empirical-Bayes or e-process diagnostics can describe a population panel; they cannot automatically issue a safe live certificate for one question.

Sources: research/mathematical_foundations.md

- [Hoeffding (1963), Probability inequalities for sums of bounded random variables](https://doi.org/10.1080/01621459.1963.10500830)
- [Howard et al. (2021), Time-uniform, nonparametric, nonasymptotic confidence sequences](https://doi.org/10.1214/20-AOS1991)

## Slide 13 • 14:45–15:45 • The freeze makes claims traceable

The reproducibility freeze binds each reported number to a selected source and a documented transformation. On this Windows checkout, canonical LF content and exact local file bytes can have different hashes, so the manifest records both and checks the intended rule. That solves line-ending identity; it does not excuse other changes. The mathematical tests compare Bellman values against independently enumerated admissible policies in 4,374 finite reward trees and check stake ordering in another 2,187 bounded systems. These are meaningful verification of identities and implementations, not empirical evidence about language models. The new prefix model has a serialized coefficient order, scaler, calibrators, source hashes, task roles and row-level label reconstruction audit. The original archived corpus remains unchanged. A preserved copy of the baseline freeze used in training prevents a later manifest refresh from obscuring the training inputs. Missing historical environment details remain limitations rather than being replaced by the current workstation's versions.

Sources: data_manifest_v1.json; research/tests/test_mathematical_foundations.py; research/tests/test_prefix_stopping_model.py; research/outputs/semester2/prefix_model_v1/training_protocol.json


## Slide 14 • 15:45–17:00 • Repair and corruption use different denominators

This table connects the conditional identity to observable transition counts. At the step-two GSM8K prefix, the current incorrect candidates form the repair denominator and the current correct candidates form the corruption denominator. The counts are 3,145 repairs among 14,818 incorrect candidates and 1,170 corruptions among 4,682 correct candidates. Their difference divided by 19,500 is the accuracy change, and subtracting point zero five gives net utility drift about plus point zero five one three. At step four, the positive accuracy change is about point zero three seven three; after the same cost, net drift is about minus point zero one two seven. Thus the answer accuracy can still improve while continuation is economically unproductive under this utility. These are pooled panel quantities over model configurations and temperatures. They do not reveal which individual task should stop. Uncertainty uses task clusters rather than treating all repeated trajectories as independent questions.

Sources: research/outputs/thesis_v1/evidence/boundary_domain_step_metrics.csv


## Slide 15 • 17:00–18:05 • The observed crossing depends on the cell

The selected-cell contrasts show why a universal step-two or step-three boundary is not supported. In the Qwen2.5-7B GSM8K cell, expected accuracy improvement minus the response cost is positive at the step-four prefix and negative at step five. For Qwen2.5-32B on MATH, the displayed transition is positive at step five and negative at step six. The same declared utility can therefore produce different descriptive crossing locations across model-domain cells. Each cell has 500 questions at three temperatures, giving 1,500 trajectories; those are not 1,500 independent questions. The graph plots stored net drifts for these exact prefixes, not a fitted smooth curve. It should not be extrapolated to other cells, an uninterrupted token-level thought chain, or a policy-optimal per-instance stopping time. The theorem's persistent conditional signs remain an additional assumption.

Sources: research/outputs/thesis_v1/evidence/boundary_selected_cell_step_metrics.csv


## Slide 16 • 18:05–19:15 • More expressive estimators did not always help

Matched comparisons are more informative than comparing a new arm to an unrelated aggregate. The empirical-Bayes hazard replacement improves the recorded step utility by about point zero zero seven eight per trajectory relative to its matched local logistic estimator. Lagged features add a smaller positive contrast. Gradient boosting and isotonic calibration show substantial negative utility contrasts in their recorded experiments. Those outcomes should remain in the thesis; they constrain the claim that a richer score or a calibration layer automatically improves stopping. The intervals are percentile resamples of the 52 selected model-domain cells, not independent generation seeds and not uncertainty over every possible language model. The historical plus 593.55 figure for the hazard arm is a summed utility change over 75,965 trajectories. It is not a peer-agreement effect and must not be reassigned to that mechanism. Nor does the isotonic result prove that calibration is inherently harmful: the model, sample, target and downstream threshold jointly determine policy performance.

Sources: research/outputs/thesis_v1/evidence/algorithm_v2_normalized_effects.csv


## Slide 17 • 19:15–20:25 • A paired precision contrast has a large local effect

The numerical precision comparison is a paired local result. BF16 produces 418 correct candidates and four-bit weights 204 at the recorded step-two endpoint, among the same 1,500 task-temperature trajectories. The difference is 214 over 1,500, or 14.27 percentage points. It is an absolute accuracy contrast, not a fourteen percent relative reduction and not a result for every quantized language model. The task-cluster interval is approximately 11.13 to 17.53 points. A separate cap comparison for Mistral-Small-22B on GSM8K finds 454 losses in each of the 256- and 512-token arms with zero discordant loss indicators. That negative result applies to the specified binary policy-loss endpoint. It does not prove that token sequences, timing or all correctness outcomes are identical, and it does not rule out truncation across other cells. Both comparisons retain their original endpoint, unit and paired design.

Sources: research/outputs/thesis_v1/evidence/algorithm_v2_normalized_effects.csv; ThesisDocs/chapters/chapter4_empirical.md


## Slide 18 • 20:25–21:30 • Pooled causal discrimination hides a weak domain

The standardized corpus allows a prefix-safe sequence comparison under task-grouped evaluation. The causal GRU's micro AUC is point eight seven four three, but task-macro and domain-macro summaries are lower. The worst-domain GPQA score is about point six three one three. This difference matters for deployment: a pooled score can be driven by easier or larger groups while leaving a weak subgroup. Task-macro AUC is defined only for tasks with both classes; 2,679 of the 2,948 task groups are eligible. The causal RoPE transformer has slightly lower micro AUC and slightly higher recorded step utility than the GRU. Their descriptive fold intervals overlap, so the summaries do not establish a unique architecture winner. These are stored archive predictions. They support retrospective causal-prefix scoring and replay; they are not the new portable model's live AUC, and they do not establish conditional calibration.

Sources: research/outputs/thesis_v1/evidence/tournament_balanced_summary.csv; research/outputs/thesis_v1/evidence/tournament_per_domain.csv


## Slide 19 • 21:30–23:00 • The retrospective 0.955 score is not a live guarantee

The most attractive historical number needs the clearest qualification. The stacked hybrid has stored AUC point nine five five one five six, versus point nine four three two two three for its reduced-feature LightGBM control. The task-bootstrap lift interval describes their difference on this development evaluation. It is not an absolute AUC interval. The bidirectional component reads all five saved steps and copies a trajectory score to earlier rows, while centered smoothing also uses later observations. Those inputs are unavailable at an early runtime stop. In addition, upstream meta-training is not confined to each reported outer training fold, so task-grouped scoring does not remove that dependence. The control labeled No Peers retains committee aggregates, preventing a pure peer-effect interpretation. The point nine five five and point eight seven four three scores therefore cannot be presented as before-and-after evidence that removing future information caused a particular AUC drop. They are different evaluations. The portable live stopper consumes neither of these full-sequence scores.

Sources: ThesisDocs/chapters/chapter4_empirical.md; ThesisDocs/chapters/chapter6_discussion.md


## Slide 20 • 23:00–24:20 • The controller prevents future response calls

The controller accepts one newly completed Observation with no gold, future row or full-sequence score. It owns the prefix, rejects skipped or repeated steps, and permanently closes after stopping. Strict JSON is necessary for trusted confidence: a legacy fallback default must remain missing. A peer panel requires the complete declared same-step roster, a barrier before the decision and charged peer computation. The default live policy uses no peers or verifier. The final saved-MATH benchmark has 100 problems, 20 repeats and 18,880 combined heuristic/control decisions. Median latency is 0.0034 ms and maximum 0.4383 ms. This covers prepared-observation validation, policy features, selection and closure, excluding telemetry extraction, loading, generation and peer waits. It establishes a worst-observed under-ten-millisecond result on this workload, not a worst-case execution-time proof. Separately, the learned main-panel benchmark makes 4,000 decisions over 100 problems and 20 repeats. Its maximum is 0.9369 ms, including prefix features, both calibrated heads and the controller, and excluding generation, loading, tokenization and peer waits.

Sources: research/online_stopping_controller.py; research/online_generation.py; research/outputs/semester2/online_stopping_20261002/latency_summary.json; research/outputs/semester2/online_stopping_20261002/learned_main/learned_latency_summary.json


## Slide 21 • 24:20–25:25 • A portable trained model estimates current and next correctness

The thesis now includes a trained deployable artifact rather than only a confidence heuristic. It fits two probabilities: correctness of the latest nonempty currently available candidate and correctness of the same selector after one more response. Public task identity fixes roles before fitting or parser availability. All 1,500 archived Qwen2.5-0.5B tasks are retained: 902 training, 322 calibration and 276 evaluation. The scaler and base logistic coefficients are fitted only on training tasks; Platt calibrators use only the independent calibration tasks. Evaluation enters neither fit. The feature contract contains current and prior answer churn, response and thought length, observed confidence when strict-valid, fixed text-density summaries, step and public domain. It excludes labels, timestamps and future rows. Archive-only gold regrading aligns reconstructed live-parser candidates with targets: 4,269 of 7,500 candidate strings differ from saved strings, but only 43 correctness labels change. The original corpus is preserved. These held-out scores validate marginal archive prediction, not runtime conditional certainty.

Sources: research/prefix_stopping_model.py; research/train_prefix_stopping_model.py; research/outputs/semester2/prefix_model_v1/training_protocol.json; research/outputs/semester2/prefix_model_v1/label_reconstruction_audit.csv


## Slide 22 • 25:25–26:45 • The trained drift policy nearly reduces to fixed two

The declared policy is myopic: subtract current probability and the fixed response cost from the next-correctness estimate, and stop at the first nonpositive result after the floor. It does not approximate a validated Bellman continuation value. On the 276 held-out archive tasks, 275 stop at step two and one at step three. The learned policy has essentially the same accuracy as fixed two and slightly greater recorded cost. It therefore does not demonstrate added policy value over that simple baseline. Relative to full horizon, its paired accuracy difference is plus 1.45 percentage points with a task-cluster interval from minus 1.45 to plus 4.35 points. Replay completion-token savings are 51.29 percent, with a task-cluster interval about 49.31 to 53.09 percent; those are archived counts, not measured live savings. Archive outputs contain no strict-valid JSON records, so confidence-related features lack training support. Live JSON prompting and charged EOS tokens also differ. A historical in-sample 7B replay saved 54.34 percent while losing 6.33 accuracy points; it remains a separate development diagnostic.

Sources: research/outputs/semester2/prefix_model_v1/evaluation.json; research/outputs/semester2/prefix_model_v1/heldout_policy_replay.csv; research/outputs/thesis_v1/evidence/offline_replay_metrics.csv


## Slide 23 • 26:45–27:50 • Actual generation yields different policy trade-offs

The frozen learned policy now has actual generation results. It produces 9,674 completion tokens versus the prior full-horizon arm's 22,244, a 56.51 percent reduction; the task bootstrap interval is 54.83 to 58.01 percent. It answers seven of 100 tasks correctly versus six for baseline. The conservative paired interval for the one-point improvement is minus 4.27 to plus 6.21 points. No noninferiority margin was registered. Every task stops at two, so the result cannot establish added value over fixed two. Shared prefixes match on all 100 tasks. This learned arm is separately generated, not replay; its baseline is the actually executed prior arm linked by hash. The original heuristic saved 2.94 percent with six correct in each arm and shared-prefix agreement on 94 tasks. Baseline JSON success is only 88 of 500 responses. Absolute accuracy is low. Prompt work, padded slots and model seconds have separate ledgers. These are one-model development outcomes; no label was used to tune the frozen policy.

Sources: research/outputs/semester2/online_stopping_20261002/live_metrics.json; research/outputs/semester2/online_stopping_20261002/live_paired_results.csv; research/outputs/semester2/online_stopping_20261002/learned_main/live_metrics.json; research/outputs/semester2/online_stopping_20261002/learned_main/live_uncertainty.json


## Slide 24 • 27:50–28:55 • Trap questions probe the policy's failure modes

The frozen bank contains 20 mathematical traps with independent short derivations: changing percentage bases, dependent draws, harmonic rates, endpoint counting, exponent precedence and answer contracts. The learned arm emits 2,268 completion tokens against 4,736 for full horizon, saving 52.11 percent; descriptive resampling gives 45.86 to 57.17 percent. Both arms answer one question correctly. All 20 learned tasks stop at two, and every shared prefix matches. The conservative iid-reference accuracy-change interval spans minus 19.68 to plus 19.68 points. Because the bank is handpicked, this is not randomized-population adversarial coverage. The heuristic separately saves 3.34 percent with one correct in each arm. A wrong stable answer or a missed late repair can defeat the policy without a software bug. Malformed observations and forged peer receipts belong to the separate state-machine tests. No trap result changes the coefficients, calibrators or threshold. Classic puzzle forms may occur in pretraining; the evidence is robustness on this bank and protocol, not exhaustive worst-case risk.

Sources: research/adversarial_tasks_v1.jsonl; research/adversarial_gold_v1.jsonl; research/outputs/semester2/online_stopping_20261002/adversarial_live/live_metrics.json; research/outputs/semester2/online_stopping_20261002/learned_adversarial/live_metrics.json; research/outputs/semester2/online_stopping_20261002/learned_adversarial/live_uncertainty.json


## Slide 25 • 28:55–30:00 • The contribution is a testable stopping framework

The thesis contributes a rigorous decision framework, a traceable empirical record and an implementable causal stopping system. The general optimum is the Snell-envelope first-contact rule; a first drift crossing needs persistent conditional signs. The empirical panels show local repair, corruption, cost crossings and estimator contrasts, while preserving negative outcomes and uncertainty at the correct cluster level. The engineering system actually prevents future response generation and keeps labels outside its runtime interface. The trained predictor is serializable and calibrated on separate task groups, but its archive policy largely collapses to fixed two and its live transport has limited support. These are honest limits of the completed research result. A confirmatory extension should freeze an unseen task panel, prompt, fitted model and cost preference, state a meaningful accuracy noninferiority margin, use several generation seeds, and report prompt, completion, peer and timing costs. Defense, committee signatures and institutional deposit remain real external events; a generated package cannot substitute for them. I welcome questions about the mathematics, evidence boundaries and deployment trade-offs.

Sources: research/mathematical_foundations.md; ThesisDocs/chapters/chapter6_discussion.md; ThesisDocs/requirements_acceptance_matrix.md


