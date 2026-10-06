# Chapter 4 Empirical evidence of overthinking

## 4.1 Population transitions and net value

The boundary analysis measures whether another response increment improves average correctness enough to pay its cost. For a fixed panel, the next-step change in accuracy equals repair frequency minus corruption frequency. Subtracting the step penalty produces the empirical net gain. This is an exact decomposition of observed labels on a common transition-eligible panel, not an assumption about the model's internal reasoning.

[[BOUNDARY_TABLE]]

For the pooled GSM8K panel, step two has accuracy 0.2401. There are 3,145 repairs among 14,818 currently incorrect candidates and 1,170 corruptions among 4,682 currently correct candidates. The estimated repair and corruption probabilities are 0.2122 and 0.2499, respectively. Despite the larger conditional corruption probability, repairs outnumber corruptions because many more candidates are currently wrong. The net next-step gain after the 0.05 penalty is positive, approximately 0.0513.

At step four, the panel contains 1,830 repairs and 1,102 corruptions among 19,500 transition-eligible trajectories. Their difference increases accuracy by about 0.0373, which is less than the 0.05 penalty. Net gain is therefore approximately -0.0127. This example demonstrates why a negative net gain does not mean accuracy must fall: accuracy can still improve while the value of improvement is below its assigned cost.

Each of these two panels comprises 500 task clusters and 19,500 trajectories. The reported task-bootstrap intervals condition on the observed model and temperature panel. They characterize uncertainty in a population transition contrast, rather than a probability statement that a particular current answer is correct.

[[BOUNDARY_FIGURE]]

## 4.2 Model and domain differences

Selected cells show different empirical crossings. In Qwen2.5-7B/GSM8K, net gain at step four is positive, approximately 0.0193, and step five is negative, approximately -0.0253. In Qwen2.5-32B/MATH, step five is positive, approximately 0.0187, and step six is negative, approximately -0.0133. Each cell contains 1,500 trajectories over 500 questions. The intervals in the evidence tables quantify variation over those questions.

These results support a model- and domain-dependent continuation window. They do not support a universal claim that accuracy peaks at step two or three across the project. Nor do they prove that the first negative empirical gain is the globally best stopping point. The recorded curves can return to positive gain later. Selecting a final positive-to-negative crossing after seeing the complete curve is a retrospective descriptive choice, not a live stopping time.

The scale comparison also needs careful language. A larger model can have a longer useful revision window in a particular domain, but the design does not isolate parameter count from all architecture and training differences. The observed ladder is evidence about these specific models, settings, and questions. A universal scaling law would require additional models, independent generations, and a prespecified functional relationship tested on new data.

## 4.3 Controlled estimator comparisons

[[CONTROLLED_TABLE]]

Empirical-Bayes step hazards improve mean step utility by 0.00781 per trajectory relative to the matched cell-local logistic baseline, with a recorded 52-cell interval approximately [0.00276, 0.01355]. Lagged logistic features improve utility by approximately 0.00331, and the step-two churn threshold contrast improves it by approximately 0.00210. These are mean controlled contrasts, not hundreds of percentage points of accuracy.

The distinction between a controlled effect and an aggregate score matters. The historical '+593.55' for the pooled-hazard arm is a sum of utility differences over 75,965 trajectories. Dividing by that denominator yields the per-trajectory effect. It is a hazard-estimation result; it is not, by itself, a causal estimate of the value of peer agreement. Assigning the summed number to a differently named mechanism would change the experiment being described.

The gradient-boosted probe and isotonic calibration arms reduce utility relative to their matched logistic controls, by approximately 0.05694 and 0.06165 per trajectory. These negative effects constrain the claim that a more expressive probability model necessarily yields a better stopping policy. They do not prove that all nonlinear models overfit or that probability calibration is inherently harmful. Their effect depends on the fitted estimator, target, sample size, and policy threshold.

## 4.4 Token cap and numerical precision

The matched token-cap experiment compares 256 and 512 completion tokens for Mistral-Small-22B/GSM8K. Both arms have 454 losses among 1,500 trajectories under the recorded binary endpoint: the hazard policy has lower utility than never stopping. There are no discordant paired loss indicators. The empirical difference is zero, and the paired bootstrap is degenerate for that particular indicator.

This equality concerns the paired binary loss indicator. Answers, token counts and timings can differ while that indicator remains unchanged. The contrast therefore measures sensitivity of the recorded policy verdict to this token cap, rather than identifying the contribution of truncation across other model-domain cells.

In the matched Qwen2.5-7B/GSM8K precision comparison, 418 of 1,500 step-two answers are correct under BF16 and 204 under 4-bit weights. The observed difference is 214/1,500, or 14.27 percentage points, with the recorded task interval approximately [11.13, 17.53] points. This is an absolute accuracy difference for that model, step, task panel, and implementation. It is not a 14.3 percent relative reduction shared by every quantized model.

## 4.5 Ranking and stopping performance

[[DETECTOR_TABLE]]

The stored prefix-safe sequence comparison uses gated recurrent units [Cho2014]. Its causal GRU has micro AUC 0.8743, task-macro AUC 0.8214, and domain-macro AUC 0.8102. Its worst domain is GPQA at approximately 0.6313. The contrast between pooled and worst-domain performance shows why a pooled score alone is inadequate for a deployment claim.

A causal transformer with rotary position embeddings [Su2021] has lower pooled AUC than the causal GRU but slightly higher recorded micro step utility, approximately 0.3331 compared with 0.3264. Its token utility is approximately 0.3634 compared with 0.3631. The descriptive fold intervals overlap, so these stored summaries do not establish a statistically unique architecture winner. They do show that ranking and stopping utility need not rank configurations identically.

Task-macro AUC is defined only for questions whose evaluated rows contain both correct and incorrect answers. There are 2,679 such tasks out of 2,948. Omitting the remaining tasks is appropriate for an undefined within-task ranking statistic, but the omission must be stated. Accuracy and utility can still include constant-label tasks.

Additional analyses were completed before the live controller study. On the standardized corpus, strict task-grouped tabular and text baselines have independently recomputed saved out-of-fold AUCs of 0.849510 and 0.808976. Their persisted probabilities include the original task-disjoint calibration stage. A legacy analysis recorded under an anonymous closed-barrier peer-feature contract has raw AUC 0.954664 versus 0.945336 for its matched baseline without the additional peer-dynamics features. The recorded baseline retains vote, count and agreement inputs; this contrast concerns the extra dynamics block. Fixing the roster to thirteen members reduces the corresponding scores to 0.940009 and 0.931266. The saved scores and paired folds reproduce these contrasts, but the exact executed runner and peer-feature module were not recovered from reachable source history. The peer contrast therefore has partial executable-source provenance and remains qualified historical evidence.

Selected-answer analyses use one causally chosen candidate per barrier rather than every model-row target. The no-batch-timing profile and the medium-capacity causal-dynamics profile have raw task-grouped AUCs of 0.934350 and 0.937037 over 14,740 decisions and 2,948 tasks. These endpoints and populations differ from the row-level comparisons, so their AUCs must not be ranked as a common benchmark. Configuration selection remains development work. The selected-answer and committee reporting calibrators are fit across previously computed out-of-fold scores and are not fully nested outer-fold calibration; their calibrated Brier and ECE summaries are diagnostic. The raw AUCs above use the original held-out scores. Appendix E records the completed analyses and distinguishes them from fresh prospective stopping evidence.

[[DETECTOR_FIGURE]]

## 4.6 The stacked retrospective diagnostic

The historical stacked hybrid has stored AUC 0.955156, compared with 0.943223 for its reduced-feature LightGBM control. The observed lift is 0.011933. The stored task-bootstrap lift interval is [0.010384, 0.013480]. That interval describes the difference between two scores on this development evaluation; it is not an interval for the absolute stacked AUC.

Two design details limit its interpretation. First, a bidirectional sequence component reads all five saved steps and copies a trajectory score to earlier rows. A centered smoothing feature also uses later observations. The feature set therefore includes information unavailable to a live decision at an early step. Second, meta-training is not confined to the outer training partition of each reported scoring fold. Task-grouped scoring alone does not remove that upstream dependence.

The control also retains committee and independent-vote aggregates, so its label 'No Peers' does not denote a pure peer-free ablation. The numerical difference is a valid stored diagnostic of the implemented comparison, but it cannot isolate the causal value of peers or certify online correctness prediction. Bootstrap repetition cannot repair either information leakage or non-nested model selection.

The empirical proportion of positive lifts among 10,000 bootstrap draws is 100 percent. It is not a conventional p-value, a proof of superiority on all future questions, or evidence of independent generation seeds. Bootstrap randomness repeatedly resamples the same empirical development distribution. Reporting what it resamples is more informative than describing the resamples as 10,000 independent stress experiments.

## 4.7 Replay and failure interpretation

The historical Qwen2.5-7B/GSM8K replay counts 827,804 full-horizon completion tokens and 377,960 tokens up to the selected stopping steps. The implied saving is 54.34 percent. Accuracy declines from 70.53 percent to 64.20 percent, an observed loss of 6.33 points. The policy is fit and evaluated on the same 1,500 traces, so this result is a development diagnostic. Chapter 5 compares it with the newly implemented runtime experiments.

The paired failure audit covers all 75,965 sanitized variable-horizon trajectories. The archived hazard policy wins in 68,095 trajectories (89.64 percent), ties in 2,135 (2.81 percent), and loses in 5,735 (7.55 percent), relative to the full recorded horizon under the stored step utility. These are utility verdicts, not accuracy percentages. The audit reconstructs both binary endpoint labels from their utility and step cost, checks unique paired trajectories, and verifies that the taxonomy partitions every loss exactly once.

[[FAILURE_TABLE]]

Every observed utility loss stopped on an incorrect candidate and ended with a correct full-horizon candidate. This pattern describes the current records; a correctness gain after an early stop can outweigh the accumulated cost. Among those losses, 196 stop states have an empty extracted candidate. Another 460 traces contain an earlier correct decision-eligible answer, and 600 contain a correct step-one answer but no correct eligible answer before stopping. The remaining losses first reach an eligible correct answer one step after stopping (1,706) or at least two steps later (2,773). The categories follow a fixed priority order, so an empty stop takes precedence over other properties of the same trace.

The larger late-repair group is 48.35 percent of losses. It does not mean that these traces invariably repaired at step five, or that all online predictors must fail. Horizons vary in this matrix. The recomputation uses frozen labels and does not execute a new regrade or train a new probe. Old probe AUCs embedded in the classification script's descriptive tags are excluded from this table. A failed linear predictor of a late repair is evidence about that predictor and its recorded features; it is not an information-theoretic impossibility proof for every possible online signal.

The taxonomy is conditional on the archived labels and the stored policy. The versioned grader coverage described in Section 3.5 tests specified normalization and equivalence behaviors; independent corpus adjudication would assess whether labeling errors alter this partition.

## 4.8 Empirical conclusions

The stored evidence supports cost-sensitive continuation decisions that vary by model and domain. Repairs can dominate early aggregate changes, later gains can fall below their cost, and estimator changes can improve or worsen utility. It also supplies direct counterexamples to several overly broad claims: pooled AUC is not live policy accuracy, a zero token-cap contrast does not establish universal absence of truncation, and replay savings need not preserve accuracy.

Chapter 5 evaluates frozen policies that use only the observed prefix and directly control subsequent response generation. Their accuracy and computation measurements assess the operational consequence of these continuation trade-offs.
