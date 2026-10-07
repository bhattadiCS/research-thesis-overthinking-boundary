# Chapter 4 Empirical evidence of overthinking

## 4.1 Population transitions and net value

The next-step accuracy change on a common transition-eligible panel equals repair frequency minus corruption frequency. Subtracting the step penalty gives empirical net gain. This exact decomposition of observed labels does not assume a model of internal reasoning.

[[BOUNDARY_TABLE]]

At step two, repairs outnumber corruptions despite a lower conditional repair probability because more candidates are currently incorrect. The next-step gain after the 0.05 penalty is positive, approximately 0.0513 (Table 4).

At step four, accuracy increases by about 0.0373, below the 0.05 penalty, leaving net gain approximately -0.0127. Thus negative net gain can accompany improving accuracy.

Each panel comprises 500 task clusters and 19,500 trajectories. Its task-bootstrap intervals condition on the observed model and temperature panel and describe a population transition contrast, rather than certifying a particular answer.

[[BOUNDARY_FIGURE]]

## 4.2 Model and domain differences

Selected cells show different empirical crossings. In Qwen2.5-7B/GSM8K, net gain at step four is positive, approximately 0.0193, and step five is negative, approximately -0.0253. In Qwen2.5-32B/MATH, step five is positive, approximately 0.0187, and step six is negative, approximately -0.0133. Each cell contains 1,500 trajectories over 500 questions. The intervals in the evidence tables quantify variation over those questions.

These crossings describe model- and domain-dependent continuation windows. Curves can return to positive gain, so neither a universal accuracy peak at step two or three nor optimality of the first negative gain follows. A final crossing selected after observing the complete curve is a retrospective description, not a live stopping time.

The design does not isolate parameter count from architecture and training differences. A longer revision window therefore characterizes the specific models, settings and questions. A universal scaling law requires additional models, independent generations and a prespecified relationship tested on new data.

## 4.3 Controlled estimator comparisons

[[CONTROLLED_TABLE]]

Empirical-Bayes step hazards improve mean step utility by 0.00781 per trajectory over the matched cell-local logistic baseline, with a 52-cell interval approximately [0.00276, 0.01355]. Lagged logistic features and the step-two churn contrast also improve utility; Table 5 reports their mean controlled effects.

The historical '+593.55' is a summed utility difference over 75,965 trajectories; Table 5 normalizes it per trajectory. It measures the hazard-estimation contrast, rather than an accuracy percentage-point gain or an isolated causal effect of peer agreement.

The gradient-boosted and isotonic arms reduce utility by approximately 0.05694 and 0.06165 per trajectory against their logistic controls. Greater capacity or calibration can therefore worsen this stopping objective, without establishing universal overfitting or harm from calibration. The result depends on the estimator, target, sample size and policy threshold.

## 4.4 Token cap and numerical precision

The Mistral-Small-22B/GSM8K token-cap comparison, 256 versus 512 completion tokens, gives 454 losses among 1,500 trajectories in each arm: the hazard policy has lower utility than never stopping. With no discordant paired loss indicators, the difference is zero and its bootstrap is degenerate. Answers, token counts and timings may still differ; this binary endpoint cannot isolate truncation effects across other model-domain cells.

In the matched Qwen2.5-7B/GSM8K precision comparison, 418 of 1,500 step-two answers are correct under BF16 and 204 under 4-bit weights. The difference is 214/1,500, or 14.27 percentage points, with task interval approximately [11.13, 17.53] points. This model-, step- and implementation-specific absolute accuracy contrast is not a universal relative quantization loss.

## 4.5 Ranking and stopping performance

[[DETECTOR_TABLE]]

The stored prefix-safe causal gated recurrent unit (GRU) [Cho2014] has micro AUC 0.8743, task-macro AUC 0.8214 and domain-macro AUC 0.8102. GPQA is its worst domain at approximately 0.6313, limiting what the pooled score implies for deployment.

The causal transformer with rotary position embeddings (RoPE) [Su2021] has lower pooled AUC than the GRU but slightly higher step utility (approximately 0.3331 versus 0.3264) and token utility (0.3634 versus 0.3631). Overlapping descriptive fold intervals leave a unique architecture winner unestablished; ranking and stopping utility can order configurations differently.

Task-macro AUC includes the 2,679 of 2,948 tasks whose evaluated rows contain both correct and incorrect answers. Within-task AUC is undefined for the others; accuracy and utility still include constant-label tasks.

On the standardized corpus, strict task-grouped tabular and text baselines have independently recomputed saved out-of-fold (OOF) AUCs of 0.849510 and 0.808976, including their original task-disjoint calibration. Appendix D gives the matched peer scores. That comparison adds dynamics above a baseline retaining vote, count and agreement inputs. Its scores and paired folds reproduce, but the executed runner and peer-feature module were not recovered, leaving partial executable-source provenance.

Selected-answer analyses target one causally chosen candidate per barrier over 14,740 decisions and 2,948 tasks, rather than every model-row; their AUCs are not a common benchmark with the row-level results. Configuration selection remains development work. Peer and selected-answer reporting calibrators fit previously computed OOF scores without fully nested outer-fold calibration, so calibrated Brier and ECE summaries are diagnostic. Appendix D retains the original held-out scores, sample sizes and paired contrasts; these are historical analyses, not fresh prospective stopping evidence.

[[DETECTOR_FIGURE]]

## 4.6 The stacked retrospective diagnostic

The stacked hybrid's stored AUC is 0.955156 versus 0.943223 for its reduced-feature LightGBM control, a lift of 0.011933. The task-bootstrap interval [0.010384, 0.013480] describes this development difference, rather than the absolute stacked AUC.

The bidirectional sequence component reads all five saved steps and copies a trajectory score to earlier rows; centered smoothing also uses later observations. Meta-training is not confined to each scoring fold's outer training partition. These dependencies prevent treating the score as an online prefix predictor despite task-grouped scoring.

The control retains committee and independent-vote aggregates, so 'No Peers' is not a peer-free ablation. The stored contrast cannot isolate causal peer value or certify online correctness prediction.

All 10,000 bootstrap lifts were positive (100 percent), a resampling proportion rather than a conventional p-value, universal superiority or independent generation-seed replication. Repeated resampling of the development distribution cannot repair feature leakage or non-nested model selection.

## 4.7 Replay and failure interpretation

The Qwen2.5-7B/GSM8K replay reduces completion tokens from 827,804 to 377,960 (54.34 percent), while accuracy falls from 70.53 to 64.20 percent, a loss of 6.33 percentage points. Fitting and evaluating on the same 1,500 traces makes this a development diagnostic; Chapter 5 reports actual runtime experiments.

Across 75,965 sanitized variable-horizon trajectories, the archived hazard policy wins in 68,095 (89.64 percent), ties in 2,135 (2.81 percent) and loses in 5,735 (7.55 percent) against the full horizon under stored step utility. These are utility verdicts, not accuracy percentages. The audit reconstructs both binary endpoint labels from utility and cost, checks unique pairs and partitions every loss exactly once.

[[FAILURE_TABLE]]

Every observed utility loss stops on an incorrect candidate and ends correctly. Table 7 applies mutually exclusive priority rules, with empty stops first. The 48.35-percent late-repair group first reaches an eligible correct answer at least two steps after stopping; variable horizons prevent interpreting this as a universal step-five outcome.

The taxonomy conditions on archived labels and the stored policy. Recomputation performs no fresh regrading or probe training, and excludes old classification-tag AUCs. These patterns cannot prove that all online predictors must fail. Section 3.5's grader checks cover specified behaviors; independent adjudication is needed to assess labeling errors' effect on the partition.

## 4.8 Empirical conclusions

Continuation value varies with model, domain and cost. Matched estimators can improve or worsen utility; ranking quality and replay savings alone do not establish accuracy preservation.
