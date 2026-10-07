# Chapter 4 Results

## 4.1 Population continuation value

On the pooled GSM8K transition panel, step two has accuracy 0.2401. Repairs occur in 3,145 of 14,818 currently incorrect candidates, and corruptions in 1,170 of 4,682 correct candidates. Repairs are more numerous despite the lower conditional repair hazard. After the 0.05 penalty, next-step net gain is approximately 0.0513. At step four, accuracy still increases by about 0.0373, but net gain is approximately -0.0127: improving accuracy need not justify its cost.

Each transition panel contains 500 task clusters and 19,500 eligible trajectories. Figure 2 reports task-cluster intervals conditional on the observed models and temperatures. These population estimates do not certify correctness of an individual answer or identify its optimal stopping time.

[[POPULATION_FIGURE]]

Recorded crossings differ by cell. Qwen2.5-7B/GSM8K has positive net gain at step four (0.0193) and negative gain at five (-0.0253); Qwen2.5-32B/MATH remains positive at five (0.0187) and negative at six (-0.0133). Each cell contains 1,500 trajectories over 500 questions. Later positive gains can follow a negative crossing, so a hindsight crossing is descriptive rather than an admissible stopping rule or universal scale law.

## 4.2 Controlled contrasts and retrospective diagnostics

Table 2 normalizes matched effects per trajectory rather than interpreting summed utility as accuracy percentage points. Empirical-Bayes hazards improve recorded utility relative to cell-local logistic hazards, while gradient boosting and isotonic calibration reduce it relative to their controls. Better capacity or calibration can therefore worsen this stopping objective under the tested estimator, target, sample and threshold.

[[CONTROLLED_TABLE]]

The matched 256/512-token comparison gives 454 policy losses among 1,500 trajectories in each arm, with no discordant loss indicators. Its zero contrast and degenerate bootstrap concern that binary verdict; answers and token counts may still differ. The Qwen2.5-7B precision comparison gives 418/1,500 correct step-two answers under BF16 versus 204 under 4-bit weights, an absolute 14.27-percentage-point contrast. Neither result generalizes to every token cap or quantized model.

Strict task-grouped tabular/text saved OOF AUCs are 0.849510/0.808976 and include their original task-disjoint calibration. The causal GRU has micro AUC 0.8743 and worst-domain GPQA AUC 0.6313. Within-task AUC is defined for only 2,679 of 2,948 tasks with mixed labels. These grouped summaries describe ranking; they do not establish a statistically unique model winner or calibrated live policy.

The historical stacked AUC 0.955156 is a non-nested retrospective diagnostic. Its bidirectional component and centered smoothing use future steps; upstream meta-training is not confined to each outer training partition. Its reduced-feature control also retains vote aggregates. Consequently, the result neither isolates causal peer value nor qualifies as an online predictor. Positive lifts in all 10,000 bootstrap draws do not repair those dependencies or constitute independent-generation evidence.

Development replay on 1,500 Qwen2.5-7B/GSM8K traces saves 54.34% of completion tokens while accuracy falls from 70.53% to 64.20%, a 6.33-percentage-point loss. Fitting and evaluation reuse the same cell, so this is a development diagnostic.

The broader archived failure audit partitions 5,735 utility losses among 75,965 trajectories. Every loss stops incorrectly and ends correctly, with 48.35% first reaching an eligible correct answer at least two steps later. This supports delayed-repair as a practical failure mode, conditional on the archived labels and policy. It is not an impossibility result for all online predictors. Full taxonomies and qualified peer/selected-answer analyses remain in the extended report.

## 4.3 Actual stopping execution

Table 3 reports physically generated responses, not tokens inferred solely from stopped-step indices. The heuristic provides small savings with weak absolute accuracy. The learned rule stops all 120 live/trap tasks at response two, giving substantially larger savings but no demonstrated adaptive advantage over a fixed-two budget. Accuracy-change intervals permit meaningful losses and establish neither superiority nor noninferiority.

[[LIVE_TABLE]]

Only 88/500 main baseline responses and 14/100 trap baseline responses satisfy the strict JSON contract. Malformed outputs remain in denominators. Heuristic shared prefixes agree on 94/100 main pairs, limiting exact continuation interpretation; all learned main and trap prefixes agree. Trap results concern a small prespecified challenge bank, not broad adversarial robustness.

Completion savings are not total-cost guarantees. Learned main prompt tokens fall from 142,498 to 43,874; prompt-plus-completion savings are 67.50%, and measured model time falls from 1,127.05 to 343.25 seconds. Trap prompt-plus-completion savings are 67.93%. There are no verifier or peer generations. Batching, padded slots and environment-specific timing remain separately recorded; energy was not measured.

On 4,000 decisions from repeated actual baseline prefixes, learned-controller latency has median 0.059 ms, p99 0.3143 ms and maximum 0.9369 ms. It includes extraction, both heads, calibration, validation and drift; it excludes loading, generation, tokenization and peer waits. The observed maximum meets the under-ten-millisecond decision target, without guaranteeing unchanged end-to-end latency.

[[ACTUAL_COST_FIGURE]]

Held-out archive replay is also nearly fixed: 275 of 276 tasks stop at two and one at three. Accuracy is 11.96% versus 10.51% at the full horizon, with paired interval [-1.45, +4.35] percentage points; replayed token saving is 51.29%, interval [49.31, 53.09]%. This produces an inspectable causal fitted artifact, while providing little evidence that its predictions add useful adaptation beyond the floor.
