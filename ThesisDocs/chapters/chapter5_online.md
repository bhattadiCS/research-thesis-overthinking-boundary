# Chapter 5 Online stopping and computation trade-offs

## 5.1 Causal controller contract

The online controller accepts one newly completed response observation at a time. An observation contains its step number, candidate answer, parsing status, generated completion tokens, and any explicitly available confidence or instrumentation. The API has no field for the reference answer, the correctness label, a future response, or a score computed from the full trajectory. The controller stores its own prefix and rejects skipped, repeated, or out-of-order observations.

The policy enforces a minimum of two completed increments and a finite maximum horizon. Stopping closes the controller, so feeding a subsequent response is an error rather than an implicit restart. The decision records a selected answer and step, the reason for termination, the policy fingerprint, and measured latency. Invalid confidence, incomplete parsing, and future timestamps receive explicit handling instead of becoming undocumented numeric defaults.

Peer agreement is optional. When used, the API requires a complete, same-step roster whose barrier closes before the decision. An incomplete panel, a duplicate peer, or a vote that completes later cannot support a current stopping decision. These checks establish the declared event contract; timestamps alone cannot establish that an external caller supplied authentic observations. A fleet experiment must retain generation events and charge every peer's computation.

## 5.2 Policy choice and limitations

The first runtime policy is a frozen confidence-and-stability heuristic. Model-reported confidence is uncalibrated, and a stable answer can be wrong. Default selection carries forward the latest nonempty answer; confidence-drop and answer-wobble branches can retain a prior candidate. Neither retention branch fired in the reported live panels. The controller also provides fixed-step and never-stop controls under the same floor and horizon. These transparent policies make it possible to compare the engineering effect of termination separately from the value of correctness prediction.

The retrospective 0.955156 stacked score is excluded from the controller. Its bidirectional sequence score and future-step smoothing violate the runtime information contract, and a new causal implementation cannot inherit its AUC by reusing the name of the detector. Stored task-held-out predictions can support replay comparisons, but they are not a deployable fitted model unless the training artifact and its causal feature transformation are available and verified.

The heuristics are engineering choices rather than instances of the general Bellman optimum. A confidence drop may justify caution without proving that all future repair opportunities are exhausted. Oscillation can signal instability without distinguishing a wrong candidate from a correct one. Chapter 2's delayed-repair counterexample applies directly: immediate evidence against continuing does not, without structural assumptions, establish a globally optimal stop.

## 5.3 Generation integration

The generation adapter requests an increment, constructs its permitted observation, evaluates the controller, and requests the next increment only if the decision continues. A stop therefore prevents future model calls. A caller cancellation is propagated through the adapter's generation lifecycle. Baseline and active policies use the same response-boundary and parsing conventions so the stopping comparison does not silently change the response contract.

This integration stops between complete reasoning responses. It does not claim that the risk heuristic identifies an optimal token inside a response. Cancellation and a common response-completion boundary can interrupt a call for engineering reasons, but they are different mechanisms from the policy's step-level decision. Completion-token savings refer to avoided response generation under this declared granularity.

The active and full-horizon arms are executed separately. For deterministic greedy generation, the evaluation checks whether their shared prefixes agree. Changing batch membership or kernels can affect numeric results, so deterministic sampling alone is not proof of identical prefixes. A measured mismatch must be reported, because it limits interpreting the full arm as an exact counterfactual continuation of the active arm.

## 5.4 Decision latency

[[LATENCY_TABLE]]

The latency experiment evaluates decisions on one hundred math-question prefixes, retaining per-step timings and the benchmark context. It reports the median, upper quantiles, and maximum, with the number of measured decisions. A single fast average is not sufficient to establish a worst-observed under-ten-millisecond target. These measurements cover the controller call on already prepared observations, not the language-model forward pass or external peers.

Unit and adversarial state-machine checks complement this timing result. They verify the minimum floor, horizon termination, terminal closure, future-step independence, parsing guards, exact token accumulation, peer barriers, and cancellation. Their passing result establishes implementation behavior on the covered cases. It does not establish that the heuristic preserves accuracy on a distribution of language-model questions.

## 5.5 Live paired evaluation

[[LIVE_TABLE]]

The live experiment freezes the task panel, policy, and generation settings before opening evaluation outcomes. Public task records are supplied to generation separately from the reference-answer ledger. Grading occurs after the generation events are written. This separation makes the information boundary inspectable and prevents a runtime stopping decision from accidentally consuming the benchmark label.

The primary paired quantities are final-answer accuracy, selected-answer accuracy, generated completion tokens, executed response increments, prompt-token counts, and wall time. The report states whether auxiliary scoring or peer generations occur. The ratio-of-total completion-token saving is computed from actual generated counts in the two arms. A completion-token reduction is not automatically an equal reduction in wall time, energy, or total inference cost.

The sample is a local development-scale experiment on one available model. Even with a policy frozen before this run, tasks previously present in the project's corpus cannot be called an untouched external test set. A new run supplies genuine prospective execution evidence, but task novelty, model generalization, and independence from development selection remain separate properties.

The accuracy interval is reported alongside savings. An observed gain or zero difference is not a noninferiority proof unless the interval and a prespecified tolerance establish that claim. This thesis makes no unconditional assertion of zero accuracy loss. Likewise, a result from Qwen2.5-0.5B does not validate the earlier Qwen2.5-7B replay or the thirteen-model detector panel as a live system.

The completed heuristic run answers six of one hundred questions correctly in each arm. No paired correctness indicators change, but the conservative paired interval still allows an accuracy change between approximately -4.29 and +4.29 percentage points under the declared iid task-pair reference model. Completion tokens decline from 22,244 to 21,591, a saving of 2.94 percent; its descriptive task-bootstrap interval is approximately [0.85, 5.63] percent. Only six tasks stop by the confidence-and-stability rule, and ninety-four reach the maximum horizon. The mean stopping step is 4.85. These observations fall well short of the roadmap's suggested thirty-to-forty-percent saving target.

The response contract is a substantial limitation. Only 88 of the 500 full-horizon responses are strictly valid complete JSON, and final baseline accuracy is six percent. The controller can retain a previously extracted nonempty answer when a later increment is empty, but that selection rule does not recover arbitrary malformed reasoning. All resulting failures remain included in the reported denominator. A study on a larger, protocol-capable model is needed before this experiment can support useful deployed answer quality.

Shared prefixes are exactly identical on 94 of the 100 paired tasks, rather than all of them. The remaining differences constrain a trajectory-specific counterfactual interpretation even though both arms use greedy sampling and the same frozen generation adapter. Measured completion tokens are supplemented by prompt tokens (142,498 versus 137,574), model time (1,127.05 versus 1,099.21 seconds), padded prefill slots (231,856 versus 222,781), and padded decoding slots (57,856 versus 56,128). Prompt plus completion tokens fall by 3.39 percent. There are no auxiliary verifier or peer generations. These quantities describe different costs and are not interchangeable efficiency percentages.

## 5.6 Adversarial questions and traps

[[ADVERSARIAL_TABLE]]

The adversarial bank contains twenty independently specified mathematical questions, with separate public prompts and evaluation labels. It includes changing percentage bases, harmonic average speed, conditional probability, dependent sampling, inclusive endpoint counts, reciprocal rates, exponent precedence, answer-contract distinctions, and irrelevant details. Exact short derivations accompany the evaluation labels so they can be checked independently of model output.

This bank is designed to probe common reasoning traps, not to estimate the worst-case failure rate of every possible attack. Some classic puzzle forms may have appeared in model pretraining even though these exact project records are newly written. The measured outcome is therefore robustness on the frozen bank under this model and prompt protocol. The software's malformed-input and forged-peer tests remain a separate engineering evaluation.

The completed heuristic trap run answers one of twenty questions correctly in each arm. Completion tokens decrease from 4,736 to 4,578, a saving of 3.34 percent; all twenty shared prefixes match exactly. Only two tasks stop by stable confidence, and the other eighteen reach the horizon. The mean stopping step is 4.75. Full-horizon strict JSON validity is fourteen of one hundred responses. The conservative iid-reference accuracy difference interval is approximately [-19.68, +19.68] percentage points, and the descriptive token-saving interval is [0.00, 8.80] percent. This small, low-accuracy panel provides no strong accuracy-preservation or broad adversarial-robustness claim.

A wrong but stable high-confidence answer is particularly informative: it demonstrates a limitation of the confidence-and-stability heuristic even when the controller functions exactly as intended. A late repair after termination similarly tests the stopping approximation. The evaluation retains these adverse outcomes rather than choosing a new threshold on the same questions and relabeling the adjusted result confirmatory.

The conservative paired accuracy interval uses simultaneous 97.5-percent Clopper-Pearson intervals on the improvement and worsening probabilities. A union bound gives at least 95-percent simultaneous coverage under iid task-pair sampling, without assuming the two discordance categories are independent. Subtracting the interval endpoints then bounds their difference. The task-bootstrap saving interval resamples paired questions and recomputes the ratio of total tokens. Because this trap bank is handpicked, its interval is a conditional reference calculation and resampling diagnostic; it does not establish randomized adversarial-population coverage.

## 5.7 Deployable prefix probability model

The second runtime policy uses a pair of fitted, portable probability heads. One estimates correctness of the latest nonempty candidate available in the observed prefix, $\widehat q_t$. The other estimates correctness of the candidate selected by the same rule after one additional increment, $\widehat p_{t+1}$. Direct prediction of the latter avoids dividing the small sample into separate repair and corruption strata. The controller recomputes the estimated one-step gain as

$$\widehat\mu_t=(v+p)(\widehat p_{t+1}-\widehat q_t)-\lambda.$$

The frozen policy uses $v=1$, $p=0$, and $\lambda=0.05$. It stops at the first nonpositive estimated gain after the two-step floor, otherwise at step five. It does not query a next-step predictor at the terminal horizon. This is a myopic fitted rule. No persistence assumption or learned Bellman continuation law is established for these generations.

Training uses all 1,500 archived Qwen2.5-0.5B GSM8K-training and MATH trajectories, containing 7,500 rows. A fixed hash of public task identity allocates 902 tasks to fitting, 322 to calibration, and 276 to evaluation. Standardization and unweighted logistic coefficients are fit only on training tasks. Each one-dimensional Platt logistic calibrator uses only calibration tasks. The evaluation tasks enter neither fit. All rows from a task share its partition, and the live hundred-task and twenty-trap public identities are explicitly excluded. Features use only current and prior observations, including step, token count, answer changes, thought-text summaries, and public domain.

The training targets match runtime answer selection. Reconstructing candidates with the declared parser produces 4,269 candidate-string disagreements with the saved candidates, so the reconstructed candidates are regraded against the archived references using the existing verification rule. Only 43 row-level correctness labels change. This new training-label version and a row-level audit are separate artifacts; the frozen source corpus and the primary empirical chapter's stored labels remain unchanged. No trajectory is filtered because of later candidate agreement. Selecting whole trajectories by agreement of all five responses would condition an early-step evaluation on future observations and change the eligible population.

[[PREDICTOR_TABLE]]

The fitted artifact serializes both scalers, coefficient vectors and calibrators, allowing standard-library probability evaluation without scikit-learn at runtime. Its byte hash is checked by the controller against the frozen policy. The calibrated held-out Brier scores improve slightly over their uncalibrated versions (0.09854 for the current target and 0.10068 for the next target). Those aggregate scores and ten-bin calibration summaries are marginal development diagnostics, rather than certified conditional probabilities for a live prefix.

Transport from the archive remains a material limitation. None of the 7,500 archived outputs satisfies the live strict JSON contract. Consequently, confidence and strict-parsing features have no observed archive variation, while the live JSON prompt creates a different feature distribution. Archived token counts also omit emitted EOS tokens that the live ledger includes. The evaluation must disclose these differences rather than describe task-disjoint archive calibration as live calibration.

On the 276 held-out archived tasks, the learned rule stops 275 times at step two and once at step three. Its accuracy is 11.96 percent, compared with 10.51 percent for the full horizon; the task-bootstrap paired accuracy interval includes zero, approximately [-1.45, +4.35] percentage points. Replayed completion-token saving is 51.29 percent, with an approximate interval [49.31, 53.09] percent. The near agreement with a fixed-two-step policy means this evaluation demonstrates little additional stopping value from the learned heads. It does establish a reproducible, causally computed fitted artifact that can be evaluated prospectively.

## 5.8 Actual trained-policy evaluation

[[LEARNED_TABLE]]

[[LEARNED_ADVERSARIAL_TABLE]]

The trained policy generates its own actual stopped trajectories. It is paired with the already executed full-horizon baseline, whose manifest, model bytes, public prompts, generation adapter, batch order and results are linked by hash. Reusing an actually generated baseline avoids unnecessary repeated GPU work and does not turn the newly executed learned arm into saved-prefix replay. The measured shared-prefix agreement and physical accounting must nevertheless be reported, because earlier baseline execution and later active batching can produce numeric differences.

The predictor artifact and policy are frozen before learned generation. No live or trap outcomes enter its coefficient, calibrator, feature, cost or threshold fitting. The comparison remains a development experiment on one small model and two declared panels. It does not deploy the retrospective stacked hybrid, establish per-instance safety, or prove that the learned rule improves on a cheap fixed-step policy.

The completed trained-policy run stops at step two on all one hundred GSM8K questions. It answers seven correctly, compared with six at the full horizon, and saves 56.51 percent of completion tokens (22,244 versus 9,674). The descriptive saving interval is approximately [54.83, 58.01] percent. The conservative paired accuracy-change interval is approximately [-4.27, +6.21] percentage points, so the observed one-point improvement does not demonstrate superiority or accuracy noninferiority. Every shared prefix matches exactly. Since every task stops at the same step, this realized policy is equivalent to a fixed-two-step budget on the collected panel; the experiment supplies no evidence of useful adaptive stopping beyond that budget.

The trained trap run also stops at step two on all twenty tasks, with one correct answer in both arms. Completion tokens decline from 4,736 to 2,268, saving 52.11 percent, with a descriptive task-resampling interval approximately [45.86, 57.17] percent. All twenty shared prefixes match. The handpicked-bank accuracy interval retains the broad iid-reference bounds of approximately [-19.68, +19.68] percentage points. The large compute reduction accompanies weak absolute answer accuracy and does not establish general adversarial robustness.

Accounting includes repeated prompt processing: the trained main arm uses 43,874 prompt tokens instead of 142,498, and the trap arm uses 8,115 instead of 27,641. Prompt-plus-completion savings are approximately 67.50 and 67.93 percent. Measured main model time is 343.25 seconds rather than 1,127.05, and trap model time is 74.12 rather than 245.14. Padded prefill and decoding slots are separately recorded, with no verifier or peer generations. These are observations on this execution environment; they do not guarantee the same latency reduction under another model or batching regime.

On one hundred actual baseline prefixes repeated twenty times, the trained controller executes 4,000 decisions with median 0.059 ms, p99 0.3143 ms and maximum 0.9369 ms. This benchmark includes feature extraction, both probability heads, calibration, validation and drift computation. It excludes language-model loading, generation, tokenization and peer waits. The measured decision overhead satisfies the under-ten-millisecond target at the observed maximum.

## 5.9 Replay and Pareto analysis

[[PARETO_TABLE]]

A development replay compares the frozen heuristic with fixed-step policies and the full horizon on already saved trajectories. It is useful because all policies can be evaluated against the same recorded continuations at low additional cost. Its counterfactual validity still requires that stopping does not change the preceding generation protocol. It cannot measure live controller overhead, asynchronous batching effects, or energy.

The accuracy-cost plane reports each policy's observed accuracy against its measured or replay-counted completion tokens. A policy is dominated when another policy achieves at least as much accuracy with no greater cost and improves at least one quantity. Nondominated policies form the empirical Pareto set. A point on that set is not automatically preferred: choosing among points requires a declared utility weight or an accuracy constraint.

[[PARETO_FIGURE]]

[[ACTUAL_PARETO_FIGURE]]

Policy variants chosen after examining the same replay outcomes remain development comparisons. The diagram does not certify which point will be nondominated on new tasks. Confidence intervals and a frozen selection rule are necessary for the next external evaluation. A larger computation budget can also move a point by producing different answers, so a comparison should specify whether it changes the maximum response horizon or only the stopping rule.

## 5.10 Online conclusions

The online work establishes a causal runtime interface, direct control over future response scheduling, and reproducible measurements of decision cost and answer-cost trade-offs. The live and replay results have distinct roles. Live events demonstrate actual avoided generation; replay provides broader development comparisons on a shared stored panel; constructed tests verify the event contract.

The evidence supports reporting those results at their actual scope. A fast stopping decision and a positive completion-token saving are useful engineering outcomes. Preserved accuracy, optimality, safe calibration, and transfer to another model are further claims that require their own evidence. Keeping those requirements separate makes the controller and its limitations suitable for critical mathematical and scientific review.
