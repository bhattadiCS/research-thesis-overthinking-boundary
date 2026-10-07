# Chapter 5 Online stopping and computation trade-offs

## 5.1 Causal controller contract

The controller accepts completed responses in step order, storing only their observed prefix: candidate answer, parsing status, completion tokens and available confidence or instrumentation. Its API excludes reference answers, correctness labels, future responses and full-trajectory scores, and rejects skipped, repeated or out-of-order observations.

The policy enforces a two-increment floor, a finite horizon and terminal closure. Decisions record the selected answer and step, termination reason and latency; the manifest binds the policy fingerprint. Validation covers confidence, parsing and future timestamps.

Live manifests preserve the executed source versions for generation and latency. Subsequent validation and accounting revisions reproduce all 480 saved decision histories across the four collections, including reused baseline rows, and leave final-answer labels unchanged. This replay verifies existing observations; it does not replace the original generation-cost or latency provenance.

Optional peer agreement requires a complete same-step roster closed before scoring; incomplete panels, duplicate peers and late votes are rejected. Timestamps verify event order, not an external caller's authenticity. A fleet experiment must retain generation events and charge every peer's computation.

## 5.2 Policy choice and limitations

The first policy is a frozen confidence-and-stability heuristic. Confidence is uncalibrated, and a stable answer can be wrong. Selection uses the latest nonempty answer, with confidence-drop and answer-wobble branches that can retain a prior candidate; neither fired in the live panels. Fixed-step and never-stop controls share the floor and horizon, separating termination's engineering effect from correctness prediction.

The retrospective stacked score is excluded because its future-step information violates this contract (Section 4.6). Stored task-held-out predictions support replay, but deployment requires a verified fitted artifact and causal feature transformation; a new implementation cannot inherit the retrospective AUC.

These heuristics do not implement the general Bellman optimum. Confidence drops and oscillation need not exhaust future repair opportunities or identify incorrect answers. Chapter 2's delayed-repair counterexample shows why immediate evidence against continuation needs additional structural assumptions to justify an optimal stop.

## 5.3 Generation integration

The adapter requests an increment, constructs its permitted observation and evaluates the controller before requesting another. Stopping therefore prevents future model calls; caller cancellation propagates through the generation lifecycle. Baseline and active policies share response-boundary and parsing conventions.

The policy stops between complete reasoning responses, rather than locating an optimal token within a response. Engineering cancellation can interrupt a call, but is distinct from this step-level decision. Completion-token savings measure avoided generation at the declared response granularity.

Separately executed greedy arms are checked for shared-prefix agreement. Batch membership and kernels can change numeric results, so deterministic sampling alone does not ensure matching prefixes. Mismatches limit treating the full arm as an exact counterfactual continuation.

## 5.4 Decision latency

[[LATENCY_TABLE]]

Table 8 reports decision counts, median, upper quantiles and maximum on one hundred saved math-question prefixes. The under-ten-millisecond target concerns the observed maximum, not a fast average. Timing covers prepared controller observations, excluding language-model forward passes and external peers.

State-machine checks cover the floor, horizon, closure, future-step independence, parsing, token accumulation, peer barriers and cancellation. They verify the tested implementation behavior, rather than distributional accuracy preservation.

## 5.5 Live paired evaluation

[[LIVE_TABLE]]

The live experiment freezes tasks, policy and generation settings before opening outcomes. Generation receives public tasks separately from the reference ledger; grading follows event recording, keeping benchmark labels outside runtime decisions.

Paired endpoints include final- and selected-answer accuracy, completion tokens, response increments, prompt tokens and wall time, with auxiliary or peer generation disclosed. Completion-token saving is the ratio of total actual token differences to total baseline tokens; wall time, energy and total inference cost are separate outcomes.

The model is Qwen2.5-0.5B-Instruct [Qwen2024]. Previously observed corpus tasks remain development tasks despite a policy frozen before new generation. Prospective execution, task novelty and cross-model transfer are distinct properties; this run does not validate the Qwen2.5-7B replay or thirteen-model panel as a live system.

Both heuristic arms answer six of one hundred questions correctly. Six tasks stop early and ninety-four reach the horizon. Completion-token saving is 2.94 percent, with descriptive task-bootstrap interval [0.85, 5.63] percent (Table 9). The conservative iid-reference accuracy interval [-4.29, +4.29] percentage points allows meaningful losses; observed equality does not establish noninferiority without a prespecified tolerance.

Only 88 of 500 full-horizon responses satisfy the complete JSON contract. Carrying forward a nonempty candidate cannot recover arbitrary malformed reasoning, and all failures remain in the denominator. The weak six-percent baseline requires evaluation on a larger, protocol-capable model before useful deployed answer quality can be claimed.

Shared prefixes match on 94 of 100 pairs, limiting trajectory-specific counterfactual interpretation. Additional costs are prompt tokens (142,498 versus 137,574), model time (1,127.05 versus 1,099.21 seconds), padded prefill slots (231,856 versus 222,781) and decoding slots (57,856 versus 56,128). Prompt-plus-completion tokens fall 3.39 percent; no verifier or peers are generated. These costs are not interchangeable efficiency percentages.

## 5.6 Adversarial questions and traps

[[ADVERSARIAL_TABLE]]

The twenty-question bank separates public prompts from independently checkable labels and short exact derivations. It covers percentage bases, harmonic average speed, conditional probability, dependent sampling, inclusive endpoints, reciprocal rates, exponent precedence, answer contracts and irrelevant details.

The handpicked bank probes these traps under the frozen model and prompt protocol; classic puzzle forms may occur in pretraining. It does not estimate a general worst-case attack rate. Malformed-input and forged-peer tests separately evaluate software behavior.

Both heuristic arms answer one of twenty questions correctly. Two stop early and eighteen reach the horizon; all twenty shared prefixes match. Strict JSON validity is fourteen of one hundred baseline responses. The 3.34-percent completion-token saving has interval [0.00, 8.80] percent; the conservative iid-reference accuracy interval is [-19.68, +19.68] percentage points (Table 10). This small, low-accuracy panel establishes neither accuracy preservation nor broad adversarial robustness.

A stable high-confidence error challenges the heuristic; a late repair challenges its stopping approximation. These adverse outcomes are retained without tuning a new threshold on the same questions and calling the adjusted result confirmatory.

Paired accuracy intervals subtract simultaneous 97.5-percent Clopper-Pearson bounds for improvement and worsening probabilities. A union bound gives at least 95-percent coverage under iid task-pair sampling without independence of discordance categories (Appendix C). Token intervals resample paired questions and recompute the ratio of total tokens. For the handpicked trap bank these are reference calculations and resampling diagnostics, not randomized adversarial-population coverage.

## 5.7 Deployable prefix probability model

The second runtime policy uses a pair of fitted, portable probability heads. One estimates correctness of the latest nonempty candidate available in the observed prefix, $\widehat q_t$. The other estimates correctness of the candidate selected by the same rule after one additional increment, $\widehat p_{t+1}$. Direct prediction of the latter avoids dividing the small sample into separate repair and corruption strata. The controller recomputes the estimated one-step gain as

$$\widehat\mu_t=(v+p)(\widehat p_{t+1}-\widehat q_t)-\lambda.$$

The frozen policy uses $v=1$, $p=0$, and $\lambda=0.05$. It stops at the first nonpositive estimated gain after the two-step floor, otherwise at step five. It does not query a next-step predictor at the terminal horizon. This is a myopic fitted rule. No persistence assumption or learned Bellman continuation law is established for these generations.

Training uses all 1,500 archived Qwen2.5-0.5B GSM8K-training and MATH trajectories, containing 7,500 rows. A fixed hash of public task identity allocates 902 tasks to fitting, 322 to calibration, and 276 to evaluation. Standardization and unweighted logistic coefficients are fit only on training tasks. Each one-dimensional Platt logistic calibrator [Platt1999] uses only calibration tasks. The evaluation tasks enter neither fit. All rows from a task share its partition, and the live hundred-task and twenty-trap public identities are explicitly excluded. Features use only current and prior observations, including step, token count, answer changes, thought-text summaries, and public domain.

The training targets match runtime answer selection. Reconstructing candidates with the declared parser produces 4,269 candidate-string disagreements with the saved candidates, so the reconstructed candidates are regraded against the archived references using the existing verification rule. Only 43 row-level correctness labels change. This new training-label version and a row-level audit are separate artifacts; the frozen source corpus and the primary empirical chapter's stored labels remain unchanged. No trajectory is filtered because of later candidate agreement. Selecting whole trajectories by agreement of all five responses would condition an early-step evaluation on future observations and change the eligible population.

[[PREDICTOR_TABLE]]

The byte-checked artifact serializes scalers, coefficients and calibrators for standard-library evaluation without scikit-learn. Held-out Brier scores are 0.09760 for the current target and 0.09970 for the next, versus uncalibrated 0.09854 and 0.10068. These scores and ten-bin calibration summaries are marginal development diagnostics, not certified live-prefix conditional probabilities.

None of the 7,500 archived outputs meets the live JSON contract, leaving confidence and strict-parsing features without archive variation. The live prompt shifts that distribution, and its ledger includes emitted EOS tokens omitted by archived counts. Archive calibration therefore does not establish transport to the live protocol.

On 276 held-out archived tasks, the rule stops 275 times at step two and once at step three. Accuracy is 11.96 versus 10.51 percent at the full horizon; the task-bootstrap paired interval [-1.45, +4.35] percentage points includes zero. Replayed completion-token saving is 51.29 percent, interval [49.31, 53.09] percent. This near-fixed budget demonstrates little additional adaptation, but provides a reproducible causal artifact for prospective evaluation.

## 5.8 Actual trained-policy evaluation

[[LEARNED_TABLE]]

[[LEARNED_ADVERSARIAL_TABLE]]

The trained policy generates its own stopped trajectories against the previously executed full-horizon baseline. Hashes bind baseline manifests, model bytes, prompts, adapter, batch order and results. Predictor and policy are frozen before generation; live and trap outcomes enter no coefficient, calibration, feature, cost or threshold fitting. This remains a development study of one small model and two panels, not deployment of the retrospective stack or a per-instance safety guarantee.

On all one hundred GSM8K questions, the trained policy stops at step two with identical shared prefixes. It answers seven correctly versus six at the full horizon and saves 56.51 percent of completion tokens (Table 12; saving interval [54.83, 58.01] percent). The conservative accuracy-change interval [-4.27, +6.21] percentage points establishes neither superiority nor noninferiority. Uniform stopping makes the realized policy equivalent to a fixed-two-step budget on this panel, with no demonstrated adaptive advantage.

All twenty trained trap runs also stop at step two with matching prefixes and one correct answer in each arm. Completion-token saving is 52.11 percent (Table 13; interval [45.86, 57.17] percent), while the handpicked-bank iid-reference accuracy bounds remain [-19.68, +19.68] percentage points. Large savings accompany weak absolute accuracy, without establishing general adversarial robustness.

Prompt-plus-completion savings are 67.50 percent on the main panel and 67.93 percent on traps; Tables 12–13 retain prompt counts. Model times are 343.25 versus 1,127.05 seconds and 74.12 versus 245.14 seconds, respectively. Padded prefill and decoding slots are recorded separately, with no verifier or peer generations. These environment-specific costs do not guarantee equivalent reductions under another model or batching regime.

On one hundred baseline prefixes repeated twenty times, 4,000 trained-controller decisions have median 0.059 ms, p99 0.3143 ms and maximum 0.9369 ms. Timing includes feature extraction, both heads, calibration, validation and drift computation, excluding model loading, generation, tokenization and peer waits. The observed maximum meets the under-ten-millisecond decision target.

## 5.9 Replay and Pareto analysis

[[PARETO_TABLE]]

Replay compares frozen heuristic, fixed-step and full-horizon policies on common saved continuations. Counterfactual validity requires unchanged preceding generation; replay cannot measure live overhead, asynchronous batching effects or energy.

The accuracy-cost plane pairs accuracy with measured or replay-counted completion tokens. A policy is dominated when another has no lower accuracy and no greater cost, improving at least one. Choosing within the empirical Pareto set requires a utility weight or accuracy constraint.

[[PARETO_FIGURE]]

[[ACTUAL_PARETO_FIGURE]]

The displayed set is panel-specific and may change with tasks, generation seeds or response budgets. Independent evaluation must freeze policy selection and distinguish a changed maximum horizon from a changed stopping rule within it.
