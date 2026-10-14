# Chapter 6 Discussion and limitations

## 6.1 Scientific interpretation

Continuation can repair an answer, corrupt it or consume computation without sufficient improvement. The binary-transition decomposition measures these changes, while optimal stopping values future opportunities beyond the next increment. A population curve mixes difficulty, models and sampling outcomes; its crossing alone need not identify the best action for every prefix.

Runtime allocation therefore needs a conditional predictor, a continuation model or an acknowledged heuristic evaluated using permitted information. A fixed-step policy can save tokens without recognizing correctness; a richer detector must justify its additional scoring or peer cost through measured improvement in the final objective.

## 6.2 Correctness labels and mathematical validity

Grader-defined correctness measures reference agreement rather than mathematical understanding. Wording can defeat numeric extraction, domain restrictions can affect symbolic equivalence, and a correct MCQ option can accompany a wrong rationale. These errors affect training targets and repair/corruption estimates.

Regression tests cover specified parser behaviors. Broader validity requires stratified human adjudication or an independent trusted verifier. Symbolic normalization can merge expressions with different domain restrictions or miss equivalent ones; an audit should distinguish parsing failures, reference issues and reasoning errors.

The tournament retains archived labels; Section 5.7 separately versions reconstructed predictor targets and regrading. The post-review grader audit preserves final-answer labels in all four live panels and the original corpus summaries. A changed label definition changes the estimand and requires a row-level audit and recomputation of dependent results.

## 6.3 Theory and deployable beliefs

Correctness, repair and corruption probabilities are conditional expectations under the true data-generating distribution. Interpreting a classifier as their estimate requires assumptions about inputs, targets, sampling and calibration. Class-balanced loss can support ranking while targeting a reweighted distribution instead of the deployment posterior.

Chapter 2 characterizes optimal decisions when the relevant conditional expectations are known; it does not certify fitted estimates. Finite-system checks verify specified algebra and dynamic programs, without establishing the true continuation law of an unseen deployment.

Question-level confidence intervals describe population comparisons under their sampling or resampling assumptions. A confidence sequence along one reasoning trajectory instead requires a valid sequential concentration or martingale construction for that process. The reported intervals provide no per-instance error guarantee.

## 6.4 Generalization and adaptation

One generation seed per setting leaves seed variability largely unmeasured in the fixed model/domain panel. Task-held-out detector folds cannot rule out benchmark exposure during language-model pretraining. Revision prompts, temperatures, response limits and grading conventions also define the population to which these results apply.

Repeated configuration comparisons can turn an outer task holdout into a development set. Confirmation requires a frozen policy and feature contract, an untouched evaluation panel, and recorded deviations from the prespecified experiment.

Strict task-grouped baselines, matched peer ablations, fixed-budget controls and grader audits support the qualified retrospective and protocol-specific findings. Fresh collection is needed for claims requiring independent prospective validation, including calibrated peer-fleet benefit. Independent training seeds or resampling do not constitute independently generated response trajectories.

GPQA's weak causal sequence results expose domain variation masked by pooled scores. New benchmarks, model families or adversarial prompts can change confidence and agreement signals, requiring monitoring and reevaluation beyond the four-domain panel.

## 6.5 Costs beyond completion tokens

Avoided completion tokens measure one part of inference cost. Prompt processing, verifier passes, peer generation, batching, synchronization, network delay and controller execution also contribute. Wall-time and energy claims require direct instrumentation.

Thirteen peers can make a shorter target trajectory more costly than a single-model baseline. Their outputs become available only after generation, whose cost must be charged. Synchronization and asynchronous scheduling further change the policy, latency and cost.

Controller timings cover available observations, including learned feature extraction, both heads, calibration and drift. They exclude loading, generation, tokenization, peer waits and network communication. Meeting the under-ten-millisecond decision target does not establish unchanged total response time.

## 6.6 Adversarial behavior and answer selection

Shared misconceptions can create stable, confident agreement. Mandatory continuation can replace a correct first answer, while an apparently unpromising trajectory can later repair. These cases challenge different heuristic assumptions.

Constructed-prefix tests check software contracts; the twenty prespecified traps use independently checked keys and measured generation costs to test model responses. This complementary evidence does not make the small handpicked bank representative of broader adversarial prompts.

Baseline and learned policies select the latest nonempty candidate. The heuristic's retention branches (Section 5.2) did not fire in the live panels. Retaining an earlier answer is a distinct intervention that can recover a candidate lost during mandatory continuation; its selector must use permitted prefix information. A richer selector needs a separate comparison, while reference-guided selection is an oracle.

## 6.7 Evidence needed for stronger claims

Accuracy-preserving savings require a prespecified noninferiority margin, justified sample size and paired evaluation of a frozen policy against the full horizon. The confidence interval must exclude losses beyond that margin; observed equality is insufficient. Independent tasks and generation seeds strengthen transfer and reproducibility evidence.

An optimal learned boundary needs a supported state representation and transition law. A scalar correctness estimate need not summarize future revision dynamics; predictive sufficiency requires showing that omitted prefix information does not materially improve continuation prediction under the target distribution. The full-history formulation avoids assuming scalar sufficiency.

Historical reproduction requires original features, splits, fits, software and stochastic controls. The freeze and current environment lock identify available artifacts, without reconstructing missing training details. Reanalysis under a fully captured new environment is a separately versioned experiment.

## 6.8 Conclusion

Useful continuation depends on the model, domain and assigned cost. The controlled comparisons show that better ranking or added calibration need not improve stopping utility. General optimal stopping requires conditional continuation values; myopic drift rules need additional structure and fail in the counterexamples. The live prototype prevents further generation, but every learned live task stops at step two: its savings demonstrate a reduced fixed budget, without demonstrated adaptive advantage. Weak absolute accuracy and broad paired intervals leave accuracy noninferiority unestablished. Broader claims require independent validation and full accounting for prompt processing, peers, verifiers and controller cost.
