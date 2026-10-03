# Defense slide source

Exactly 25 slides. Allocated speaking time: 30 minutes.

## Slide 1: Cost aware stopping boundaries in reasoning language models

Master's thesis defense • Applied and Computational Mathematics

- Aditya Bhatt • Johns Hopkins University
- Research adviser: Dr. Zerotti Woods
- Second reader: Dr. Moustapha Pemy
- Research evidence frozen October 2026

Sources: ThesisDocs/Masters_Thesis_Draft_v1.md; ThesisDocs/ADVISOR_MEETING_SEMESTER_2_ROADMAP.md

## Slide 2: When is another reasoning increment worth its cost?

Research question

- Repair: an incorrect candidate becomes correct.
- Corruption: a correct candidate becomes incorrect.
- Unproductive continuation: benefit falls below declared cost.
- A causal stop uses the observed prefix and a declared answer selector.

Sources: ThesisDocs/chapters/chapter1_intro.md; research/mathematical_foundations.md

## Slide 3: Two corpora answer different questions

Experimental scope

- The corpora overlap; their counts must not be added.
- GSM8K, MATH, ARC-Challenge and GPQA are the recorded benchmarks.

| Quantity | Revision matrix | Detector corpus |
|---|---|---|
| Model-domain cells | 52 | 52 |
| Model configurations | 13 | 13 (different selection) |
| Trajectories | 75,965 | 28,888 |
| Step rows | 798,770 raw | 144,440 |
| Unique tasks | 1,948 | 2,948 |
| Horizon | 8 / 10 / 14 by domain | 5 in every cell |

Visible evidence qualification: Overlapping corpora; counts are not additive.

Sources: data_manifest_v1.json; ThesisDocs/chapters/chapter3_methodology.md

## Slide 4: The observation unit is a complete revision response

Generation protocol

- One increment produces a thought, candidate answer and telemetry.
- The next prompt contains the previously observed response history.
- The floor is two increments; it is a protocol constraint.
- Historical temperatures are repeated settings of one recorded seed.

| Setting | Declared value |
|---|---|
| Historical seed / temperatures | 7 / 0.1, 0.6, 1.0 |
| Live generator | Local Qwen2.5-0.5B, greedy |
| Live completion cap | 128 tokens per response |
| Live floor / horizon | 2 / 5 responses |

Visible evidence qualification: The floor and horizon are declared protocol constraints.

Sources: ThesisDocs/chapters/chapter3_methodology.md; research/online_generation.py; research/online_stopping_controller.py

## Slide 5: Correctness is hidden from the stopping policy

Filtration and admissibility

- Fₜ contains declared decision information and completed prefix observations.
- Cₜ is the selected candidate's correctness against an unobserved reference.
- qₜ = E[Cₜ | Fₜ] is a conditional probability, not a known label.
- τ is admissible when {τ ≤ t} belongs to Fₜ; m ≤ τ ≤ N.

| Runtime permits | Evaluation keeps separate |
|---|---|
| Current / prior answer and thought | Reference answer and correctness |
| Strict parsing and observed confidence | Future response and future label |
| Charged tokens and public domain | Bidirectional full-trajectory score |

Visible evidence qualification: The runtime interface excludes gold and future responses; statistical measurability has separate assumptions.

Sources: research/mathematical_foundations.md; research/online_stopping_controller.py

## Slide 6: Accuracy, utility and computation are separate outcomes

Objective and accounting

- Gₜ = qₜ − Kₜ; maximize E[Gτ] over admissible stops.
- A response-cost example uses Kₜ = 0.05(t − 1).
- Affine stakes give Gₜ = (v + p)qₜ − p − Kₜ.
- Completion tokens, repeated prompts, peers and wall time need separate ledgers.

Sources: research/mathematical_foundations.md; ThesisDocs/chapters/chapter5_online.md

## Slide 7: Conditional repair and corruption yield the exact drift

Binary correctness identity

- αₜ = P(Cₜ₊₁ = 1 | Cₜ = 0, Fₜ).
- βₜ = P(Cₜ₊₁ = 0 | Cₜ = 1, Fₜ).
- μₜ = (1 − qₜ)αₜ − qₜβₜ − cₜ.
- cₜ = E[Kₜ₊₁ − Kₜ | Fₜ]; zero denominators have zero mass.

E[Gₜ₊₁ − Gₜ | Fₜ] = (1 − qₜ)αₜ − qₜβₜ − cₜ

Sources: research/mathematical_foundations.md

## Slide 8: Bellman continuation value defines the optimal stop

Finite-horizon Snell envelope

- S_N = G_N; Sₜ = max{Gₜ, E[Sₜ₊₁ | Fₜ]}.
- τ* = first t ≥ m with Sₜ = Gₜ.
- Δ*ₜ = μₜ + E[Sₜ₊₁ − Gₜ₊₁ | Fₜ] ≥ μₜ.
- Nonpositive immediate drift alone does not imply an optimal stop.

Sₜ = max(Gₜ, E[Sₜ₊₁ | Fₜ])

Sources: research/mathematical_foundations.md; research/tests/test_mathematical_foundations.py

## Slide 9: A first drift crossing is valid under persistent signs

Sufficient condition, not a universal law

- τμ = first admissible time with μₜ ≤ 0, otherwise N.
- If all later conditional drifts stay nonpositive pathwise, τμ is optimal.
- Proof: telescope the stopped drift sum before and after the crossing.
- A declining population mean is weaker than this assumption.

E[Gσ − Gτ] = Σₛ E[(1{σ>s} − 1{τ>s}) μₛ]

Sources: research/mathematical_foundations.md; research/tests/test_mathematical_foundations.py

## Slide 10: Delayed repair defeats a myopic stopping rule

Exact counterexample

- q₀ = 0, q₁ = 0, q₂ = 1; each continuation costs 0.10.
- Immediate drift at t = 0 is −0.10.
- Stopping now earns 0; continuing to t = 2 earns 0.80.
- The missed future repair is a continuation option.

| Time | Correctness q | Cumulative cost | Stop reward |
|---|---|---|---|
| 0 | 0 | 0.00 | 0.00 |
| 1 | 0 | 0.10 | −0.10 |
| 2 | 1 | 0.20 | 0.80 |

Visible evidence qualification: Immediate drift is negative at zero, yet optimal continuation value is 0.80.

Sources: research/mathematical_foundations.md; research/tests/test_mathematical_foundations.py

## Slide 11: A ranking score is not a calibrated probability

Prediction assumptions

- AUC measures ordering, not agreement with outcome frequencies.
- Class weighting changes the fitted posterior unless corrected.
- A small set of prefix features need not be a sufficient state.
- Independent task calibration supports marginal checks, not certainty at every prefix.

Sources: research/mathematical_foundations.md; research/prefix_stopping_model.py; research/outputs/semester2/prefix_model_v1/evaluation.json

## Slide 12: Repeated checking needs a valid sequential target

Uncertainty guarantees

- A simultaneous upper bound can control false negative-drift stops.
- Its coverage must hold over all decision times for the actual target.
- Conditional probe independence and pre-probe information must be stated.
- Population diagnostics do not certify one task's unseen continuation.

Sources: research/mathematical_foundations.md

## Slide 13: The freeze makes claims traceable

Reproducibility and provenance

- Exact bytes and canonical LF content have separate hashes.
- Corpus identity, exclusions, partitions and analysis units are explicit.
- Generated proofs are verified by exact finite-system enumeration.
- A hash records identity; it does not certify truth or representativeness.

Sources: data_manifest_v1.json; research/tests/test_mathematical_foundations.py; research/tests/test_prefix_stopping_model.py; research/outputs/semester2/prefix_model_v1/training_protocol.json

## Slide 14: Repair and corruption use different denominators

Pooled GSM8K transitions

- 19,500 trajectories over 500 task clusters at each shown transition.
- Net drift subtracts the declared 0.05 response cost.
- Negative utility drift can coexist with increasing accuracy.

| Prefix t | Repair / eligible | Corruption / eligible | Net utility drift |
|---|---|---|---|
| 2 | 3145 / 14818 | 1170 / 4682 | +0.0513 |
| 4 | 1830 / 11661 | 1102 / 7839 | -0.0127 |

Visible evidence qualification: This population utility drift can turn negative while average accuracy still increases.

Sources: research/outputs/thesis_v1/evidence/boundary_domain_step_metrics.csv

## Slide 15: The observed crossing depends on the cell

Selected model-domain contrasts

- 7B/GSM: crossing 4 → 5.
- 32B/MATH: crossing 5 → 6.
- 500 tasks per cell; three temperatures.

Editable chart data: {"kind": "column", "categories": ["7B/GSM, t=4", "7B/GSM, t=5", "32B/MATH, t=5", "32B/MATH, t=6"], "series": [{"name": "Net drift", "values": [0.019333333333333324, -0.025333333333333333, 0.018666666666666658, -0.013333333333333338]}], "x_title": "Model / domain and completed prefix", "y_axis_title": "Accuracy change − 0.05 cost", "y_axis_min": -0.04, "y_axis_max": 0.04}

Sources: research/outputs/thesis_v1/evidence/boundary_selected_cell_step_metrics.csv

## Slide 16: More expressive estimators did not always help

Matched development comparisons

- Effects are utility differences per trajectory relative to matched controls.
- Intervals resample 52 model-domain cells.
- These analyses do not establish a universally best estimator.

| Matched modification | Utility / trajectory | 95% cell interval |
|---|---|---|
| Empirical-Bayes hazards | +0.00781 | [+0.00276, +0.01355] |
| Lagged logistic | +0.00331 | [+0.00062, +0.00633] |
| Gradient-boosted probe | -0.05694 | [-0.08716, -0.03145] |
| Isotonic calibration | -0.06165 | [-0.09168, -0.03547] |

Visible evidence qualification: Matched utility contrasts, with 52-cell descriptive bootstrap intervals.

Sources: research/outputs/thesis_v1/evidence/algorithm_v2_normalized_effects.csv

## Slide 17: A paired precision contrast has a large local effect

Qwen2.5-7B / GSM8K / step two

- BF16: 418 correct out of 1,500; 4-bit: 204 out of 1,500.
- Absolute accuracy difference: 14.27 percentage points.
- Task-cluster interval: [11.13, 17.53] points.

Editable chart data: {"kind": "column", "categories": ["BF16", "4-bit"], "series": [{"name": "Step-two accuracy", "values": [0.2786666666666667, 0.136]}], "x_title": "Weight precision", "y_axis_title": "Correct / 1,500 trajectories", "y_axis_min": 0, "y_axis_max": 0.35}

Sources: research/outputs/thesis_v1/evidence/algorithm_v2_normalized_effects.csv; ThesisDocs/chapters/chapter4_empirical.md

## Slide 18: Pooled causal discrimination hides a weak domain

Five-step task-grouped detector evaluation

- GRU micro AUC: 0.8743.
- Worst domain GPQA: 0.6313.
- AUC and policy utility rank differently.

Editable chart data: {"kind": "column", "categories": ["Micro", "Task macro", "Domain macro", "Worst domain"], "series": [{"name": "Causal GRU AUC", "values": [0.8743264229443097, 0.8213923134886576, 0.8102415230146596, 0.6312532126006337]}], "x_title": "Aggregation rule", "y_axis_title": "ROC AUC", "y_axis_min": 0.5, "y_axis_max": 1.0}

Sources: research/outputs/thesis_v1/evidence/tournament_balanced_summary.csv; research/outputs/thesis_v1/evidence/tournament_per_domain.csv

## Slide 19: The retrospective 0.955 score is not a live guarantee

Information and evaluation audit

- Stacked development AUC: 0.955156; reduced-feature control: 0.943223.
- Bidirectional scoring and centered smoothing use future responses.
- Meta-training is not nested within the scoring outer fold.
- The 0.8743 causal result comes from a different evaluation.

| Evidence | Permitted claim |
|---|---|
| Stacked AUC 0.955156 | Retrospective development diagnostic |
| Causal GRU AUC 0.874326 | Task-grouped archive prefix ranking |
| Portable prefix model | Separate train / calibration / holdout |
| Actual online run | Measured generated-cost / accuracy trade-off |

Visible evidence qualification: Future-step inputs and non-nested meta-fitting preclude a live 0.955 claim.

Sources: ThesisDocs/chapters/chapter4_empirical.md; ThesisDocs/chapters/chapter6_discussion.md

## Slide 20: The controller prevents future response calls

Runtime implementation and measured latency

- It owns the prefix, rejects future/out-of-order inputs, and closes permanently on stop.
- Strict JSON telemetry authorizes confidence; fallbacks carry confidence=None.
- Peers require a complete same-step barrier and fully charged computation.
- Saved-MATH benchmark: 18,880 decisions; worst observed 0.4383 ms.

| Prepared-observation decisions | Latency |
|---|---|
| Median (saved MATH) | 0.0034 ms |
| 95th percentile (saved MATH) | 0.0079 ms |
| 99th percentile (saved MATH) | 0.0130 ms |
| Maximum (saved MATH) | 0.4383 ms |
| Learned maximum (4,000 main decisions) | 0.9369 ms |

Visible evidence qualification: Controller timing excludes LM generation/loading, tokenization and peer waits.

Sources: research/online_stopping_controller.py; research/online_generation.py; research/outputs/semester2/online_stopping_20261002/latency_summary.json; research/outputs/semester2/online_stopping_20261002/learned_main/learned_latency_summary.json

## Slide 21: A portable trained model estimates current and next correctness

Fixed task-disjoint probability fitting

- 1,500 archived tasks: 902 train, 322 calibration, 276 evaluation.
- Two unweighted logistic models; train-only scaling and separate Platt calibration.
- 21 features use only current/prior Observation fields plus public domain.
- No live GSM8K-test or trap outcomes select the model or threshold.

| Held-out target | Rows | AUC | Calibrated Brier |
|---|---|---|---|
| Selected current correctness | 1380 | 0.7101 | 0.0976 |
| Selected next correctness | 1104 | 0.6923 | 0.0997 |

Visible evidence qualification: Separate task roles; archive calibration does not establish live calibration (0 / 7,500 strict-valid JSON).

Sources: research/prefix_stopping_model.py; research/train_prefix_stopping_model.py; research/outputs/semester2/prefix_model_v1/training_protocol.json; research/outputs/semester2/prefix_model_v1/label_reconstruction_audit.csv

## Slide 22: The trained drift policy nearly reduces to fixed two

Held-out archive replay and transport limits

- Stop when p̂next − q̂current − 0.05 ≤ 0, with floor 2 / horizon 5.
- 275 of 276 held-out tasks stop at two; one stops at three.
- No material policy advantage over fixed-two is demonstrated.
- Archive strict JSON support is zero: live prompt transport is a major limitation.

| Archive holdout policy | Accuracy | Completion saving | Mean stop |
|---|---|---|---|
| Full horizon | 10.51% | 0.00% | 5.000 |
| Fixed two | 11.96% | 51.33% | 2.000 |
| Learned drift | 11.96% | 51.29% | 2.004 |

Visible evidence qualification: Learned drift nearly equals fixed two; these archived token counts are not measured live savings.

Sources: research/outputs/semester2/prefix_model_v1/evaluation.json; research/outputs/semester2/prefix_model_v1/heldout_policy_replay.csv; research/outputs/thesis_v1/evidence/offline_replay_metrics.csv

## Slide 23: Actual generation yields different policy trade-offs

100-task GSM8K test development panel

- Learned completion saving: 56.51%; interval [54.83%, 58.01%].
- Learned accuracy: 7 / 100 versus 6 / 100; change interval [-4.27, +6.21] points.
- All 100 learned tasks stop at two; no registered noninferiority margin.
- Baseline strict JSON: 88 / 500; learned shared prefixes: 100 / 100.

| Actual outcome | Never | Heuristic | Learned |
|---|---|---|---|
| Selected-answer accuracy | 6.00% | 6.00% | 7.00% |
| Generated completion tokens | 22244 | 21591 | 9674 |
| Completion-token saving | 0% | 2.94% | 56.51% |
| Mean stopping step | 5.00 | 4.85 | 2.00 |
| Shared-prefix identical tasks | Reference | 94 / 100 | 100 / 100 |

Visible evidence qualification: All learned tasks stop at two; added value over fixed two and accuracy noninferiority are not established.

Sources: research/outputs/semester2/online_stopping_20261002/live_metrics.json; research/outputs/semester2/online_stopping_20261002/live_paired_results.csv; research/outputs/semester2/online_stopping_20261002/learned_main/live_metrics.json; research/outputs/semester2/online_stopping_20261002/learned_main/live_uncertainty.json

## Slide 24: Trap questions probe the policy's failure modes

Adversarial bank and limits

- 20 mathematical prompts with separately derived gold answers.
- Changing bases, dependent sampling, rates, precedence and answer contracts.
- Wrong stable answers and delayed repair can defeat a working controller.
- One small model, one frozen prompt and a development panel limit generalization.

| Trap policy | Accuracy | Completion saving | Paired changes |
|---|---|---|---|
| Heuristic | 1 / 20 | 3.34% | 0 worsened / 0 improved |
| Learned | 1 / 20 | 52.11% | 0 worsened / 0 improved |

Visible evidence qualification: Handpicked bank: descriptive robustness only; all learned tasks stop at two.

Sources: research/adversarial_tasks_v1.jsonl; research/adversarial_gold_v1.jsonl; research/outputs/semester2/online_stopping_20261002/adversarial_live/live_metrics.json; research/outputs/semester2/online_stopping_20261002/learned_adversarial/live_metrics.json; research/outputs/semester2/online_stopping_20261002/learned_adversarial/live_uncertainty.json

## Slide 25: The contribution is a testable stopping framework

Conclusion and next confirmatory study

- Exact conditional drift and finite-horizon optimal-stopping proofs.
- Frozen empirical contrasts retain subgroup and negative results.
- A causal runtime and serialized task-disjoint trained predictor.
- Next: unseen tasks, prespecified accuracy margin, multiple seeds and full costs.

Sources: research/mathematical_foundations.md; ThesisDocs/chapters/chapter6_discussion.md; ThesisDocs/requirements_acceptance_matrix.md

