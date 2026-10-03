# Defense question bank

These answers describe the frozen research result and its limitations. They do not imply committee approval or degree completion. Numerical answers refer to the final prefix artifact with byte SHA-256 92fe0af86ac0f204d514a6938d0a29dace2b3cdffa24800cc4e2c85a2d51879f. All prospective heuristic and learned main/trap outcomes now have final ledgers.

## Mathematical model and proof

### 1. What is the exact optimization problem?

With finite horizon \(N\), earliest stop \(m\), declared observable filtration \((\mathcal F_t)\), causally selected answer \(A_t\), correctness \(C_t\), and integrable adapted cumulative cost \(K_t\), optimize

\[
\sup_{\tau:\;m\leq\tau\leq N,\;\tau\ {\rm stopping\ time}}
\mathbb E[C_\tau-K_\tau].
\]

For \(q_t=\mathbb E[C_t\mid\mathcal F_t]\), the same value is obtained using observable reward \(G_t=q_t-K_t\). The model assumes the specified potential generation process and answer selector; altering them changes the problem.

### 2. Why may hidden correctness be replaced by its conditional probability at a random stop?

Because admissibility makes \(\{\tau=t\}\) measurable in \(\mathcal F_t\). Therefore
\(\mathbb E[\mathbf1_{\{\tau=t\}}C_t]
=\mathbb E[\mathbf1_{\{\tau=t\}}q_t]\).
Sum over the finite possible stopping times and subtract the common expected cost. This proves equality of objectives for every admissible stop, not equality of realized labels and scores.

### 3. Derive the repair-minus-corruption identity without assuming a Markov chain.

Pointwise,
\[
C_{t+1}-C_t
=\mathbf1_{\{C_t=0,C_{t+1}=1\}}
-\mathbf1_{\{C_t=1,C_{t+1}=0\}}.
\]
Conditional expectation given \(\mathcal F_t\), followed by the tower property for \(q_{t+1}\), gives
\[
\mu_t=(1-q_t)\alpha_t-q_t\beta_t-c_t.
\]
Here \(c_t=\mathbb E[K_{t+1}-K_t\mid\mathcal F_t]\). Neither independence nor a Markov state is required for the identity.

### 4. What does conditioning on an unobserved correctness state mean?

It defines a conditional probability within the joint statistical model, not a runtime query to the gold ledger. Define repair joint mass \(r_t=P(C_t=0,C_{t+1}=1\mid\mathcal F_t)\) and use \(\alpha_t=r_t/(1-q_t)\) when \(q_t<1\). Define corruption similarly. On a zero-probability state, an arbitrary value in \([0,1]\) is harmless because its multiplier is zero. A runtime estimator must infer these quantities from permitted observations or estimate next correctness directly.

### 5. Why not subtract repair hazard and corruption hazard directly?

Their eligible denominators differ. Repair is conditional on current incorrectness and corruption on current correctness. Accuracy drift is their joint-mass difference, namely \((1-q)\alpha-q\beta\). With a common panel, it is also \((\#\mathrm{repairs}-\#\mathrm{corruptions})/n\). A subtraction of the two conditional hazards alone generally has neither interpretation.

### 6. Give the finite-horizon optimal stopping proof.

Define \(S_N=G_N\), \(S_t=\max(G_t,E[S_{t+1}\mid\mathcal F_t])\). Backward induction makes \(S\) an integrable supermartingale dominating \(G\), and shows any other dominating supermartingale is at least \(S\). For every bounded admissible \(\sigma\ge t\), optional stopping gives \(E[G_\sigma\mid\mathcal F_t]\le S_t\). Before first contact, \(S_s=E[S_{s+1}\mid\mathcal F_s]\); after contact the stopped process stays constant. The stopped process is therefore a martingale from time \(t\), and first contact attains \(E[G_{\tau^*}\mid\mathcal F_t]=S_t\). The first contact is a stopping time and exists by the finite terminal horizon. This is the finite-horizon Snell-envelope argument, not a claim that the fitted controller computes \(S\). See [Ferguson's finite-horizon treatment](https://www.math.ucla.edu/~tom/Stopping/sr3.pdf) and [Peskir and Shiryaev](https://doi.org/10.1007/978-3-7643-7390-0).

### 7. Why is negative immediate drift insufficient for optimal stopping?

\[
E[S_{t+1}\mid\mathcal F_t]-G_t
=\mu_t+E[S_{t+1}-G_{t+1}\mid\mathcal F_t].
\]
The second term is nonnegative future option value. A negative first term can be outweighed by future opportunities. The exact example \(q=(0,0,1)\) with cost \(0.1\) per increment gives myopic value zero and optimal value \(0.8\). It is realizable using one common reference answer.

### 8. Under precisely what condition does the first nonpositive drift become optimal?

Under the sufficient pathwise persistent-sign assumption: before the first admissible nonpositive drift, drifts are positive; from that crossing onward, every later conditional drift remains nonpositive along the realized path. The bounded Doob drift decomposition then shows that earlier stopping omits positive terms and later stopping adds nonpositive terms. A declining population average or a negative mean at each time does not establish this assumption.

### 9. Is \(q_t\) necessarily a martingale?

No. The selected answer changes with time, so \(C_t\) is a changing target. The tower property gives \(E[q_{t+1}\mid\mathcal F_t]=E[C_{t+1}\mid\mathcal F_t]\), not \(q_t\). The compensated reward \(G_t-G_m-\sum_{s=m}^{t-1}\mu_s\) is the relevant martingale. A posterior about one fixed latent event is a different object.

### 10. Does a minimum of two responses follow from the theorem?

No. \(m\) is an admissibility constraint. The theorem works for any declared deterministic floor within the horizon. The implementation chooses two complete responses. Its optimizer is optimal within that restricted class, and its utility comparison must use the same floor. Evidence does not establish a universally optimal floor of two.

### 11. What if the controller selects an earlier or a peer answer?

Then \(A_t\) and \(C_t\) must refer to that selected candidate. The default selector carries forward the latest nonempty answer; baseline and learned arms use it. The heuristic can retain the previous answer on a confidence drop or the highest-confidence valid observed answer on wobble, with earliest-step tie-breaking. Neither branch triggered in the reported live panels. The current and next probability targets use the learned/baseline selector. Reusing raw current-row correctness for a carry-forward policy mislabels both its baseline and its drift.

### 12. Are sigma-fields a model of computational inability?

No. They describe statistical information. If gold is a known deterministic measurable function of a fully revealed task and candidate, then correctness is mathematically measurable even without a gold API field. Nontrivial hidden-reference uncertainty needs a latent-reference statistical model or a coarser declared information process such as the permitted decision statistics. All stopping identities remain true in the degenerate case \(q=C\). The deployed model uses domain and prefix statistics, not complete task semantics; the thesis does not provide a theory of arbitrary computational hardness.

### 13. What assumption is needed for a compact Markov state or Bellman approximation?

A state must be sufficient for the conditional distribution of future rewards and costs under the generation protocol, not merely correlated with correctness. Without transition sufficiency, a Bellman recursion on step, confidence and stability can discard relevant history. The finite-horizon theorem works on the full filtration. The new model predicts one-step outcomes from fixed prefix features; it does not assert a Markov state or estimate a validated full continuation value.

### 14. How do higher answer stakes change the optimal stopping time?

For fixed candidates, potential generation law, filtration, floor/horizon and nondecreasing costs, write \(w=v+p>0\). The normalized advantage of an admissible continuation is
\[
E[C_\sigma-C_t-(K_\sigma-K_t)/w\mid\mathcal F_t].
\]
It is nondecreasing in \(w\), since accumulated future cost is nonnegative. Its supremum has the same property. Thus earliest optimal stopping sets shrink as stakes rise, and earliest optimal stop is pathwise nondecreasing. The conclusion fails to follow when changing stakes changes prompts, candidates, costs or the generation law.

## Calibration and statistical inference

### 15. What does an AUC of 0.8743 establish?

It describes ranking discrimination in the stored task-grouped causal-prefix detector evaluation. It does not certify calibration, a stopping threshold, conditional accuracy or live savings. Task-macro AUC is evaluated only where both classes occur. The GPQA worst-domain AUC is about 0.6313, so the pooled score should not be a deployment guarantee.

### 16. Why can class weighting invalidate a probability interpretation?

For population weighted binary log loss, weights \(a,b>0\) give optimum
\(r=aq/(aq+b(1-q))\), not \(q\).
The inverse is \(q=br/[a(1-r)+br]\) under that model. A ranking score can still be useful, but a drift calculation using \(r\) as an uncorrected probability changes the utility scale. The new portable models use no class weights and separate calibration tasks.

### 17. What is actually validated by Brier score and calibration bins?

Average squared probability error and observed outcome frequencies on the held-out panel. They are marginal finite-sample diagnostics. They do not prove that every individual prefix has a correct posterior, nor that a shifted prompt will preserve calibration. Correlated steps within one task do not become independent questions. Evaluation tasks do not fit the base model or calibrator.

### 18. State the valid upper-bound stopping guarantee.

If \(P(\forall t,\ \mu_t\le U_t)\ge1-\delta\), then stopping on \(U_t\le0\) cannot occur on a positive true immediate drift except on the coverage-failure event. This is a sign-error guarantee. It is not Bellman optimality, a zero-accuracy-loss guarantee or evidence that the fitted bounds have the stipulated coverage.

### 19. Why distinguish pre-probe and post-probe information?

After probes are observed, their outcomes are part of the decision information. A conditional-independent Hoeffding statement must be made given the pre-probe sigma-field and a named mean target. A bound on that target does not automatically bound the post-probe conditional drift. The probes can themselves update the candidate correctness posterior. A transport or comparison argument is needed. [Hoeffding's inequality](https://doi.org/10.1080/01621459.1963.10500830) does not supply that modeling step.

### 20. Why is a product called an e-process only under conditional assumptions?

Nonnegative test-martingale or supermartingale constructions require conditional expected multiplicative increments no greater than one under the null. Unconditional favorable means or correlated repeated signals do not suffice. Reusing a shared random sign across increments can break conditional validity after the first sign is observed. Population e-process diagnostics therefore cannot automatically certify an individual live question. See [Howard et al.'s time-uniform framework](https://doi.org/10.1214/20-AOS1991).

### 21. What uncertainty cluster is appropriate here?

Repeated responses and task-temperature trajectories share task information. Accuracy transitions and policy replay use task clusters. The normalized estimator contrasts have a stated 52-cell bootstrap. Detector fold intervals are descriptive summaries of their folds. A bootstrap's resampling unit determines its scope; 10,000 resamples are not 10,000 independent generation seeds or fresh experiments.

### 22. Does zero observed accuracy change prove no harm?

No. An interval and a prespecified acceptable margin are needed for a noninferiority conclusion. The original 100-task live heuristic answers six tasks correctly in both arms, but its reported paired accuracy interval is about \([-4.29,+4.29]\) percentage points. The interval allows a practically meaningful loss. A nonsignificant McNemar result is also not a proof of equivalence.

## Training artifact and information safety

### 23. What exactly was trained?

Two standardized unweighted logistic models: current selected-candidate correctness \(q\), and next selected-candidate correctness \(p_{\mathrm{next}}\). Each has an independent one-dimensional Platt calibrator. Current labels use all five observed prefixes; next labels use only prefixes one through four. The policy is first \(\hat p_{\mathrm{next}}-\hat q-0.05\le0\) after the floor, otherwise the terminal horizon.

### 24. How are fitting, calibration and evaluation separated?

A salted hash of domain and task ID assigns 60/20/20 roles before parsing, availability or labels. Final counts are 902 training, 322 calibration and 276 evaluation tasks. A task's repeated runs remain in one role. Scalers and base coefficients fit only training tasks. Calibrators fit only calibration tasks. Evaluation metrics and paired replay use only evaluation tasks. There is no evaluation-selected cost or threshold.

### 25. Which training features can actually exist at runtime?

The 21-feature contract uses current/prior step, strict parse status, answer presence, observed confidence when valid, prefix confidence summaries, completion length, cumulative length, answer changes and streak, answer/thought characters, fixed thought numeric/keyword densities, public domain and predetermined interactions. It consumes only the Observation sequence and domain. It does not accept a gold field, labeled table, hidden-state feature, future response or hindsight score.

### 26. Why were candidates regraded, and did this contaminate live evaluation?

The archived four-line parser and live JSON/fallback parser reconstruct different candidate strings. Reusing saved correctness for a different answer would create an invalid target. The offline trainer regrades reconstructed candidates against archived gold under the existing verification rule, preserving the original corpus and a 7,500-row audit. There are 4,269 candidate disagreements and 43 correctness changes. This reads archive training labels only. Live and trap gold are excluded, and the model was frozen before their learned runs.

### 27. Why preserve the provisional artifact?

The first reconstruction attempt retained only trajectories whose five candidates agreed with saved candidates. It selected 355 of 1,500 trajectories partly on later parser outcomes. That eligibility rule was rejected before learned live generation. The provisional directory preserves the audit trail; its metrics are superseded. The final model retains all complete trajectories and unchanged features, task-role rule, hyperparameters, calibration, cost and threshold.

### 28. What do the final held-out predictor metrics show?

Current-correctness AUC is 0.710094, calibrated Brier 0.097599 and ten-bin ECE 0.022227 over 1,380 prefix rows. Next-correctness AUC is 0.692308, Brier 0.099697 and ECE 0.020928 over 1,104 rows. These describe the 276-task archived development holdout. They do not imply pointwise confidence coverage or live calibration.

### 29. Does the learned stopper outperform a fixed-two baseline?

No material added policy value is demonstrated. It stops 275 of 276 archive holdout tasks at two and one at three. Its accuracy equals fixed two on that panel and its cost is slightly higher. Against full horizon, the observed accuracy change is +1.45 points with task-cluster interval \([-1.45,+4.35]\), and archived completion savings are 51.29%. These comparisons are development replay; prospective learned measurements are separate.

### 30. What is the largest transport limitation?

Zero archived rows satisfy the live strict JSON contract. Confidence-related predictors are therefore constant/untrained in the archive, and live JSON wording changes thought and answer distributions. The live generator also charges emitted EOS tokens whereas archived completion counts omit them. Matching model identity alone does not remove a prompt, parser or measurement shift. Final live outcomes assess one transported policy without tuning it on their labels.

### 31. How is the deployment artifact verified?

The runtime is standard-library only. It validates schema, feature order, dimensions, finite coefficients and calibrators, positive scales, supported domains and the two-to-five horizon. Its loader computes the SHA-256 of the exact JSON bytes. Tests verify independent vector probability calculations, all task-role separations, public live/trap exclusions, row-level reconstruction totals, candidate carry-forward, earlier-feature invariance under later-row changes, terminal guards and the preserved training freeze.

## Empirical scope and runtime

### 32. Why are there two different corpus counts?

The revision matrix supports longer domain-dependent response trajectories and has 75,965 analyzed trajectories. The standardized five-step detector corpus has 28,888 trajectories and 144,440 rows. They overlap and use somewhat different model selection, so counts must not be added. Tables state their own denominator. The selected recorded domains are GSM8K, MATH, ARC-Challenge and GPQA.

### 33. What is wrong with using 0.955 as the live accuracy claim?

The stored stacked AUC is a retrospective development score. Its bidirectional component and centered smoothing use future steps, and meta-training is not nested inside the scoring outer folds. It is not accuracy, not a live confidence guarantee, and not consumed by the runtime. The 0.8743 causal archive score comes from a different evaluation; their difference cannot isolate a causal effect of removing future information.

### 34. Do the pooled curves prove accuracy peaks at step two or three?

No universal peak is established. Positive accuracy improvement can coexist with negative utility drift because of cost. The selected 7B/GSM cell changes net-drift sign between prefixes four and five, and 32B/MATH between five and six. A population crossing is descriptive and is not automatically a task-specific optimal stopping boundary.

### 35. How do you interpret negative estimator comparisons?

They show lower recorded stopping utility for those matched experimental configurations. Gradient boosting and isotonic calibration have negative controlled contrasts against the matching logistic baselines. They do not prove that all nonlinear models overfit or calibration is intrinsically harmful. The source includes the exact target, comparator, resampling unit and denominator.

### 36. Is the precision contrast a universal scaling law?

No. It is BF16 versus four-bit weights for Qwen2.5-7B/GSM8K at step two on a paired task-temperature panel. Correct counts are 418/1,500 versus 204/1,500, an absolute 14.27-point accuracy difference, with task interval approximately [11.13,17.53] points. It should not become a relative percentage or a claim about all models.

### 37. What does the 256-versus-512 cap null result establish?

The recorded Mistral-Small-22B/GSM8K binary policy-loss indicator has 454 losses in each arm and no discordant pairs. It establishes zero observed difference for that endpoint. It does not establish equal answers, token counts, time, or absence of truncation in every cell.

### 38. Is stopping an actual interruption or just a replay label?

Actual runtime stopping removes a task from future response scheduling. No later model call is requested for it. The stopping rule operates between completed response increments; within-call abort/cancellation is separate. Replay only counts what a frozen policy would have emitted on recorded continuations and does not measure actual saved GPU execution.

### 39. Which costs are counted?

Actual emitted completion tokens including EOS, repeated prompt tokens, padded prefill slots, decode slots and model seconds have separate ledgers. Peer and verifier generation tokens are zero for the current default policy. Completion savings are a ratio of total generated counts; they are not automatically equal to energy, time or total inference savings.

### 40. What does the sub-ten-millisecond result cover?

Prepared-observation controller computation: input validation, prefix update, feature and answer selection, decision and closure. The final 18,880-decision saved-MATH benchmark has maximum observed 0.4383 ms. The separate final learned-main benchmark has 4,000 decisions over 100 problems and 20 repeats, with maximum 0.9369 ms, including both calibrated heads. Both exclude loading the model, forward generation, tokenization and peer waits. They are worst observed values on declared workloads, not worst-case execution-time proofs.

### 41. Why check shared prefixes even under greedy generation?

Different batches or kernels can change floating-point execution and candidate outputs. Greedy decoding alone does not prove equality. The original main run has identical shared prefixes on 94/100 tasks. Differences limit treating the full arm as the exact continuation of the active arm. Both outcomes and the measured agreement count remain in the report.

### 42. What did the original live heuristic actually achieve?

On 100 GSM8K-test tasks, it generated 21,591 completion tokens versus 22,244 under never stopping: 2.94% savings. Accuracy is 6/100 in both arms and mean stop is 4.85. Only 88/500 baseline responses are strict-valid JSON. This is a modest engineering saving with low absolute task accuracy, far below a 30–40% savings target. It is not the learned model's result.

The separately generated frozen learned arm emits 9,674 tokens (56.51% saving, task-bootstrap interval [54.83%,58.01%]), with 7/100 correct against 6/100. Its conservative paired accuracy-change interval is [-4.27,+6.21] points. All 100 tasks stop at two and shared prefixes all agree. This is actual avoided generation, but it does not establish superiority over fixed two or accuracy noninferiority.

### 43. What do twenty adversarial questions establish?

Performance on the frozen bank under the declared model and prompt. They test known traps with independently written short derivations, not an exhaustive worst-case rate. Exact project records are new, but classic forms may be in model pretraining. The original heuristic has 1/20 correct in both arms and about 3.34% completion savings. Those results do not justify retuning on the same outcomes and calling the new result confirmatory.

The frozen learned arm saves 52.11% of actual completion tokens and remains 1/20 correct, stopping every task at two. Its iid-reference accuracy interval is [-19.68,+19.68] points; the handpicked bank does not support randomized adversarial-population coverage.

### 44. Can a data hash certify the labels or scientific validity?

No. A hash binds an artifact's identity under its specified byte-normalization rule. It does not show that the initial run was honest, the grader correct on every answer, or the sample representative. Grader regression tests establish covered cases. Historical labels and new reconstructed labels have distinct versions and audits; the exact training freeze is preserved.

### 45. What remains for a stronger empirical conclusion?

An unseen task panel, a frozen model/prompt/policy, prespecified utility and accuracy margin, multiple generation seeds, task-cluster uncertainty, subgroup diagnostics and complete computation accounting. A positive saving with a wide accuracy interval is not confirmation of harmless stopping. Training a Bellman continuation model is a separate research extension.

### 46. What remains for a completed Master's degree?

Actual committee review, approval, defense, signatures, institution-specific submission and clearance. A complete editable presentation, thesis manuscript and reproducibility package are research deliverables. They cannot create those external institutional events.

## Primary proof references

- [Ferguson, Optimal Stopping and Applications, finite horizon](https://www.math.ucla.edu/~tom/Stopping/sr3.pdf).
- [Peskir and Shiryaev, Optimal Stopping and Free-Boundary Problems, 2006](https://doi.org/10.1007/978-3-7643-7390-0).
- [Hoeffding, Probability inequalities for sums of bounded random variables, 1963](https://doi.org/10.1080/01621459.1963.10500830).
- [Howard et al., Time-uniform, nonparametric, nonasymptotic confidence sequences, 2021](https://doi.org/10.1214/20-AOS1991).

Local authoritative derivations and empirical sources are research/mathematical_foundations.md, the six thesis chapters, research/outputs/thesis_v1/evidence/, and the frozen online/prefix-model ledgers named in the slide notes.
