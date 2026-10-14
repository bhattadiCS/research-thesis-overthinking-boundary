# Agent A independent v7 core mathematics review

Initial review recorded 2026-10-13, before receiving another reviewer's mathematical verdicts. Reviewer: Agent A. Read `PROTOCOL.md` and `baseline_hashes.json` first. Read repository `CLAUDE.md`, `README.md`, `CURRENT_FORMAL_THESIS.md`, all five concise v7 chapter sources, v7 builder, compiled abstract and relevant compiled line mappings. The user/current-source hierarchy controls over historical canonical-math language. Raw-byte hashes of all five sources, compiled v7 and its builder were independently checked and agree with the protocol baseline.

The initial deductive assessment finds **no demonstrated mathematical error in concise v7 Chapter 2 or its paired-uncertainty derivation**. This is a conditional assessment, not empirical establishment of its assumptions. Central claims remain `NOT ESTABLISHED` in the initial ledger only because the protocol requires independent second review before their final verified status. Their independently reconstructed first-review result is recorded separately. Empirical cells, source-code correspondence, reference metadata and complete PDF correspondence belong to the coordinator/Agent C; this review does not silently count them as verified.

## Recorded assumptions and definitions

Times are integers in `{0,...,N}`, with deterministic finite N and deterministic floor m in that set. All random variables and selectors/graders used are measurable on the stated probability space. The filtration increases with t; admissible policies are stopping times in that filtration, bounded between m and N. C_t is binary but need not be observable. Its conditional mean q_t is an adapted version in [0,1]. Costs are nonnegative, adapted, nondecreasing, and E[K_N]<infinity. These are stated in concise Chapter 2 lines 5-17. Since K_t<=K_N and q_t<=1, every G_t=q_t-K_t is integrable and |G_t|<=1+K_N. Finite time permits simultaneous almost-sure choices of versions and all required finite sums.

The theoretical model is a law of potential full trajectories, with prefix-compatible execution needed when transferring replay to an actually stopped generator. It does not prove that an archived fitted score is a true posterior, that a compute-limited agent can evaluate all measurable functions, that a reference grader is semantically correct, or that a policy-dependent generation protocol has unchanged prefixes. V7 explicitly makes these distinctions.

## Independent derivation D1: correctness at a random stopping time

For every permitted t, `{tau=t}` is F_t-measurable. Thus

`E[1{tau=t} C_t] = E[1{tau=t} E[C_t|F_t]] = E[1{tau=t} q_t]`.

Summing t=m,...,N gives E[C_tau]=E[q_tau]. Since 0<=K_tau<=K_N, cost is integrable and subtracting E[K_tau] gives Lemma 1 exactly. No assumption that C_t itself is F_t-measurable, that tau is deterministic, or that the q process is a martingale is used. The finite sum is essential to the stated simple proof; the manuscript does not assert its infinite-horizon extension.

## Independent derivation D2: hazards, zero denominators and drift

Define the F_t-measurable joint conditional probabilities

`r_t=E[1{C_t=0,C_{t+1}=1}|F_t]`,
`s_t=E[1{C_t=1,C_{t+1}=0}|F_t]`.

They satisfy `0<=r_t<=1-q_t` and `0<=s_t<=q_t`. Define alpha=r/(1-q) where q<1 and beta=s/q where q>0, assigning any value in [0,1] on the respective zero-at-risk sets. Then `(1-q)alpha=r` and `q beta=s`, including the boundaries. The binary identity `C_{t+1}-C_t=1{0->1}-1{1->0}` gives

`E[C_{t+1}|F_t]=q_t+r_t-s_t=(1-q_t)alpha_t+q_t(1-beta_t)`.

Because F_t is a sub-sigma-field of F_{t+1}, the tower property gives `E[q_{t+1}|F_t]=E[C_{t+1}|F_t]`. Subtract q_t and the conditional cost increment to obtain exactly

`mu_t=(1-q_t)alpha_t-q_t beta_t-c_t`.

There is no division by zero and no Markov, independent-transition or stationary-hazard assumption. The manuscript's 'arbitrary version' means a finite probability version, so it creates no 0 times infinity issue. For a common finite transition panel, `sum(C_next-C_now)=repairs-corruptions` exactly; dividing by panel size produces accuracy change. Conditional hazard denominators instead count the appropriate at-risk labels. These identities require a common eligible panel rather than unrelated changing denominators, as the text records.

## Independent derivation D3: finite Snell recursion, attainment and ties

Backward recursion makes each V_t F_t-measurable and integrable. One explicit domination is `-K_N<=G_t<=V_t<=1`, so `|V_t|<=1+K_N`. The upper bound follows inductively from conditional expectation. For every admissible sigma>=t, split `{sigma=t}` and `{sigma>t}`. On the second event the induction hypothesis at t+1 bounds `E[G_sigma|F_{t+1}]` by V_{t+1}; conditioning back proves `E[G_sigma|F_t]<=V_t`.

Define theta_N=N and, backwards, theta_t=t on `{G_t>=E[V_{t+1}|F_t]}`, otherwise theta_t=theta_{t+1}. Each theta_t is a permitted stopping time. Induction gives `E[G_theta_t|F_t]=V_t`. Because V_t is the displayed maximum, the stop event is exactly `{V_t=G_t}`. Therefore theta_m is the first equality, terminates by N, and attains E[V_m]. Equality is resolved by stopping; later optimal stops can coexist on ties, so no uniqueness is claimed. Lemma 1 transfers the optimum to C_tau-K_tau.

The theorem uses all F_t-adapted policies; a learned architecture's representable policy class need not attain it. Zero costs do not imply that waiting is strictly better: e.g. a constant correctness belief makes every time optimal, and the prescribed earliest stop is still optimal. At m=N (including N=m=0), the recursion has only its terminal value and the unique permitted time is optimal. No positive cost or nonempty crossing assumption is necessary.

The primary Ferguson text was accessed directly at https://www.math.ucla.edu/~tom/Stopping/sr3.pdf (Chapter 3, especially sections 3.1-3.2). It agrees with equality-based earliest stopping and distinguishes infinite-horizon existence assumptions. The finite proof above supplies the assumptions actually needed here; no infinite-horizon theorem is being imported without its conditions. Chapter 5 at https://www.math.ucla.edu/~tom/Stopping/sr5.pdf was also accessible. The original Clopper-Pearson DOI endpoint returned an internal access error, so no historical source quotation or metadata verification is claimed.

## Independent derivation D4: persistence at a random crossing

Let rho=tau_mu. Since mu_t is F_t-measurable, rho is a bounded stopping time. Before rho, `mu_t>0`; on `{rho<=t}`, persistence supplies `mu_t<=0`. For an arbitrary permitted stopping time sigma, the pathwise stopped telescoping identity is

`G_rho-G_sigma = sum_{t=m}^{N-1} (1{rho>t}-1{sigma>t})(G_{t+1}-G_t)`.

Both indicator events are F_t-measurable. Taking expectations replaces the increment by mu_t. On `{sigma<=t<rho}` the coefficient is +1 and mu_t>0. On `{rho<=t<sigma}` the coefficient is -1 and mu_t<=0. Else the coefficient is zero. Every summand has nonnegative expectation; hence `E[G_rho]>=E[G_sigma]` for every sigma. This proves the full optimality claim, including policies that stop earlier on some paths and later on others, and a genuinely random crossing. It avoids incorrectly applying a deterministic-time argument to a random crossing. One can also interpret the manuscript's remaining-phase supermartingale argument through the adapted process `G_t-G_{t wedge rho}`, whose conditional increments are `1{rho<=t}mu_t<=0`.

Strictly positive drift before rho implies Snell continuation is strictly better there. Persistence bounds all permitted later conditional rewards at rho by G_rho; therefore rho coincides with the earliest Snell equality, although the manuscript only needs optimality. On a zero-drift tie, stopping can be optimal without being unique. Persistence is a sufficient structural assumption and is explicitly not empirically established by observed panel curves or fitted heads. A marginal mean drift curve cannot establish this almost-sure conditional condition.

## Independent derivation D5: delayed repair and same-q counterexample

The manuscript's deterministic example gives q=C=(0,0,1), K=(0,.10,.20), G=(0,-.10,.80), mu=(-.10,.90). With m=0,N=2, the myopic rule stops at 0 for reward 0, while the Snell rule stops at 2 for .80. Both the counterexample and its stated violation of persistence are exact. It establishes a failure of unconditional myopic optimality, not its deployed frequency. The distinct live floor m=2 is explicitly qualified.

To check that q alone does not determine the Bellman action, take zero current cost, one step left, current q=.5 in two observed regimes. In regime R the next correctness mean is 1, and in regime S it is 0; charge next-step cost .1 in either. Current reward is .5 in both, but continuation rewards .9 and -.1 yield opposite decisions. Such conditional means can be realized using unobserved Bernoulli correctness and an observed regime. Thus future law/cost information is needed in general. No theorem of a universal q threshold follows from the binary drift identity.

## Independent derivation D6: paired intervals and zero discordance

For each iid task pair, active correctness minus baseline correctness equals I-W, so delta=pi_I-pi_W. Across pairs, each marginal count is Binomial(n,pi); I and W within the same pair are mutually exclusive, and independence between them is not required. Two marginal confidence sets with coverage at least .975 each fail jointly with probability at most .025+.025=.05. On simultaneous coverage, subtracting endpoints proves `[L_I-U_W,U_I-L_W]` covers delta. The argument permits the resulting interval to extend outside the physical [-1,1] parameter range; intersecting would be optional and conservative.

With zero observations of one discordance type, its equal-tail 97.5% CP lower endpoint is 0 and the upper endpoint solves `(1-U)^n=.0125`, giving `U=1-.0125^(1/n)`. It is strictly positive for every finite n>=1. For zero counts of both types, delta's displayed interval is [-U,U]. Independently computed U is .0428740302384385 for n=100 and .19675967970462538 for n=20, agreeing with the rounded +/-4.29 and +/-19.68 percentage-point displays. The iid task-pair sampling assumption is essential for coverage; the manuscript explicitly limits the handpicked traps to an iid-reference calculation. Bootstrap ratio intervals do not automatically inherit this exact coverage proof or time-uniform coverage.

## Independent derivation D7: costs, ranking and probability qualifications outside Chapter 2

The utility definitions imply `.05/.0002=250` completion tokens per response-increment penalty. For positive baseline lengths b_i, total completion saving equals `sum_i [b_i/sum_j b_j]*(1-a_i/b_i)`, proving the baseline-length weighting statement. Completion-only accounting does not account for additional prompt/probe/peer resources; a shorter target trace alone cannot show lower complete system cost. The two-response floor can exclude a first-response-only correct answer; it is a feasible-action restriction, not a consequence of the optimal stopping theorem.

ROC-AUC is invariant to strictly increasing transforms of a score, while score values/probability error can change, so ranking performance alone cannot establish conditional probabilities. A constant .5 score is marginally calibrated in a balanced mixture whose observed regime reveals label 0 or 1; it is not the posterior given the regime. Class-weighted binary log loss with weights w1,w0 has population minimizer `s=w1*p/(w1*p+w0*(1-p))`; equalizing class totals at prevalence .1 can yield s=.5 rather than p=.1. Thus the qualified discussion of archive calibration and reweighted targets is mathematically sound.

Question-level coverage does not imply coverage for all repeatedly queried times; a union over times would need additional control. Likewise, bootstrap draws recycle observations and cannot create new independent data or remove leakage in the original score construction. Noninferiority requires a defined tolerance and valid one-sided comparison; the recorded intervals including meaningful negative differences do not establish it. These are limits of inference rather than failures of the conditional theory.

## Supplemental exact finite-state checks

Ran `python ThesisDocs/verification/math_reliability_audit_2026-10-13/agent_a/finite_tree_check.py` using exact `fractions.Fraction` arithmetic. It enumerates every adapted deterministic policy on a fair binary observation tree of horizon two, all 3^7 correctness-mean assignments from {0,1/2,1}, all costs {0,1/10,1/2}, and floors {0,1,2}. Results: 19,683 tree/cost/floor cases, 17,280 satisfying persistence, including 3,324 with a random crossing; zero Snell mismatches and zero persistence-theorem mismatches. There are 686 suboptimal myopic cases without persistence. N=0, m=N, zero costs, equality ties and the delayed-repair example are separately checked. `finite_tree_results.json` records the result. These finite checks supplement, and do not replace, D1-D6.

## Initial scope limits and corrections

No manuscript correction is justified by this first review. An optional expanded proof of Theorem 2 can use D4 to make the random-crossing argument easier to inspect, but the existing concise proof is valid and is not classified as an error or missing assumption. The core theorem does not require additional Markov or hazard independence assumptions. Standard exact binomial coverage is conditional on iid task pairs already stated in Chapter 3; the manuscript does not claim adversarial-population coverage.

The ledger explicitly separates mathematical derivations from whether frozen empirical rows and actual executed decision paths support their numerical or implementation premises. Numerical arithmetic from displayed counts is independently checked, but not counted as source-row reproduction. Agent C/coordinator evidence remains required for empirical quantities. No new collection, generation, training, regrading, manuscript/code/frozen-evidence mutation, commit, push, or external message occurred. Only this agent's authorized directory was written. No PDF page image was inspected by Agent A in the initial review; complete rendered correspondence remains with the coordinator. Required repository test execution also remains with the coordinator rather than being inferred from prior receipts.

The preceding record preserves the initial assessment. Subsequent reconciliation: Agent B independently derived the concise source results before opening A's claim verdicts and recorded `agent_b/cross_review_a.md`, B-XA-01 through B-XA-06. That second review covers the random-stop lemma, zero-at-risk hazards and drift, Snell recursion/ties/boundaries, persistent random crossing against arbitrary policies, delayed repair and exact paired intervals. It agrees with D1-D6 and identifies no manuscript correction. The initial ledger and coverage map are preserved as `claim_ledger_initial.json/.csv` and `coverage_map_initial.json`; the current ledger records the final conditional verified outcome for all fifteen previously pending central claims, and the final coverage map incorporates that evidence. Twenty distinct ledger claims (including surrounding definitions already valid on first review) now carry explicit independent B second-review mappings. This does not change the empirical/implementation/PDF limits recorded above.
