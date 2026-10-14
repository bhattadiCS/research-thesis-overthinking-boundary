# B second review of A's central v7 proofs

Derived from concise_v7/chapter2_mathematics.md itself on 2026-10-13, after recording my independent v5 initial review and before opening A's claim verdicts. A's agreement/disagreement is not the basis of the results below. All statements concern the exact baseline manuscript and the recorded statistical information model; no empirical posterior or deployment guarantee is inferred.

## B-XA-01 — Model and random-time reward replacement

Source: concise_v7/chapter2_mathematics.md:5–21. Outcome: **VERIFIED UNDER RECORDED ASSUMPTIONS**. Dependencies: fixed measurable causal selector and binary grader, nested cumulative-history filtration, deterministic finite horizon/floor and nonnegative nondecreasing adapted integrable K. This is the same mathematical claim as B-CH2-01/04/05, independently reapplied to the concise source.

For a bounded admissible tau, {tau=t}={tau<=t}\{tau<=t-1} is in F_t. Since q_t=E(C_t|F_t), E[1_{tau=t}(C_t-q_t)]=0 for each t. A finite sum proves E C_tau=E q_tau. Costs are integrable because K_tau<=K_N and E K_N<infinity. Subtract their expectations to obtain Lemma 1. Neither C_t nor Y* must be observable; what is needed is the stopping-event measurability. If correctness is already measurable because the reference is determined by the task, q=C and the equality still holds. Runtime interface exclusion of gold fields does not itself imply nondegenerate probabilistic uncertainty; the concise manuscript explicitly includes that qualification.

## B-XA-02 — Repair identity including null states

Source: concise_v7/chapter2_mathematics.md:27–39. Outcome: **VERIFIED UNDER RECORDED ASSUMPTIONS**. Dependencies: B-XA-01, binary target and same conditioning. Corresponding extended claims: B-CH2-07/08/10.

Let r=E[1_{C_t=0,C_next=1}|F_t] and s=E[1_{C_t=1,C_next=0}|F_t]. On 1-q>0 define alpha=r/(1-q), and on q>0 define beta=s/q; assign any [0,1] version on the null conditioning state. Since r<=1-q and s<=q, the zero multipliers are valid also at q=0 or q=1. Then E(C_next|F_t)=r+q-s=(1-q)alpha+q(1-beta). Nested conditioning gives E(q_next|F_t)=E(C_next|F_t). Subtract q and the expected cost increment to obtain mu=(1-q)alpha-q beta-c. No conditional independence or Markov assumption occurs. Finite-sample analogues require common transition-eligible rows and correct at-risk denominators; heterogeneous marginal means do not justify products of means.

## B-XA-03 — Snell recursion, first contact, ties and boundary cases

Source: concise_v7/chapter2_mathematics.md:45–53. Outcome: **VERIFIED UNDER RECORDED ASSUMPTIONS**. Dependencies: B-XA-01; finite horizon; all computation included in K; fixed selector and potential process. Corresponding extended claims: B-CH2-12/13.

Because |G_t|<=1+K_N, all rewards and conditional expectations are integrable. At N the only permitted rule stops and V_N=G_N. Inductively suppose V_{t+1} is the conditional maximum attained from t+1. A rule from t either stops on an F_t event or continues and attains at most E(V_{t+1}|F_t); hence its conditional reward is at most their pointwise maximum V_t. This maximum is attained by selecting stop on {G_t>=E(V_{t+1}|F_t)} and otherwise the attaining future rule. Thus the recursion applies to arbitrary histories, without assuming scalar-state Markov structure. The stop event is measurable and equals {G_t=V_t}, including equality of stop and continuation. Since contact holds at N, the first contact is finite.

An alternative direct bound uses the integrable supermartingale V and finite conditional optional sampling: E G_tau<=E V_tau<=E V_m. Before first contact the recursion forces V_t=E(V_{t+1}|F_t), so the process stopped at contact is a martingale and achieves equality. Lemma 1 transfers this optimum to realized C-K.

The earliest-contact tie convention is coherent. At a tie, stopping attains the maximum, although a later rule can also be optimal. A rule that stops with positive probability before contact must stop on a set where the conditional continuation value is strictly larger. Switching on that measurable set to the attaining future policy strictly improves its expectation, so no optimal rule can stop earlier. This proves earliest optimality in addition to the stated optimality. If m=N (including N=0), there is only the terminal action, the proof reduces to Lemma 1 and the drift search is empty.

## B-XA-04 — Random first crossing and persistent nonpositive drift

Source: concise_v7/chapter2_mathematics.md:57–63. Outcome: **VERIFIED UNDER RECORDED ASSUMPTIONS**. Dependencies: B-XA-01/02/03, and the explicit probability-one condition 1_{tau_mu<=t}mu_t<=0 for every t<N. Corresponding extended claims: B-CH2-15/16.

Write sigma=tau_mu. It is an F-stopping time because every mu_t is F_t measurable, the search starts at deterministic m and the finite fallback is N. For any admissible tau,

E G_tau = E G_m + sum_{t=m}^{N-1} E[1_{tau>t}mu_t].

This follows by telescoping the bounded stop, then conditioning each increment using the measurable indicator. On each common probability-one path before sigma, mu_t>0 and 1_{sigma>t}=1. At/after sigma, the given premise implies mu_t<=0 and 1_{sigma>t}=0. Therefore [1_{sigma>t}-1_{tau>t}]mu_t>=0 for every t, pathwise. Sum and take expectations: E G_sigma>=E G_tau. This establishes global optimality against all admissible rules, including ones stopping before sigma or choosing later adaptive continuation.

The concise optional-sampling argument can also be made formal. The post-crossing difference process X_t=G_t-G_{t wedge sigma} has increment 1_{sigma<=t}(G_{t+1}-G_t) and nonpositive conditional drift by the premise. Finite optional sampling from sigma to any later bounded rule gives conditional expected later reward no larger than G_sigma. Before sigma, positive mu implies positive full continuation advantage because V_{t+1}>=G_{t+1}. Thus no optimal rule stops before sigma, and stopping at sigma attains the best remaining value. Finite horizon and integrability, rather than any unbounded optional-stopping theorem, justify this step. A monotone or negative population mean curve does not verify the pathwise conditional premise, and the manuscript correctly states that limitation.

## B-XA-05 — Delayed repair

Source: concise_v7/chapter2_mathematics.md:65–69. Outcome: **VERIFIED UNDER RECORDED ASSUMPTIONS**. Dependencies: deterministic sequence C=(0,0,1), K=(0,1/10,1/5), N=2,m=0. Corresponding extended claim: B-CH2-17.

The rewards are (0,-1/10,4/5), so mu_0=-1/10 and mu_1=9/10. The first nonpositive drift rule stops at zero with reward zero. Backward recursion gives V_2=4/5,V_1=4/5,V_0=4/5, attained at time two. Persistence fails exactly because mu_1>0 after the first crossing. One common latent reference can realize the example with surely incorrect placeholders at zero and one and a terminal generator that emits the reference. Alternatively deterministic correctness is allowed by the model's own degenerate measurable-reference case. Shifting times/floor by two preserves the comparison, so the different example floor does not invalidate it.

## B-XA-06 — Paired binomial intervals

Source: concise_v7/chapter3_methods.md:49–53; independently derived in review.md D18 from appendices.md:31–39 before reading A's ledger. Outcome: **VERIFIED UNDER RECORDED ASSUMPTIONS**. I_i=1{baseline wrong, active correct}, W_i=1{baseline correct, active wrong}; iid task pairs make their marginal sums Bin(n,pi_I) and Bin(n,pi_W). Each exact two-sided 97.5% binomial interval has noncoverage <=.025. Regardless of the dependence of the two counts, the union bound gives simultaneous coverage >=.95, and subtracting simultaneous limits yields delta in [L_I-U_W,U_I-L_W]. With zero discordances, the lower marginal endpoint is zero and the upper endpoint solves (1-U)^n=.0125, giving U=1-.0125^(1/n)>0 for finite n. Nothing here supplies a probability-sampling guarantee for a fixed handpicked bank; both manuscripts explicitly qualify that use.

## Exact A-claim reconciliation

After the above independent derivations were recorded, A's IDs were read for mapping: B-XA-01 covers A-014/015; B-XA-02 covers A-017/018/019/020/021; B-XA-03 covers A-028/029/030; B-XA-04 covers A-033/034/035/036; B-XA-05 covers A-038/039; B-XA-06 covers A-051/052/053/054. All these mapped claims receive VERIFIED UNDER RECORDED ASSUMPTIONS from this second review. The mathematical model's runtime information restrictions are assumptions here; actual code conformance is separately audited by C.

## Reconciliation limit

No correction of the central v7 deductive mathematics is justified by this review. The short proof in Theorem 2 is valid; the telescoping derivation above supplies a fully explicit check of the random-crossing step. These conclusions support conditional validity only. The learned controller's posterior calibration, continuation law, persistence and accuracy noninferiority remain unestablished premises or empirical limitations, exactly as the manuscript says. Source/PDF correspondence, all empirical units and institutional acceptance are separate work. After this file was recorded, exact A ledger IDs can be mapped without using their verdicts as evidence.
