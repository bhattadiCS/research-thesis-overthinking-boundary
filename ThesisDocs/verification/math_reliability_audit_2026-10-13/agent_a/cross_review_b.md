# Agent A independent cross-review of consequential v5 bounds

Recorded after the initial v7 review files were saved and before receiving Agent B's conclusions. Reviewed actual source `ThesisDocs/chapters/chapter2_theory.md`, sections 2.3, 2.6 and 2.7, affine-stakes corollary in 2.2, and `ThesisDocs/appendices.md` Appendix C. Source-to-compiled mappings: these Chapter 2 lines map to `Masters_Thesis_Formal_v5.md` with offset +93; Appendix C lines map with offset +1263. The independent cross-review finds the bounds valid under the assumptions actually stated, with the limited meanings described below. It finds no justified manuscript correction. After this derivation was saved, B's IDs were reconciled in `cross_review_b_map.json`: AXB1 maps B-CH2-11/14; AXB2 maps B-CH2-22; AXB3 maps B-CH2-23; AXB4 maps B-CH2-24; AXB5 maps B-CH2-25/26; AXB6 maps B-CH2-27/28; AXB7 maps B-M246 through257 and B-P243 through246. B's independent derivations agree, and its small audit-prose endpoint typo was corrected without changing manuscript rounding.

## AXB1: affine stakes and monotonicity

Source Chapter 2 lines 152-175 and 235-263. Let w=v+p>0. The realized payoff is `w C_tau-p-K_tau`; the stopped conditional-mean identity gives `w q_tau-p-K_tau`, and one-step drift is `w*((1-q)alpha-q beta)-c`. Both formulas are exact. The additive -p does not change decisions.

For a decision at t<N, the value of requiring one more response is the essential supremum over bounded admissible sigma>=t+1. Normalizing by w and subtracting the current reward gives exactly

`Delta_t(w)/w = ess sup_sigma E[q_sigma-q_t-(K_sigma-K_t)/w | F_t]`.

The essential supremum is legitimate because finite Snell recursion attains it. Under a fixed law, filtration, candidates, costs, floor and horizon, the policy set is the same at both w values. Since K_sigma-K_t>=0, every member of the family is nondecreasing in w, and so is its essential supremum. Thus positive continuation advantage at w1 implies positive advantage at w2>=w1. Earliest optimal stopping continues strictly when Delta>0 and stops when Delta<=0. On every path where w1 has continued so far, w2 has also continued; therefore tau*(w1)<=tau*(w2), including ties at zero and common mandatory terminal stopping. Equal w but different v,p changes only a constant and leaves contact times equal. No claim that the actual correctness path is monotone, that accuracy improves strictly, that an estimated drift boundary is ordered, or that stakes can change the generation law is used. The nondecreasing-cost assumption is material to this proof.

## AXB2: class weighting and drift sensitivity

Source Chapter 2 lines 408-433. Differentiating `-w1*p*log(r)-w0*(1-p)*log(1-r)` gives derivative `-w1*p/r+w0*(1-p)/(1-r)`. The second derivative is `w1*p/r^2+w0*(1-p)/(1-r)^2>0` for p,r in (0,1), positive weights. Solving yields `r=w1*p/(w1*p+w0*(1-p))`; rearranging gives the stated inverse. The positive endpoint masses make risk diverge at r=0,1, so the unique interior critical point is the global optimum. This is an unrestricted population-risk result; regularized/misspecified fitted models need not satisfy it exactly.

For prevalence p=1/10, inverse-frequency balanced class weights have ratio w1/w0=9, yielding r=1/2. At alpha=1/50,beta=1/200,c=1/100, drift at q=1/10 is `18/1000-1/2000-1/100=3/400`; at q=1/2 it is `1/100-1/400-1/100=-1/400`. The counterexample to treating reweighted probabilities as the original target is exact. AUC invariance to a strictly increasing score transform and the balanced constant-score calibration example also hold, but give no universal conditional error bound.

## AXB3: Proposition 6 perturbation and rectangle bounds

Source Chapter 2 lines 435-474. For f(q,a,b,c)=(1-q)a-qb-c, the exact subtraction is

`f(qhat,ahat,bhat,chat)-f(q,a,b,c) = (1-qhat)(ahat-a)-qhat(bhat-b)-(qhat-q)(a+b)-(chat-c)`.

Taking absolute values and using qhat in [0,1], a+b<=2 gives exactly the manuscript bound `2*eps_q+(1-qhat)*eps_a+qhat*eps_b+eps_c`. All error tolerances are nonnegative; the premise is pointwise error control, rather than something implied by cross-validation or calibration summaries.

The partial derivatives/signs are `df/da=1-q>=0`, `df/db=-q<=0`, `df/dq=-(a+b)<=0`, `df/dc=-1`. Therefore on a nonempty rectangle with probability endpoints in [0,1], its minimum occurs at (q_U,a_L,b_U,c_U) and maximum at (q_L,a_U,b_L,c_L). The stated L,U are exact extrema of that rectangle, even if some monotonicities are non-strict. True parameters may obey additional joint constraints, making a rectangle larger than their feasible set; that only makes propagation conservative. The word 'exact' describes rectangle extrema, not a claim of exact statistical coverage. Simultaneous input-interval coverage is explicitly required and not produced by this proposition.

## AXB4: first-crossing guarantee

Source Chapter 2 lines 478-500. On the event that mu_t<=U_t simultaneously at all eligible decision times, any first U_t<=0 before N satisfies mu_tau<=U_tau<=0. Thus `{tau_U<N,mu_tau>0}` is a subset of the simultaneous noncoverage event, proving probability <=delta for meaningful 0<delta<1. U_t must be F_t-observable to make the first-crossing rule a runtime stopping time; this is stated. No persistence assumption is needed for this one-step sign statement. Persistence or an upper confidence bound on Bellman advantage is required for the respective stronger optimality statement. Neither sign coverage nor Bellman optimality implies retained accuracy without a separate argument.

## AXB5: conditional Hoeffding and failure allocation

Source Chapter 2 lines 502-537. Condition on the pre-probe H_t. The count n_t>=1 is fixed given that information, the bounded probes are conditionally independent and have common conditional mean theta_t. For a variable Z in [0,1] of mean p, convexity bounds its centered exponential moment by the centered endpoint Bernoulli moment. For `psi(s)=log((1-p)*exp(-sp)+p*exp(s(1-p)))`, the tilted Bernoulli variance gives psi''(s)<=1/4, and psi(0)=psi'(0)=0. Integrating gives psi(s)<=s^2/8. Rescaling [a,b] and multiplying independent conditional moments, exponential Markov and optimization give

`P(theta_t > Xbar_t+epsilon | H_t) <= exp(-2*n_t*epsilon^2/(b-a)^2)`.

Substitute epsilon=(b-a)*sqrt(log(1/delta_t)/(2*n_t)) to obtain the stated one-sided bound. If a=b, theta equals the observation and the conclusion holds directly; the formula has zero width. For r=t-m+1, `sum_{r=1}^M delta/[r(r+1)] = delta*(1-1/(M+1))<=delta`, so conditioning away and union bounding proves simultaneous coverage of theta_t. Independence across decision times is unnecessary. Within-prefix dependence, adaptive probe-count choice after seeing probes, or a changed post-probe target are not covered by this fixed-count result. All are explicitly qualified in the source. The pre-probe/post-probe distinction is substantive: information acquired by a probe may alter E[G_next-G_now|F_t] even when a pre-probe target is accurately bounded.

Primary-source access: the Hoeffding DOI returned HTTP403; no claim of reading its inaccessible original text is made. The proof above independently reconstructs the bounded-variable inequality. The Howard et al. DOI resolves to its primary Annals of Statistics article landing page (https://doi.org/10.1214/20-AOS1991), but the tool exposed an iframe rather than article text; this review does not infer detailed theorem assumptions from that page. The manuscript's elementary e-process is checked directly below.

## AXB6: elementary e-process, mixtures, Ville bound and dependence witness

Source Chapter 2 lines 539-562. Under the conditional-mean null, eta in [0,1) and X_i in [-1,1], each factor 1-eta X_i is positive and at most2. Thus E_n is adapted, integrable at every finite n, nonnegative and starts at1. Its conditional expectation is `E_{n-1}*(1-eta E[X_n|G_{n-1}])<=E_{n-1}`. A fixed convex mixture of these nonnegative supermartingales remains a supermartingale by conditional Tonelli and has initial value1. Data-dependent mixture weights chosen using future samples would need another argument; the source specifies fixed weights.

For any threshold h>0 let T_h be its first attained crossing. For every finite M, optional sampling at T_h wedge M gives `1>=E[E_{T_h wedge M}]>=h*P(T_h<=M)`. Let M increase to infinity to bound the probability of an attained crossing by 1/h. To include paths whose supremum equals 1/delta without attaining it, take thresholds h_k<1/delta increasing to 1/delta. The decreasing intersection of the crossing events is `{sup_n E_n>=1/delta}`, so continuity of probabilities gives the stated <=delta bound. Neither uniform integrability at infinity nor independence is needed. This explicit limiting step prevents confusing a supremum with an attained maximum.

For the shared-sign example B=+/-1 with equal probabilities, X_i=B has marginal mean0, but its time10 product at eta=1/2 on B=-1 is `(3/2)^10=59049/1024=57.6650390625>20`. Hence crossing probability is at least1/2, contradicting a purported5% guarantee based only on marginal means/boundedness. There is no contradiction with the valid theorem: after the first observation of a negative sign, the later conditional means are -1 and violate its null. The example and its limiting interpretation are correct.

## AXB7: Appendix C paired accuracy bounds

Source Appendix C lines31-39; identical mathematical argument to concise v7 Chapter3 lines49-53. The identity active-baseline=I-W, marginal Binomial(n,pi) laws under iid pairs, marginal97.5% exact intervals, union-bound95% simultaneous coverage, and endpoint subtraction are all valid without within-pair independence. With no discordances, each marginal upper bound solves `(1-U)^n=.0125`, yielding `1-.0125^(1/n)`. The limitation to an iid-reference model for fixed handpicked traps is explicit. The bootstrap ratio intervals are described as resampling intervals rather than inheriting the exact binomial guarantee. See initial `review.md` D6 for independently computed n100/n20 endpoints. No noninferiority inference follows merely from equal observed accuracy.

## Initial cross-review conclusion and remaining evidence

These are favorable deductive reviews under recorded assumptions, not evidence that the fitted implementation satisfies the coverage/error premises. Agent C must independently verify the claims about actual detector code and empirical outputs. This cross-review did not rerun generation/training/collection/regrading or modify any manuscript/frozen evidence. No expanded bound or altered hypothesis is needed to make the stated v5 results valid. The only pending reconciliation is mapping to Agent B's independently recorded IDs and resolving any concrete counterargument he reports.
