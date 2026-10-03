# Mathematical foundations for causal, cost-aware stopping

**Status:** canonical mathematical reference for the thesis draft, 2026-10-02.
The results below supersede the unqualified optimality and deployment-safety
claims in `overthinking_boundary.md`, `OBE_SYSTEM_THESIS.md`, and
`../ThesisDocs/archive/thesis_stopping_rule_algorithm.md`. Historical results
and fitted equations remain empirical artifacts; this note does not promote
them to a calibrated online policy.

## 1. Probability model and admissible information

Fix a finite horizon $N$ and a deterministic earliest permitted stop
$m\in\{0,\ldots,N\}$. Work on a probability space
$(\Omega,\mathcal A,\mathbb P)$. Let $Y^*$ be the reference answer,
$H_t$ the information actually available after decision step $t$, and

$$
\mathcal F_t=\sigma(H_0,\ldots,H_t),\qquad
A_t=a_t(H_0,\ldots,H_t),\qquad
C_t=\mathbf1\{A_t=Y^*\}.
$$

All observations, executed peer calls, verifier outputs, and controller
randomness used by a decision belong in this filtration. The runtime
interface does not supply offline ground-truth fields, future revisions,
future peer responses, or features computed using an unfinished trace.
These are interface restrictions; mathematical measurability is governed
by the information model and the caveat below. The correctness indicator
need not be observable, even though the answer is.

The filtration is a statistical information model, not a model of
computational hardness. If the reference answer is a known deterministic
measurable function of a fully revealed task, then the selected candidate's
correctness is $\mathcal F_t$-measurable and $q_t=C_t$, even without a gold
input field. All results below still hold in that degenerate case.
Nondegenerate conditional uncertainty requires a specified latent-reference
generative model or an explicitly coarser filtration of decision statistics,
with admissible policies restricted to that information. A fitted probability
based on prefix summaries does not establish a posterior conditioned on all
mathematical semantics of a task.

The candidate $A_t$ means the answer that the specified policy would emit
if it stopped at $t$. If the policy selects among current, earlier, or peer
answers, define $A_t$ by that causal selection procedure. Results for the
current generator answer do not automatically apply to a different selected
answer. If every available candidate in a finite, observable candidate
set can be selected freely, the optimal
immediate correctness reward is
$\max_{a\in\mathcal B_t}\mathbb P(Y^*=a\mid\mathcal F_t)$, where
$\mathcal B_t$ is the candidate set available at $t$.

A policy's stop time $\tau$ is admissible when

$$
m\le\tau\le N,\qquad \{\tau\le t\}\in\mathcal F_t\quad\text{for every }t.
$$

Let $K_t$ be the nonnegative, nondecreasing, integrable, adapted cumulative cost of **all**
computation executed by $t$, including peers and probes. Put

$$
q_t=\mathbb E[C_t\mid\mathcal F_t],\qquad G_t=q_t-K_t,
\qquad c_t=\mathbb E[K_{t+1}-K_t\mid\mathcal F_t].
$$

The objective is $\sup_{\tau}\mathbb E[C_\tau-K_\tau]$. The step-cost
specialization is $K_t=\lambda(t-m)$ for $t\ge m$; previously incurred
constant costs do not change the optimizer. The empirical choice
$\lambda=0.05$ is a utility convention, not a universal monetary cost or
an assertion about token savings.

**Lemma 1 (observable reward reduction).** For every admissible stop time,

$$
\mathbb E[C_\tau-K_\tau]=\mathbb E[G_\tau].
$$

**Proof.** The event $\{\tau=t\}$ belongs to $\mathcal F_t$, so
$\mathbb E[\mathbf1_{\{\tau=t\}}C_t]
=\mathbb E[\mathbf1_{\{\tau=t\}}q_t]$. Sum this equality over the finite
set $t=m,\ldots,N$, and subtract $\mathbb E[K_\tau]$. □

For stopping-only comparisons, a full potential trace may be defined even
after a policy would stop. Its prefixes must have the same law as the actual
generation process up to stopping. A controller that changes prompts,
sampling, peer allocation, or answer selection changes the process; a
retrospective evaluation must account for that change.

## 2. Exact binary correctness drift

For $t<N$, define the conditional joint transition masses

$$
r_t=\mathbb P(C_t=0,C_{t+1}=1\mid\mathcal F_t),\qquad
s_t=\mathbb P(C_t=1,C_{t+1}=0\mid\mathcal F_t),
$$

and the conditional repair and corruption probabilities

$$
\alpha_t=\begin{cases}r_t/(1-q_t),&q_t<1,\\0,&q_t=1,\end{cases}
\qquad
\beta_t=\begin{cases}s_t/q_t,&q_t>0,\\0,&q_t=0.\end{cases}
$$

The value assigned on a zero-probability conditioning state is arbitrary
in $[0,1]$: its multiplier is zero. These quantities are probabilities
per decision transition, not continuous-time intensities. No Markov,
independence, monotonicity, or observability of $C_t$ is needed for the
following identity.

**Theorem 2 (conditional drift identity).** Almost surely,

$$
\mu_t:=\mathbb E[G_{t+1}-G_t\mid\mathcal F_t]
=(1-q_t)\alpha_t-q_t\beta_t-c_t.
$$

**Proof.** Pointwise,
$C_{t+1}-C_t=\mathbf1_{\{C_t=0,C_{t+1}=1\}}
-\mathbf1_{\{C_t=1,C_{t+1}=0\}}$.
The tower property and nested filtrations give

$$
\mathbb E[q_{t+1}\mid\mathcal F_t]
=\mathbb E[\mathbb E[C_{t+1}\mid\mathcal F_{t+1}]\mid\mathcal F_t]
=\mathbb E[C_{t+1}\mid\mathcal F_t].
$$

Subtract $q_t=\mathbb E[C_t\mid\mathcal F_t]$, use the indicator identity,
and subtract $c_t$. □

Consequently, with $D_m=0$ and $D_t=\sum_{s=m}^{t-1}\mu_s$,
$M_t=G_t-G_m-D_t$ is an integrable martingale. This is a discrete Doob
decomposition. The process $q_t$ need not be a martingale: it concerns
the changing answer $C_t$, rather than the conditional probability of
one fixed event.

For a sample of $n$ observed transitions, let $n_0,n_1$ be the numbers
of current incorrect and correct states and let $n_{01},n_{10}$ count
repairs and corruptions. On that same sample,

$$
\left(1-\frac{n_1}{n}\right)\frac{n_{01}}{n_0}
-\frac{n_1}{n}\frac{n_{10}}{n_1}
=\frac{n_{01}-n_{10}}{n}
=\frac1n\sum_i(C_{i,t+1}-C_{i,t}),
$$

with the zero-denominator convention above. Frequencies $n_{01}/n$
and $n_{10}/n$ are already joint event frequencies; multiplying them
again by $1-q$ and $q$ is incorrect. Matching this sample identity
verifies arithmetic, not an online probability model. A product of separately
averaged, instance-specific probabilities generally differs from the
average of their products.

**Corollary 2.1 (affine stakes).** Suppose a correct answer earns $v\ge0$
and an incorrect answer loses $p\ge0$, with $v+p>0$, and computation still
costs $K_t$. The observable stop reward and its conditional drift become

$$
G_t^{v,p}=v q_t-p(1-q_t)-K_t=(v+p)q_t-p-K_t,
$$

$$
\mu_t^{v,p}=(v+p)\bigl((1-q_t)\alpha_t-q_t\beta_t\bigr)-c_t.
$$

**Proof.** The realized terminal reward is
$vC_t-p(1-C_t)-K_t$. Apply Lemma 1 and the linearity of conditional
expectation to Theorem 2. □

For constant computation cost, the correctness-versus-compute trade-off is
therefore determined by cost divided by $v+p$; $-p$ is an additive constant.
Increasing stakes can change the optimal policy. Under a fixed potential
trace law and nondecreasing computation costs, optimal continuation regions
expand as $v+p$ increases; Corollary 3.1 proves the corresponding ordering
of earliest optimal stop times. That ordering need not hold for empirically
thresholded policies, or when stakes change the generation process, answer
selection, or costs. All comparisons must state both scales.

## 3. General finite-horizon optimal stopping

**Theorem 3 (Bellman recursion and Snell envelope).** Set

$$
S_N=G_N,\qquad
S_t=\max\{G_t,\mathbb E[S_{t+1}\mid\mathcal F_t]\},
\quad t=N-1,\ldots,m.
$$

Then $S$ is the smallest integrable supermartingale dominating $G$.
For each $t\ge m$,

$$
S_t=\operatorname*{ess\,sup}_{\tau:\,t\le\tau\le N}
\mathbb E[G_\tau\mid\mathcal F_t],\qquad
\tau_t^*=\inf\{s\in\{t,\ldots,N\}:S_s=G_s\}
$$

is an optimal stopping time, and
$S_t=\mathbb E[G_{\tau_t^*}\mid\mathcal F_t]$.

**Proof.** Backward induction gives integrability, adaptation,
$S_t\ge G_t$, and $S_t\ge\mathbb E[S_{t+1}\mid\mathcal F_t]$.
If $W$ is any integrable supermartingale with $W_s\ge G_s$, then
$W_N\ge S_N$. If $W_{s+1}\ge S_{s+1}$, the supermartingale property
and domination give
$W_s\ge\max\{G_s,\mathbb E[S_{s+1}\mid\mathcal F_s]\}=S_s$.
This proves minimality by induction.

For any bounded admissible $\tau$, finite optional sampling, or the
finite telescoping sum of conditional supermartingale increments, gives
$\mathbb E[G_\tau\mid\mathcal F_t]
\le\mathbb E[S_\tau\mid\mathcal F_t]\le S_t$.
The first contact time is measurable and no later than $N$.
On $\{s<\tau_t^*\}$, $S_s>G_s$, so the recursion forces
$S_s=\mathbb E[S_{s+1}\mid\mathcal F_s]$.
Thus $(S_{s\wedge\tau_t^*})_{s=t}^N$ is a martingale. At contact
$S_{\tau_t^*}=G_{\tau_t^*}$, giving equality in the upper bound.
Attainment establishes the essential-supremum representation. □

This is the standard finite-horizon optimal stopping construction; see
[Ferguson, Chapter 3, Section 3.2](https://www.math.ucla.edu/~tom/Stopping/sr3.pdf).
The proof here specializes it to the observable correctness-minus-cost
reward and includes the explicit protocol floor.

The optimal continuation advantage is

$$
\Delta_t^*=\mathbb E[S_{t+1}\mid\mathcal F_t]-G_t
=\mu_t+\mathbb E[S_{t+1}-G_{t+1}\mid\mathcal F_t]\ge\mu_t.
$$

The second term is the value of future choices. Therefore $\mu_t>0$
is sufficient to favor continuation at $t<N$; $\mu_t\le0$
alone is insufficient to justify stopping. At $N-1$ the terms coincide.
The optimal policy stops at the first $\Delta_t^*\le0$, or at $N$.

**Corollary 3.1 (stake monotonicity under a fixed process).** Compare affine
stakes with $0<w_1\le w_2$, where $w_j=v_j+p_j$. Suppose that the
filtration, candidate answers, potential trace law, horizon, floor, and
nondecreasing cumulative cost process are identical in both problems.
Then their earliest optimal contact times satisfy
$\tau^*(w_1)\le\tau^*(w_2)$ almost surely.

**Proof.** For $t<N$, the normalized advantage of being required to
continue at least once is

$$
\frac{\Delta_t^*(w)}{w}
=\operatorname*{ess\,sup}_{\tau:\,t+1\le\tau\le N}
\mathbb E\left[q_\tau-q_t-\frac{K_\tau-K_t}{w}
\,\middle|\,\mathcal F_t\right].
$$

The constant incorrect-answer offset cancels. Since $K_\tau-K_t\ge0$,
each conditional expectation is nondecreasing in $w$, as is its essential
supremum. Consequently $\Delta_t^*(w_1)>0$ implies
$\Delta_t^*(w_2)>0$. Along any fixed path, the larger-stakes policy cannot
stop while the smaller-stakes policy has continued at every decision so
far: each of those strict continuation conditions also holds for larger
stakes. With the same tie convention and forced terminal stop, this proves
the stated ordering. □

This result does not assert that accuracy itself increases strictly, that
all fitted drift boundaries obey the ordering, or that a negative drift
becomes positive merely because stakes increase.

## 4. When a drift-sign boundary is optimal

**Theorem 4 (persistent nonpositive drift).** Define

$$
T_c=\min\bigl(\{t\in\{m,\ldots,N-1\}:\mu_t\le0\}\cup\{N\}\bigr).
$$

Assume, almost surely, that $\mu_s\le0$ for every
$s\in\{T_c,\ldots,N-1\}$. Then $T_c$ maximizes
$\mathbb E[G_\tau]$ over all admissible stop times.

**Proof.** For any such $\tau$, telescoping and the measurability of
$\{\tau>s\}$ yield

$$
\mathbb E[G_\tau]=\mathbb E[G_m]+
\mathbb E\sum_{s=m}^{N-1}\mathbf1_{\{\tau>s\}}\mu_s.
$$

For $s<T_c$, $\mu_s>0$ and
$\mathbf1_{\{T_c>s\}}=1$. For $s\ge T_c$, $\mu_s\le0$
and $\mathbf1_{\{T_c>s\}}=0$. Accordingly,

$$
(\mathbf1_{\{T_c>s\}}-\mathbf1_{\{\tau>s\}})\mu_s\ge0
$$

on every path. Sum, take expectations, and apply the telescoping identity
to $T_c$. □

This is a condition on the **conditional drifts along sample paths**.
A single crossing of an empirical population mean curve does not establish
it. Theorem 4 is a special case of monotone stopping problems, whose
one-stage look-ahead conditions are discussed in
[Ferguson, Chapter 5](https://www.math.ucla.edu/~tom/Stopping/sr5.pdf).

**Corollary 4.1 (a sufficient structural condition).** Suppose cost is
constant $c_t=\lambda$, and, almost surely along every path,
$q_{t+1}\ge q_t$, $\alpha_{t+1}\le\alpha_t$, and
$\beta_{t+1}\ge\beta_t$ for $m\le t<N-1$, for specified versions
of the hazards. Then $\mu_t$ is nonincreasing and Theorem 4 applies.

**Proof.** Subtract adjacent drift expressions:

$$
\begin{aligned}
\mu_{t+1}-\mu_t={}&(1-q_{t+1})(\alpha_{t+1}-\alpha_t)
+\alpha_t(q_t-q_{t+1})\\
&-q_{t+1}(\beta_{t+1}-\beta_t)
-\beta_t(q_{t+1}-q_t).
\end{aligned}
$$

Every term is nonpositive because all probabilities lie in $[0,1]$.
With variable predictable cost an extra term $-(c_{t+1}-c_t)$
appears; nondecreasing $c_t$ is an additional sufficient condition. □

These sufficient assumptions are strong. They are not established by a
high correctness AUC, a good average utility curve, or a monotone fitted
hazard curve. A noisily estimated posterior may increase and decrease
after new observations even when the marginal accuracy improves.

### Exact counterexample 1: delayed repair

Use times $0,1,2$, constant cost $1/10$, and correctness probabilities
$q=(0,0,1)$. An exact-answer realization has a latent binary reference
$Y^*$, a surely incorrect placeholder at times 0 and 1, and a generator
that outputs $Y^*$ at time 2. No information at time 0 or 1 reveals which
reference answer will be returned. Then
$G=(0,-1/10,4/5)$ and $\mu_0=-1/10$.
The myopic boundary stops at 0 with reward 0. The optimal stop is 2 with
reward $4/5$. A second system with $q=(0,0,0)$ has the same current
$(q_0,\alpha_0,\beta_0,c_0)=(0,0,0,1/10)$ but optimally stops at 0.
Thus even the exact current hazard triple cannot identify a general
multi-step optimal policy. The example can be shifted to a floor $m=2$
without changing the comparison.

### Exact counterexample 2: a population curve misses adaptive value

Take $q_0=1/2$ and cost $1/10$. At time 1, a fair observed signal gives
branch A with $q_1=2/5$ or branch B with $q_1=3/5$. At time 2, set
$q_2=1$ on A and $q_2=0$ on B. Both unconditional one-step drifts
equal $-1/10$, and the deterministic-time expected rewards are
$1/2,2/5,3/10$. Yet continuing once, then continuing on A and stopping
on B, achieves

$$
\tfrac12(1-2/10)+\tfrac12(3/5-1/10)=13/20>1/2.
$$

For an exact-answer realization, let $Y^*\in\{0,1\}$ have prior probability
$1/2$ of 1, and use candidate answer 1 at times 0 and 1. Set
$\mathbb P(A,Y^*=1)=1/5$, $\mathbb P(A,Y^*=0)=3/10$,
$\mathbb P(B,Y^*=1)=3/10$, and $\mathbb P(B,Y^*=0)=1/5$.
At time 2 the generator returns $Y^*$ on branch A and a surely incorrect
placeholder on branch B. The controller uses only the observed branch
to make its time-1 decision. Thus the examples respect a common reference
answer and use no hindsight information to make decisions.

## 5. Partial observation and feature models

For a causal feature vector $Z_t=\phi_t(H_0,\ldots,H_t)$, define

$$
\widetilde q_t=\mathbb P(C_t=1\mid Z_t),\quad
\widetilde\alpha_t=\mathbb P(C_{t+1}=1\mid C_t=0,Z_t),\quad
\widetilde\beta_t=\mathbb P(C_{t+1}=0\mid C_t=1,Z_t).
$$

Their hazard expression equals
$\mathbb E[C_{t+1}-C_t\mid Z_t]$ exactly, subject to the same null-state
convention. This feature-conditional quantity need not equal the
history-conditional drift in Theorem 2. In particular, unrelated
$\sigma(Z_t)$ at different times need not form a filtration.

A Bellman equation on a compressed state requires sufficiency: conditional
on that state, the law of future observations, reward, and cost must not
depend on omitted history. Alternatively a correctly specified partially
observed Markov model can use the full posterior over its latent state as
the belief state. A scalar correctness posterior and two current hazards
do not by themselves provide that transition model. Without such
assumptions, a fitted hazard rule is an empirical myopic approximation.

Retrospective correctness classifiers, continuation predictors, and optimal
stopping policies are different estimands. Training with labels is allowed;
using those labels or future observations to decide on the same live
instance is not. A prefix-stable feature transform must be fitted before
evaluation or updated solely from information available by that prefix.
Full-family or full-trace normalizations may change earlier decisions when
future rows are added, even without directly using correctness labels.

## 6. Calibration and perturbation bounds

Discrimination is not probability accuracy. AUC is unchanged by a strictly
increasing transform, while the drift sign generally changes. Even marginal
calibration $\mathbb E[C_t\mid\widehat q_t]=\widehat q_t$ does not imply
$\widehat q_t=\mathbb E[C_t\mid\mathcal F_t]$: a constant $1/2$ score
is marginally calibrated in a balanced population whose available history
perfectly separates correct and incorrect cases.

**Proposition 5 (class weighting changes the population target).** For a
binary target with conditional probability $p\in(0,1)$, positive class
weights $w_1,w_0$, and an unconstrained probability prediction $r$, the
weighted Bernoulli log loss is uniquely minimized at

$$
r=\frac{w_1p}{w_1p+w_0(1-p)},\qquad
p=\frac{w_0r}{w_1(1-r)+w_0r}.
$$

**Proof.** Differentiate
$-w_1p\log r-w_0(1-p)\log(1-r)$; setting the derivative to zero yields
the first formula. Strict positivity of the second derivative gives
uniqueness. Algebra gives the inverse. □

`equation_analysis.py` and `trace_analysis.py` fit class-balanced
classifiers. Thus their uncorrected probability outputs are not guaranteed
to be target-distribution $q,\alpha,\beta$, even if ranking is useful.
For example, balanced weighting in a population with prevalence $1/10$
maps its constant posterior to $1/2$. With
$\alpha=1/50,\beta=1/200,\lambda=1/100$, changing only $q$ from
$1/10$ to $1/2$ changes drift from $3/400>0$ to $-1/400<0$.
The inverse formula describes a population optimum; model misspecification,
regularization, estimated weights, and distribution shift still require
independent probability validation. ECE and Brier scores are useful
diagnostics but do not provide uniform conditional error guarantees.

**Proposition 6 (drift perturbation and interval propagation).** If
$|\widehat q-q|\le\varepsilon_q$,
$|\widehat\alpha-\alpha|\le\varepsilon_\alpha$,
$|\widehat\beta-\beta|\le\varepsilon_\beta$, and
$|\widehat c-c|\le\varepsilon_c$, with all probabilities in $[0,1]$,
then

$$
|\widehat\mu-\mu|\le
2\varepsilon_q+(1-\widehat q)\varepsilon_\alpha
+\widehat q\varepsilon_\beta+\varepsilon_c.
$$

If simultaneous valid intervals are
$q\in[q_L,q_U]$, $\alpha\in[a_L,a_U]$,
$\beta\in[b_L,b_U]$, and $c\in[c_L,c_U]$, their exact rectangular
drift bounds, provided the three probability intervals are contained in
$[0,1]$, are

$$
L=(1-q_U)a_L-q_Ub_U-c_U,\qquad
U=(1-q_L)a_U-q_Lb_L-c_L.
$$

**Proof.** Subtract the true expression from the fitted expression as

$$
\widehat\mu-\mu=(1-\widehat q)(\widehat\alpha-\alpha)
-\widehat q(\widehat\beta-\beta)
-(\widehat q-q)(\alpha+\beta)-(\widehat c-c).
$$

Apply the triangle inequality and $\alpha+\beta\le2$.
The drift is increasing in $\alpha$ and decreasing in
$q,\beta,c$ on their domains, so the stated corners attain its
minimum and maximum on the rectangle. □

This propagation result assumes genuine interval coverage of the relevant
conditional probabilities. It cannot manufacture such intervals from AUC,
a chosen safety margin, cross-validation alone, or calibration plots.

## 7. What sequential uncertainty control can certify

**Proposition 7 (a precisely scoped first-crossing guarantee).** Suppose
$U_t$ is observable at decision $t$ and

$$
\mathbb P(\exists t\in\{m,\ldots,N-1\}:\mu_t>U_t)\le\delta.
$$

Let $\tau_U$ be the first $U_t\le0$ after the floor, with fallback
$N$. Then

$$
\mathbb P(\tau_U<N\text{ and }\mu_{\tau_U}>0)\le\delta.
$$

**Proof.** On the simultaneous-coverage event,
$\mu_{\tau_U}\le U_{\tau_U}\le0$ whenever $\tau_U<N$.
Every violation is therefore contained in the coverage-failure event. □

This certifies a nonpositive **one-step** drift, not optimality, retained
accuracy, or proximity to a hindsight oracle. A corresponding simultaneous
upper bound on $\Delta_t^*$ certifies the Bellman stop condition.
To interpret a bound on $\mu_t$ as a statement about the myopic boundary
one must separately impose the relevant structural assumptions.

For a probe construction, distinguish the pre-probe sigma-field
$\mathcal H_t$ from the post-probe decision information $\mathcal F_t$.
Let $\theta_t$ be an $\mathcal H_t$-measurable target, and assume the
$n_t$ probes $X_{t,i}\in[a,b]$ are conditionally independent given
$\mathcal H_t$, each with conditional mean $\theta_t$. Fix their count
before observing them. Conditional Hoeffding gives a bound for
$\theta_t$, not automatically for $\mu_t$:

$$
U_t^{\mathrm{pre}}=\overline X_t+(b-a)\sqrt{\frac{\log(1/\delta_t)}{2n_t}}.
$$

Taking $\delta_t=\delta/[r(r+1)]$ for decision index
$r=t-m+1\ge1$ gives $\sum_t\delta_t\le\delta$, so a union bound
establishes simultaneous coverage of the $\theta_t$ targets. Decisions can depend on earlier data;
independence between different decision times is unnecessary. Within-prefix
conditional independence and the correct conditional mean are essential.
Adaptively choosing the number of probes requires a bound also uniform in
probe count, or a further failure-budget allocation. The bounded-sample
result is due to
[Hoeffding (1963)](https://doi.org/10.1080/01621459.1963.10500830).

Probe observations and their cost must enter the actual decision
filtration. If they reveal information that changes the live conditional
continuation gain, a bound for the pre-probe conditioning target is not
automatically a bound for the post-probe target. Proposition 7 follows
from this construction only if $\theta_t=\mu_t$ after the probes, or if
an additional justified comparison supplies a bound for the post-probe
$\mu_t$. Shared unknown ground truth
can make apparently independent generations dependent after conditioning
only on the visible prefix. Verified probe labels are also not generally
available at deployment. An empirical-Bernstein formula requires its own
theorem, variance convention, and sampling assumptions; merely inserting a
sample variance does not prove those conditions. The modern confidence
sequence framework is developed by
[Howard et al. (2021)](https://doi.org/10.1214/20-AOS1991).

For completeness, a valid elementary e-process can be obtained when
$X_i\in[-1,1]$ is adapted to a sampling filtration $\mathcal G_i$
and the null satisfies $\mathbb E[X_i\mid\mathcal G_{i-1}]\ge0$
for every $i$. For any fixed $\eta\in[0,1)$,

$$
E_n^{(\eta)}=\prod_{i=1}^n(1-\eta X_i)
$$

is a nonnegative supermartingale starting at 1: its next conditional
multiplier has expectation at most 1. A fixed convex mixture of such
processes has the same property. Finite optional sampling at the first
threshold crossing, followed by monotone limits, gives

$$
\mathbb P(\sup_n E_n\ge1/\delta)\le\delta.
$$

This rejects that conditional-mean null for the specified sampling process.
It does not estimate a different instance's present continuation gain.
If $X_i=B$ for all $i$, with one shared fair sign $B$, all marginal
means are zero, yet $(1-X_i/2)$'s ten-factor product exceeds 20 on
the negative branch with probability $1/2$. Hence marginal means and
boundedness alone do not justify a 5% sequential guarantee.

In `trace_analysis.py`, empirical-Bernstein bounds and mixture products
are computed across labeled task rows within a step, then used for a common
step-level diagnostic. That construction is not an observable per-instance
confidence bound. Repeated runs of the same task also need dependence
handling for population inference. This note makes no claim that the
repository's raw fitted probabilities or historical detectors satisfy
Proposition 7's coverage premise.

## 8. Scope of the empirical claims

| Mathematical object | Required information or assumptions | Defensible interpretation |
| --- | --- | --- |
| Exact conditional hazard identity | Binary reward, correct conditioning, common transition sample | Algebraic one-step decomposition |
| Snell/Bellman policy | Conditional future law and all computation costs | Optimal finite-horizon causal stop |
| First nonpositive true drift | Persistent nonpositive drift after first crossing | Optimal under Theorem 4 |
| Fitted first nonpositive drift | Causal features and validated probability estimates | Empirical myopic stopping rule |
| First nonpositive upper bound | Simultaneous coverage of the stated target | Probabilistic one-step sign certificate |
| Hindsight best recorded step | Completed labels and trace | Infeasible upper benchmark |
| Correctness AUC | Held-out target labels and scores | Ranking quality on that population |

A hindsight maximum satisfies
$\mathbb E[\max_{m\le t\le N}(C_t-K_t)]
\ge\sup_\tau\mathbb E[C_\tau-K_\tau]$, because it dominates every
realized admissible stop reward. Its maximizing time generally fails the
stopping-time condition. Therefore a hindsight oracle gap includes
unavailable information, and a reported early/late stop relative to that
oracle is not a formal Type I error rate.

A minimum-step floor is a protocol constraint. It restricts the policy
class and can improve a particular fitted policy empirically; it cannot
improve the unconstrained optimum, because that optimum already includes
every policy respecting the floor. For positive constant step cost,
$q_t$ measures accuracy while $q_t-\lambda(t-m)$ measures utility;
their peaks need not coincide. Any savings claim must compare actual
executed tokens/calls with a defined baseline, not convert stopped step
indices into hardware savings without measurement.

The results are relative to the correctness target $C_t$. A passed grading
test suite does not prove every empirical label correct, and an exact-answer
benchmark theorem is not a theorem about open-ended semantic correctness.
The standard optimal stopping machinery here is not claimed as a novel
theorem or a universal law across model architectures.

## 9. Reproducible exact checks

Run from the repository root:

```powershell
python research/tests/test_mathematical_foundations.py
```

The standard-library suite uses rational arithmetic rather than Monte
Carlo. It checks finite conditional probability spaces, all causal policies
on small binary trees, Bellman values against exhaustive policy evaluation,
the qualified drift boundary, the counterexamples above, interval corners,
class-weight posterior distortion, and e-process dependence conditions.
These checks can expose algebra or implementation mistakes and supplement
the proofs; finite enumeration is not a proof of the general theorems.

## References verified against primary sources

- Thomas S. Ferguson, *Optimal Stopping and Applications*, UCLA electronic
  text, [Chapter 3](https://www.math.ucla.edu/~tom/Stopping/sr3.pdf) and
  [Chapter 5](https://www.math.ucla.edu/~tom/Stopping/sr5.pdf).
- Goran Peskir and Albert Shiryaev (2006), *Optimal Stopping and
  Free-Boundary Problems*, Birkhauser, Lectures in Mathematics ETH Zurich.
  [Publisher record](https://link.springer.com/book/10.1007/978-3-7643-7390-0),
  DOI 10.1007/978-3-7643-7390-0. General background, not the source of any
  claimed new LLM-specific theorem.
- Wassily Hoeffding (1963), "Probability Inequalities for Sums of Bounded
  Random Variables," *Journal of the American Statistical Association*
  **58**(301), 13-30.
  [DOI](https://doi.org/10.1080/01621459.1963.10500830).
- Steven R. Howard, Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon
  (2021), "Time-uniform, nonparametric, nonasymptotic confidence sequences,"
  *The Annals of Statistics* **49**(2), 1055-1080.
  [DOI](https://doi.org/10.1214/20-AOS1991),
  [authors' preprint](https://arxiv.org/abs/1810.08240).
