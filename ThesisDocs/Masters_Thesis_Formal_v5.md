# Cost aware stopping boundaries in reasoning language models

by

Aditya Bhatt

A thesis submitted to Johns Hopkins University in conformity with the requirements for the degree of Master of Science

Baltimore, Maryland

October 2026

# Abstract

Additional reasoning can repair an incorrect answer, replace a correct answer, or consume computation without sufficient improvement. This thesis formulates response-level stopping as a finite-horizon decision based on observable prefixes and hidden correctness. It derives the binary repair-corruption drift identity, applies the standard Bellman/Snell optimal stopping construction, gives a sufficient persistence condition for a drift-sign rule, and constructs exact counterexamples to unconditional myopic optimality. The experimental record separates a variable-horizon matrix from a standardized corpus of 144,440 saved rows, 28,888 trajectories, and 2,948 tasks. Recomputed cluster-based tables show model- and domain-dependent continuation value and matched effects of estimator, token-cap, and precision changes. The historical stacked ROC-AUC of 0.955156 is retained as a retrospective non-nested diagnostic rather than an online performance guarantee. A fitted prefix controller enforces a two-step floor and prevents future generation after stopping. In a newly executed evaluation on 100 GSM8K questions, its actual arm saves 56.51 percent of completion tokens with 7 correct answers versus 6 at the full horizon. On 20 traps it saves 52.11 percent, with one correct answer in each arm. Every learned run stops at step two; adaptive benefit and accuracy noninferiority remain unestablished. The results support protocol-specific, cost-sensitive stopping experiments while identifying limits of myopic rules, probability calibration, and accuracy preservation.

Research adviser: Zerotti Woods

Second reader: Moustapha Pemy

# Chapter 1 Introduction and motivation

## 1.1 Reasoning as a sequential allocation problem

Language models can answer mathematical and scientific questions by producing intermediate text before committing to an answer. Chain-of-thought prompting makes that intermediate work explicit [1]. The resulting sequence can contain useful calculations, checks, and repairs. It can also contain repetition, an incorrect reinterpretation of the problem, or a revision that replaces a correct candidate with an incorrect one. Additional generation therefore has an uncertain benefit and a measurable cost. A stopping policy must decide when the expected benefit of another increment justifies that cost.

The problem is especially clear when a model supplies several successive candidate answers to the same question. A trajectory may begin incorrectly, recover, and then become incorrect again. Its final answer alone conceals both the earlier opportunity to stop and the risk of stopping before a later repair. Comparing only the first and last answers similarly conceals the sequence of changes. The appropriate observational unit is a trajectory with ordered candidate answers, and the relevant decision is made using the prefix available at the moment of stopping.

This thesis studies that decision in a deliberately controlled setting. Each reasoning increment is a complete model response containing a candidate answer. The model can be asked to revise that response at the next increment. The experiments are therefore about repeated response-and-revision trajectories. They should not be interpreted as observations of every latent thought, every token of an uninterrupted internal reasoning process, or the model's unobserved cognition. This distinction determines what a controller can interrupt and how savings must be measured.

The motivating application is economical inference: spend computation when it improves the answer enough to be worthwhile, and terminate when further computation has low expected value. Economical inference is not identical to maximizing accuracy at any cost. A policy can improve a cost-sensitive utility while reducing accuracy; conversely, a policy can increase accuracy while increasing total computation. Both outcomes must be reported. Describing either one solely as a successful stop rate would hide the actual trade-off.

## 1.2 Operational meaning of overthinking

Let a saved candidate be correct when it satisfies the benchmark's frozen reference grader. A repair is an incorrect-to-correct transition between successive saved candidates. A corruption is a correct-to-incorrect transition. These terms describe an observable change in the answer label. They do not assert that the model was internally uncertain or that its narrative accurately reports its reasoning.

Three phenomena must be distinguished. First, **answer corruption** means that a later candidate is wrong after an earlier candidate was correct. Second, **unproductive continuation** means that additional generation fails to produce enough improvement to offset its cost. Third, **policy regret** means that a stopping decision has lower utility than a specified comparator. Corruption can occur without large regret when it is rare or inexpensive to repair. Unproductive continuation can occur even when accuracy is constant. Regret can arise because the policy stops before a repair, not only because it continues beyond a correct answer.

Published work on overthinking motivates the possibility that longer reasoning is inefficient [8]. This thesis asks a narrower, measurable question: on the recorded model-domain panels, when does another response increment have negative net value, and how much of that information can a causal controller use? The literature supplies context; the project's measured effects come from its own frozen artifacts.

At the population level, an accuracy curve can show where the average benefit of continuation changes sign. That curve is not a per-question stopping oracle. An early difficulty assessment, answer stability, or model confidence may differentiate questions that share the same average curve. Conversely, two indistinguishable prefixes may have different unobserved continuations. A scientifically useful controller must acknowledge both possibilities rather than treating the panel's best step as a universally correct stopping time.

## 1.3 Research questions

This study asks how repair and corruption explain changes in graded correctness and whether the resulting gains justify computation cost. It also asks when a one-step drift criterion is an optimal stopping rule. Chapter 2 distinguishes conditional hazards from joint transition frequencies and establishes the finite-horizon solution, a sufficient persistence condition, and counterexamples.

The empirical question concerns how model family, scale, domain, temperature, token cap, and numerical precision relate to the observed trade-off. The operational question is whether a prefix-only controller makes timely decisions and whether its loop avoids further generation. Chapters 3-5 distinguish descriptive associations, controlled contrasts, controller latency, live completion tokens, and replayed counterfactual costs.

## 1.4 Contributions

The contribution is a protocol-specific study of accuracy-cost trade-offs in complete response-and-revision trajectories. It combines established stopping theory with controlled empirical contrasts and a causal implementation. The findings depend on the declared generation, answer-selection, grading, and cost conventions; Section 1.6 compares them with prior work.

Chapter 2 applies finite-horizon Bellman/Snell theory [11], [12] to a specified decision filtration, selected answer, and cumulative computation cost. It derives conditional repair-corruption drift and distinguishes one-step gain from optimal multi-step continuation value. It proves a sufficient persistence condition for myopic optimality and gives counterexamples without that condition. Drift-error bounds and stake comparisons have explicit assumptions. These results apply established theory; they do not prove the fitted controller optimal.

The empirical work compares matched hazard estimators and paired token-cap and precision settings, evaluates detectors with task-level grouping, and classifies policy failures. It distinguishes ranking from stopping utility and future-informed diagnostics from deployable prefix predictors. In the recorded matched contrasts, more expressive estimators and calibration can reduce utility. These negative results identify limitations of particular stopping approximations.

The analysis keeps the variable-horizon boundary and five-step detector corpora separate, attaching roster, split, sample size, and horizon to each result. A prefix-only controller with a response floor ceases future generation. Separate fitting, calibration, and evaluation tasks yield a serializable probability model without runtime reference labels. Checks cover future-step independence, terminal behavior, token accounting, and adversarial trajectories; manifests identify executed sources and artifacts.

In both live panels, the learned rule stops every task at the minimum two-response budget. Observed savings therefore do not establish an adaptive advantage over that fixed budget. Low absolute accuracy and broad paired intervals limit accuracy-preservation claims. These negative findings accompany the working causal implementation. Comparative superiority over prior methods and a universally best reasoning horizon remain unestablished.

## 1.5 Evidence and reproducibility

Frozen manifests identify the selected data, label versions, and retained software evidence. They distinguish historical runtime records from the current workstation environment. Hashes establish artifact identity under the declared byte-normalization rule; they do not establish label validity, representative sampling, or bit-identical regeneration.

Historical tables retain their recorded labels. Later grader changes and reconstructed predictor targets have separate versioned results. Regression tests establish the checked behaviors, while semantic label validity requires independent adjudication. Appendix A maps the evidence and locates the electronic verification and reanalysis guide; Chapter 6 discusses these limits.

## 1.6 Related work and the thesis position

Chain-of-thought prompting studies how explicit intermediate reasoning can improve language-model performance [1]. Its benefit does not imply that every additional reasoning increment is valuable. Verifier training uses completed solutions to learn candidate selection [2], while process supervision provides feedback on intermediate mathematical steps [6]. The present endpoint is the selected answer at a saved response boundary. It does not certify every preceding mathematical statement, so the project's answer-level detectors are distinct from process verifiers. These approaches establish why intermediate work and evaluation can help; the stopping question concerns the conditional value of continuing a chosen protocol after a particular prefix.

Adaptive computation predates current language-model reasoning systems. Adaptive Computation Time learns a halting unit for repeated recurrent-state updates and includes a penalty for computation in its training objective [16]. PonderNet instead models the conditional probability of halting, optimizes expected prediction loss over possible halting steps, and regularizes the halting distribution toward a prior [17]. Both learn how much internal computation to perform within a trainable architecture. Their computational increments are state updates, whereas this thesis observes complete generated responses from a fixed generator. The connection is the allocation of effort according to the evolving state; the difference is the information available to the policy and the operation whose execution it can prevent.

Confident Adaptive Language Modeling, or CALM, chooses intermediate-layer exits during autoregressive decoding and calibrates local per-token decisions against sequence-level performance constraints [18]. The present controller instead decides whether to request another complete response. Layer execution and response generation measure different resources, and a response-boundary heuristic does not inherit CALM's calibration guarantees.

Self-consistency samples several reasoning paths and aggregates their answers [7]. Adaptive-Consistency makes that sampling budget variable using agreement among the samples already generated [19]. It is a direct precedent for prefix-dependent allocation, with the prefix consisting of completed sampled solutions. The present revision trajectory instead conditions later responses on earlier generated work and uses a declared causal answer selector. Independent-path aggregation and successive revision need not have the same transition law or correctness endpoint. Agreement features can be useful in either setting, but their acquisition cost and timing must be included. In particular, thirteen completed peer trajectories cannot be treated as a free observation in a cost comparison with a single generator.

The overthinking literature motivates inefficiency from excessive test-time reasoning [8]. More recent work addresses termination within a chain of thought. Liu and Wang examine answer convergence and evaluate answer-consistency stopping, changes to end-of-reasoning signals, and a supervised stopping predictor using internal activations [20]. Their work shows that stability is a serious candidate stopping signal, rather than a new idea introduced by this thesis. Stability, however, is a statement about answer agreement; the project separately measures correctness and the possibility of a later repair. Its complete-response boundaries and observable prefix summaries also differ from intervention within a single reasoning trace or access to internal activations.

REFRAIN combines a reflective-redundancy discriminator with a sliding-window upper-confidence-bound controller that adapts stopping thresholds [21]. This provides a recent training-free alternative to a fitted probability model. A redundancy score, a stable answer, and a probability of correctness are different quantities, and each policy must specify how its score determines a continuation decision. The present confidence-and-stability heuristic and trained drift rule are evaluated under their own fixed thresholds and response protocol; REFRAIN's reported results are contextual evidence, not estimates of this implementation's savings or accuracy.

Optimal stopping supplies the established mathematical distinction between immediate reward and conditional continuation value [11], [12]. A particularly close reasoning application is OS-Pruner, a July 2026 preprint that optimizes answer accuracy minus a token-length penalty after observing a reasoning prefix [22]. It intervenes at paragraph boundaries, elicits a final answer on termination, and trains a stopping policy using hidden-state features and precomputed rewards. It also presents a Bellman interpretation and distinguishes continuation value from a fixed correctness threshold. Thus, applying an accuracy-cost optimal-stopping objective to reasoning is already explicit in prior work. This thesis studies selected answers at complete-response boundaries and estimates one-step drift from observable summaries. Its fitted rule approximates a different target from a learned full continuation-value policy, and its finite counterexamples explain the extra conditions needed to connect myopic drift to optimal stopping.

Probability calibration is another established component. Platt's sigmoid mapping fits probabilities from classifier scores [23], and Guo et al. distinguish probability calibration from classification accuracy and evaluate post-processing methods for neural networks [24]. The project uses independently held-out task groups to fit its Platt calibrators and reports ranking and probability errors separately. A calibration fit on archived prefixes does not establish calibration under the live generator, parser, or stopping-induced state distribution. The actual live model is the instruction-tuned Qwen2.5-0.5B release [25]; identifying its model snapshot is essential because the wider archived model panels do not validate transport to every Qwen release or to larger reasoning-specialized models.

Sequential uncertainty theory addresses a further question: how can an interval or decision certificate retain its stated error control under repeated observation? Hoeffding's concentration bound concerns bounded observations under specified sampling assumptions [13]. Confidence sequences provide time-uniform coverage under the conditions of their construction [14]. Such results differ from fitting a calibrated probability score, and neither an arbitrary confidence threshold nor repeated use of an ordinary bootstrap interval inherits their guarantees. Chapter 2 states sufficient assumptions for its uncertainty results; Chapters 4 and 5 report empirical uncertainty at the declared experimental unit. This keeps a mathematical stopping certificate distinct from a descriptive interval for a measured policy contrast.

Prior work establishes learned halting, adaptive sampling, within-chain truncation, and accuracy-cost stopping. This thesis studies complete-response revision under a specified generation, grading, and cost protocol, comparing repair-corruption rules with fixed budgets. Its contribution is protocol-specific derivations and controlled findings, including failures. Prior methods were not reproduced under a shared protocol, so comparative superiority remains untested.

## 1.7 Thesis organization

Chapter 2 develops the stopping theory and Chapter 3 specifies the experiments. Chapters 4-5 report empirical and online results; Chapter 6 interprets them. Appendices supply the evidence map, claim scope, paired uncertainty, supporting analyses and diagrams.


# Chapter 2 Mathematical formulation and stopping theory

## 2.1 Probability model and admissible information

Fix a finite horizon $N$ and a deterministic earliest permitted stop
$m\in\{0,\ldots,N\}$. Work on a probability space
$(\Omega,\mathcal A,\mathbb P)$. Let $Y^*$ be the reference answer,
$H_t$ the information actually available after decision step $t$, and

$$
\mathcal F_t=\sigma(H_0,\ldots,H_t),\qquad
A_t=a_t(H_0,\ldots,H_t),\qquad
C_t=\mathbf1\{A_t=Y^*\}.
$$

For empirical evaluation, correctness is the versioned domain grader $C_t=g_d(A_t,Y^*)\in\{0,1\}$. Exact answer equality is the special case displayed above. The binary-reward arguments remain unchanged for this grading predicate. This notation identifies the measured endpoint; it does not certify semantic correctness of every stored label. Appendix E illustrates the information restrictions and delayed-repair counterexample.

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
$\mathcal B_t$ is the candidate set available at $t$. With the empirical grading predicate, the analogous immediate reward is $\max_{a\in\mathcal B_t}\mathbb E[g_d(a,Y^*)\mid\mathcal F_t]$.

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

## 2.2 Exact binary correctness drift

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

## 2.3 General finite-horizon optimal stopping

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

## 2.4 When a drift-sign boundary is optimal

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

## 2.5 Partial observation and feature models

For a causal feature vector $Z_t=\phi_t(H_0,\ldots,H_t)$, define

$$
\begin{aligned}
\widetilde q_t&=\mathbb P(C_t=1\mid Z_t),\\
\widetilde\alpha_t&=\mathbb P(C_{t+1}=1\mid C_t=0,Z_t),\\
\widetilde\beta_t&=\mathbb P(C_{t+1}=0\mid C_t=1,Z_t).
\end{aligned}
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

## 2.6 Calibration and perturbation bounds

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

## 2.7 What sequential uncertainty control can certify

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

## 2.8 Scope of the empirical claims

**Table 1. Mathematical objects and assumptions.**

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



# Chapter 3 Experimental setup and methods

## 3.1 Corpus separation

The variable-horizon response-and-revision matrix supports boundary and controlled-policy analyses: 75,965 sanitized trajectories and 798,770 raw saved step rows over 52 model-domain cells. Sanitization handles malformed fragments and inconsistent records, so raw row and eligible trajectory counts have different denominators. The standardized five-step detector corpus supports training and task-grouped scoring: 28,888 trajectories, 144,440 rows, and 2,948 unique task identifiers across 52 cells.

The collections overlap in scientific subject matter and include repeated benchmark questions. They are not independent replications and must not be summed into a single sample size. The earlier replay experiment also reuses a 1,500-trajectory Qwen2.5-7B/GSM8K cell from existing evidence. Its replayed decisions do not create new language-model generations.

**Table 2. Standardized five-step corpus.**

| Domain | Effective split | Tasks | Trajectories | Rows |
| --- | --- | --- | --- | --- |
| GSM8K | train | 1,000 | 8,064 | 40,320 |
| MATH-500 | test | 500 | 6,500 | 32,500 |
| ARC-Challenge | test | 1,000 | 8,500 | 42,500 |
| GPQA main | train (inferred; request test) | 448 | 5,824 | 29,120 |
| Total | four domains | 2,948 | 28,888 | 144,440 |

Different benchmark allocations explain the standardized corpus's larger task count. Counts use source-qualified trajectory keys; unequal model-domain sample sizes prevent deriving row totals from unique questions multiplied by models and steps.

## 3.2 Model roster

The canonical matrix includes thirteen model configurations: DeepSeek-R1-Distill-Qwen-1.5B and 7B; Qwen2.5-0.5B, 3B, 7B, 14B, and 32B Instruct [25]; InternLM3-8B-Instruct; Llama-3.1-8B-Instruct; Mistral-7B-Instruct-v0.3; Mistral-Small-Instruct-2409; Phi-4-mini-instruct; and Yi-1.5-9B-Chat. The standardized detector collection replaces InternLM3 with Qwen3.5-9B. These are distinct rosters, even though both have thirteen entries.

**Table 3. Model configurations. Source: cell metadata.**

| Model ID | Recorded scale | Membership |
| --- | --- | --- |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | Matrix + detector |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | Matrix + detector |
| internlm/InternLM3-8B-Instruct | 8B | Matrix only |
| meta-llama/Llama-3.1-8B-Instruct | 8B | Matrix + detector |
| mistralai/Mistral-7B-Instruct-v0.3 | 7B | Matrix + detector |
| mistralai/Mistral-Small-Instruct-2409 | 22B | Matrix + detector |
| microsoft/Phi-4-mini-instruct | 4B | Matrix + detector |
| Qwen/Qwen2.5-0.5B-Instruct | 0.5B | Matrix + detector |
| Qwen/Qwen2.5-14B-Instruct | 14B | Matrix + detector |
| Qwen/Qwen2.5-32B-Instruct | 32B | Matrix + detector |
| Qwen/Qwen2.5-3B-Instruct | 3B | Matrix + detector |
| Qwen/Qwen2.5-7B-Instruct | 7B | Matrix + detector |
| 01-ai/Yi-1.5-9B-Chat | 9B | Matrix + detector |
| Qwen/Qwen3.5-9B | 9B | Detector only |

The historical alias `mistral_small_24b_2409` denotes the 2409 model with a recorded specification of 22B. Tables use documented specifications; aliases locate source files. Saved manifests and trace metadata determine model inclusion.

The panel supports descriptive family and scale comparisons. Architecture, pretraining data, instruction tuning, distillation, and tokenization were not randomized, so family differences do not isolate parameter count. A common Qwen family reduces some confounding without constituting a controlled intervention on scale alone.

## 3.3 Benchmark tasks and splits

GSM8K provides grade-school mathematical word problems [2]; MATH provides competition-style mathematics [3]. ARC-Challenge contains difficult multiple-choice science questions [4], and GPQA contains graduate-level multiple-choice scientific questions [5]. The following protocol specifies this project's subsets and grading.

The canonical registry specifies GSM8K train, MATH test, ARC-Challenge test, and GPQA main train. The standardized collection uses the same first three splits. Its GPQA metadata records `test`, while the current loader always requests `train` and saved identifiers use `gpqa_main`. The historical loader revision is not recorded, so the effective split is inferred from the current implementation and stored identities; both requested and inferred splits are reported. Detector folds are distinct from benchmark split names: a task holdout drawn from benchmark training data is internal validation, not an official test-set evaluation.

SVAMP does not appear in the authoritative standardized tournament manifest. It is excluded from the reported four-domain totals. The complete prompt, task identifier, reference answer, split, and any deterministic choice shuffle must remain associated with a question. An MCQ label such as B only has meaning relative to the exact displayed ordering. The project stores or reconstructs that ordering before evaluating the option label.

## 3.4 Response and revision protocol

A saved step is a complete response generated under the collection's prompt contract. For the canonical registry, seed 7 and dataset shuffle seed 17 are recorded. The temperature panel uses 0.1, 0.6, and 1.0, and the common completion-token cap is 256. Response horizons are domain-specific: ten increments for GSM8K and GPQA, fourteen for MATH, and eight for ARC. Individual metadata remains authoritative when a cell records a deviation or recovery event.

The standardized detector corpus has five saved increments per trajectory. Its generation settings include temperature 0.6 and seed 7. A repeated call with a revision prompt is not equivalent to revealing another fragment of one uninterrupted chain of thought. The revision prompt itself can change the distribution of the next answer. The stopping target is therefore defined relative to this complete sequential protocol, including prompt construction and sampling.

The minimum admissible stopping step is two in the principal experiments. This floor is a design choice informed by development analyses. It is not a theorem that every question requires a revision. Some trajectories are correct only at step one, and the floor can exclude their best candidate. Allowing an answer-selection policy to return a previously saved candidate would define a different action space; such a policy must be compared separately.

At every decision, computation already incurred is sunk. The continuation decision should charge the incremental cost of future generation, not charge the prefix again. At evaluation, however, total policy cost includes every generated response up to the selected step. The distinction keeps the Bellman recursion and empirical cost accounting consistent.

## 3.5 Recorded signals and grading

Recorded signals can include candidate text, normalized answer, correctness, completion-token count, generation runtime, answer changes, log-probability summaries, entropy, hidden-state displacement, and peer agreement. Availability varies across experiments. A missing signal should be recorded as missing or excluded by a documented rule; treating it as a meaningful zero can alter a learned boundary.

Correctness labels use the benchmark's answer type. Numeric tasks use extraction and normalization; symbolic MATH tasks use the project's mathematical equivalence checks; multiple-choice tasks compare the parsed option with the exact displayed reference. Normalization must preserve information that matters to equivalence. For example, reducing an arbitrary plane equation to its right-hand side would identify distinct mathematical objects, and treating the phrase 'the third option' as the fraction one third would corrupt an MCQ answer.

The historical grader receipt covers thirty specific cases. Following the software review, the revised grader exposes ninety-eight tests covering normalization, mathematical equivalence, ambiguous option labels and bounded symbolic parsing. The receipts identify the tested source versions, and finite-system stopping tests separately check the minimum boundary and termination behavior. Regression coverage concerns these specified behaviors; validity across the full answer corpus requires independent adjudication.

The primary boundary tables use the archived `correct` values. The predictor training in Section 5.7 uses a separately versioned reconstruction and regrade of its selected candidates. A subsequent comparison of the historical and revised graders found no changes to the final-answer labels in the four live collections. These checks leave the original labels and results intact; a corpus-wide regrade would define a new analysis version with its own label changes and dependent estimates.

## 3.6 Statistical units and grouping

Rows from one trajectory are dependent, and trajectories from the same task share the question and reference. Task-held-out detector folds keep all rows for a question outside the detector's training data. A source-qualified key combines the cell with its run identifier to prevent accidental merging of trajectories from different model-domain cells. Checking raw identifier collisions is useful, but source qualification remains the defined identity rule.

The strongest stored task-grouped detector analyses use five outer folds. For meta-model stacking, every upstream fitted component must also respect the outer split: its parameters and any calibrated thresholds must be fit using outer-training data alone. Producing a base score out of fold somewhere in the development pipeline does not automatically make a later outer-fold stacked result nested. Chapter 4 explains why the historical 0.955156 result falls short of that standard.

Cluster bootstraps resample questions when the same question appears in several trajectories. Model-domain estimator comparisons use the recorded cell bootstrap for their reported interval. These intervals answer different questions. A question bootstrap estimates variability associated with the observed question population conditional on the fixed model panel; a cell bootstrap summarizes variability across the existing panel. Neither one is a multi-seed generation replication.

The archived N2/N3 probe and hazard harness uses run-group folds for upstream prediction, followed by task-group folds for threshold selection. Other-temperature trajectories of a question can enter upstream fitting, so the full pipeline is a controlled development contrast rather than an untouched-question evaluation. The strict tabular and text analyses and the new prefix predictor follow separately recorded task-disjoint contracts. Appendix D details retrospective prediction analyses; Section 5.7 specifies the predictor protocol. Appendix E illustrates offline fitting and runtime information flow.

## 3.7 Outcomes and computation measures

Accuracy is the mean binary correctness label at the policy's selected answer. Historical per-trajectory step utility is $U_i(\tau_i)=C_{i,\tau_i}-0.05(\tau_i-1)$, so the mandatory first response is a common baseline and each additional increment incurs the penalty. This penalty is a utility convention, not five percent of physical power consumption or a universal economic price. Token utility is $C_{i,\tau_i}-0.0002\sum_{s=1}^{\tau_i}L_{i,s}$, where $L_{i,s}$ is measured completion tokens; 250 completion tokens carry the same penalty as one response increment. Charging every generated token includes the first response, which cancels in paired policy differences on the same trace. The theoretical reward in Chapter 2 can subtract the common prefix cost without changing the optimal continuation decision.

Token savings are one minus the ratio of total stopped-policy completion tokens to total full-horizon completion tokens, calculated on a common task panel. The ratio of totals differs from the average per-question saving and weights long generations more heavily. Both the definition and denominator must accompany a percentage. Prompt tokens, scoring passes, peer generations, controller inference, and runtime may have separate costs and cannot be omitted from a claim about total compute savings.

ROC-AUC measures ranking, with half credit for tied scores. Brier loss measures squared error of probabilistic predictions. Neither alone establishes a stopping guarantee. Class-balanced raw probability outputs generally target a reweighted posterior, as derived in Chapter 2; their held-out ranking or marginal calibration does not validate natural-distribution conditional hazards. A detector with high pooled AUC can be poorly calibrated around the selected stopping threshold or fail on an underrepresented domain. The evaluation therefore includes micro, task-macro, domain-macro, worst-domain, and policy utility summaries.

## 3.8 Reproducibility and evidence freeze

Evidence manifests record exact local byte hashes and canonical LF content hashes, cross-check historical fingerprints, and identify supporting analyses. Verification also checks file sizes, row counts, and source-qualified trajectory membership. Appendix A maps the evidence and locates the electronic verification and reanalysis guide.

The original profile, `data_manifest_v1.json`, binds revision `09225c95`; `data_manifest_post_review_v1.json` binds grader, controller, and audit revisions at `6e4378be`. Both preserve the fifty-two standardized trace files, fitted predictor, and recorded generation outcomes. Each live-generation manifest binds the source copies executed for that run. Post-review replay checks revised decisions on saved prefixes; it produces no new generations or timing measurements.

Stored Blackwell metadata records a CUDA 13.0 runtime and PyTorch 2.13.0+cu130; the workstation lock describes a different, current environment. Compiler libraries, hardware, drivers, random-state handling, and nondeterministic kernels also affect repetition. The freeze supports inspection and re-analysis of stored results; bit-identical regeneration is not established.


# Chapter 4 Empirical evidence of overthinking

## 4.1 Population transitions and net value

The next-step accuracy change on a common transition-eligible panel equals repair frequency minus corruption frequency. Subtracting the step penalty gives empirical net gain. This exact decomposition of observed labels does not assume a model of internal reasoning.

**Table 4. GSM8K transition panel; 19,500 trajectories and 500 task clusters per row. Event cells give the count and at-risk denominator above the conditional probability; the final column gives net gain above its 95% interval.**

| Step | Accuracy | Repair events / at risk<br>Probability | Corruption events / at risk<br>Probability | Net gain<br>95% interval |
| --- | --- | --- | --- | --- |
| 2 | 0.2401 | 3,145/14,818<br>0.2122 | 1,170/4,682<br>0.2499 | +0.0513<br>[+0.0433, +0.0593] |
| 4 | 0.4020 | 1,830/11,661<br>0.1569 | 1,102/7,839<br>0.1406 | -0.0127<br>[-0.0186, -0.0065] |

At step two, repairs outnumber corruptions despite a lower conditional repair probability because more candidates are currently incorrect. The next-step gain after the 0.05 penalty is positive, approximately 0.0513 (Table 4).

At step four, accuracy increases by about 0.0373, below the 0.05 penalty, leaving net gain approximately -0.0127. Thus negative net gain can accompany improving accuracy.

Each panel comprises 500 task clusters and 19,500 trajectories. Its task-bootstrap intervals condition on the observed model and temperature panel and describe a population transition contrast, rather than certifying a particular answer.

![Four-domain population curves](images/thesis_v2/population_transitions.png)

**Figure 1.** Population accuracy and net gain. Bands are task-cluster bootstrap intervals. The horizontal line marks zero gain, not zero accuracy.

## 4.2 Model and domain differences

Selected cells show different empirical crossings. In Qwen2.5-7B/GSM8K, net gain at step four is positive, approximately 0.0193, and step five is negative, approximately -0.0253. In Qwen2.5-32B/MATH, step five is positive, approximately 0.0187, and step six is negative, approximately -0.0133. Each cell contains 1,500 trajectories over 500 questions. The intervals in the evidence tables quantify variation over those questions.

These crossings describe model- and domain-dependent continuation windows. Curves can return to positive gain, so neither a universal accuracy peak at step two or three nor optimality of the first negative gain follows. A final crossing selected after observing the complete curve is a retrospective description, not a live stopping time.

The design does not isolate parameter count from architecture and training differences. A longer revision window therefore characterizes the specific models, settings and questions. A universal scaling law requires additional models, independent generations and a prespecified relationship tested on new data.

## 4.3 Controlled estimator comparisons

**Table 5. Matched effects. Estimator intervals resample 52 cells; systems intervals resample 500 tasks.**

| Arm | Units | Matched effect | 95% interval | Endpoint |
| --- | --- | --- | --- | --- |
| N1 LOCO | 75,965 | +0.01210 | [+0.00522, +0.01907] | step utility per trajectory |
| N1 LOMO | 75,965 | +0.01286 | [+0.00683, +0.01911] | step utility per trajectory |
| N2a | 75,965 | -0.05694 | [-0.08716, -0.03145] | step utility per trajectory |
| N2b | 75,965 | -0.06165 | [-0.09168, -0.03547] | step utility per trajectory |
| N2c | 75,965 | +0.00331 | [+0.00062, +0.00633] | step utility per trajectory |
| N3 | 75,965 | +0.00781 | [+0.00276, +0.01355] | step utility per trajectory |
| N4 | 75,965 | +0.00210 | [+0.00049, +0.00403] | step utility per trajectory |
| N5 | 1,500 | +0.00000 | [+0.00000, +0.00000] | loss-risk difference |
| N6 | 1,500 | +0.14267 | [+0.11133, +0.17533] | accuracy difference |

Empirical-Bayes step hazards improve mean step utility by 0.00781 per trajectory over the matched cell-local logistic baseline, with a 52-cell interval approximately [0.00276, 0.01355]. Lagged logistic features and the step-two churn contrast also improve utility; Table 5 reports their mean controlled effects.

The historical '+593.55' is a summed utility difference over 75,965 trajectories; Table 5 normalizes it per trajectory. It measures the hazard-estimation contrast, rather than an accuracy percentage-point gain or an isolated causal effect of peer agreement.

The gradient-boosted and isotonic arms reduce utility by approximately 0.05694 and 0.06165 per trajectory against their logistic controls. Greater capacity or calibration can therefore worsen this stopping objective, without establishing universal overfitting or harm from calibration. The result depends on the estimator, target, sample size and policy threshold.

## 4.4 Token cap and numerical precision

The Mistral-Small-22B/GSM8K token-cap comparison, 256 versus 512 completion tokens, gives 454 losses among 1,500 trajectories in each arm: the hazard policy has lower utility than never stopping. With no discordant paired loss indicators, the difference is zero and its bootstrap is degenerate. Answers, token counts and timings may still differ; this binary endpoint cannot isolate truncation effects across other model-domain cells.

In the matched Qwen2.5-7B/GSM8K precision comparison, 418 of 1,500 step-two answers are correct under BF16 and 204 under 4-bit weights. The difference is 214/1,500, or 14.27 percentage points, with task interval approximately [11.13, 17.53] points. This model-, step- and implementation-specific absolute accuracy contrast is not a universal relative quantization loss.

## 4.5 Ranking and stopping performance

**Table 6. Stored causal detectors; common task-held-out development evaluation.**

| Configuration | Micro AUC | Task macro | Domain macro | Worst AUC | Step utility |
| --- | --- | --- | --- | --- | --- |
| Causal GRU | 0.8743 | 0.8214 | 0.8102 | 0.6313 | 0.3264 |
| Causal Fourier neural operator | 0.8708 | 0.8172 | 0.8045 | 0.6152 | 0.3289 |
| Selective SSM | 0.8663 | 0.8116 | 0.7989 | 0.6078 | 0.3326 |
| Causal residual TCN | 0.8647 | 0.8088 | 0.7965 | 0.6057 | 0.3315 |
| Causal RoPE transformer | 0.8638 | 0.8085 | 0.7953 | 0.6079 | 0.3331 |
| Truncated Beta mixture | 0.8630 | 0.8140 | 0.7951 | 0.6204 | 0.3311 |
| Five-expert causal MoE | 0.8583 | 0.8005 | 0.7882 | 0.5955 | 0.3310 |
| MoE plus hysteresis | 0.8583 | 0.8005 | 0.7882 | 0.5955 | 0.3217 |
| Task-grouped linear baseline | 0.8294 | 0.7773 | 0.7612 | 0.6000 | 0.3244 |

The stored prefix-safe causal gated recurrent unit (GRU) [9] has micro AUC 0.8743, task-macro AUC 0.8214 and domain-macro AUC 0.8102. GPQA is its worst domain at approximately 0.6313, limiting what the pooled score implies for deployment.

The causal transformer with rotary position embeddings (RoPE) [10] has lower pooled AUC than the GRU but slightly higher step utility (approximately 0.3331 versus 0.3264) and token utility (0.3634 versus 0.3631). Overlapping descriptive fold intervals leave a unique architecture winner unestablished; ranking and stopping utility can order configurations differently.

Task-macro AUC includes the 2,679 of 2,948 tasks whose evaluated rows contain both correct and incorrect answers. Within-task AUC is undefined for the others; accuracy and utility still include constant-label tasks.

On the standardized corpus, strict task-grouped tabular and text baselines have independently recomputed saved out-of-fold (OOF) AUCs of 0.849510 and 0.808976, including their original task-disjoint calibration. Appendix D gives the matched peer scores. That comparison adds dynamics above a baseline retaining vote, count and agreement inputs. Its scores and paired folds reproduce, but the executed runner and peer-feature module were not recovered, leaving partial executable-source provenance.

Selected-answer analyses target one causally chosen candidate per barrier over 14,740 decisions and 2,948 tasks, rather than every model-row; their AUCs are not a common benchmark with the row-level results. Configuration selection remains development work. Peer and selected-answer reporting calibrators fit previously computed OOF scores without fully nested outer-fold calibration, so calibrated Brier and ECE summaries are diagnostic. Appendix D retains the original held-out scores, sample sizes and paired contrasts; these are historical analyses, not fresh prospective stopping evidence.

![Causal detector domain scores](images/thesis_v2/causal_detector_domains.png)

**Figure 2.** Micro, domain-macro, and worst-domain ranking differ substantially. Stored grouped outputs, not live-run results.

## 4.6 The stacked retrospective diagnostic

The stacked hybrid's stored AUC is 0.955156 versus 0.943223 for its reduced-feature LightGBM control, a lift of 0.011933. The task-bootstrap interval [0.010384, 0.013480] describes this development difference, rather than the absolute stacked AUC.

The bidirectional sequence component reads all five saved steps and copies a trajectory score to earlier rows; centered smoothing also uses later observations. Meta-training is not confined to each scoring fold's outer training partition. These dependencies prevent treating the score as an online prefix predictor despite task-grouped scoring.

The control retains committee and independent-vote aggregates, so 'No Peers' is not a peer-free ablation. The stored contrast cannot isolate causal peer value or certify online correctness prediction.

All 10,000 bootstrap lifts were positive (100 percent), a resampling proportion rather than a conventional p-value, universal superiority or independent generation-seed replication. Repeated resampling of the development distribution cannot repair feature leakage or non-nested model selection.

## 4.7 Replay and failure interpretation

The Qwen2.5-7B/GSM8K replay reduces completion tokens from 827,804 to 377,960 (54.34 percent), while accuracy falls from 70.53 to 64.20 percent, a loss of 6.33 percentage points. Fitting and evaluating on the same 1,500 traces makes this a development diagnostic; Chapter 5 reports actual runtime experiments.

Across 75,965 sanitized variable-horizon trajectories, the archived hazard policy wins in 68,095 (89.64 percent), ties in 2,135 (2.81 percent) and loses in 5,735 (7.55 percent) against the full horizon under stored step utility. These are utility verdicts, not accuracy percentages. The audit reconstructs both binary endpoint labels from utility and cost, checks unique pairs and partitions every loss exactly once.

**Table 7. Mutually exclusive archived-policy loss patterns; 5,735 utility losses among 75,965 trajectories.**

| Observed loss pattern | Count | Share of losses |
| --- | --- | --- |
| No extracted candidate at the stop | 196 | 3.42% |
| An earlier decision-eligible candidate was correct | 460 | 8.02% |
| Step one was correct; no eligible pre-stop answer was correct | 600 | 10.46% |
| First eligible correct answer arrived one step after the stop | 1,706 | 29.75% |
| First eligible correct answer arrived at least two steps after the stop | 2,773 | 48.35% |

Every observed utility loss stops on an incorrect candidate and ends correctly. Table 7 applies mutually exclusive priority rules, with empty stops first. The 48.35-percent late-repair group first reaches an eligible correct answer at least two steps after stopping; variable horizons prevent interpreting this as a universal step-five outcome.

The taxonomy conditions on archived labels and the stored policy. Recomputation performs no fresh regrading or probe training, and excludes old classification-tag AUCs. These patterns cannot prove that all online predictors must fail. Section 3.5's grader checks cover specified behaviors; independent adjudication is needed to assess labeling errors' effect on the partition.

## 4.8 Empirical conclusions

Continuation value varies with model, domain and cost. Matched estimators can improve or worsen utility; ranking quality and replay savings alone do not establish accuracy preservation.


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

**Table 8. Controller-only latency on 100 saved math-question prefixes.**

| Scope | Decisions | Median ms | p99 ms | Maximum ms |
| --- | --- | --- | --- | --- |
| All measured decisions | 18880 | 0.00340 | 0.01300 | 0.43830 |
| confidence_stability_v1 | 8880 | 0.00420 | 0.01534 | 0.43830 |
| never | 10000 | 0.00340 | 0.01060 | 0.11090 |

Table 8 reports decision counts, median, upper quantiles and maximum on one hundred saved math-question prefixes. The under-ten-millisecond target concerns the observed maximum, not a fast average. Timing covers prepared controller observations, excluding language-model forward passes and external peers.

State-machine checks cover the floor, horizon, closure, future-step independence, parsing, token accumulation, peer barriers and cancellation. They verify the tested implementation behavior, rather than distributional accuracy preservation.

## 5.5 Live paired evaluation

**Table 9. Actual paired local generation on 100 GSM8K questions.**

| Endpoint | Measured value |
| --- | --- |
| Questions | 100 |
| Full-horizon accuracy | 6.00% |
| Stopped accuracy | 6.00% |
| Accuracy difference | +0.00 percentage points |
| Full-horizon completion tokens | 22,244 |
| Stopped completion tokens | 21,591 |
| Full-horizon prompt tokens | 142,498 |
| Stopped prompt tokens | 137,574 |
| Completion-token saving | 2.94% |
| Mean stopped step | 4.85 |
| Identical shared prefixes | 94 |
| Conservative paired 95% accuracy interval | [-4.29, +4.29] percentage points |
| Task-bootstrap saving interval | [0.85%, 5.63%] |

The live experiment freezes tasks, policy and generation settings before opening outcomes. Generation receives public tasks separately from the reference ledger; grading follows event recording, keeping benchmark labels outside runtime decisions.

Paired endpoints include final- and selected-answer accuracy, completion tokens, response increments, prompt tokens and wall time, with auxiliary or peer generation disclosed. Completion-token saving is the ratio of total actual token differences to total baseline tokens; wall time, energy and total inference cost are separate outcomes.

The model is Qwen2.5-0.5B-Instruct [25]. Previously observed corpus tasks remain development tasks despite a policy frozen before new generation. Prospective execution, task novelty and cross-model transfer are distinct properties; this run does not validate the Qwen2.5-7B replay or thirteen-model panel as a live system.

Both heuristic arms answer six of one hundred questions correctly. Six tasks stop early and ninety-four reach the horizon. Completion-token saving is 2.94 percent, with descriptive task-bootstrap interval [0.85, 5.63] percent (Table 9). The conservative iid-reference accuracy interval [-4.29, +4.29] percentage points allows meaningful losses; observed equality does not establish noninferiority without a prespecified tolerance.

Only 88 of 500 full-horizon responses satisfy the complete JSON contract. Carrying forward a nonempty candidate cannot recover arbitrary malformed reasoning, and all failures remain in the denominator. The weak six-percent baseline requires evaluation on a larger, protocol-capable model before useful deployed answer quality can be claimed.

Shared prefixes match on 94 of 100 pairs, limiting trajectory-specific counterfactual interpretation. Additional costs are prompt tokens (142,498 versus 137,574), model time (1,127.05 versus 1,099.21 seconds), padded prefill slots (231,856 versus 222,781) and decoding slots (57,856 versus 56,128). Prompt-plus-completion tokens fall 3.39 percent; no verifier or peers are generated. These costs are not interchangeable efficiency percentages.

## 5.6 Adversarial questions and traps

**Table 10. Actual paired local generation on the 20-question trap bank.**

| Endpoint | Measured value |
| --- | --- |
| Questions | 20 |
| Full-horizon accuracy | 5.00% |
| Stopped accuracy | 5.00% |
| Accuracy difference | +0.00 percentage points |
| Full-horizon completion tokens | 4,736 |
| Stopped completion tokens | 4,578 |
| Full-horizon prompt tokens | 27,641 |
| Stopped prompt tokens | 26,143 |
| Completion-token saving | 3.34% |
| Mean stopped step | 4.75 |
| Identical shared prefixes | 20 |
| Conservative paired 95% accuracy interval | [-19.68, +19.68] percentage points |
| Task-bootstrap saving interval | [0.00%, 8.80%] |

The twenty-question bank separates public prompts from independently checkable labels and short exact derivations. It covers percentage bases, harmonic average speed, conditional probability, dependent sampling, inclusive endpoints, reciprocal rates, exponent precedence, answer contracts and irrelevant details.

The handpicked bank probes these traps under the frozen model and prompt protocol; classic puzzle forms may occur in pretraining. It does not estimate a general worst-case attack rate. Malformed-input and forged-peer tests separately evaluate software behavior.

Both heuristic arms answer one of twenty questions correctly. Two stop early and eighteen reach the horizon; all twenty shared prefixes match. Strict JSON validity is fourteen of one hundred baseline responses. The 3.34-percent completion-token saving has interval [0.00, 8.80] percent; the conservative iid-reference accuracy interval is [-19.68, +19.68] percentage points (Table 10). This small, low-accuracy panel establishes neither accuracy preservation nor broad adversarial robustness.

A stable high-confidence error challenges the heuristic; a late repair challenges its stopping approximation. These adverse outcomes are retained without tuning a new threshold on the same questions and calling the adjusted result confirmatory.

Paired accuracy intervals subtract simultaneous 97.5-percent Clopper-Pearson bounds for improvement and worsening probabilities. A union bound gives at least 95-percent coverage under iid task-pair sampling without independence of discordance categories (Appendix C). Token intervals resample paired questions and recompute the ratio of total tokens. For the handpicked trap bank these are reference calculations and resampling diagnostics, not randomized adversarial-population coverage.

## 5.7 Deployable prefix probability model

The second runtime policy uses a pair of fitted, portable probability heads. One estimates correctness of the latest nonempty candidate available in the observed prefix, $\widehat q_t$. The other estimates correctness of the candidate selected by the same rule after one additional increment, $\widehat p_{t+1}$. Direct prediction of the latter avoids dividing the small sample into separate repair and corruption strata. The controller recomputes the estimated one-step gain as

$$\widehat\mu_t=(v+p)(\widehat p_{t+1}-\widehat q_t)-\lambda.$$

The frozen policy uses $v=1$, $p=0$, and $\lambda=0.05$. It stops at the first nonpositive estimated gain after the two-step floor, otherwise at step five. It does not query a next-step predictor at the terminal horizon. This is a myopic fitted rule. No persistence assumption or learned Bellman continuation law is established for these generations.

Training uses all 1,500 archived Qwen2.5-0.5B GSM8K-training and MATH trajectories, containing 7,500 rows. A fixed hash of public task identity allocates 902 tasks to fitting, 322 to calibration, and 276 to evaluation. Standardization and unweighted logistic coefficients are fit only on training tasks. Each one-dimensional Platt logistic calibrator [23] uses only calibration tasks. The evaluation tasks enter neither fit. All rows from a task share its partition, and the live hundred-task and twenty-trap public identities are explicitly excluded. Features use only current and prior observations, including step, token count, answer changes, thought-text summaries, and public domain.

The training targets match runtime answer selection. Reconstructing candidates with the declared parser produces 4,269 candidate-string disagreements with the saved candidates, so the reconstructed candidates are regraded against the archived references using the existing verification rule. Only 43 row-level correctness labels change. This new training-label version and a row-level audit are separate artifacts; the frozen source corpus and the primary empirical chapter's stored labels remain unchanged. No trajectory is filtered because of later candidate agreement. Selecting whole trajectories by agreement of all five responses would condition an early-step evaluation on future observations and change the eligible population.

**Table 11. Deployable probability heads on 276 archived held-out tasks; repeated rows are not independent trials.**

| Target | Held-out rows | AUC | Brier | ECE (10 bins) |
| --- | --- | --- | --- | --- |
| p_next | 1104 | 0.6923 | 0.0997 | 0.0209 |
| q_current | 1380 | 0.7101 | 0.0976 | 0.0222 |

The byte-checked artifact serializes scalers, coefficients and calibrators for standard-library evaluation without scikit-learn. Held-out Brier scores are 0.09760 for the current target and 0.09970 for the next, versus uncalibrated 0.09854 and 0.10068. These scores and ten-bin calibration summaries are marginal development diagnostics, not certified live-prefix conditional probabilities.

None of the 7,500 archived outputs meets the live JSON contract, leaving confidence and strict-parsing features without archive variation. The live prompt shifts that distribution, and its ledger includes emitted EOS tokens omitted by archived counts. Archive calibration therefore does not establish transport to the live protocol.

On 276 held-out archived tasks, the rule stops 275 times at step two and once at step three. Accuracy is 11.96 versus 10.51 percent at the full horizon; the task-bootstrap paired interval [-1.45, +4.35] percentage points includes zero. Replayed completion-token saving is 51.29 percent, interval [49.31, 53.09] percent. This near-fixed budget demonstrates little additional adaptation, but provides a reproducible causal artifact for prospective evaluation.

## 5.8 Actual trained-policy evaluation

**Table 12. Actual trained-policy generation paired with the existing 100-question live baseline.**

| Endpoint | Measured value |
| --- | --- |
| Questions | 100 |
| Full-horizon accuracy | 6.00% |
| Stopped accuracy | 7.00% |
| Accuracy difference | +1.00 percentage points |
| Full-horizon completion tokens | 22,244 |
| Stopped completion tokens | 9,674 |
| Full-horizon prompt tokens | 142,498 |
| Stopped prompt tokens | 43,874 |
| Completion-token saving | 56.51% |
| Mean stopped step | 2.00 |
| Identical shared prefixes | 100 |
| Conservative paired 95% accuracy interval | [-4.27, +6.21] percentage points |
| Task-bootstrap saving interval | [54.83%, 58.01%] |

**Table 13. Actual trained-policy generation paired with the existing trap-bank live baseline.**

| Endpoint | Measured value |
| --- | --- |
| Questions | 20 |
| Full-horizon accuracy | 5.00% |
| Stopped accuracy | 5.00% |
| Accuracy difference | +0.00 percentage points |
| Full-horizon completion tokens | 4,736 |
| Stopped completion tokens | 2,268 |
| Full-horizon prompt tokens | 27,641 |
| Stopped prompt tokens | 8,115 |
| Completion-token saving | 52.11% |
| Mean stopped step | 2.00 |
| Identical shared prefixes | 20 |
| Conservative paired 95% accuracy interval | [-19.68, +19.68] percentage points |
| Task-bootstrap saving interval | [45.86%, 57.17%] |

The trained policy generates its own stopped trajectories against the previously executed full-horizon baseline. Hashes bind baseline manifests, model bytes, prompts, adapter, batch order and results. Predictor and policy are frozen before generation; live and trap outcomes enter no coefficient, calibration, feature, cost or threshold fitting. This remains a development study of one small model and two panels, not deployment of the retrospective stack or a per-instance safety guarantee.

On all one hundred GSM8K questions, the trained policy stops at step two with identical shared prefixes. It answers seven correctly versus six at the full horizon and saves 56.51 percent of completion tokens (Table 12; saving interval [54.83, 58.01] percent). The conservative accuracy-change interval [-4.27, +6.21] percentage points establishes neither superiority nor noninferiority. Uniform stopping makes the realized policy equivalent to a fixed-two-step budget on this panel, with no demonstrated adaptive advantage.

All twenty trained trap runs also stop at step two with matching prefixes and one correct answer in each arm. Completion-token saving is 52.11 percent (Table 13; interval [45.86, 57.17] percent), while the handpicked-bank iid-reference accuracy bounds remain [-19.68, +19.68] percentage points. Large savings accompany weak absolute accuracy, without establishing general adversarial robustness.

Prompt-plus-completion savings are 67.50 percent on the main panel and 67.93 percent on traps; Tables 12-13 retain prompt counts. Model times are 343.25 versus 1,127.05 seconds and 74.12 versus 245.14 seconds, respectively. Padded prefill and decoding slots are recorded separately, with no verifier or peer generations. These environment-specific costs do not guarantee equivalent reductions under another model or batching regime.

On one hundred baseline prefixes repeated twenty times, 4,000 trained-controller decisions have median 0.059 ms, p99 0.3143 ms and maximum 0.9369 ms. Timing includes feature extraction, both heads, calibration, validation and drift computation, excluding model loading, generation, tokenization and peer waits. The observed maximum meets the under-ten-millisecond decision target.

## 5.9 Replay and Pareto analysis

**Table 14. Frozen variants replayed on 1,500 saved trajectories. These are development comparisons.**

| Policy | Accuracy | Completion tokens | Saving | Nondominated |
| --- | --- | --- | --- | --- |
| never | 9.53% | 418,845 | 0.00% | False |
| fixed_2 | 9.60% | 204,011 | 51.29% | True |
| fixed_3 | 9.93% | 278,444 | 33.52% | True |
| fixed_4 | 9.73% | 349,254 | 16.61% | False |
| confidence_80 | 9.60% | 374,129 | 10.68% | False |
| confidence_90 | 9.60% | 377,172 | 9.95% | False |
| confidence_95 | 9.53% | 382,981 | 8.56% | False |

Replay compares frozen heuristic, fixed-step and full-horizon policies on common saved continuations. Counterfactual validity requires unchanged preceding generation; replay cannot measure live overhead, asynchronous batching effects or energy.

The accuracy-cost plane pairs accuracy with measured or replay-counted completion tokens. A policy is dominated when another has no lower accuracy and no greater cost, improving at least one. Choosing within the empirical Pareto set requires a utility weight or accuracy constraint.

![Replay accuracy cost comparison](images/thesis_v2/replay_pareto.png)

**Figure 3.** Development replay accuracy against completion-token cost. Labels identify fixed-step and confidence variants; observed nondominance is sample-specific.

![Actual live accuracy and completion-token comparison](images/thesis_v2/actual_live_pareto.png)

**Figure 4.** Actual generated arms on two development panels. Each learned arm reuses its panel's actually generated full-horizon baseline and stops at step two on every task. The plotted point therefore supplies no evidence of adaptation beyond a fixed-two-step budget. Point estimates omit uncertainty, which is reported in Tables 9, 10, 12 and 13.

The displayed set is panel-specific and may change with tasks, generation seeds or response budgets. Independent evaluation must freeze policy selection and distinguish a changed maximum horizon from a changed stopping rule within it.


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


# References

[1] Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q. V., and Zhou, D. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS 35. https://arxiv.org/abs/2201.11903

[2] Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. (2021). Training Verifiers to Solve Math Word Problems. https://arxiv.org/abs/2110.14168

[3] Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. (2021). Measuring Mathematical Problem Solving With the MATH Dataset. Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, 1. https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html

[4] Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. (2018). Think You Have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. https://arxiv.org/abs/1803.05457

[5] Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., Michael, J., and Bowman, S. R. (2023). GPQA: A Graduate-Level Google-Proof Q&A Benchmark. COLM 2024. https://arxiv.org/abs/2311.12022

[6] Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. (2023). Let's Verify Step by Step. ICLR 2024. https://arxiv.org/abs/2305.20050

[7] Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., and Zhou, D. (2022). Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023. https://arxiv.org/abs/2203.11171

[8] Chen, X., Xu, J., Liang, T., He, Z., Pang, J., Yu, D., Song, L., Liu, Q., Zhou, M., Zhang, Z., Wang, R., Tu, Z., Mi, H., and Yu, D. (2024). Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs. https://arxiv.org/abs/2412.21187

[9] Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., and Bengio, Y. (2014). Learning Phrase Representations Using RNN Encoder-Decoder for Statistical Machine Translation. EMNLP, 1724-1734. https://aclanthology.org/D14-1179/

[10] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., and Liu, Y. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. https://arxiv.org/abs/2104.09864

[11] Ferguson, T. S. Optimal Stopping and Applications. UCLA electronic text. Chapters 3 and 5. https://www.math.ucla.edu/~tom/Stopping/sr3.pdf and https://www.math.ucla.edu/~tom/Stopping/sr5.pdf

[12] Peskir, G., and Shiryaev, A. (2006). Optimal Stopping and Free-Boundary Problems. Birkhauser, Lectures in Mathematics ETH Zurich. https://doi.org/10.1007/978-3-7643-7390-0

[13] Hoeffding, W. (1963). Probability Inequalities for Sums of Bounded Random Variables. Journal of the American Statistical Association 58(301), 13-30. https://doi.org/10.1080/01621459.1963.10500830

[14] Howard, S. R., Ramdas, A., McAuliffe, J., and Sekhon, J. (2021). Time-uniform, nonparametric, nonasymptotic confidence sequences. Annals of Statistics 49(2), 1055-1080. https://doi.org/10.1214/20-AOS1991

[15] Clopper, C. J., and Pearson, E. S. (1934). The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial. Biometrika 26(4), 404-413. https://doi.org/10.1093/biomet/26.4.404

[16] Graves, A. (2016). Adaptive Computation Time for Recurrent Neural Networks. arXiv:1603.08983. https://arxiv.org/abs/1603.08983

[17] Banino, A., Balaguer, J., and Blundell, C. (2021). PonderNet: Learning to Ponder. 8th ICML Workshop on Automated Machine Learning; arXiv:2107.05407. https://arxiv.org/abs/2107.05407

[18] Schuster, T., Fisch, A., Gupta, J., Dehghani, M., Bahri, D., Tran, V. Q., Tay, Y., and Metzler, D. (2022). Confident Adaptive Language Modeling. Advances in Neural Information Processing Systems 35. https://papers.neurips.cc/paper_files/paper/2022/hash/6fac9e316a4ae75ea244ddcef1982c71-Abstract-Conference.html

[19] Aggarwal, P., Madaan, A., Yang, Y., and Mausam. (2023). Let's Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with LLMs. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 12375-12396. https://aclanthology.org/2023.emnlp-main.761/

[20] Liu, X., and Wang, L. (2025). Answer Convergence as a Signal for Early Stopping in Reasoning. Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 17896-17907. https://aclanthology.org/2025.emnlp-main.904/

[21] Sun, R., Cheng, W., Li, D., Chen, H., and Wang, W. (2026). Stop When Enough: Adaptive Early-Stopping for Chain-of-Thought Reasoning. Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 27250-27268. https://aclanthology.org/2026.acl-long.1256/

[22] Ehab, M., El Gadarri, A., Farias, V. F., Jozefiak, A., and Moallemi, C. C. (2026). OS-Pruner: Pruning Chains-of-Thought of Reasoning Models via Optimal Stopping. arXiv:2607.11089, preprint. https://arxiv.org/abs/2607.11089

[23] Platt, J. C. (2000). Probabilities for SV Machines. In A. J. Smola, P. Bartlett, B. Schölkopf, and D. Schuurmans (eds.), Advances in Large-Margin Classifiers, 61-74. MIT Press. https://doi.org/10.7551/mitpress/1113.003.0008

[24] Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. Proceedings of the 34th International Conference on Machine Learning, Proceedings of Machine Learning Research 70, 1321-1330. https://proceedings.mlr.press/v70/guo17a.html

[25] Qwen Team. (2024). Qwen2.5 Technical Report. arXiv:2412.15115, first submitted December 19, 2024; revised January 3, 2025. https://arxiv.org/abs/2412.15115. Official Qwen2.5-0.5B-Instruct model card (accessed October 4, 2026): https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct


# Appendix A Reproducibility and evidence sources

The original research profile is fixed at revision `09225c95c676ab3cae1c4b9586d74f7f803f519f` by `data_manifest_v1.json`. The post-review profile at `6e4378bef9a98037c20d1381eb9c7b61462a6578` uses `data_manifest_post_review_v1.json` for revised grading, controller validation and behavior checks. Both preserve the collected corpus, fitted predictor and recorded generation outcomes. Each live manifest binds the sources actually executed in its `locked_code` directory.

**Table 15. Claim evidence sources.**

| Claim or output | Authoritative repository source |
| --- | --- |
| Tournament corpus | data_manifest_v1.json and ultimate_tournament_manifest.json |
| Boundary and matched effects | research/outputs/thesis_v1/evidence/*.csv |
| Proofs and exact finite checks | research/mathematical_foundations.md; test_mathematical_foundations.py |
| Controller execution | online_stopping_controller.py; online_generation.py |
| Latency and replay | research/outputs/semester2/online_stopping_20261002/ |
| Adversarial bank | research/adversarial_tasks_v1.jsonl; adversarial_gold_v1.jsonl |
| Archived policy failures | research/reports/thesis_failure_audit_v1/audit_summary.json |
| Deployable prefix predictor | research/outputs/semester2/prefix_model_v1/ |
| PDF authoring | tools/build_master_thesis.py; tools/render_thesis_math.mjs |

The electronic supplement, `ThesisDocs/formal/supplements/v5/reproduction_and_history.txt`, gives commands for detached checkouts, data verification and isolated reanalysis. Verification uses `tools/freeze_research_data.py`; `tools/recompute_thesis_evidence.py` reconstructs tables, and `tools/verify_post_review_behavior.py` checks saved controller behavior. Analysis outputs go to new scratch directories, and uncertainty analysis operates on copied live panels.

The recorded receipts distinguish thirty historical grader cases from ninety-eight revised tests, and 107 passing historical-suite tests from 243 post-review repository tests. Saved-prefix verification reproduces 480 decision histories across four live collections; reused baseline rows are not additional generations. These counts identify their source versions, rather than measuring corpus-wide label validity.

The fifty-two standardized trace files are mandatory freeze inputs. Supporting evidence includes the variable-horizon matrix, paired study arms, stored predictions and live event ledgers. Exact-byte and canonical-LF hashes identify selected evidence; they do not establish bit-identical regeneration of the partially recorded historical GPU environment. The current authoring manifests separately identify the rendering environment and exact PDF bytes.

# Appendix B Claim scope and mathematical review

**Table 16. Claim scope and required evidence.**

| Claim | Evidence required | Supported interpretation |
| --- | --- | --- |
| Binary repair-corruption identity | Common transition panel or conditional probability proof | Exact decomposition in Chapter 2 |
| General optimal stopping | Full conditional continuation law and all costs | Finite-horizon theorem; not a calibrated learned deployment |
| First drift crossing is optimal | Pathwise persistence after the crossing | Conditional theorem; not established universally by the empirical curves |
| 0.955156 AUC | Stored historical tournament result | Retrospective non-nested diagnostic |
| Causal detector ranking | Task-grouped causal sequence outputs | Internal development evaluation |
| Replay completion-token saving | Frozen traces and selected prefixes | Counterfactual development quantity |
| Live completion-token saving | Actual generation events from both arms | Model-, task-, and protocol-specific measurement |
| No meaningful accuracy loss | Prespecified tolerance and adequate paired confidence interval | Not inferred merely from observed equality |
| Grader regression coverage | Versioned test inputs and execution receipts | Thirty historical cases and ninety-eight revised tests cover specified parser behaviors; corpus-wide validity requires separate adjudication |

# Appendix C Paired accuracy uncertainty

For iid task pairs, let $I$ indicate an incorrect baseline answer paired with a correct active answer, and $W$ a correct baseline answer paired with an incorrect active answer. The accuracy difference is $\delta=\pi_I-\pi_W$, where $\pi_I=\Pr(I=1)$ and $\pi_W=\Pr(W=1)$. The indicators are mutually exclusive within a task; their independence is not assumed.

Across $n$ independent pairs, each marginal discordance count is binomial. Exact two-sided 97.5% Clopper-Pearson intervals $[L_I,U_I]$ for $\pi_I$ and $[L_W,U_W]$ for $\pi_W$ each have noncoverage probability at most 0.025 [15]. A union bound gives simultaneous coverage at least 0.95 regardless of dependence between the counts. Subtraction yields

$$\delta\in[L_I-U_W,\ U_I-L_W].$$

This is the conservative 95% paired interval in the live tables. Exact multinomial enumeration supplements the binomial coverage argument. With no discordances, both lower bounds are zero and both upper bounds equal $1-0.0125^{1/n}$. Observed equality therefore leaves nonzero uncertainty; no noninferiority tolerance or hypothesis was prespecified.

The twenty handpicked traps form a fixed challenge bank, so the iid-reference interval gives no randomized adversarial-population coverage. Completion-token intervals bootstrap paired tasks and the ratio of total token differences to baseline tokens; these are descriptive resampling intervals.

# Appendix D Retrospective prediction analyses

Table 17 distinguishes row-level correctness from correctness of one causally selected answer per closed barrier. The endpoints and eligible populations differ, so the scores do not form a common architecture ranking.

**Table 17. Additional retrospective prediction analyses.**

| Analysis | Units and tasks | Saved OOF AUC | Endpoint |
| --- | --- | --- | --- |
| Strict tabular | 144,440 rows; 2,948 tasks | 0.849510 | Current-candidate correctness |
| Strict text | 144,440 rows; 2,948 tasks | 0.808976 | Current-candidate correctness |
| Anonymous peer features | 144,440 rows; 2,948 tasks | 0.954664 | Current-candidate correctness |
| Matched anonymous baseline | Same rows and tasks | 0.945336 | Current-candidate correctness |
| Fixed thirteen-member peers | 98,280 rows; 1,512 tasks | 0.940009 | Current-candidate correctness |
| Matched fixed-roster baseline | Same rows and tasks | 0.931266 | Current-candidate correctness |
| Selected answer without timing | 14,740 decisions; 2,948 tasks | 0.934350 | Selected-candidate correctness |
| Selected causal dynamics | 14,740 decisions; 2,948 tasks | 0.937037 | Selected-candidate correctness |

The strict baseline and paired peer scores were recomputed from saved predictions, with source hashes and task-fold memberships checked. The anonymous peer lift is 0.009328, with paired task-bootstrap interval [0.008028, 0.010639]; the fixed-roster lift is 0.008743, with interval [0.006653, 0.010882]. These intervals describe the archived panels. Selected-answer source and five-fold checkpoint coverage were also checked.

Strict baseline probabilities include the original task-disjoint calibration. Peer and selected-answer scores use original uncalibrated outputs; this check applies no additional reporting calibration. The later reporting calibrators use previously computed out-of-fold scores across folds and are not fully nested, so their Brier and ECE summaries are diagnostic.

Strict-baseline and selected-answer source hashes match retained code. The legacy peer runner and feature-module hashes do not match any recovered Git version, including line-ending reconstructions, although their common base script resolves to the recorded July commit. The saved arrays and paired folds are auditable; exact reproduction of the executed peer-feature pipeline remains incomplete. This qualification covers both peer populations in Table 17.

The historical peer candidate has no assigned stopping threshold. Its full-horizon collector and small smoke ledgers establish neither calibrated prospective fleet stopping nor avoided peer generation. The electronic supplement records the development history and distinguishes reused predictions from independently generated trajectories.

# Appendix E Information flow and delayed repair diagrams

![Offline fitting and runtime information flow](images/thesis_v4/information_flow.png)

**Figure 5. Offline fitting and runtime information flow.** Reference labels enter fitting and evaluation offline; runtime decisions receive observed prefixes and frozen parameters. The continuation loop incurs additional generation cost. This schematic describes the information contract and does not assert a Bellman-optimal fitted controller or an executed peer fleet.

The fitted single-model controller estimates current and next selected-answer correctness and uses a one-step drift rule, with a declared response floor and horizon. The general theorem instead uses conditional multi-step continuation value.

![Decision tree for the delayed-repair counterexample](images/thesis_v4/delayed_repair_tree.png)

**Figure 6. Delayed repair defeats a myopic stopping rule.** The exact counterexample has horizon two, earliest permitted stop zero and incremental cost 0.10. Correctness follows zero, zero, one. Immediate drift at zero is -0.10, whereas continuation to the horizon yields reward 0.80. This abstract floor differs from the live experiments' floor of two; the diagram illustrates the Chapter 2 counterexample rather than new generated data.
