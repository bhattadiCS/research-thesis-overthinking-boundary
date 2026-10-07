# Chapter 2 Mathematical formulation

## 2.1 Information, answers and computation cost

Fix a finite integer response horizon $N\ge0$ and earliest permitted stop $m\in\{0,\ldots,N\}$ on a probability space $(\Omega,\mathcal A,\mathbb P)$. Let $H_t$ contain the observations available after response $t$, and define $\mathcal F_t=\sigma(H_0,\ldots,H_t)$. The selected candidate $A_t=a_t(H_0,\ldots,H_t)$ is the answer the specified measurable causal selector emits if stopping at $t$. For domain $d$, reference $Y^*$ and measurable binary grader $g_d$, put

$$C_t=g_d(A_t,Y^*)\in\{0,1\},\qquad q_t=\mathbb E[C_t\mid\mathcal F_t].$$

The selector and grader are fixed within a comparison. Changing either changes the endpoint. Offline reference labels, future revisions and future peer outputs are excluded from runtime inputs. Completed peer or verifier observations may enter the filtration only after their costs have been incurred.

Statistical information does not model computational difficulty. If the reference is a deterministic measurable function of a fully observed task, correctness can be $\mathcal F_t$-measurable and $q_t=C_t$. Nondegenerate conditional uncertainty requires a latent-reference model or a coarser filtration to which admissible policies are restricted. A fitted prefix score does not establish the true posterior given all task semantics.

An admissible stop time satisfies $m\le\tau\le N$ and $\{\tau\le t\}\in\mathcal F_t$. Let $K_t$ be nonnegative, nondecreasing, adapted and integrable cumulative cost, including every executed generation, peer and probe call. The objective is

$$\sup_{\tau}\mathbb E[C_\tau-K_\tau].$$

Define the observable reward $G_t=q_t-K_t$ and expected incremental cost $c_t=\mathbb E[K_{t+1}-K_t\mid\mathcal F_t]$. Previously incurred common costs are sunk for the continuation decision, while total reported policy cost still includes them.

**Lemma 1.** For every admissible stop, $\mathbb E[C_\tau-K_\tau]=\mathbb E[G_\tau]$.

**Proof.** Since $\{\tau=t\}\in\mathcal F_t$, conditional expectation gives $\mathbb E[\mathbf1_{\{\tau=t\}}C_t]=\mathbb E[\mathbf1_{\{\tau=t\}}q_t]$. Sum over the finite set $t=m,\ldots,N$ and subtract expected cost. □

A potential full trajectory can support replay only when stopping leaves the law of its preceding prefixes unchanged. Shared-prefix agreement is therefore checked in the live experiments, rather than inferred from deterministic sampling alone.

## 2.2 Repair-corruption drift

For $t<N$, define conditional repair and corruption probabilities

$$\alpha_t=\mathbb P(C_{t+1}=1\mid C_t=0,\mathcal F_t),\qquad \beta_t=\mathbb P(C_{t+1}=0\mid C_t=1,\mathcal F_t).$$

Where an at-risk event has conditional probability zero, its hazard can be assigned an arbitrary version; the corresponding weighted term is zero. Splitting next-step correctness by current correctness gives

$$\mathbb E[C_{t+1}\mid\mathcal F_t]=(1-q_t)\alpha_t+q_t(1-\beta_t).$$

The tower property also gives $\mathbb E[q_{t+1}\mid\mathcal F_t]=\mathbb E[C_{t+1}\mid\mathcal F_t]$. Consequently,

$$\mu_t:=\mathbb E[G_{t+1}-G_t\mid\mathcal F_t]=(1-q_t)\alpha_t-q_t\beta_t-c_t.$$

This is an exact binary-reward identity. Empirical repair probability divides repair events by currently incorrect candidates; corruption probability divides by currently correct candidates. Dividing each event count by all eligible trajectories instead gives its joint frequency. Their frequency difference is the accuracy change on a common panel. A higher conditional corruption hazard can coexist with more repairs when incorrect candidates are more numerous.

The historical step penalty is $0.05$ per additional response. It defines a utility convention, rather than a monetary or energy conversion. Positive accuracy change can therefore accompany negative net utility gain. Population accuracy and estimated drift also differ from a conditional decision for one prefix.

## 2.3 Finite-horizon optimum

The standard Bellman/Snell construction [Ferguson], [Peskir2006] sets

$$V_N=G_N,\qquad V_t=\max\{G_t,\mathbb E[V_{t+1}\mid\mathcal F_t]\},\quad m\le t<N.$$

**Theorem 1.** The earliest permitted time with $V_t=G_t$, denoted $\tau_V$, is optimal and attains $\mathbb E[V_m]$.

**Proof.** Backward induction bounds the conditional reward of any admissible continuation from $t+1$ by $V_{t+1}$. At $t$, stopping yields $G_t$ and continuing yields at most its conditional expectation; their maximum is $V_t$. The bound is attained by stopping at equality and otherwise following the attaining policy from the next step. The hitting event is measurable because $G_t$ and $V_t$ are adapted; finite $N$ ensures termination. Integrability follows from bounded binary rewards and integrable $K_N$. Lemma 1 transfers the result to the original reward. □

The continuation term includes all later opportunities and permitted information. The recursion is not generally a threshold on $q_t$ alone: identical correctness beliefs can accompany different future repair laws or costs. A fitted one-step predictor supplies an approximation, not the conditional multi-step law used by the theorem.

## 2.4 A conditional myopic rule and its failure

Let $\tau_\mu=\inf\{t\ge m:t<N,\ \mu_t\le0\}\wedge N$, using infinity when the crossing set is empty. Suppose nonpositive true drift persists after this first crossing:

$$\mathbf1_{\{\tau_\mu\le t\}}\mu_t\le0\quad\text{almost surely for every }t<N.$$

**Theorem 2.** Under this persistence condition, $\tau_\mu$ is optimal.

**Proof.** Before the crossing, $G_t<\mathbb E[G_{t+1}\mid\mathcal F_t]\le\mathbb E[V_{t+1}\mid\mathcal F_t]$, so stopping cannot be optimal. After the crossing, the condition makes reward a supermartingale on the remaining phase. Finite conditional optional sampling bounds every later admissible reward by the reward at the crossing. Stopping there attains that bound. □

Persistence is substantive and is not established by the observed curves or fitted heads. A deterministic delayed-repair example has $N=2$, $m=0$, correctness $(0,0,1)$ and cumulative costs $(0,0.10,0.20)$. Immediate drift at zero is $-0.10$, yet continuing to the horizon yields reward $0.80$. The first nonpositive drift rule stops too early; drift subsequently becomes positive and violates persistence.

[[DELAYED_REPAIR_FIGURE]]

The example uses a different floor from the live experiments' $m=2$. It establishes a mathematical failure mode, not an estimate of its frequency in deployed models. The extended manuscript provides a second adaptive-information counterexample, calibration and perturbation bounds, and assumptions for sequential certificates. None supplies an unconditional optimality or per-instance safety guarantee for the implemented controller.
