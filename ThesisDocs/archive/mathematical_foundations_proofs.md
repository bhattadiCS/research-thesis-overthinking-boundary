# Mathematical Foundations and Proofs of the Overthinking Boundary in Reasoning LLMs

**Master's Thesis Research Note — Mathematical Derivations (625.803–804)**  
**Author:** Aditya Bhatt (M.S. in Applied and Computational Mathematics, Johns Hopkins University)  
**Research Adviser:** Dr. Zerotti Woods  
**Proposed Second Reader:** Dr. Moustapha Pemy  

---

## 1. Introduction and Setup

This document provides a self-contained, mathematically rigorous exposition of the optimal stopping boundary for chain-of-thought reasoning in Large Language Models (LLMs). We detail:
1. The step-by-step derivation of the predictable drift equation ($\mu_t$) from scratch.
2. The formal proof of the **Structural One-Crossing Theorem** (Theorem 1).
3. The optimal stopping formulation using One-Step Look-Ahead (OSLA) theory.
4. The statistical safety proofs for the **Anytime Empirical-Bernstein** and **Mixture E-Process** stopping rules.

---

## 2. Rigorous Mathematical Re-derivation of the Drift Equation

### 2.1. Probability Space and Filtration
Let $(\Omega, \mathcal{F}, (\mathcal{F}_t)_{t \ge 0}, \mathbb{P})$ be a filtered probability space. The discrete-time index $t \in \mathbb{N}_0$ represents the step count in a generated chain of thought.
*   Let $Y^* \in \mathcal{Y}$ be the unique, latent ground-truth answer.
*   Let $A_t \in \mathcal{Y}$ be the intermediate candidate answer extracted from the reasoning trace at step $t$.
*   Let $C_t = \mathbf{1}\{A_t = Y^*\}$ be the binary indicator of correctness.
*   Let $\mathcal{F}_t = \sigma(R_{1:t}, A_{1:t}, Z_{1:t})$ be the observable filtration, where $R_{1:t}$ is the token history, $A_{1:t}$ is the history of candidate answers, and $Z_{1:t}$ represents intermediate observables (e.g., hidden states, token logprobs).
*   Let $q_t = \mathbb{P}(C_t = 1 \mid \mathcal{F}_t) = \mathbb{E}[C_t \mid \mathcal{F}_t]$ represent the model's belief state of correctness at step $t$.

### 2.2. The Value Process
Let $v > 0$ denote the utility scalar for finalizing a correct answer, and $-c < 0$ (with $c > 0$) denote the penalty for finalizing an incorrect answer. The expected utility of stopping at step $t$ is:
$$V(q_t) = q_t \cdot v - (1 - q_t) \cdot c = q_t(v + c) - c$$
To account for cumulative computational cost, we introduce a constant per-step cost parameter $\lambda > 0$. The stopped utility value process $V_t$ is defined as:
$$V_t = V(q_t) - \lambda t = q_t(v + c) - c - \lambda t$$

### 2.3. Tower Property and Correctness Transition
The predictable drift of $V_t$ is the conditional expectation of its increment:
$$\mu_t = \mathbb{E}[V_{t+1} - V_t \mid \mathcal{F}_t]$$
Substituting the definition of $V_t$:
$$\mu_t = \mathbb{E}\left[ q_{t+1}(v+c) - c - \lambda(t+1) - \left( q_t(v+c) - c - \lambda t \right) \;\middle|\; \mathcal{F}_t \right]$$
$$\mu_t = \mathbb{E}[q_{t+1} - q_t \mid \mathcal{F}_t](v+c) - \lambda$$

To compute the expected belief change $\mathbb{E}[q_{t+1} - q_t \mid \mathcal{F}_t]$, we apply the tower property of conditional expectation. Since the filtration is nested ($\mathcal{F}_t \subset \mathcal{F}_{t+1}$), we have:
$$\mathbb{E}[q_{t+1} \mid \mathcal{F}_t] = \mathbb{E}\left[ \mathbb{E}[C_{t+1} \mid \mathcal{F}_{t+1}] \;\middle|\; \mathcal{F}_t \right] = \mathbb{E}[C_{t+1} \mid \mathcal{F}_t]$$
Since $q_t = \mathbb{E}[C_t \mid \mathcal{F}_t]$ is $\mathcal{F}_t$-measurable, we write:
$$\mathbb{E}[q_{t+1} - q_t \mid \mathcal{F}_t] = \mathbb{E}[C_{t+1} - C_t \mid \mathcal{F}_t]$$

### 2.4. Decomposing transitions into hazards
The binary difference $C_{t+1} - C_t$ can only take three values in $\{-1, 0, 1\}$. We decompose its expectation by conditioning on the indicator of the current state $C_t$:
$$\mathbb{E}[C_{t+1} - C_t \mid \mathcal{F}_t] = \mathbb{E}\left[ (C_{t+1} - C_t)\mathbf{1}\{C_t = 0\} + (C_{t+1} - C_t)\mathbf{1}\{C_t = 1\} \;\middle|\; \mathcal{F}_t \right]$$
*   If $C_t = 0$, then $C_{t+1} - C_t = C_{t+1} - 0 = C_{t+1}$.
*   If $C_t = 1$, then $C_{t+1} - C_t = C_{t+1} - 1 = -\mathbf{1}\{C_{t+1} = 0\}$.

Thus,
$$\mathbb{E}[C_{t+1} - C_t \mid \mathcal{F}_t] = \mathbb{E}[C_{t+1}\mathbf{1}\{C_t = 0\} \mid \mathcal{F}_t] - \mathbb{E}[\mathbf{1}\{C_{t+1} = 0, C_t = 1\} \mid \mathcal{F}_t]$$
$$\mathbb{E}[C_{t+1} - C_t \mid \mathcal{F}_t] = \mathbb{P}(C_{t+1} = 1, C_t = 0 \mid \mathcal{F}_t) - \mathbb{P}(C_{t+1} = 0, C_t = 1 \mid \mathcal{F}_t)$$

Now, we define the **conditional repair hazard** $\alpha_t$ and the **conditional corruption hazard** $\beta_t$ as:
$$\alpha_t = \mathbb{P}(C_{t+1} = 1 \mid C_t = 0, \mathcal{F}_t)$$
$$\beta_t = \mathbb{P}(C_{t+1} = 0 \mid C_t = 1, \mathcal{F}_t)$$

Applying the definition of conditional probability:
$$\mathbb{P}(C_{t+1} = 1, C_t = 0 \mid \mathcal{F}_t) = \mathbb{P}(C_{t+1} = 1 \mid C_t = 0, \mathcal{F}_t)\mathbb{P}(C_t = 0 \mid \mathcal{F}_t) = \alpha_t (1 - q_t)$$
$$\mathbb{P}(C_{t+1} = 0, C_t = 1 \mid \mathcal{F}_t) = \mathbb{P}(C_{t+1} = 0 \mid C_t = 1, \mathcal{F}_t)\mathbb{P}(C_t = 1 \mid \mathcal{F}_t) = \beta_t q_t$$

Substituting these back into the expectation:
$$\mathbb{E}[q_{t+1} - q_t \mid \mathcal{F}_t] = (1 - q_t)\alpha_t - q_t\beta_t$$

Finally, substituting the expected belief revision into the predictable drift:
$$\mu_t = \left[ (1 - q_t)\alpha_t - q_t\beta_t \right](v+c) - \lambda$$

Setting the utility scale $v+c = 1$ (without loss of generality), we obtain the normalized continuation drift:
$$\mu_t = (1 - q_t)\alpha_t - q_t\beta_t - \lambda \quad \text{Q.E.D.}$$

---

## 3. Proof of the Structural One-Crossing Theorem

We formalize our structural hypothesis—stating that the predictable drift $\mu_t$ crosses zero at most once—as Theorem 1.

### Theorem 1
Let $\mu_t = (1 - q_t)\alpha_t - q_t\beta_t - \lambda$. If:
1.  The correctness belief $q_t$ is nondecreasing in $t$ almost surely,
2.  The repair hazard $\alpha_t$ is nonincreasing in $t$ almost surely,
3.  The corruption hazard $\beta_t$ is nondecreasing in $t$ almost surely,

then the predictable drift $\mu_t$ is nonincreasing in $t$ almost surely.

### Proof
We compute the difference of the drift at consecutive steps:
$$\mu_{t+1} - \mu_t = (1 - q_{t+1})\alpha_{t+1} - q_{t+1}\beta_{t+1} - \lambda - \left[ (1 - q_t)\alpha_t - q_t\beta_t - \lambda \right]$$
$$\mu_{t+1} - \mu_t = (1 - q_{t+1})\alpha_{t+1} - (1 - q_t)\alpha_t - q_{t+1}\beta_{t+1} + q_t\beta_t$$

We decompose this difference by adding and subtracting cross-terms $(1 - q_{t+1})\alpha_t$ and $q_{t+1}\beta_t$:
$$\mu_{t+1} - \mu_t = (1 - q_{t+1})\alpha_{t+1} - (1 - q_{t+1})\alpha_t + (1 - q_{t+1})\alpha_t - (1 - q_t)\alpha_t - q_{t+1}\beta_{t+1} + q_{t+1}\beta_t - q_{t+1}\beta_t + q_t\beta_t$$

Factoring the terms yields:
$$\mu_{t+1} - \mu_t = (1 - q_{t+1})(\alpha_{t+1} - \alpha_t) + \alpha_t(q_t - q_{t+1}) - q_{t+1}(\beta_{t+1} - \beta_t) - \beta_t(q_{t+1} - q_t)$$

We now analyze the signs of each of the four terms under the stated monotonicity conditions:
1.  **Term 1:** $(1 - q_{t+1})(\alpha_{t+1} - \alpha_t)$
    *   $q_{t+1} \in [0, 1] \implies 1 - q_{t+1} \ge 0$.
    *   $\alpha_t$ is nonincreasing $\implies \alpha_{t+1} - \alpha_t \le 0$.
    *   Therefore, $(1 - q_{t+1})(\alpha_{t+1} - \alpha_t) \le 0$.
2.  **Term 2:** $\alpha_t(q_t - q_{t+1})$
    *   $\alpha_t \ge 0$ since it is a probability hazard.
    *   $q_t$ is nondecreasing $\implies q_t - q_{t+1} \le 0$.
    *   Therefore, $\alpha_t(q_t - q_{t+1}) \le 0$.
3.  **Term 3:** $-q_{t+1}(\beta_{t+1} - \beta_t)$
    *   $q_{t+1} \ge 0$ since it is a probability belief.
    *   $\beta_t$ is nondecreasing $\implies \beta_{t+1} - \beta_t \ge 0 \implies -(\beta_{t+1} - \beta_t) \le 0$.
    *   Therefore, $-q_{t+1}(\beta_{t+1} - \beta_t) \le 0$.
4.  **Term 4:** $-\beta_t(q_{t+1} - q_t)$
    *   $\beta_t \ge 0$ since it is a probability hazard.
    *   $q_t$ is nondecreasing $\implies q_{t+1} - q_t \ge 0 \implies -(q_{t+1} - q_t) \le 0$.
    *   Therefore, $-\beta_t(q_{t+1} - q_t) \le 0$.

Since $\mu_{t+1} - \mu_t$ is a sum of four nonpositive terms, we conclude:
$$\mu_{t+1} - \mu_t \le 0 \implies \mu_{t+1} \le \mu_t \quad \text{a.s.} \quad \text{Q.E.D.}$$

---

## 4. Optimal Stopping Theory and OSLA Optimality

We establish why stopping at the first step where $\mu_t \le 0$ is optimal under sequential decision theory.

### 4.1. Monotone Optimal Stopping Problems
Following Chow, Robbins, and Siegmund (1971), an optimal stopping problem for a sequence of random variables $V_t$ is called **monotone** (or satisfies the one-step look-ahead property) if the stopping region:
$$G_t = \{ \omega : V_t(\omega) \ge \mathbb{E}[V_{t+1}(\omega) \mid \mathcal{F}_t] \}$$
is closed under transitions. That is, if $\omega \in G_t$, then $\omega \in G_s$ for all $s > t$ almost surely.

### 4.2. Proof of Optimality of the OSLA Rule
Using the definition of predictable drift, we write the stopping region as:
$$G_t = \{ \omega : \mathbb{E}[V_{t+1} - V_t \mid \mathcal{F}_t] \le 0 \} = \{ \omega : \mu_t(\omega) \le 0 \}$$
By Theorem 1, $\mu_t$ is nonincreasing almost surely. Thus, if $\mu_t(\omega) \le 0$, it follows that for any $s > t$, $\mu_s(\omega) \le \mu_t(\omega) \le 0$, which implies $\omega \in G_s$. Thus, the stopping problem is monotone.

Let $\mathcal{T}$ denote the set of all stopping times adapted to the filtration $\mathcal{F}_t$. Let the OSLA stopping time be:
$$T^* = \inf \{ t \ge 0 : \mu_t \le 0 \}$$
We wish to show that $T^*$ maximizes expected utility over all stopping times $\tau \in \mathcal{T}$.

### Proof
By the Doob Decomposition, any adapted process $V_t$ can be written as:
$$V_t = V_0 + M_t + A_t$$
where $M_t$ is a martingale with $M_0 = 0$, and $A_t = \sum_{s=0}^{t-1} \mu_s$ is the predictable drift process.
For any bounded stopping time $\tau \in \mathcal{T}$, the optional stopping theorem implies $\mathbb{E}[M_\tau] = 0$.
Thus:
$$\mathbb{E}[V_\tau] = V_0 + \mathbb{E}[A_\tau] = V_0 + \mathbb{E}\left[ \sum_{s=0}^{\tau-1} \mu_s \right]$$

Now we compare the expected stopped value of $T^*$ and $\tau$:
$$\mathbb{E}[V_{T^*} - V_\tau] = \mathbb{E}\left[ \sum_{s=0}^{T^*-1} \mu_s - \sum_{s=0}^{\tau-1} \mu_s \right]$$
We split the sum into two disjoint events: $\{\tau < T^*\}$ and $\{\tau > T^*\}$:
$$\mathbb{E}[V_{T^*} - V_\tau] = \mathbb{E}\left[ \sum_{s=\tau}^{T^*-1} \mu_s \mathbf{1}\{\tau < T^*\} - \sum_{s=T^*}^{\tau-1} \mu_s \mathbf{1}\{\tau > T^*\} \right]$$

We evaluate the sign of each term inside the expectation:
1.  **On the event $\{\tau < T^*\}$:** The summation range is $s \in [\tau, T^*-1]$. For all such $s$, we have $s < T^*$. By definition of $T^*$, we must have $\mu_s > 0$. Thus, the sum is a sum of positive terms:
    $$\sum_{s=\tau}^{T^*-1} \mu_s \mathbf{1}\{\tau < T^*\} \ge 0 \quad \text{a.s.}$$
2.  **On the event $\{\tau > T^*\}$:** The summation range is $s \in [T^*, \tau-1]$. For all such $s$, we have $s \ge T^*$. By definition of $T^*$ and the monotonicity of $\mu_t$, we must have $\mu_s \le 0$. Thus, the sum is nonpositive, which implies its negative is nonnegative:
    $$-\sum_{s=T^*}^{\tau-1} \mu_s \mathbf{1}\{\tau > T^*\} \ge 0 \quad \text{a.s.}$$

Since both terms inside the expectation are almost surely nonnegative, we conclude:
$$\mathbb{E}[V_{T^*} - V_\tau] \ge 0 \implies \mathbb{E}[V_{T^*}] \ge \mathbb{E}[V_\tau] \quad \text{Q.E.D.}$$

---

## 5. Sequential Safety and Anytime-Valid Bounds

To deploy the stopping rule online, we require statistical estimators that are valid uniformly over time. We evaluate two methods: the **Anytime Empirical-Bernstein Bound** and the **Mixture E-Process**.

### 5.1. Anytime Empirical-Bernstein Bound
Let $\Delta_t^{(i)} = V_{t+1}^{(i)} - V_t^{(i)}$ be independent realization samples of the continuation gain at step $t$ across different prompts, bounded in a known interval $[a, b]$.
Let the sample mean be $\widehat{\mu}_t = \frac{1}{m}\sum_{i=1}^m \Delta_t^{(i)}$ and sample variance be $\widehat{v}_t = \frac{1}{m}\sum_{i=1}^m (\Delta_t^{(i)} - \widehat{\mu}_t)^2$.

For each step $t$, we construct the empirical-Bernstein upper bound:
$$U_t^{\mathrm{EB}} = \widehat{\mu}_t + \sqrt{\frac{2\widehat{v}_t\log(3/\delta_t)}{m}} + \frac{3(b-a)\log(3/\delta_t)}{m}$$
where $\delta_t = \frac{6\delta}{\pi^2 (t+1)^2}$ is a summable confidence schedule ensuring $\sum_{t=1}^\infty \delta_t = \delta$.

### Proposition 2
The stopping rule $\tau_{\mathrm{safe}} = \inf\{t \ge 2 : U_t^{\mathrm{EB}} \le 0\}$ controls the false-early stop rate at level $\delta$:
$$\mathbb{P}(\text{false-early stop}) \le \delta$$

### Proof
A false-early stop occurs if we stop at some step $t = \tau_{\mathrm{safe}}$ when the true continuation gain is positive ($\mu_t > 0$).
By definition of $\tau_{\mathrm{safe}}$, stopping implies $U_t^{\mathrm{EB}} \le 0$.
If the true gain $\mu_t > 0$, then we must have $\mu_t > U_t^{\mathrm{EB}}$.
This implies:
$$\text{false-early stop} \subseteq \{\exists t \ge 2 : \mu_t > U_t^{\mathrm{EB}}\}$$

By union bound:
$$\mathbb{P}(\exists t \ge 2 : \mu_t > U_t^{\mathrm{EB}}) \le \sum_{t=2}^\infty \mathbb{P}(\mu_t > U_t^{\mathrm{EB}})$$
From the fixed-time empirical-Bernstein inequality, for any fixed $t$, $\mathbb{P}(\mu_t > U_t^{\mathrm{EB}}) \le \delta_t$.
Thus:
$$\mathbb{P}(\text{false-early stop}) \le \sum_{t=2}^\infty \delta_t < \sum_{t=1}^\infty \delta_t = \delta \quad \text{Q.E.D.}$$

---

### 5.2. Mixture E-Process Stopping Rule
We reformulate the sequential stopping decision using e-values under the **testing-by-betting** framework.
At step $t$, we test the null hypothesis:
$$H_0: \mu_t \ge 0 \quad (\text{continuation is beneficial}) \quad \text{vs.} \quad H_1: \mu_t < 0 \quad (\text{stop, overthinking started})$$

Let $X_1, \ldots, X_m$ be independent observations of the continuation gain $\Delta_t$ at step $t$. Let $M = \max(|a|, |b|)$ be the scale parameter. We normalize the observations to $Y_i = X_i / M \in [-1, 1]$. Under the null hypothesis $H_0$, $\mathbb{E}[Y_i] \ge 0$.

For any parameter $\lambda \in [0, 1)$, we define the betting outcomes:
$$E_i(\lambda) = 1 - \lambda Y_i$$
Since $Y_i \ge -1$, $E_i(\lambda) = 1 - \lambda Y_i \ge 1 - \lambda > 0$, ensuring nonnegativity. The expected value under the null is:
$$\mathbb{E}[E_i(\lambda) \mid \mathcal{F}_{i-1}] = 1 - \lambda \mathbb{E}[Y_i \mid \mathcal{F}_{i-1}] \le 1 - 0 = 1$$

Since the samples are independent, the product process:
$$E^{(k)}(\lambda) = \prod_{i=1}^k (1 - \lambda Y_i)$$
is a nonnegative supermartingale starting at $E^{(0)}(\lambda) = 1$.

To eliminate dependence on a fixed $\lambda$, we define a mixture e-process over a grid of parameters $\Lambda \subset [0, 1)$ (e.g., 19 linearly spaced values between $0.05$ and $0.95$):
$$E^{(k)} = \frac{1}{|\Lambda|} \sum_{\lambda \in \Lambda} E^{(k)}(\lambda)$$
Since the sum of nonnegative supermartingales is a nonnegative supermartingale, $E^{(k)}$ is a valid e-process with $\mathbb{E}[E^{(k)}] \le 1$ under $H_0$.

We define the e-process stopping rule:
$$\tau_{\text{e-proc}} = \inf \left\{ t \ge 2 : E^{(m)} \ge \frac{1}{\delta_t} \right\}$$
where $\delta_t$ is the same summable confidence schedule.

By Ville's inequality for nonnegative supermartingales, for any threshold $K > 0$:
$$\mathbb{P}(\exists k \ge 1 : E^{(k)} \ge K) \le \frac{\mathbb{E}[E^{(0)}]}{K} = \frac{1}{K}$$
Setting $K_t = 1/\delta_t$, the probability of a false stop at step $t$ is bounded by $\delta_t$. Applying the union bound across all steps $t \ge 2$ guarantees:
$$\mathbb{P}(\text{false-early stop}) \le \sum_{t=2}^\infty \delta_t < \delta \quad \text{Q.E.D.}$$
