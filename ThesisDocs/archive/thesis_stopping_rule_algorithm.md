# Thesis Stopping-Rule Section

## Purpose

This note is written as thesis-ready prose for the stopping-rule section of the final document. It keeps the theorem-facing object, the empirical estimator, and the deployable stopping time distinct.

## Algorithm Box

### Algorithm 1. Hazard-Based Stopping from Continuation Value

Let $q_t = \mathbb{P}(C_t = 1 \mid \mathcal{F}_t)$ denote the current correctness belief, let $\alpha_t$ be the one-step repair hazard, let $\beta_t$ be the one-step corruption hazard, and let $\lambda > 0$ be the per-step compute cost. The continuation value is

$$
\mu_t = (1-q_t)\alpha_t - q_t\beta_t - \lambda,
$$

and the latent overthinking boundary is

$$
T_c = \inf\{t \ge 0 : \mu_t \le 0\}.
$$

The observable procedure uses trace features $Z_{1:t}$ to estimate the latent objects and then separates a plug-in scientific boundary from a sequentially safe deployment rule.

```text
Input: observable trace features Z_1:Z_T; cost lambda; estimator family for q_t, alpha_t, beta_t;
       anytime-valid upper bound constructor U_t
Output: plug-in boundary T_hat_c and safe stopping time tau_safe

Initialize T_hat_c as undefined
for t = 1, 2, ..., T:
    update features from the current trace prefix
    estimate q_hat_t = P(C_t = 1 | F_t)
    estimate alpha_hat_t = P(C_{t+1} = 1 | C_t = 0, F_t)
    estimate beta_hat_t = P(C_{t+1} = 0 | C_t = 1, F_t)
    compute mu_hat_t = (1 - q_hat_t) * alpha_hat_t - q_hat_t * beta_hat_t - lambda

    if T_hat_c is undefined and mu_hat_t <= 0:
        set T_hat_c = t

    compute U_t, an anytime-valid upper bound on the one-step continuation gain
    if t >= 2 and U_t <= 0:
        return T_hat_c, tau_safe = t

if T_hat_c is undefined:
    set T_hat_c = T
return T_hat_c, tau_safe = T
```

## Interpretation

The algorithm contains three distinct stopping objects.

1. The latent boundary $T_c$ is the theorem-facing optimal stopping time. It is defined in terms of the true hazards and is the object that appears in the structural argument.
2. The plug-in boundary $\widehat{T}_c = \inf\{t \ge 1 : \widehat{\mu}_t \le 0\}$ is the empirical analogue used in the hazard audit and in cross-model comparison. It is appropriate for scientific reporting because it exposes how the observed boundary depends on the estimated repair and corruption hazards.
3. The safe stopping time $\tau_{\mathrm{safe}} = \inf\{t \ge 2 : U_t \le 0\}$ is the deployment-oriented rule. It is stricter than the plug-in crossing because it requires an anytime-valid upper confidence bound to cross zero, not merely a negative point estimate.

This separation matters. The thesis claim is not that one should deploy the first negative plug-in estimate without uncertainty control. The claim is that the latent continuation value defines the correct boundary object, that the plug-in estimator is the right empirical witness, and that a sequentially valid rule is required when the stopping time is itself chosen online.

## Implementation Notes For This Repo

- The current empirical pipeline estimates $\widehat{q}_t$, $\widehat{\alpha}_t$, and $\widehat{\beta}_t$ from observable trace features such as entropy, hidden-state drift, answer revisions, and confidence proxies.
- The theorem-facing plug-in boundary is reported from the corrected conditional-hazard drift, not from the legacy pooled proxy.
- The sequentially safe deployment path is instantiated with an anytime empirical-Bernstein upper bound and audited in parallel against a mixture e-process diagnostic.
- All reported cross-family comparisons should therefore distinguish the corrected boundary estimate from the safer online deployment rule.