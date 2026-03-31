# Overthinking Boundary in Reasoning LLMs

## Scope

This note is an externalized research log rather than private chain-of-thought. It extends the earlier repo baseline in five concrete ways:

1. It broadens the literature base to a 30-source map spanning overthinking, adaptive compute, reward hacking, uncertainty estimation, latent-state control, and anytime-valid inference.
2. It strengthens the semimartingale story with a simple structural theorem showing when monotone hazards imply a one-crossing drift.
3. It upgrades the sequential detector from an anytime Hoeffding rule to a tighter anytime empirical-Bernstein rule.
4. It extends the simulator with observable proxies and additional baselines.
5. It runs a real open-weight pilot locally, and records exactly where the environment still blocks stronger evidence.

Related artifacts in this repo:

- `research/literature_map.csv`
- `research/literature_synthesis.md`
- `research/model_shortlist.md`
- `research/hypothesis_table.md`
- `research/open_questions.md`
- `research/real_trace_experiments.py`
- `research/trace_analysis.py`
- `research/outputs/summary.csv`
- `research/outputs/real_traces_qwen/`

## 1. Literature Update

The literature sweep changed the picture in three important ways.

First, the empirical field has converged on a short list of observables that really do carry stopping information: late-layer hidden states, token entropy spikes, answer-first positions, path deviation markers, trace length, and calibrated judge or verifier outputs. Papers such as ROM, TERMINATOR, DiffAdapt, EAT, REFRAIN, ReBalance, CREST, and TRACE all point in that direction.

Second, several recent papers now treat test-time compute as an allocation problem rather than a prompting trick. CODA, ODAR, Conformal Thinking, CaRT, and latent pondering methods all make explicit that extra reasoning is valuable only when its marginal benefit exceeds its cost.

Third, the reward-model literature now provides a much sharper account of why proxy-only stopping fails. Inference-Time Reward Hacking, Revisiting the (Sub)Optimality of Best-of-N, Reward Under Attack, Robust Reward Modeling via Causal Rubrics, and related work show that proxy improvement can coexist with truth degradation because reward models often overvalue stylistic fluency, verbosity, or spurious rubrics.

The key hole is still the same one the earlier note identified: most papers optimize or detect something useful, but only a few articulate the continuation value itself, and even fewer distinguish pointwise concentration from sequential validity.

## 2. Surviving Theory

Let:

- $Y^*$ be ground truth,
- $A_t$ be the current answer after reasoning step $t$,
- $
\mathcal{F}_t = \sigma(R_{1:t}, A_{1:t}, Z_{1:t})$
  be the observable filtration,
- $C_t = \mathbf{1}\{A_t = Y^*\}$ be correctness,
- $q_t = \mathbb{P}(C_t = 1 \mid \mathcal{F}_t)$ be the current correctness belief,
- $\lambda > 0$ be per-step compute cost.

Define the stop-value process

$$
V_t = q_t - \lambda t.
$$

Let the repair and corruption hazards be

$$
\alpha_t = \frac{\mathbb{P}(C_t = 0, C_{t+1} = 1 \mid \mathcal{F}_t)}{1 - q_t},
$$

$$
\beta_t = \frac{\mathbb{P}(C_t = 1, C_{t+1} = 0 \mid \mathcal{F}_t)}{q_t}.
$$

Then

$$
\mathbb{E}[q_{t+1} - q_t \mid \mathcal{F}_t] = (1-q_t)\alpha_t - q_t\beta_t,
$$

and the predictable drift of $V_t$ is

$$
\mu_t = \mathbb{E}[V_{t+1} - V_t \mid \mathcal{F}_t] = (1-q_t)\alpha_t - q_t\beta_t - \lambda.
$$

This remains the central object. Overthinking begins at the first time $\mu_t \le 0$.

### 2.1 Structural One-Crossing Theorem

The previous note assumed a one-crossing drift. We can now derive it from more primitive monotonicity.

### Theorem 1

If $q_t$ is nondecreasing, $\alpha_t$ is nonincreasing, and $\beta_t$ is nondecreasing, then $\mu_t$ is nonincreasing.

### Proof

Using

$$
\mu_t = (1-q_t)\alpha_t - q_t\beta_t - \lambda,
$$

we have

$$
\mu_{t+1} - \mu_t
= (1-q_{t+1})(\alpha_{t+1}-\alpha_t)
+ \alpha_t(q_t-q_{t+1})
- q_{t+1}(\beta_{t+1}-\beta_t)
- \beta_t(q_{t+1}-q_t).
$$

Under the stated monotonicity conditions, every term on the right-hand side is nonpositive, hence $\mu_{t+1} \le \mu_t$. QED.

This matters because it turns the earlier one-crossing assumption into a falsifiable structural condition: diminishing repair opportunity, rising corruption pressure, and improving correctness belief are sufficient.

### 2.2 Optimal Boundary

Under the one-crossing condition,

$$
T_c = \inf\{t \ge 0 : \mu_t \le 0\}
$$

is the optimal stopping time for maximizing $\mathbb{E}[V_\tau]$ over bounded stopping times.

That part of the earlier argument survives unchanged.

### 2.3 Reward-Hacking Region

Let a proxy score $P_t$ have predictable drift

$$
d_t = \mu_t + \kappa_t,
$$

where $\kappa_t$ is proxy optimism bias. Then the reward-hacking region is

$$
\mathcal{H} = \{t : \mu_t < 0 < \mu_t + \kappa_t\}.
$$

This is still the cleanest formal explanation of why PRM-only stopping can fail: extra reasoning is truly harmful while the proxy still says continue.

## 3. Sequential Detection

The earlier note already corrected one important mistake: pointwise bounds are not enough for first-crossing stopping. That remains true.

Suppose bounded rollout probes at step $t$ estimate the one-step continuation gain:

$$
\widehat{\mu}_t = \frac{1}{m}\sum_{i=1}^m \Delta_t^{(i)}, \qquad \Delta_t^{(i)} \in [a,b].
$$

### 3.1 Anytime Hoeffding Baseline

With

$$
\delta_t = \frac{6\delta}{\pi^2 (t+1)^2},
$$

the earlier detector used

$$
U_t^{\mathrm{H}} = \widehat{\mu}_t + (b-a)\sqrt{\frac{\log(1/\delta_t)}{2m}}.
$$

Stopping at the first $t$ with $U_t^{\mathrm{H}} \le 0$ is sequentially valid, but conservative.

### 3.2 Anytime Empirical-Bernstein Upgrade

Let $\widehat{v}_t$ be the sample variance of the probes. Define

$$
U_t^{\mathrm{EB}} = \widehat{\mu}_t
+ \sqrt{\frac{2\widehat{v}_t\log(3/\delta_t)}{m}}
+ \frac{3(b-a)\log(3/\delta_t)}{m}.
$$

### Proposition 2

If the per-time empirical-Bernstein bound is valid for each fixed $t$, then by the same summable schedule and union-bound argument,

$$
\mathbb{P}(\exists t \ge 0 : \mu_t > U_t^{\mathrm{EB}}) \le \delta.
$$

So the first-crossing rule based on $U_t^{\mathrm{EB}} \le 0$ is also sequentially safe.

This is not yet a mixture-bound or e-process style confidence sequence. It is still a stitched union-bound construction. But it is materially tighter than Hoeffding whenever probe variance is substantially below the worst-case range.

## 4. Extended Simulation

The simulator now includes:

- the earlier exact boundary, anytime Hoeffding, naive pointwise, and PRM argmax rules,
- a new anytime empirical-Bernstein detector,
- a CUSUM baseline,
- an entropy-threshold baseline,
- observable proxies for entropy and hidden-state shift.

All synthetic claims in this section come from `research/simulate_overthinking_boundary.py` as executed in this workspace on 2026-03-31.

### 4.1 Main Synthetic Results

The exact boundary still tracks the oracle closely:

- helpful reasoning: oracle $16.985$, true boundary $16.585$,
- overthinking: oracle $12.950$, true boundary $13.075$,
- reward hacking: oracle $12.123$, true boundary $12.110$.

The new empirical-Bernstein detector improves over anytime Hoeffding in every scenario:

- helpful reasoning: gap $0.1057$ vs $0.1578$,
- overthinking: gap $0.0315$ vs $0.0448$,
- reward hacking: gap $0.0315$ vs $0.0408$.

The improvement is largest where probe variance is low relative to the bounded support, exactly where a variance-adaptive rule should help.

The practical baselines performed worse than expected:

- CUSUM lagged badly, with gaps from $0.1140$ to $0.3538$,
- the entropy threshold baseline was much too late, with gaps from $0.2287$ to $0.3119$.

The proxy anti-baseline remains the sharpest reward-hacking demonstration:

- in `reward_hacking`, PRM argmax stops at $19.81$ versus an oracle near $12.12$ and loses $0.1529$ value.

### 4.2 Takeaway

The core synthetic story now looks stronger than before:

1. The drift-sign boundary still matches the oracle.
2. A sequentially valid detector can be made materially tighter than the old Hoeffding baseline.
3. Observable-only baselines such as simple entropy thresholds are not competitive by themselves.
4. Proxy-based stopping still breaks exactly where the theory says it should.

## 5. Real Open-Weight Pilot

### 5.1 Environment Reality

The local machine has a GTX 1650 with 4 GB VRAM, but the active PyTorch environment in this workspace is CPU-only. That constraint dominates model feasibility.

### 5.2 What Actually Ran

The repo now contains a reproducible harness:

- `research/real_trace_experiments.py`
- `research/trace_analysis.py`

Actual completed run:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- backend: `transformers+torch(cpu)`
- hidden states: accessible
- token logprobs: accessible
- protocol: forced incremental answer updates over 3 steps on 3 exact-answer tasks at temperatures $0.2$ and $0.8$
- outputs: `research/outputs/real_traces_qwen/`

Attempted but not completed:

- model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- status: weights downloaded and model initialized successfully, but meaningful trace execution was too slow on CPU to finish within a practical research iteration.

### 5.3 What the Qwen Pilot Showed

The completed pilot is scientifically useful mainly as a failure case.

The 0.5B Qwen control model remained in a low-skill regime under the structured incremental tracing protocol:

- $q_t = 0$ at every observed step,
- repair rate $\alpha_t = 0$ at every observed transition,
- no observed corruption transitions because the model was never correct,
- empirical continuation utility stayed at $-\lambda = -0.05$.

As a result, the oracle under the chosen utility was immediate stop after the first answer attempt, and every later detector was necessarily worse.

This is not evidence against the drift-sign theory. It is evidence that the chosen model-protocol pair never entered the regime where overthinking is even meaningful. In other words, the pilot found a tractable but underpowered control model, not a capable reasoning model with a rich repair-versus-corruption tradeoff.

### 5.4 What Failed and Why It Matters

Two empirical failures were important:

1. The small Qwen control model frequently imitated the formatting scaffold instead of producing semantically useful answer updates.
2. The DeepSeek distill was locally feasible to download and initialize, but not locally feasible to run at meaningful scale in the CPU-only environment.

These failures are worth recording because they delimit the real scope of the current empirical claims:

- the real-model harness is now in place and tested,
- hidden-state and logprob collection works,
- but the present workstation is not strong enough to turn the preferred reasoning models into a dense trace dataset within one autonomous iteration.

## 6. What Survived the Cross-Disciplinary Sweep

Three frameworks still look worth keeping.

### 6.1 Semimartingale Drift with Hazards

This remains the cleanest theorem-bearing core because it directly exposes the continuation value and explains reward hacking with one extra bias term.

### 6.2 Hazard-Based Operational Layer

For real traces, the best operational decomposition is still in terms of:

- current correctness belief $q_t$,
- repair hazard $\alpha_t$,
- corruption hazard $\beta_t$,
- optionally proxy bias $\kappa_t$.

This is more identifiable than a fully general POMDP and more falsifiable than a pure change-point story.

### 6.3 Anytime Inference Layer

Confidence sequences, e-processes, conformal risk control, and empirical-Bernstein-style sequential bounds are not the whole theory, but they are the right safety layer on top of it.

## 7. What Got Rejected or Demoted

- Pure entropy-threshold stopping: useful feature, weak stopping object.
- Pure path-deviation heuristics: good diagnostics, weak theorems.
- Dynamical-systems metaphors as the main formalism: interesting language, poor identifiability.
- POMDPs as the main model: too general to be useful without collapsing back to hazards or continuation value.
- Small-model local pilot as substantive evidence about overthinking in strong LRMs: rejected; the model was simply not capable enough under the protocol.

## 8. Immediate Questions, Answered Explicitly

1. Does the drift-sign boundary continue to explain real open-weight traces?

Not yet for a capable reasoning model. The Qwen 0.5B pilot was too weak: it stayed in a regime with no successful repairs and no observed corruption events. The harness is now ready for stronger models, but the present local evidence is only a negative control.

2. Can $q_t$, $\alpha_t$, and $\beta_t$ be estimated well enough online to drive a practical stop rule?

On exact-verifiable tasks, $q_t$ can collapse to a verifier-based quantity. The harder part is estimating $\alpha_t$ and $\beta_t$ from enough real transitions. The current local pilot did not generate a rich enough transition set for that claim to survive empirically.

3. Which observables look best?

From literature plus simulation: entropy dynamics, hidden-state drift, answer revisions, path deviation, trace length, and calibrated judge features. From the local real pilot alone: no ranking is trustworthy because the model never entered a useful repair regime.

4. What exact failure mode produces reward hacking?

The strongest surviving account is still

$$
\mu_t < 0 < \mu_t + \kappa_t,
$$

where stylistic or verbosity-sensitive proxy components keep rising after true correctness-minus-cost has turned negative.

5. Which framework is cleanest and still empirically defensible?

The hazard-based semimartingale model, with a sequential-valid detection layer on top.

6. Where do the semimartingale assumptions break on actual traces?

The local break was not mathematical dependence first; it was identifiability. A too-small model under a structured tracing scaffold did not generate the repair/corruption events needed to estimate the latent quantities.

7. Can a tighter detector beat the earlier anytime-safe rule without losing formal safety?

Yes in simulation. The anytime empirical-Bernstein detector improved over anytime Hoeffding in all three scenarios while preserving the same sequential-validity logic via the summable $\delta_t$ schedule.

## 9. Current Best Research Position

The project is now in a better state than the baseline, but not finished.

What is now stronger:

- the literature review is materially broader,
- the semimartingale theory is sharper,
- the sequential detector is tighter,
- the simulator is more realistic and has stronger baselines,
- the real-model harness exists and has been exercised locally,
- the feasibility boundary for local experiments is now explicit rather than guessed.

What is still missing:

- successful real-trace evidence on at least one genuinely capable reasoning model,
- dense transition data for estimating $q_t$, $\alpha_t$, and $\beta_t$ online,
- a stronger sequential detector built from mixture boundaries or e-processes rather than stitched fixed-time inequalities,
- a real PRM or verifier-conditioned proxy study instead of only synthetic reward-hacking evidence.

## 10. Generated Artifacts

- `research/literature_map.csv`
- `research/literature_synthesis.md`
- `research/model_shortlist.md`
- `research/hypothesis_table.md`
- `research/open_questions.md`
- `research/real_trace_experiments.py`
- `research/trace_analysis.py`
- `research/outputs/summary.csv`
- `research/outputs/observable_signals.png`
- `research/outputs/representative_trajectories.png`
- `research/outputs/monte_carlo_gaps.png`
- `research/outputs/average_drifts.png`
- `research/outputs/real_traces_qwen/metadata.json`
- `research/outputs/real_traces_qwen/trace_steps.csv`
- `research/outputs/real_traces_qwen/detector_comparison.csv`
