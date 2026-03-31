---
name: "Autonomous Overthinking Research"
description: "Continue the ResearchThesis overthinking-boundary program through literature review, mathematical ideation, real open-weight model experiments, and theory-driven stopping rules."
argument-hint: "Optional: override model shortlist, benchmark priorities, compute budget, or preferred mathematical directions"
agent: "agent"
---

# Autonomous Research Prompt: Overthinking Boundary in Reasoning LLMs

You are continuing an existing research program in this repository. This is not a greenfield task. Begin from the current repo state, preserve what is already correct, and push the work all the way through the next major milestones.

## 0. Starting Context

Read these files first and treat them as current ground truth unless your new evidence proves otherwise:

- [Current theory note](../../research/overthinking_boundary.md)
- [Current simulator](../../research/simulate_overthinking_boundary.py)
- [Current summary metrics](../../research/outputs/summary.csv)
- [Representative trajectories](../../research/outputs/representative_trajectories.png)
- [Gap distributions](../../research/outputs/monte_carlo_gaps.png)
- [Average drifts](../../research/outputs/average_drifts.png)

What is already established in this repo:

- The current note formalizes overthinking as a sign change in the predictable drift

$$
\mu_t = (1-q_t)\alpha_t - q_t\beta_t - \lambda.
$$

- The current simulator validates that the exact theoretical boundary closely matches the oracle stopping point under synthetic dynamics.
- The current simulator also shows that a PRM-only stopping rule can fail badly under reward hacking.
- The current note already corrected one important mistake: pointwise confidence bounds are not sufficient for a first-crossing sequential stopping claim.

What is not yet established:

- Real open-weight model evidence.
- Real trace collection and hidden-state or logit diagnostics.
- Estimation procedures for $q_t$, $\alpha_t$, $\beta_t$, and $\kappa_t$ from actual traces.
- Tighter sequential detectors than the current conservative anytime Hoeffding-style rule.
- A broader cross-disciplinary mathematical search beyond the initial semimartingale framing.

## 1. Mission

Continue autonomously until the following research question is materially advanced:

> When and why does additional reasoning compute in a large language model change from beneficial to harmful, and what mathematically principled stopping rule can identify that transition before quality degrades?

Your job is to:

1. Expand the literature review far beyond the currently cited papers.
2. Sweep across machine learning, statistics, applied mathematics, stochastic processes, optimization, information theory, control theory, dynamical systems, and any adjacent field that may contribute a usable formalism.
3. Generate, test, and prune multiple candidate theories.
4. Move from synthetic simulations toward real open-weight model experiments.
5. Produce new files, code, metrics, and a revised mathematical note that are stronger than the current baseline.

Do not stop at brainstorming. Implement, test, reject, refine, and only finish once you have a materially improved theory-plus-evidence package.

## 2. Non-Negotiable Rules

1. Do not restart from scratch. Build on the existing note and simulator.
2. Do not claim private chain-of-thought. Provide concise research logs, explicit assumptions, derivations, and evidence.
3. Prefer verified literature over memory. Use arXiv and web tools aggressively.
4. When a formal claim depends on a sequential bound, check whether the bound is pointwise or time-uniform.
5. When a proxy metric looks good, compare it against a true utility proxy such as correctness-minus-compute, not just another proxy.
6. If a candidate framework fails in simulation or on real traces, document the failure and pivot.
7. If actual open-weight model execution is infeasible in the environment, leave a reproducible harness and clearly label what remains unexecuted.
8. Every major claim should map to one of: theorem, simulation, empirical trace result, ablation, or literature-backed observation.

## 3. Preferred Open-Weight Model Suite

Use open-weight or openly downloadable models for real-trace experiments. Do not anchor on a single family.

### Primary shortlist

| Role | Preferred models | Why they matter |
| --- | --- | --- |
| Primary reasoning family | DeepSeek-R1 and available DeepSeek-R1 distills | Best direct fit for long reasoning and overthinking behavior. |
| Strong generalist control | Llama 3.3 70B Instruct or closest open Llama reasoning-capable instruct model available in the environment | Gives a major open family that is not specialized around the exact same training recipe. |
| Alternative reasoning family | QwQ-32B, Qwen reasoning or instruct models with strong math/code performance | Useful to separate DeepSeek-specific behavior from broader reasoning-model behavior. |
| Medium-size instrumentation model | Gemma 3, Mistral Small, Phi-family reasoning or instruct variants, or another medium open-weight model that can be run with hidden-state access | Useful when the large models are too expensive for dense instrumentation. |

### Minimum viable empirical set

If resources are limited, prioritize this order:

1. One DeepSeek-R1 family model or distill.
2. One Llama-family instruct model.
3. One Qwen or QwQ family model.

### Model selection criteria

For each model, record:

- exact model name,
- parameter count,
- quantization if any,
- inference backend,
- maximum context length used,
- whether token logprobs are accessible,
- whether hidden states are accessible,
- whether full reasoning traces are accessible or hidden,
- sampling parameters,
- hardware/runtime limits.

### If the environment cannot run the preferred set

Use the best available open-weight substitutes and explicitly note the substitution.

## 4. Immediate Questions to Answer

You must answer these explicitly in the updated research package:

1. Does the drift-sign boundary continue to explain real open-weight traces, not just synthetic ones?
2. Can $q_t$, $\alpha_t$, and $\beta_t$ be estimated well enough online to drive a practical stopping rule?
3. Which observables best predict the boundary: entropy, semantic divergence, answer revisions, verifier scores, hidden-state projections, reward-model outputs, or combinations thereof?
4. What exact failure mode produces reward hacking in reasoning traces?
5. Which mathematical framework yields the cleanest theorem that still survives empirical testing?
6. Where do the current semimartingale assumptions break when applied to actual traces?
7. Can a tighter detector beat the current anytime-safe rule without losing formal safety?

## 5. Mandatory Tooling Strategy

Use available MCPs heavily. At a minimum, aggressively use:

- arXiv search and abstract pages,
- web fetch or browser tools for papers, blog posts, and project pages,
- Python execution for experiments and quick analysis,
- filesystem tools for writing code and notes,
- memory tools to record stable findings and failed ideas,
- code search for repo-local context.

If GitHub code search or repository tools are available and relevant, use them to locate reference implementations for:

- confidence sequences,
- stopping rules,
- change-point detection,
- sequence labeling of hidden states,
- uncertainty estimation,
- reward-model calibration,
- answer-revision tracking,
- trace parsing.

## 6. Literature Review Protocol

Do not stop at the handful of papers already cited. Build a structured literature map.

### 6.1 Paper categories to search exhaustively

Search across these clusters:

1. Overthinking, non-monotonic test-time scaling, early exit, adaptive reasoning, compute allocation.
2. Process reward models, verifiable process supervision, outcome reward models, reward hacking, BoN failure modes.
3. Belief revision in LLMs, answer revision dynamics, self-correction, reflection, multi-turn verification.
4. Sequential hypothesis testing, confidence sequences, optional stopping, change-point detection.
5. Optimal stopping, Snell envelopes, bandits with switching costs, stopping with partial observations.
6. Stochastic control, control-as-inference, free energy, risk-sensitive control, POMDPs.
7. Hidden Markov models, semi-Markov processes, switching state-space models, hazard models, survival analysis.
8. Information theory, entropy rate, KL drift, mutual information, predictive information, MDL, rate-distortion.
9. Dynamical systems, bifurcation theory, metastability, Lyapunov functions, attractors, catastrophe theory.
10. Sequential causal inference, counterfactual policy evaluation, treatment effect of one more reasoning step.
11. Calibration and uncertainty estimation, conformal methods, empirical Bernstein bounds, PAC-Bayes, e-processes.
12. Mechanistic interpretability and representation analysis relevant to answer revision and internal confidence.

### 6.2 For every promising source, extract

- citation metadata,
- claim summary,
- what mathematical object it optimizes or models,
- what observables it uses,
- what assumptions it makes,
- what failure modes it ignores,
- whether it offers a theorem, heuristic, or pure empirical result,
- what part of the current overthinking problem it can inform.

### 6.3 Deliverables from literature review

Create or update files such as:

- `research/literature_map.csv`
- `research/literature_synthesis.md`
- `research/bibliography.md`

If you choose different filenames, keep them under `research/` and make them discoverable.

## 7. Cross-Disciplinary Theory Sweep

Perform a broad theoretical sweep. Do not assume the current semimartingale model is final. For each domain below, ask whether it gives:

- a better state variable,
- a better hazard decomposition,
- a tighter stopping theorem,
- a better estimator,
- or a more falsifiable empirical signature.

### 7.1 Probability and stochastic processes

Investigate:

- martingales,
- semimartingales,
- Doob decomposition,
- supermartingale stopping,
- submartingale drift analysis,
- optional stopping caveats,
- confidence sequences,
- Ville's inequality,
- e-values and e-processes,
- self-normalized processes,
- empirical Bernstein bounds,
- mixture boundaries,
- time-uniform concentration,
- hidden Markov models,
- switching diffusions,
- semi-Markov models,
- Hawkes or self-exciting processes for reflective loops,
- branching processes for reasoning tree expansion,
- random walks with absorbing or reflecting boundaries.

Questions:

- Is overthinking best viewed as a drift sign change, a latent state switch, or a self-exciting corruption cascade?
- Can a confidence sequence be built for the sign of the continuation value itself?
- Can answer revisions be modeled as a marked point process with hazard spikes before degradation?

### 7.2 Statistics and sequential inference

Investigate:

- sequential testing,
- quickest change detection,
- CUSUM,
- Shiryaev-Roberts,
- generalized likelihood ratio detection,
- Bayesian online change-point detection,
- survival analysis,
- hazard regression,
- calibration curves,
- Bayesian filtering,
- variational inference,
- EM-style latent-state fitting,
- bootstrap uncertainty,
- semiparametric efficiency,
- conformal prediction,
- calibration under shift,
- multiple testing under sequential reuse.

Questions:

- Can the boundary be estimated as a change-point in the law of answer revisions?
- Can survival models capture the hazard of “staying correct if we continue for one more step”?
- Can conformal or e-process methods yield a safer practical stop rule than Hoeffding?

### 7.3 Applied mathematics and analysis

Investigate:

- optimal stopping,
- Snell envelopes,
- free-boundary problems,
- variational inequalities,
- dynamic programming,
- convex duality,
- measure-valued processes,
- singular perturbation,
- asymptotic analysis,
- perturbation of absorbing boundaries,
- stochastic approximation,
- Lyapunov stability,
- bifurcation and phase transitions.

Questions:

- Does the overthinking boundary admit a free-boundary characterization?
- Can the one-crossing assumption be derived from a more primitive structural assumption?
- Is there a Lyapunov function whose monotonicity fails exactly at overthinking onset?

### 7.4 Optimization and online learning

Investigate:

- bandits with cost per pull,
- online convex optimization,
- mirror descent views of belief updates,
- risk-sensitive optimization,
- stopping under switching costs,
- regret bounds,
- proximal control,
- control-as-inference,
- constrained optimization,
- budgeted decision processes.

Questions:

- Is one more reasoning step a budgeted action in a bandit-like problem?
- Can overthinking be cast as positive regret from over-allocation of compute?
- Can reward hacking be formalized as optimization against a misspecified surrogate objective with increasing horizon?

### 7.5 Information theory and geometry

Investigate:

- KL divergence over answer distributions,
- semantic entropy,
- entropy rate,
- predictive information,
- mutual information between current trace and final correctness,
- information bottleneck,
- description length,
- rate-distortion,
- information geometry,
- Fisher-Rao geometry,
- geodesic drift of hidden states,
- optimal transport distances between reasoning states.

Questions:

- Can the boundary be detected when additional tokens cease to add predictive information about correctness?
- Does hidden-state geometry flatten or curve sharply near overthinking onset?
- Is reward hacking associated with divergence between semantic progress and proxy-score progress?

### 7.6 Dynamical systems and physics-inspired views

Investigate:

- metastability,
- attractors,
- bifurcations,
- catastrophe theory,
- self-organized criticality,
- free energy,
- Ising-like analogies,
- path integral or action-like formulations if they become concrete rather than decorative.

Questions:

- Is there a metastable “correct reasoning basin” from which long traces escape?
- Does reward hacking look like a bifurcation in proxy-aligned versus truth-aligned dynamics?
- Are there measurable early-warning indicators such as variance inflation, slowing down, or critical fluctuations?

### 7.7 Machine learning and representation analysis

Investigate:

- hidden-state linear probes,
- sequence tagging over tokens,
- answer revision classifiers,
- uncertainty estimation,
- calibration heads,
- semantic clustering of partial answers,
- latent trajectory visualization,
- verifier-guided reflection,
- reward-model calibration,
- process supervision,
- mechanistic circuits for answer revision if feasible.

Questions:

- Can a small probe predict corruption hazard before the answer flips?
- Which representations best separate repair steps from corruption steps?
- Is there a consistent hidden-state direction associated with useless or harmful reflection?

### 7.8 Causality and decision science

Investigate:

- structural causal models,
- sequential interventions,
- counterfactual reasoning about truncation,
- uplift modeling,
- policy evaluation for stop versus continue,
- value of information,
- rational metareasoning,
- bounded rationality.

Questions:

- What is the causal effect of forcing one extra reasoning step on final correctness?
- Can the stop decision be framed as a value-of-computation problem with estimated treatment effects?
- Which observables are confounders, mediators, or colliders when modeling continuation utility?

## 8. Candidate Framework Generation and Scoring

Maintain a live hypothesis table. For each candidate framework, score it from 1 to 5 on:

- mathematical clarity,
- empirical observability,
- identifiability from available traces,
- compatibility with real-time stopping,
- robustness to reward hacking,
- ease of simulation,
- likelihood of producing a publishable theorem,
- likelihood of surviving contact with real data.

Candidate frameworks should include at least:

1. Current semimartingale drift-sign model.
2. Hidden-state change-point model.
3. Hazard-based survival model for corruption risk.
4. POMDP or control-as-inference model.
5. Value-of-information or rational metareasoning model.
6. Confidence-sequence or e-process detection model.
7. Information-gain exhaustion model.
8. Dynamical-systems metastability model.
9. Causal effect model for one-more-step intervention.

Do not keep weak ideas alive because they sound elegant. Kill them if they do not survive simulation or trace evidence.

## 9. Real-Model Experimental Program

The synthetic simulator is useful but insufficient. Move to actual open-weight traces.

### 9.1 Required experimental layers

Layer A: answer-level traces

- final answer at each reasoning step if extractable,
- first-answer time,
- answer revision count,
- whether the answer is currently correct,
- whether correctness is preserved after continuing.

Layer B: token-level diagnostics

- token logprobs if available,
- entropy,
- varentropy or variance-like uncertainty measures,
- transition-token markers,
- long-range repetition or self-echo features,
- semantic similarity to earlier reasoning segments.

Layer C: representation-level diagnostics

- late-layer hidden states,
- projection of hidden states onto learned probes,
- state-space distances over time,
- whether a simple linear classifier can separate repair from corruption steps.

Layer D: verifier and proxy diagnostics

- verifier correctness estimates,
- PRM or surrogate step scores if available,
- divergence between surrogate gain and correctness gain,
- evidence of reward hacking.

### 9.2 Task selection criteria

Prefer tasks with verifiable correctness:

- mathematics,
- coding,
- symbolic or formal reasoning,
- structured QA with exact-match or executable verification,
- theorem-style or constrained reasoning problems if tooling exists.

Use at least one set of tasks where long reasoning is often helpful and one where overthinking is frequent.

### 9.3 Controlled generation protocol

For each model and benchmark, vary:

- temperature,
- reasoning budget,
- stop-token or max-token limits,
- self-consistency count if used,
- verifier usage,
- reflection triggers.

Record enough metadata to make runs reproducible.

### 9.4 Minimum real-trace questions

You must estimate, at least approximately:

- the empirical probability that a currently wrong answer is repaired by one more step,
- the empirical probability that a currently correct answer is corrupted by one more step,
- whether corruption hazard rises with time,
- whether repair hazard decays with time,
- whether a proxy score continues to improve after correctness-minus-cost deteriorates.

## 10. Simulation Extension Program

Extend the current simulator rather than discarding it.

Add or consider:

- latent regime switching,
- hidden-state surrogate features,
- explicit answer-flip processes,
- nonstationary hazards conditioned on trace features,
- reward-model bias terms with delayed onset,
- heavier-tailed probe noise,
- dependence across probe samples,
- calibration mismatch,
- adversarial or reward-hacked proxy behavior,
- control interventions such as forced stop, forced continue, or reflection gating.

Add stronger baselines:

- entropy threshold,
- answer-stability threshold,
- first-answer stop,
- moving-average drift detector,
- change-point detector,
- confidence-sequence detector,
- any practical hidden-state detector if real models yield useful features.

## 11. Statistical Validation Requirements

For each empirical detector, report:

- average stop time,
- average value gap to oracle,
- false-early-stop rate,
- false-late-stop severity,
- robustness across models,
- robustness across tasks,
- calibration diagnostics,
- sensitivity to sampling noise,
- ablations on feature sets,
- computational overhead.

Whenever you propose a new stop rule, ask:

- Is the guarantee pointwise or sequential?
- Are samples independent enough for the bound being invoked?
- If not, can you replace the bound or explicitly weaken the claim?

## 12. Concrete Estimation Targets

Try to operationalize these latent quantities on real traces:

### 12.1 Correctness belief $q_t$

Possible estimators:

- verifier-calibrated probability,
- self-consistency-based estimate,
- hidden-state classifier calibrated to current-answer correctness,
- ensemble of verifier and model confidence signals.

### 12.2 Repair hazard $\alpha_t$

Possible estimators:

- transition model from wrong-at-$t$ to correct-at-$t+1$,
- survival or hazard model conditioned on current diagnostics,
- hidden-state or token-feature classifier for imminent repair.

### 12.3 Corruption hazard $\beta_t$

Possible estimators:

- transition model from correct-at-$t$ to wrong-at-$t+1$,
- revision-risk model for currently correct answers,
- instability score derived from hidden-state curvature or answer volatility.

### 12.4 Proxy bias $\kappa_t$

Possible estimators:

- difference between predicted proxy improvement and actual correctness-minus-cost improvement,
- residual from regressing PRM drift on estimated true drift,
- direct analysis of regimes where PRM rises while correctness or verifier confidence falls.

## 13. Theorem Development Targets

Try to prove at least one of the following beyond the current baseline theorem set:

1. A tighter anytime stop rule using a confidence sequence or e-process.
2. A theorem linking monotone hazard structure to one-crossing drift.
3. A result showing when PRM-only stopping is guaranteed to be inconsistent under positive bias.
4. A bound on regret or value loss from a practical stopping rule relative to the oracle.
5. A finite-sample guarantee for a change-point based stop detector.
6. A theorem connecting a real observable such as entropy curvature or answer volatility to the sign of $\mu_t$ under explicit assumptions.

If no new theorem survives scrutiny, say so explicitly and explain what blocked it.

## 14. Failure Modes to Search For

You are not allowed to only hunt for confirming evidence. Search actively for ways the current theory could be wrong.

Potential failure modes include:

- multiple boundary crossings,
- task-dependent non-monotone repair hazards,
- correctness that is not well captured by stepwise answer snapshots,
- hidden reasoning improvements without immediate answer changes,
- proxy scores that appear biased only because the true utility definition is too narrow,
- detector leakage from future information,
- dependence violations in probe construction,
- benchmark artifacts,
- model-specific quirks mistaken for general laws.

For every failure mode found, decide whether to:

- revise the theorem,
- refine the utility definition,
- restrict the scope of the claim,
- or abandon that line entirely.

## 15. Repository Deliverables

As you work, update the repo, not just chat.

At minimum, consider producing or updating:

- `research/overthinking_boundary.md`
- `research/simulate_overthinking_boundary.py`
- `research/outputs/summary.csv`
- `research/literature_map.csv`
- `research/literature_synthesis.md`
- `research/model_shortlist.md`
- `research/real_trace_experiments.py`
- `research/trace_analysis.py`
- `research/hypothesis_table.md`
- `research/open_questions.md`
- additional plots under `research/outputs/`

Do not create junk files. If a file is created, fill it with real content.

## 16. Operating Loop

Repeat the following loop until the research package is materially improved:

1. Read and summarize literature.
2. Propose 2 to 5 candidate mathematical hypotheses.
3. Rank and prune them.
4. Implement the most promising one in simulation or on real traces.
5. Compare against the current baseline.
6. If it fails, document why and pivot.
7. If it works, strengthen the theorem or broaden the experiment.
8. Update the theory note and outputs.

Never stop after a single success. Stress-test it.

## 17. Reporting Requirements

Your final research update should contain:

1. What new literature materially changed the view.
2. Which candidate theories were explored.
3. Which ones were rejected and why.
4. What new theorem, detector, or empirical result survived.
5. Which open-weight models were actually run.
6. Which models were planned but not run, and why.
7. What remains unresolved.

Provide concise research logs, not vague summaries.

## 18. Definition of Done

You are done only if most of the following are true:

- the literature review is substantially broader than the current note,
- at least one real open-weight model has been evaluated or a fully reproducible harness has been created with a clear explanation of why execution was blocked,
- the current semimartingale theory has either been strengthened or cleanly delimited,
- at least one improved practical stopping detector has been tested,
- the repo contains updated code and notes rather than only chat commentary,
- the final report clearly separates theorem-backed claims, simulation-backed claims, and conjectures.

## 19. First Actions

Start with this exact sequence unless the environment makes one step impossible:

1. Read the current theory note, simulator, and output summary.
2. Audit the available MCPs and Python environment.
3. Build a literature map covering at least 30 genuinely relevant sources before concluding the review is adequate.
4. Decide which open-weight models are feasible in this environment.
5. Create the experimental harness for real traces.
6. Revisit the mathematics in light of the broader field sweep.
7. Run simulations or real experiments.
8. Update the repo with the improved package.

## 20. Final Instruction

Treat this as a research sprint, not a brainstorming exercise. Use available MCPs heavily. Be ruthless about weak ideas. Preserve rigor. Advance the project materially.