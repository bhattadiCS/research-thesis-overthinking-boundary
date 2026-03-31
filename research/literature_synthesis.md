# Literature Synthesis

## Main Shift

The literature no longer supports treating overthinking as a vague side effect of long chain-of-thought. The stronger papers now frame it as a compute-allocation failure, an uncertainty or path-deviation event, or a proxy-misalignment problem.

## Stable Findings

1. Longer reasoning is not monotone-helpful.

ROM, TERMINATOR, REFRAIN, EAT, DiffAdapt, ReBalance, ThinkPrune, and related papers all report substantial token savings with little or no accuracy loss once stopping becomes adaptive.

2. Hidden states and entropy are the most consistently useful observables.

ROM uses late hidden states directly. DiffAdapt, EAT, and path-deviation monitoring use entropy-based signals. CREST and latent pondering methods show that representation-space steering can suppress inefficient trajectories.

3. Reward proxies can improve outcomes and still remain misaligned.

Best-of-N, causal-rubric, and PRM papers agree that proxy optimization can remain practically useful while still admitting reward hacking or stylistic exploitation.

4. Risk control and stopping validity are still underdeveloped in the LLM literature.

Conformal Thinking is one of the few papers to phrase compute allocation as explicit risk control. The broader statistical literature on safe anytime-valid inference and e-values is still much sharper than most LLM stopping work.

## What This Means For This Repo

The literature supports a layered view:

- core object: continuation value or drift,
- latent decomposition: correctness belief plus repair and corruption hazards,
- observables: entropy, answer revisions, hidden-state change, verifier signals,
- safety layer: anytime-valid sequential inference.

That is exactly why the semimartingale hazard model survives this sweep better than a pure entropy rule, a pure path-deviation heuristic, or a general POMDP.

## The Most Useful Imports

- From adaptive reasoning papers: difficulty-aware compute allocation is real and operationally important.
- From overthinking detection papers: late hidden states and entropy spikes are actionable covariates.
- From reward-hacking papers: proxy optimism must be modeled explicitly.
- From statistics: time-uniform guarantees matter whenever a stop rule scans over time.

## What Still Looks Missing

- A strong theorem that ties a real observable directly to the sign of $\mu_t$.
- A practical online estimator for $\alpha_t$ and $\beta_t$ that survives model and task shift.
- A convincing real-model study on capable open reasoning models collected under dense instrumentation.
