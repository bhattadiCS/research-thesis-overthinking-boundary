# Technical Appendix: Dual-Boundary Interpretation

## Purpose

This appendix explains why the empirical thesis narrative now reports both a **first boundary** and a **late boundary**. The distinction is necessary because the observed plug-in drift can be nonmonotone even when the latent theory is written as a one-crossing stopping problem.

## 1. The Latent Boundary

The theorem-facing continuation value is

$$
\mu_t = (1-q_t)\alpha_t - q_t\beta_t - \lambda,
$$

with latent stopping boundary

$$
T_c = \inf\{t \ge 0 : \mu_t \le 0\}.
$$

If the true drift crosses zero once and stays negative, this object is unambiguous. In that idealized setting, one stopping boundary is enough.

## 2. Why The Empirical Audit Needs Two Boundaries

The empirical pipeline does not observe $\mu_t$ directly. It estimates $\widehat{\mu}_t$ from finite samples, pooled trajectories, and difficulty mixtures. As a result, the estimated drift path can be nonmonotone:

- an early step can look negative,
- a repair-dominant block can appear later,
- and the drift can finally collapse below zero again.

When that happens, collapsing the whole trace to the first negative value throws away the main scientific signal: whether the model ever entered a sustained positive-drift repair window.

## 3. Definitions Used In This Thesis

For an empirical drift sequence $\widehat{\mu}_1, \ldots, \widehat{\mu}_T$, define

$$
T_c^{first} = \inf\{t \ge 1 : \widehat{\mu}_t \le 0\}.
$$

This is the earliest negative-drift warning.

Next define the last positive step

$$
s^\star = \sup\{s \ge 1 : \widehat{\mu}_s > 0\},
$$

and the late boundary

$$
T_c^{late} = \inf\{t > s^\star : \widehat{\mu}_t \le 0\}.
$$

Equivalently, when a positive-drift block exists, $T_c^{late}$ is the first negative step after the **last** positive step. If the drift never returns positive after its first crossing, then $T_c^{late} = T_c^{first}$.

## 4. Interpretation Of The Three Stop Objects

The thesis now keeps three stopping objects separate:

1. **Latent boundary $T_c$**: the theorem-facing optimal stopping object defined by the true hazards.
2. **First empirical boundary $T_c^{first}$**: the earliest plug-in warning that continuation may already be harmful.
3. **Late empirical boundary $T_c^{late}$**: the final scientifically usable repair window, defined by the last positive-to-negative crossing.

In deployment, a fourth object still matters:

4. **Safe stopping time $\tau_{safe}$**: the online stop produced by an anytime-valid upper bound or e-process rule.

The thesis does **not** claim that one should deploy $T_c^{late}$ directly without uncertainty control. The claim is that $T_c^{late}$ is the right empirical witness for whether a family or stratum truly exhibits a late repair regime.

## 5. Why $T_c^{late}$ Is The Main Scientific Witness

The scientific question is not merely whether a noisy early estimate can dip below zero once. The scientific question is whether additional reasoning remains useful for some nontrivial interval.

If a slice shows

- an early negative estimate,
- a middle block with positive continuation value,
- and a later final collapse,

then $T_c^{first}$ is too conservative to summarize the effective repair window. In that setting, $T_c^{late}$ better captures the final point at which another reasoning step still has positive expected value.

## 6. Examples From The Current Workspace

The `Medium` difficulty stratum is the clearest place to see the distinction:

| Model | Medium n | $T_c^{first}$ | $T_c^{late}$ | $\alpha/\beta$ | Step-1 acc | Peak acc | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen 0.5B | 51 | 2 | 2 | 0.774 | 25.5% | 32.7% | Small recoverable subset, no extended late window |
| DeepSeek 1.5B | 194 | 2 | 2 | 0.381 | 12.5% | 30.9% | Early repair pulse but no durable late regime |
| Mistral 7B | 136 | 1 | 3 | 0.435 | 17.9% | 32.6% | Genuine but short late repair window |
| Qwen 7B | 194 | 1 | 6 | 0.918 | 15.1% | 75.4% | Strong late repair window |

Two examples matter most for the defense:

- **Qwen 7B medium**: the first negative estimate occurs at Step 1, but the drift becomes strongly positive over the next several steps and only finally crosses back below zero at Step 6. This is the clearest example of why a first-crossing summary alone is inadequate.
- **Mistral 7B medium**: the same qualitative shape appears, but the repair block is shorter and weaker, so the late boundary arrives at Step 3 rather than Step 6.

Hard strata collapse immediately and therefore do not need a dual-boundary treatment. Easy strata can also show late boundaries, but they are less diagnostic because many of those tasks are already solved at Step 1.

## 7. Recommended Reporting Convention

For thesis writing and defense slides, use the following convention:

1. Report the aggregate corrected boundary for each family.
2. Report both $T_c^{first}$ and $T_c^{late}$ for the `Medium` stratum.
3. Use the difficulty-stratified drift grid together with the alpha/beta scatter to show both the mechanism and the predictive relationship.
4. Keep the deployable anytime-valid stopping rule separate from the scientific late-boundary summary.

## 8. Short Slide Version

If this appendix has to be compressed into one defense slide, the message is:

- The latent theory uses one stopping boundary.
- The empirical drift can be nonmonotone.
- Therefore the audit reports both an early warning ($T_c^{first}$) and the final usable repair window ($T_c^{late}$).
- Qwen 7B medium and Mistral 7B medium are the canonical examples of why the distinction matters.