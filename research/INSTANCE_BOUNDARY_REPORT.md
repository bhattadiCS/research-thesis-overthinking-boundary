# Instance-Level Boundary Estimation

## Overview

A central finding in recent research literature on iterative reasoning is that optimal reasoning termination points are highly instance-specific (Wei et al. 2025, *Evolution of Thought*). Generating a single summary boundary step across an entire dataset forces temporal smoothing and obscures valid sub-populations of tasks requiring extended computation.

Phase B constructed per-problem step features exactly equivalent to the macro $\mu_t$ analysis, yielding single-task boundaries.

## Findings

For every problem ($N=300$) inside each model scope, we computed a continuation boundary $T_c$.

| Model | Mean $T_c$ | Median $T_c$ | Std. Dev. $T_c$ | Early ($\leq 2$) | Late ($\geq 5$) |  Never Cross |
|---|---|---|---|---|---|---|
| Qwen 0.5B | 1.04 | 1.0 | 0.20 | 99.6% | 0.0% | 0.0% |
| Mistral 7B | 1.15 | 1.0 | 0.41 | 98.7% | 0.0% | 0.0% |
| DeepSeek 1.5B | 1.28 | 1.0 | 0.51 | 97.3% | 0.0% | 0.0% |
| Qwen 7B | 1.11 | 1.0 | 0.38 | 98.3% | 0.0% | 0.0% |

### Analysis of Data Limitations

The table above illustrates a critical limitation of the per-instance protocol: instance boundaries aggressively collapse to $T_c=1$. **This does not falsify instance-specific boundaries.** Instead, it originates from sampling constraints: each problem features only 3 observation runs (T=0.1, 0.6, 1.0).

With only three observations:
1. Transition sets between step $t$ and $t+1$ typically possess $N \in [0, 3]$.
2. Zero or one active path provides insufficient variance to yield meaningful, stabilized estimates of repair ($\alpha_t$) or corruption ($\beta_t$).
3. The resultant boundary systematically truncates at the boundary floor.

## Conclusion

The per-problem resolution analysis proved too granular for the data volume retrieved inside the automated GPU cycle. Granularity must default to the difficulty-stratification level analyzed in Phase A, which retains an average of $N=100-200$ problems ($300-600$ observations) per subset path, permitting stabilized conditional probability metrics.

True instance-specific measurements would require a dataset re-scoping where observation iterations scale proportional to $N_{runs} \geq 20$ paths per unique task prompt to ensure dense feature representations.
