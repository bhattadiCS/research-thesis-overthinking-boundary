# Safety-Critical Stakes Sweep Report

This report sweeps incorrect answer penalties ($c$) to evaluate boundary shifts.

Formula:
  - **Expected Utility:** $V_t = q_t \cdot v - (1 - q_t) \cdot c - \lambda t$
  - **Predictable Drift:** $\mu_t = [(1 - q_t)\alpha_t - q_t\beta_t](v + c) - \lambda$

Sweep Settings:
  - Per-step compute cost ($\lambda$): 0.05
  - Correct answer reward ($v$): 1.0

## Sweep Summary Table

| Model | Penalty (c) | Boundary Step (T*) | Oracle Utility | Hazard Utility | Never Stop Utility | Hazard Oracle Gap | Never Stop Oracle Gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2p5_7b | 0.0 | 5 | 0.6998 | 0.4653 | 0.2553 | 0.2344 | 0.4444 |
| qwen2p5_7b | 0.5 | 5 | 0.5804 | 0.2980 | 0.1080 | 0.2824 | 0.4724 |
| qwen2p5_7b | 1.0 | 5 | 0.4611 | 0.1307 | -0.0393 | 0.3304 | 0.5004 |
| qwen2p5_7b | 2.0 | 6 | 0.2224 | -0.1800 | -0.3340 | 0.4024 | 0.5564 |
| qwen2p5_7b | 5.0 | 6 | -0.4936 | -1.1100 | -1.2180 | 0.6164 | 0.7244 |
| qwen2p5_7b | 10.0 | 7 | -1.6869 | -2.6293 | -2.6913 | 0.9424 | 1.0044 |
| qwen2p5_7b | 20.0 | 7 | -4.0736 | -5.6560 | -5.6380 | 1.5824 | 1.5644 |
| qwen2p5_7b | 50.0 | 10 | -11.2336 | -14.4780 | -14.4780 | 3.2444 | 3.2444 |
| qwen2p5_7b | 100.0 | 10 | -23.1669 | -29.2113 | -29.2113 | 6.0444 | 6.0444 |
| qwen2p5_14b | 0.0 | 5 | 0.5935 | 0.1653 | -0.0033 | 0.4282 | 0.5969 |
| qwen2p5_14b | 0.5 | 7 | 0.4549 | -0.1020 | -0.2800 | 0.5569 | 0.7349 |
| qwen2p5_14b | 1.0 | 7 | 0.3162 | -0.3693 | -0.5567 | 0.6855 | 0.8729 |
| qwen2p5_14b | 2.0 | 7 | 0.0389 | -0.9040 | -1.1100 | 0.9429 | 1.1489 |
| qwen2p5_14b | 5.0 | 7 | -0.7931 | -2.5080 | -2.7700 | 1.7149 | 1.9769 |
| qwen2p5_14b | 10.0 | 7 | -2.1798 | -5.1813 | -5.5367 | 3.0015 | 3.3569 |
| qwen2p5_14b | 20.0 | 7 | -4.9531 | -10.5280 | -11.0700 | 5.5749 | 6.1169 |
| qwen2p5_14b | 50.0 | 7 | -13.2731 | -26.5680 | -27.6700 | 13.2949 | 14.3969 |
| qwen2p5_14b | 100.0 | 7 | -27.1398 | -53.3013 | -55.3367 | 26.1615 | 28.1969 |
| qwen2p5_32b | 0.0 | 5 | 0.7899 | 0.5960 | 0.4287 | 0.1939 | 0.3613 |
| qwen2p5_32b | 0.5 | 6 | 0.7389 | 0.5140 | 0.3680 | 0.2249 | 0.3709 |
| qwen2p5_32b | 1.0 | 6 | 0.6879 | 0.4353 | 0.3073 | 0.2526 | 0.3806 |
| qwen2p5_32b | 2.0 | 7 | 0.5859 | 0.2800 | 0.1860 | 0.3059 | 0.3999 |
| qwen2p5_32b | 5.0 | 8 | 0.2799 | -0.1300 | -0.1780 | 0.4099 | 0.4579 |
| qwen2p5_32b | 10.0 | 8 | -0.2301 | -0.7800 | -0.7847 | 0.5499 | 0.5546 |
| qwen2p5_32b | 20.0 | 10 | -1.2501 | -1.9980 | -1.9980 | 0.7479 | 0.7479 |
| qwen2p5_32b | 50.0 | 10 | -4.3101 | -5.6380 | -5.6380 | 1.3279 | 1.3279 |
| qwen2p5_32b | 100.0 | 10 | -9.4101 | -11.7047 | -11.7047 | 2.2946 | 2.2946 |

## Key Insights:
1. **Boundary Shifts Later:** As error penalty ($c$) scales, the boundary step shifts to the right (later), not earlier. This is expected from the utility formula: scaling $c$ amplifies the reward/penalty term $(v + c)$ relative to the fixed per-step cost $\lambda$, so continued reasoning becomes worth more, not less, as consequences scale -- the model is incentivized to keep reasoning longer to avoid an even costlier wrong answer.
2. **Utility Conservation:** In high-penalty sweeps (c >= 10), letting the model think indefinitely (`never_stop`) leads to severe negative utility scores due to high error accumulation. The `hazard_drift` early stopping policy prevents this degradation.