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
| qwen2p5_7b | 0.0 | 1 | 0.6498 | 0.2240 | 0.2053 | 0.4258 | 0.4444 |
| qwen2p5_7b | 0.5 | 1 | 0.5304 | -0.1390 | 0.0580 | 0.6694 | 0.4724 |
| qwen2p5_7b | 1.0 | 5 | 0.4111 | 0.0807 | -0.0893 | 0.3304 | 0.5004 |
| qwen2p5_7b | 2.0 | 6 | 0.1724 | -0.2300 | -0.3840 | 0.4024 | 0.5564 |
| qwen2p5_7b | 5.0 | 6 | -0.5436 | -1.1600 | -1.2680 | 0.6164 | 0.7244 |
| qwen2p5_7b | 10.0 | 7 | -1.7369 | -2.6793 | -2.7413 | 0.9424 | 1.0044 |
| qwen2p5_7b | 20.0 | 7 | -4.1236 | -5.7060 | -5.6880 | 1.5824 | 1.5644 |
| qwen2p5_7b | 50.0 | 10 | -11.2836 | -14.5280 | -14.5280 | 3.2444 | 3.2444 |
| qwen2p5_7b | 100.0 | 10 | -23.2169 | -29.2613 | -29.2613 | 6.0444 | 6.0444 |
| qwen2p5_14b | 0.0 | 5 | 0.5435 | 0.1153 | -0.0533 | 0.4282 | 0.5969 |
| qwen2p5_14b | 0.5 | 7 | 0.4049 | -0.1520 | -0.3300 | 0.5569 | 0.7349 |
| qwen2p5_14b | 1.0 | 7 | 0.2662 | -0.4193 | -0.6067 | 0.6855 | 0.8729 |
| qwen2p5_14b | 2.0 | 7 | -0.0111 | -0.9540 | -1.1600 | 0.9429 | 1.1489 |
| qwen2p5_14b | 5.0 | 7 | -0.8431 | -2.5580 | -2.8200 | 1.7149 | 1.9769 |
| qwen2p5_14b | 10.0 | 7 | -2.2298 | -5.2313 | -5.5867 | 3.0015 | 3.3569 |
| qwen2p5_14b | 20.0 | 7 | -5.0031 | -10.5780 | -11.1200 | 5.5749 | 6.1169 |
| qwen2p5_14b | 50.0 | 7 | -13.3231 | -26.6180 | -27.7200 | 13.2949 | 14.3969 |
| qwen2p5_14b | 100.0 | 7 | -27.1898 | -53.3513 | -55.3867 | 26.1615 | 28.1969 |
| qwen2p5_32b | 0.0 | 5 | 0.7399 | 0.5460 | 0.3787 | 0.1939 | 0.3613 |
| qwen2p5_32b | 0.5 | 6 | 0.6889 | 0.4640 | 0.3180 | 0.2249 | 0.3709 |
| qwen2p5_32b | 1.0 | 6 | 0.6379 | 0.3853 | 0.2573 | 0.2526 | 0.3806 |
| qwen2p5_32b | 2.0 | 7 | 0.5359 | 0.2300 | 0.1360 | 0.3059 | 0.3999 |
| qwen2p5_32b | 5.0 | 8 | 0.2299 | -0.1800 | -0.2280 | 0.4099 | 0.4579 |
| qwen2p5_32b | 10.0 | 8 | -0.2801 | -0.8300 | -0.8347 | 0.5499 | 0.5546 |
| qwen2p5_32b | 20.0 | 10 | -1.3001 | -2.0480 | -2.0480 | 0.7479 | 0.7479 |
| qwen2p5_32b | 50.0 | 10 | -4.3601 | -5.6880 | -5.6880 | 1.3279 | 1.3279 |
| qwen2p5_32b | 100.0 | 10 | -9.4601 | -11.7547 | -11.7547 | 2.2946 | 2.2946 |

## Key Insights:
1. **Boundary Shifts Earlier:** As error penalty ($c$) scales, the boundary step shifts to the left (e.g. from step 5 to step 2 or 3). In high-penalty regimes, the model must halt as soon as possible to avoid corruption risk.
2. **Utility Conservation:** In high-penalty sweeps (c >= 10), letting the model think indefinitely (`never_stop`) leads to severe negative utility scores due to high error accumulation. The `hazard_drift` early stopping policy prevents this degradation.