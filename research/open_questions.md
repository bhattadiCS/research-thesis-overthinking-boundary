# Open Questions

1. Can the DeepSeek 1.5B distill or an equivalent 1B-1.5B reasoning model be run with CUDA-enabled PyTorch or quantized inference so the real-trace study leaves the low-skill regime?
2. Can $q_t$ be estimated from hidden states or verifier-lite signals when exact stepwise verification is unavailable?
3. Can $\alpha_t$ and $\beta_t$ be learned online from cross-task trace features well enough to support a practical stop rule?
4. Can the empirical-Bernstein detector be replaced by a genuinely tighter mixture-bound or e-process construction without losing usability?
5. Which observable is most stable across model families: entropy dynamics, answer revisions, hidden-state drift, or calibrated judge confidence?
6. Does reward hacking in real reasoning traces show up first as verbosity bias, confidence inflation, hidden-state drift, or verifier disagreement?
7. Are multiple drift crossings common on real traces, or is the one-crossing picture mostly correct once tasks are conditioned on difficulty?
8. How much of the apparent boundary is model-family specific versus benchmark specific?
