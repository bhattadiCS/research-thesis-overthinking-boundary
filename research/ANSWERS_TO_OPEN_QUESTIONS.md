# Answers to Open Questions

## Question 1
Can the DeepSeek 1.5B distill or an equivalent 1B-1.5B reasoning model be run with CUDA-enabled PyTorch or quantized inference so the real-trace study leaves the low-skill regime?

Yes. The completed L4 run used CUDA-backed transformers inference on 900 runs covering 300 GSM8K tasks, with step-1 competence $q_1=0.237$ and at-least-once correctness in 621 runs. That is enough to leave the low-skill regime and estimate continuation hazards on real traces rather than toy tasks.

## Question 2
Can $q_t$ be estimated from hidden states or verifier-lite signals when exact stepwise verification is unavailable?

Provisionally yes. The correctness probe achieved mean Brier 0.2217 and mean AUC 0.6137, with answer revision flag (answer_changed, coeff=-0.618) as the strongest signal. This run still uses exact GSM8K verification for supervision, so the evidence is about signal availability rather than full label-free deployment, but it is strong enough to justify a verifier-lite estimator.

## Question 3
Can $lpha_t$ and $eta_t$ be learned online from cross-task trace features well enough to support a practical stop rule?

Partially yes. The hazard-based stop rule reached mean oracle gap 0.4121 with false-late rate 0.733, while the empirical-Bernstein detector reached 0.7141. The pooled repair and corruption rates were 0.189 and 0.461, so the hazards are learnable enough to drive a practical detector, although still conservatively.

## Question 4
Can the empirical-Bernstein detector be replaced by a genuinely tighter mixture-bound or e-process construction without losing usability?

Still unresolved. In the current run the empirical-Bernstein rule achieved mean oracle gap 0.7141, which improves materially over never-stop at 0.7463 but still trails the fitted hazard rule at 0.4121. No mixture-bound or e-process detector was implemented and validated in this cycle, so the tighter-safe replacement remains an open follow-up rather than a completed result.

## Question 5
Which observable is most stable across model families: entropy dynamics, answer revisions, hidden-state drift, or calibrated judge confidence?

Within the current deepseek 1.5b l4 run, the most stable currently supported observable is answer revision flag (answer_changed, coeff=-0.618), while the strongest corruption-side signal is answer revision flag (answer_changed, coeff=0.396). That keeps hidden-state drift, entropy, and verbosity-linked signals in the lead, but true cross-family stability is not settled until a stronger second family is run at comparable scale.

## Question 6
Does reward hacking in real reasoning traces show up first as verbosity bias, confidence inflation, hidden-state drift, or verifier disagreement?

In the current traces it shows up earliest through answer revision flag (answer_changed, coeff=0.396). The hazard drift crosses zero at step 7, and the never-stop policy still loses 0.7463 utility on average. That pattern is more consistent with corruption through unstable internal state and verbosity-linked overrun than with harmless extra verification.

## Question 7
Are multiple drift crossings common on real traces, or is the one-crossing picture mostly correct once tasks are conditioned on difficulty?

The pooled hazard curve is currently much closer to a one-crossing story than a repeated-crossing story: the first zero crossing occurs at step 7, and the aggregate hazard sign changes 2 time(s). That supports the one-crossing picture at the population level, but the present artifact stack does not yet fit per-task latent-state crossing models, so repeated crossings cannot be ruled out on difficult outlier tasks.

## Question 8
How much of the apparent boundary is model-family specific versus benchmark specific?

Still unresolved from this cycle. The completed large run is concentrated on DeepSeek 1.5B over GSM8K, so it identifies a real boundary for that model-benchmark pair but cannot yet decompose family effects from benchmark effects. A comparable Qwen, Llama, or larger DeepSeek follow-up is still needed before attributing the boundary to model family rather than task distribution.
