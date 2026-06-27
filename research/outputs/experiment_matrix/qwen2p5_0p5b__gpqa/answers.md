# Answers to Open Questions

## Question 1
Can the DeepSeek 1.5B distill or an equivalent 1B-1.5B reasoning model be run with CUDA-enabled PyTorch or quantized inference so the real-trace study leaves the low-skill regime?

Yes. The completed Qwen2.5 instruct 0.5B L4 run used CUDA-backed transformers inference on 1344 runs covering 448 GSM8K tasks, with step-1 competence $q_1=0.204$ and at-least-once correctness in 398 runs. That clears the capability gate used for cross-family boundary claims, so this run leaves the low-skill regime and supports continuation-hazard estimation on real traces rather than toy tasks.

## Question 2
Can $q_t$ be estimated from hidden states or verifier-lite signals when exact stepwise verification is unavailable?

Provisionally yes. The correctness probe achieved mean Brier 0.2485 and mean AUC 0.5878, with verbosity-confidence proxy (verbose_confidence_proxy, coeff=-0.175) as the strongest signal. This run still uses exact GSM8K verification for supervision, so the evidence is about signal availability rather than full label-free deployment, but it is strong enough to justify a verifier-lite estimator.

## Question 3
Can $\alpha_t$ and $\beta_t$ be learned online from cross-task trace features well enough to support a practical stop rule?

Partially yes. The hazard-based stop rule reached mean oracle gap 0.0666 with false-late rate 0.112, while the empirical-Bernstein detector reached 0.4111, and the new mixture e-process reached 0.0544. The corrected conditional hazard drift crosses at step 2 and the raw empirical utility drift crosses at step 2, while the fitted hazard drift estimate crosses at step 2. The pooled repair and corruption hazards were 0.027 and 0.134, so the hazards are learnable enough to drive a practical detector, although still conservatively.

## Question 4
Can the empirical-Bernstein detector be replaced by a genuinely tighter mixture-bound or e-process construction without losing usability?

Partially yes. The implemented mixture e-process detector reduced mean oracle gap from 0.4111 under empirical-Bernstein to 0.0544, and reduced false-late rate from 0.996 to 0.000. It also improves on the fitted hazard rule at 0.0666, so the stronger sequential detector is currently the best pooled stop rule in the repo on this run.

## Question 5
Which observable is most stable across model families: entropy dynamics, answer revisions, hidden-state drift, or calibrated judge confidence?

Across the currently completed model families, the most stable currently supported observable is verbosity-confidence proxy (verbose_confidence_proxy, coeff=-0.175), while the strongest corruption-side signal is answer revision flag (answer_changed, coeff=0.844). True cross-family stability is still unsettled until another family is run at comparable scale, but the current run cleanly identifies the leading signals for this model.

## Question 6
Does reward hacking in real reasoning traces show up first as verbosity bias, confidence inflation, hidden-state drift, or verifier disagreement?

In the current traces it shows up earliest through answer revision flag (answer_changed, coeff=0.844). The corrected conditional hazard drift crosses zero at step 2, and the never-stop policy still loses 0.4626 utility on average. That pattern is more consistent with corruption through instability in the model's observable state than with harmless extra verification.

## Question 7
Are multiple drift crossings common on real traces, or is the one-crossing picture mostly correct once tasks are conditioned on difficulty?

The corrected conditional hazard curve is currently much closer to a one-crossing story than a repeated-crossing story: the first zero crossing occurs at step 2, and the aggregate corrected hazard sign changes 0 time(s). That supports the one-crossing picture at the population level, but the present artifact stack does not yet fit per-task latent-state crossing models, so repeated crossings cannot be ruled out on difficult outlier tasks.

## Question 8
How much of the apparent boundary is model-family specific versus benchmark specific?

Still unresolved from this cycle. The completed large run is concentrated on Qwen2.5 instruct 0.5B over GSM8K, so it identifies a boundary story for that model-benchmark pair but cannot yet cleanly decompose family effects from benchmark effects. A comparable higher-capability cross-family follow-up is still needed before attributing the boundary to model family rather than task distribution.
