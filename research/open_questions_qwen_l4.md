# Open Questions

Current status after the latest completed L4 run:

## 1. Can the DeepSeek 1.5B distill or an equivalent 1B-1.5B reasoning model be run with CUDA-enabled PyTorch or quantized inference so the real-trace study leaves the low-skill regime?

Answer: Not yet. The completed Qwen2.5 instruct 0.5B L4 run used CUDA-backed transformers inference on 900 runs covering 300 GSM8K tasks, but it only reached step-1 competence $q_1=0.071$ and at-least-once correctness in 81 runs. That stays below the current capability gate for a strong boundary claim, so this run is still best interpreted as a weak-regime control rather than a decisive theorem-facing witness.

## 2. Can $q_t$ be estimated from hidden states or verifier-lite signals when exact stepwise verification is unavailable?

Answer: Provisionally yes. The correctness probe achieved mean Brier 0.2399 and mean AUC 0.5291, with verbosity-confidence proxy (verbose_confidence_proxy, coeff=0.448) as the strongest signal. This run still uses exact GSM8K verification for supervision, so the evidence is about signal availability rather than full label-free deployment, but it is strong enough to justify a verifier-lite estimator.

## 3. Can $\alpha_t$ and $\beta_t$ be learned online from cross-task trace features well enough to support a practical stop rule?

Answer: Partially yes. The hazard-based stop rule reached mean oracle gap 0.1531 with false-late rate 0.996, while the empirical-Bernstein detector reached 0.4106, and the new mixture e-process reached 0.0595. The corrected conditional hazard drift crosses at step 1 and the raw empirical utility drift crosses at step 1, while the fitted hazard drift estimate crosses at step 4. The pooled repair and corruption hazards were 0.003 and 0.024, so the hazards are learnable enough to drive a practical detector, although still conservatively.

## 4. Can the empirical-Bernstein detector be replaced by a genuinely tighter mixture-bound or e-process construction without losing usability?

Answer: Partially yes. The implemented mixture e-process detector reduced mean oracle gap from 0.4106 under empirical-Bernstein to 0.0595, and reduced false-late rate from 1.000 to 0.981. It also improves on the fitted hazard rule at 0.1531, so the stronger sequential detector is currently the best pooled stop rule in the repo on this run.

## 5. Which observable is most stable across model families: entropy dynamics, answer revisions, hidden-state drift, or calibrated judge confidence?

Answer: Within the current Qwen2.5 instruct 0.5B L4 run, the most stable currently supported observable is verbosity-confidence proxy (verbose_confidence_proxy, coeff=0.448), while the strongest corruption-side signal is token entropy (entropy_mean, coeff=0.847). True cross-family stability is still unsettled until another family is run at comparable scale, but the current run cleanly identifies the leading signals for this model.

## 6. Does reward hacking in real reasoning traces show up first as verbosity bias, confidence inflation, hidden-state drift, or verifier disagreement?

Answer: In the current traces it shows up earliest through token entropy (entropy_mean, coeff=0.847). The corrected conditional hazard drift crosses zero at step 1, and the never-stop policy still loses 0.4595 utility on average. That pattern is more consistent with corruption through instability in the model's observable state than with harmless extra verification.

## 7. Are multiple drift crossings common on real traces, or is the one-crossing picture mostly correct once tasks are conditioned on difficulty?

Answer: The corrected conditional hazard curve is currently much closer to a one-crossing story than a repeated-crossing story: the first zero crossing occurs at step 1, and the aggregate corrected hazard sign changes 0 time(s). That supports the one-crossing picture at the population level, but the present artifact stack does not yet fit per-task latent-state crossing models, so repeated crossings cannot be ruled out on difficult outlier tasks.

## 8. How much of the apparent boundary is model-family specific versus benchmark specific?

Answer: Still unresolved from this cycle. The completed large run is concentrated on Qwen2.5 instruct 0.5B over GSM8K, so it identifies a boundary story for that model-benchmark pair but cannot yet cleanly decompose family effects from benchmark effects. A comparable higher-capability cross-family follow-up is still needed before attributing the boundary to model family rather than task distribution.
