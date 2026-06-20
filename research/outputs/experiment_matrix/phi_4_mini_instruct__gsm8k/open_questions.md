# Open Questions

Current status after the latest completed L4 run:

## 1. Can the DeepSeek 1.5B distill or an equivalent 1B-1.5B reasoning model be run with CUDA-enabled PyTorch or quantized inference so the real-trace study leaves the low-skill regime?

Answer: Yes. The completed Phi 4 4B L4 run used CUDA-backed transformers inference on 1500 runs covering 500 GSM8K tasks, with step-1 competence $q_1=0.333$ and at-least-once correctness in 1081 runs. That clears the capability gate used for cross-family boundary claims, so this run leaves the low-skill regime and supports continuation-hazard estimation on real traces rather than toy tasks.

## 2. Can $q_t$ be estimated from hidden states or verifier-lite signals when exact stepwise verification is unavailable?

Answer: Provisionally yes. The correctness probe achieved mean Brier 0.2314 and mean AUC 0.6365, with answer revision flag (answer_changed, coeff=-0.565) as the strongest signal. This run still uses exact GSM8K verification for supervision, so the evidence is about signal availability rather than full label-free deployment, but it is strong enough to justify a verifier-lite estimator.

## 3. Can $\alpha_t$ and $\beta_t$ be learned online from cross-task trace features well enough to support a practical stop rule?

Answer: Partially yes. The hazard-based stop rule reached mean oracle gap 0.2935 with false-late rate 0.459, while the empirical-Bernstein detector reached 0.6194, and the new mixture e-process reached 0.3407. The corrected conditional hazard drift crosses at step 2 and the raw empirical utility drift crosses at step 2, while the fitted hazard drift estimate crosses at step 3. The pooled repair and corruption hazards were 0.159 and 0.343, so the hazards are learnable enough to drive a practical detector, although still conservatively.

## 4. Can the empirical-Bernstein detector be replaced by a genuinely tighter mixture-bound or e-process construction without losing usability?

Answer: Partially yes. The implemented mixture e-process detector reduced mean oracle gap from 0.6194 under empirical-Bernstein to 0.3407, and reduced false-late rate from 0.981 to 0.644. It still trails the fitted hazard rule at 0.2935, so the stronger sequential detector exists now, but it is not yet the best practical stopping rule in the repo.

## 5. Which observable is most stable across model families: entropy dynamics, answer revisions, hidden-state drift, or calibrated judge confidence?

Answer: Within the current Phi 4 4B L4 run, the most stable currently supported observable is answer revision flag (answer_changed, coeff=-0.565), while the strongest corruption-side signal is answer revision flag (answer_changed, coeff=0.355). True cross-family stability is still unsettled until another family is run at comparable scale, but the current run cleanly identifies the leading signals for this model.

## 6. Does reward hacking in real reasoning traces show up first as verbosity bias, confidence inflation, hidden-state drift, or verifier disagreement?

Answer: In the current traces it shows up earliest through answer revision flag (answer_changed, coeff=0.355). The corrected conditional hazard drift crosses zero at step 2, and the never-stop policy still loses 0.6594 utility on average. That pattern is more consistent with corruption through instability in the model's observable state than with harmless extra verification.

## 7. Are multiple drift crossings common on real traces, or is the one-crossing picture mostly correct once tasks are conditioned on difficulty?

Answer: The corrected conditional hazard curve is currently much closer to a one-crossing story than a repeated-crossing story: the first zero crossing occurs at step 2, and the aggregate corrected hazard sign changes 0 time(s). That supports the one-crossing picture at the population level, but the present artifact stack does not yet fit per-task latent-state crossing models, so repeated crossings cannot be ruled out on difficult outlier tasks.

## 8. How much of the apparent boundary is model-family specific versus benchmark specific?

Answer: Still unresolved from this cycle. The completed large run is concentrated on Phi 4 4B over GSM8K, so it identifies a boundary story for that model-benchmark pair but cannot yet cleanly decompose family effects from benchmark effects. A comparable higher-capability cross-family follow-up is still needed before attributing the boundary to model family rather than task distribution.
