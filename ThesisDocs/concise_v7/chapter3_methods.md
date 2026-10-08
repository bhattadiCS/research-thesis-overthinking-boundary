# Chapter 3 Experimental methods

## 3.1 Corpora, models and labels

Two separately defined collections support the analysis. The variable-horizon model-domain matrix contains 798,770 raw saved rows and 75,965 sanitized trajectories over 52 cells. Its boundary and matched policy analyses distinguish malformed raw records from eligible trajectories. The standardized detector corpus has 144,440 rows, 28,888 five-response trajectories and 2,948 task identifiers (Table 1). The collections overlap in benchmark questions and are not independent replications; their counts must not be added.

[[CORPUS_TABLE]]

The thirteen-model matrix includes DeepSeek-R1-Distill-Qwen-1.5B and 7B; Qwen2.5-0.5B, 3B, 7B, 14B and 32B Instruct; InternLM3-8B-Instruct; Llama-3.1-8B-Instruct; Mistral-7B-Instruct-v0.3; Mistral-Small-Instruct-2409; Phi-4-mini-instruct; and Yi-1.5-9B-Chat. The detector roster replaces InternLM3-8B with Qwen3.5-9B. The historical `mistral_small_24b_2409` alias denotes the recorded 22B model. Cell allocations are unequal, so totals cannot be inferred by multiplying task count, roster size and horizon. Architecture, training and tokenization are not randomized; a family or scale association is not an isolated parameter-count effect.

GSM8K, MATH, ARC-Challenge and GPQA define the four task domains [Cobbe2021], [Hendrycks2021], [Clark2018], [Rein2023]. Standardized GPQA metadata requests `test`, but the current loader uses `train` and saved identities use `gpqa_main`. Its effective split is inferred because the executed historical loader is unrecorded. Table 1 reports both interpretations. Detector task holdouts can come from benchmark training splits and are distinct from official benchmark test evaluation.

Each increment is a complete generated response followed, if permitted, by a revision prompt. Grading uses numeric extraction, symbolic-equivalence checks or the displayed MCQ ordering. The primary historical tables retain archived labels; predictor training separately versions candidate reconstruction and regrading. Parser regression coverage addresses specified cases, while corpus-wide semantic label validity remains unestablished.

The canonical matrix registry records generation seed 7, dataset shuffle seed 17, temperatures 0.1, 0.6 and 1.0, and a 256-token completion cap. Recorded horizons are ten responses for GSM8K and GPQA, fourteen for MATH and eight for ARC; individual cell metadata takes precedence for deviations or recovery events. The standardized detector corpus uses five saved responses and records temperature 0.6 and seed 7. These settings specify successive revision, not additional fragments of one uninterrupted thought. Complete prompts, task identities, references and displayed MCQ orderings remain attached to the frozen evidence.

## 3.2 Experimental units and comparisons

Rows within a trajectory and trajectories sharing a task are dependent. Source-qualified cell/run keys identify trajectories; task-disjoint folds keep a question's rows together. Strict tabular, text and portable-prefix fits follow their recorded task-disjoint contracts. Some earlier hazard probes instead use upstream run-group folds followed by task-group threshold folds, allowing other-temperature versions of a question into upstream fitting. Their controlled contrasts retain development scope.

Matched estimator comparisons resample the recorded 52 cells; paired systems comparisons and population transitions resample task clusters, conditional on the fixed model panel. One generation seed per setting leaves seed variability largely unmeasured. Repeated configuration selection can also turn a held-out panel into development data. Ten thousand bootstrap draws resample existing observations rather than create independent generations.

Accuracy is the mean grader label of the selected answer. Index $i$ denotes a trajectory, and $L_{i,s}$ its recorded completion tokens at response $s$. Historical step utility is $C_{i,\tau_i}-0.05(\tau_i-1)$; token utility uses $C_{i,\tau_i}-0.0002\sum_{s=1}^{\tau_i}L_{i,s}$. The mandatory first response is common in paired step-utility comparisons; total token accounting charges it. Thus 250 completion tokens carry the same penalty as one response increment under this convention. Actual savings use one minus the ratio of total stopped completion tokens to total baseline completion tokens. This weights longer generations more heavily than the mean per-question saving. Prompt processing, scoring, peers, timing and energy are distinct costs; a stopped-step index alone cannot establish physical savings.

ROC-AUC assesses ranking, Brier loss probability error. Causal sequence baselines include gated recurrent units [Cho2014] and transformers with rotary position embeddings [Su2021]. High pooled AUC or marginal calibration does not establish conditional continuation probabilities, a unique architecture winner or useful stopping utility.

## 3.3 Runtime policies and fitting

The controller consumes completed responses in order, enforces a two-response floor and five-response horizon, records selected answers and costs, and rejects later input after stopping. The adapter requests another response only after a continue decision, so termination prevents future model calls. Reference labels and future-step scores are excluded. Optional peer observations require a completed same-step roster and charged generation costs; no substantial live thirteen-peer experiment is reported.

The floor was selected during development and can exclude answers correct only at step one. It is a protocol choice, not a theorem that every task needs revision. The learned controller first charges and records the completed response. Before the floor it continues; at the horizon it stops without querying a next-step head. At intermediate eligible steps it evaluates the fitted drift and requests another response only when that drift is positive.

A frozen confidence/stability heuristic is compared with the full horizon. Confidence is uncalibrated; optional prior-answer retention branches did not fire in the live panels. The learned rule selects the latest nonempty candidate and estimates current correctness $\widehat q_t$ and next-selected correctness $\widehat p_{t+1}$. It stops at the first eligible nonpositive $\widehat p_{t+1}-\widehat q_t-0.05$, or at the horizon. This fitted myopic rule is not the Bellman policy.

The heuristic's stability branch requires valid parsing, confidence at least 90/100 and the same normalized answer in two consecutive responses. From step three, a valid observation can also trigger a 15-point confidence drop, retaining the preceding candidate, or two changes among valid answers, retaining the highest-confidence candidate with earliest-step tie breaking. No peers are required in these single-model runs. The learned policy uses reward one, wrong-answer penalty zero and the stated 0.05 continuation cost.

Fitting uses 1,500 archived Qwen2.5-0.5B GSM8K/MATH trajectories, comprising 7,500 rows. Task hashes allocate 902 tasks to fitting, 322 to calibration and 276 to evaluation. Scalers and unweighted logistic heads use fitting tasks; Platt mappings use calibration tasks [Platt1999]. All rows of a task share a partition, and the live/trap identities are excluded. Prefix features include step, tokens, answer changes, thought-text summaries and domain, without future-trajectory filtering.

The portable artifact contains 21 features, training-only standardization, both logistic coefficient vectors and separate one-dimensional Platt calibrators. Its current and next-step labels use the same latest-nonempty answer selector. Missing confidence is represented explicitly. The serialized parameters and feature transformation determine runtime scoring; evaluation tasks fit neither head nor calibrator.

Reconstruction changes 4,269 candidate strings but only 43 correctness labels; the versioned training audit preserves the original corpus. None of the 7,500 archive outputs meets the live JSON contract, leaving confidence and strict-parsing features without archive variation. Live prompts and EOS-inclusive token accounting differ. Archive calibration therefore does not establish transport to live generation.

## 3.4 Paired evaluation and uncertainty

Actual paired executions use Qwen2.5-0.5B-Instruct [Qwen2024] on 100 GSM8K questions and twenty prespecified traps, with separately stored public prompts, reference keys and event ledgers. Tasks already observed during corpus development remain development tasks. Active/full arms are executed separately; shared-prefix agreement is measured because batching and kernels can change greedy outputs.

All four live collections record greedy generation, temperature zero, seed 20261002, maximum batch size 32 and at most 128 newly generated tokens per response. A response ends at the first strictly valid complete JSON object, EOS or the token cap. The model snapshot is `7ae557604adf67be50417f59c2c2f167def9a775`; manifests bind its files and executed source copies. EOS-inclusive live token counts differ from the archive convention. Learned arms reuse the previously collected full-horizon baseline, as marked in Table 3.

For $n\ge1$ iid task pairs, let $I$ flag baseline incorrect/active correct and $W$ baseline correct/active incorrect. Write $\pi_I=\Pr(I=1)$ and $\pi_W=\Pr(W=1)$, so $\delta=\pi_I-\pi_W$ is the accuracy difference. Their marginal counts are binomial but the within-pair indicators are not assumed independent. Exact two-sided 97.5% Clopper-Pearson bounds $[L_I,U_I]$ and $[L_W,U_W]$ for the respective marginal probabilities [Clopper1934] have joint coverage at least 95% by a union bound; subtraction gives

$$\delta\in[L_I-U_W,\ U_I-L_W].$$

No discordances still leave nonzero uncertainty, with each marginal upper bound $1-0.0125^{1/n}$. No noninferiority margin was prespecified. Token intervals bootstrap paired tasks and the ratio of totals. The handpicked traps form a fixed challenge bank; iid-reference intervals do not provide randomized coverage of an adversarial population.
