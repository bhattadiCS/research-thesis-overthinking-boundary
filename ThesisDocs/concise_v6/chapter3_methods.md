# Chapter 3 Experimental methods

## 3.1 Corpora, models and labels

Two separately defined collections support the analysis. The variable-horizon model-domain matrix contains 798,770 raw saved rows and 75,965 sanitized trajectories over 52 cells. Its boundary and matched policy analyses distinguish malformed raw records from eligible trajectories. The standardized detector corpus has 144,440 rows, 28,888 five-response trajectories and 2,948 task identifiers (Table 1). The collections overlap in benchmark questions and are not independent replications; their counts must not be added.

[[CORPUS_TABLE]]

The thirteen-model matrix spans DeepSeek, Qwen, InternLM, Llama, Mistral, Phi and Yi. The detector roster replaces InternLM3-8B with Qwen3.5-9B. The historical `mistral_small_24b_2409` alias denotes the recorded 22B model. Complete identifiers, sampling settings, seeds and response horizons are in the preserved extended methods and cell metadata. Architecture, training and tokenization are not randomized; a family or scale association is not an isolated parameter-count effect.

GSM8K, MATH, ARC-Challenge and GPQA define the four task domains [Cobbe2021], [Hendrycks2021], [Clark2018], [Rein2023]. Standardized GPQA metadata requests `test`, but the current loader uses `train` and saved identities use `gpqa_main`. Its effective split is inferred because the executed historical loader is unrecorded. Table 1 reports both interpretations. Detector task holdouts can come from benchmark training splits and are distinct from official benchmark test evaluation.

Each increment is a complete generated response followed, if permitted, by a revision prompt. Grading uses numeric extraction, symbolic-equivalence checks or the displayed MCQ ordering. The primary historical tables retain archived labels; predictor training separately versions candidate reconstruction and regrading. Parser regression coverage addresses specified cases, while corpus-wide semantic label validity remains unestablished.

## 3.2 Experimental units and comparisons

Rows within a trajectory and trajectories sharing a task are dependent. Source-qualified cell/run keys identify trajectories; task-disjoint folds keep a question's rows together. Strict tabular, text and portable-prefix fits follow their recorded task-disjoint contracts. Some earlier hazard probes instead use upstream run-group folds followed by task-group threshold folds, allowing other-temperature versions of a question into upstream fitting. Their controlled contrasts retain development scope.

Matched estimator comparisons resample the recorded 52 cells; paired systems comparisons and population transitions resample task clusters, conditional on the fixed model panel. One generation seed per setting leaves seed variability largely unmeasured. Repeated configuration selection can also turn a held-out panel into development data. Ten thousand bootstrap draws resample existing observations rather than create independent generations.

Accuracy is the mean grader label of the selected answer. Historical step utility is $C_{i,\tau_i}-0.05(\tau_i-1)$; token utility uses $C_{i,\tau_i}-0.0002\sum_{s=1}^{\tau_i}L_{i,s}$. Thus 250 completion tokens carry the same penalty as one response increment under this convention. Actual savings use one minus the ratio of total stopped completion tokens to total baseline completion tokens. Prompt processing, scoring, peers, timing and energy are distinct costs; a stopped-step index alone cannot establish physical savings.

ROC-AUC assesses ranking, Brier loss probability error. Causal sequence baselines include gated recurrent units [Cho2014] and transformers with rotary position embeddings [Su2021]. High pooled AUC or marginal calibration does not establish conditional continuation probabilities, a unique architecture winner or useful stopping utility.

## 3.3 Runtime policies and fitting

The controller consumes completed responses in order, enforces a two-response floor and five-response horizon, records selected answers and costs, and rejects later input after stopping. The adapter requests another response only after a continue decision, so termination prevents future model calls. Reference labels and future-step scores are excluded. Optional peer observations require a completed same-step roster and charged generation costs; no substantial live thirteen-peer experiment is reported.

A frozen confidence/stability heuristic is compared with the full horizon. Confidence is uncalibrated; optional prior-answer retention branches did not fire in the live panels. The learned rule selects the latest nonempty candidate and estimates current correctness $\widehat q_t$ and next-selected correctness $\widehat p_{t+1}$. It stops at the first eligible nonpositive $\widehat p_{t+1}-\widehat q_t-0.05$, or at the horizon. This fitted myopic rule is not the Bellman policy.

Fitting uses 1,500 archived Qwen2.5-0.5B GSM8K/MATH trajectories, comprising 7,500 rows. Task hashes allocate 902 tasks to fitting, 322 to calibration and 276 to evaluation. Scalers and unweighted logistic heads use fitting tasks; Platt mappings use calibration tasks [Platt1999]. All rows of a task share a partition, and the live/trap identities are excluded. Prefix features include step, tokens, answer changes, thought-text summaries and domain, without future-trajectory filtering.

Reconstruction changes 4,269 candidate strings but only 43 correctness labels; the versioned training audit preserves the original corpus. None of the 7,500 archive outputs meets the live JSON contract, leaving confidence and strict-parsing features without archive variation. Live prompts and EOS-inclusive token accounting differ. Archive calibration therefore does not establish transport to live generation.

## 3.4 Paired evaluation and uncertainty

Actual paired executions use Qwen2.5-0.5B-Instruct [Qwen2024] on 100 GSM8K questions and twenty prespecified traps, with separately stored public prompts, reference keys and event ledgers. Tasks already observed during corpus development remain development tasks. Active/full arms are executed separately; shared-prefix agreement is measured because batching and kernels can change greedy outputs.

For $n\ge1$ iid task pairs, let $I$ flag baseline incorrect/active correct and $W$ baseline correct/active incorrect. Write $\pi_I=\Pr(I=1)$ and $\pi_W=\Pr(W=1)$, so $\delta=\pi_I-\pi_W$ is the accuracy difference. Their marginal counts are binomial but the within-pair indicators are not assumed independent. Exact two-sided 97.5% Clopper-Pearson bounds $[L_I,U_I]$ and $[L_W,U_W]$ for the respective marginal probabilities [Clopper1934] have joint coverage at least 95% by a union bound; subtraction gives

$$\delta\in[L_I-U_W,\ U_I-L_W].$$

No discordances still leave nonzero uncertainty, with each marginal upper bound $1-0.0125^{1/n}$. No noninferiority margin was prespecified. Token intervals bootstrap paired tasks and the ratio of totals. The handpicked traps form a fixed challenge bank; iid-reference intervals do not provide randomized coverage of an adversarial population.
