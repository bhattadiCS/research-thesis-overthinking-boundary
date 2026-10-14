## Abstract

Reasoning language models generate intermediate answers that may repair an error, preserve the current answer, or corrupt an answer that was already correct. This report formulates that behavior as a finite-horizon optimal-stopping problem and documents the Semester 1 experimental program used to measure it. The work produced a repeatable trace-generation and grading system, a 52-cell matrix spanning 13 models and four reasoning domains, task-grouped evaluation, controlled estimator comparisons, and regression tests. The principal empirical result is not a universal stopping step: one-step net gain depends on model, domain, step, and assigned computation cost. In a fixed 13-model GSM8K panel, its estimate is positive at step 2 and negative at step 4, with task-cluster bootstrap intervals excluding zero. A non-nested retrospective stack attained ROC-AUC 0.955 versus 0.943 for a legacy reduced-feature control; both retain committee-derived inputs, and the stack reads later steps, so the contrast is descriptive rather than causal. An in-sample replay exposes an efficiency-accuracy trade-off. These results establish the experimental and mathematical foundation for the final thesis, while live stopping and locked external validation remain Semester 2 work.

## 1. Problem, hypothesis, and Semester 1 contributions

For trajectory \(i\), let \(g(i)\) identify its underlying benchmark question. A **reasoning trace** is the ordered sequence of complete responses for one question-model-temperature-seed condition. Throughout this report, one reasoning **increment** (or **step**) is one complete saved model response, not one token. The answer at step \(t\) can be wrong and become correct (**repair**), or correct and become wrong (**corruption**). The research question is therefore: *when does the expected gain from another response cease to justify its risk and computation cost?*

The working hypothesis was that the marginal one-step net gain from continuing changes sign along at least some traces, and that this sign change can be estimated from information available at the current step. Semester 1 delivered four concrete contributions:

1. a repeatable pipeline for generation, trace capture, answer parsing, domain-specific grading, and analysis;
2. controlled sweeps over model family and size, dataset, temperature, per-response completion-token cap, numerical precision, and candidate stopping features;
3. a mathematical framework connecting correctness, repair, corruption, cost, and a stopping time; and
4. task-held-out experiments, uncertainty estimates, diagnostic plots, and software regression tests.

[[PIPELINE_FIGURE]]

The detector is a statistical scoring rule applied to stored traces; it is not the language model that solves the task. Every result in this report is retrospective, and the offline replay does not terminate live generation.

## 2. Mathematical formulation and claim status

Let \(A_{i,t}\) be the parsed answer after increment \(t\), \(g(i)\) its benchmark-question cluster, \(Y_{g(i)}\) its reference, and \(G_d^{(v(i))}\) the domain grader version used to create the frozen label. With \(N_t\) observed trajectories, correctness and step accuracy are

$$ C_{i,t}=\mathbf{1}\{G_d^{(v(i))}(A_{i,t},Y_{g(i)})=1\},\qquad \widehat{\mathrm{Acc}}(t)=\frac{1}{N_t}\sum_{i=1}^{N_t}C_{i,t}. $$

Thus **correct** means accepted by that version-locked automated grader; it is not independent human certification of the complete reasoning or a formal proof.

Suppressing trajectory index \(i\) below, let \((\mathcal{F}_t)_{t=1}^{H}\) be an increasing filtration containing only information available to a stopping rule through increment \(t\); it excludes \(Y_{g(i)}\) and the grader label \(C_{i,t}\). Define current correctness probability \(q_t\), repair hazard \(\alpha_t\), and corruption hazard \(\beta_t\) by

$$ q_t=\Pr(C_t=1\mid\mathcal{F}_t),\quad \alpha_t=\Pr(C_{t+1}=1\mid C_t=0,\mathcal{F}_t),\quad \beta_t=\Pr(C_{t+1}=0\mid C_t=1,\mathcal{F}_t). $$

Here a **hazard** is simply a conditional probability of changing correctness state on the next complete response.

**Proposition 1 (exact one-step decomposition).** The conditional change in correctness satisfies

$$ \mathbb{E}[C_{t+1}-C_t\mid\mathcal{F}_t]=(1-q_t)\alpha_t-q_t\beta_t. $$

**Proof.** The increment is \(+1\) only for a repair and \(-1\) only for a corruption; unchanged transitions contribute zero. Conditioning on \(C_t=0\) and \(C_t=1\) gives the identity. This statement is algebraic and does not depend on a fitted model.

<!-- PAGE BREAK -->

With per-increment cost \(\lambda>0\), define realized utility, the conditional **stop-now reward**, and the \(\mathcal{F}_t\)-measurable one-step conditional utility drift:

$$ U_i(t)=C_{i,t}-\lambda(t-1),\quad V_t=\mathbb{E}[U_i(t)\mid\mathcal{F}_t],\quad \mu_t=\mathbb{E}[V_{t+1}-V_t\mid\mathcal{F}_t]=(1-q_t)\alpha_t-q_t\beta_t-\lambda. $$

Reported boundary experiments use \(\lambda=0.05\): another complete response is valuable only if its expected correctness gain exceeds five percentage points.

For the paired set \(I_t\) of traces containing both increments \(t\) and \(t+1\), the plotted estimator is

$$ \widehat D_t=\frac{1}{|I_t|}\sum_{i\in I_t}\bigl(C_{i,t+1}-C_{i,t}-\lambda\bigr). $$

This is a fixed-panel average estimator; it does not estimate the individual conditional drift \(\mu_t\) for every trace.

**Proposition 2 (sufficient condition for one crossing).** If \(q_{t+1}\ge q_t\), \(\alpha_{t+1}\le\alpha_t\), and \(\beta_{t+1}\ge\beta_t\) almost surely, then \(\mu_{t+1}\le\mu_t\). Indeed,

$$ \mu_{t+1}-\mu_t=(1-q_{t+1})(\alpha_{t+1}-\alpha_t)+\alpha_t(q_t-q_{t+1})-q_{t+1}(\beta_{t+1}-\beta_t)-\beta_t(q_{t+1}-q_t)\le0. $$

**Theorem 1 (oracle OSLA optimality).** For finite horizon \(H\), integrable adapted \(V_t\), and pathwise nonincreasing \(\mathcal{F}_t\)-measurable one-step drifts \(\mu_t\), define the **oracle one-step-look-ahead (OSLA)** rule [5]

$$ \tau_{\mathrm{OSLA}}=\min\left(\{t\in\{t_{\min},\ldots,H-1\}:\mu_t\le0\}\cup\{H\}\right),\qquad t_{\min}=2. $$

An **admissible stopping time** is an \(\mathcal{F}_t\)-stopping time in \(\{t_{\min},\ldots,H\}\): it selects a step using only information available by that step and cannot inspect a future response. Then \(\tau_{\mathrm{OSLA}}\) maximizes \(\mathbb{E}[U_i(\tau)]\) among admissible stopping times. It is an oracle benchmark because it uses the true drift; an implemented policy must substitute an estimate. **Proof.** Since \(\mu_s\) is \(\mathcal{F}_s\)-measurable, \(\tau_{\mathrm{OSLA}}\) is a stopping time. For any admissible \(\tau\), \(\{\tau=t\}\in\mathcal{F}_t\) and \(\{\tau>s\}\in\mathcal{F}_s\); conditioning on these events and applying the tower property gives

$$ \mathbb{E}[U_i(\tau)]=\mathbb{E}[V_\tau]=\mathbb{E}[V_{t_{\min}}]+\mathbb{E}\left[\sum_{s=t_{\min}}^{H-1}\mathbf{1}\{\tau>s\}\mu_s\right]. $$

Before the first crossing \(\mu_s>0\), and from the crossing onward \(\mu_s\le0\). Therefore, for every admissible \(\tau\) and every \(s\),
\(\mathbf{1}\{\tau>s\}\mu_s\le\mathbf{1}\{\tau_{\mathrm{OSLA}}>s\}\mu_s\) almost surely. The OSLA inner sum retains every strictly positive term and excludes every negative term, so summing and taking expectations proves optimality. Zero-drift ties can yield additional optimal rules. In plain language, if correctness becomes more likely, remaining repair becomes no more likely, and corruption becomes no less likely, then the one-step net gain can move only downward and cross zero at most once. This theorem is conditional: Semester 1 did **not** verify these monotonicity assumptions for every trace.

Sparse model-domain-step hazards were stabilized by empirical-Bayes shrinkage [6],

$$ \widetilde{\alpha}_{c,t}=\frac{R_{c,t}+k\overline{\alpha}_t}{N^{(0)}_{c,t}+k},\qquad \widetilde{\beta}_{c,t}=\frac{B_{c,t}+k\overline{\beta}_t}{N^{(1)}_{c,t}+k}, $$

where \(c\) indexes a model-domain cell; \(R,B\) count repairs and corruptions; \(N^{(0)},N^{(1)}\) count eligible wrong and correct states; \(\overline{\alpha}_t,\overline{\beta}_t\) are pooled step-\(t\) training-data rates; and \(k=10\) is the pseudo-count. This is Beta-binomial pseudo-count shrinkage and is empirical Bayes because its prior means are estimated from pooled data. Conditional on those plug-in means, Equation (9) is the posterior mean under a Beta prior with concentration \(k\).

For saved signal \(z_{i,t}\), the implementation zero-pads unavailable pre-trace lags and recursively computes

$$ \Delta^1z_{i,t}=z_{i,t}-z_{i,t-1},\qquad \Delta^r z_{i,t}=\Delta^{r-1}z_{i,t}-\Delta^{r-1}z_{i,t-1},\quad r=2,\ldots,5. $$

These are higher-order **discrete differences**, not physical derivatives. Exponential averages use \(m_t=\rho z_t+(1-\rho)m_{t-1}\) with recursive weighting (\(adjust=False\)), initialize as \(m_1=z_1\), and use 16 exact weights: 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, and 0.95. Larger \(\rho\) emphasizes the newest response; smaller \(\rho\) changes more slowly. These are retrospective numerical summaries, not motion variables.

## 3. Scientific method and experimental design

Confirmatory matched arms followed a fixed sequence: state the sign-change hypothesis and comparison; hold task, prompt, seed, and grader definitions fixed; vary the planned factor; save every response and measurement; grade each step; compare against the declared baseline; quantify uncertainty over whole task clusters or model-domain cells; and retain null or adverse results. Descriptive boundary analyses used a declared fixed panel, while exploratory architecture and feature searches used grouped development evaluation and were not treated as causal evidence. This separation avoids treating repeated rows from one problem, training epochs, or model initializations as independent experimental evidence.

### Data and trace lineage

<div class="table-caption"><strong>Table 1.</strong> Standardized five-step corpus. “Tasks” counts unique benchmark questions; tasks may repeat across models, and every standardized trajectory uses temperature 0.6 and seed 7.</div>

| Domain | Source and effective split | Tasks | Trajectories | Step rows | Grader accepts |
|---|---|---:|---:|---:|---|
| Grade-school arithmetic | GSM8K `main`, train [1] | 1,000 | 8,064 | 40,320 | normalized numeric, fraction, unit, or prose-wrapped answer |
| Competition math | MATH-500, test subset [2, 11] | 500 | 6,500 | 32,500 | boxed extraction plus symbolic/numeric equivalence |
| Science multiple choice | ARC-Challenge, test [3] | 1,000 | 8,500 | 42,500 | normalized option label after deterministic relabeling |
| Graduate science | GPQA `gpqa_main`, train [4] | 448 | 5,824 | 29,120 | deterministic choice shuffling plus normalized option label |
| **Total** | **four domains** | **2,948** | **28,888** | **144,440** | **five saved steps per trajectory** |

<!-- PAGE BREAK -->

The canonical variable-horizon matrix contains 75,965 sanitized trajectories and 798,770 raw step rows across 52 analyzed model-domain cells with complete derived artifacts; metadata record 13,505.12 aggregate collection minutes, or 225.09 aggregate CUDA collection job-hours (the sum of job wall times, not utilization-weighted GPU-compute hours). Its exact roster is DeepSeek-R1-Distill-Qwen-{1.5B, 7B}; Qwen2.5-{0.5B, 3B, 7B, 14B, 32B}-Instruct; InternLM3-8B-Instruct; Llama-3.1-8B-Instruct; Mistral-7B-Instruct-v0.3; Mistral-Small-Instruct-2409 (22B); Phi-4-mini-instruct; and Yi-1.5-9B-Chat. The standardized five-step corpus above is a separate detector experiment, replaces InternLM3 with Qwen3.5-9B, and is not added to the canonical total. Boundary analyses use the canonical matrix; detector tournaments use the standardized corpus. The offline replay reuses the canonical 1,500-trajectory Qwen2.5-7B/GSM8K cell; it is not a third independent corpus.

### Controlled factors, controls, and recorded variables

The principal independent variables were model family and scale; domain; temperatures 0.1, 0.6, and 1.0 (the sampling randomness setting); per-response completion-token cap; 16-bit floating point (BF16) versus 4-bit weight quantization in paired arms; and alternative step-local, temporal, consensus, and shrinkage features. Saved sequence horizon varied between corpora, but N5 changed `max_new_tokens` per complete response rather than the number of saved reasoning increments. Matched comparisons kept problem identifiers, prompt contract, reference answers, and grading logic fixed. Controlled studies tested leave-cell/model calibration, nonlinear probability models, monotone probability recalibration, prior-step features, pooled repair/corruption estimates, step-2 answer changes, token caps, and numerical precision. Development-only neural sweeps screened recurrent, attention, and mixture-of-experts configurations; grouped evaluation, rather than fit count or epoch count, supplies the comparative evidence. Generation used seed 7 and dataset shuffle seed 17 in the canonical matrix; this is a broad model/domain sweep, not a multi-seed replication study.

Depending on the experiment, each saved step recorded the parsed answer, correctness label, answer change, completion-token count, runtime, confidence, next-token uncertainty (entropy or log-probability summaries), internal-vector (hidden-state) displacement, and agreement among peer models. The main harness generates a complete step and then performs a separate scoring pass; it does not yet stop token generation inside `model.generate()`. `CausalAttractorGRU` is a project-specific historical class name, not a theorem or standard model. Its gated recurrent unit [9] is bidirectional, so the backward direction reads later stored steps; it is retrospective, not a **prefix-safe** controller restricted to current and earlier steps.

### Evaluation and software checks

The standardized detector tournament used five task-grouped outer folds: every task was evaluated only by a detector trained on other tasks. Controlled estimator variants used their recorded run/task fold designs. Boundary and matched-effect intervals use 10,000 percentile-bootstrap resamples of whole task clusters or model-domain cells [7]. They are conditional on exchangeability of the resampled clusters; a 52-cell interval does not establish generalization to unseen model families, domains, or seeds. Descriptive fold intervals use \(\overline{m}\pm t_{0.975,4}s_m/\sqrt{5}\); because outer-fold training sets overlap, these summarize fold-to-fold variation rather than independent replications. Accuracy is the mean frozen binary grader label. ROC-AUC is a ranking statistic, not percent correct [8]:

$$ \mathrm{AUC}=\Pr(\widehat q^+>\widehat q^-)+\frac{1}{2}\Pr(\widehat q^+=\widehat q^-). $$

Here \(\widehat q^+\) and \(\widehat q^-\) are evaluation-fold scores for randomly selected correct and incorrect rows. “Micro” AUC pools eligible rows; “task-macro” averages per-task AUCs for tasks containing both classes; “domain-macro” weights the four domain AUCs equally. Reported step utility is the arithmetic mean of \(U_i(\tau_i)\) from Equation (4) over the \(N\) trajectories; token utility replaces its step penalty with

$$ \widehat U_{\mathrm{tok}}=\frac{1}{N}\sum_i\left[C_{i,\tau_i}-0.0002\sum_{s=1}^{\tau_i}L_{i,s}\right],\qquad 0.0002=\frac{0.05}{250}. $$

Thus 250 completion tokens receive the same cost as one complete reasoning increment. Token utility charges all generated completion tokens; the mandatory first response is common to policies on the same trace and cancels in paired comparisons.

### Retrospective detector evaluation

<div class="table-caption"><strong>Table 2.</strong> Detector evaluations and the exact scope of each conclusion.</div>

| Evaluation | Denominator and result | What it establishes |
|---|---|---|
| Balanced sequence ranking | 28,888 trajectories; five task-held-out folds. A prefix-safe unidirectional GRU [9] achieved micro AUC 0.8743; task-macro 0.8214 over 2,679/2,948 valid tasks; domain-macro/worst-domain 0.8102/0.6313; ARC 0.8679, GPQA 0.6313, GSM8K 0.8808, MATH 0.8610. Its five fold AUCs were 0.8791, 0.8721, 0.8685, 0.8795, and 0.8728; fold mean 0.8744, descriptive 95% t interval [0.8685, 0.8803]. | Task and domain balance preserve the ranking result, but GPQA is the limiting domain. |
| Balanced stopping utility | On the same folds, a causal attention model with rotary position encoding (RoPE) [10] had task-macro/domain-macro/worst-domain AUC 0.8085/0.7953/0.6079. Its micro step/token utility was 0.3331/0.3634; domain-macro 0.3068/0.3361; per-domain ARC 0.6178/0.6365, GPQA 0.1715/0.1846, GSM8K 0.3369/0.3913, MATH 0.1010/0.1318. RoPE fold-mean step utility was 0.3331 [0.3220, 0.3442] and token utility 0.3634 [0.3536, 0.3732]; corresponding GRU values were 0.3264 [0.3172, 0.3357] and 0.3631 [0.3539, 0.3723]. | AUC ranking and stopping utility select different architectures. The descriptive fold intervals overlap; no singular detector winner was established. |
| Stored-step ranking diagnostic | 144,440 rows, 2,948 task groups, five scoring folds. A stacked hybrid achieved AUC 0.955156 versus 0.943223 for a legacy reduced-feature LightGBM control retaining four committee/vote aggregates: observed difference +0.011933; task-bootstrap mean +0.011937, CI [0.010384, 0.013480]. | Retrospective, potentially optimistic development diagnostic only: not strict held-out validation, a causal feature ablation, or a live stop rate. |
| Held-out-domain stress test | Train on three domains; evaluate 5,824 GPQA trajectories (29,120 rows; 448 tasks): AUC 0.700305. | Generalization to an unseen domain is materially weaker; this arm has no stored uncertainty interval. |

The answer-parser/grader script passes 30/30 regression cases (20 normalization/extraction and 10 mathematical-equivalence cases), and the minimum-boundary script passes 6/6 cases. These are directly executable regression scripts rather than test cases discovered by `pytest`. Later parser corrections changed the comparison target, so no current-code full-corpus disagreement rate is claimed: all empirical tables use the frozen stored `correct` field. These checks verify defined software behavior, not independent human adjudication of every answer.

<!-- PAGE BREAK -->

## 4. Results, metrics, and evaluation of conclusions

[[RESULTS_FIGURE]]

For the stacked diagnostic, **non-nested** means that meta-training was not confined to each outer training fold. The trajectory-level sequence score consumes all five saved steps, including steps later than an early row, and the same score is copied to every row in that trajectory; a centered five-point Savitzky-Golay entropy feature also reads later saved steps. The resulting AUC is retrospective and potentially optimistic: it is neither prefix-safe nor a strict held-out estimate. The 0.943223 control is likewise not a pure no-peer ablation: it retains `independent_vote_count`, `independent_vote_fraction`, `committee_has_independent_match`, and `committee_nonempty_fraction`.

The panel accuracy \(\widehat p_t\) below is distinct from an individual trace's conditional probability \(q_t\).

<div class="table-caption"><strong>Table 3.</strong> Pooled GSM8K transition decomposition at two representative steps. Each row contains 19,500 trajectories grouped into 500 task clusters; intervals are 10,000-draw task-cluster bootstraps. Source: <code>boundary_domain_step_metrics.csv</code>.</div>

| Step | Panel accuracy \(\widehat p_t\), 95% CI | Repairs / eligible wrong; \(\widehat\alpha_t\), 95% CI | Corruptions / eligible correct; \(\widehat\beta_t\), 95% CI | One-step net gain \(\widehat D_t\), 95% CI |
|---:|---|---|---|---|
| 2 | 0.2401 [0.2260, 0.2548] | 3,145/14,818; 0.2122 [0.2003, 0.2245] | 1,170/4,682; 0.2499 [0.2354, 0.2653] | +0.0513 [0.0433, 0.0593] |
| 4 | 0.4020 [0.3855, 0.4184] | 1,830/11,661; 0.1569 [0.1481, 0.1660] | 1,102/7,839; 0.1406 [0.1321, 0.1490] | -0.0127 [-0.0186, -0.0065] |

<div class="table-caption"><strong>Table 4.</strong> Selected controlled, system, and replay results. All bracketed intervals are 95% percentile-bootstrap intervals over the stated task or cell cluster. Sources: <code>boundary_selected_cell_step_metrics.csv</code>, <code>algorithm_v2_normalized_effects.csv</code>, and <code>offline_replay_metrics.csv</code>.</div>

| Result | Experimental unit and estimate | Interpretation |
|---|---|---|
| Cell-specific crossings | Qwen2.5-7B/GSM8K: \(\widehat D_4=+0.0193\), CI [0.0060, 0.0340], then \(\widehat D_5=-0.0253\), CI [-0.0353, -0.0147]. Qwen2.5-32B/MATH: \(\widehat D_5=+0.0187\), CI [0.0040, 0.0333], then \(\widehat D_6=-0.0133\), CI [-0.0240, -0.0027]. Each cell has 1,500 traces and 500 tasks. | The exploratory crossings are model- and domain-dependent, not universal. |
| Controlled estimator change | Empirical-Bayes hazards versus matched cell-local logistic hazards on 75,965 trajectories (1,948 tasks; 52 cells; seed 7; temperatures 0.1, 0.6, 1.0): +0.00781 utility/trajectory, 52-cell bootstrap CI [0.00276, 0.01355]. | Pooling sparse transition evidence produced the largest positive matched effect among the five estimator comparisons in Figure 2(c). |
| Paired systems controls | Mistral-Small-22B/GSM8K, 500 tasks x 3 temperatures per arm: the hazard-drift rule had lower utility than never-stop on 454/1,500 trajectories under both 256- and 512-token caps; loss-risk difference 0, CI [0, 0]. Qwen2.5-7B/GSM8K, 1,500 matched pairs: step-2 correctness 418 using BF16 versus 204 using 4-bit weights; difference +0.1427, CI [0.1113, 0.1753]. | The larger cap had no measured effect on this binary loss endpoint; other cap-dependent endpoints were not tested by this contrast. The 4-bit effect is panel-specific. |
| In-sample offline replay | Canonical Qwen2.5-7B/GSM8K cell, 500 tasks x 3 temperatures: counted tokens 827,804 to 377,960; savings 54.34%, task CI [53.42%, 55.23%]. Accuracy 70.53% to 64.20%; change -6.33 points, task CI [-7.73, -5.00]. The policy was fit and replayed on the same 1,500 traces. | Development trade-off diagnostic only: neither held-out evidence nor live stopping. |

For \(N\) trajectories and a policy \(\pi\), let \(\tau_i^\pi\) be its selected step, \(L_{i,s}\) the completion-token count at increment \(s\), and \(T_i\) the final saved step. Token savings and stopped-versus-final accuracy change are

$$ S_{\mathrm{tok}}=1-\frac{\sum_i\sum_{s=1}^{\tau_i^\pi}L_{i,s}}{\sum_i\sum_{s=1}^{T_i}L_{i,s}}. $$

$$ \Delta_{\mathrm{acc}}=\frac{1}{N}\sum_i\bigl(C_{i,\tau_i^\pi}-C_{i,T_i}\bigr). $$

Relative to the four proposal objectives, analytical formalization is complete; empirical calibration and estimator upgrades are complete at the retrospective-development stage; structural characterization is complete as a conditional theorem whose assumptions remain to be tested trace by trace; and generalization/validation is partially complete through task-held-out and held-out-domain studies, with a prefix-safe live controller and locked external evaluation reserved for Semester 2.

<!-- PAGE BREAK -->

<div class="table-caption"><strong>Table 5.</strong> Semester 1 research-program inventory. The final column prevents engineering or exploratory activity from being misread as independent confirmatory evidence.</div>

| Workstream | Completed work | Evidentiary role |
|---|---|---|
| Mathematics and exploratory analysis | Repair/corruption drift identity, cost-sensitive stopping rule, conditional OSLA proof, synthetic regimes, and equation-, difficulty-, and trajectory-level screens. | Theory and hypothesis generation; assumptions and language-model evidence remain explicitly separate. |
| Data and instrumentation | Four loaders/graders; 52-cell canonical matrix; standardized five-step corpus; answer, token, runtime, logit-summary, hidden-vector, and peer-signal capture. | Auditable measurement foundation; no token-level live interception is claimed. |
| Controlled and predictive studies | N1-N6 matched studies; empirical-Bayes, lag, churn, cap, and precision tests; task-grouped linear, GRU, TCN, FNO, SSM, RoPE, MoE, and tree/stack screens. | Matched effects and grouped development estimates; fits, seeds, and epochs are not experimental denominators. |
| Stress testing and reproducibility | Held-out GPQA, cross-domain/family, peer and selected-answer perturbations, prefix-safety analyses, manifests/hashes, regression scripts, and prospective-protocol tamper checks. | Boundary-condition and integrity evidence; protocol checks are not prospective stopping results. |

## 5. Semester 1 conclusions, claim ledger, and next steps

### What is established

<div class="table-caption"><strong>Table 6.</strong> Claim ledger separating mathematical results, empirical evidence, and present limitations.</div>

| Claim class | Supported statement | Boundary of the claim |
|---|---|---|
| Algebraic identity | Equation (3) weights the repair hazard by \(1-q_t\) and the corruption hazard by \(q_t\). | This weighted identity does not identify a universal stopping time. |
| Conditional theorem | The oracle OSLA rule in Theorem 1 is optimal under the stated finite-horizon, adapted-information, and pathwise one-crossing assumptions. | Those assumptions have not been verified for every collected trajectory, and the true drift is not observable online. |
| Empirical finding | Estimated one-step net gain crosses zero in the pooled GSM8K panel and in selected Qwen GSM8K/MATH cells, with task-cluster intervals. | Results apply to the fixed model, prompt, seed, cost, and dataset panel. |
| Predictive finding | A non-nested retrospective development stack ranks likely-correct stored steps better than the legacy reduced-feature control. | Both configurations retain committee-derived inputs and differ in multiple components; AUC is not accuracy, and this is neither strict held-out validation nor a causal feature lift. |
| Systems finding | In-sample offline replay can reduce counted tokens, but the tested rule also reduces accuracy. | Prospective latency, energy, and live token savings remain unmeasured. |

Semester 1 completed the data schema, four benchmark loaders and graders, 52-cell model/domain sweep, trace instrumentation, mathematical and synthetic analyses, boundary estimators, empirical-Bayes and temporal variants, controlled N1-N6 arms, peer/selected-answer and cross-domain stress tests, task-grouped tournaments, prospective-protocol checks, diagnostic plots, and regression scripts. The central scientific lesson is that stopping must be defined as a cost-sensitive, model- and domain-conditional decision; confidence, answer stability, or a fixed token limit alone is insufficient.

### Scope of the current evidence

The evaluation supports development-stage conclusions because denominators are stated, tasks rather than rows define the properly held-out tournament folds, uncertainty resamples the experimental cluster, and accuracy and computation are reported together. It remains limited by adaptive model selection on the same development corpus, one canonical generation seed, fixed temperature 0.6 and seed 7 in the standardized corpus, an effective GPQA training split that must be frozen consistently in metadata, corpus imbalance, and no prospective in-generation stopping hook. The stacked AUC is non-nested and uses future-step information, so it cannot serve as strict external validation or an online prefix-only score. The replay policy was fit and evaluated on the same 1,500 traces. Bootstrap intervals cover task variation in the fixed panel, not new model families or seeds. The September evidence freeze will version-lock the parser/grader and rerun the label census. **Unique challenges or problems: none at present; these limitations are planned Semester 2 research tasks, not blockers.**

<!-- PAGE BREAK -->

### Semester 2 completion plan

- **September:** freeze the evidence manifest and GPQA split; implement the prefix-safe live controller; test prompt noise, reasoning traps, and out-of-distribution tasks; and complete policy optimization using untouched held-out problems.
- **October:** make thesis writing the primary work product. Formalize all assumptions, complete the optimal-stopping and boundary-error arguments, write the experimental and results chapters, and iterate with Dr. Zerotti Woods until each definition, derivation, concept, and empirical claim is correct and defensible. The revise-and-review loop begins during this month.
- **November and early December:** continue advisor and reader revisions, finalize publication figures with colorblind-safe palettes, conduct mock defenses, complete the oral defense, apply post-defense revisions, archive reproducibility materials, and submit the electronic thesis and required approvals.

## 6. Conclusion

The Semester 1 work supports a precise but limited conclusion: additional reasoning has measurable, cost-dependent value that can become negative, and pooled hazard estimates can improve retrospective stopping utility. It does not yet support a universal boundary or a claim of harmless live early stopping. The experimental foundation is complete; Semester 2 is centered on writing and defending the mathematics while prospectively validating the final controller.

**Implementation and evidence sources.** Generation/grading: `research/real_trace_experiments.py`; boundary selection: `research/trace_analysis.py::first_zero_crossing`; cluster intervals: `tools/compute_progress_report_review_metrics.py::cluster_ratio_bootstrap`; controlled experiments: `research/algorithm_v2_experiments_n2_n3.py`; prefix-safe tournament: `research/run_ultimate_multi_day_tournament.py`; stacked diagnostic: `research/run_ultimate_blackwell_5day_tournament.py`; replay: `research/run_active_stopping.py::run_active_stopping_simulation`; derived evidence and hashes: `research/outputs/progress_report_review_metrics/`.

### References

<div class="references">
[1] Cobbe, K. et al. “Training Verifiers to Solve Math Word Problems.” <em>arXiv:2110.14168</em>, 2021.<br>
[2] Hendrycks, D. et al. “Measuring Mathematical Problem Solving With the MATH Dataset.” <em>NeurIPS</em> 34, 2021.<br>
[3] Clark, P. et al. “Think You Have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge.” <em>arXiv:1803.05457</em>, 2018.<br>
[4] Rein, D. et al. “GPQA: A Graduate-Level Google-Proof Q&A Benchmark.” <em>COLM</em>, 2024.<br>
[5] Chow, Y. S., Robbins, H., and Siegmund, D. <em>Great Expectations: The Theory of Optimal Stopping</em>. Houghton Mifflin, 1971.<br>
[6] Robbins, H. “An Empirical Bayes Approach to Statistics.” <em>Proceedings of the Third Berkeley Symposium</em>, vol. 1, pp. 157-163, 1956. doi:10.1525/9780520313880-015.<br>
[7] Field, C. A. and Welsh, A. H. “Bootstrapping Clustered Data.” <em>JRSS B</em> 69(3):369-390, 2007. doi:10.1111/j.1467-9868.2007.00593.x.<br>
[8] Bamber, D. “The Area Above the Ordinal Dominance Graph and the Area Below the Receiver Operating Characteristic Graph.” <em>Journal of Mathematical Psychology</em> 12(4):387-415, 1975. doi:10.1016/0022-2496(75)90001-2.<br>
[9] Cho, K. et al. “Learning Phrase Representations Using RNN Encoder-Decoder for Statistical Machine Translation.” <em>EMNLP</em>, pp. 1724-1734, 2014. doi:10.3115/v1/D14-1179.<br>
[10] Su, J. et al. “RoFormer: Enhanced Transformer with Rotary Position Embedding.” <em>arXiv:2104.09864</em>, 2021.<br>
[11] Lightman, H. et al. “Let’s Verify Step by Step.” <em>ICLR</em>, 2024; <em>arXiv:2305.20050</em>.
</div>
