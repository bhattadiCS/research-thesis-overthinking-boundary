# Thesis Research Progress Report

**Project:** Overthinking Boundary in Reasoning Large Language Models: Dynamic Stop Policies via Optimal Stopping  
**Student:** Aditya Bhatt  
**Program:** M.S. in Applied and Computational Mathematics, Johns Hopkins University  
**Research Advisor:** Dr. Zerotti Woods  
**Proposed Second Reader:** Dr. Moustapha Pemy  
**Report date:** August 2, 2026  
**Evidence snapshot:** July 30, 2026  
**Target completion:** December 2026  
**Repository snapshot:** branch `main`, HEAD `7400d3a6`, 469 commits reachable from HEAD

## Scope, evidence policy, and executive findings

This report reconciles three evidence classes:

1. the three-page enrollment status PDF, `JHU_ACM_625.804_Thesis_Enrollment_Status_Report_Aditya_Bhatt.pdf`;
2. all 469 commits reachable from repository HEAD, from `6f89d5b` on March 31, 2026 through `7400d3a` on July 23, 2026; and
3. the code, manifests, trace files, result tables, audit notes, and current worktree artifacts present on July 30, 2026.

The following evidence precedence is used whenever numbers conflict:

1. current trace-level files and machine-readable manifests;
2. current result JSON/CSV files;
3. code that generated or analyzes those files;
4. committed Markdown reports;
5. commit messages;
6. the enrollment-status PDF.

This ordering is necessary because several headline numbers in prose are stale or refer to different experimental corpora. The principal reconciled findings are:

- The canonical variable-horizon experiment matrix contains 52 successful cells, 75,965 analyzed trajectories after sanitation, and 798,770 raw step rows. It evaluates 13 models across four datasets and three temperatures.
- The later standardized five-step corpus also contains 52 cells, but it is a different experiment: 13 models, 28,888 trajectories, 144,440 step rows, temperature $0.6$, and exactly five steps per trajectory.
- The union of successful full-matrix and standardized-corpus models contains 14 model identifiers. A Gemma 4 text-only pilot raises the total number of successfully recorded model identifiers to 15. Thus, “15 models in 52 cells” is not a literal description of either complete 52-cell design.
- The current standardized corpus contains 28,888 trajectories and 144,440 rows. The PDF’s 28,928 trajectories and 192,640 steps remain in an older log line but are not reproduced by the current 52 trace files.
- The current supported final Blackwell stacked detector result is OOF ROC-AUC $0.955156$, against a no-peer control AUC of $0.943223$. The bootstrap mean lift is $0.011937$ with a task-clustered 95% interval $[0.010384,0.013480]$ over 10,000 draws.
- The PDF’s $97.17\%$ AUC statement is not supported by a matching final result file. Commit `e299697` states $0.971662$, but the result JSON stored in that commit reports a 50-task smoke run with deep-hybrid AUC $0.760762$ and stacked AUC $0.942603$. No current result JSON reports $0.9717$.
- The current inference harness performs step-level generation followed by a separate forward scoring pass. It does not intercept token-by-token generation. `run_active_stopping.py` replays already-generated trajectories; it does not terminate `model.generate`. Its measured trade-off is 54.34% fewer replay-counted tokens and an accuracy change from 70.53% to 64.20%, a decrease of 6.33 percentage points.
- The mathematical stopping rule is implemented and tested offline. A deployable, prefix-safe, in-generation stopping hook remains unfinished.
- The stopping-boundary hypothesis now has direct measured support with task-cluster uncertainty. In the fixed 13-model GSM8K panel, net drift is $+0.0513$ at step 2 with 95% CI $[+0.0433,+0.0593]$, remains positive at step 3, and is $-0.0127$ at step 4 with interval $[-0.0186,-0.0065]$. Qwen2.5-7B/GSM8K crosses between steps 4 and 5, while Qwen2.5-32B/MATH crosses between steps 5 and 6. ARC and GPQA controls remain negative at the earliest admissible stop.
- Algorithm-v2 totals are now normalized by their experimental units. Across 75,965 trajectories, empirical-Bayes hazard shrinkage has the largest positive matched effect, $+0.00781$ utility per trajectory with 52-cell bootstrap 95% CI $[+0.00276,+0.01355]$. Temporal lags and churn modulation also have intervals above zero; gradient boosting and isotonic calibration have intervals below zero.
- Tournament model selection is now macro- and domain-balanced. The GRU leads micro, task-macro, domain-macro, and worst-domain AUC and has the lowest worst-domain loss rate. The RoPE transformer leads mean step utility and corrected raw-token utility. Five-fold intervals overlap, so the report retains a three-candidate set rather than declaring a unique winner.
- Phase I has completed 6/6 defined infrastructure gates. Across the combined Phase I/II research objective, 7/9 gates are achieved; the missing gates are true in-generation stopping and a locked external generalization test.
- No advisor feedback comments, annotations, requested revisions, or approval notes appear in the enrollment-status PDF. The document identifies the advisor and second reader, but it does not record feedback content.
- The PDF does not label a “summer semester” plan. It records Semester 1 accomplishments and a Semester 2 plan beginning in September 2026. Consequently, none of its Fall milestones is overdue as of the August 2, 2026 submission date.

## Registrar-facing submission and completion status

**Overall determination: conditionally ready for approval.** This document is the status report intended to be attached to the official MATH ACM 625.804 Semester 2 Thesis Approval Form. It documents the research status, unique challenges, and completion path. It does **not** replace the official form, advisor/reader signatures, program approvals, or Registrar submission. Those administrative actions remain outstanding in the current worktree.

### Approval and routing status

The enrollment-status PDF names the following people and administrative destinations. The status below records only evidence found in the repository; it does not assume that an approval has been granted.

| Routing item | Named person or destination | Evidence in current artifacts | Required action |
|---|---|---|---|
| Research advisor | Dr. Zerotti Woods | Named in the enrollment-status PDF and this report. | Obtain advisor review, requested revisions, and signature on the official form; record each decision in the feedback ledger. |
| Proposed second reader | Dr. Moustapha Pemy | Named in the enrollment-status PDF and this report. | Obtain second-reader review and signature on the official form; record each decision in the feedback ledger. |
| Program administration | Dr. Beryl Castello, Program Manager; Dr. James Spall, Program Chair | Both are listed in the enrollment-status PDF. | Confirm the official routing and obtain any required program-level approval or chair sign-off. |
| Registrar | JHU Registrar, through the prescribed submission channel | No Registrar receipt or completed approval form is present in the worktree. | After every required approval is obtained, attach this report to the signed form, submit the package to the Registrar, and retain the submission receipt. |

### Requirement-to-evidence check

| Registrar instruction | Where this report addresses it | Determination |
|---|---|---|
| Attach a status report. | This document is the evidence-qualified status report; Appendix A maps every major claim to a repository artifact. | Ready to attach; the official form is still required. |
| Mention unique challenges or problems. | The explicit challenge register below, Sections 4.11 and 6.1-6.4, and Appendix C identify scientific, data, deployment, and archival risks with mitigations. | Addressed explicitly. |
| Clearly indicate what must be done for completion by the end of Semester 2. | The critical-path schedule below and Sections 6.1-6.14 define dated deliverables, dependencies, acceptance criteria, defense milestones, ETD submission, and final administrative clearance. | Addressed with measurable gates through December 7, 2026. |
| Submit after all approvals. | The routing table above separates research readiness from approval and Registrar submission. | Pending signatures and official routing; no premature submission is claimed. |

### Explicit challenge register

| Challenge or problem | Current impact | Mitigation and completion gate |
|---|---|---|
| Effective GPQA split differs from its metadata. | The loader uses `train` while metadata labels the split `test`; affected metrics cannot be called archival until the split is frozen. | Decide the authoritative split, regenerate or explicitly exclude affected traces, hash the dataset revision, and pass the Phase 1 evidence-freeze gate. |
| The strongest neural detectors are retrospective. | `CausalAttractorGRU` and the Blackwell stack score stored trajectories; the current replay does not stop a live `model.generate()` call. | Implement and test a prefix-safe controller with byte-identical paired prefixes, real token counts, latency, and GPU-time measurements. |
| External validity is weakest on GPQA. | Leave-one-domain-out GPQA AUC is $0.700305$, below the proposed $0.75$ gate. | Run the locked external/OOD suite; either meet the pre-registered gate or carry the failure explicitly into the thesis limitations and model-selection decision. |
| Several historical headline numbers are not reproducible from current artifacts. | The 97.17% AUC and 28,928/192,640 corpus description are not supported by the current authoritative JSON/CSV bundle. | Use the reconciled values in this report, archive one manifest, and label all superseded prose as non-authoritative. |
| The 100-instance ultra sweep lacks a durable trial ledger. | Code specifies the folds, seeds, probes, and epochs, but the final JSON does not prove every attempted run. | Persist per-trial seed, fold, split hash, epoch count, checkpoint, metrics, and failure status before final model selection. |
| Correctness has parser regression coverage but not full independent adjudication. | The 30/30 regression and targeted extraction audit validate mechanics and a sample, not every benchmark row. | Obtain an independent adjudication sample and report agreement and disagreements before the final manuscript tag. |
| Administrative feedback and approvals are not present in the source artifacts. | No advisor annotations, approval signatures, feedback ledger, or Registrar receipt was found. | Obtain reviews/signatures, maintain the feedback ledger, and submit the signed package only after all required approvals. |

### Critical path to completion by the end of Semester 2

The end-of-semester completion target is **December 7, 2026**. A task is complete only when its acceptance evidence exists; code or prose alone does not satisfy the gate.

| Window | Required work | Acceptance evidence and dependency |
|---|---|---|
| August 3-28 | Freeze and repair traces; resolve GPQA metadata; implement the prefix-safe stopping controller; run prompt-noise, reasoning-trap, and initial OOD tests. | `data_manifest_v1.json`, clean-clone hash match, no future-step feature reads, paired-prefix identity, and Phase 1 decision report. |
| August 31-September 25 | Run nested task-grouped hyperparameter search and select conservative, utility, and never-stop policies. | Complete trial ledger with at least 1,000 attempted inner trials, calibration package, outer-fold metrics, and advisor-approved selection memo. |
| September 28-October 23 | Complete theorem statements/proofs and empirical assumption checks; integrate Chapters 1-5; submit manuscript Draft v1 by October 23. | Version-controlled manuscript, proof/assumption audit, frozen tables, and advisor/reader Draft v1 receipt. |
| October 26-November 10 | Produce publication-quality figures; revise manuscript from advisor/reader feedback; finalize technical-paper package. | Figure inventory with captions, vector exports where possible, feedback ledger, and revised manuscript. |
| November 11-20 | Conduct at least two recorded mock defenses and resolve critical questions. | Deck, recordings or logs, question ledger, and documented corrections. |
| November 23-30 | Complete the public oral defense. | Defense completion record and required committee decisions. |
| December 1-3 | Apply post-defense revisions, complete ETD formatting, freeze code/data/manuscript hashes, and archive the reproducibility appendix. | Final ETD manuscript, release tag, environment lock, licenses, manifests, figures, and defense package. |
| December 4-7 | Submit the ETD and, after all required approvals are signed, attach this status report to the official form and submit the package to the Registrar; obtain chair/grade clearance. | ETD receipt, signed-form copy, Registrar submission receipt, and final sign-off record. |

If a P0 gate fails, the student must notify the advisor immediately, document the exception, and revise the completion decision. The report does not claim that unresolved scientific gates or missing signatures are already complete.

## Plain-language reader's guide and claim audit

This section translates the most technical phrases into the exact operation performed by the repository. The report uses three different layers that should not be conflated:

1. **The language model** generates a sequence of complete reasoning increments.
2. **The grader** assigns a binary correctness label to each parsed answer.
3. **A detector** estimates correctness or continuation value from recorded features. It does not itself solve the benchmark question.

### What “Kinematics of Thought” means in this implementation

“Kinematics” is an analogy, not a claim that a language model has physical velocity. A trajectory contains one row per completed reasoning increment. Let $z_t$ be one scalar signal at increment $t$, such as peer support, panel entropy, or a peer-margin count. The code uses discrete finite differences:

$$
v_t=z_t-z_{t-1},\qquad
a_t=v_t-v_{t-1},\qquad
j_t=a_t-a_{t-1},
$$

$$
s_t=j_t-j_{t-1},\qquad
c_t=s_t-s_{t-1}.
$$

The project names these columns `v_`, `a_`, `j_`, `snap_`, and `crackle_`. “Crackle” is simply the fifth finite difference; it is not a physical measurement. The differences measure how a signal changes from one reasoning increment to the next and can help a classifier recognize a turn, flattening, or reversal in a trajectory.

The implementation in `research/run_ultimate_blackwell_5day_tournament.py:104-150` applies this calculation to three per-step signals: `peer_support_fraction`, `panel_response_entropy`, and `peer_support_margin_count`. It then computes 16 exponential moving averages (EMAs) with

$$
\alpha\in\{0.001,0.005,0.01,0.02,0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95\},
$$

using $m_t=\alpha z_t+(1-\alpha)m_{t-1}$. A small $\alpha$ changes slowly and suppresses short-term noise; a large $\alpha$ follows the latest increment closely. The phrase “smoothing noisy token data” is therefore shorthand: the code does **not** smooth each generated token or make a token-by-token stopping decision. It smooths per-increment telemetry derived after a completed response. These high-order features are retrospective unless a future-safe prefix implementation is separately demonstrated.

The EMA and derivative calculations reset at each `trajectory_id`. Missing predecessors at the beginning of a trajectory are zero-filled, so the first few derivative values are initialization features rather than measurements of a physical velocity or acceleration.

### What `CausalAttractorGRU` is

`CausalAttractorGRU` is a project-specific class, not a standard architecture or a theorem. The class is defined in `research/run_40day_deep_iteration_suite.py:98-126` and `research/run_ultimate_blackwell_5day_tournament.py:331-350`.

| Code component | Plain-language meaning |
|---|---|
| `Linear(input_dim, 256)` | Projects each row of engineered telemetry into a 256-dimensional feature vector. |
| Three-layer `nn.GRU(..., bidirectional=True)` | Reads the sequence in both directions and summarizes how features change across the entire stored trajectory. |
| Residual addition plus `LayerNorm` | Adds the projected input back to the GRU output and rescales the result for stable optimization. |
| SiLU, dropout, and linear head | Converts the final sequence representation into one logit for binary final-correctness prediction. |
| Final-step gather | Selects the representation at the last available step of the stored sequence. |

The name is potentially misleading. “Causal” is a historical label, but the bidirectional GRU and final-step gather use the full recorded sequence; the implementation is not a prefix-safe online controller. “Attractor” refers to engineered phase-space features such as `attractor_distance_3d` and `trajectory_curvature`; it does not establish a dynamical-systems attractor mathematically. The model predicts a correctness score for a stored trajectory. It does not directly output a stopping time and it does not terminate `model.generate()`.

The reproducible `iteration_suite_v1` artifact reports a trajectory-level PyTorch probe AUC of $0.946214$ and a subsequent stacked AUC of $0.955424$ on 28,888 stored sequences (144,440 rows) and 2,948 task groups. The separate `blackwell_5day_tournament_v1` artifact reports the current supported stacked AUC of $0.955156$. These are distinct artifacts and neither should be described as “97.17% accuracy.”

### Why the experiment trained many configurations

The “100 PyTorch models over 1,000 epochs” statement describes a planned ultra-depth training mode, not 100 different language models and not 100 proven stopping boundaries. In the `ultra` branch of `research/run_ultimate_blackwell_5day_tournament.py:523-537`, the code sets 1,000 passes through the training data, 10 random seeds, and five outer folds. The training function instantiates two top-level probes per seed and fold, so the intended count is

$$
5\ \text{folds}\times10\ \text{seeds}\times2\ \text{probes}=100\ \text{trained instances}.
$$

One epoch means one pass through the training batches; it is not 1,000 independent experiments. The purpose of the sweep is to measure sensitivity to initialization and optimization depth and to compare candidate feature extractors. A candidate is selected using held-out metrics, calibration, utility, and balanced domain checks. The phrase “singular most optimal boundary detector” is too strong: no single universal optimum was established, and the final report retains three candidates because fold intervals overlap. The saved Blackwell JSON does not contain a per-seed/per-fold trial ledger, so the completed count of all 100 instances cannot be independently verified from the final artifact. Keep this count separate from the 1,170 Algorithm-v2 grid-search configurations (implementation capacity without a frozen trial ledger) and the nine frozen standardized-tournament rows.

### Correctness, accuracy, probability, and AUC are different quantities

For each benchmark item, `verify_answer()` compares the parsed answer with the reference answer after domain-specific normalization. Numeric tasks use guarded numeric, symbolic, and unit-aware comparisons; multiple-choice tasks use the normalized option label. The resulting label is

$$
C_{i,t}=\mathbf{1}\{A_{i,t}\equiv Y_i^*\},
$$

where $C_{i,t}=1$ means that the answer at step $t$ matches the benchmark reference after normalization, and $C_{i,t}=0$ means that it does not.

That binary label is **correctness**. **Accuracy** is its sample mean, for example $\operatorname{Acc}(t)=N_t^{-1}\sum_i C_{i,t}$; it is the fraction of answers that were graded correct. A detector's $\widehat q_{i,t}$ is different: it is a predicted probability that $C_{i,t}=1$.

The 30/30 grader regression and the 0.188% extraction re-grade audit check the grading mechanics and a targeted sample; they are not independent human adjudication of every benchmark row. The report therefore treats the normalized reference answer and parser contract as the evaluation oracle and lists external adjudication as future validation.

| Metric | How it is computed | What it answers | What it does not answer |
|---|---|---|---|
| Step accuracy | Mean of $C_{i,t}$ at a fixed step. | How often answers at that step are correct. | Whether stopping there saves compute. |
| Stopped-policy accuracy | Mean of $C_{i,\tau_i^\pi}$ at the detector's selected step. | How often the policy's selected answers are correct. | Whether the probability estimates are calibrated. |
| ROC-AUC | $\Pr(\widehat q^+>\widehat q^-)+\frac{1}{2}\Pr(\widehat q^+=\widehat q^-)$ over positive/negative correctness pairs. | Whether correct examples tend to receive higher scores than incorrect examples. | The fraction of answers that are correct, or the quality of an actual stopping policy. |
| Brier score / ECE | Squared probability error / agreement between predicted and observed frequencies in probability bins. | Whether $\widehat q$ behaves like a probability. | Whether the model will stop safely under a new generation protocol. |
| Step utility | $C_{i,t}-0.05(t-1)$. | Whether correctness gain covers the cost of another reasoning increment. | Wall-clock latency or GPU energy. |
| Win/tie/loss | Compare policy utility with never-stop utility for the same trajectory. | How often a policy helps, ties, or harms under the defined utility. | A human-judged “overthinking error” count unless that term is explicitly defined as a utility loss. |

### How out-of-fold AUC was computed, and what “97.17%” means

For the current standardized tournament, the five outer folds are grouped by `task_id`. For each fold, the detector is fit only on training tasks; scalers and learned thresholds are also fit only there. The detector then assigns scores to held-out tasks. The held-out predictions are concatenated once, and the ROC curve is summarized by the pairwise AUC formula above. The final Blackwell stacked result is $0.955156$ (95.5156% as an AUC scale), against a no-peer control of $0.943223$, with a task-cluster bootstrap mean lift of $0.011937$ and 95% interval $[0.010384,0.013480]$.

The $0.971662$ value (97.17%) appears in commit prose, but the corresponding committed JSON is a 50-task smoke result whose deep-hybrid AUC is $0.760762$ and stacked AUC is $0.942603$. No current machine-readable artifact reproduces $0.971662$. Therefore the accurate answer to “how did we compute 97.17%?” is: the repository does not preserve a reproducible computation for that number, so it must not be presented as a verified result.

### What “60.2% fewer live overthinking errors” actually means

The older Tier-3 table compares two policies on 4,500 stored trajectories. It reports baseline win/tie/loss counts $3,027/305/1,168$ and gated self-consistency counts $1,516/2,519/465$. The percentage is arithmetic:

$$
\frac{1,168-465}{1,168}=0.60188\approx60.2\%.
$$

Here “loss” means the policy's realized utility is below its never-stop comparator under $U=C-0.05(t-1)$; it is not a separately annotated human error category. The implementation in `research/run_final_tournament.py` evaluates already-recorded trajectories and, when uncertainty triggers, adds stored $k=2$ self-consistency token counts. It does not stop a live `model.generate()` call. Its five-fold split groups by `run_id`, not the current task-grouped `task_id` protocol. Thus 60.2% is an offline, corpus-specific reduction in utility losses, not a verified live reduction in overthinking errors and not directly comparable to the current Blackwell tournament.

### Completeness audit: work included and work still open

| Workstream | Included in this report | Evidence | Still not established |
|---|---|---|---|
| Data collection and grading | Four benchmark loaders, prompt contracts, token counts, answer normalization, and correctness labels. | `research/real_trace_experiments.py`; Sections 3.1-3.5. | A new external task family and a new generation seed panel. |
| State and uncertainty telemetry | Log probabilities, entropy, hidden-state displacement, answer changes, runtime, and token counts from a second forward pass. | Sections 2.5 and 5.6. | Token-by-token interception inside `generate()`. |
| Boundary mathematics | $q_t$, repair hazard $\alpha_t$, corruption hazard $\beta_t$, step cost, drift, oracle, and stopping rules. | Sections 1.1, 4.1, 4.2.1, and 4.4. | A theorem whose assumptions hold for every collected trajectory. |
| Feature and detector development | Logistic hazards, shrinkage, lags, churn, high-order features, sequence probes, ensembles, calibration, and balanced tournament metrics. | Sections 4.4-4.8 and Phase II chronology. | A single universally best detector. |
| Controlled comparisons | N1-N6 denominators, matched arms, bootstrap intervals, precision and token-cap tests. | Section 4.7 and `progress_report_review_metrics/`. | Multi-seed replication for the small causal arms. |
| Stress tests and generalization | Peer corruption, roster changes, permutation, leave-one-domain-out transfer, and GPQA weakness. | Sections 4.9 and 6.1-6.3. | Noisy prompts, adversarial traps, and a locked external holdout. |
| Deployment | Offline replay and a mathematical policy specification. | Section 4.10 and `run_active_stopping.py`. | A prefix-safe Hugging Face stopping hook with prospective compute/accuracy measurements. |

The remaining claims are deliberately labeled as limitations or future work rather than silently omitted. This distinction keeps the report complete without implying that retrospective prediction is already a live stopping system.

# 1. Research Objectives and Scientific Alignment

## 1.1 Original research statement

The enrollment-status PDF frames the research problem as the unnecessary continuation of reasoning after a model has reached its best answer. Continued inference can:

- preserve the current answer while consuming additional tokens;
- repair an incorrect intermediate answer;
- corrupt a previously correct answer;
- increase latency and computational cost without an offsetting performance gain.

The proposed object of study is an optimal stopping boundary $T^*$: the first admissible reasoning step at which the expected value of one additional reasoning step is nonpositive. The PDF describes this through “Kinematics of Thought,” using confidence velocity, acceleration, higher-order trajectory derivatives, entropy, answer changes, and hidden-state movement as observable signals.

The codebase gives this idea a more precise stochastic formulation. Let:

- $Y^*$ be the task’s ground-truth answer;
- $A_t$ be the answer extracted after reasoning step $t$;
- $C_t=\mathbf{1}\{A_t\equiv Y^*\}$ be the graded correctness indicator;
- $\mathcal{F}_t$ be the information available through step $t$;
- $q_t=\Pr(C_t=1\mid\mathcal{F}_t)$ be the conditional probability that the current answer is correct;
- $\alpha_t=\Pr(C_{t+1}=1\mid C_t=0,\mathcal{F}_t)$ be the repair hazard;
- $\beta_t=\Pr(C_{t+1}=0\mid C_t=1,\mathcal{F}_t)$ be the corruption hazard;
- $v>0$ be the value of a correct answer;
- $c>0$ be the magnitude of the penalty for an incorrect answer; and
- $\lambda>0$ be the cost of one additional reasoning step.

The stop value is

$$
V_t=q_t(v+c)-c-\lambda(t-1).
$$

The one-step predictable continuation drift is

$$
\mu_t
=\mathbb{E}[V_{t+1}-V_t\mid\mathcal{F}_t]
=\left[(1-q_t)\alpha_t-q_t\beta_t\right](v+c)-\lambda.
$$

The implemented normalized case sets $v+c=1$ and $\lambda=0.05$:

$$
\mu_t=(1-q_t)\alpha_t-q_t\beta_t-0.05.
$$

The operational boundary is

$$
T^*=\inf\{t\ge T_{\min}:\mu_t\le 0\},\qquad T_{\min}=2.
$$

The $T_{\min}=2$ floor is essential. Step 1 is treated as a forced initial commitment, not a valid stopping decision. Commit `6f832ea` corrected the boundary operator, and commit `8ce9b9f` propagated the $T_{\min}=2$ floor into ad hoc scripts.

## 1.2 Core hypothesis

The thesis hypothesis has a structural and an empirical component.

**Structural hypothesis.** If $q_t$ is nondecreasing, $\alpha_t$ is nonincreasing, and $\beta_t$ is nondecreasing along a trajectory, then

$$
\mu_{t+1}\le \mu_t,
$$

so $\mu_t$ crosses zero at most once. Under a finite horizon, integrability, and the monotone stopping-region assumptions stated in `ThesisDocs/archive/mathematical_foundations_proofs.md`, the one-step-look-ahead rule

$$
\tau_{\mathrm{OSLA}}=\inf\{t\ge2:\mu_t\le0\}
$$

is optimal for the stated step-cost utility.

**Empirical hypothesis.** Prefix-observable features $\mathbf{x}_t$ contain sufficient information to estimate $q_t$, $\alpha_t$, and $\beta_t$ accurately enough that an out-of-sample stopping policy can reduce expected compute while controlling correctness loss:

$$
\widehat{\tau}
=\inf\left\{t\ge2:
(1-\widehat q_t)\widehat\alpha_t
-\widehat q_t\widehat\beta_t
-\lambda
\le 0
\right\}.
$$

For deployment, this hypothesis requires three simultaneous conditions:

$$
\mathbb{E}[B_{\widehat\tau}]<\mathbb{E}[B_T],
$$

$$
\mathbb{E}[C_{\widehat\tau}-\lambda(\widehat\tau-1)]
\ge
\mathbb{E}[C_T-\lambda(T-1)],
$$

and a pre-registered safety constraint such as

$$
\Pr(C_{\widehat\tau}<C_T)\le \epsilon.
$$

The repository has strong evidence for offline correctness prediction and mixed evidence for utility-preserving stopping. It does not yet establish the third condition under live generation.

## 1.3 Scientific process mapping

| Scientific-method stage | Operationalization | Repository evidence | Result |
|---|---|---|---|
| Observation | Models revise, repair, preserve, or corrupt answers over repeated reasoning steps. | `research/real_trace_experiments.py`; trace-level `correct`, `answer_changed`, `repair_rate`, and `corruption_rate` columns. | Recurrent repair and corruption transitions are directly measurable. |
| Hypothesis | Continuation value is governed by correctness belief, repair hazard, corruption hazard, and cost. | `research/trace_analysis.py:170`; measured tables in `research/outputs/progress_report_review_metrics/`. | Implemented with $\lambda=0.05$ and measured: pooled GSM8K drift is positive at steps 2-3 and negative at step 4; Qwen2.5-7B/GSM8K and Qwen2.5-32B/MATH have task-cluster intervals that exclude zero on both sides of their crossings. |
| Controlled design | Hold prompt protocol, seed, token cap, attention backend, and dataset fixed while changing model, temperature, precision, or budget. | `research/outputs/experiment_matrix/matrix_manifest.json`; Algorithm-v2 protocols N1–N8. | 52-cell matrix plus targeted causal arms. |
| Data collection | Generate repeated stepwise outputs, retaining parsed answers, logits-derived uncertainty, hidden summaries, runtime, and token counts. | `generate_batch_with_diagnostics()` and `run_batch_traces()` in `research/real_trace_experiments.py`. | 75,965 sanitized variable-horizon trajectories; 28,888 standardized five-step trajectories. |
| Falsification | Test empty-answer guards, $T_{\min}=3$, token caps, precision, threshold transfer, nonlinear probes, and hazard shrinkage. | `ThesisDocs/rigor_audit/05_offline_experiment_results.md`; `08_algorithm_v2_results_analysis.md`. | Several plausible alternatives failed; empirical-Bayes hazards, lags, and two-parameter churn survived. |
| Out-of-sample analysis | Prevent trajectory or task reuse between fit and evaluation folds. | `trace_analysis.out_of_sample_eval()`; task-grouped outer folds in `run_ultimate_multi_day_tournament.py`. | Later pipelines use task-grouped OOF evaluation; early cell analyses group only by `run_id`. |
| Stress testing | Corrupt peer signals, leave out domains, change roster size, permute panels, vary token cap and quantization. | `peer_dynamics_stress_test_report.json`; N5/N6 artifacts. | GPQA transfer is the largest observed weakness; quantization has a measurable effect. |
| Revision | Repair graders, boundary floors, corrupted CSV checkpoints, data leakage, precision, and memory failures. | Commits `ef45dda`, `f89ff38`, `ce0167c`, `8485341`, and July 14 memory fixes. | Scientific and systems assumptions became materially stricter over time. |

The process is scientific in the sense that alternatives were explicitly tested and rejected. It is not complete because the final stopping policy has not been evaluated prospectively inside a generation loop on a locked external holdout.

## 1.4 Status-report delta: claims versus current evidence

| PDF statement | Current evidence | Reconciled status |
|---|---|---|
| 52-cell pipeline | 52 successful canonical matrix cells; one additional failed Mistral-Small-3.1 VLM manifest entry has zero runs. | Supported after excluding the zero-run failed cell. |
| 15 models from 0.5B to 32B | 15 successful identifiers exist across all experiments; only 13 occur in either complete 52-cell design. | Supported only as an all-experiment roster count. |
| 28,928 traces and 192,640 steps | Current five-step files contain 28,888 trajectories and 144,440 rows. | Stale or unreconciled. |
| Four benchmarks | GSM8K, MATH-500, ARC-Challenge, and GPQA Main are implemented. | Supported. |
| Temperatures 0.1, 0.6, 1.0 | Canonical matrix uses all three; standardized global sweep uses only 0.6. | Supported for the matrix, not the later global corpus. |
| Full precision versus 4-bit | NF4/double-quantization code and N6 paired experiment exist. | Supported. |
| $14.3\%$ step-2 degradation under quantization | N6 reports 418/1,500 BF16 correct versus 204/1,500 4-bit correct, a $0.14267$ gap with task-cluster 95% CI $[0.11133,0.17533]$ and $Z=9.79$. | Supported for the specified Qwen-7B/GSM8K arm; the interval covers tasks, not new generation seeds. |
| 60.2% loss reduction, 1,168 to 465 | Tier-3 deep-research corpus reports an offline policy-loss comparison: $3,027/305/1,168$ versus $1,516/2,519/465$ win/tie/loss. | Supported only for the distinct 4,500-trajectory/58,000-step Tier-3 corpus; not a live-generation error reduction. |
| 97.17% OOF AUC | Commit prose states 0.971662; no matching final artifact does. Current final stacked result is 0.955156. | Not currently reproducible. |
| Active early stopping saves compute without harming correctness | Offline replay saves 54.34% of counted completion tokens but accuracy falls by 6.33 percentage points. | Compute reduction supported; “without harm” is contradicted by the artifact. |
| 100 neural configurations over 1,000 epochs | The Blackwell script exposes 1,000-epoch modes and commit prose claims 100 configurations. The final JSON does not serialize a completed 100-model ledger. | Implementation intent supported; completed-run count requires a durable trial manifest. |
| Advisor feedback incorporated | No feedback annotations or revision requests appear in the PDF. | No evidence in the named source document. |

## 1.5 Goal-by-goal completion ledger

The PDF’s plan begins in September 2026; it does not contain a summer-semester schedule. The table below evaluates early completion of its non-overlapping Fall work products as of August 2, 2026, using the July 30 evidence snapshot.

The scoring rubric is:

- 0%: no artifact found;
- 25%: outline, protocol, or component notes exist;
- 50%: substantial implementation or prose exists, but validation or integration is incomplete;
- 75%: validated technical artifact exists, but advisor-ready packaging or final generalization is incomplete;
- 100%: final deliverable exists, is internally consistent, and satisfies its acceptance criterion.

| Planned work product | PDF timing | Evidence | Completion |
|---|---:|---|---:|
| Complete feature and hyperparameter tuning; integrate advanced concepts | September | Algorithm-v2 N1–N8, 1,170-model grid-search support, nested Optuna infrastructure, 199/225-feature tournaments. No frozen final search ledger or selected deployable policy. | 75% |
| Expand stress tests to additional variables | September | Precision, token cap, threshold transfer, peer corruption, roster scaling, permutation, and leave-one-domain-out tests. Prompt noise and true OOD task families remain. | 75% |
| Chapter 1: introduction and literature review | September | Proposal, novelty review, layperson guide, and research question exist; no thesis-form chapter file. | 25% |
| Chapter 2: methodology | September | `Scientific_Methodology_and_Experimental_Report.md` and proof notes provide substantial content. | 75% |
| Chapter 3: implementation and experimental setup | September | Detailed code, manifests, runbooks, and methodology prose exist; no integrated chapter. | 75% |
| Explore edge cases and related research ideas | October | Failure taxonomy, perfect-detector stress tests, causal and peer-dynamics variants exist; adversarial reasoning traps remain. | 50% |
| Finalize analysis and publication-quality charts | October | 323 PNG files are tracked, including aggregate and advisor figures; no final figure inventory, caption audit, vector export, or manuscript linkage. | 50% |
| Chapter 4: results | October | Several results reports and tables exist; the conflicting headline numbers are not yet reconciled in a thesis chapter. | 75% |
| Chapter 5: discussion and future work | October | Failure analyses and roadmaps exist; limitations and claim boundaries are not integrated into a final chapter. | 50% |
| Full manuscript version 1 | October 23 | Component Markdown reports exist; no tracked `.tex`, `.docx`, or chapter-complete manuscript exists. | 25% |
| Defense slide deck | November | No tracked `.ppt` or `.pptx`. | 0% |
| Publication-quality technical paper | November | Proposal and long methodology reports exist; no paper-form manuscript with abstract, related work, methods, results, and references. | 25% |
| Incorporate advisor and reader feedback | November 1–10 | No feedback ledger or annotated draft found. | 0% |
| Mock defenses | November 11–20 | No deck, script, recording, or question log found. | 0% |
| Formal 60-minute defense | November 23–30 | Future milestone. | 0% |
| Code archival and repository tag | December | Repository is extensive but has no Git tag; the current worktree contains untracked result directories and database/WAL files. | 25% |
| Refereed journal or conference submission | December | No submission package or confirmation found. | 0% |
| Final ETD polish | December 1–3 | No ETD-formatted manuscript. | 0% |
| ETD submission | December 4–5 | Future milestone. | 0% |
| Advisor/chair final sign-off | December 6–7 | Future milestone. | 0% |
| Document student contribution above the required 60% | Before approval | 388 of 469 commits use author name “Aditya Bhatt,” but aliases, automation, and generated artifacts make commit count insufficient as a formal contribution audit. | 25% |

Two aggregate scores are useful:

$$
\text{Research/chapter-preparation readiness}
=\frac{550}{9\times100}
=61.1\%.
$$

$$
\text{Full plan completion}
=\frac{650}{21\times100}
=31.0\%.
$$

These are evidence-readiness scores, not elapsed-time estimates. Because the plan starts in September, the 31.0% represents work completed ahead of the PDF schedule rather than schedule slippage.

# 2. Comprehensive Model and Architecture Exploration

## 2.1 Successfully recorded model roster

Parameter labels in the second column are the repository catalog labels. Architecture and nominal context information comes from the referenced model configurations/cards; it is not serialized in the experimental metadata. “Test horizon” is the actual repository protocol, which is far shorter than every nominal context window.

| Repository alias and exact identifier | Catalog size | Architecture and attention | Nominal model context | Repository participation |
|---|---:|---|---:|---|
| `deepseek_r1_distill_1p5b` — `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1.5B | Qwen2 dense decoder; 28 layers; GQA 12 query/2 KV heads | 131,072 | Canonical matrix and standardized global corpus |
| `deepseek_r1_distill_7b` — `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 7B | Qwen2 dense decoder; 28 layers; GQA 28/4 | 131,072 | Canonical matrix and standardized global corpus |
| `gemma_4_e4b_it` — `principled-intelligence/gemma-4-E4B-it-text-only` | E4B label; 7.52B text-only total | Dense Gemma 4 text decoder with 42 layers; five local sliding-attention layers per global layer; GQA 8/2; 512-token local window | 131,072 | Pilot only; not in either complete 52-cell corpus |
| `internlm3_8b_instruct` — `internlm/InternLM3-8B-Instruct` | 8B | Dense decoder; 48 layers; GQA 32/2; dynamic RoPE | 32,768 configured; long-context card evaluates through 128K | Canonical matrix only |
| `llama_3p1_8b_instruct` — `meta-llama/Llama-3.1-8B-Instruct` | 8B | Optimized dense decoder with GQA and RoPE | 128K | Canonical matrix and standardized global corpus |
| `mistral_7b_instruct_v0p3` — `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Dense Mistral decoder; 32 layers; GQA 32/8; no configured sliding window | 32,768 | Canonical matrix and standardized global corpus |
| `mistral_small_24b_2409` — `mistralai/Mistral-Small-Instruct-2409` | 22B | Dense Mistral decoder; 56 layers; GQA 48/8; alias retains an obsolete “24b” label | 32,768 | Canonical matrix and standardized global corpus |
| `phi_4_mini_instruct` — `microsoft/Phi-4-mini-instruct` | 4B label; official card 3.8B | Dense decoder-only transformer with GQA and shared input/output embedding | 128K | Canonical matrix and standardized global corpus |
| `qwen2p5_0p5b` — `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | Qwen2 dense decoder; 24 layers; GQA 14/2 | 32,768 configured | Canonical matrix and standardized global corpus |
| `qwen2p5_3b` — `Qwen/Qwen2.5-3B-Instruct` | 3B | Qwen2 dense decoder with RoPE, SwiGLU, RMSNorm, QKV bias, and GQA | 32,768 configured | Canonical matrix and standardized global corpus |
| `qwen2p5_7b` — `Qwen/Qwen2.5-7B-Instruct` | 7B label; official card 7.61B | 28-layer Qwen2 dense decoder; GQA 28/4 | 32,768 configured; 131,072 with documented YaRN extension | Canonical matrix and standardized global corpus |
| `qwen2p5_14b` — `Qwen/Qwen2.5-14B-Instruct` | 14B | Qwen2 dense decoder with GQA | 32,768 configured; family supports YaRN extension | Canonical matrix and standardized global corpus |
| `qwen2p5_32b` — `Qwen/Qwen2.5-32B-Instruct` | 32B | Qwen2 dense decoder with GQA | 32,768 configured; family supports YaRN extension | Canonical matrix and standardized global corpus |
| `qwen_3p5_9b` — `Qwen/Qwen3.5-9B` | 9B | Hybrid language backbone: repeated Gated DeltaNet blocks plus gated-attention blocks; 32 layers; attention blocks use GQA 16/4 | 262,144 native | Standardized global corpus only |
| `yi_1p5_9b_chat` — `01-ai/Yi-1.5-9B-Chat` | 9B | 48-layer Llama-style dense decoder; GQA 32/4 | 4,096 | Canonical matrix and standardized global corpus |

Primary architecture references: [DeepSeek-R1-Distill-Qwen-7B model/config](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B), [Qwen2.5-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [Gemma 4 E4B model card](https://huggingface.co/google/gemma-4-E4B-it), [InternLM3 configuration](https://huggingface.co/internlm/internlm3-8b-instruct/blob/main/config.json), [Llama 3.1 model card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [Mistral 7B configuration](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/blob/main/config.json), [Mistral Small model card](https://huggingface.co/mistralai/Mistral-Small-Instruct-2409), [Phi-4-mini model card](https://huggingface.co/microsoft/Phi-4-mini-instruct), [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B), and [Yi-1.5-9B configuration](https://huggingface.co/01-ai/Yi-1.5-9B-Chat/blob/main/config.json).

Catalog entries for Gemma 4 31B, Gemma 4 26B-A4B, Qwen3.5 35B-A3B, GPTQ, and AWQ variants exist in `MODEL_CATALOG`, but no successful trace metadata was found for them. They must not be listed as evaluated models.

## 2.2 Experimental model-count lineage

| Corpus | Successful model count | Inclusion rule |
|---|---:|---|
| Canonical variable-horizon matrix | 13 | InternLM3 included; Qwen3.5 excluded |
| Standardized five-step global corpus | 13 | Qwen3.5 included; InternLM3 excluded |
| Union of the two complete corpora | 14 | One-for-one exchange of InternLM3 and Qwen3.5 |
| All successful model metadata, including pilot | 15 | Adds Gemma 4 E4B text-only |

This lineage explains the PDF’s model-count ambiguity without discarding any successful experiment.

## 2.3 Precision, quantization, and attention backends

`load_model()` in `research/real_trace_experiments.py:1491` implements:

- `bfloat16` on CUDA devices reporting BF16 support;
- `float16` otherwise;
- optional 8-bit bitsandbytes loading;
- 4-bit NF4 with double quantization;
- BF16/FP16 compute for the quantized linear operations;
- FP32 CPU offload;
- `device_map="auto"` or model-specific low-memory mappings;
- Flash Attention 2 when explicitly selected and supported;
- fallback to PyTorch SDPA if Flash Attention 2 fails;
- eager attention as an available diagnostic option.

The canonical matrix commands record:

- quantization `none`;
- attention implementation `sdpa`;
- NVIDIA-hosted CUDA execution;
- temperatures $0.1$, $0.6$, and $1.0$;
- nucleus sampling $p=0.95$;
- repetition penalty $1.02$;
- seed 7;
- maximum 256 new tokens per reasoning step.

The N6 precision arm directly compares Qwen2.5-7B in unquantized BF16 and 4-bit form on GSM8K. Step-2 correctness is $0.2787$ in BF16 and $0.1360$ in 4-bit:

$$
\Delta p_2=0.2787-0.1360=0.1427,\qquad Z=9.79.
$$

This is an absolute 14.27-percentage-point effect based on 1,500 matched trajectories per arm: 500 tasks at three temperatures with generation seed 7. A 10,000-draw task-cluster bootstrap gives

$$
\mathrm{CI}_{0.95}(\Delta p_2)
=[0.11133,0.17533].
$$

The interval quantifies task heterogeneity inside this panel. It does not cover new random seeds, models, or datasets, so the result should not be generalized without replication.

## 2.4 Recorded runtime and hardware evidence

The PDF identifies two hardware regimes:

- early experimentation on Google Colab L4 GPUs;
- later matrix and tournament work on an NVIDIA RTX PRO 6000 Blackwell Server Edition.

The ultimate tournament preflight records:

- CUDA device `NVIDIA RTX PRO 6000 Blackwell Server Edition`;
- compute capability `sm_120`;
- total GPU memory 97,250 MiB in PyTorch and 97,887 MiB in `nvidia-smi`;
- PyTorch `2.13.0+cu130`;
- BF16 precision;
- 600 W power limit.

The canonical matrix manifest records 225.085 collection hours across 52 successful cells. These are end-to-end cell collection durations, not normalized single-request latency measurements:

| Model alias | Four-cell collection hours | Fastest cell | Slowest cell |
|---|---:|---:|---:|
| `deepseek_r1_distill_1p5b` | 14.39 h | 91.66 min | 411.97 min |
| `deepseek_r1_distill_7b` | 19.29 h | 172.14 min | 424.50 min |
| `internlm3_8b_instruct` | 32.52 h | 248.95 min | 722.33 min |
| `llama_3p1_8b_instruct` | 13.10 h | 114.13 min | 299.19 min |
| `mistral_7b_instruct_v0p3` | 24.93 h | 191.62 min | 598.10 min |
| `mistral_small_24b_2409` | 20.44 h | 183.79 min | 531.43 min |
| `phi_4_mini_instruct` | 20.24 h | 183.18 min | 483.44 min |
| `qwen2p5_0p5b` | 8.89 h | 30.35 min | 242.77 min |
| `qwen2p5_3b` | 9.77 h | 64.36 min | 275.07 min |
| `qwen2p5_7b` | 17.03 h | 78.33 min | 438.47 min |
| `qwen2p5_14b` | 17.57 h | 110.33 min | 517.91 min |
| `qwen2p5_32b` | 12.44 h | 120.83 min | 262.08 min |
| `yi_1p5_9b_chat` | 14.46 h | 126.02 min | 396.60 min |

The durations are confounded by dataset horizon, task count, output length, batch size, resume behavior, and hardware state. They document compute investment, not architectural speed rankings.

## 2.5 Inference and state interception

The inference path is implemented in `generate_batch_with_diagnostics()` at `research/real_trace_experiments.py:1759`.

### Generation

For a batch of rendered prompts, the function:

1. tokenizes with padding and an 8-token alignment on CUDA;
2. calls `model.generate()` with the full per-step token budget;
3. returns complete step responses;
4. appends the parsed response to the history;
5. renders the next step as a new prompt containing all previous step summaries.

Thus, one “reasoning step” is a complete generation call, not a single decoder token.

### Logit extraction

The code does not request `output_scores=True` from `generate()`. Instead, it performs a second forward pass over prompt plus generated completion:

$$
\ell_{t,j}\in\mathbb{R}^{|\mathcal{V}|},
\qquad
p_{t,j,k}
=\frac{\exp(\ell_{t,j,k})}
{\sum_{v\in\mathcal{V}}\exp(\ell_{t,j,v})}.
$$

For generated token $y_{t,j}$, it records

$$
\overline{\log p}_t
=\frac{1}{L_t}\sum_{j=1}^{L_t}
\log p_{t,j,y_{t,j}},
$$

and mean token entropy

$$
\overline H_t
=-\frac{1}{L_t}
\sum_{j=1}^{L_t}
\sum_{k\in\mathcal{V}}
p_{t,j,k}\log p_{t,j,k}.
$$

The low-memory path calls the base transformer, takes completion-position hidden states, and applies the model output embedding/head. If that path fails, it falls back to the wrapper model’s logits and hidden states.

### Hidden-state extraction

The final-layer completion vectors are mean-pooled:

$$
\bar{\mathbf h}_t
=\frac{1}{L_t}\sum_{j=1}^{L_t}\mathbf h_{t,j}^{(L)}.
$$

The trajectory features include

$$
d_t^{(2)}
=\|\bar{\mathbf h}_t-\bar{\mathbf h}_{t-1}\|_2
$$

and cosine displacement

$$
d_t^{(\cos)}
=1-
\frac{\bar{\mathbf h}_t^\top\bar{\mathbf h}_{t-1}}
{\|\bar{\mathbf h}_t\|_2\|\bar{\mathbf h}_{t-1}\|_2}.
$$

When extended observables are enabled, layers near $L/3$ and $2L/3$ are mean-pooled on GPU, copied to CPU, and projected to 64 dimensions with a cached random projection seeded at 42. July 14 commits `5a06073` and `646b379` moved pooling before CPU transfer, reducing the copied tensor size by approximately three orders of magnitude and preventing allocator stalls.

### What is not intercepted

- No token-by-token callback is passed into `model.generate`.
- No Hugging Face `StoppingCriteria` implementation invokes the learned boundary.
- No generation KV-cache state is exposed to the stopping estimator.
- No gradient is computed during inference; “gradient state changes” are not an implemented stopping signal.
- Logit-delta threshold stopping is not implemented as a primary detector.
- The Blackwell trajectory MoE uses the full five-step trajectory to predict final correctness. It is retrospective and cannot be treated as prefix-safe.

# 3. Data Engineering and Collection Framework

## 3.1 Corpus lineage

| Corpus | Models | Cells | Trajectories/runs | Step rows | Horizon | Primary purpose |
|---|---:|---:|---:|---:|---|---|
| Canonical experiment matrix | 13 | 52 | 75,965 analyzed | 798,770 raw | GSM8K 10, MATH 14, ARC 8, GPQA 10 | Multi-temperature model/dataset behavior and classical detectors |
| Standardized global corpus | 13 | 52 | 28,888 | 144,440 | Exactly 5 | Cross-model task-grouped detector tournaments |
| Tier-3 deep-research corpus | Separate Qwen-7B pilots | Not interchangeable with global cells | 4,500 | 58,000 reported | Variable | N7/N8 telemetry and GRU/LSTM/gated self-consistency |
| BF16 ladder replay corpus | Qwen2.5-7B | GSM8K ladder | 1,500 | 15,000 | 10 | Offline active-stop token/accuracy simulation |

The four corpora must remain separate in all manuscript tables.

## 3.2 Dataset sources, splits, and grading

| Domain | Exact source | Effective split | Current standardized trajectories | Standardized rows | Grader and curation |
|---|---|---:|---:|---:|---|
| Grade-school mathematics | `openai/gsm8k`, configuration `main` | `train` | 8,064 | 40,320 | Extract reference after `####`; normalize numeric forms, commas, currency, units, fractions, and prose-wrapped numbers. |
| Competition mathematics | `HuggingFaceH4/MATH-500` | `test` | 6,500 | 32,500 | Pinned source; hard failure instead of corpus substitution; boxed-answer extraction, exact/symbolic/numeric comparison, guarded SymPy simplification. |
| Science multiple choice | `allenai/ai2_arc`, `ARC-Challenge` | `test` | 8,500 | 42,500 | Relabel options to A–E and map the gold answer key to the new label. |
| Graduate science QA | `Idavidrein/gpqa`, `gpqa_main` | `train` in loader | 5,824 | 29,120 | Gated dataset; deterministic per-item shuffling of one correct and three incorrect answers. |

There is a protocol metadata defect for GPQA: global metadata files state `dataset_split: "test"`, but `load_gpqa_tasks()` at `research/real_trace_experiments.py:1139` hardcodes `split="train"`. The effective split is therefore `train`. This mismatch must be corrected before archival.

Canonical-matrix analyzed counts are:

| Dataset | Analyzed trajectories | Raw trajectory IDs | Raw step rows |
|---|---:|---:|---:|
| ARC | 19,500 | 19,500 | 156,000 |
| GPQA | 17,466 | 17,496 | 174,770 |
| GSM8K | 19,500 | 19,500 | 195,000 |
| MATH | 19,499 | 19,500 | 273,000 |
| **Total** | **75,965** | — | **798,770** |

The 31-run difference between the matrix manifest’s 75,996 completed runs and the 75,965 analyzed runs is attributable to sanitation and repaired/corrupted checkpoint rows, concentrated in GPQA and one MATH trajectory.

## 3.3 Standardized corpus balance

The five-step global sweep is not a balanced 500-task-per-cell design:

- ARC has 8,500 trajectories: four Qwen2.5 cells contain 1,000 tasks; nine cells contain 500.
- GPQA has 5,824 trajectories: every cell contains the complete 448-item `gpqa_main` split.
- GSM8K has 8,064 trajectories: five Qwen2.5 cells contain 1,000, seven cells contain 500, and Qwen3.5 contains only 64.
- MATH has 6,500 trajectories: every cell contains 500.

Model-level trajectory totals are:

| Model | Trajectories | Step rows |
|---|---:|---:|
| DeepSeek-R1-Distill-Qwen-1.5B | 1,948 | 9,740 |
| DeepSeek-R1-Distill-Qwen-7B | 1,948 | 9,740 |
| Llama-3.1-8B-Instruct | 1,948 | 9,740 |
| Mistral-7B-Instruct-v0.3 | 1,948 | 9,740 |
| Mistral-Small-Instruct-2409 | 1,948 | 9,740 |
| Phi-4-mini-instruct | 1,948 | 9,740 |
| Qwen2.5-0.5B-Instruct | 2,948 | 14,740 |
| Qwen2.5-3B-Instruct | 2,448 | 12,240 |
| Qwen2.5-7B-Instruct | 2,948 | 14,740 |
| Qwen2.5-14B-Instruct | 2,948 | 14,740 |
| Qwen2.5-32B-Instruct | 2,448 | 12,240 |
| Qwen3.5-9B | 1,512 | 7,560 |
| Yi-1.5-9B-Chat | 1,948 | 9,740 |
| **Total** | **28,888** | **144,440** |

Task-grouped evaluation prevents the same task from appearing in training and test folds, but the original aggregate table is micro-averaged toward the overrepresented Qwen/ARC/GSM8K cells. Section 4.8 now adds task-macro, domain-macro, worst-domain, per-domain, and fold-level results from the frozen OOF predictions.

## 3.4 Prompt construction and continuation protocol

`conversation_prompt()` at `research/real_trace_experiments.py:764` builds each incremental prompt from:

- task ID;
- domain;
- difficulty label;
- full task text;
- current step number and maximum step count;
- summaries of all preceding steps;
- an instruction to perform one short reasoning or verification step even if the answer is already known;
- a structured response contract.

The canonical matrix uses a single JSON object:

```json
{"thought": "...", "answer": "...", "confidence": 0, "stop": false}
```

The standardized global corpus uses exactly four lines:

```text
THOUGHT: <one short sentence>
ANSWER: <current best final answer only>
CONFIDENCE: <integer 0-100>
STOP: <yes or no>
```

The tokenizer’s model-specific chat template is applied when available. The default system message identifies a logical reasoning assistant. Parsing records whether extraction was exact, partial, fallback free-form, empty, or truncated.

Synthetic data generation is not used to create benchmark questions. “Synthetic” content in this repository refers to:

- model-generated reasoning traces;
- derivative features generated from observed trace signals;
- randomized projections of hidden states;
- deterministic peer-panel aggregates;
- controlled corruption of peer signals for stress tests.

## 3.5 Tokenization and output distributions

Across the current 144,440 standardized step rows:

| Scope | Step rows | Generated model tokens | Mean | P25 | Median | P75 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 144,440 | 14,538,601 | 100.655 | 39 | 51 | 175 | 256 | 256 |
| ARC | 42,500 | 3,916,828 | 92.161 | 37 | 45 | 80 | 256 | 256 |
| GPQA | 29,120 | 3,463,458 | 118.937 | 46 | 63 | 256 | 256 | 256 |
| GSM8K | 40,320 | 3,410,989 | 84.598 | 36 | 45 | 72 | 256 | 256 |
| MATH | 32,500 | 3,747,326 | 115.302 | 44 | 62 | 256 | 256 | 256 |

The mass at 256 shows that long/truncated completions are not rare, particularly for GPQA and MATH.

The column `thought_token_count` is not a tokenizer count. At `research/real_trace_experiments.py:2399`, it is computed as

$$
W_t=\left|\operatorname{regex}_{\backslash w+}(\text{parsed thought}_t)\right|.
$$

It therefore counts word-like regex spans. Its current distribution is:

| Scope | Total regex words | Mean | P25 | Median | P75 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| All | 2,980,636 | 20.636 | 10 | 16 | 25 | 54 | 233 |
| ARC | 810,263 | 19.065 | 11 | 16 | 23 | 45 | 233 |
| GPQA | 747,607 | 25.673 | 14 | 21 | 32 | 63 | 223 |
| GSM8K | 693,552 | 17.201 | 9 | 13 | 20 | 46 | 215 |
| MATH | 729,214 | 22.437 | 11 | 16 | 27 | 61 | 227 |

The manuscript should rename this field “thought word count” or state its construction wherever it is used.

## 3.6 Training, validation, and test lineage

No base language model is fine-tuned in this repository. The LLM weights are frozen and used only for inference. Training refers to correctness probes, hazard models, sequence models, calibration models, or meta-ensembles.

Three validation generations appear in history:

1. **Early/cell-level:** five-fold `GroupKFold` by `run_id` in `trace_analysis.out_of_sample_eval()`. This prevents steps from one trajectory crossing folds, but different temperatures for the same task can enter different folds.
2. **Standardized task-level:** five outer folds grouped by raw `task_id` in `run_ultimate_multi_day_tournament.py`. This blocks the same benchmark item across all model trajectories from entering both train and test.
3. **Blackwell stacked detector:** five-fold `GroupKFold` by `task_id`. The control and meta-ensemble are row-level correctness detectors. The trajectory MoE predicts final correctness from the entire five-step sequence and is therefore not deployable at earlier prefixes.

The ultimate task-grouped fold sizes are:

| Fold | Outer-train trajectories | Outer-test trajectories | Train task IDs | Test task IDs |
|---|---:|---:|---:|---:|
| 1 | 23,109 | 5,779 | 2,357 | 591 |
| 2 | 23,113 | 5,775 | 2,358 | 590 |
| 3 | 23,110 | 5,778 | 2,359 | 589 |
| 4 | 23,110 | 5,778 | 2,359 | 589 |
| 5 | 23,110 | 5,778 | 2,359 | 589 |

Within each outer training fold, the multi-day tournament further separates tuning-train, tuning-validation, model-fit, and calibration task groups. Scalers are fit only on outer-training data.

# 4. Overthinking Detection and Mathematical Metrics

## 4.1 Formal definition of overthinking

For a trajectory of maximum step $T$, the repository uses three related definitions.

**Correction corruption.** A step is harmful when

$$
C_t=1,\qquad C_{t+1}=0.
$$

**Utility overthinking.** Continuing from step $t$ is harmful in conditional expectation when

$$
\mu_t
=\mathbb{E}[U_{t+1}-U_t\mid\mathcal{F}_t]
\le0.
$$

**Oracle lateness.** For realized trajectory utility

$$
U_t=C_t-\lambda(t-1),
$$

the hindsight oracle is

$$
\tau_i^{\mathrm{oracle}}
=\arg\max_{t\in\{T_{\min},\ldots,T_i\}}U_{i,t}.
$$

A detector is false-late on trajectory $i$ when

$$
\widehat\tau_i>\tau_i^{\mathrm{oracle}},
$$

with severity

$$
L_i^{\mathrm{late}}
=\max(\widehat\tau_i-\tau_i^{\mathrm{oracle}},0).
$$

These definitions are not interchangeable. Corruption is a realized one-step event; negative drift is a conditional decision criterion; oracle lateness is a hindsight comparison.

## 4.2 Accuracy and task performance

For task $i$ at step $t$:

$$
C_{i,t}
=\mathbf{1}\{g_d(A_{i,t},Y_i^*)=1\},
$$

where $g_d$ is the dataset-specific grader.

Step-$t$ accuracy over $N_t$ available trajectories is

$$
\operatorname{Acc}(t)
=\frac{1}{N_t}\sum_{i=1}^{N_t}C_{i,t}.
$$

Stopped-policy accuracy is

$$
\operatorname{Acc}(\widehat\tau)
=\frac{1}{N}\sum_{i=1}^{N}C_{i,\widehat\tau_i}.
$$

Never-stop accuracy is

$$
\operatorname{Acc}(T)
=\frac{1}{N}\sum_{i=1}^{N}C_{i,T_i}.
$$

For numeric tasks, equivalence includes normalized integers, decimals, fractions, signs, commas, currency/unit wrappers, and guarded symbolic equivalence. For multiple-choice tasks, it is normalized label equality.

### 4.2.1 Measured boundary evidence with uncertainty

The definitions above are now paired with direct empirical estimates from the 52-cell canonical matrix. For every trajectory that contains both steps $t$ and $t+1$, define

$$
\widehat q_t
=\frac{\sum_i C_{i,t}}{N_t},
\qquad
\widehat\alpha_t
=\frac{\sum_i\mathbf{1}\{C_{i,t}=0,C_{i,t+1}=1\}}
{\sum_i\mathbf{1}\{C_{i,t}=0\}},
$$

$$
\widehat\beta_t
=\frac{\sum_i\mathbf{1}\{C_{i,t}=1,C_{i,t+1}=0\}}
{\sum_i\mathbf{1}\{C_{i,t}=1\}},
$$

and

$$
\widehat\mu_t
=(1-\widehat q_t)\widehat\alpha_t
-\widehat q_t\widehat\beta_t
-0.05
=\operatorname{Acc}(t+1)-\operatorname{Acc}(t)-0.05.
$$

The final equality is an audit identity for the paired transition sample: repair events add correctness, corruption events remove correctness, and continuation costs $0.05$. A positive interval for $\mu_t$ means that another reasoning step improves expected utility; a negative interval means that the expected correctness gain is smaller than corruption plus compute cost.

The uncertainty calculation resamples complete `task_id` clusters 10,000 times with replacement using master seed 804. All model and temperature observations for a task remain together. The resulting percentile intervals therefore account for dependence among the 13 fixed models and three temperatures on the same question. They quantify task-sampling uncertainty for the fixed model panel; they do not quantify variation across new model families or new generation seeds.

| Domain | Unique tasks | Trajectories at each step | Maximum step | Fixed model cells |
|---|---:|---:|---:|---:|
| ARC | 500 | 19,500 | 8 | 13 |
| GPQA | 448 | 17,466 | 10 | 13 |
| GSM8K | 500 | 19,500 | 10 | 13 |
| MATH | 500 | 19,499 | 14 | 13 |

The pooled domain curves over the decision-relevant early steps are:

| Domain | Step $t$ | $\widehat q_t$ (95% CI) | $\widehat\mu_t$ (95% CI) |
|---|---:|---:|---:|
| ARC | 2 | 0.7013 [0.6846, 0.7172] | -0.0404 [-0.0455, -0.0352] |
| ARC | 3 | 0.7110 [0.6939, 0.7270] | -0.0420 [-0.0466, -0.0375] |
| ARC | 4 | 0.7190 [0.7032, 0.7341] | -0.0463 [-0.0507, -0.0419] |
| ARC | 5 | 0.7227 [0.7067, 0.7383] | -0.0464 [-0.0503, -0.0423] |
| GPQA | 2 | 0.2637 [0.2463, 0.2814] | -0.0392 [-0.0447, -0.0337] |
| GPQA | 3 | 0.2745 [0.2568, 0.2923] | -0.0427 [-0.0473, -0.0380] |
| GPQA | 4 | 0.2818 [0.2640, 0.2995] | -0.0482 [-0.0527, -0.0438] |
| GPQA | 5 | 0.2836 [0.2655, 0.3018] | -0.0477 [-0.0514, -0.0439] |
| GSM8K | 2 | 0.2401 [0.2260, 0.2548] | +0.0513 [+0.0433, +0.0593] |
| GSM8K | 3 | 0.3414 [0.3247, 0.3582] | +0.0106 [+0.0039, +0.0173] |
| GSM8K | 4 | 0.4020 [0.3855, 0.4184] | -0.0127 [-0.0186, -0.0065] |
| GSM8K | 5 | 0.4393 [0.4235, 0.4548] | -0.0287 [-0.0337, -0.0235] |
| MATH | 2 | 0.0791 [0.0683, 0.0903] | -0.0147 [-0.0199, -0.0093] |
| MATH | 3 | 0.1144 [0.1010, 0.1282] | -0.0271 [-0.0311, -0.0230] |
| MATH | 4 | 0.1373 [0.1230, 0.1522] | -0.0309 [-0.0347, -0.0271] |
| MATH | 5 | 0.1564 [0.1412, 0.1722] | -0.0360 [-0.0393, -0.0326] |

GSM8K supplies the clearest pooled boundary: net continuation value is significantly positive at steps 2 and 3 and significantly negative at step 4. ARC is already easy for capable models, so little repair headroom remains. GPQA is too difficult for the tested panel to realize sustained repair. The pooled MATH curve is negative because 11 of 13 model cells cannot realize the available headroom; this aggregate does not erase late boundaries in the strongest cells.

The following hazard decompositions show the two late-boundary cases and two floor-boundary controls. Each row reports actual transition events and conditional denominators.

| Model-domain | Step | $N$ / tasks | $\widehat q_t$ | Repair events / eligible; $\widehat\alpha_t$ (95% CI) | Corruptions / eligible; $\widehat\beta_t$ (95% CI) | $\widehat\mu_t$ (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B / GSM8K | 4 | 1,500 / 500 | 0.5960 | 107/606; 0.1766 [0.1452, 0.2092] | 3/894; 0.0034 [0.0000, 0.0078] | +0.0193 [+0.0060, +0.0340] |
| Qwen2.5-7B / GSM8K | 5 | 1,500 / 500 | 0.6653 | 46/502; 0.0916 [0.0667, 0.1187] | 9/998; 0.0090 [0.0039, 0.0153] | -0.0253 [-0.0353, -0.0147] |
| Qwen2.5-32B / MATH | 5 | 1,500 / 500 | 0.3973 | 114/904; 0.1261 [0.1031, 0.1505] | 11/596; 0.0185 [0.0084, 0.0299] | +0.0187 [+0.0040, +0.0333] |
| Qwen2.5-32B / MATH | 6 | 1,500 / 500 | 0.4660 | 63/801; 0.0787 [0.0598, 0.0989] | 8/699; 0.0114 [0.0040, 0.0210] | -0.0133 [-0.0240, -0.0027] |
| Qwen2.5-7B / ARC | 2 | 1,500 / 500 | 0.8953 | 15/157; 0.0955 [0.0474, 0.1533] | 12/1,343; 0.0089 [0.0038, 0.0149] | -0.0480 [-0.0553, -0.0407] |
| Qwen2.5-7B / GPQA | 2 | 1,344 / 448 | 0.3103 | 54/927; 0.0583 [0.0426, 0.0749] | 40/417; 0.0959 [0.0667, 0.1278] | -0.0396 [-0.0537, -0.0247] |

For Qwen2.5-7B on GSM8K, the repair hazard falls from 0.1766 to 0.0916 as the pool of still-wrong trajectories shrinks; corruption remains small, but the remaining repair benefit no longer covers the 0.05 step cost. The sign changes between steps 4 and 5, and both confidence intervals exclude zero. Qwen2.5-32B on MATH exhibits the same pattern between steps 5 and 6. The ARC and GPQA controls remain negative at the earliest permitted stop. These measurements support a difficulty-conditioned stopping boundary, not a universal claim that every model-domain pair has a late boundary.

Complete domain-step and selected-cell tables are stored in `research/outputs/progress_report_review_metrics/boundary_domain_step_metrics.csv` and `boundary_selected_cell_step_metrics.csv`.

## 4.3 Step utility, token utility, and overthink ratios

The primary step-cost utility in `utility()` at `research/real_trace_experiments.py:752` is

$$
U_{i,t}^{\mathrm{step}}
=C_{i,t}-0.05(t-1).
$$

For a policy $\pi$:

$$
\overline U_{\pi}^{\mathrm{step}}
=\frac{1}{N}\sum_{i=1}^{N}
\left[C_{i,\tau_i^\pi}-0.05(\tau_i^\pi-1)\right].
$$

The policy delta against never-stop is

$$
\Delta U_i^\pi
=U_{i,\tau_i^\pi}^{\mathrm{step}}
-U_{i,T_i}^{\mathrm{step}}.
$$

The repository classifies:

$$
\mathrm{win}_i=\mathbf{1}\{\Delta U_i^\pi>10^{-12}\},
$$

$$
\mathrm{tie}_i=\mathbf{1}\{|\Delta U_i^\pi|\le10^{-12}\},
$$

$$
\mathrm{loss}_i=\mathbf{1}\{\Delta U_i^\pi<-10^{-12}\}.
$$

The multi-day tournament also reports token utility with token price $\rho=0.0002$:

$$
U_{i,t}^{\mathrm{token}}
=C_{i,t}
-0.0002
\sum_{s=1}^{t}B_{i,s},
$$

where $B_{i,s}$ is `raw_generation_tokens` for the primary path. Optional $k=2$ self-consistency tokens are charged only if invoked.

For direct compute accounting, define

$$
\operatorname{Savings}_i(\pi)
=1-
\frac{\sum_{s=1}^{\tau_i^\pi}B_{i,s}}
{\sum_{s=1}^{T_i}B_{i,s}}.
$$

A realized oracle-tail overthink ratio suitable for the manuscript is

$$
\operatorname{OTR}_i
=
\frac{\sum_{s=\tau_i^{\mathrm{oracle}}+1}^{T_i}B_{i,s}}
{\sum_{s=1}^{T_i}B_{i,s}}.
$$

This last ratio is a recommended derived statistic; it is not currently materialized as a canonical CSV column.

## 4.4 Boundary functions and stopping criteria

Let the prefix feature vector be

$$
\mathbf{x}_t
=
\left[
t,\overline H_t,\operatorname{sd}(H_t),
\operatorname{confidence}_t,
\operatorname{answerChanged}_t,
W_t,
d_t^{(2)},d_t^{(\cos)},
\operatorname{lexicalEcho}_t,
\operatorname{verboseProxy}_t
\right].
$$

The baseline score function is

$$
S(t)
=f(\mathbf{x}_t,\mathcal{H}_t)
=
(1-\widehat q_t)\widehat\alpha_t
-\widehat q_t\widehat\beta_t
-\lambda,
$$

where $\mathcal{H}_t$ is the prefix history. The stopping rule is

$$
\widehat\tau
=\inf\{t\ge2:S(t)\le0\}.
$$

The three probabilities are estimated with standardized, class-balanced logistic regressions:

$$
\widehat q_t
=\sigma(\mathbf w_q^\top\mathbf x_t+b_q),
$$

$$
\widehat\alpha_t
=\sigma(\mathbf w_\alpha^\top\mathbf x_t+b_\alpha),
$$

$$
\widehat\beta_t
=\sigma(\mathbf w_\beta^\top\mathbf x_t+b_\beta).
$$

Repair and corruption models are fit only on genuine transitions with $t\ge2$ after the audit fix.

The implemented detector set in `build_detector_frame()` includes:

| Detector | Stop definition | Causality |
|---|---|---|
| Oracle | $\arg\max_{t\ge2} U_t$ | Hindsight only |
| First answer | First nonempty normalized answer | Prefix-safe |
| Verifier first-correct | First $t$ with $C_t=1$ | Uses ground truth; not deployable |
| Answer stability | First $t\ge2$ with answer streak $\ge2$ and confidence $\ge55$ | Prefix-safe |
| Entropy plateau | First $t\ge2$ with $\overline H_t\le h^*$ and answer streak $\ge2$ | Prefix-safe if $h^*$ is trained out of fold |
| Hazard drift | First $t\ge2$ with $\widehat\mu_t\le0$ | Prefix-safe |
| Empirical Bernstein | First $t\ge2$ whose upper drift bound is nonpositive | Population-step bound; deployed per learned global step |
| Mixture e-process | First $t\ge2$ whose e-process crosses its threshold | Sequential population test |
| Never stop | Final available step | Baseline |

The entropy threshold $h^*$ is chosen from the training quantiles

$$
\{Q_{0.20},Q_{0.35},Q_{0.50},Q_{0.65},Q_{0.80}\}
$$

to minimize training-fold mean oracle gap.

The empirical-Bernstein upper bound is

$$
U_t^{\mathrm{EB}}
=\widehat\mu_t
+\sqrt{\frac{2\widehat v_t\log(3/\delta_t)}{m_t}}
+\frac{3(b-a)\log(3/\delta_t)}{m_t},
$$

with

$$
\delta_t=\frac{6\delta}{\pi^2(t+1)^2},
\qquad
a=-(1+\lambda),
\qquad
b=1-\lambda.
$$

The mixture e-value over $\Lambda=\{0.05,\ldots,0.95\}$ is

$$
E_t
=\frac{1}{|\Lambda|}
\sum_{\eta\in\Lambda}
\prod_{i=1}^{m_t}
\left(1-\eta\frac{\Delta_{t}^{(i)}}{M}\right),
$$

where $M=\max(|a|,|b|)$.

## 4.5 Calibration and detector-quality metrics

For labels $y_j\in\{0,1\}$ and predicted correctness probabilities $p_j$:

$$
\operatorname{Brier}
=\frac{1}{n}\sum_{j=1}^{n}(p_j-y_j)^2.
$$

For $K$ calibration bins:

$$
\operatorname{ECE}
=\sum_{k=1}^{K}
\frac{|B_k|}{n}
\left|
\frac{1}{|B_k|}\sum_{j\in B_k}y_j
-
\frac{1}{|B_k|}\sum_{j\in B_k}p_j
\right|.
$$

ROC-AUC is

$$
\operatorname{AUC}
=\Pr(p^+>p^-)
+\frac12\Pr(p^+=p^-).
$$

AUC evaluates ranking, not probability calibration and not stopping utility. A model can increase AUC while increasing policy losses. The P3d result demonstrates this distinction.

Oracle regret is

$$
\operatorname{Regret}_i^\pi
=U_{i,\tau_i^{\mathrm{oracle}}}
-U_{i,\tau_i^\pi},
$$

and mean oracle gap is

$$
\overline G^\pi
=\frac1N\sum_i\operatorname{Regret}_i^\pi.
$$

## 4.6 Canonical matrix and audit results

The repaired canonical matrix’s never-stop comparison contains:

- 68,095 wins;
- 2,135 ties;
- 5,735 losses;
- loss rate $5,735/75,965=7.5495\%$.

The audit falsification table is:

| Test | Result | Interpretation |
|---|---:|---|
| Empty-answer guard | $\Delta U=-116.55$ | Rejected |
| Raise $T_{\min}$ from 2 to 3 | $\Delta U=-1,912.35$ | Rejected; step-2 wins are structurally important |
| Token-cap guard | $\Delta U=-8,033.65$ | Rejected |
| Temperature monotonicity | 4/4 model checks pass | Supported within tested cells |
| Extraction re-grade | 0.188% label delta; 30 DeepSeek-scoped cases | Small aggregate effect, locally relevant |
| Cost sweep | See below | $\lambda=0.20$ is nearly degenerate |

| $\lambda$ | Win rate | Loss rate |
|---:|---:|---:|
| 0.01 | 87.65% | 6.60% |
| 0.02 | 88.84% | 6.85% |
| 0.05 | 89.64% | 7.55% |
| 0.10 | 90.86% | 7.06% |
| 0.20 | 98.78% | 0.27% |

At $\lambda=0.20$, aggressive early stopping wins mainly because step cost dominates correctness. This is not evidence of a universally superior detector.

Per-cell threshold calibration P3d produces:

$$
\Delta U_{\mathrm{OOF}}=+1,545.85.
$$

However:

- losses rise from 5,735 to 6,191;
- loss rate rises from 7.55% to 8.15%;
- 631 baseline losses are repaired;
- 982 baseline wins become losses;
- mean threshold shift is $+0.080$;
- 45 of 52 cells favor earlier stopping;
- seven cells favor more patience.

P3d is therefore a utility improvement with a worse tail-safety profile.

## 4.7 Algorithm-v2 controlled experiments

The original experiment log reports utility sums. A value such as $+919.20$ is not a mean and cannot be compared to a 1,500-trajectory causal arm without its denominator. The corrected presentation separates:

1. the raw aggregate printed by the experiment;
2. the matched controlled contrast;
3. the mean controlled effect per experimental unit; and
4. a 95% uncertainty interval.

### 4.7.1 Evaluation design and denominators

| Experiments | Experimental unit and corpus | Holdout or fold design | Repetition and randomness |
|---|---|---|---|
| N1 LOCO/LOMO | 75,965 trajectories in 52 model-domain cells; 1,948 unique tasks | 52 leave-one-cell-out or 13 leave-one-model-out meta-calibration evaluations after 5-fold within-cell OOF probe fitting | One completed deterministic analysis; generation seed 7 at temperatures 0.1, 0.6, and 1.0; dataset shuffle seed 17 |
| N2a/N2b/N2c/N3 | Same 75,965 trajectories and 52 cells | Five run-group OOF folds for probes, followed by five task-group folds for threshold selection inside each cell | One completed analysis; gradient boosting uses random state 42; other estimators are deterministic |
| N4 | Same 75,965 trajectories and 52 cells | Five task-group folds per cell; one- and two-parameter rules use identical test trajectories | One completed deterministic analysis on the same seed-7 trace corpus |
| N5 | 1,500 matched Mistral-Small-22B/GSM8K trajectories per arm: 500 tasks x 3 temperatures | Paired 256-token versus 512-token cap; no fitted comparison model | One generation seed, 7, per task-temperature; 500 task clusters |
| N6 | 1,500 matched Qwen2.5-7B/GSM8K trajectories per precision arm: 500 tasks x 3 temperatures | Paired BF16 versus 4-bit step-2 correctness; no fitted comparison model | One generation seed, 7, per task-temperature; 500 task clusters |

The three temperatures are controlled generation conditions, not independent random-seed replications. Consequently, the intervals below measure heterogeneity across tasks or model-domain cells, not across new sampling seeds.

For N1-N4, the reported interval is a 10,000-draw percentile bootstrap over the 52 model-domain cells:

$$
\overline{\Delta U}^{*(b)}
=
\frac{\sum_{j\in I_b}\Delta U_j}
{\sum_{j\in I_b}N_j},
$$

where $I_b$ is a size-52 resample of cells. For N5-N6, complete `task_id` clusters are resampled 10,000 times. All intervals use master seed 804.

### 4.7.2 Normalized results

For N1-N4, "raw aggregate" is the sum of per-trajectory step-utility deltas. "Mean controlled effect" is the controlled total divided by 75,965 trajectories.

| Experiment | Raw aggregate printed by the run | Matched controlled contrast | Mean controlled effect (95% CI) | Decision |
|---|---:|---:|---:|---|
| N1 LOCO | $+919.20$ versus frozen hazard drift | $+919.20$; 59.5% of the $+1,545.85$ P3d target | $+0.01210$ [$+0.00522$, $+0.01907$] utility/trajectory | Positive transfer, but fails the pre-registered 60% recovery criterion |
| N1 LOMO | $+976.80$ versus frozen hazard drift | $+976.80$; 63.2% of target | $+0.01286$ [$+0.00683$, $+0.01911$] | Passes the model-transfer criterion |
| N2a | $-2,785.05$ total versus hazard drift | $-4,325.10$ versus matched logistic baseline | $-0.05694$ [$-0.08716$, $-0.03145$] | Reject gradient boosting |
| N2b | $-3,142.90$ total versus hazard drift | $-4,682.95$ versus matched logistic baseline | $-0.06165$ [$-0.09168$, $-0.03547$] | Reject isotonic calibration |
| N2c | $+1,791.70$ total versus hazard drift | $+251.65$ versus matched unlagged logistic baseline | $+0.00331$ [$+0.00062$, $+0.00633$] | Retain temporal lags |
| N3 | $+2,133.60$ total versus hazard drift | $+593.55$ versus matched cell-local logistic hazards | $+0.00781$ [$+0.00276$, $+0.01355$] | Retain empirical-Bayes shrinkage |
| N4 | $+1,700.00$ total for the two-parameter rule | $+159.50$ versus matched one-parameter threshold | $+0.00210$ [$+0.00049$, $+0.00403$] | Retain churn modulation |

The N2a and N2b rows expose an ambiguity in the earlier table: $-2,785.05$ and $-3,142.90$ are totals relative to frozen hazard drift, while the controlled decrements relative to the matched logistic arm are $-4,325.10$ and $-4,682.95$. N2c, N3, and N4 have cell-bootstrap intervals above zero. N3 has the largest positive matched effect, which supports the conclusion that variance control in late-step hazard estimates is the strongest Algorithm-v2 improvement.

The combined N4 plus N3 configuration has total $\Delta U=+2,179.60$ versus frozen hazard drift. It is the largest aggregate Algorithm-v2 total, but it is not an additional independently randomized experiment; it combines two retained changes and should not be interpreted as a separate causal effect.

The causal collection arms use different outcome units and therefore appear separately:

| Experiment | Arm counts | Controlled effect (95% task-cluster CI) | Additional diagnostic | Decision |
|---|---:|---:|---:|---|
| N5: 256 to 512 tokens | 454/1,500 losses and summed $\Delta U=-45.00$ in each arm; 0 discordant pairs | Loss-risk difference $0.0000$ [0.0000, 0.0000] | 500 matched tasks; three temperatures; one seed | No token-cap effect in this exact paired arm; new seeds are still required for generalization |
| N6: BF16 to 4-bit | 418/1,500 BF16 correct versus 204/1,500 4-bit correct at step 2 | Accuracy difference $+0.14267$ [$+0.11133$, $+0.17533$] | 348 BF16-only correct and 134 4-bit-only correct pairs; reported $Z=9.79$ | Precision has a causal effect in the tested task-temperature panel |

The full machine-readable table is `research/outputs/progress_report_review_metrics/algorithm_v2_normalized_effects.csv`.

## 4.8 Standardized detector tournaments

The prefix-safe ultimate tournament contains 28,888 source-qualified trajectories, five steps per trajectory, 2,948 unique task groups, and five outer `GroupKFold` splits by `task_id`. Its domain composition is not balanced:

| Domain | Trajectories | Share of micro average | Unique task groups | Model cells |
|---|---:|---:|---:|---:|
| ARC | 8,500 | 29.42% | 1,000 | 13 |
| GPQA | 5,824 | 20.16% | 448 | 13 |
| GSM8K | 8,064 | 27.91% | 1,000 | 13 |
| MATH | 6,500 | 22.50% | 500 | 13 |
| Total | 28,888 | 100.00% | 2,948 | 52 |

The original tournament CSV computed its token column from `thought_token_count`, which is a regex thought-word count even though the log called it generated reasoning tokens. The table below corrects that accounting by applying $\rho=0.0002$ to `raw_generation_tokens`. AUC, ECE, step utility, and win/tie/loss exactly reproduce the frozen tournament artifact; raw-token utility is the audit recomputation.

| Configuration | Micro OOF AUC | ECE | Micro step utility | Micro raw-token utility | Win/Tie/Loss |
|---|---:|---:|---:|---:|---:|
| Causal GRU | 0.874326 | 0.062469 | 0.326449 | 0.363126 | 21,559/6,382/947 |
| Causal Fourier neural operator | 0.870849 | 0.065370 | 0.328870 | 0.359883 | 23,796/3,777/1,315 |
| Selective SSM | 0.866304 | 0.061928 | 0.332553 | 0.358266 | 24,256/3,111/1,521 |
| Causal residual TCN | 0.864735 | 0.066292 | 0.331532 | 0.361147 | 24,199/3,284/1,405 |
| Causal RoPE transformer | 0.863837 | 0.064875 | 0.333123 | 0.363403 | 23,474/4,175/1,239 |
| Truncated Beta mixture | 0.863040 | 0.067063 | 0.331117 | 0.362804 | 22,933/4,764/1,191 |
| Five-expert causal MoE | 0.858343 | 0.066104 | 0.331025 | 0.355112 | 24,867/2,283/1,738 |
| MoE plus hysteresis | 0.858343 | 0.066104 | 0.321729 | 0.317286 | 22,956/4,681/1,251 |
| Task-grouped linear baseline | 0.829416 | 0.083296 | 0.324373 | 0.361571 | 21,597/6,050/1,241 |

### 4.8.1 Macro and fold-balanced correctness results

Task-macro AUC is the unweighted mean of within-task AUCs. Of 2,948 task groups, 2,679 contain both correctness classes and therefore have a defined AUC; 269 single-class groups are excluded from task-macro AUC but remain in utility calculations. Domain-macro AUC gives each of ARC, GPQA, GSM8K, and MATH weight $1/4$, independent of trajectory count. The fold interval is

$$
\overline{m}
\pm t_{0.975,4}\frac{s_m}{\sqrt{5}},
$$

computed over the five untouched outer-fold metrics. With only five folds, this interval describes fold variation; it is not a claim that overlapping model intervals imply equivalence.

| Configuration | Micro AUC | Task-macro AUC | Domain-macro AUC | Worst-domain AUC | Fold AUC 95% interval |
|---|---:|---:|---:|---:|---:|
| Causal GRU | 0.8743 | 0.8214 | 0.8102 | 0.6313 | [0.8685, 0.8803] |
| Causal Fourier neural operator | 0.8708 | 0.8172 | 0.8045 | 0.6152 | [0.8620, 0.8798] |
| Selective SSM | 0.8663 | 0.8116 | 0.7989 | 0.6078 | [0.8595, 0.8733] |
| Causal residual TCN | 0.8647 | 0.8088 | 0.7965 | 0.6057 | [0.8553, 0.8742] |
| Causal RoPE transformer | 0.8638 | 0.8085 | 0.7953 | 0.6079 | [0.8502, 0.8777] |
| Truncated Beta mixture | 0.8630 | 0.8140 | 0.7951 | 0.6204 | [0.8565, 0.8697] |
| Five-expert causal MoE | 0.8583 | 0.8005 | 0.7882 | 0.5955 | [0.8488, 0.8681] |
| MoE plus hysteresis | 0.8583 | 0.8005 | 0.7882 | 0.5955 | [0.8488, 0.8681] |
| Task-grouped linear baseline | 0.8294 | 0.7773 | 0.7612 | 0.6000 | [0.8265, 0.8324] |

The per-domain AUCs reveal that GPQA is the worst ranking domain for every architecture:

| Configuration | ARC | GPQA | GSM8K | MATH | Domain macro |
|---|---:|---:|---:|---:|---:|
| Causal GRU | 0.8679 | 0.6313 | 0.8808 | 0.8610 | 0.8102 |
| Causal Fourier neural operator | 0.8653 | 0.6152 | 0.8818 | 0.8558 | 0.8045 |
| Selective SSM | 0.8594 | 0.6078 | 0.8790 | 0.8493 | 0.7989 |
| Causal residual TCN | 0.8629 | 0.6057 | 0.8739 | 0.8437 | 0.7965 |
| Causal RoPE transformer | 0.8563 | 0.6079 | 0.8763 | 0.8405 | 0.7953 |
| Truncated Beta mixture | 0.8511 | 0.6204 | 0.8670 | 0.8421 | 0.7951 |
| Five-expert causal MoE | 0.8516 | 0.5955 | 0.8700 | 0.8359 | 0.7882 |
| MoE plus hysteresis | 0.8516 | 0.5955 | 0.8700 | 0.8359 | 0.7882 |
| Task-grouped linear baseline | 0.8214 | 0.6000 | 0.8101 | 0.8131 | 0.7612 |

### 4.8.2 Macro and per-domain stopping results

The utility table prevents pooled ARC and GSM8K cells from determining model selection. Worst-domain step utility is the minimum of the four domain means. Worst-domain loss is the maximum domain loss rate under the win/tie/loss definition in Section 4.3.

| Configuration | Micro step U | Domain-macro step U | Worst-domain step U | Micro raw-token U | Domain-macro raw-token U | Worst-domain loss | Fold step-U 95% interval |
|---|---:|---:|---:|---:|---:|---:|---:|
| Causal GRU | 0.3264 | 0.2990 | 0.0895 | 0.3631 | 0.3350 | 4.02% | [0.3172, 0.3357] |
| Causal Fourier neural operator | 0.3289 | 0.3023 | 0.0948 | 0.3599 | 0.3324 | 5.78% | [0.3185, 0.3392] |
| Selective SSM | 0.3326 | 0.3065 | 0.1032 | 0.3583 | 0.3311 | 7.03% | [0.3205, 0.3446] |
| Causal residual TCN | 0.3315 | 0.3053 | 0.0985 | 0.3611 | 0.3339 | 6.24% | [0.3202, 0.3428] |
| Causal RoPE transformer | 0.3331 | 0.3068 | 0.1010 | 0.3634 | 0.3361 | 5.44% | [0.3220, 0.3442] |
| Truncated Beta mixture | 0.3311 | 0.3046 | 0.0941 | 0.3628 | 0.3353 | 5.01% | [0.3209, 0.3413] |
| Five-expert causal MoE | 0.3310 | 0.3053 | 0.1022 | 0.3551 | 0.3283 | 8.66% | [0.3207, 0.3414] |
| MoE plus hysteresis | 0.3217 | 0.2951 | 0.0806 | 0.3173 | 0.2887 | 6.09% | [0.3141, 0.3294] |
| Task-grouped linear baseline | 0.3244 | 0.2981 | 0.0892 | 0.3616 | 0.3346 | 5.89% | [0.3137, 0.3351] |

Per-domain step utility is:

| Configuration | ARC | GPQA | GSM8K | MATH | Domain macro |
|---|---:|---:|---:|---:|---:|
| Causal GRU | 0.6159 | 0.1532 | 0.3374 | 0.0895 | 0.2990 |
| Causal Fourier neural operator | 0.6164 | 0.1658 | 0.3322 | 0.0948 | 0.3023 |
| Selective SSM | 0.6174 | 0.1729 | 0.3325 | 0.1032 | 0.3065 |
| Causal residual TCN | 0.6160 | 0.1717 | 0.3350 | 0.0985 | 0.3053 |
| Causal RoPE transformer | 0.6178 | 0.1715 | 0.3369 | 0.1010 | 0.3068 |
| Truncated Beta mixture | 0.6155 | 0.1695 | 0.3391 | 0.0941 | 0.3046 |
| Five-expert causal MoE | 0.6197 | 0.1764 | 0.3229 | 0.1022 | 0.3053 |
| MoE plus hysteresis | 0.6102 | 0.1632 | 0.3266 | 0.0806 | 0.2951 |
| Task-grouped linear baseline | 0.6151 | 0.1664 | 0.3216 | 0.0892 | 0.2981 |

The balanced interpretation is narrower than the earlier pooled conclusion:

- the GRU leads micro AUC, task-macro AUC, domain-macro AUC, worst-domain AUC, and worst-domain loss safety;
- the RoPE transformer leads micro and domain-macro step utility and corrected raw-token utility;
- the SSM has the highest worst-domain step utility, but its 7.03% worst-domain loss rate is less conservative than the GRU's 4.02%;
- fold intervals overlap across the leading neural architectures, so this evidence supports a candidate set rather than a unique winner;
- the MoE's large win count does not compensate for its 8.66% worst-domain loss rate.

Accordingly, the next locked evaluation should carry three pre-declared candidates: GRU as the conservative ranking/safety policy, RoPE as the mean-utility policy, and SSM as the worst-domain-utility policy. No architecture is selected solely from the micro average.

The recomputation is stored in `research/outputs/progress_report_review_metrics/tournament_balanced_summary.csv`, `tournament_per_domain.csv`, and `tournament_per_fold.csv`.

The later Blackwell stacked correctness detector reports:

These three OOF scores use 144,440 step rows from 28,888 stored five-step sequences and 2,948 task groups. The five outer folds are grouped by `task_id`; all learned models and, where applicable, scalers or thresholds are fit within the training tasks before scores are assigned to held-out rows. The trajectory probe's final-correctness score is repeated over the corresponding stored sequence rows for the stack, so this remains a retrospective correctness-ranking evaluation rather than five-step generation-time stopping.

| Model | OOF AUC |
|---|---:|
| No-peer LightGBM control | 0.943223 |
| Full-trajectory deep hybrid MoE | 0.936430 |
| 60/40 LightGBM–HistGradientBoosting stacked ensemble | 0.955156 |

For the stacked-versus-control comparison:

$$
\widehat{\Delta\operatorname{AUC}}=0.011937,
$$

$$
\mathrm{CI}_{0.95}=[0.010384,0.013480],
$$

with 10,000 task-cluster bootstrap draws. The JSON field called `p_value_delta_greater_zero` equals 1.0, but this is an empirical bootstrap proportion $\Pr^*(\Delta>0)$, not a conventional $p$-value.

The $0.955156$ result is a correctness-ranking result, not an end-to-end stopping-policy result. The deep MoE sees the full trajectory and its score is attached to each row, so the stack is retrospective.

## 4.9 Peer-dynamics generalization

The anonymous peer-dynamics model reports pooled OOF AUC $0.954664$, Brier score $0.077421$, ECE-15 $0.012067$, and task-cluster bootstrap AUC interval $[0.949871,0.958757]$. The pooled values use 144,440 step rows and 2,948 task groups from the standardized corpus; leave-one-domain-out rows are reported separately below and are not pooled into that interval.

Per-domain AUC is:

| Domain | OOF AUC |
|---|---:|
| ARC | 0.978745 |
| GPQA | 0.714039 |
| GSM8K | 0.975494 |
| MATH | 0.928579 |

Leave-one-domain-out transfer is:

| Held-out domain | Test step rows | LODO AUC |
|---|---:|---:|
| ARC | 42,500 | 0.968836 |
| GPQA | 29,120 | 0.700305 |
| GSM8K | 40,320 | 0.927793 |
| MATH | 32,500 | 0.868882 |

GPQA is the clear external-validity bottleneck.

Corrupting $k$ peer streams changes AUC as follows:

| Corrupted peers | OOF AUC |
|---:|---:|
| 0 | 0.954664 |
| 1 | 0.953484 |
| 2 | 0.952565 |
| 3 | 0.951260 |
| 5 | 0.950622 |

The permutation arm yields AUC $0.947287$. This is a decrease of only $0.007377$ from the unpermuted baseline, so the current artifacts do not support a claim that permutation “destroys” performance. They show that much of the predictive signal survives without correctly aligned peer structure.

## 4.10 Offline active-stop replay

For 1,500 Qwen2.5-7B GSM8K trajectories and 15,000 pre-generated steps:

| Policy | Counted completion tokens | Accuracy |
|---|---:|---:|
| Never stop | 827,804 | 70.53% |
| Replayed hazard stop | 377,960 | 64.20% |

Token saving is

$$
1-\frac{377,960}{827,804}=0.5434.
$$

Accuracy change is

$$
64.20\%-70.53\%=-6.33\text{ percentage points}.
$$

Relative accuracy retained is

$$
\frac{64.20}{70.53}=91.03\%.
$$

The script fits and evaluates its logistic models on the same trace file without a train/test split. This result is exploratory and optimistic. Its Markdown claims of “optimal” stopping, “accuracy preservation,” sub-millisecond latency, and direct Hugging Face loop integration are not established by the implementation.

## 4.11 Mathematical limitations requiring proof or correction

1. The one-crossing theorem is conditional. The assumptions that $q_t$ is nondecreasing, $\alpha_t$ is nonincreasing, and $\beta_t$ is nondecreasing are not universal properties of reasoning traces; corruption itself violates monotone realized correctness.
2. OSLA optimality requires a monotone stopping region, bounded or suitably integrable stopping times, and a value process consistent with the deployed estimator. These conditions must be stated in the thesis theorem.
3. The code uses $C_t-\lambda(t-1)$, while several prose derivations use $-\lambda t$. The difference is an additive constant for fixed trajectories but must be standardized.
4. Population empirical-Bernstein and e-process bounds are not automatically individual-trajectory safety guarantees.
5. Grouping by `run_id` is insufficient when multiple temperatures share a raw task. Final claims should use task-grouped splits.
6. AUC does not imply positive stopping utility.
7. Full-trajectory features cannot be described as online or prefix-safe.

# 5. Detailed Commit Chronology and Experimental Evolution

## 5.1 Repository artifact census

HEAD tracks 6,539 files. The repository is artifact-heavy rather than source-heavy:

| Artifact class | Tracked files | Interpretation |
|---|---:|---|
| NumPy archives (`.npz`) | 3,901 | Per-run or per-trajectory hidden-state/logit feature payloads |
| CSV tables (`.csv`) | 1,465 | Step traces, cell summaries, fold results, and tournament outputs |
| Markdown (`.md`) | 338 | Protocols, analyses, audits, experiment notes, and thesis material |
| JSON (`.json`) | 328 | Manifests, configuration/state records, and machine-readable reports |
| PNG (`.png`) | 323 | Diagnostic and publication-candidate plots |
| Python (`.py`) | 98 | 84 under `research/`; loaders, inference, analysis, tournaments, validation |
| Runtime logs (`.log`) | 40 | RunAI, sweep, and tournament execution records |
| Safetensors (`.safetensors`) | 20 | Qwen3.5 35B GPTQ model-offload tensor fragments from an unsuccessful smoke path |
| Shell (`.sh`) | 12 | Autonomous, RunAI, matrix, Tier-3, and tournament launchers |
| Text (`.txt`) | 4 | Plain-text supporting material |
| HTML (`.html`) | 2 | Rendered reports |
| PDF (`.pdf`) | 2 | Tracked report artifacts; the enrollment-status PDF analyzed here is untracked |
| PowerShell (`.ps1`) | 2 | Proposal export and MCP startup |
| Jupyter (`.ipynb`) | 1 | `tools/runai/jupyter_repo_gpu_sanity.ipynb` |
| JSON Lines (`.jsonl`) | 1 | Line-oriented data artifact |

There are no tracked YAML files. Configuration is encoded primarily in Python dataclasses/constants, shell arguments, JSON manifests, and checkpoint metadata.

Four tracked Python paths match test-oriented naming:

1. `research/tests/test_boundary_floor.py`;
2. `research/tests/test_graders.py`;
3. `research/test_v2_forced_choice_protocol.py`; and
4. `research/run_perfect_overthinking_stress_test.py`.

The first two are focused regression scripts. The forced-choice file is a synthetic no-model integration test, and the fourth path is a stress-test executable. None defines pytest-style `test_*` functions; each is invoked through a `main()` or `run()` path. The numerical experiment suite therefore relies more heavily on executable validation pipelines and result audits than on isolated pytest tests.

## 5.2 Forensic method

The history corpus is the 469 commits reachable from HEAD, not every object on every local ref. It spans March 31 through July 23, 2026:

- inclusive calendar span: 115 days;
- active commit days: 38;
- merge commits: 7;
- root: `6f89d5b`, “Add Colab runner and research package”;
- HEAD: `7400d3a`, “checkpoint: Phase 1 Complete - Control Baseline OOF AUC: 0.943223.”

The author-name distribution is:

| Author string | Commits |
|---|---:|
| Aditya Bhatt | 388 |
| RunAI Matrix | 63 |
| Aditya | 8 |
| Aditya R Bhatt | 6 |
| bhattadiCS | 2 |
| GitHub Copilot | 2 |

Monthly counts are:

| Month | Commits |
|---|---:|
| March | 2 |
| April | 82 |
| May | 28 |
| June | 107 |
| July | 250 |

Git numstat totals are 14,801,579 inserted lines, 357,276 deleted lines, 4,360 binary-file touches, and 7,799 total file touches. These figures are dominated by committed CSV traces, manifests, logs, images, and checkpoint histories. They must not be interpreted as source-code growth.

## 5.3 Multi-label topic clustering

Commit subjects and changed paths were classified with case-insensitive regular-expression families. A commit can occupy multiple categories; counts therefore do not sum to 469.

| Topic | Commits | Share of history | Subject/path trigger families |
|---|---:|---:|---|
| Data collection/results | 309 | 65.9% | `result`, `trace`, `data(set)`, `collect`, `corpus`, `telemetry`, `experiment`, `sweep`, `matrix`, `checkpoint`, `output`; output/data/result/experiment path segments; `.csv`; `.npz` |
| Architecture/infrastructure | 108 | 23.0% | `infra`, `pipeline`, `runner`, `orchestrat`, `harness`, `architecture`, `gpu`, `cuda`, `runai`, `colab`, `memory`, `parallel`, `cache`, `launcher`, `model support`, `bootstrap`; `tools/`; run/launch/bootstrap/orchestration scripts; requirements/environment/setup/server paths |
| Metrics/overthinking detection | 273 | 58.2% | `metric`, `hazard`, `boundary`, `stopp`, `detector`, `overthink`, `auc`, `entropy`, `confidence`, `grader`, `utility`, `drift`, `oracle`, `correctness`, `committee`, `peer`, `algorithm`; matching analysis/tournament paths |
| Hyperparameter tuning | 55 | 11.7% | `hyper`, `tuning`, `optuna`, `grid`, `search`, `threshold`, `calibrat`, `temperature`, `quantiz`, `ablation`, `trial`, `parameter`; matching sweep/calibration paths |
| Bug fixes/hardening | 81 | 17.3% | `fix`, `repair`, `harden`, `correct`, `recover`, `sanitiz`, `stabiliz/stabilis`, `fallback`, `timeout`, `oom`, `audit`, `bug`, `corrupt`, `zero-byte`, `empty CSV`, `leak`; test/audit paths |
| Documentation | 137 | 29.2% | `doc`, `report`, `proposal`, `methodology`, `readme`, `thesis`, `advisor`, `paper`, `explain`, `analysis`; `ThesisDocs/`; README/doc paths; `.md`; `.pdf` |
| Validation/testing | 130 | 27.7% | `test`, `verify`, `validat`, `stress`, `oof`, `holdout`, `bootstrap`, `falsif`, `sanity`, `smoke`, `regression`; test/stress/audit/OOF/smoke paths |
| Unclassified | 8 | 1.7% | No trigger above |

The dominant conventional subject prefixes are `results` (90), `docs` (60), no prefix (57), matrix checkpoint (55), `fix` (43), `exp` (36), `feat` (30), and `wip` (22).

## 5.4 Developmental epochs

| Phase | Dates | Commits | Insertions | Deletions | File touches | Principal outcome |
|---|---|---:|---:|---:|---:|---|
| I. Groundwork and infrastructure | Mar 31–Apr 2 | 33 | 248,559 | 1,094 | 3,498 | Colab/L4 harness, Qwen/DeepSeek/Mistral runs, baseline protocol |
| II. Metric formulation and detector prototyping | Apr 3–Jul 12 | 262 | 12,516,024 | 352,902 | 3,688 | Model catalog, graders, 52-cell matrix, boundary corrections, OOF audit, Algorithm-v2 |
| III. Controlled model/data sweeps | Jul 13–Jul 18 | 138 | 1,428,189 | 3,026 | 499 | Tier-3 telemetry, five-step global sweep, 52-cell standardized corpus, advanced tournament |
| IV. Refactoring, optimization, and validation | Jul 19–Jul 23 | 36 | 608,807 | 254 | 114 | Prefix-safe tournament completion, peer dynamics, stress tests, Blackwell stacked detectors |

### Phase I: Groundwork and infrastructure

Key changes:

- `6f89d5b` created the research package and Colab runner.
- `92681da` introduced flexible prompt modes and an answer-only baseline.
- `2351c8f` added cross-family L4 results.
- `6b5ad36` preserved runtime context during output refresh.
- `afad4f0` recovered a Qwen-7B run and regenerated analysis.
- `ba7e28f`, `13ad1ff`, and `1641b2c` established and repaired the Mistral L4 path.

The initial detector concept was heuristic: first answer, answer stability, confidence, entropy, and hidden-state movement. The experiment harness already collected the signals required for later hazard modeling.

### Phase II: Metric formulation and overthinking detection prototyping

Infrastructure and model expansion:

- `2a883cc` and `089bf59` added frontier-model catalog entries and 4-bit L4 optimization.
- `ad6faad` fixed IO management, hidden directory handling, and multimodal loading.
- May commits added bitsandbytes CPU offload, meta-state dispatch compatibility, and low-memory scoring.
- `45072d1`, `9a99aff`, and `32963ca` separated text-only Gemma loading from unsupported multimodal paths.

Dataset and matrix construction:

- `376254b` and `589b797` hardened GSM8K loading.
- `1945022` added the experiment-matrix orchestrator and RunAI launcher.
- `0841135` added data-driven cross-family aggregation and MATH parsing.
- `122358a` added Mistral Small, GPQA, and ARC.
- `f89ff38` completed all 52 successful canonical cells after sanitizing hard-kill-corrupted rows.

Mathematical corrections:

- `ef45dda` repaired graders, leakage, and boundary logic.
- `6f832ea` corrected the canonical boundary operator and regenerated results out of sample.
- `8ce9b9f` applied $T_{\min}=2$ consistently.
- `0ea16f4` added the scientific-rigor audit and loss autopsy.
- `d2a42e8` tested P3b/P3c/P3d; only per-cell threshold calibration survived.
- `cca2113` productized the calibrated thresholds.
- `36a13c9` pre-registered Algorithm-v2 N1–N9.
- `253a05f`, `89ff672`, and `4a4f745` recorded N1–N6 outcomes.

The conceptual evolution in this phase was:

$$
\text{fixed heuristic}
\rightarrow
\text{learned correctness probe}
\rightarrow
\text{repair/corruption hazards}
\rightarrow
\text{drift boundary}
\rightarrow
\text{OOF-calibrated and shrinkage-stabilized boundary}.
$$

### Phase III: Controlled model and data sweeps

- `07fedbd` added N7 self-consistency and N8 answer-span/mid-layer observables.
- `fc69dab` collected Qwen-7B GSM8K/MATH Tier-3 telemetry.
- `d687103` recorded the deep sequence tournament.
- July 13–18 auto-commit results populated standardized global cells.
- `0a58d8b`, `fcddab3`, and later finalization commits marked repeated global-sweep checkpoints.
- `5ee5cbe`, `cf9b0d0`, and `927b79d` added Qwen3.5 MATH, ARC, and GPQA; the Qwen3.5 GSM8K cell remained at 64 tasks.
- `4363048` introduced the advanced 96-GB GPU hyper-optimization tournament.
- `8cbb546` through `352bf28` checkpointed the five outer folds.
- `f43ac16` completed the advanced tournament.

Systems optimization was scientifically consequential:

- `4599765` cached random projections;
- `ee2521d` reverted an unstable GPU projection path;
- `5a06073` and `646b379` moved pooling to GPU and reduced transfer volume;
- `49401f8`, `56759a9`, and related commits added timeouts for pathological SymPy expressions;
- `03a93b8` and `92efe50` handled zero-byte/empty CSV checkpoints.

### Phase IV: Refactoring, optimization, and testing

- `91a05b3` added the ultimate multi-day GPU tournament with nested task-grouped evaluation.
- `83314c5` synchronized final ultimate results.
- `a9bf282` added closed-barrier anonymous peer dynamics, reporting fold-mean AUC near 0.9547.
- `d365845` fixed peer-dynamics type and entropy-feature bugs.
- `a65e7fb` completed the five-arm peer stress battery.
- `0cf8c29`, `84bec18`, and `48635df` added high-order features, CausalAttractorGRU variants, ensembles, and bootstrap verification.
- `25292e5` completed a perfect-detector weakness battery.
- `e299697` recorded the unsupported $0.971662$ checkpoint claim.
- `20e6b22` added feature standardization and repeated the $0.9717$ claim.
- `0fb9bd1` and July 23 commits built the five-day Blackwell suite, 16-scale EMA and fifth-order “crackle” features, dual neural architectures, checkpointing, and larger epoch modes.
- `7400d3a` ends the reachable history with the reproducible control AUC $0.943223$.

## 5.5 Algorithmic evolution

| Generation | Detection object | Estimator | Principal weakness addressed |
|---|---|---|---|
| G0 | First answer or fixed final step | Deterministic rule | No uncertainty or transition model |
| G1 | Stable answer/low entropy/high confidence | Threshold heuristic | Captures surface stabilization only |
| G2 | Current correctness $q_t$ | Logistic probe | Ignores repair versus corruption asymmetry |
| G3 | Continuation drift $\mu_t$ | Separate $q$, $\alpha$, $\beta$ logistic heads | Maps prediction to utility-aware stopping |
| G4 | Calibrated per-cell boundary | OOF threshold shift $\delta_{\mathrm{cell}}$ | Adapts to model/dataset heterogeneity |
| G5 | Shrunk late-step hazards | Empirical-Bayes pooling | Reduces sparse-tail variance |
| G6 | Temporal and difficulty-aware boundary | Lags and two-parameter churn | Uses prefix dynamics rather than a static cell offset |
| G7 | Sequence correctness probes | GRU/LSTM/TCN/SSM/Transformer/FNO/MoE | Learns nonlinear trajectory structure |
| G8 | Peer/committee dynamics | Closed-panel consensus features | Exploits cross-model agreement |
| G9 | High-order retrospective stack | 16-scale EMA, derivatives, MoE, tree ensemble | Maximizes correctness ranking; not yet an online stopper |

The current frontier is split:

- G6/G7 contains the most defensible prefix-safe stopping work.
- G8/G9 contains the highest AUC correctness detectors but needs causality and deployment constraints before being called a stopping policy.

## 5.6 Phase I deep dive: from a benchmark question to an auditable trajectory

### Why the infrastructure is part of the scientific contribution

Phase I did more than make models run. It defined the experimental unit that every later claim depends on. A conventional benchmark records one prompt and one final answer. This thesis instead records a *trajectory*: the sequence of intermediate answers, confidence reports, generated-token statistics, hidden-state summaries, and correctness labels produced while the model is repeatedly asked to continue or verify.

For task $i$, the stored object is

$$
\mathcal{T}_i
=
\left\{
\mathbf{z}_{i,1},
\mathbf{z}_{i,2},
\ldots,
\mathbf{z}_{i,T_i}
\right\},
$$

where each step record contains

$$
\mathbf{z}_{i,t}
=
\left(
A_{i,t},
C_{i,t},
H_{i,t},
\ell_{i,t},
\mathbf{h}_{i,t},
B_{i,t},
R_{i,t}
\right).
$$

Here $A_{i,t}$ is the parsed answer, $C_{i,t}$ is correctness, $H_{i,t}$ is mean token entropy, $\ell_{i,t}$ is mean generated-token log probability, $\mathbf{h}_{i,t}$ is a pooled hidden-state representation, $B_{i,t}$ is the generated-token count, and $R_{i,t}$ contains runtime and parsing diagnostics. This schema converts an informal question - “did the model think too long?” - into a sequence of measurable state transitions.

### End-to-end Phase I pipeline

| Stage | Plain-language purpose | Implementation | Persisted evidence |
|---|---|---|---|
| 1. Load a task | Convert four benchmark formats into one common task object. | `load_gsm8k_tasks()`, `load_math_tasks()`, `load_arc_tasks()`, and `load_gpqa_tasks()` in `research/real_trace_experiments.py:992–1183`. | Task ID, source index, domain, difficulty, answer type, expected answer. |
| 2. Fix identity and randomness | Ensure a run can be resumed and compared under the same task, model, temperature, and seed. | `run_id_for()` at line 1223; seed and temperature are written into every row. | Stable `run_id`, `task_id`, seed, temperature, model identifier. |
| 3. Construct an incremental prompt | Show the model the task and prior step summaries, then request exactly one new reasoning or verification increment. | `conversation_prompt()` at line 764 and `render_prompt()` at line 812. | Prompt mode, system-prompt mode, step number, prior-answer history. |
| 4. Generate one reasoning increment | Ask the frozen language model to produce the current thought, answer, confidence, and stop flag. | `generate_batch_with_diagnostics()` at line 1759; `model.generate()` uses temperature, top-$p=0.95$, and repetition penalty $1.02$. | Raw response text and generated-token count. |
| 5. Score the generated tokens | Run a second forward pass to measure how probable and uncertain the generated tokens were. | `low_memory_scoring_pass()` and the diagnostic forward path at lines 1670–1979. | Mean log probability, entropy mean/standard deviation, final- and mid-layer summaries. |
| 6. Parse and grade | Convert free-form model text into a typed answer and compare it with the benchmark reference. | `parse_generation()`, `normalize_answer()`, and `verify_answer()` at lines 713–984. | Normalized answer, parse status, extraction source, binary correctness. |
| 7. Measure state change | Compare the current step with preceding steps. | Row assembly at lines 2320–2439. | Answer change, lexical echo, hidden-state $L_2$/cosine shift, token cap, utility. |
| 8. Persist and recover | Append durable records, remove incomplete rows, and resume interrupted GPU jobs without duplicating finished trajectories. | `append_records()` and `reconcile_existing_outputs()` at lines 1264–1472. | Trace CSV, run summary, batch metrics, metadata, reconciliation backup. |

The two-pass design in stages 4 and 5 is important. `model.generate()` produces the response, but the implementation does not request and retain every full-vocabulary score during generation. The second forward pass re-scores the completed response so entropy, token log probability, and hidden states can be collected with lower peak memory. This enables larger models on the available GPUs, but it also means the current collector observes a completed reasoning increment rather than making a stop decision inside each decoder-token loop.

### Concepts used in the trace files

| Term | Accessible definition | What it can and cannot establish |
|---|---|---|
| Token | A model-specific text unit; it may be a word, part of a word, punctuation, or whitespace. | Generated-token counts measure model work more directly than character counts. They are not identical across tokenizers. |
| Logit | An unnormalized score assigned to each possible next token before softmax converts scores to probabilities. | Logits support probability and entropy calculations. A large logit for one token does not prove the final answer is correct. |
| Entropy | A summary of how spread out the next-token probability distribution is. Low entropy means the model strongly favors a narrow set of tokens. | Low entropy measures linguistic certainty, not mathematical correctness. A model can be confidently wrong. |
| Hidden state | An internal vector representation computed by a transformer layer for a token position. | Movement can indicate a changing internal representation. Its semantic meaning is learned indirectly and is not a human-readable proof trace. |
| Quantization | Storing or computing model weights with fewer bits, such as 4-bit rather than BF16, to reduce memory. | It enables larger models on limited hardware, but N6 shows that it can change early-step correctness. |
| Calibration | Agreement between predicted probabilities and observed frequencies. Among predictions near $0.8$, approximately 80% should be correct. | Calibration is required for risk-aware stopping. Ranking quality alone does not provide calibrated probabilities. |
| Out-of-sample evaluation | Testing on tasks that were not used to fit model parameters or stopping thresholds. | It estimates generalization and limits leakage. It does not guarantee performance on a genuinely new domain. |

### Phase I completion evidence

Phase I has six explicitly auditable foundation gates:

| Foundation gate | Evidence | Status |
|---|---|---|
| Four domain loaders produce a common task schema. | GSM8K, MATH-500, ARC-Challenge, and GPQA loaders; standardized domain counts reconcile to 28,888 trajectories. | Complete |
| Model-family abstraction supports heterogeneous decoder architectures. | `MODEL_CATALOG`, attention fallbacks, device mapping, and 15 successful model identifiers across complete and pilot experiments. | Complete |
| Stepwise prompt and response contract is deterministic and parseable. | Structured JSON and four-line protocols; parse and extraction-source fields are stored per row. | Complete |
| Token-, uncertainty-, representation-, and runtime-level telemetry is recorded. | 55-column standardized trace header; 144,440 current step rows. | Complete |
| Interrupted GPU collection is resumable and sanitized. | Reconciliation, backup, zero-byte/empty-CSV handling, and 52 successful canonical cells. | Complete with documented repaired rows |
| Baseline grading and stopping primitives are regression-tested. | Boundary-floor regression: 6/6 passed; grader regression: 30/30 passed; synthetic forced-choice protocol passed and rejected both tampered variants. | Complete |

The Phase I gate score is therefore

$$
\frac{6\ \text{completed foundation gates}}
{6\ \text{defined foundation gates}}
=100\%.
$$

This is an artifact-completion score. It does not imply that every collected dataset row is equally representative or that the online stopping problem is solved.

## 5.7 Phase II deep dive: from trajectory observations to a stop decision

### Why confidence or answer stability alone is insufficient

Two real Qwen2.5-7B/GSM8K trajectories in the standardized corpus illustrate the scientific problem.

**Case A: correct, corrupted, then repaired.** Task `gsm8k_train_00008_cd4fe6cf` has reference answer 110.

| Step | Normalized answer | Correct | Reported confidence | Mean entropy | Generated tokens | Transition | Step utility |
|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 50 | No | 50 | 0.1030 | 27 | Initial answer | 0.00 |
| 2 | 110 | Yes | 80 | 0.0597 | 40 | Repair | 0.95 |
| 3 | 22 | No | 80 | 0.1337 | 30 | Corruption | -0.10 |
| 4 | 110 | Yes | 80 | 0.1104 | 36 | Repair | 0.85 |
| 5 | 110 | Yes | 80 | 0.1092 | 33 | Stable correct answer | 0.80 |

The answer is already correct at step 2. Stopping there would use 67 of the trajectory's 166 generated tokens:

$$
1-\frac{67}{166}=59.64\%\ \text{token saving}.
$$

It would also improve realized step utility from $0.80$ at step 5 to $0.95$ at step 2. This is a favorable early-stop example, not a population-level performance claim.

**Case B: confidently stable but wrong until the final repair.** Task `gsm8k_train_00006_4cf43e2d` has reference answer 10.

| Step | Normalized answer | Correct | Reported confidence | Mean entropy | Generated tokens | Answer changed | Step utility |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50 | No | 50 | 0.1398 | 36 | No prior answer | 0.00 |
| 2 | 70 | No | 80 | 0.2874 | 33 | Yes | -0.05 |
| 3 | 70 | No | 80 | 0.2225 | 29 | No | -0.10 |
| 4 | 70 | No | 80 | 0.1804 | 28 | No | -0.15 |
| 5 | 10 | Yes | 100 | 0.1741 | 43 | Yes | 0.80 |

At steps 3 and 4 the answer is stable, confidence is 80, and entropy is declining, yet the answer remains wrong. A rule that equates stability, confidence, or falling entropy with correctness would stop too early. The detector must estimate both sides of continuation: the chance that another step repairs an error and the chance that another step corrupts a correct answer.

### The three learned probabilities

| Symbol | Question answered | Training label |
|---|---|---|
| $q_t$ | “Given everything observed through step $t$, how likely is the current answer to be correct?” | Current correctness $C_t$. |
| $\alpha_t$ | “If the current answer is wrong, how likely is the next answer to become correct?” | Repair transition $C_t=0,\ C_{t+1}=1$. |
| $\beta_t$ | “If the current answer is correct, how likely is the next answer to become wrong?” | Corruption transition $C_t=1,\ C_{t+1}=0$. |

The estimated one-step value of continuing is

$$
\widehat\mu_t
=
(1-\widehat q_t)\widehat\alpha_t
-\widehat q_t\widehat\beta_t
-\lambda.
$$

The three terms have direct interpretations:

1. $(1-\widehat q_t)\widehat\alpha_t$ is expected repair benefit;
2. $\widehat q_t\widehat\beta_t$ is expected corruption risk; and
3. $\lambda$ is the explicit cost of one more reasoning step.

An illustrative early state with $\widehat q_t=0.35$, $\widehat\alpha_t=0.45$, $\widehat\beta_t=0.05$, and $\lambda=0.05$ gives

$$
\widehat\mu_t
=(0.65)(0.45)-(0.35)(0.05)-0.05
=0.225.
$$

Continuation has positive estimated value. A later state with $\widehat q_t=0.82$, $\widehat\alpha_t=0.10$, and $\widehat\beta_t=0.08$ gives

$$
\widehat\mu_t
=(0.18)(0.10)-(0.82)(0.08)-0.05
=-0.0976.
$$

The policy stops because estimated repair benefit no longer covers corruption risk and step cost. These numbers explain the rule; they are not fitted values from either case study.

### How the models are trained without giving them the answer at deployment

Ground-truth answers are used during research to create training labels. They are not inputs to the deployed feature vector. The process is:

1. Collect complete trajectories while retaining the benchmark answer in a sealed grading field.
2. Grade every recorded step to create $C_t$, repair, and corruption labels.
3. Remove the reference answer and all future-step information from the feature matrix.
4. Fit the correctness, repair, and corruption models on training tasks.
5. Apply the fitted models to held-out task prefixes.
6. Convert held-out probabilities into a stopping step and compare it with never-stop and hindsight-oracle controls.

This separation matters because an oracle detector can identify the first correct answer only by reading the ground truth. It is useful as an upper bound, but it is not deployable.

### Why out-of-fold evaluation is required

If the same task is used both to learn a threshold and to report its performance, the result can be optimistic even if the code never explicitly uses the answer as a feature. Cross-validation reduces this problem:

$$
\mathcal{D}
=
\mathcal{D}_1\cup\cdots\cup\mathcal{D}_5,
\qquad
\mathcal{D}_j\cap\mathcal{D}_k=\varnothing
\quad(j\ne k).
$$

For fold $k$, models and thresholds are fit on $\mathcal{D}\setminus\mathcal{D}_k$ and evaluated only on $\mathcal{D}_k$. Every task becomes test data exactly once. Early Phase II code grouped by `run_id`, which protected steps within one trajectory but could separate temperatures of the same raw task. The later tournament uses `task_id`, preventing the same benchmark question across models or temperatures from entering both sides of an outer fold. Final claims use the stricter task-grouped design.

### Why empirical-Bayes shrinkage improved the boundary

Late reasoning steps contain fewer observations because horizons differ and incomplete runs are removed. A cell-specific repair estimate such as one repair in two observed wrong states equals 0.5, but that estimate is extremely uncertain. Empirical-Bayes shrinkage combines the cell count with a global step-level rate:

$$
\widehat\alpha_{c,t}^{\mathrm{shrunk}}
=
\frac{R_{c,t}+k\overline{\alpha}_t}
{N_{c,t}^{\mathrm{wrong}}+k}.
$$

$R_{c,t}$ is the number of repairs, $N_{c,t}^{\mathrm{wrong}}$ is the number of eligible wrong states, $\overline{\alpha}_t$ is the broader repair rate, and $k$ controls how much evidence the cell needs before it can move far from the global estimate. In plain language, a sparse cell is prevented from making an extreme claim from one or two transitions. Across 75,965 trajectories, N3 improves matched out-of-fold utility by $+593.55$, or $+0.00781$ per trajectory with 52-cell bootstrap 95% CI $[+0.00276,+0.01355]$. This is the largest single positive Algorithm-v2 controlled effect.

### What Phase II experiments established

| Question | Controlled evidence | Conclusion |
|---|---|---|
| Is step 1 a legitimate stopping decision? | Raising the enforced floor from step 2 to step 3 changes utility by $-1,912.35$; boundary regression 6/6 passed. | Step 1 is a forced initialization; $T_{\min}=2$ is retained. |
| Should an empty answer always block stopping? | P1 utility change $-116.55$. | No. The intuitive guard is net harmful. |
| Should a token-capped step always block stopping? | P3a utility change $-8,033.65$. | No. Token caps are common in both wins and losses. |
| Does a larger nonlinear correctness model automatically help? | On 75,965 trajectories, N2a is $-4,325.10$ versus matched logistic, mean $-0.05694$ [95% CI $-0.08716,-0.03145$]; N2b is $-4,682.95$, mean $-0.06165$ [$-0.09168,-0.03547$]. | No. Flexible models overfit sparse transition tails. |
| Does short trajectory history help? | N2c adds $+251.65$ matched utility, mean $+0.00331$ [95% CI $+0.00062,+0.00633$]. | Yes. Direction and rate of change contain signal absent from a snapshot. |
| Does partial pooling stabilize sparse hazards? | N3 adds $+593.55$, mean $+0.00781$ [95% CI $+0.00276,+0.01355$]; N3 plus churn reaches total dU $2,179.60$. | Yes. Variance control is the strongest Phase II improvement. |
| Does early churn reflect within-cell difficulty? | N4 adds $+159.50$, mean $+0.00210$ [95% CI $+0.00049,+0.00403$]. | Yes, within the audited corpus. |
| Does doubling 256 to 512 tokens fix the worst token-cap arm? | N5 has 454/1,500 losses and summed dU $-45.00$ in both paired arms; 0 discordant outcomes; loss-risk difference 0.0000 [0.0000, 0.0000]. | No measured causal benefit in that exact task-temperature-seed panel. |
| Does 4-bit quantization change early reasoning? | N6 has 418/1,500 BF16 correct versus 204/1,500 4-bit correct; gap $0.14267$ [95% CI $0.11133,0.17533$], $Z=9.79$. | Yes for Qwen2.5-7B/GSM8K; new-seed replication is needed before generalizing. |

Failed hypotheses are progress because they remove attractive but unsupported rules from the design space. Phase II narrows the credible mechanism to prefix history, calibrated cell/task context, and stabilized transition hazards rather than blanket guards or unconstrained model complexity.

## 5.8 Phase I/II progress checkpoint

The following gates distinguish completed research infrastructure from remaining deployment evidence.

| Gate | Evidence required | Current evidence | Status |
|---|---|---|---|
| P1. Common task and trace schema | Four domains load into one versioned row contract. | Implemented across 52 standardized cells. | Achieved |
| P2. Reproducible multi-model inference | Stable run identities, seeds, prompt contract, model identifiers, and restart behavior. | Implemented in the current worktree; exact external revisions, environment lock, and final Git tag remain archival items. | Achieved for the current worktree |
| P3. Correct grading and boundary semantics | Regression tests for numeric, symbolic, MCQ, and $T_{\min}$ behavior. | 30/30 grader and 6/6 boundary checks passed on July 30. | Achieved |
| P4. Formal decision objective | Utility, repair, corruption, cost, and first-crossing rule agree in theory and code. | Implemented in `trace_analysis.py`; utility convention documented. | Achieved |
| P5. Honest held-out evaluation | No task overlap between fit and outer test. | Task-grouped outer folds in the standardized tournament. | Achieved |
| P6. Controlled alternatives | Pre-registered single-variable tests with success bars. | P1/P2/P3, N1–N6, precision, token-cap, and stress artifacts. | Achieved |
| P7. Prefix-safe model comparison | Multiple causal sequence architectures evaluated on the same folds. | Nine standardized configurations; GRU AUC $0.874326$. | Achieved |
| P8. True in-generation stopping | Frozen policy terminates generation before unneeded steps are produced. | No live `StoppingCriteria` or equivalent controller. | Not achieved |
| P9. Locked external generalization | Unseen datasets and model families remain untouched until one final test. | Leave-domain/model analyses exist, but no locked external holdout. | Not achieved |

Seven of the nine defined Phase I/II research gates are achieved:

$$
\frac{7}{9}=77.8\%.
$$

This gate score is deliberately stricter than “code exists.” The two remaining gates are the decisive transition from retrospective analysis to a deployable and externally validated stopping policy.

# 6. Actionable Fall Semester Roadmap

## Phase 1: Advanced stress testing and generalization

**Target window:** August 3–August 28, 2026

### What Phase 1 means for a non-specialist reader

Phase 1 asks whether the detector has learned a reusable relationship or has merely adapted to the exact models, tasks, and prompt style already collected.

| Term | Meaning in this thesis | Failure it is designed to reveal |
|---|---|---|
| Stress test | Change one controlled property of an otherwise equivalent prompt or model input. | The detector may rely on formatting, answer position, verbosity, or clean wording instead of reasoning state. |
| Generalization | Preserve performance on tasks, domains, or model families excluded from training. | A pooled score can hide failure on a specific domain such as GPQA. |
| Out-of-distribution (OOD) | Evaluate on data generated by a source or reasoning pattern outside the training distribution. | Cross-validation within the same four datasets may overestimate external performance. |
| Reasoning trap | A prompt constructed to induce anchoring, premature consensus, contradiction, or a plausible wrong intermediate answer. | Confidence and answer stability may be caused by the trap rather than correctness. |
| Online stopping | Terminate generation before unneeded future steps are generated. | Offline replay can report token savings without proving wall-clock or GPU savings. |

The current evidence entering Phase 1 is:

| Workstream | Evidence already available | Remaining gap |
|---|---|---|
| Frozen standardized corpus | 52 files with SHA-256 values in `ultimate_tournament_manifest.json`; 28,888 source-qualified trajectories; zero recorded cross-cell run-ID collisions. | GPQA split metadata, Qwen3.5 64-task GSM8K cell, package lock, model/dataset revisions, and final tag are unresolved. |
| Boundary measurement | The 75,965-trajectory matrix now has task-cluster intervals for step accuracy, repair, corruption, and drift. GSM8K crosses from significantly positive drift at step 3 to significantly negative drift at step 4; selected Qwen GSM8K/MATH cells show later crossings. | Intervals cover tasks in the fixed model/seed panel; multi-seed and external-model boundary replication remain open. |
| Balanced tournament evaluation | Micro, task-macro, domain-macro, worst-domain, per-domain, and five-fold intervals are materialized. GRU, RoPE, and SSM remain as pre-declared candidates for different safety/utility objectives. | Leading fold intervals overlap; a locked external evaluation must choose among the candidates without additional tuning. |
| Cross-domain evaluation | Leave-one-domain-out AUCs: ARC 0.968836, GPQA 0.700305, GSM8K 0.927793, MATH 0.868882. | GPQA is below the proposed 0.75 gate; no locked external dataset has been evaluated. |
| Perturbation testing | Precision, token budget, peer corruption, roster size, and panel permutation have been tested. | No controlled prompt-noise, paraphrase, option-order, contradiction, or anchoring suite exists. |
| Stopping simulation | Offline replay records 54.34% fewer completion tokens. | Accuracy falls 6.33 percentage points; the script fits and evaluates on the same file; generation is not actually terminated. |
| Regression protection | Grader 30/30, boundary floor 6/6, and synthetic forced-choice protocol pass. | These validate mechanics, not generalization or live compute saving. |

### 6.1 Freeze and repair the evidence base

Deliverables:

1. Create a read-only `data_manifest_v1.json` with SHA-256 hashes for every canonical trace and result file.
2. Correct GPQA split metadata from `test` to effective `train`.
3. Reconcile or delete stale 28,928/192,640 prose and log references.
4. Rerun or explicitly exclude the 64-task Qwen3.5 GSM8K cell.
5. Produce one source-qualified trajectory key, `source_cell::run_id`, in every downstream artifact.
6. Record package lock, CUDA/PyTorch/Transformers versions, model revisions, dataset revisions, and hardware.
7. Tag the frozen state only after a clean-clone reproduction.

Acceptance criteria:

- 52/52 standardized cells pass schema validation;
- 28,888 current trajectories or a new explicitly versioned count match every table;
- zero raw run-ID collisions after source qualification;
- zero task overlap between outer train and test;
- a clean clone regenerates all summary JSON/CSV files from the frozen traces.

### 6.2 Prompt-noise and reasoning-trap suite

Add five pre-registered perturbation families:

1. irrelevant-sentence insertion at 0%, 10%, 20%, and 40% prompt length;
2. contradictory but explicitly non-authoritative distractors;
3. paraphrase and option-order perturbations;
4. answer-anchoring traps that present a plausible incorrect intermediate result;
5. instruction-format perturbations that preserve semantics but change response syntax.

Use at least 250 locked tasks per available domain, three seeds, and two temperatures $(0.1,0.6)$ for the selected 4–6 model roster. Report:

$$
\Delta\operatorname{Acc},\quad
\Delta U,\quad
\Delta\operatorname{AUC},\quad
\Delta\operatorname{ECE},\quad
\Pr(\text{harmful early stop}).
$$

Acceptance criterion: the selected policy must remain within 2.0 percentage points of clean never-stop accuracy and retain positive mean utility in every perturbation family.

### 6.3 OOD and leave-family-out evaluation

Lock one unseen dataset per reasoning type:

- arithmetic word problems;
- symbolic algebra;
- logical deduction;
- adversarial multiple choice.

Perform:

- leave-one-dataset-out;
- leave-one-model-family-out;
- leave-one-size-band-out;
- temporal holdout by collection batch.

Acceptance criterion: no headline result may rely only on pooled micro-AUC. Report task-macro, domain-macro, and worst-domain metrics. GPQA or its replacement scientific-reasoning holdout must exceed a pre-registered minimum AUC of 0.75 or be explicitly identified as an unresolved failure.

### 6.4 Implement a real online stopping hook

Implement a Hugging Face `StoppingCriteria` or explicit one-step generation controller that:

1. generates only the next reasoning increment;
2. computes strictly prefix-available features;
3. evaluates the frozen stopping model;
4. terminates generation before future steps are produced;
5. measures policy latency, GPU utilization, KV-cache effects, and real wall-clock savings.

Acceptance criteria:

- no feature at time $t$ reads step $t+1$ or final-trajectory information;
- stopped outputs are byte-identical to the corresponding prefix of never-stop outputs under the same seed;
- actual generated-token and wall-clock reductions agree with replay estimates within a pre-registered tolerance;
- latency is measured, not asserted.

### Phase 1 execution order and exit package

The critical dependency chain is:

$$
\text{evidence freeze}
\rightarrow
\text{online controller}
\rightarrow
\text{stress and OOD evaluation}
\rightarrow
\text{locked Phase 1 report}.
$$

Prompt perturbations and external-dataset adapters can be implemented while the online controller is being built, but no final metric should be produced until the corpus, fold assignments, and policy artifact are frozen.

| Work package | Required input | Concrete output | Proof of completion |
|---|---|---|---|
| A. Evidence freeze | Current 52-cell traces and result artifacts. | `data_manifest_v1.json`, environment lock, revision ledger, corrected GPQA metadata, explicit Qwen3.5 inclusion decision. | Clean-clone reproduction and hash match for every summary input. |
| B. Online controller | One frozen prefix-safe model and preprocessing pipeline. | `online_stopping_controller.py`, causal feature contract, deterministic paired-run harness. | No future-step reads; identical generated prefix under matched seed; real token and time measurements. |
| C. Prompt stress suite | Locked clean tasks and perturbation generator. | Perturbed task manifest, per-family paired results, failure examples. | All five perturbation families, three seeds, and pre-registered accuracy/utility gates. |
| D. External generalization | Datasets and model families never used during fitting. | LODO/LOMO/OOD report with task-, domain-, and worst-group metrics. | No overlap with fit/tuning data; GPQA/science gate resolved explicitly. |
| E. Phase 1 decision report | Outputs A–D. | Advisor-facing report selecting “advance,” “revise,” or “do not deploy.” | Claim table ties every conclusion to a frozen artifact and confidence interval. |

The Phase 1 advisor checkpoint should approve four choices before final execution:

1. the maximum allowed accuracy loss $\epsilon$;
2. whether GPQA AUC 0.75 is a hard gate or a documented failure threshold;
3. the 4–6 model stress-test roster; and
4. the external datasets that will remain untouched until the final test.

## Phase 2: Systematic hyperparameter optimization

**Target window:** August 31–September 25, 2026

### What Phase 2 means for a non-specialist reader

A model *parameter* is learned from training data. A *hyperparameter* is a design choice fixed outside that training procedure: the step cost, stopping threshold, history length, model width, dropout rate, or calibration strength. Hyperparameter optimization asks which settings work best without using the final test set to make that choice.

Four concepts govern the Phase 2 design:

| Concept | Accessible explanation | Role in the protocol |
|---|---|---|
| Nested cross-validation | Use inner folds to select hyperparameters and a separate outer fold to estimate final performance. | Prevents reporting the same data used to choose the settings. |
| Bayesian optimization | Use results from completed trials to choose promising new combinations instead of exhaustively evaluating every combination. | Makes the 12-dimensional search computationally feasible. |
| Constraint | A rule the selected policy must satisfy, not merely a metric to maximize. | Prevents token savings from being purchased with unacceptable accuracy loss or miscalibration. |
| Pareto frontier | The set of policies for which one objective cannot improve without another becoming worse. | Makes the accuracy/utility/speed trade-off visible rather than collapsing it into AUC. |

Phase II results already narrow the search:

- the utility conclusion is stable for $\lambda\in[0.02,0.10]$, while $\lambda=0.20$ is cost-dominated and nearly degenerate;
- N2a and N2b show that greater model flexibility can reduce utility through sparse-tail overfitting;
- N2c, N3, and N4 support temporal lags, empirical-Bayes shrinkage, and early-churn modulation;
- the standardized GRU has the highest prefix-safe AUC and fewest losses, but the RoPE transformer has higher step utility;
- GPQA remains the weakest domain and must influence model selection rather than being hidden by pooled AUC.

### 6.5 Multi-objective search

Use nested task-grouped cross-validation. Outer folds remain untouched until final evaluation. Search on inner folds:

| Parameter | Search range | Why it matters |
|---|---|---|
| Step cost $\lambda$ | log-uniform $[0.005,0.20]$ | Converts additional reasoning into an explicit penalty; controls how early stopping is rewarded. |
| Drift offset $\delta$ | uniform $[-0.25,0.25]$ | Moves the continuation boundary earlier or later after model/dataset calibration. |
| Entropy quantile | uniform $[0.10,0.90]$ | Sets what counts as unusually concentrated token uncertainty using training data. |
| Minimum answer streak | integer $\{1,2,3,4\}$ | Requires an answer to persist before a stability-based stop is allowed. |
| Confidence floor | integer $[40,95]$ | Prevents stopping on low self-reported confidence; it cannot be trusted without calibration. |
| Hysteresis lower/upper | lower $[0.05,0.45]$, upper $[0.55,0.95]$ | Creates separate enter/exit thresholds so a noisy score does not repeatedly switch decisions. |
| Temperature scaling | log-uniform $[0.25,4.0]$ | Recalibrates predicted probabilities without changing their ranking. |
| EB confidence $\delta_{\mathrm{EB}}$ | log-uniform $[10^{-4},0.10]$ | Controls the conservativeness of the empirical-Bernstein population bound. |
| Empirical-Bayes shrinkage strength | log-uniform $[10^{-2},10^3]$ | Determines how strongly sparse cell hazards are pulled toward global step rates. |
| GRU/TCN hidden width | $\{32,64,128,256\}$ | Controls sequence-model capacity and therefore the bias/overfitting trade-off. |
| Dropout | uniform $[0,0.30]$ | Randomly masks training activations to reduce co-adaptation and overfitting. |
| Learning rate | log-uniform $[10^{-5},3\times10^{-3}]$ | Controls the size of parameter updates and training stability. |

Optimize the constrained objective

$$
\max_\theta
\mathbb{E}[U_{\tau_\theta}]
$$

subject to

$$
\operatorname{Acc}(\tau_\theta)
\ge
\operatorname{Acc}(T)-\epsilon,
$$

$$
\Pr(\Delta U_i<0)\le\eta,
$$

and

$$
\operatorname{ECE}\le\kappa.
$$

Recommended initial constraints are $\epsilon=0.02$, $\eta=0.08$, and $\kappa=0.05$, followed by advisor approval.

Run at least 200 Bayesian trials per outer fold for the selected prefix-safe architecture, with three initialization seeds for the top ten configurations. Persist every trial, fold assignment, seed, failure, and selected parameter set.

With five outer folds, the minimum planned workload is

$$
5\times200=1{,}000
$$

inner-search trials, followed by at least

$$
10\times3=30
$$

seed replications for the ten leading configurations. A failed or timed-out trial remains part of the ledger; deleting failures would bias the apparent reliability of the search.

### 6.6 Final model selection

Select from a Pareto frontier over:

- stopped accuracy;
- mean step utility;
- mean token utility;
- loss rate;
- P95 regret;
- calibration;
- real wall-clock savings.

Do not select by AUC alone.

Acceptance criterion: one frozen policy, one conservative policy, and one never-stop control, each with a serialized preprocessing pipeline and threshold.

### Phase 2 exit package and advisor decision

| Deliverable | Required contents | Acceptance gate |
|---|---|---|
| Search ledger | Trial ID, outer/inner fold, task groups, seed, full parameter vector, runtime, status, metrics, artifact hash. | At least 1,000 attempted inner trials; zero final-test access during search. |
| Calibration package | Frozen scaler, feature order, probability calibrator, threshold, and software versions. | Re-loading the package reproduces stored probabilities within numerical tolerance. |
| Outer-fold report | Accuracy, AUC, Brier, ECE, step/token utility, loss rate, P95 regret, generated tokens, and wall-clock time. | Every metric is produced once per untouched outer fold and then aggregated. |
| Pareto analysis | Policy points with uncertainty intervals and dominated/non-dominated labels. | Selected policies are non-dominated under the approved objectives. |
| Selection memo | Rationale for frozen, conservative, and never-stop policies; rejected alternatives and failure modes. | Advisor approves the policy taken into theoretical analysis and final testing. |

Phase 2 exits only when the selected policy satisfies the accuracy, loss-rate, and calibration constraints on every outer fold or when an exception is documented as a thesis limitation. A high pooled AUC alone is not an exit criterion.

## Phase 3: Formal theoretical and mathematical proofs

**Target window:** September 28–October 23, 2026

### 6.7 Standardize the mathematical object

Use one utility convention:

$$
U_t=q_t(v+c)-c-\lambda(t-1).
$$

State the probability space, filtration, horizon, measurability, boundedness, and admissible stopping-time class.

### 6.8 Convert hypotheses into theorem–assumption–test units

Theorem 1 should state:

> If $q_t$ is nondecreasing, $\alpha_t$ is nonincreasing, and $\beta_t$ is nondecreasing almost surely, then $\mu_t$ is nonincreasing and crosses zero at most once.

Then add empirical diagnostics:

$$
\widehat p_q
=\frac{1}{N}\sum_i
\mathbf{1}\{\widehat q_{i,t+1}\ge\widehat q_{i,t}\ \forall t\},
$$

with analogous $\widehat p_\alpha$ and $\widehat p_\beta$. Report violations by model and domain.

### 6.9 Prove finite-horizon OSLA optimality

Provide a self-contained proof for bounded horizon $T$:

$$
\tau^*=\inf\{t\ge2:\mu_t\le0\}\wedge T.
$$

Explicitly distinguish:

- optimality under known true hazards;
- regret under estimated hazards;
- effects of calibration error.

Derive a perturbation bound. If

$$
|\widehat q_t-q_t|\le\epsilon_q,\quad
|\widehat\alpha_t-\alpha_t|\le\epsilon_\alpha,\quad
|\widehat\beta_t-\beta_t|\le\epsilon_\beta,
$$

then a first-order uniform drift error bound is

$$
|\widehat\mu_t-\mu_t|
\le
(v+c)
\left(
\epsilon_\alpha+\epsilon_\beta
+2\epsilon_q
\right),
$$

using probabilities bounded by one. Tighten this bound and connect it to boundary displacement under a margin condition

$$
|\mu_t|\ge\gamma
$$

away from the true crossing.

### 6.10 Sequential safety

State precisely whether each bound applies:

- across prompts at a fixed step;
- uniformly across steps;
- to a population mean;
- or to an individual trajectory.

Validate empirical coverage by simulation and held-out traces. Any theorem that assumes independent prompts must exclude repeated temperatures or model outputs on the same raw task unless clustered.

Acceptance criteria:

- every theorem has assumptions, proof, and an empirical assumption check;
- no online-safety statement is inferred from an AUC result;
- notation is identical in theory, code comments, plots, and manuscript.

## Phase 4: Manuscript compilation and defense readiness

**Target window:** October 26–December 7, 2026

### 6.11 Manuscript build

Create a version-controlled LaTeX manuscript with:

1. Introduction and research question;
2. Related work;
3. Stochastic optimal-stopping framework;
4. Data and experimental protocol;
5. Detector architectures;
6. Results;
7. Falsification and failure analysis;
8. Theoretical results;
9. Limitations, ethics, and reproducibility;
10. Conclusion.

The October 23 version-1 target in the PDF should be retained by starting chapter integration before Phase 3 ends.

### 6.12 Publication figures

For each final figure:

- generate from a script and frozen CSV;
- export vector PDF/SVG when possible and 600-DPI PNG fallback;
- include units, sample size, confidence interval, and corpus version;
- use consistent model aliases and color mapping;
- include a caption that states the fold unit and whether the metric is micro- or macro-averaged.

Required figures:

1. repair and corruption hazards versus step;
2. drift crossing with uncertainty;
3. accuracy and utility versus stop step;
4. Pareto frontier of token saving versus accuracy loss;
5. per-domain and per-model generalization heat map;
6. calibration plots;
7. loss-severity distribution;
8. online wall-clock and token savings.

### 6.13 Advisor review and defense

Maintain a feedback ledger with:

- reviewer;
- date;
- section/figure;
- requested change;
- disposition;
- commit hash resolving the comment.

Schedule:

- November 1–10: advisor and reader revision cycle;
- November 11–20: at least two recorded mock defenses;
- November 23–30: formal defense;
- December 1–3: ETD polish;
- December 4–5: ETD submission;
- December 6–7: sign-off.

The defense package must contain:

- a 30-minute talk;
- a 30-minute question reserve;
- a one-page theorem/notation sheet;
- a reproducibility appendix;
- a limitations slide that includes the retrospective-versus-online distinction.

### 6.14 Archival acceptance criteria

Before the final tag:

- working tree clean;
- no transient `.sqlite3-wal`, `.sqlite3-shm`, or incomplete checkpoints tracked;
- all large artifacts documented;
- environment lock committed;
- data and model licenses listed;
- all final table numbers generated by one command;
- manuscript and defense cite the same frozen result version;
- Git tag and release notes include hashes for traces, results, manuscript, and slides.

# Appendix A. Repository source map

| Research function | Primary implementation |
|---|---|
| Model catalog | `research/real_trace_experiments.py:337` |
| Answer normalization and grading | `research/real_trace_experiments.py:713` |
| Utility | `research/real_trace_experiments.py:752` |
| Incremental prompt | `research/real_trace_experiments.py:764` |
| GSM8K loader | `research/real_trace_experiments.py:992` |
| MATH loader | `research/real_trace_experiments.py:1039` |
| ARC loader | `research/real_trace_experiments.py:1098` |
| GPQA loader | `research/real_trace_experiments.py:1139` |
| Model/quantization loading | `research/real_trace_experiments.py:1491` |
| Generation and diagnostic scoring | `research/real_trace_experiments.py:1759` |
| Per-step trace assembly | `research/real_trace_experiments.py:2395` |
| Baseline feature vector | `research/trace_analysis.py:26` |
| Temporal features | `research/trace_analysis.py:49` |
| Oracle stop | `research/trace_analysis.py:107` |
| Hazard stop | `research/trace_analysis.py:170` |
| OOF detector evaluation | `research/trace_analysis.py:469` |
| Offline active-stop replay | `research/run_active_stopping.py:105` |
| Offline active-stop result | `research/outputs/real_traces_bf16_ladder/active_stopping_simulation_report.md` |
| Boundary-floor regression | `research/tests/test_boundary_floor.py` |
| Grader regression | `research/tests/test_graders.py` |
| Synthetic forced-choice protocol | `research/test_v2_forced_choice_protocol.py` |
| Prefix-safe multi-day tournament | `research/run_ultimate_multi_day_tournament.py` |
| Retrospective Blackwell stack | `research/run_ultimate_blackwell_5day_tournament.py` |
| Canonical matrix manifest | `research/outputs/experiment_matrix/matrix_manifest.json` |
| Standardized corpus manifest | `research/outputs/experiments_v2/ultimate_tournament_manifest.json` |
| Frozen tournament OOF probabilities | `research/outputs/experiments_v2/ultimate_oof_predictions.npz` |
| Phase II trajectory examples | `research/outputs/experiments_v2/global_qwen2p5_7b_gsm8k/trace_steps.csv` |
| Prefix-safe tournament results | `research/outputs/experiments_v2/ultimate_tournament_results.csv` |
| Review-metric recomputation | `tools/compute_progress_report_review_metrics.py` |
| Boundary estimates and task-cluster intervals | `research/outputs/progress_report_review_metrics/boundary_domain_step_metrics.csv` |
| N1-N6 denominators and normalized effects | `research/outputs/progress_report_review_metrics/algorithm_v2_normalized_effects.csv` |
| Macro, per-domain, and fold tournament metrics | `research/outputs/progress_report_review_metrics/tournament_balanced_summary.csv` |
| Blackwell final report | `research/outputs/experiments_v2/blackwell_5day_tournament_v1/blackwell_tournament_report.json` |
| Peer stress results | `research/outputs/experiments_v2/peer_dynamics_stress_tests_v1/peer_dynamics_stress_test_report.json` |
| Rigor audit | `ThesisDocs/rigor_audit/` |
| Mathematical proof draft | `ThesisDocs/archive/mathematical_foundations_proofs.md` |

# Appendix B. Reproducibility and reporting corrections

The advisor-facing manuscript should apply the following corrections before using any result:

1. Replace 28,928 trajectories and 192,640 steps with the versioned corpus actually analyzed.
2. Explain the 13/14/15 model-count lineage.
3. Replace 97.17% with a result tied to a machine-readable artifact; current final value is 95.5156% for the retrospective stack.
4. Scope the 60.2% loss reduction to the 4,500-trajectory Tier-3 corpus.
5. Describe active stopping as offline replay until a true generation hook is implemented.
6. State the 6.33-percentage-point accuracy loss beside the 54.34% token saving.
7. Label `thought_token_count` as a regex thought-word count and recompute token utility from `raw_generation_tokens`; do not reuse the historical tournament token column.
8. Correct GPQA effective split metadata.
9. Use task-grouped, not run-grouped, final evaluation.
10. Report micro, task-macro, domain-macro, worst-domain, and fold-level uncertainty before selecting a model.
11. Distinguish prefix-safe sequence models from full-trajectory retrospective detectors.
12. State that the enrollment PDF contains no advisor feedback notes.
13. Report step-wise accuracy, repair and corruption denominators, net drift, and task-cluster uncertainty when claiming a stopping boundary.
14. Label every N1-N6 value as a total, mean, count, probability, or percentage-point effect and state its experimental unit.

# Appendix C. Advisor approval checklist before archival

The report is suitable as an evidence-qualified progress submission. The following items remain explicit conditions for a final archival release; none is presented as silently completed:

| Priority | Condition | Evidence in the current worktree | Required action before the final thesis tag |
|---|---|---|---|
| P0 | Freeze the effective GPQA split. | `research/real_trace_experiments.py:1139` currently loads the `train` split while the metadata label says `test`. | Decide the authoritative split, regenerate affected traces and metrics, and record the dataset revision and hash. |
| P0 | Freeze one authoritative result bundle. | The JSON/CSV artifacts under `research/outputs/experiments_v2/` and `progress_report_review_metrics/` are the current numeric sources; older prose reports can contain stale intervals. | Generate every manuscript table from one command, archive its manifest and hashes, and label superseded prose as non-authoritative. |
| P1 | Make the 100-instance ultra sweep auditable. | The code specifies five folds, ten seeds, two probe families, and 1,000 epochs, but the saved Blackwell JSON has no per-seed/per-fold ledger. | Persist a trial ledger with seed, fold, epoch count, split hashes, checkpoint path, metrics, and failure status for every attempted instance. |
| P1 | Demonstrate prospective stopping. | `run_active_stopping.py` replays stored trajectories; it does not terminate a live `model.generate()` call. | Implement a prefix-safe generation hook and report prospective accuracy, latency, raw generated tokens, and energy or GPU-time measurements. |
| P1 | Test external validity. | Current intervals quantify task clusters in the fixed panel; they do not cover new model families, sampling seeds, noisy prompts, or a locked external dataset. | Add multi-seed replication, perturbation/OOD suites, and a held-out dataset whose task identities never enter fitting or threshold selection. |
| P1 | Validate the correctness oracle independently. | The 30/30 grader regression and targeted extraction audit validate parser mechanics, not every benchmark answer by human adjudication. | Obtain a documented independent adjudication sample and report agreement, disagreements, and adjudication rules. |
| P1 | Complete release reproducibility. | The report lists environment, licensing, and Git-tag requirements, but the final release tag and environment lock are not yet frozen. | Commit the lockfile, licenses, trace/result manifests, manuscript source, figures, and release notes under one immutable tag. |

An advisor may approve the present report while recording these items as fall-semester work. Approval of the report should not be interpreted as approval of the unresolved GPQA split, a live stopping claim, or a universal detector optimum.
