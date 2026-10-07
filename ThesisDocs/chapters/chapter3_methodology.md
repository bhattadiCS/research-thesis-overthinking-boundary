# Chapter 3 Experimental setup and methods

## 3.1 Corpus separation

The variable-horizon response-and-revision matrix supports boundary and controlled-policy analyses: 75,965 sanitized trajectories and 798,770 raw saved step rows over 52 model-domain cells. Sanitization handles malformed fragments and inconsistent records, so raw row and eligible trajectory counts have different denominators. The standardized five-step detector corpus supports training and task-grouped scoring: 28,888 trajectories, 144,440 rows, and 2,948 unique task identifiers across 52 cells.

The collections overlap in scientific subject matter and include repeated benchmark questions. They are not independent replications and must not be summed into a single sample size. The earlier replay experiment also reuses a 1,500-trajectory Qwen2.5-7B/GSM8K cell from existing evidence. Its replayed decisions do not create new language-model generations.

[[CORPUS_TABLE]]

Different benchmark allocations explain the standardized corpus's larger task count. Counts use source-qualified trajectory keys; unequal model-domain sample sizes prevent deriving row totals from unique questions multiplied by models and steps.

## 3.2 Model roster

The canonical matrix includes thirteen model configurations: DeepSeek-R1-Distill-Qwen-1.5B and 7B; Qwen2.5-0.5B, 3B, 7B, 14B, and 32B Instruct [Qwen2024]; InternLM3-8B-Instruct; Llama-3.1-8B-Instruct; Mistral-7B-Instruct-v0.3; Mistral-Small-Instruct-2409; Phi-4-mini-instruct; and Yi-1.5-9B-Chat. The standardized detector collection replaces InternLM3 with Qwen3.5-9B. These are distinct rosters, even though both have thirteen entries.

[[MODEL_TABLE]]

The historical alias `mistral_small_24b_2409` denotes the 2409 model with a recorded specification of 22B. Tables use documented specifications; aliases locate source files. Saved manifests and trace metadata determine model inclusion.

The panel supports descriptive family and scale comparisons. Architecture, pretraining data, instruction tuning, distillation, and tokenization were not randomized, so family differences do not isolate parameter count. A common Qwen family reduces some confounding without constituting a controlled intervention on scale alone.

## 3.3 Benchmark tasks and splits

GSM8K provides grade-school mathematical word problems [Cobbe2021]; MATH provides competition-style mathematics [Hendrycks2021]. ARC-Challenge contains difficult multiple-choice science questions [Clark2018], and GPQA contains graduate-level multiple-choice scientific questions [Rein2023]. The following protocol specifies this project's subsets and grading.

The canonical registry specifies GSM8K train, MATH test, ARC-Challenge test, and GPQA main train. The standardized collection uses the same first three splits. Its GPQA metadata records `test`, while the current loader always requests `train` and saved identifiers use `gpqa_main`. The historical loader revision is not recorded, so the effective split is inferred from the current implementation and stored identities; both requested and inferred splits are reported. Detector folds are distinct from benchmark split names: a task holdout drawn from benchmark training data is internal validation, not an official test-set evaluation.

SVAMP does not appear in the authoritative standardized tournament manifest. It is excluded from the reported four-domain totals. The complete prompt, task identifier, reference answer, split, and any deterministic choice shuffle must remain associated with a question. An MCQ label such as B only has meaning relative to the exact displayed ordering. The project stores or reconstructs that ordering before evaluating the option label.

## 3.4 Response and revision protocol

A saved step is a complete response generated under the collection's prompt contract. For the canonical registry, seed 7 and dataset shuffle seed 17 are recorded. The temperature panel uses 0.1, 0.6, and 1.0, and the common completion-token cap is 256. Response horizons are domain-specific: ten increments for GSM8K and GPQA, fourteen for MATH, and eight for ARC. Individual metadata remains authoritative when a cell records a deviation or recovery event.

The standardized detector corpus has five saved increments per trajectory. Its generation settings include temperature 0.6 and seed 7. A repeated call with a revision prompt is not equivalent to revealing another fragment of one uninterrupted chain of thought. The revision prompt itself can change the distribution of the next answer. The stopping target is therefore defined relative to this complete sequential protocol, including prompt construction and sampling.

The minimum admissible stopping step is two in the principal experiments. This floor is a design choice informed by development analyses. It is not a theorem that every question requires a revision. Some trajectories are correct only at step one, and the floor can exclude their best candidate. Allowing an answer-selection policy to return a previously saved candidate would define a different action space; such a policy must be compared separately.

At every decision, computation already incurred is sunk. The continuation decision should charge the incremental cost of future generation, not charge the prefix again. At evaluation, however, total policy cost includes every generated response up to the selected step. The distinction keeps the Bellman recursion and empirical cost accounting consistent.

## 3.5 Recorded signals and grading

Recorded signals can include candidate text, normalized answer, correctness, completion-token count, generation runtime, answer changes, log-probability summaries, entropy, hidden-state displacement, and peer agreement. Availability varies across experiments. A missing signal should be recorded as missing or excluded by a documented rule; treating it as a meaningful zero can alter a learned boundary.

Correctness labels use the benchmark's answer type. Numeric tasks use extraction and normalization; symbolic MATH tasks use the project's mathematical equivalence checks; multiple-choice tasks compare the parsed option with the exact displayed reference. Normalization must preserve information that matters to equivalence. For example, reducing an arbitrary plane equation to its right-hand side would identify distinct mathematical objects, and treating the phrase 'the third option' as the fraction one third would corrupt an MCQ answer.

The historical grader receipt covers thirty specific cases. Following the software review, the revised grader exposes ninety-eight tests covering normalization, mathematical equivalence, ambiguous option labels and bounded symbolic parsing. The receipts identify the tested source versions, and finite-system stopping tests separately check the minimum boundary and termination behavior. Regression coverage concerns these specified behaviors; validity across the full answer corpus requires independent adjudication.

The primary boundary tables use the archived `correct` values. The predictor training in Section 5.7 uses a separately versioned reconstruction and regrade of its selected candidates. A subsequent comparison of the historical and revised graders found no changes to the final-answer labels in the four live collections. These checks leave the original labels and results intact; a corpus-wide regrade would define a new analysis version with its own label changes and dependent estimates.

## 3.6 Statistical units and grouping

Rows from one trajectory are dependent, and trajectories from the same task share the question and reference. Task-held-out detector folds keep all rows for a question outside the detector's training data. A source-qualified key combines the cell with its run identifier to prevent accidental merging of trajectories from different model-domain cells. Checking raw identifier collisions is useful, but source qualification remains the defined identity rule.

The strongest stored task-grouped detector analyses use five outer folds. For meta-model stacking, every upstream fitted component must also respect the outer split: its parameters and any calibrated thresholds must be fit using outer-training data alone. Producing a base score out of fold somewhere in the development pipeline does not automatically make a later outer-fold stacked result nested. Chapter 4 explains why the historical 0.955156 result falls short of that standard.

Cluster bootstraps resample questions when the same question appears in several trajectories. Model-domain estimator comparisons use the recorded cell bootstrap for their reported interval. These intervals answer different questions. A question bootstrap estimates variability associated with the observed question population conditional on the fixed model panel; a cell bootstrap summarizes variability across the existing panel. Neither one is a multi-seed generation replication.

The archived N2/N3 probe and hazard harness uses run-group folds for upstream prediction, followed by task-group folds for threshold selection. Other-temperature trajectories of a question can enter upstream fitting, so the full pipeline is a controlled development contrast rather than an untouched-question evaluation. The strict tabular and text analyses and the new prefix predictor follow separately recorded task-disjoint contracts. Appendix D details retrospective prediction analyses; Section 5.7 specifies the predictor protocol. Appendix E illustrates offline fitting and runtime information flow.

## 3.7 Outcomes and computation measures

Accuracy is the mean binary correctness label at the policy's selected answer. Historical per-trajectory step utility is $U_i(\tau_i)=C_{i,\tau_i}-0.05(\tau_i-1)$, so the mandatory first response is a common baseline and each additional increment incurs the penalty. This penalty is a utility convention, not five percent of physical power consumption or a universal economic price. Token utility is $C_{i,\tau_i}-0.0002\sum_{s=1}^{\tau_i}L_{i,s}$, where $L_{i,s}$ is measured completion tokens; 250 completion tokens carry the same penalty as one response increment. Charging every generated token includes the first response, which cancels in paired policy differences on the same trace. The theoretical reward in Chapter 2 can subtract the common prefix cost without changing the optimal continuation decision.

Token savings are one minus the ratio of total stopped-policy completion tokens to total full-horizon completion tokens, calculated on a common task panel. The ratio of totals differs from the average per-question saving and weights long generations more heavily. Both the definition and denominator must accompany a percentage. Prompt tokens, scoring passes, peer generations, controller inference, and runtime may have separate costs and cannot be omitted from a claim about total compute savings.

ROC-AUC measures ranking, with half credit for tied scores. Brier loss measures squared error of probabilistic predictions. Neither alone establishes a stopping guarantee. Class-balanced raw probability outputs generally target a reweighted posterior, as derived in Chapter 2; their held-out ranking or marginal calibration does not validate natural-distribution conditional hazards. A detector with high pooled AUC can be poorly calibrated around the selected stopping threshold or fail on an underrepresented domain. The evaluation therefore includes micro, task-macro, domain-macro, worst-domain, and policy utility summaries.

## 3.8 Reproducibility and evidence freeze

Evidence manifests record exact local byte hashes and canonical LF content hashes, cross-check historical fingerprints, and identify supporting analyses. Verification also checks file sizes, row counts, and source-qualified trajectory membership. Appendix A maps the evidence and locates the electronic verification and reanalysis guide.

The original profile, `data_manifest_v1.json`, binds revision `09225c95`; `data_manifest_post_review_v1.json` binds grader, controller, and audit revisions at `6e4378be`. Both preserve the fifty-two standardized trace files, fitted predictor, and recorded generation outcomes. Each live-generation manifest binds the source copies executed for that run. Post-review replay checks revised decisions on saved prefixes; it produces no new generations or timing measurements.

Stored Blackwell metadata records a CUDA 13.0 runtime and PyTorch 2.13.0+cu130; the workstation lock describes a different, current environment. Compiler libraries, hardware, drivers, random-state handling, and nondeterministic kernels also affect repetition. The freeze supports inspection and re-analysis of stored results; bit-identical regeneration is not established.
