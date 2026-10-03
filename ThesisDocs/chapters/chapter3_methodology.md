# Chapter 3 Experimental setup and methods

## 3.1 Corpus separation

The project contains two main collections with different purposes. The canonical model-domain matrix is a variable-horizon response-and-revision collection used for boundary and controlled policy analyses. Its current analysis describes 75,965 sanitized trajectories and 798,770 raw saved step rows over 52 model-domain cells. The word sanitized matters: raw malformed fragments and inconsistent records are handled by the analysis pipeline, so a raw row count and an eligible trajectory count are different denominators. The standardized detector collection contains 144,440 rows from 28,888 five-step trajectories and 2,948 unique task identifiers. Its 52 cells support detector training and task-grouped scoring.

The collections overlap in scientific subject matter and include repeated benchmark questions. They are not independent replications and must not be summed into a single sample size. The earlier replay experiment also reuses a 1,500-trajectory Qwen2.5-7B/GSM8K cell from existing evidence. Its replayed decisions do not create new language-model generations.

[[CORPUS_TABLE]]

The standardized task count exceeds the canonical matrix's task count because the collection uses different benchmark sample allocations. Each standardized trajectory has five rows, so 28,888 multiplied by five gives exactly 144,440. Multiplying the unique task count by thirteen and five would be incorrect because model-domain cells have unequal sample sizes. Counts are derived from source-qualified trajectory keys, not from that shortcut.

## 3.2 Model roster

The canonical matrix includes thirteen model configurations: DeepSeek-R1-Distill-Qwen-1.5B and 7B; Qwen2.5-0.5B, 3B, 7B, 14B, and 32B Instruct; InternLM3-8B-Instruct; Llama-3.1-8B-Instruct; Mistral-7B-Instruct-v0.3; Mistral-Small-Instruct-2409; Phi-4-mini-instruct; and Yi-1.5-9B-Chat. The standardized detector collection replaces InternLM3 with Qwen3.5-9B. These are distinct rosters, even though both have thirteen entries.

[[MODEL_TABLE]]

The historical alias `mistral_small_24b_2409` identifies the 2409 model, whose recorded specification is 22B. An alias is not a reliable parameter count. Scientific tables use the documented model specification and retain the alias only where needed to locate source files. Likewise, a speculative entry in a software catalog is not evidence that a model was included in a completed experiment; the actual manifests and trace metadata determine inclusion.

The model panel allows descriptive comparisons of families and scale. It does not randomize architecture, pretraining data, instruction tuning, distillation, or tokenization. A difference between two model families therefore cannot be attributed uniquely to scale. Within the Qwen ladder, common family identity reduces some confounding, but still does not create a controlled intervention on parameter count alone.

## 3.3 Benchmark tasks and splits

GSM8K contains grade-school mathematical word problems and was introduced in the verifier study of Cobbe and colleagues [Cobbe2021]. MATH provides competition-style mathematical problems [Hendrycks2021]. ARC-Challenge provides difficult multiple-choice science questions [Clark2018]. GPQA provides graduate-level multiple-choice scientific questions [Rein2023]. These references identify the benchmark tasks; they do not validate this project's chosen subsets or grading implementation.

The canonical experiment registry specifies GSM8K train, MATH test, ARC-Challenge test, and GPQA main train. The standardized collection also uses GSM8K train, MATH test, and ARC-Challenge test. Its GPQA metadata records `test`, but the current loader unconditionally requests the `train` split; the saved identifiers use `gpqa_main`. The requested and inferred effective split are retained as a provenance discrepancy rather than silently equated. The historical loader revision is not recorded, so this effective-split interpretation uses the current implementation and stored task identities. Thus a dataset's published split name and a detector's held-out fold mean different things. A detector can be evaluated on a task-held-out fold drawn from a benchmark training split. That is an internal task holdout, not an official benchmark test-set evaluation.

SVAMP does not appear in the authoritative standardized tournament manifest. It is excluded from the reported four-domain totals. The complete prompt, task identifier, reference answer, split, and any deterministic choice shuffle must remain associated with a question. An MCQ label such as B only has meaning relative to the exact displayed ordering. The project stores or reconstructs that ordering before evaluating the option label.

## 3.4 Response and revision protocol

A saved step is a complete response generated under the collection's prompt contract. For the canonical registry, seed 7 and dataset shuffle seed 17 are recorded. The temperature panel uses 0.1, 0.6, and 1.0, and the common completion-token cap is 256. Response horizons are domain-specific: ten increments for GSM8K and GPQA, fourteen for MATH, and eight for ARC. Individual metadata remains authoritative when a cell records a deviation or recovery event.

The standardized detector corpus has five saved increments per trajectory. Its generation settings include temperature 0.6 and seed 7. A repeated call with a revision prompt is not equivalent to revealing another fragment of one uninterrupted chain of thought. The revision prompt itself can change the distribution of the next answer. The stopping target is therefore defined relative to this complete sequential protocol, including prompt construction and sampling.

The minimum admissible stopping step is two in the principal experiments. This floor is a design choice informed by development analyses. It is not a theorem that every question requires a revision. Some trajectories are correct only at step one, and the floor can exclude their best candidate. Allowing an answer-selection policy to return a previously saved candidate would define a different action space; such a policy must be compared separately.

At every decision, computation already incurred is sunk. The continuation decision should charge the incremental cost of future generation, not charge the prefix again. At evaluation, however, total policy cost includes every generated response up to the selected step. The distinction keeps the Bellman recursion and empirical cost accounting consistent.

## 3.5 Recorded signals and grading

Recorded signals can include candidate text, normalized answer, correctness, completion-token count, generation runtime, answer changes, log-probability summaries, entropy, hidden-state displacement, and peer agreement. Availability varies across experiments. A missing signal should be recorded as missing or excluded by a documented rule; treating it as a meaningful zero can alter a learned boundary.

Correctness labels use the benchmark's answer type. Numeric tasks use extraction and normalization; symbolic MATH tasks use the project's mathematical equivalence checks; multiple-choice tasks compare the parsed option with the exact displayed reference. Normalization must preserve information that matters to equivalence. For example, reducing an arbitrary plane equation to its right-hand side would identify distinct mathematical objects, and treating the phrase 'the third option' as the fraction one third would corrupt an MCQ answer.

The grader regression script checks thirty specific cases. The minimum-boundary script checks six cases. Their execution is recorded by command, because historical standalone scripts are not necessarily thirty and six discoverable pytest functions. These checks defend the tested semantics. They do not constitute exhaustive validation of every mathematical answer or all historical parser versions.

All empirical labels used in the primary boundary tables are the stored `correct` values. The data-freeze tools do not mutate them. A future regrade must preserve the previous labels, record the grader version, compare changes by domain and answer type, and publish derived results under a new version. That procedure avoids silently changing the estimand halfway through a comparison.

## 3.6 Statistical units and grouping

Rows from one trajectory are dependent, and trajectories from the same task share the question and reference. Task-held-out detector folds keep all rows for a question outside the detector's training data. A source-qualified key combines the cell with its run identifier to prevent accidental merging of trajectories from different model-domain cells. Checking raw identifier collisions is useful, but source qualification remains the defined identity rule.

The strongest stored task-grouped detector analyses use five outer folds. For meta-model stacking, every upstream fitted component must also respect the outer split: its parameters and any calibrated thresholds must be fit using outer-training data alone. Producing a base score out of fold somewhere in the development pipeline does not automatically make a later outer-fold stacked result nested. Chapter 4 explains why the historical 0.955156 result falls short of that standard.

Cluster bootstraps resample questions when the same question appears in several trajectories. Model-domain estimator comparisons use the recorded cell bootstrap for their reported interval. These intervals answer different questions. A question bootstrap estimates variability associated with the observed question population conditional on the fixed model panel; a cell bootstrap summarizes variability across the existing panel. Neither one is a multi-seed generation replication.

## 3.7 Outcomes and computation measures

Accuracy is the mean binary correctness label at the policy's selected answer. Historical per-trajectory step utility is $U_i(\tau_i)=C_{i,\tau_i}-0.05(\tau_i-1)$, so the mandatory first response is a common baseline and each additional increment incurs the penalty. This penalty is a utility convention, not five percent of physical power consumption or a universal economic price. Token utility is $C_{i,\tau_i}-0.0002\sum_{s=1}^{\tau_i}L_{i,s}$, where $L_{i,s}$ is measured completion tokens; 250 completion tokens carry the same penalty as one response increment. Charging every generated token includes the first response, which cancels in paired policy differences on the same trace. The theoretical reward in Chapter 2 can subtract the common prefix cost without changing the optimal continuation decision.

Token savings are one minus the ratio of total stopped-policy completion tokens to total full-horizon completion tokens, calculated on a common task panel. The ratio of totals differs from the average per-question saving and weights long generations more heavily. Both the definition and denominator must accompany a percentage. Prompt tokens, scoring passes, peer generations, controller inference, and runtime may have separate costs and cannot be omitted from a claim about total compute savings.

ROC-AUC measures ranking, with half credit for tied scores. Brier loss measures squared error of probabilistic predictions. Neither alone establishes a stopping guarantee. Class-balanced raw probability outputs generally target a reweighted posterior, as derived in Chapter 2; their held-out ranking or marginal calibration does not validate natural-distribution conditional hazards. A detector with high pooled AUC can be poorly calibrated around the selected stopping threshold or fail on an underrepresented domain. The evaluation therefore includes micro, task-macro, domain-macro, worst-domain, and policy utility summaries.

## 3.8 Reproducibility and evidence freeze

The freeze records exact local byte hashes and canonical LF content hashes for the selected tournament files, cross-checks historical fingerprints, and pins supporting boundary and analysis artifacts. It also records source-code identities. File size and row count are checked in addition to content identity, because a compact summary alone can obscure selection errors. The reproducibility appendix gives the verification commands.

The workstation lock and recorded historical environment are separate evidence. Stored Blackwell metadata records a CUDA 13.0 runtime and PyTorch 2.13.0+cu130, whereas the current workstation has different versions. Compiler libraries, drivers, hardware, random-state handling, and nondeterministic kernels can affect a repetition even with an exact package list. The freeze supports inspection and re-analysis of the stored corpus; bit-identical regeneration is not established.

Every newly produced table and figure includes or is associated with its source paths and a hash manifest. The thesis build uses the recomputed evidence directory, not hand-edited numerical summaries. The separation allows a reader to inspect a claim, find the table that supplies it, and then verify the underlying frozen sources without reconstructing the author's conversational history.
