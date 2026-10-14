# Semester 1: progress to date

This three-page summary accompanies the official MATH ACM 625.804 Semester 2 Thesis Approval Form. It is a plain-language update for my advisor, second reader, and program reviewers. The full technical report remains available for detailed methods, code paths, and audit tables.

## Research question and terms

My thesis studies when a reasoning language model should stop writing intermediate work. An additional reasoning step can repair an incorrect answer, leave the answer unchanged, or replace a correct answer with an incorrect one. I treat that choice as an optimal-stopping problem: continue only when the expected benefit of one more step exceeds its expected risk and computation cost.

A **reasoning trace** is the saved sequence of intermediate responses produced while a model answers one benchmark question. A **step** is one response in that sequence. **Correctness** means that the answer at a step matches the benchmark reference answer after the appropriate numerical, symbolic, unit, or multiple-choice normalization. **Accuracy** is the fraction of answers graded correct. A **token** is a unit of model output; counted completion tokens are a consistent proxy for the amount of generated computation.

## What I completed

- Built a repeatable pipeline that prompts models, saves their reasoning traces, extracts intermediate answers, grades every saved answer, and records the measurements needed for later analysis.
- Collected 75,965 cleaned reasoning traces from 52 successful model, dataset, and sampling-setting experiments. A separate standardized collection contains 28,888 five-step traces and 144,440 saved step records.
- Evaluated four distinct problem domains: GSM8K grade-school arithmetic, MATH-500 competition mathematics, ARC-Challenge science reasoning, and GPQA graduate-level questions. This spread tests whether a finding is limited to one style of problem.
- Compared models ranging from approximately 0.5B to 32B parameters, including DeepSeek, Qwen, Llama, Mistral, Phi, InternLM, and Yi families.
- Ran controlled experiments: within each comparison, I held the benchmark questions, prompts, and evaluation rule fixed while varying one or more planned factors such as model family or size, output-token budget, numeric calculation precision, and temperature. Temperature is the sampling setting that controls how deterministic or variable a model's next response is.
- Saved intermediate answers, answer changes, token counts, runtime, confidence and uncertainty signals, and model-agreement signals. I then produced step-by-step plots of correctness, repairs, corruptions, and computation use.
- Added repeatability tests for answer grading and stopping-boundary rules. The answer-grading test suite passes 30 of 30 cases, and the boundary-floor test suite passes 6 of 6 cases.

## Evidence from the first analysis

To measure whether another step is worthwhile, I computed **net continuation value**: the change in correctness after one additional step minus a fixed cost for that step. A positive value favors continuing; a negative value favors stopping. In a fixed 13-model GSM8K panel containing 19,500 traces from 500 problem clusters, the estimated value was +0.051 at step 2 and -0.013 at step 4. The 95% uncertainty intervals were obtained by repeatedly resampling whole problems, rather than individual rows, so that a problem with many saved steps did not dominate the estimate. This is evidence that the useful boundary can change during a trace; it is not a universal stopping step for every model or dataset.

<!-- PAGE BREAK -->

# Semester 1: what I completed and learned

## How the scientific method was applied

1. **Hypothesis.** Additional reasoning eventually ceases to help on at least some traces, because the chance of repairing an error can be overtaken by the chance of changing a correct answer or by the cost of more computation.
2. **Controlled variants.** I held the task set, prompts, and scoring rule fixed within a comparison and varied planned factors such as model family, output budget, temperature, and numerical precision.
3. **Trace collection.** For each run, I saved the sequence of answers and the measurements associated with each response rather than only the final answer. This makes it possible to study how performance changes while the model continues to reason.
4. **Independent grading.** Each intermediate answer was evaluated against the benchmark reference answer. This produces a binary correctness label for every step and makes accuracy a directly defined quantity.
5. **Data analysis.** I compared repair and corruption rates, net continuation value, token use, and accuracy across steps, models, and domains. The plots show the trajectory of these quantities rather than relying on a single final-answer statistic.
6. **Validation checks.** I used task-held-out evaluation, uncertainty intervals based on resampling entire tasks, matched comparisons, and regression tests. These safeguards reduce the risk that a result is caused by leakage, a particular split of questions, or a grading change.

## Further evaluation results

I also evaluated a learned scoring rule that estimates whether an intermediate answer is likely to be correct. Its area under the ROC curve (AUC) was 0.955, compared with 0.943 for the version without peer-model information, on 144,440 saved records from 2,948 tasks. I used five task-held-out folds: each task was scored only by a rule fitted on other tasks. AUC is a ranking measure: it asks whether a correct response tends to receive a higher score than an incorrect response. It is not answer accuracy, and it does not by itself establish a live stopping system.

In an offline replay on 1,500 Qwen2.5-7B GSM8K traces, a preliminary stop rule used 54.34% fewer counted completion tokens, while accuracy changed from 70.53% to 64.20%. This trade-off identifies a real optimization target for Semester 2: save computation without accepting an excessive loss in correctness.

## What I learned

- The value of another reasoning step is empirical, not something that can be assumed from a model's confidence or from a fixed token limit. In the GSM8K panel, the estimated continuation value changed from positive early in the trace to negative later in the trace.
- Models and datasets differ. A stopping rule that works for arithmetic need not have the same behavior on competition mathematics, science reasoning, or graduate-level questions. This is why the project uses four datasets and will retain a final external test set.
- A detector that ranks stored answers well is promising evidence, but it is different from a controller that safely stops a model during generation. The latter is a specific Semester 2 implementation and validation task.
- Accuracy and efficiency must be reported together. The offline replay demonstrated that token savings can come with a measurable accuracy loss; the final policy must make that trade-off explicit rather than reporting only a single favorable metric.
- The experimental foundation is complete: the data schema, trace-collection pipeline, benchmark graders, controlled comparisons, plots, and repeatability tests are in place. No project-stopping problems arose during Semester 1.

## Current position

The core data-collection and initial-analysis work is complete before the final-semester manuscript and defense phase. This places the project ahead of the usual sequence in which the research pipeline and evidence base are still being built while the thesis must be written. The remaining research work is refinement and validation: a prefix-safe live stopping controller and a locked external evaluation set. These are planned completion tasks, not blockers to the progress already achieved.

<!-- PAGE BREAK -->

# Semester 2: plan to completion

All remaining work is organized by month across the upcoming semester, with completion targeted for late November or early December. Mathematical formalization will continue throughout the empirical work, but the central Semester 2 priority is writing a rigorous, defensible thesis and preparing to defend it.

## September: freeze evidence, test generalization, and optimize the stopping policy

I will freeze the data and result manifest, a versioned inventory of the exact inputs and outputs used for the final analysis; document the exact software, model, dataset, and hardware versions; and resolve the effective GPQA split. I will implement a prefix-safe one-step stopping controller, meaning that it can use only information available before the next response is generated. I will then stress-test it with noisy prompts, reasoning traps, and problems outside the original training mix.

In the same month, I will systematically vary the stopping threshold, available features, output budget, temperature, and computation cost. Each candidate will be chosen using held-out problem sets, not the final evaluation set. I will compare a conservative policy, an efficiency-oriented policy, and a never-stop baseline using accuracy, tokens saved, and worst-domain behavior. The goal is not to maximize one score in isolation; it is to choose a policy with a defensible accuracy-efficiency trade-off.

## October: formalize the mathematics and write the thesis

I will consolidate the mathematical development that accompanies the experiments: state the optimal-stopping model precisely, define the repair and corruption probabilities, document the assumptions behind the one-crossing argument (the point where continuing changes from helpful to harmful), and connect estimation error to error in the proposed boundary. In parallel, I will write the introduction, mathematical model, experimental design, results, and discussion chapters from the frozen evidence.

By the middle of the fall term, I will begin an iterative review and revision cycle with Dr. Woods. The goal is to make every mathematical definition, assumption, derivation, and empirical claim correct, internally consistent, and defensible before the oral defense.

## November: revise, defend, and submit

I will continue the advisor and reader revision cycle, prepare publication-quality figures, conduct mock defenses, and prepare the oral defense. After the defense, I will complete final thesis revisions, archive the reproducibility materials, complete electronic-thesis formatting, and obtain the required approvals and receipts. Any final administrative or archival work will be completed in early December.

## Completion evidence

- A versioned data, code, and result manifest that reproduces the final tables and figures.
- A live-controller evaluation with held-out accuracy, token-use, generalization, and safety results.
- A completed thesis, mathematical proof and assumption audit, defense materials, required signatures, and electronic-thesis submission records.

## Submission sequence

1. Send this summary and the full technical report to the advisor and second reader.
2. Incorporate requested revisions into the manuscript and evidence package.
3. Obtain required advisor, reader, program, and chair approvals on the official form.
4. Attach this summary to the signed form, submit it to the Registrar, and retain the receipt.

After advisor and second-reader review, I will complete the required program and chair approvals. Dr. Beryl Castello and Dr. James Spall are the program contacts listed in the enrollment-status report.

**Bottom line:** Semester 1 produced the data, experimental system, controlled comparisons, validation checks, plots, and initial conclusions needed for the thesis. Semester 2 will refine and validate the stopping method while prioritizing the mathematical write-up, completed thesis, and oral defense.
