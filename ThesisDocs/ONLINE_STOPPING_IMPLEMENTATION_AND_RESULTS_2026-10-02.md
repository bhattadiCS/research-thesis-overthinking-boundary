# Online stopping implementation and measured development results

**Evidence date:** 2 October 2026 (local time; the ledgers use UTC).

A real, local Qwen2.5-0.5B-Instruct generator executed the stopping policy, with future reasoning calls removed after STOP. The trained prefix policy saved **56.51% of emitted completion tokens on 100 GSM8K test problems** and **52.11% on 20 arithmetic traps**. Correct answers changed from 6/100 to 7/100 and remained 1/20, respectively. These are measured development results with a weak baseline. Every learned-policy problem stopped at step two, so the experiment demonstrates executable prefix stopping and cost reduction; it does not demonstrate an advantage over a fixed two-step budget, population accuracy preservation, or reliable adversarial performance.

The frozen heuristic also ran in the actual generator. It saved 2.94% and 3.34% of completion tokens on the same two panels. Its accuracy remained 6/100 and 1/20. The earlier full-sequence detector with reported AUC 0.955 was not deployed: it has no verified prefix-safe inference contract.

## Runtime boundary and invariants

The public runtime input is `PublicTask(task_id, prompt, answer_type, domain, difficulty)` plus one new completed `Observation` at a time. Observations contain only current/past generated text, answer, explicitly parsed confidence, token costs, and completion timestamp. There are no correctness, gold, oracle-stop, future-row, or retrospective ensemble-score inputs. The controller stores its own immutable prefix view and rejects repeated, skipped, future, or post-STOP observations.

Both policies require at least two completed steps. Step five always terminates before any learned next-step head is queried. The default selection is the latest nonempty already observed answer, carrying it forward when a later response supplies no answer. The heuristic can explicitly retain a prior answer after a confidence drop or choose among already observed answers after answer wobble. Offline grading uses the selected answer in each arm.

`run_online_generation` and the batched collection loops remove a stopped problem before the next `generate_step`/`generate_batch` call. They do not generate a full five-step trajectory and subsequently truncate it. Deterministic fake-decoder tests verify both the call boundary and exact accounting. The learned loop also tests a two-step stopped task alongside another task that continues to step five.

**The risk policy acts between completed incremental reasoning responses.** The Hugging Face adapter's token stopping criterion ends each response at its first strictly valid complete JSON object, EOS, or 128-token cap. It can also honor caller cancellation after a token is generated. Caller abort is distinct from policy eligibility and is not evidence of risk-driven interruption inside a response. Both baseline and active arms use the same response boundary.

Strict JSON requires text `thought` and nonempty text `answer`, an integer confidence from 0 to 100, and a boolean `stop`. Incomplete or incorrectly typed output cannot authorize a confidence-based stop. Failed strict parsing retains the existing fallback candidate with missing confidence. The model's `stop` flag cannot bypass the two-step floor.

Optional peer inputs require a frozen, unique, complete roster at the current step and monotonic receipt times satisfying `completed <= panel_closed <= decision`. Partial, duplicate, stale, and future panels fail validation; all peer completion work, including disagreement, must be charged. Actual asynchronous fleet studies must preserve genuine arrival/barrier receipts in the prospective ledger. All four measured runs used one model, no peer features, and zero peer/verifier/diagnostic generations. The tests validate the peer contract; these experiments do not establish a live multi-model fleet result.

Implementation: [base controller](../research/online_stopping_controller.py), [generation adapter](../research/online_generation.py), [learned controller](../research/learned_online_stopping_controller.py), [live evaluator](../research/run_online_stopping_evaluation.py), and [learned live evaluator](../research/run_learned_online_stopping.py).

## Frozen prefix predictor

The final artifact is [prefix_model.json](../research/outputs/semester2/prefix_model_v1/prefix_model.json), SHA-256 `92fe0af86ac0f204d514a6938d0a29dace2b3cdffa24800cc4e2c85a2d51879f`. It fits unweighted logistic heads for selected-current correctness `q_t` and selected-next-step correctness `p_next`, with independent Platt calibration on separate task groups. A fixed public-task hash allocation retains all 1,500 archived complete Qwen2.5-0.5B GSM8K-train/MATH trajectories: 902 training, 322 calibration, and 276 evaluation tasks. Candidate reconstruction and gold regrading occur offline in that training corpus; original archive labels and files remain intact. Targets use the same latest-nonempty candidate selector as the runtime.

The 21 features are functions of the observed prefix and public GSM8K/MATH domain: step, parsing/answer availability, trusted or missing confidence, prior confidence changes, answer stability/churn, current and accumulated token counts, and text lengths/densities. No runtime labels, future observations, task IDs, model aliases, or timing are predictor features. The archive task allocation, feature contract, parameter bytes, utility constants, and source copies were frozen before learned generation. Neither live GSM8K nor trap outcomes enter fitting or calibration.

The controller recomputes `mu = (p_next - q_t) * (v + c) - lambda` with frozen `v=1`, `c=0`, and `lambda=0.05`, and stops for `mu <= 0` after the floor. It validates both probabilities and artifact identity rather than trusting a supplied stop score. This is a myopic one-step policy, not a proved Bellman-optimal stopping policy.

Archive holdout behavior is already almost fixed two-step: 275/276 tasks stop at two and one at three. Both live panels stop every task at two. The archive has zero strict-JSON rows, while the live instrument requests JSON, so confidence features were not trained on valid live-style confidence. Archived token lengths omit EOS whereas the live generator charges EOS. These instrument differences and marginal task-disjoint calibration limit transport; they cannot support a conditional calibration guarantee on these live prefixes. The provisional candidate-agreement subset was withdrawn before any learned deployment because eligibility depended on future parser agreement. Only the all-1,500-task artifact above was executed.

Training and calibration evidence: [training protocol](../research/outputs/semester2/prefix_model_v1/training_protocol.json), [archive evaluation](../research/outputs/semester2/prefix_model_v1/evaluation.json), and [portable prefix model](../research/prefix_stopping_model.py).

## Actual generation experiment

The main panel contains 100 cached GSM8K test questions, selected by a frozen shuffle with seed 20261002. The 20-question bank is independently constructed arithmetic/stability/delayed-repair material with separately checked rationales; it is not a comprehensive safety benchmark. Public prompts and sealed evaluation gold are separate files. Gold is used for allocation/export and subsequent grading, not by the online policy or generation collector. The panels remain development evidence: source novelty against the broader historic corpus is not assumed, and no noninferiority margin was registered.

All calls used cached Qwen2.5-0.5B-Instruct snapshot `7ae557604adf67be50417f59c2c2f167def9a775`, greedy decoding, a five-step maximum, 128 completion tokens per response, float16 CUDA/SDPA/KV cache, and initial batches of at most 32. The learned runs generated new active trajectories and reused the earlier **actually generated** never-stop arms. Model/tokenizer bytes, prompts, adapter/controller source, task order, caps, and baseline ledgers were checked against the original manifest. Shared prefixes are measured below. Reuse avoids another baseline execution; it is not saved-trace replay of the learned active arm.

| Actual policy / panel | Baseline → active correct | Improved / worsened | Completion tokens baseline → active | Completion savings | Mean stop step | Identical shared prefixes |
|---|---:|---:|---:|---:|---:|---:|
| Heuristic / GSM8K 100 | 6 → 6 | 0 / 0 | 22,244 → 21,591 | 2.94% | 4.85 | 94/100 |
| Heuristic / traps 20 | 1 → 1 | 0 / 0 | 4,736 → 4,578 | 3.34% | 4.75 | 20/20 |
| Learned / GSM8K 100 | 6 → 7 | 1 / 0 | 22,244 → 9,674 | 56.51% | 2.00 | 100/100 |
| Learned / traps 20 | 1 → 1 | 0 / 0 | 4,736 → 2,268 | 52.11% | 2.00 | 20/20 |

Actual strict-format success is poor: main never-stop 88/500 steps, heuristic 86/485, and learned 83/200; traps 14/100, 11/95, and 10/40. Baseline accuracy of 6%/5% and failed formatting dominate interpretation. The heuristic main prefixes differ on six problems after batching changes, despite greedy decoding; no deterministic counterfactual suffix identity is assumed. Learned shared prefixes match all 120 paired problems, and learned results coincide with fixed-two replay on those actual baseline prefixes.

A paired-problem bootstrap with 10,000 replicates and seed 20261002 gives learned completion-savings intervals of **54.81–58.01%** for GSM8K and **45.87–57.27%** for traps. The learned GSM8K accuracy delta is +1 percentage point with a descriptive bootstrap interval [0, +3] points; the other three empirical deltas are zero with degenerate bootstrap intervals. Every exact paired McNemar test has p=1. These intervals/tests do not establish zero population loss, accuracy noninferiority, or adversarial safety. In particular, the trap baseline already fails 19/20 problems. Fixed-budget controls are necessary before attributing a benefit to adaptive probability estimation.

## Costs beyond completion length

The generator records emitted tokens including EOS, repeated prompt/input tokens, auxiliary/peer/verifier work, actual padded prefill/decode slots, model-call time, tokenization time, load time, and peak allocated CUDA memory. Decoder padding is excluded from semantic completion totals and reported separately. No unit-step proxy is substituted for measured length. Auxiliary, peer, and verifier token counts are zero because these runs make no such calls; learned probability inference consumes CPU time, included in controller latency below.

| Actual policy / panel | Input tokens baseline → active | Input + completion savings | Model-call seconds baseline → active | Padded prefill slots baseline → active | Decode slots baseline → active | Model calls baseline → active |
|---|---:|---:|---:|---:|---:|---:|
| Heuristic / GSM8K 100 | 142,498 → 137,574 | 3.39% | 1,127.05 → 1,099.21 | 231,856 → 222,781 | 57,856 → 56,128 | 20 → 20 |
| Heuristic / traps 20 | 27,641 → 26,143 | 5.11% | 245.14 → 231.67 | 41,560 → 38,847 | 12,660 → 12,030 | 5 → 5 |
| Learned / GSM8K 100 | 142,498 → 43,874 | 67.50% | 1,127.05 → 343.25 | 231,856 → 58,624 | 57,856 → 23,564 | 20 → 8 |
| Learned / traps 20 | 27,641 → 8,115 | 67.93% | 245.14 → 74.12 | 41,560 → 10,420 | 12,660 → 5,100 | 5 → 2 |

Original main/trap collection took 2,248.69/510.52 seconds including load (both baseline and heuristic arms). New learned-only collection took 369.52/99.78 seconds including load, reusing the linked prior baseline execution. Respective model-load times are 20.29, 32.43, 25.10, and 24.85 seconds; peak allocated CUDA memory is 2,276,269,568, 1,640,875,520, 1,598,428,672, and 1,319,676,928 bytes. Each folder preserves `live_collection_runtime.json` and per-batch tokenization time. Sequential development timing is measured execution time, not a randomized production service-latency estimate or a dollar-cost estimate.

## Decision latency and verification

The final base-controller benchmark uses **100 distinct competition MATH problems**, 20 repetitions, and both the actual confidence/stability and never-stop policies. Across 18,880 decisions, mean latency is 0.004306 ms, p99 0.013 ms, and maximum 0.4383 ms; no decision exceeds 10 ms. The heuristic-only p99 is 0.015342 ms. Observation assembly, model generation/loading, tokenization, and peer waits are outside this controller benchmark.

The trained policy separately processes prefixes from **100 actually executed GSM8K baseline problems**, 20 repetitions, including feature extraction, both probability heads, Platt calibration, validation, selection, and drift computation. Across 4,000 decisions, mean latency is 0.080343 ms, p99 0.314315 ms, and maximum **0.9369 ms**, with zero decisions above 10 ms. This is CPU decision time; it is not complete solver latency.

Actual active-only decision telemetry is also recorded: heuristic main/traps p99 0.032548/0.028724 ms; learned main/traps p99 0.378534/0.409053 ms. The learned main actual maximum is 0.7038 ms. `live_metrics.json` reports the combined paired-arm distribution; `final_evidence_summary.json` additionally reports active-only distributions to avoid mixing learned inference with the reused baseline.

All **30 owned controller/generation/replay-accounting tests pass**. They cover the mandatory floor, terminal closure, invalid/conflicting telemetry, gold/future exclusion, prefix invariance, actual future-call prevention, caller cancellation, peer completeness/timing/accounting, immutable predictor identity, finite probabilities, recomputed utility, and fair terminal selection with missing final answers. A confidently wrong stable answer with a later correct repair is an explicit heuristic failure test, rather than evidence of robustness. All four frozen-source/live-event accounting audits pass. [Test log](../research/outputs/semester2/online_stopping_20261002/online_and_learned_test_results.txt). The independent prefix-model suite separately reports **14 passing tests**, including portable-vector serialization, task/public-panel exclusions, the 7,500-row regrading audit, future-feature invariance, confidence fallback, terminal guards, and training-freeze hashes: [prefix test summary](../research/outputs/semester2/prefix_model_v1/test_summary.json).

## Retrospective controls and Pareto evidence

The archive replay is separately labelled retrospective. It covers 1,500 archived trajectories with measured main and auxiliary completion lengths. Prompt lengths are unavailable, so full input-inclusive savings are **unknown**, not zero-cost inputs. Both arms use the same latest-nonempty terminal selector; the fair never-stop comparator is 143/1,500 (9.53%), rather than the archive's raw final-row label when that row is empty.

| Archive replay policy | Selected-answer correct | Main completion savings | Completion + auxiliary savings | Paired improved / worsened |
|---|---:|---:|---:|---:|
| Never | 143/1,500 | 0.00% | 0.00% | 0 / 0 |
| Fixed two | 144/1,500 | 51.29% | 51.20% | 32 / 31 |
| Fixed three | 149/1,500 | 33.52% | 33.32% | 26 / 20 |
| Fixed four | 146/1,500 | 16.61% | 16.49% | 21 / 18 |
| Confidence 80 | 144/1,500 | 10.68% | 10.50% | 5 / 4 |
| Confidence 90 | 144/1,500 | 9.95% | 9.77% | 5 / 4 |
| Confidence 95 | 143/1,500 | 8.56% | 8.39% | 4 / 4 |

Fixed two and fixed three are nondominated in this displayed archive policy set. Replay neither prevents already completed work nor demonstrates these savings in new generation. Its legacy parser/confidence instrument also differs from the strict live format.

Each actual run has a Pareto CSV and PNG/SVG that distinguish actual live points from **replay of the live baseline's saved prefixes**. These plots are descriptive development comparisons among predeclared variants. The learned main/trap point exactly overlaps fixed-two replay, reinforcing the absence of demonstrated adaptive added value. The actual heuristic on GSM8K is dominated in this displayed set by that fixed-two replay point; replay controls still need their own actual-generation evaluation before claiming a general live policy ranking.

## Reproduction and provenance

The primary evidence directory is [online_stopping_20261002](../research/outputs/semester2/online_stopping_20261002/). The four actual result folders are the primary directory, `adversarial_live/`, `learned_main/`, and `learned_adversarial/`. Each contains frozen task/gold/policy/model/source hashes, exact executed `locked_code/` copies, generation events and decisions, paired CSVs, per-batch costs, metrics, audit, and separate runtime environment provenance. `final_evidence_summary.json` aggregates the nested metrics and uncertainty; `source_revision_notes.json` maps preserved executed sources to final working sources.

All actual jobs used **global Python** `C:/Users/bhatt/AppData/Local/Programs/Python/Python312/python.exe`, Python 3.12.4, torch 2.11.0+cu128, transformers 5.8.0, numpy 2.3.5, and pandas 2.2.2 on an NVIDIA GeForce GTX 1650. They did not use the repository `.venv`. [Runtime environment](../research/outputs/semester2/online_stopping_20261002/runtime_environment.json) preserves the package/executable observation separately so original manifests remain intact. The final runners automatically record executable/package versions and copy sources before new generation. Post-collection changes improve replay fairness, missing-cost reporting, benchmark coverage, chart readability, and future provenance capture; they do not rewrite the executed source snapshots or actual paired metrics.

These PowerShell commands use existing cached data/model files and fresh result directories. They attempt no model download. Use a new directory for each real run; an existing frozen live manifest is rejected.

```powershell
$stoppingPython = 'C:/Users/bhatt/AppData/Local/Programs/Python/Python312/python.exe'

# Controller timing and explicitly retrospective archive replay.
& $stoppingPython research/run_online_stopping_evaluation.py --benchmark --replay `
  --output-dir research/outputs/semester2/reproduced_replay

# Actual never-stop + frozen heuristic, 100 cached GSM8K test questions.
& $stoppingPython research/run_online_stopping_evaluation.py --live --max-tasks 100 `
  --output-dir research/outputs/semester2/reproduced_main

# Actual arithmetic trap panel, with gold outside the runtime task schema.
& $stoppingPython research/run_online_stopping_evaluation.py --live `
  --task-file research/adversarial_tasks_v1.jsonl `
  --gold-file research/adversarial_gold_v1.jsonl `
  --output-dir research/outputs/semester2/reproduced_traps

# Actual learned active generation paired with the completed actual baseline.
& $stoppingPython research/run_learned_online_stopping.py `
  --baseline-dir research/outputs/semester2/reproduced_main `
  --predictor-artifact research/outputs/semester2/prefix_model_v1/prefix_model.json `
  --expected-predictor-sha256 92fe0af86ac0f204d514a6938d0a29dace2b3cdffa24800cc4e2c85a2d51879f `
  --output-dir research/outputs/semester2/reproduced_learned_main

& $stoppingPython research/analyze_online_stopping_results.py `
  --output-dir research/outputs/semester2/reproduced_main

& $stoppingPython -m unittest discover -s research/tests -p 'test_online*.py'
& $stoppingPython -m unittest discover -s research/tests -p 'test_learned_online*.py'
```

For other public tasks, the JSONL schema is `task_id`, `prompt`, `answer_type`, `domain`, and `difficulty`; gold JSONL separately supplies matching `task_id`, string `expected_answer`, and `answer_type`, with optional evaluator-only rationale. The learned artifact supports only `gsm8k`/`math` domain names. GPU use is local; `--device cpu` is available for the base evaluator if CUDA is unavailable.

The development implementation and latency/token measurements are complete. A confirmation claim requires a fresh, frozen evaluation with a capable, format-reliable solver, a registered accuracy-loss margin, actual fixed-budget comparators, and a separately instrumented fleet if peer consensus is claimed. The current evidence supports the executed stopping boundary and reported local costs, with the accuracy and transport limitations above.
