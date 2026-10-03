# Current advisor meeting brief — October 2, 2026

This is the current meeting account of the completed research package. The original roadmap retains historical hypotheses and proposed dates; use this brief and the final evidence below when describing the present results. Committee transmission, feedback, rehearsals, defense, approval and institutional submission have not occurred through this work. A separate draft archival candidate passes PDF/A-2b validation; the final approved revision and its validation remain pending.

## Sixty-second pitch

> “I have completed a six-chapter, 86-page research draft, formal stopping proofs, a frozen evidence package, an executable online controller and a trained predictor that uses only the available reasoning prefix. The learned policy saved 56.51% of completion tokens on 100 GSM8K questions and 52.11% on 20 arithmetic traps. Accuracy was 7 versus 6 correct and 1 versus 1, respectively, so these small, weak-baseline panels do not prove accuracy preservation. Every learned task stopped at step two; there is no demonstrated adaptive benefit over that fixed budget. The earlier 0.955 AUC is a future-informed retrospective result, not the live detector. There is no universal early peak. The 25-page paper and 25-slide defense package are ready for review. I now need agreement on committee review, the remaining empirical claims, rehearsal and defense arrangements, and the applicable program requirements.”

## What is ready for review

| Deliverable | Verified present state | Evidence |
| --- | --- | --- |
| Scientific freeze | 52 tournament files plus 679 supporting files: 731 selected files, verified under their declared exact-byte or canonical-LF contracts. Current environment inventories and partial historical provenance remain separate. | [Master manifest](../data_manifest_v1.json); [software provenance](../software_provenance_v1.json) |
| Mathematical foundation | Conditional repair/corruption identity; cost-aware finite-horizon Bellman/Snell solution; sufficient conditions and counterexamples for myopic stopping; uncertainty and calibration limits. Twelve exact-system checks are included in the recorded test suite. | [Canonical proofs](../research/mathematical_foundations.md); [test record](verification/final_test_results.txt) |
| Online engineering | Actual generation stops future reasoning calls after STOP. The predictor receives only current/past observations and public domain. Frozen sources and event ledgers preserve four measured paired collections. | [Implementation and results](ONLINE_STOPPING_IMPLEMENTATION_AND_RESULTS_2026-10-02.md) |
| Thesis draft | Six chapters, 86 pages, 17,046 words, four figures and fourteen numbered tables, with references and appendices; all pages visually reviewed. | [Review PDF](../output/pdf/Masters_Thesis_Draft_v1_Aditya_Bhatt.pdf); [exact-byte build manifest](thesis_build_manifest.json) |
| Research papers | Separate 25-page, 9,269-word research paper; anonymous NeurIPS preparation with seven main-text pages and 22 total pages, plus a tested anonymous supplement. The venue package is prepared, not submitted. | [Long-paper manifest](paper/paper_build_manifest.json); [NeurIPS manifest](neurips/build_manifest.json); [supplement manifest](../output/neurips/anonymous_supplement_build_manifest.json) |
| Defense preparation | Editable 25-slide PowerPoint, 13 native tables and three native charts, 1,800 seconds of speaker notes and 46 defense questions. This is preparation for an actual talk. | [Editable deck](../output/presentation/Thesis_Defense_v1_Aditya_Bhatt.pptx); [defense source and notes](defense/); [build manifest](defense/final_build_manifest.json) |

The final freeze content fingerprint is `4c950d56495aa204a50a4db9b1daa823b279ff9d582d033832769d6b7bb4800c`. The recorded combined suite and the actual extracted anonymous supplement each passed 107 tests. Those results establish the tested software properties, not the correctness of every historical label or a live performance guarantee.

## Empirical result and its scope

All four actual collections use local Qwen2.5-0.5B-Instruct, a five-response horizon and a minimum of two responses. Savings below count emitted completion tokens, including EOS; repeated prompt costs and padded decoding work are reported separately. They are not energy measurements.

| Policy and panel | Full-horizon correct | Active correct | Completion-token savings |
| --- | --- | --- | --- |
| Confidence heuristic, 100 GSM8K tasks | 6/100 | 6/100 | 2.94% |
| Learned prefix policy, same 100 tasks | 6/100 | 7/100 | 56.51% |
| Confidence heuristic, 20 traps | 1/20 | 1/20 | 3.34% |
| Learned prefix policy, same 20 traps | 1/20 | 1/20 | 52.11% |

The learned GSM8K accuracy change is +1 percentage point; its conservative paired 95% interval is approximately **[−4.27, +6.21] points** under an iid task-pair reference model. For the handpicked traps the corresponding reference interval is **[−19.68, +19.68] points**, with no representative adversarial-population coverage. No accuracy noninferiority margin was prespecified. Equal observed counts and tests with p=1 do not prove zero population loss. See the independent [main uncertainty record](../research/outputs/semester2/online_stopping_20261002/learned_main/live_uncertainty.json) and [trap uncertainty record](../research/outputs/semester2/online_stopping_20261002/learned_adversarial/live_uncertainty.json).

Every learned task stopped at response two, and all 120 learned shared prefixes matched their actual full-horizon baselines. The outcomes coincide with fixed-two replay on those prefixes; an independently generated fixed-budget comparison and a capable, format-reliable solver are needed before claiming adaptive added value. The baseline produced valid requested JSON in only 88/500 GSM8K responses. The predictor's archived training instrument had zero strict-JSON rows, so archive calibration does not establish calibration on the live instrument.

The historical **0.955156 AUC** uses future-step features and non-nested meta-training. It is a retrospective development diagnostic, distinct from the separate causal-prefix GRU result of 0.874326 and from the executed two-head model's held-out current/next AUC of 0.710094/0.692308. None is a live accuracy percentage. Neither a universal response-2/3 peak nor the combined promise of 30–40% savings with zero accuracy loss is established. The general optimal rule uses conditional continuation value; the executed estimated one-step drift policy is not proved Bellman optimal.

Observed controller overhead met the under-10-ms benchmark: maximum 0.4383 ms on 18,880 saved-MATH decisions and 0.9369 ms on 4,000 repeated learned-prefix decisions. These scoped measurements exclude LLM generation and do not prove a universal latency bound. Optional peer receipt/accounting controls are tested; no measured run used a multi-model peer fleet.

## Decisions and real events still needed

1. **Committee review:** confirm recipients and the review schedule, transmit the complete draft, then record actual comments and resolutions in the [revision ledger](committee_revision_ledger_v1.csv). Draft v1.1 and v2.0 should reflect the two real feedback cycles; internal agent review cannot replace them.
2. **Scientific scope:** agree whether the thesis defends the current audited development findings or requires a separate confirmatory evaluation. Any new evaluation should freeze an accuracy-loss margin, task allocation and fixed-budget controls before inspecting its outcomes. Preserve the present freeze and report new evidence separately.
3. **Rehearsal and defense:** arrange a consenting peer recorded mock, an adviser dry run, confirmed committee availability, the defense format and logistics, and the actual announcement. Use the prepared timing notes and question bank; record actual timing and questions.
4. **Program and institutional confirmation:** confirm the EP program’s internal approval/grade cutoff, defense announcement lead time, binding requirement and accepted PDF/A profile. The [current EP plan](completion_and_submission_plan_2026.md) records the July 2026 process, November 30 graduation-application deadline and holiday overlap. After actual approval, create the final revision, validate its exact PDF bytes with a dedicated PDF/A validator if required, obtain real signatures, deposit it and retain acceptance/clearance records.

The [completion audit](milestone_completion_audit_v1.md) and [committee delivery package](committee_review_package_v1.md) distinguish completed digital preparation from these pending human and institutional milestones. The full W1–W14 objective remains active until the required events and records exist.
