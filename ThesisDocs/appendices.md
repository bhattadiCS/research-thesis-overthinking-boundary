# Appendix A Reproducibility and evidence sources

Run the following commands from the repository root. The environment provenance distinguishes the current workstation from the historical training environment. A successful verification confirms the frozen evidence identity under the chosen byte rule; it does not regenerate the language-model corpus.

```powershell
python tools/recompute_thesis_evidence.py
python research/tests/test_graders.py
python research/tests/test_boundary_floor.py
python research/tests/test_mathematical_foundations.py
python research/tests/test_online_controller.py
python -m pytest research/tests/test_prefix_stopping_model.py research/tests/test_learned_online_controller.py research/tests/test_live_uncertainty.py -q
python tools/summarize_failure_audit.py
python tools/analyze_live_stopping_uncertainty.py research/outputs/semester2/online_stopping_20261002 research/outputs/semester2/online_stopping_20261002/adversarial_live research/outputs/semester2/online_stopping_20261002/learned_main research/outputs/semester2/online_stopping_20261002/learned_adversarial
python tools/build_master_thesis.py
```

Data-freeze and live-evaluation commands are documented in the corresponding implementation reports. Use their exact model-local path and frozen task/policy arguments when reproducing a generation experiment. The thesis build requires the pinned KaTeX dependency recorded in `tools/thesis_pdf_package.json`, the Python packages used by the builder, and a local Chrome or Edge executable. It does not fetch external mathematical assets while rendering.

[[EVIDENCE_TABLE]]

The standardized tournament manifest lists the mandatory fifty-two trace files. Supporting evidence adds the matrix, paired arms, cached estimator comparisons, stored OOF predictions, analysis implementations, and newly collected events. This source selection is explicit, so a reader can distinguish a fingerprint of selected evidence from a census of every incidental file in the repository.

# Appendix B Claim scope and mathematical review

| Claim | Evidence required | Current interpretation |
| --- | --- | --- |
| Binary repair-corruption identity | Common transition panel or conditional probability proof | Exact decomposition in Chapter 2 |
| General optimal stopping | Full conditional continuation law and all costs | Finite-horizon theorem; not a calibrated learned deployment |
| First drift crossing is optimal | Pathwise persistence after the crossing | Conditional theorem; not established universally by the empirical curves |
| 0.955156 AUC | Stored historical tournament result | Retrospective non-nested diagnostic |
| Causal detector ranking | Task-grouped causal sequence outputs | Internal development evaluation |
| Replay completion-token saving | Frozen traces and selected prefixes | Counterfactual development quantity |
| Live completion-token saving | Actual generation events from both arms | Model-, task-, and protocol-specific measurement |
| No meaningful accuracy loss | Prespecified tolerance and adequate paired confidence interval | Not inferred merely from observed equality |
| Grader has no errors | Independent corpus adjudication | Thirty regression cases do not prove this |
| Thesis accepted and degree cleared | Committee and institutional records | Pending human and institutional milestones |

The proof audit explicitly checks the filtration, hidden correctness target, measurability of stopping events, finite-horizon integrability, cost accounting, and the difference between conditional beliefs and sample averages. Exact finite-system tests supplement those proofs by checking enumerated policy values and counterexamples. Neither a test suite nor a plot substitutes for a general mathematical argument.

# Appendix C Revision and final submission

The roadmap's two committee cycles require actual feedback. For each cycle, record the reviewer comment, affected statement, evidence or proof needed, resolution, and verification. Do not identify a draft as approved without the corresponding approval. The planned defense deck, paper, mock defenses, oral defense, signatures, library deposit, and registrar clearance remain traceable in the milestone audit.

The electronic-thesis draft uses one-inch digital margins, a body font of at least ten points, double-spaced body text and abstract, Roman front-matter numbering, and Arabic main-text numbering. These are formatting targets derived from the current Sheridan Libraries requirements [JHUFormat]. Final PDF/A export and conformance validation follow the final committee-approved revision. No conformance or submission claim is made for an ordinary review PDF.

# Appendix D Paired accuracy uncertainty

For a task sampled under a common independent and identically distributed task law, let $I$ indicate an incorrect baseline answer repaired by the active policy and let $W$ indicate a correct baseline answer corrupted by it. The population accuracy difference is $\delta=\pi_I-\pi_W$, where $\pi_I=\Pr(I=1)$ and $\pi_W=\Pr(W=1)$. The two indicators are mutually exclusive within a task; the analysis does not assume their independence.

Across $n$ independent task pairs, each marginal discordance count is binomial. Construct an exact two-sided 97.5% Clopper–Pearson interval $[L_I,U_I]$ for $\pi_I$ and a separate interval $[L_W,U_W]$ for $\pi_W$ [Clopper1934]. Each interval has noncoverage probability at most 0.025. The union bound therefore gives simultaneous coverage at least 0.95, irrespective of the dependence between the two counts. On that event, subtraction yields

$$\delta\in[L_I-U_W,\ U_I-L_W].$$

This is the conservative 95% paired interval reported for the live panels. Exact finite multinomial outcome enumeration supplements the coverage argument; it does not replace the binomial assumptions. With no observed discordances, both marginal lower bounds are zero and both upper bounds equal $1-0.0125^{1/n}$. Thus observed equality gives a nonzero uncertainty interval, rather than evidence of exact equivalence. No accuracy tolerance or noninferiority hypothesis was prespecified.

The twenty hand-selected adversarial tasks are a fixed challenge bank. An interval calculated using an independent-task reference model does not provide randomized coverage for an adversarial population. Completion-token intervals instead use a paired task bootstrap of the ratio of total token differences to total baseline tokens. These are descriptive resampling intervals, rather than exact finite-sample certificates.
