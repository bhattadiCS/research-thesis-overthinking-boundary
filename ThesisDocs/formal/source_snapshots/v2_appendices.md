# Appendix A Reproducibility and evidence sources

The empirical results have two software profiles. The original research snapshot is preserved at revision `09225c95c676ab3cae1c4b9586d74f7f803f519f` with `data_manifest_v1.json`. Revision `6e4378bef9a98037c20d1381eb9c7b61462a6578` contains the subsequent software review and `data_manifest_post_review_v1.json`. The latter changes validation, grading and controller input handling while preserving the original corpus, fitted predictor, generation ledgers and reported results. Each live manifest additionally binds the exact executed source copies in its `locked_code` directory.

The following PowerShell commands create detached checkouts and new analysis directories outside the frozen result folders. Run them from a clone containing the recorded evidence, using a Python analysis environment described by `software_provenance_v1.json`. The historical tournament environment is only partially recorded; these commands verify and reanalyse saved data rather than regenerate the original model corpus.

```powershell
$historicalRevision = `
  "09225c95c676ab3cae1c4b9586d74f7f803f519f"
$reviewedRevision = `
  "6e4378bef9a98037c20d1381eb9c7b61462a6578"
$scratch = Join-Path $env:TEMP `
  ("stopping-thesis-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $scratch | Out-Null
$historical = Join-Path $scratch "historical"
git worktree add --detach $historical $historicalRevision
Push-Location $historical
python tools/freeze_research_data.py verify `
  --manifest data_manifest_v1.json `
  --allow-line-ending-changes
python research/tests/test_graders.py
python research/tests/test_boundary_floor.py
python -m pytest `
  research/tests/test_mathematical_foundations.py `
  research/tests/test_data_freeze.py `
  research/tests/test_online_controller.py `
  research/tests/test_online_replay_accounting.py `
  research/tests/test_prefix_stopping_model.py `
  research/tests/test_learned_online_controller.py `
  research/tests/test_live_uncertainty.py -q
python tools/recompute_thesis_evidence.py `
  --output-dir (Join-Path $scratch "tables")
$liveCopies = Join-Path $scratch "live-panels"
Copy-Item -LiteralPath `
  "research/outputs/semester2/online_stopping_20261002" `
  -Destination $liveCopies -Recurse
$panels = @($liveCopies, `
  (Join-Path $liveCopies "adversarial_live"), `
  (Join-Path $liveCopies "learned_main"), `
  (Join-Path $liveCopies "learned_adversarial"))
python tools/analyze_live_stopping_uncertainty.py @panels
Pop-Location

$reviewed = Join-Path $scratch "reviewed"
git worktree add --detach $reviewed $reviewedRevision
Push-Location $reviewed
python tools/freeze_research_data.py verify `
  --manifest data_manifest_post_review_v1.json `
  --allow-line-ending-changes
python research/tests/test_graders.py
python tools/verify_post_review_behavior.py `
  --output-dir (Join-Path $scratch "review-behavior")
Pop-Location
```

The historical grader receipt contains thirty cases, whereas the revised grader exposes ninety-eight tests. The recorded verification receipts contain 107 passing tests for the historical research suite and 243 for the broader post-review repository suite. These counts refer to their named source versions. The post-review behavior check reproduces all 480 saved decision histories across the four live collections and audits copied ledgers; it performs no model generation or latency measurement. The repeated baseline rows in the learned collections do not constitute additional generations.

The uncertainty command writes only into the copied panels, and the table entry point receives an explicit scratch output path. New generation experiments require the documented model snapshot, task allocation, policy and generation settings, together with a fresh result directory. The manuscript build manifests record the rendering dependencies, including the pinned KaTeX package and local browser engine.

[[EVIDENCE_TABLE]]

The standardized tournament manifest lists the mandatory fifty-two trace files. Supporting evidence adds the matrix, paired arms, cached estimator comparisons, stored OOF predictions, analysis implementations, and newly collected events. This source selection is explicit, so a reader can distinguish a fingerprint of selected evidence from a census of every incidental file in the repository.

# Appendix B Claim scope and mathematical review

**Table 16. Claim scope and required evidence.**

| Claim | Evidence required | Supported interpretation |
| --- | --- | --- |
| Binary repair-corruption identity | Common transition panel or conditional probability proof | Exact decomposition in Chapter 2 |
| General optimal stopping | Full conditional continuation law and all costs | Finite-horizon theorem; not a calibrated learned deployment |
| First drift crossing is optimal | Pathwise persistence after the crossing | Conditional theorem; not established universally by the empirical curves |
| 0.955156 AUC | Stored historical tournament result | Retrospective non-nested diagnostic |
| Causal detector ranking | Task-grouped causal sequence outputs | Internal development evaluation |
| Replay completion-token saving | Frozen traces and selected prefixes | Counterfactual development quantity |
| Live completion-token saving | Actual generation events from both arms | Model-, task-, and protocol-specific measurement |
| No meaningful accuracy loss | Prespecified tolerance and adequate paired confidence interval | Not inferred merely from observed equality |
| Grader regression coverage | Versioned test inputs and execution receipts | Thirty historical cases and ninety-eight revised tests cover specified parser behaviors; corpus-wide validity requires separate adjudication |

The proof audit explicitly checks the filtration, hidden correctness target, measurability of stopping events, finite-horizon integrability, cost accounting, and the difference between conditional beliefs and sample averages. Exact finite-system tests supplement those proofs by checking enumerated policy values and counterexamples. Neither a test suite nor a plot substitutes for a general mathematical argument.

# Appendix C Supporting research artifacts

The research is accompanied by a prepared twenty-five-page paper, an anonymous conference manuscript with a technical appendix, and a native twenty-five-slide presentation with thirty minutes of speaker notes. The presentation contains thirteen editable tables and three native charts. These artifacts communicate the mathematical argument and the frozen empirical results in formats suited to review and presentation.

The preserved eighty-six-page historical v1 archival candidate has a byte-bound veraPDF report confirming PDF/A-2b conformance. Its validation establishes the document-format result for those bytes. An oral defense and institutional deposit are separate events and are not recorded by this research evidence.

# Appendix D Paired accuracy uncertainty

For a task sampled under a common independent and identically distributed task law, let $I$ indicate an incorrect baseline answer repaired by the active policy and let $W$ indicate a correct baseline answer corrupted by it. The population accuracy difference is $\delta=\pi_I-\pi_W$, where $\pi_I=\Pr(I=1)$ and $\pi_W=\Pr(W=1)$. The two indicators are mutually exclusive within a task; the analysis does not assume their independence.

Across $n$ independent task pairs, each marginal discordance count is binomial. Construct an exact two-sided 97.5% Clopper–Pearson interval $[L_I,U_I]$ for $\pi_I$ and a separate interval $[L_W,U_W]$ for $\pi_W$ [Clopper1934]. Each interval has noncoverage probability at most 0.025. The union bound therefore gives simultaneous coverage at least 0.95, irrespective of the dependence between the two counts. On that event, subtraction yields

$$\delta\in[L_I-U_W,\ U_I-L_W].$$

This is the conservative 95% paired interval reported for the live panels. Exact finite multinomial outcome enumeration supplements the coverage argument; it does not replace the binomial assumptions. With no observed discordances, both marginal lower bounds are zero and both upper bounds equal $1-0.0125^{1/n}$. Thus observed equality gives a nonzero uncertainty interval, rather than evidence of exact equivalence. No accuracy tolerance or noninferiority hypothesis was prespecified.

The twenty hand-selected adversarial tasks are a fixed challenge bank. An interval calculated using an independent-task reference model does not provide randomized coverage for an adversarial population. Completion-token intervals instead use a paired task bootstrap of the ratio of total token differences to total baseline tokens. These are descriptive resampling intervals, rather than exact finite-sample certificates.
