# Data freeze and software provenance verification

The Week 1 freeze identifies the retained tournament corpus and the sources used to recompute thesis evidence. It checks file content and trajectory structure. It does not establish bit-identical model regeneration, semantic correctness of every stored label, nested validity of the historical ensemble, or effectiveness of an online stopping policy.

## Verified tournament coverage

The selection follows `research/outputs/experiments_v2/ultimate_tournament_manifest.json` exactly. All 52 selected CSVs match its archived SHA256 after CRLF and bare-CR normalization to LF. Windows checkout bytes differ in every CSV; their separate raw SHA256 and byte lengths are retained in `data_manifest_v1.json`. No source file was rewritten to obtain a match.

| Benchmark | Trace files | Rows | Trajectories | Unique task identifiers |
| --- | ---: | ---: | ---: | ---: |
| ARC Challenge | 13 | 42,500 | 8,500 | 1,000 |
| GPQA main | 13 | 29,120 | 5,824 | 448 |
| GSM8K | 13 | 40,320 | 8,064 | 1,000 |
| MATH 500 | 13 | 32,500 | 6,500 | 500 |
| Total | 52 | 144,440 | 28,888 | 2,948 |

Every trajectory has exactly the five distinct consecutive steps 1 through 5 and maps to one task identifier. Every step position has 28,888 rows. Correctness labels are binary, with 54,838 correct and 89,602 incorrect rows. No raw run identifier occurs in multiple cells. The manifest still uses `source_cell::run_id` as the defined trajectory key. The model roster contains thirteen configurations, and SVAMP is absent from this selection.

The original legacy aggregate fingerprint is `59551e6e5fc1b1b6577671001e169026fe69c78434d05ca9ac31989ac89e6eb9`. Reconstructing the original list of LF byte lengths and file hashes reproduces it exactly. The current runner hashes a different JSON format and obtains `879e83d1fd84581ecc5e253ac529ed1a2ee22d9b9f70a9f69bf0692e945c1775` for the same LF file contents; these two digest formats are recorded separately.

## Supporting evidence and corpus distinctions

The auxiliary section enumerates matrix trace files, metadata and detector comparisons; both algorithm ablation caches; paired token-limit and quantization arms; replay inputs; tournament OOF predictions and retained final reports; their producer scripts; and the recomputed thesis evidence tables. The final refreshed freeze also includes public adversarial questions, separate gold answers, online controller sources, and the completed online evaluation directory. File membership is audited as well as bytes, so new matching files cannot enter an existing freeze silently.

The model-domain matrix has 52 available trace cells, while its registry contains 53 attempted entries. An independent CSV scan counted 798,770 raw saved records, 75,996 raw source-qualified run identifiers, and 1,954 raw task identifiers, including 38 records with malformed width or missing identifiers. Those raw identifiers are not the eligible analysis denominators. The analysis sanitizer and caches define the separately reported 75,965 eligible trajectories and 1,948 task identifiers. Matrix counts must not be added to tournament totals.

All thirteen standardized GPQA metadata files record a requested `test` split. The retained `load_gpqa_tasks` implementation unconditionally requests `train`, ignoring that argument. The freeze records the configured split, effective split interpretation, and source of this discrepancy. This is a loader interpretation rather than a recorded original source revision. GSM8K metadata records `train`; ARC and MATH metadata and per-task notes corroborate `test`.

## Software evidence

`requirements.lock.txt` and its descriptive alias `requirements.local-environment.lock.txt` pin the 334 distinct installed distributions observed on the executing workstation. `software_provenance_v1.json` identifies Python 3.12.4, the platform, scientific package versions, retained later-analysis runtime contracts, and repository-declared generation requirements. This is an inventory of the current audit and reanalysis environment, rather than a clean minimal environment or historical regeneration recipe. Package indexes, wheel hashes, system libraries, and direct-install source identities are not reconstructed.

The historical tournament preflight instead records PyTorch 2.13.0+cu130, CUDA 13.0 and an NVIDIA RTX PRO 6000 Blackwell Server Edition. Only these observed historical values support `requirements.historical-tournament.partial.txt`. The historical Python, Transformers, LightGBM and other unrecorded versions, model revisions and dataset revisions remain unknown. In particular, the workstation's PyTorch 2.11.0+cu128 and Transformers 5.8.0 do not reconstruct the original training environment. The older `requirements-colab.txt` declares Transformers 4.53.3 and several lower bounds; declarations are not an observed environment lock.

The new failure-classification command explicitly used the repository `.venv`. Its separate current inventory contains 94 distributions, including PyTorch 2.5.1+cu121, Transformers 5.4.0, NumPy 2.4.4 and pandas 3.0.2. `requirements.repository-venv.lock.txt` and `software_repository_venv_observed_v1.json` retain that distinction. The classification summary, table recomputation, prefix training and actual live jobs used base Python; live jobs additionally preserve their own `runtime_environment.json`. None of these observed inventories reconstructs historical experiment software.

The prefix training protocol records the master manifest's path and byte hash as they existed at fitting. Its exact bytes are preserved as `prefix_model_v1/data_freeze_at_training.json`, with SHA256 `6e2d1add18b88efbb8fd1f5f87f6efc80cb8db9453734cbc84259a2575ffa4c9`. The trainer checks its two immutable trace sources against that manifest. The final master manifest supersedes the earlier auxiliary membership, while the saved snapshot preserves the protocol's original reference.

## Integrity checks and adversarial gold review

The seventeen standard-library tests cover changed content, missing or added cells, duplicate and noncontiguous steps, changed task identity, invalid labels, metadata disagreement, forged summaries, auxiliary membership, path escapes, safe output creation, binary hashing, text chunk boundaries, declared bytecode exclusions, and separate local and historical software evidence. Exact rational calculations also check all twenty adversarial golds against the public task IDs. Each answer matches the intended mathematical contract. The race question assumes unchanged runner speeds across both races; that assumption should remain explicit in its public prompt.

The final exhaustive strict audit passes for all 52 tournament sources and their declared supporting membership after all live and analysis producer processes finish. The machine-readable manifest is authoritative for the final auxiliary count and content fingerprint; the fingerprint is not repeated inside this pinned report. All four actual run directories, their original locked code, executed predictor, independent uncertainty and ledger audits, corrected replay, final source-revision notes and runtime provenance are included. The deployed predictor retains its original raw SHA256 `92fe0af86ac0f204d514a6938d0a29dace2b3cdffa24800cc4e2c85a2d51879f`.

Git attributes preserve the exact bytes of new semester2 outputs, reconstructed failure-audit files and derived thesis evidence, including their CRLF JSON/CSV artifacts. Those files contain raw-byte bindings to one another and must not be silently normalized. New manuscript and presentation source directories separately preserve their authoring identities; they are not empirical observations. Master/provenance records, observed locks, public adversarial inputs and the canonical proof use explicit LF. No legacy tournament CSV was rewritten.

```powershell
python -m unittest research.tests.test_data_freeze -v
python tools/freeze_research_data.py freeze --replace
python tools/freeze_research_data.py verify
```

For a new Git checkout that changes only text line endings, use `verify --allow-line-ending-changes`. Its result lists every accepted line-ending-only difference. Strict verification remains the default. Environment capture is a separate explicit command, `python tools/freeze_research_data.py environment`; existing outputs require `--replace`, and a refresh still describes the new local environment rather than the historical experiment.
