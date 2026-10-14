# Final evidence-gap search

Read-only follow-up, 2026-10-13. No model fit, generation, regrading or pickle object loading.

Searched 20 workspace archives (19 ZIPs and one unrelated environment tar), five retained checkpoint paths, 3,970 NPZ files (3,917 distinct hashes), 13 prepared pickle metadata files, and 52 N1 cache records. The ZIP entry inventories are saved under gap_zip_inventories/. gap_search.json records every path, hash, query pattern, CSV header, matching JSON key, checkpoint opcode metadata and source-save anchor.

## Archive inventories

| Archive | Entries | Declared-manifest check |
| --- | ---: | --- |
| output/research/Thesis_v7_Publication_Package_2026-10-08.zip | 141 | 141 declared;0 missing;0 unlisted;0 payload hash mismatches |
| output/research/Thesis_v6_Publication_Package_2026-10-07.zip | 132 | 132 declared;0 missing;0 unlisted;0 payload hash mismatches |
| output/research/Thesis_v5_Publication_Package_2026-10-06.zip | 297 | 295 declared;0 missing;2 unlisted;0 payload hash mismatches |
| output/research/Thesis_v4_Publication_Package_2026-10-06.zip | 276 | 274 declared;0 missing;2 unlisted;0 payload hash mismatches |
| output/research/Thesis_v4_Audited_Research_Supplement.zip | 256 | 254 declared;0 missing;2 unlisted;0 payload hash mismatches |
| output/neurips/anonymous_stopping_supplement_v1.zip | 171 | not a research manifest |
| research/outputs/real_traces_colab_smoke.zip | 17 | not a research manifest |
| research/outputs/real_traces_colab.zip | 14 | not a research manifest |
| tmp/publication/checkout_transport/output/research/Thesis_v4_Publication_Package_2026-10-06.zip | 276 | duplicate package hash |
| tmp/publication/checkout_transport/output/research/Thesis_v4_Audited_Research_Supplement.zip | 256 | duplicate package hash |
| research/outputs/real_traces_colab_smoke_deepseek_r1_distill_1p5b.zip | 20 | not a research manifest |
| tmp/pdfs/recheck_v2/poppler/Release-26.09.0-0.zip | 567 | not a research manifest |
| tmp/pdfs/pdfa_tools/verapdf-installer.zip | 4 | not a research manifest |
| tmp/bug_review/historical_checkout/output/neurips/anonymous_stopping_supplement_v1.zip | 171 | duplicate package hash |
| tmp/pdfs/formal_thesis/v5/checkout_transport/output/research/Thesis_v5_Publication_Package_2026-10-06.zip | 297 | duplicate package hash |
| tmp/pdfs/content_audit_v7_2026-10-08/staged_checkout/output/research/Thesis_v7_Publication_Package_2026-10-08.zip | 141 | duplicate package hash |
| tmp/pdfs/compact_v6/staged_checkout/output/research/Thesis_v6_Publication_Package_2026-10-07.zip | 132 | duplicate package hash |
| tmp/neurips_build/official_template.zip | 3 | not a research manifest |
| tmp/neurips_build/tectonic.zip | 1 | not a research manifest |

The two unlisted entries in the early packages are their own embedded manifest and README (`PUBLICATION_MANIFEST.json`/`PUBLICATION_README.txt`, or `SUPPLEMENT_MANIFEST.json`/`SUPPLEMENT_README.txt`). They contain no hidden prediction payload. Every declared payload SHA-256 matches.

The v4–v7 publication packages and audited research supplement contain no NPZ, NPY, PKL, PTH or PT payloads. Their saved peer/selected-answer CSVs are inventoried; they are different endpoints. The NeurIPS supplement contains paired portable/heuristic replay records, not the original 7B fitted replay. No nested ZIPs were found.

## Exact gap scopes

**C-018: NOT ESTABLISHED.** No exact N1 held-out selected thresholds/predictions or per-cell contributions found. Cache retains features, targets and profiles; source computes held-out predictions in memory and prints only aggregate totals. Exact effects and intervals require a refit or retained full-precision held-out evidence.

**C-014: NOT ESTABLISHED.** No exact Blackwell oof_control/oof_hybrid arrays found in available package inventories, local NPZ keys or checkpoint metadata. The Blackwell save function only invokes git; main writes aggregate JSON/Markdown. The advanced checkpoint retains another experiment's OOF arrays (202,440 float64 entries for its first key), distinct from the 144,440-row Blackwell stack. The ultimate checkpoint belongs to another linear/MoE experiment. No pickle object was executed.

**C-015: NOT ESTABLISHED.** The original 7B report retains heads rounded to four decimals and the complete raw feature order, with no scaler. The baseline's 827,804 tokens and 1,058 correct answers are independently verified. Rounding uncertainty leaves 97 of 1,500 stop decisions uncertified; rounded point weights yield 377,774 tokens versus the headline's 377,960, with conservative possible token range [374,891, 380,891]. Insufficient precision does not contradict the full-precision historical endpoint. No full-precision fitted parameters or original per-run stops were found in available package/local named inventories.

## Rounded7B precision assessment

The retained report provides all three heads at four decimals and the historical source provides the complete unscaled feature order. Each coefficient/intercept rounding cell has half-width 0.00005. For a raw feature vector x augmented with an intercept 1, the unknown logit is within 0.00005*sum(abs(x)) of the reported logit. Sigmoid monotonicity gives q/repair/corruption intervals. Drift decreases with q and corruption and increases with repair, so opposite interval corners provide valid drift bounds. A stop is certified only when all earlier decisions and that stop have definite signs.

This certifies 1,403 trajectory stop decisions; 97 remain unclassified by this conservative calculation. Reported rounded weights yield 377,774 tokens and 963 correct answers; the full-precision historical headline is 377,960 tokens and 963 correct. Conservative outer aggregate bounds are 374,891–380,891 tokens and 958–964 correct answers. All trajectories share the same coefficient vectors, so the outer endpoints are not claimed to be jointly attainable by any single coefficient choice. Full coefficient precision or original stopped rows is still needed to certify the exact stopped endpoint. The baseline endpoint is separately reproduced directly from all 15,000 raw rows: 827,804 tokens and 1,058 of 1,500 correct.

## Retained checkpoints are different experiments

The 33,641,015-byte advanced checkpoint has SHA-256 `7405818cdec82813b38317547475bd4d5f5f3d25bf67ca2f695ff54550c4c036`. Pickle disassembly shows fold 3, smoke_test False, and five named OOF arrays for linear, dynamic, BiGRU, TCN and BetaLikelihood models; the first array has 202,440 float64 entries. It also contains per-policy results dataframes. This is the advanced hyper-optimization experiment, distinct from the 144,440-row Blackwell stack and the 1,500-run 7B replay. Current ultimate checkpoint keys instead contain linear/MoE q, repair and corruption dictionaries. All objects were inspected using pickletools; no GLOBAL/REDUCE opcode was executed.

## Limits

- This is an inventory of accessible local workspace/package evidence, not proof that missing files never existed or are absent from external/remote/unreachable Git storage.
- NPZ headers and pickle opcode metadata inspected without object deserialization. Exact contents of arbitrary binary blobs not interpreted as unknown custom formats.
- Source-only reading of historical scripts: never invoked collection/training/git-write helper functions. No model refit or new-generation execution.
- No automatic identification by merely similar AUC or array size: advanced/ultimate, peer, portable and heuristic replay outputs remain distinct experiments.
- Rounded coefficient interval analysis assumes decimal table values are nearest four-decimal representations of originally applied coefficients and historical raw source features are exact. Dependencies across coefficient effects make the aggregate bounds conservative.
