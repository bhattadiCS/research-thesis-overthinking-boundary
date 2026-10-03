# NeurIPS Main Track preparation

This is an anonymous research-paper draft, separate from the longer thesis paper. It uses the unmodified official 2026 NeurIPS style. The 2026 full-paper deadline has passed; this package supports preparation for a future cycle. Future-cycle rules must be checked when published. No paper has been submitted, and acceptance or completed human author review is not claimed.

The official [Main Track handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook) limits main content to nine pages; references, technical appendix, and checklist follow the main text. The source places `\label{sec:main-text-end}` immediately before `\clearpage` and the references. The build must verify that this label is on page nine or earlier without altering the official fonts, margins, or style.

Source files:

- `paper.tex`: anonymous main text and numerical claims.
- `references.tex`: seventeen primary-source bibliography entries.
- `technical_appendix.tex`: full proofs, common-reference counterexamples, predictor specification, uncertainty calculations, actual execution/accounting, and verification limits.
- `paper_checklist.tex`: all sixteen official questions, headings, and guidelines with draft-specific answers.
- `build_checklist.py` and `checklist_answers.json`: reproducible checklist filling and provenance. The generator removes the template instruction block and changes only answer/justification placeholders.
- `template/` and `template_provenance.json`: retained official originals and verified download identities.

Build from the repository root:

```powershell
python tools/build_neurips_paper.py
```

The root build uses the verified portable Tectonic engine, checks the main-text page label, extracts searchable text, and renders the PDF for visual inspection. The presentation/PDF operation-start markers have already been run once for their respective artifacts and must not be repeated by this package.

The draft distinguishes the retrospective stacked AUC of 0.955156 from the separate causal-prefix GRU AUC of 0.874326. The final trained two-head model has held-out current/next AUC 0.710094/0.692308. The actual learned 100-task run saves 56.51% of completion tokens with 7/100 correct against 6/100 full horizon, but every task stops at two and the accuracy interval includes harm. No live calibration, noninferiority, Bellman-optimal learned policy, energy saving, or added value over fixed two is claimed.

The accompanying local anonymous review capsule is `output/neurips/anonymous_stopping_supplement_v1.zip`, SHA256 `0e65afa642d7c16bbe0227aafa10c6d1930ec50cc32538c84fed0dfa3eaf7a57`. It includes 169 selected source/evidence files, including ten explicit sanitized review copies with original/review identities. Large raw archives and model weights are omitted. The capsule was tested on its actual extracted copies; it is not a complete public license-reviewed release.

Dependency identities:

- Canonical mathematical source SHA256: `5c49b379f53cf0f6e521e9aeff957cd4bb69b705ead8452dd72dddfaf7a09dbf`.
- Final master data-freeze content fingerprint: `4c950d56495aa204a50a4db9b1daa823b279ff9d582d033832769d6b7bb4800c`.
- Executed prefix-model artifact SHA256: `92fe0af86ac0f204d514a6938d0a29dace2b3cdffa24800cc4e2c85a2d51879f`.
- Unmodified official style SHA256: `c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a`.

The checklist is intentionally not all “Yes.” The source documents the new fitting/decoding settings and local auditable records, but complete historical reexecution metadata and total project compute are unavailable. Public anonymous licensed release, complete asset terms review, and human author verification/ethics attestation remain pending. The disclosure identifies LLM coding agents' substantive contributions to theory, audits, implementation, and manuscript preparation.
