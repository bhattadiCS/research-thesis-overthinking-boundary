# Independent mathematical reliability audit protocol

Recorded 2026-10-13 before substantive claim judgments. Repository baseline HEAD: 5304575ce8d421673ee85e3429a128fbddeee697. Exact raw-byte SHA-256 identities are recorded in baseline_hashes.json. Existing untracked user files are outside authorized edits.

## Source hierarchy and scope

CURRENT_FORMAL_THESIS.md identifies concise v7 as primary; concise_v7 chapter sources, references.md and build_concise_thesis_v7.py generate the compiled v7 manuscript. V5 is extended supporting material, not a separate experiment. Determine v5 chapter identities from build/source-integrity receipts and actual comparisons. Repository README/CLAUDE historical canonical-math assertions do not override this user-designated hierarchy. Receipts and previous tests identify evidence only.

Inventory all equations, definitions, mathematical assertions, quantitative table cells, captions and conclusions in both compiled manuscripts, with mappings to chapter sources, implementation and frozen evidence. Record duplicates rather than silently omitting them. Supporting sources are audited insofar as they support report claims. Bibliographic metadata, novelty, institutional acceptance, raw label semantics and population generalization are distinct from mathematical validity.

## Outcomes and evidence standard

- VERIFIED UNDER RECORDED ASSUMPTIONS: independently reconstructible derivation or exact numerical reproduction, explicit sufficient assumptions, and independent second review for central theorems/headline quantities.
- INCORRECT: explicit contradiction, counterexample satisfying stated assumptions, invalid algebra, or reproducible numerical mismatch beyond declared rounding. Reviewer unfamiliarity is insufficient.
- REQUIRES CORRECTION OR ADDITIONAL ASSUMPTIONS: necessary qualification absent/ambiguous or unsupported theorem application; record sufficient conditions without claiming empirical establishment.
- NOT ESTABLISHED: insufficient accessible evidence, unresolved assumptions, or incomplete verification; neither pass nor failure.

Finding categories: demonstrable error; missing/ambiguous assumption; valid conditional result; empirical limitation; presentation issue; unresolved question. Severity reflects consequences, independently of category. Do not conflate validity, empirical support, originality or institutional acceptance.

The same standard governs favorable and unfavorable findings. Deductive claims require checked definitions, quantifiers, boundary cases and every proof step. Standard theorem applications require primary-source assumptions checked against this setting. Empirical identities require source rows and reproducible arithmetic; uncertainty claims additionally require sampling and dependence assumptions. Code correspondence requires actual decision paths and artifact inspection, not docstrings. Numerical tests supplement, never prove general theorems. Frozen labels are versioned endpoints, not independently adjudicated truth.

## Independence and reconciliation

Agent A owns v7 core theory; Agent B owns v5 extended bounds; Agent C owns empirical and implementation mathematics. Each receives the same source baseline and writes only in its own directory. Initial reviews precede exchange of verdicts. B then cross-reviews A's central proofs; A cross-reviews B's consequential bounds. Coordinator independently recomputes C's headline quantities, owns source/PDF correspondence, references, combined inventory and corrections. Resolve disagreements by derivation, admissible counterexample or reproducible evidence, not votes. Track dependencies and propagate every material correction.

## Preservation and completion

No new generation, collection, training, production regrading, publication, submission, messages to faculty, commits or pushes. Do not change frozen dependencies, published sources/PDFs, data or receipts. Only coordinator edits a separate versioned manuscript when justified; rebuild and visually review if manuscript mathematics changes. Run existing mathematical_foundations and live_uncertainty tests in a suitable existing environment, recording exact commands and limits. Review actual rendered PDF images as well as text. Record precisely which pages and claims were reviewed; never infer complete visual verification from extraction or prior receipts.

Deliver claim/dependency ledger, findings, derivations, numerical checks, tests/commands, hashes, corrections/dependency effects, disagreements, advisor assessment, separate verdicts and exact unresolved evidence. Exhaustive inventory is not exhaustive verification: unresolved ledger entries must remain explicit. No blanket all-mathematics verdict unless every claim and required second review is complete.
