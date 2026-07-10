# ⚠️ SUPERSEDED DATA — do not cite

Everything in this directory dates to **2026-06-12**, before the 2026-06-20 audit
remediation (`ef45dda`/`6f832ea`). In particular `cross_family_summary.csv` contains
**pre-floor boundary values (`corrected_boundary_step = 1.0`)** for
`deepseek_r1_distill_1p5b`, `deepseek_r1_distill_7b`, and `phi_4_mini_instruct` that
violate the t≥2 decision floor and were the source of a documented downstream
contradiction (see `ThesisDocs/rigor_audit/00_repo_state_and_staleness.md` §B3).

Current cross-family results live under
`research/outputs/experiment_matrix/_aggregate/{dataset}/`. The narrative report for
this stale tree (`research/CROSS_FAMILY_REPORT.md`) has carried a SUPERSEDED banner
since commit `9819ee2`; this README extends that banner to the raw CSVs, which
previously carried no warning of their own.
