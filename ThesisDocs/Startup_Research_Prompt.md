# Startup Research & Planning Prompt for Claude Fable 5
## Overthinking Boundary × AIC: A Multi-Agent Diligence & YC Application Operating Brief

*Target Runtime: **Claude Fable 5 for heavy strategic/technical reasoning, Claude Sonnet 5 for mechanical/verification work**, run inside Claude Code or an equivalent agentic harness with subagent dispatch, multi-agent workflow orchestration, WebSearch/WebFetch, and file read/write access. Model and reasoning-effort are set **per subagent**, not once for the whole session — see Section 1.*

*This is not a single-turn Q&A prompt. It is an operating brief for a long-running, multi-agent research-and-planning session that produces and updates real files on disk, spanning **two repositories**, not just a chat reply.*

*Prepared 2026-07-02 by a Sonnet 5 planning pass. Sections 0.4–0.5 contain seed web research from that date; Section 0.1 and 0.6 contain findings from a direct read of both repos' current state — **verify and deepen this, don't re-discover it from scratch.***

---

## ⏰ The One Fact That Changes Your Time Budget

**If Y Combinator is in scope at all, the current Fall 2026 batch's on-time application deadline is July 27, 2026, 8:00pm PT.** [UNVERIFIED — confirm live, see §0.5] As of this writing that is roughly 25 days out. YC now runs **four** batches a year (Winter/Spring/Summer/Fall, expanded from two in 2025), so missing this one means a ~3-month wait, not ~6. Decisions land ~Aug 28; the batch itself runs Oct–Dec 2026 in San Francisco. This deadline should gate your entire prioritization — see Module 6.

**Second fact that changes your scope:** this is not a green-field startup exercise. `C:\Aditya_Data\Personal\algorithm-x-optimizer` (AIC, GitHub `bhattadiCS/goldenSpheres`) is a **live, tested, mid-buildout product with a near-submission-ready YC application already drafted.** Read §0.1 before you do anything else — building a fresh pitch in ignorance of what already exists there is the single biggest way this session could waste its budget.

---

## How to Run This Prompt

1. **Start your Claude Code session with working directory = `C:\Aditya_Data\Personal\algorithm-x-optimizer`.** That is where the company's actual product, tests, and near-ready YC application live, and where "ready to build out" has to happen. Paste this whole document (it can continue to live in the ResearchThesis repo) as your first message. Give the session **read access to `C:\Aditya_Data\Personal\ResearchThesis`** as well — Module 0 requires reading files there directly (Read/Grep/Glob on an absolute path outside the cwd works fine in Claude Code; you do not need to copy files over).
2. **Read `ResearchThesis\ThesisDocs\July_1_Checkin.md` and this document's §0.1 before Module 1.** The first is the from-scratch walkthrough of the math; the second is what already exists in the repo you're sitting in. Skipping either means re-deriving or re-building something that already exists.
3. **Treat every numeric claim in §0.2 as "reported," not "verified."** Module 0 exists because this document's own research corpus contains internal inconsistencies (§0.3). Resolve them before they reach a pitch deck or a YC application.
4. **Use real agent orchestration, not a simulated one — and use the right model tier for each subagent.** You have a Task/Agent tool for subagent dispatch and, if available, a Workflow-style tool with per-call `model` and `effort` options. Section 1 gives a concrete roster: strategic-judgment and narrative-writing agents run on **Fable 5 at high/max effort**; mechanical fact-finding, reconciliation, and scheduling agents run on **Sonnet 5** (max effort on Sonnet is a good default when you do offload — you're trading model tier for cost/speed, not trying to also trade away care). Don't spend Fable 5's budget re-discovering public facts or re-reading files a cheaper pass can summarize.
5. **Most of your outputs are updates to existing files, not new ones.** `docs/gtm/yc_application.md` and `docs/gtm/deck_outline.md` are close to submission-ready — your job on those is a careful, tracked update, not a rewrite from a blank page. Section 3 is the full manifest, including which older docs need a superseded-banner rather than a rewrite.
6. **Completion criterion (the mission's own stopping rule):** you are done when (a) every file in the Section 3 manifest exists or has been updated, (b) `docs/gtm/yc_application.md` contains real, submittable prose you'd be willing to send with light editing, (c) the Positioning Fork (§0.6) has an explicit, reasoned resolution that every downstream document is consistent with, and (d) Module 6 ends in an explicit **go / no-go / go-with-changes** call. Don't keep researching past the point of marginal value — a little on the nose, given the subject matter, but the point stands.

---

## Section 0: Context & Technical Background

### 0.1 The Sibling Project — AIC / algorithm-x-optimizer (Not Dead — Read Carefully)

An earlier draft of this document described AIC as a failed prior attempt whose "overthinking detection did not work," full stop. That framing is **incomplete enough to be actively misleading.** Here is the corrected picture, current as of 2026-07-02:

**What actually happened:** AIC shipped an "overthinking detector" with an accuracy claim, could not make that claim survive scrutiny against real model traces, and — this is the important part — **the founder killed the feature rather than keep a fabricated headline number.** The commit history has explicit `de-theater:` commits removing a mock interceptor (`random.randint` standing in for a real model call), a fake build toolchain, and fabricated telemetry, replacing them with a real, tested interceptor. This is not a dead project; it's a project that went through a credible honesty pass and is now positioned as a **security/reliability control, explicitly not a cost optimizer and explicitly not an overthinking/model-quality judgment**:

> "We are not claiming to detect 'overthinking' or judge answer quality — that's a model-quality bet we explicitly refuse to make. We detect a loop and enforce a ledger." — `docs/gtm/yc_application.md`

**Current engineering state (verified by direct read, not the repo's own claims):** a real Python + TypeScript SDK with independently-counted **181 Python test functions across 31 files and 26 TypeScript tests across 3 files** (closely matching the repo's own claimed 192+26), a streaming circuit breaker, an atomic cross-call spend ledger (fail-closed, SQLite/Redis backends), and a tamper-evident, hash-chained compliance evidence export mapped to OWASP LLM10 / EU AI Act Art. 12/14 / ISO 42001. A FastAPI+React dashboard exists but is explicitly labeled internal-demo-only, not a product surface. The Rust/eBPF/"Nitro" proxy is explicitly unbuilt and self-documented as such, including a known fail-open defect noted in its own design doc. The cross-customer "fingerprint network" data moat is a plan, not a shipped asset — the repo's own YC application says so plainly.

**Two specific, currently-disabled flags are exactly where the new thesis research is relevant:** `enable_convergence_stop` and `enable_corruption_stop`, in `sdk/python/algorithm_x/interceptor.py` (~line 224–227), both default `False`. The code comment is explicit about why: *"used as a convergence STOP signal only when `enable_convergence_stop` is True (default off until validated on convergent traces)."* That is the exact failure mode the ResearchThesis work fixed — single-trace anecdote replaced by a 75,996-run, out-of-sample-validated statistical model (§0.2). **Nobody has yet checked whether the new validated model actually fixes these specific disabled flags, or whether it would pass the repo's own existing test suite (`tests/test_parity_golden.py`'s `convergence_on`/`corruption_on` cases, `scripts/validate_convergence.py`, the controlled 3-arm benchmark).** This is Module 0's highest-value question — see §0.6 for why it's not a purely technical question.

**A near-submission-ready YC application already exists:** `docs/gtm/yc_application.md` (2026-06-15) is disciplined, evidence-tagged (every claim marked shipped / being-built / explicitly-not-claimed), and has real, specific answers to what-you-do, why-now, unfair-insight, moat, "what happens when a platform ships this natively," and why-you — including acquisition comps (Protect AI→Palo Alto ~$650–700M, Lakera→Check Point ~$300M, CalypsoAI→F5 $180M) and a pre-registered kill/continue criterion for the GTM validation sprint. A current deck (`docs/gtm/deck_outline.md`, 16KB, 2026-06-15) also exists. **Do not draft a competing application from scratch. Update this one.**

**What's stale and needs a supersession banner, not a rewrite-in-place:** `docs/yc_strategy/*` (dated 2026-05-25, i.e. the oldest strategy content in the repo) — `moat_defense_strategy.md`, `final_battle_test_verdict.md`, `differentiation_from_static_routers.md`, `threat_registry_architecture_v2.md`, `market_research_validation.md` — is contradicted by the later honesty pass (aspirational moat claims presented as built, a roleplayed "YC partner verdict" declaring readiness, fabricated-traction slides). `AIC_MATH_AUDIT.md` (2026-06-06) is a rigorous theory-only audit (three proved theorems, an SDE/Inverse-Gaussian derivation, specific calibration numbers) whose premises were later empirically contradicted by the README's honest disclaimer — and it **does not reference the ResearchThesis project at all**, so it's a useful record of which theoretical framing didn't survive contact with real traces, not a source of current numbers. `dow_scan_report.md` and `PROJECT.md` are older fossils of superseded concepts. The pitch deck outline and README already carry explicit superseded/honesty banners; the rest of `docs/yc_strategy/` doesn't yet — that's a concrete action item (§3).

**Also note:** `algorithm-x-optimizer/startup_research_prompt.md` (root, lowercase) is a same-day, less-developed sibling of this exact document. Treat **this** document (in ResearchThesis) as authoritative; consider replacing the other with a short pointer once you've read both, rather than letting two prompt files drift apart.

### 0.2 The Verified Breakthrough (52-cell grid, commit f89ff38, 2026-06-27)

All results below are **full-precision (bf16)**, run via `research/run_experiment_matrix.py` on an RTX PRO 6000 Blackwell (96GB) box, and passed a 79-agent adversarial audit (`research/reports/deep_code_audit.md`) that fixed grader bugs, eliminated train/test leakage (all analysis is now out-of-sample via GroupKFold-by-run-id), and re-graded 165,000 labels (0.33% flip rate). Treat the audit's existence as a feature to cite, not a footnote to hide — "we found our own bugs before a diligence process would have" is a credibility asset, and it rhymes directly with AIC's own de-theater story (§0.1) — both projects have a real "we killed our own bad number" moment. That's a founder-narrative asset worth making explicit in Module 3.

**Scale:** 13 model families × 4 benchmarks (GSM8K, MATH, ARC-Challenge, GPQA) × 3 temperatures = 52 cells, 75,996 total runs, ~760,000 step-level observations.

**Headline findings:**
1. **Overthinking is real and mathematically characterized, not anecdotal.** Accuracy on late-boundary tasks rises, peaks, then decays (corruption) or plateaus (pure token waste) with continued chain-of-thought.
2. **Capability ladder (Qwen family, GSM8K):** boundary step moves monotonically with scale — 0.5B boundary at step 1, 3B at step 4, **7B/14B/32B converge at step 5.** Probe AUC (predicting correctness from observables) rises 0.57→0.96 in-sample, 0.53–0.91 honest out-of-sample.
3. **The boundary is benchmark-dependent — this is a product-scoping fact, not just a research footnote.** The *late, correctable* overthinking pattern (confidently right → corrupts) concentrates on GSM8K (5 families clearly late) and MATH (Qwen-only, real but narrower). On **ARC**, capable models are already right at step 1, so the policy's job degrades to "notice nothing to gain, stop immediately" — still useful (94.63% win rate), just a trivial win rather than a rescue. On **GPQA**, accuracy sits near a hard ceiling (~20–35%) regardless of step count — there is no signal to correct, so the policy's value is again "don't waste tokens on an unwinnable question" (92.05% win rate) rather than "caught a corruption." **This is a feature to pitch, not hide:** the system characterizes *which* of a customer's traffic is even eligible for savings before acting, rather than blindly truncating everything.
4. **Global statistical significance:** across 74,540 completed runs, the `hazard_drift` policy beat "never stop" in 89.59% of cases (95% CI [89.37%, 89.81%], Z ≈ 353.5 from the null of 50%). Stable across temperature 0.1/0.6/1.0 (89.0–89.9%). This is your headline stats-credibility number for a YC application — lead with it.
5. **Near the theoretical online ceiling.** On the 12 cells with a genuine late boundary, the policy captures **50.9% of the achievable oracle-vs-baseline utility gap** — against a decision-theoretic online (no-foresight) upper bound generally cited around 50% (and a Prophet-Inequality-style worst case of 1/e ≈ 36.8%). Translate for non-technical reviewers: an omniscient policy that could see the future would only do about twice as well as yours, and no online policy can see the future.
6. **In-flight (real) deployment, not just post-hoc simulation:** hooking the `hazard_drift` policy directly into the Hugging Face generation loop for Qwen2.5-7B (v=1, c=0, λ=0.05) cut total generated tokens by **54.34%** (827,804 → 377,960) while preserving **91.0%** of baseline accuracy (64.20% vs. 70.53%). Decision latency is a ~5-coefficient logistic-regression dot product — sub-microsecond; see Module 4 for where the *actual* latency risk hides.
7. **Stakes-awareness is a real, controllable dial.** Sweeping the wrong-answer penalty $c$ upward in $\mu_t = [(1 - q_t)\alpha_t - q_t\beta_t](v + c) - \lambda$ pushes the boundary **later** (more patience) — e.g., Qwen-7B's boundary moves from step 1 (c=0) to step 5 (c=1) to step 7 (c=10). This is the literal mechanism behind "let it think longer on a medical/legal query, cut it off fast on a low-stakes chat completion" — a single tunable parameter, not a separate model per vertical.
8. **Prompted vs. distilled reasoning diverge sharply.** Qwen2.5-7B (prompted CoT) has low corruption risk (β = 5.72%) and a comparatively high repair rate (α = 10.94%) — it's worth letting it keep working. DeepSeek-R1-Distill-7B (RL-distilled reasoning) has a **much higher corruption rate (β = 31.49%)** and a boundary that arrives almost immediately — its stopping behavior appears "pre-baked" by RL training, and pushing it past that point is comparatively dangerous. **Product implication:** the two model families need different default policies out of the box.

### 0.3 ⚠️ Known Reconciliation Items — Do Not Skip (This Is Module 0's Job)

Two internal inconsistencies surfaced while preparing this prompt. Do not silently pick a number and move on — resolve them, or explicitly scope which experimental convention each number belongs to, before either goes in front of an investor.

- **Qwen-7B's boundary is reported as both step 5 and step 1.** The capability-ladder narrative (§0.2.2, corroborated across `project-overthinking-state` notes and `July_1_Checkin.md`) says step 5. But `research/outputs/real_traces_bf16_ladder/stakes_sweep_report.md`'s own **c=0 baseline row** for `qwen2p5_7b` shows **T\* = 1** (14B and 32B show T\*=5 at the same c=0 setting in the same table — only 7B looks different). The likely explanation is a differing default stakes/λ convention between the main experiment-matrix pipeline and this later ad-hoc sweep script, not a factual contradiction — **but this must be confirmed by inspecting both scripts' actual default parameters, not assumed.**
- **DeepSeek-R1-Distill's boundary is reported as both step 2 and step 1.** The original draft of this prompt asserted step 2; `research/outputs/real_traces_bf16_ladder/prompted_vs_distilled_report.md` and the cross-family memory note ("Reasoning-distill ... stay early (step 1)") both say step 1. Prefer step 1 unless you find a source that justifies step 2, and note which script/config produced whichever number you cite.
- **`stakes_sweep_report.md`'s own prose contradicts its own data table.** Its "Key Insights" text claims the boundary "shifts earlier"/"to the left" as the penalty $c$ increases. Its data table shows the opposite — the boundary moves later/right as $c$ increases (consistent with §0.2.7 above and with basic incentive logic). Do not repeat the report's prose claim; trust the table.

**Module 0 deliverable:** a short reconciliation memo stating, for every headline number that ends up in the pitch, exactly which script/commit/config produced it. If you can't reconcile a number in the time available, mark it "directionally correct, exact value pending re-verification" rather than asserting false precision.

### 0.4 Seed Competitive & Market Research [gathered via live web search, 2026-07-02 — verify before citing to investors]

**Confirmed gap in the gateway/router market — nobody does adaptive, hazard-based stopping:**

| Player | What it is | Pricing | Overthinking/adaptive-reasoning feature? |
|---|---|---|---|
| LiteLLM | Open-source proxy, self-hosted | Enterprise $250–$2,500/mo | None |
| Portkey | Gateway + observability + guardrails | Was $49/mo Pro | **Acquired by Palo Alto Networks, announced Apr 30 2026**, folded into Prisma AIRS as an agent-security control plane. Notable as M&A validation of the "AI gateway" category, exiting toward security not cost — directly relevant to AIC's own security positioning (§0.1), not just the thesis pitch. |
| Helicone (YC W23) | Observability/gateway | $79–$799/mo | None |
| OpenRouter | Usage-based router | 5.5% fee on credit top-ups | None |
| Cloudflare AI Gateway | Free-tier gateway | Free core; added static **dollar-denominated** spend caps in June 2026 | Confirms the *problem* is acknowledged at the infra layer, but the cap is static. **Already cited in AIC's own `yc_application.md`** as the "naive budget caps are commoditizing" comparison — cross-check before re-researching. |
| Martian | Per-query model router | $9M seed (2023); a "$1.3B valuation" figure circulating Apr 2026 traces only to aggregator sites — **unverified rumor** | None |
| Not Diamond / Unify.ai | Meta-model / benchmark-driven routers | Unify from $99/mo | None |

**Note:** AIC's own `docs/gtm/yc_application.md` and the superseded `docs/yc_strategy/market_research_validation.md` already cover Martian, Not Diamond, RouteLLM, Helicone, Portkey, LiteLLM, LangSmith/Braintrust, LangChain/LangGraph, Vercel AI SDK, and a real, closer clone called **TokenFence** ("cost circuit breaker" tagline). They do **not** cover OpenRouter, Cloudflare AI Gateway's June 2026 spend-cap launch, Unify.ai, or ReasonBlocks. Module 1 (Competitive Landscape) should merge these two research passes, not duplicate the parts that already exist.

**Direct competitive risk in the *thesis* space — read this before writing Module 2:** the academic space is not white space. A 2025 TMLR survey ("Stop Overthinking," arxiv.org/abs/2503.16419) plus at least eight 2025–26 papers (REFRAIN, DVS-LR, S-GRPO, BudgetThinker, SelfBudgeter, TALE, "Stop When Enough" [arxiv.org/abs/2510.10103], MARS, "When More Thinking Hurts") report 20–90% token savings on similar problems. **Differentiation must rest on rigor and deployment, not on being first to notice the phenomenon:** 75,996 runs, out-of-sample validation, an adversarial audit trail, and a real in-generation-loop deployment are the actual moat inputs.

**Closest direct commercial competitor found (to the thesis/overthinking angle specifically):** **ReasonBlocks** (YC Spring 2026, 2-person team) — mid-run reasoning compression/correction via a cross-run "reasoning library," 52% token reduction + a 42% accuracy lift on SWE-Bench Pro. Mechanism differs (cross-run memory on agentic coding vs. single-CoT real-time stopping on reasoning benchmarks), but close enough that **whichever module ends up owning the "overthinking" narrative must contain an explicit, named "how are we different from ReasonBlocks" paragraph.**

**Adjacent funded players (context):** Keywords AI, Lemma, The Context Company, Compresr (YC W26), Oximy (YC W26, AI usage cost governance), CodeIntegrity ($4.8M seed 2026), Guardrails AI ($7.5M). Larger raises signaling category heat: Arize ($70M Series C, Feb 2025), Braintrust ($80M Series B @ $800M valuation, Feb 2026), Langfuse (acquired by ClickHouse, Jan 2026).

**Market sizing hooks:** AI inference market ~$106–118B (2025/26) → ~$254–255B by 2030 (17.5–19.2% CAGR, three independent estimates converge). Gartner: total AI spend >$2.5T in 2026 (+47% YoY); separately, Gartner predicts **>40% of agentic AI projects will be canceled by end of 2027** over escalating cost, unclear value, and inadequate risk controls — a strong "the waste is real and executives already know it" hook. AIC's own application already cites sharper, more specific DoW numbers (Uber's Claude Code budget, $4.2k/$6k/$47k documented runaways, OWASP LLM10, 79% overspend) — **prefer those over generic market-sizing stats when the two overlap.**

### 0.5 Seed YC Mechanics Research [gathered via live web search, 2026-07-02 — verify before acting]

- **Batch cadence changed.** YC expanded from two batches/year to **four** (Winter/Spring/Summer/Fall) starting in 2025. Don't plan around the old two-batch cycle.
- **Current deadline:** Fall 2026 batch on-time applications due **July 27, 2026, 8:00pm PT**; decisions by ~Aug 28; batch runs Oct–Dec 2026 in SF. [UNVERIFIED — re-confirm live at ycombinator.com/apply before treating as load-bearing.]
- **Application format:** what-you'll-build question, a founder-achievement question, a "wildcard" question, equity-split disclosure, no application fee. A ~1-minute founder video is part of the flow; unresolved whether it's mandatory.
- **Possibly relevant to you specifically:** Spring 2026 reportedly added an option to submit a Claude Code/Codex transcript as evidence of AI-assisted building — unconfirmed whether this persists into Fall 2026. If it does, note it explicitly: this whole multi-repo, multi-agent operating session is itself exactly that kind of transcript.
- **Useful comps:** **The Token Company (YC W26)** — solo (18-year-old) founder, LLM prompt-compression API, led the application with a benchmark rather than customers. **Wafer, f.k.a. Herdora (YC S25)** — real research backgrounds, de-risked via paid consulting before productizing. **Helicone (YC W23)** — contrast case: emerged from a product/timing insight, not a research result.
- **YC's own stated interest (Requests for Startups, checked 2026-07-02):** "Inference Chips for Agent Workflows" and "Software for Agents" both currently listed as open, wanted territory. No partner statement found calling this space saturated. YC's broader 2026 messaging suggests fatigue specifically with generic "AI wrapper" apps, as distinct from infrastructure plays — AIC and the thesis work are both in the latter category; make that distinction explicit in the application.

### 0.6 The Central Strategic Fork — Resolve This Before Module 1

AIC's current, near-ready pitch has a specific credibility asset at its climax: *"I killed a feature of my own product because its headline number wasn't real... a founder who fabricates a metric is exactly the founder who would ship a guardrail that fails silently."* The entire "why you" answer is built on having refused to make an unvalidated model-quality bet.

The ResearchThesis work has now independently, rigorously validated a version of *exactly that bet* — 75,996 runs, out-of-sample, adversarially audited. That's genuinely good news scientifically. It is **not automatically good news for AIC's pitch**, and treating it as a simple merge is the most likely way this session produces something worse than what already exists. Three options, in increasing order of risk:

- **(A) Keep fully separate.** The thesis research becomes its own pitch/company (roughly what the original, un-corrected version of this prompt assumed), and AIC's existing application is untouched. Cleanest story for each, zero risk to AIC's existing credibility asset — but two half-built things instead of one strong one, and a solo founder splitting 25 days across two applications.
- **(B) Cautious, clearly-labeled opt-in upgrade (recommended starting lean).** Flip `enable_convergence_stop`/`enable_corruption_stop` from "default-off, unvalidated research" to "default-off, but available, backed by a published-quality 75,996-run out-of-sample study" — a v2/opt-in efficiency feature sitting alongside, not replacing, the core mechanical DoW story. The founder-story climax stays true ("I killed it when it *wasn't* validated" — now: "and here's the rigorous, independent validation that changes that, which is why it's back as an opt-in, not a default"). Lowest risk to the existing application's credibility, real product upside.
- **(C) Full repositioning.** Lead the pitch with both the mechanical DoW story *and* the validated efficiency story as co-equal pillars. Highest upside if it lands, highest risk of diluting the sharp, disciplined positioning that makes the current application unusually credible for a pre-revenue solo-founder pitch.

**Required action:** an agent running at Fable 5 / max effort must read both repos' full context and explicitly decide among (A)/(B)/(C) — or a variant — with stated reasoning, before Module 2 (moat) or Module 3 (YC application) proceeds. Every downstream module must be consistent with whatever this agent decides. Do not let different modules silently assume different answers.

---

## Section 1: Agent Orchestration Plan (Required Execution Structure)

Do not answer the modules below as a single linear chat response. Decompose into named subagents with explicit dependencies and **explicit model/effort tiers** — spend Fable 5 on judgment and prose that will be read by a YC partner or that decides company strategy; spend Sonnet 5 on fact-finding, reconciliation, and scheduling that a careful-but-cheaper pass handles fine.

| # | Agent | Mission | Model | Effort | Depends on | Output |
|---|---|---|---|---|---|---|
| 0a | Research Ground-Truth Verifier | Resolve §0.3; re-derive current headline numbers from `research/outputs/real_traces_bf16_ladder/` and the underlying scripts | Sonnet 5 | high | — | `docs/startup_plan/02_research_ground_truth.md` |
| 0b | AIC State Auditor | Catalogue what's shipped/being-built/stale in algorithm-x-optimizer; list which `docs/yc_strategy/*` files need superseded banners | Sonnet 5 | high | — | `docs/startup_plan/03_aic_state_audit.md` |
| 0c | **Positioning Fork Resolver** | Resolve §0.6 (A/B/C) with full reasoning, reading both repos | **Fable 5** | **max** | — | `docs/startup_plan/01_positioning_fork_decision.md` |
| 1 | Competitive Landscape | Merge §0.4 with AIC's existing market research; fill the confirmed gaps (OpenRouter, Cloudflare's June cap launch, Unify.ai, ReasonBlocks) | Sonnet 5 | high | 0a,0b,0c | `docs/startup_plan/04_competitive_landscape.md` |
| 2 | **Technical Synthesis** | Does the validated hazard-drift model actually fix `enable_convergence_stop`/`enable_corruption_stop`? Attempt to re-score AIC's own stored traces/telemetry against the new methodology; check whether it would pass `test_parity_golden.py`'s convergence/corruption cases | **Fable 5** (design/judgment) — delegate the actual scripting/replay to a nested Sonnet 5 sub-task if your harness supports it | **max** | 0a,0b,0c | `docs/startup_plan/05_technical_synthesis_convergence_stop.md` |
| 3 | Product & Architecture | Module 1 + Module 4 | Fable 5 | high | 0c, 2 | `docs/startup_plan/06_product_and_architecture.md` |
| 4 | Moat & Defensibility | Module 2 (rewrite, supersedes `moat_defense_strategy.md`) | Fable 5 | max | 0c, 1, 2 | `docs/startup_plan/07_moat_and_defensibility.md` |
| 5 | Business Model & GTM | Pricing/packaging, building on `yc_application.md`'s existing pricing logic | Fable 5 | high | 0c, 1 | `docs/startup_plan/08_business_model.md` |
| 6 | **YC Application Updater** | Update `docs/gtm/yc_application.md` and `docs/gtm/deck_outline.md` **in place**, per the Fork decision | **Fable 5** | **max** | 0c, 1, 2, 3, 4, 5 | `docs/gtm/yc_application.md`, `docs/gtm/deck_outline.md` (diffs, not replacements) |
| 7 | Red Team / Bear Case | Module 5 — argue against the Fork decision too, not just the product | Fable 5 | max | 3, 4, 5, 6 | `docs/startup_plan/09_red_team_bear_case.md` |
| 8 | 90-Day Roadmap | Module 6, dated checklist anchored on the real deadline | Sonnet 5 | high | 6, 7 | `docs/startup_plan/10_90_day_roadmap.md` |
| 9 | Executive Synthesis | One navigable summary + explicit go/no-go/go-with-changes call | Fable 5 | max | all | `docs/startup_plan/00_EXECUTIVE_SUMMARY.md` |

**Phasing:**

```
Phase 1 (parallel):  0a (Sonnet/high)  +  0b (Sonnet/high)  +  0c (Fable/max)
Phase 2 (parallel):  1 (Sonnet/high)  +  2 (Fable/max)  +  3 (Fable/high)  +  4 (Fable/max)  +  5 (Fable/high)
Phase 3 (parallel):  6 (Fable/max)  +  7 (Fable/max)
Phase 4 (sequential): 8 (Sonnet/high)  ->  9 (Fable/max)
```

If your harness exposes a `Workflow`-style tool with `parallel()`/`pipeline()` and per-call `model`/`effort` options, this maps directly, e.g.:

```js
phase('Phase 1')
const [groundTruth, aicState, fork] = await parallel([
  () => agent(GROUND_TRUTH_PROMPT, {model: 'claude-sonnet-5', effort: 'high'}),
  () => agent(AIC_AUDIT_PROMPT,    {model: 'claude-sonnet-5', effort: 'high'}),
  () => agent(FORK_PROMPT,         {model: 'claude-fable-5',  effort: 'max'}),
])

phase('Phase 2')
const [competitive, synthesis, product, moat, gtm] = await parallel([
  () => agent(COMPETITIVE_PROMPT(groundTruth, aicState, fork), {model: 'claude-sonnet-5', effort: 'high'}),
  () => agent(SYNTHESIS_PROMPT(groundTruth, aicState, fork),   {model: 'claude-fable-5',  effort: 'max'}),
  // ...product, moat (max), gtm
])
// Phase 3, Phase 4 follow the same pattern — see the roster table for exact model/effort per agent.
```

If your harness doesn't expose per-call model selection, fall back to dispatching the Sonnet-tier agents (0a, 0b, 1, 8) as plain Task calls from whatever cheaper mode is available, and reserve the Fable-5 session itself for the rows marked Fable above. **Every subagent prompt must be self-contained** — include the relevant facts from Section 0 inline, state which file to write or update, and tell it not to duplicate research another agent already owns.

---

## Section 2: The Research & Planning Mission — Modules

### Module 0: Ground-Truth Verification (Agents 0a/0b/0c — do this first)
1. Resolve both items in §0.3 with sourced, exact citations.
2. Catalogue AIC's real vs. stale state per §0.1; list every `docs/yc_strategy/*` file needing a superseded banner.
3. Resolve the Positioning Fork (§0.6) with explicit reasoning. This gates Modules 2 and 3.

### Module 1: Product-Market Fit & Product Offering
1. **The core product wrapper**, evaluated against AIC's *actual* shipped integration point (the LiteLLM `async_post_call_streaming_iterator_hook`, per `yc_application.md`), not a hypothetical one:
   - *Option A:* API Gateway Proxy — this is closest to what's already shipped.
   - *Option B:* Client-side SDK — already exists in Python/TypeScript; the question is whether the new stopping model is a config flag on the existing SDK or a separate product.
   - *Option C:* In-flight orchestrator for private vLLM/TensorRT-LLM deployments — best supported by §0.2.6's real in-loop deployment result, but currently unbuilt in AIC (the Rust/Nitro proxy is explicitly quarantined). Make the build-vs-defer case explicitly.
2. **The "traffic classifier" surface.** §0.2.3 established the boundary is benchmark/difficulty-dependent. Design the feature that detects whether a request is in the "correctable late-boundary" regime, the "already solved, stop immediately" regime, or the "capped, no signal" regime — and what the product does differently in each.
3. **Product tiers.** How does "stakes sweeping" (§0.2.7) become a packaged control — e.g., a per-endpoint or per-request-tag $c$ value, with sane defaults for "medical/legal/financial" vs. "chatbot small talk"? Reconcile with AIC's existing security-buyer pricing logic (per-protected-agent meter, not % of model bill) rather than proposing a conflicting model.
4. **Pricing model**, benchmarked against §0.4's confirmed competitor pricing and AIC's existing pricing brief.

### Module 2: The Moat & Defensibility
1. **Rewrite, don't append** — this module supersedes `docs/yc_strategy/moat_defense_strategy.md`. Ground every claim in what's *actually* built (fingerprint store status per §0.1, evidence chain, 200+ tests) plus whatever the Fork Resolver (0c) decided about the thesis research's role.
2. The required ReasonBlocks differentiation paragraph (§0.4), if the Fork decision keeps or reintroduces the overthinking angle at all.
3. The "Frontier Lab Obsolescence" threat: if OpenAI/Anthropic ship native early-stopping, or LiteLLM/Cloudflare ship the full DoW control natively — note AIC's application *already has a strong, specific answer to this* (§0.1); extend it, don't replace it.
4. Academic-crowding response: the one-paragraph answer to "isn't this just [paper X]?", grounded in scale/validation/audit trail — and consistent with whatever the Fork decided about whether this argument is even part of AIC's pitch.

### Module 3: Y Combinator Seed Pitch & Metrics
1. **This is an update to `docs/gtm/yc_application.md`, not a fresh draft.** Read it in full first. It already has strong, specific, evidence-disciplined answers — your job is to determine exactly which answers change given the Fork decision (most likely candidates: the product description's shipped/being-built table, "unfair insight," and "moat" — least likely: "why you," which is founder-story and doesn't depend on this research) and make surgical updates, preserving its existing tone and rigor discipline (the shipped/being-built/not-claimed table pattern — reuse it, don't invent a new format).
2. Fold in the strongest thesis stats (§0.2.4, §0.2.6) only where the Fork decision says the thesis narrative belongs in this pitch at all.
3. Update `docs/gtm/deck_outline.md` to match.
4. Re-verify the application format and deadline (§0.5) live before finalizing anything date-dependent.

### Module 4: High-Throughput System Architecture
1. **Latency budget — be specific about where the risk actually is.** The logistic regression itself (§0.2.6) is a near-free dot product. The real risk is **feature extraction**: `entropy_mean` requires a softmax over the full vocabulary at every generated token, and `hidden_l2_shift` requires diffing hidden states across steps. Benchmark these, not the regression.
2. Should the wrapper compile to ONNX inside AIC's (currently unbuilt) Rust/Nitro proxy, or is a Python interceptor extension — i.e., adding this as a third detector alongside the existing convergence/corruption stop logic in `interceptor.py` — sufficient given the regression's near-zero cost? The lower-risk path is almost certainly extending the existing, tested Python interceptor rather than reviving the quarantined Rust proxy on a 25-day clock.
3. Concrete build milestones: (a) extract the three trained logistic-regression models as standalone artifacts; (b) determine whether they can directly replace the `running_stability >= convergence_patience` heuristic at `interceptor.py:738` or need new hook points; (c) get `test_parity_golden.py`'s `convergence_on`/`corruption_on` cases passing against the new model; (d) benchmark real per-token overhead under load; (e) dogfood on this repo's own `aic_telemetry.jsonl` / benchmark harness as the first "customer."

### Module 5: Red Team — The Bear Case
Argue the strongest case against this being fundable/buildable right now, including against the Fork decision itself, then rebut honestly (concede where you can't):
1. Does re-introducing any version of the overthinking claim risk the exact credibility AIC's "why you" answer is built on?
2. The academic space is crowded (§0.4) — is "better validated" a defensible moat once a well-funded competitor replicates the method?
3. If the in-loop result (§0.2.6) is Qwen-7B-specific, how much per-model-family engineering closes the gap to "sellable," and does that cost undercut a "10-minute integration" pitch?
4. Enterprise security-buyer sales cycles run 6–12 months (this is explicitly AIC's own buyer profile) — compatible with a solo founder's runway and a 25-day application clock?
5. Pre-mortem: what would have to be true in 12 months for this to have clearly been the wrong call?

### Module 6: 90-Day Actionable Roadmap
Anchor every date on the real, re-verified YC deadline. Produce a dated checklist:
- [ ] **T-minus N days:** Module 0 complete (ground truth + AIC audit + Fork decision).
- [ ] **T-minus N-x days:** Module 2 (Technical Synthesis) has a real answer on whether the new stopping model passes AIC's own test suite — even a partial, one-model-family result.
- [ ] **T-minus N-y days:** `docs/gtm/yc_application.md` and deck updated and internally consistent with the Fork decision; red-team pass incorporated as objection-handling.
- [ ] **Application deadline (confirm live date):** submit.
- [ ] **Decision date (~Aug 28, confirm live):** contingency for both outcomes.
- [ ] Explicit **go / no-go / go-with-changes** call, informed by Module 5.

---

## Section 3: Output Manifest

All new files live under `algorithm-x-optimizer/docs/startup_plan/`. Two files are **in-place updates**, not new files — treat them as diffs against real, already-good content:

| File | Agent | Type | Must contain |
|---|---|---|---|
| `docs/startup_plan/00_EXECUTIVE_SUMMARY.md` | 9 | new | Navigable summary + explicit go/no-go call |
| `docs/startup_plan/01_positioning_fork_decision.md` | 0c | new | A/B/C decision (or variant) with full reasoning |
| `docs/startup_plan/02_research_ground_truth.md` | 0a | new | §0.3 resolved; citable-numbers table |
| `docs/startup_plan/03_aic_state_audit.md` | 0b | new | Shipped/stale catalogue; superseded-banner list |
| `docs/startup_plan/04_competitive_landscape.md` | 1 | new | Merged §0.4 + AIC's existing research, gaps filled |
| `docs/startup_plan/05_technical_synthesis_convergence_stop.md` | 2 | new | Does the new model fix the disabled flags — with evidence, not just argument |
| `docs/startup_plan/06_product_and_architecture.md` | 3 | new | Module 1 + 4, wrapper decision explicitly made |
| `docs/startup_plan/07_moat_and_defensibility.md` | 4 | new | Supersedes `moat_defense_strategy.md` |
| `docs/startup_plan/08_business_model.md` | 5 | new | Pricing decision, reconciled with existing pricing brief |
| `docs/gtm/yc_application.md` | 6 | **update in place** | Surgical edits consistent with the Fork decision, existing tone/rigor preserved |
| `docs/gtm/deck_outline.md` | 6 | **update in place** | Matches the updated application |
| `docs/startup_plan/09_red_team_bear_case.md` | 7 | new | Honest, including any point not fully rebuttable |
| `docs/startup_plan/10_90_day_roadmap.md` | 8 | new | Dated checklist |

**Also required (part of Agent 0b's mandate):** add a `⚠️ SUPERSEDED` banner (matching the existing pattern in `docs/gtm/yc_strategy/yc_pitch_deck_outline.md` and `README.md`) to `moat_defense_strategy.md`, `final_battle_test_verdict.md`, `differentiation_from_static_routers.md`, `threat_registry_architecture_v2.md`, `market_research_validation.md`, `dow_scan_report.md`, `AIC_MATH_AUDIT.md`, and `PROJECT.md` — each pointing to whatever supersedes it.

---

## Section 4: Operating Principles

- **Be mathematically precise, systems-minded, and highly strategic.** Every claim should trace back to the predictable hazard-drift math (§0.2) or a specific, sourced result from the live repos, not a stale reference to any single earlier experiment.
- **Reuse AIC's own rigor convention.** `docs/gtm/yc_application.md` already established a shipped / being-built / explicitly-not-claimed table format. Use it for any new claim rather than inventing a different rigor-signaling style — consistency here is itself part of the credibility story.
- **Cite sources inline** for external claims, with a plain **[UNVERIFIED]** tag for anything not independently confirmed. A pitch that states an unverified figure as fact is worse than one that says "directionally X, confirming exact figure."
- **No placeholders.** If an agent can't complete its section, it should say what's missing and why, not fill the space with generic advice.
- **Self-review before declaring done:** re-read the Section 3 manifest against what actually exists on disk, re-read the Fork decision, re-read Module 6's completion criterion. Only then produce the executive summary and the go/no-go call.
