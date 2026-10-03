"""Rebuild exactly 25 editable-slide content records and their speaker notes.

This generates presentation content, not a PPTX renderer. Numerical evidence is
read from the frozen analysis outputs. Missing prospective outcomes are explicitly
pending. PowerPoint authoring/export/visual QA is a separate build step.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EVIDENCE = ROOT / "research/outputs/thesis_v1/evidence"
ONLINE = ROOT / "research/outputs/semester2/online_stopping_20261002"
MODEL = ROOT / "research/outputs/semester2/prefix_model_v1"


def records(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_if_exists(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def percent(value, digits=2):
    return f"{100 * value:.{digits}f}%"


def add(slides, title, kicker, bullets, seconds, notes, sources, **extra):
    slides.append({
        "number": len(slides) + 1, "title": title, "kicker": kicker, "bullets": bullets,
        "seconds": seconds, "speaker_notes": notes.strip(),
        "sources": sources, **extra,
    })


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    slides = []
    boundaries = records(EVIDENCE / "boundary_domain_step_metrics.csv")
    cells = records(EVIDENCE / "boundary_selected_cell_step_metrics.csv")
    effects = records(EVIDENCE / "algorithm_v2_normalized_effects.csv")
    summary = records(EVIDENCE / "tournament_balanced_summary.csv")
    learned = load_if_exists(MODEL / "evaluation.json")
    protocol = load_if_exists(MODEL / "training_protocol.json")
    live = load_if_exists(ONLINE / "live_metrics.json")
    learned_live = load_if_exists(ONLINE / "learned_main/live_metrics.json")
    trap = load_if_exists(ONLINE / "adversarial_live/live_metrics.json")
    learned_trap = load_if_exists(ONLINE / "learned_adversarial/live_metrics.json")
    learned_main_ci = load_if_exists(ONLINE / "learned_main/live_uncertainty.json")
    learned_trap_ci = load_if_exists(ONLINE / "learned_adversarial/live_uncertainty.json")
    learned_latency = load_if_exists(ONLINE / "learned_main/learned_latency_summary.json")
    controller_latency = load_if_exists(ONLINE / "latency_summary.json")
    math_source = "research/mathematical_foundations.md"
    stopping_refs = [
        {"title": "Ferguson, Optimal Stopping and Applications, finite horizon", "url": "https://www.math.ucla.edu/~tom/Stopping/sr3.pdf"},
        {"title": "Peskir and Shiryaev, Optimal Stopping and Free-Boundary Problems (2006)", "url": "https://doi.org/10.1007/978-3-7643-7390-0"},
    ]
    add(slides,
        "Cost aware stopping boundaries in reasoning language models",
        "Master's thesis defense • Applied and Computational Mathematics",
        ["Aditya Bhatt • Johns Hopkins University", "Research adviser: Dr. Zerotti Woods", "Second reader: Dr. Moustapha Pemy", "Research evidence frozen October 2026"],
        45,
        """My thesis studies a decision that a reasoning system must make repeatedly: after the current answer, is another response worth its computation? The study concerns complete response-and-revision increments. It does not observe every internal thought or locate an optimal token inside a response. I will first establish the mathematical decision problem, then examine the two frozen experimental corpora, and finally show the implementable causal stopper and its measured limits. The key separation throughout is between information available at a decision and information visible only after a full trajectory. These are research findings and a defense package; institutional approval and degree completion require the committee's actual review.""",
        ["ThesisDocs/Masters_Thesis_Draft_v1.md", "ThesisDocs/ADVISOR_MEETING_SEMESTER_2_ROADMAP.md"],
        layout="cover")
    add(slides, "When is another reasoning increment worth its cost?",
        "Research question",
        ["Repair: an incorrect candidate becomes correct.", "Corruption: a correct candidate becomes incorrect.", "Unproductive continuation: benefit falls below declared cost.", "A causal stop uses the observed prefix and a declared answer selector."],
        55,
        """A final-answer score hides important changes along a revision trajectory. A model can repair an initially wrong answer, and it can replace a correct answer with a wrong revision. I use repair and corruption as changes in externally graded candidates, not as claims about hidden cognition. There is also a separate economic outcome: continuation can be unproductive even if average accuracy still increases, because the increase may be smaller than the cost. The first research question is where this happens in the recorded panels. The second is what a policy can infer from a permitted prefix. The third is whether actual generation can be avoided while retaining an acceptable accuracy trade-off. The thesis evaluates these separately. Neither a high classifier AUC nor a fast controller call establishes that the system preserves accuracy.""",
        ["ThesisDocs/chapters/chapter1_intro.md", math_source])
    add(slides, "Two corpora answer different questions",
        "Experimental scope",
        ["The corpora overlap; their counts must not be added.", "GSM8K, MATH, ARC-Challenge and GPQA are the recorded benchmarks."],
        65,
        """The canonical revision matrix and the standardized detector corpus have different observation rules and denominators. The first supports the boundary and matched estimator analyses. Its 75,965 observations here are trajectories, not reasoning steps; the corresponding raw step table contains 798,770 rows. The second supports five-step detector evaluation and contains 28,888 trajectories, or 144,440 rows. Both have 52 model-domain cells, but the standardized selection replaces InternLM with Qwen3.5-9B. Their task sets overlap, so summing the task or trajectory counts would double count evidence. A table or claim must identify its own corpus. GPQA also has a recorded metadata-versus-loader split discrepancy; the manuscript retains that provenance limitation. No SVAMP result is claimed in this selected corpus.""",
        ["data_manifest_v1.json", "ThesisDocs/chapters/chapter3_methodology.md"],
        table={"headers": ["Quantity", "Revision matrix", "Detector corpus"], "rows": [
            ["Model-domain cells", "52", "52"],
            ["Model configurations", "13", "13 (different selection)"],
            ["Trajectories", "75,965", "28,888"],
            ["Step rows", "798,770 raw", "144,440"],
            ["Unique tasks", "1,948", "2,948"],
            ["Horizon", "8 / 10 / 14 by domain", "5 in every cell"],
        ]})
    add(slides, "The observation unit is a complete revision response",
        "Generation protocol",
        ["One increment produces a thought, candidate answer and telemetry.", "The next prompt contains the previously observed response history.", "The floor is two increments; it is a protocol constraint.", "Historical temperatures are repeated settings of one recorded seed."],
        75,
        """The repeated-response experiment supplies a concrete point at which to observe a candidate and decide whether to schedule another response. A revision is therefore not merely another token in one uninterrupted internal chain. The archive records candidate answers, parse outcomes, token counts and additional instrumentation. In the live adapter, the required JSON response boundary is shared by active and full-horizon arms. Both enforce a floor of two and maximum of five. The floor protects the declared task protocol; it is not a theorem that two increments are universally necessary or sufficient. The historical panel uses seed seven at three temperatures. Those are three temperature settings, not independent generation seeds. A controller can stop future calls only after the currently observed increment completes. Within-call cancellation is a separate engineering control.""",
        ["ThesisDocs/chapters/chapter3_methodology.md", "research/online_generation.py", "research/online_stopping_controller.py"],
        table={"headers": ["Setting", "Declared value"], "rows": [
            ["Historical seed / temperatures", "7 / 0.1, 0.6, 1.0"],
            ["Live generator", "Local Qwen2.5-0.5B, greedy"],
            ["Live completion cap", "128 tokens per response"],
            ["Live floor / horizon", "2 / 5 responses"],
        ]})
    add(slides, "Correctness is hidden from the stopping policy",
        "Filtration and admissibility",
        ["Fₜ contains declared decision information and completed prefix observations.", "Cₜ is the selected candidate's correctness against an unobserved reference.", "qₜ = E[Cₜ | Fₜ] is a conditional probability, not a known label.", "τ is admissible when {τ ≤ t} belongs to Fₜ; m ≤ τ ≤ N."],
        70,
        """The filtration formalizes the declared decision information. It grows when a response or a probe finishes. Offline correctness labels evaluate execution and are not supplied to the runtime. The candidate selector is part of the policy contract. The default carries forward the latest nonempty available answer; baseline and learned arms use it. The heuristic can retain the previous answer on a confidence drop or the best observed valid confidence on wobble. Neither branch triggered in the reported runs. C denotes correctness of the selected candidate, and q is its theoretical conditional expectation; reported confidence is not automatically q. A stopping time is admissible when deciding to stop by time t uses information present by t. A full-trajectory score or unseen future answer violates that declared interface. Mathematical measurability is a separate assumption: a known deterministic reference can make q equal C even without a gold field.""",
        [math_source, "research/online_stopping_controller.py"],
        table={"headers": ["Runtime permits", "Evaluation keeps separate"], "rows": [
            ["Current / prior answer and thought", "Reference answer and correctness"],
            ["Strict parsing and observed confidence", "Future response and future label"],
            ["Charged tokens and public domain", "Bidirectional full-trajectory score"],
        ]})
    add(slides, "Accuracy, utility and computation are separate outcomes",
        "Objective and accounting",
        ["Gₜ = qₜ − Kₜ; maximize E[Gτ] over admissible stops.", "A response-cost example uses Kₜ = 0.05(t − 1).", "Affine stakes give Gₜ = (v + p)qₜ − p − Kₜ.", "Completion tokens, repeated prompts, peers and wall time need separate ledgers."],
        65,
        """The cost-sensitive reward is expected correctness minus cumulative computation cost. The cost process is adapted, integrable and nondecreasing in the natural computation setting. A unit response penalty of point zero five is a declared preference: it is not a measured price of every response. The token-weighted analyses use their separately declared scaling, and actual completion-token savings do not inherit that scaling as a dollar or energy saving. For a correct answer worth v and an incorrect answer penalized by p, the coefficient of correctness becomes v plus p. Under a fixed potential process, unchanged prompts, selector and nondecreasing costs, higher stakes produce nested earliest optimal stopping sets. That corollary does not apply when changing stakes also changes generation. The empirical report therefore presents accuracy, utility, generated tokens, prompt work and time side by side rather than converting one into all the others.""",
        [math_source, "ThesisDocs/chapters/chapter5_online.md"])
    add(slides, "Conditional repair and corruption yield the exact drift",
        "Binary correctness identity",
        ["αₜ = P(Cₜ₊₁ = 1 | Cₜ = 0, Fₜ).", "βₜ = P(Cₜ₊₁ = 0 | Cₜ = 1, Fₜ).", "μₜ = (1 − qₜ)αₜ − qₜβₜ − cₜ.", "cₜ = E[Kₜ₊₁ − Kₜ | Fₜ]; zero denominators have zero mass."],
        80,
        """The drift identity follows by separating the only two transitions that change a binary correctness label. The increment C next minus C current equals the repair indicator minus the corruption indicator. Conditional expectation given the current observed information gives the repair joint mass minus the corruption joint mass. Factoring these masses gives one minus q times alpha and q times beta. Subtract the expected next cost to obtain mu. The conditional probabilities are defined through these joint masses, so an event with zero conditional probability contributes zero rather than an undefined ratio. This resolves a common denominator error: repair rate divides by currently incorrect candidates, whereas corruption rate divides by currently correct candidates. Subtracting those two conditional rates directly would not equal population accuracy drift. With affine stakes, the accuracy part is multiplied by v plus p, then the same incremental cost is subtracted. This identity is exact; estimating its terms is a separate statistical problem.""",
        [math_source], references=stopping_refs,
        equation="E[Gₜ₊₁ − Gₜ | Fₜ] = (1 − qₜ)αₜ − qₜβₜ − cₜ")
    add(slides, "Bellman continuation value defines the optimal stop",
        "Finite-horizon Snell envelope",
        ["S_N = G_N; Sₜ = max{Gₜ, E[Sₜ₊₁ | Fₜ]}.", "τ* = first t ≥ m with Sₜ = Gₜ.", "Δ*ₜ = μₜ + E[Sₜ₊₁ − Gₜ₊₁ | Fₜ] ≥ μₜ.", "Nonpositive immediate drift alone does not imply an optimal stop."],
        105,
        """The finite-horizon theorem is the main optimality result. Start at the terminal reward, then recursively compare stopping now with the conditional value of continuing one step and following an optimal later policy. Backward induction makes S adapted and integrable, shows it dominates G, and makes it a supermartingale. Any integrable supermartingale dominating G must dominate S by the same backward induction, so S is the smallest such process. Optional stopping in the finite bounded horizon gives an upper bound S at the current time for every admissible stopping reward. Before the first contact with G, the recursion uses conditional continuation with equality. The Snell process stopped at that first contact is therefore a martingale from the chosen starting time. At contact its value equals the reward, so that stopping rule attains the upper bound. The displayed continuation advantage subtracts G now from expected S next. It splits into immediate drift plus a nonnegative future option value. A positive drift forces continuation in this reward model. A nonpositive drift can still be outweighed by that future value.""",
        [math_source, "research/tests/test_mathematical_foundations.py"], references=stopping_refs,
        equation="Sₜ = max(Gₜ, E[Sₜ₊₁ | Fₜ])")
    add(slides, "A first drift crossing is valid under persistent signs",
        "Sufficient condition, not a universal law",
        ["τμ = first admissible time with μₜ ≤ 0, otherwise N.", "If all later conditional drifts stay nonpositive pathwise, τμ is optimal.", "Proof: telescope the stopped drift sum before and after the crossing.", "A declining population mean is weaker than this assumption."],
        95,
        """The boundary result is deliberately narrower than the Bellman theorem. Suppose the conditional drift is positive before its first nonpositive value and remains nonpositive afterward along every admissible realized path. For any competing stop, decompose the expected reward difference using the Doob decomposition and the bounded stopping times. If the competitor stops before the crossing, it omits positive conditional drifts. If it stops after the crossing, it adds only nonpositive conditional drifts. Both differences are nonnegative in favor of the first crossing. This proves optimality under the persistent-sign condition. One sufficient structural pattern is increasing q, decreasing repair hazard, increasing corruption hazard, and constant or nondecreasing predictable cost, all pathwise. These are strong hypotheses, not claims established for every language model. Average negative drift at every time still permits an adaptive policy to exploit subgroups. The source includes a second exact counterexample where all population drifts are negative yet observation-dependent continuation improves value. The empirical crossing curves are descriptive unless the required conditional structure is established.""",
        [math_source, "research/tests/test_mathematical_foundations.py"],
        equation="E[Gσ − Gτ] = Σₛ E[(1{σ>s} − 1{τ>s}) μₛ]")
    add(slides, "Delayed repair defeats a myopic stopping rule",
        "Exact counterexample",
        ["q₀ = 0, q₁ = 0, q₂ = 1; each continuation costs 0.10.", "Immediate drift at t = 0 is −0.10.", "Stopping now earns 0; continuing to t = 2 earns 0.80.", "The missed future repair is a continuation option."],
        95,
        """This small deterministic example is enough to disprove a universal first-negative-drift rule. The first extra response remains wrong. The second extra response produces the correct answer. Each response costs one tenth. A one-step policy at the first time sees no correctness improvement and a positive cost, so it stops and receives zero. A policy that pays for both responses earns one minus two tenths, or point eight. The Bellman recursion also gives point eight at the first time because it carries forward the future repair opportunity. This example is realizable with candidates judged against one fixed reference answer; it is not an impossible collection of unrelated marginal probabilities. The mathematical source supplies that realization and independently enumerated verification. The conclusion is about scope: a learned estimate of immediate drift can be causally valid and still fail to be globally optimal. Empirical success of a drift heuristic does not remove this counterexample. Approximating Bellman value would require a transition model or direct continuation-value learning with appropriate validation.""",
        [math_source, "research/tests/test_mathematical_foundations.py"],
        table={"headers": ["Time", "Correctness q", "Cumulative cost", "Stop reward"], "rows": [
            ["0", "0", "0.00", "0.00"], ["1", "0", "0.10", "−0.10"], ["2", "1", "0.20", "0.80"],
        ]})
    add(slides, "A ranking score is not a calibrated probability",
        "Prediction assumptions",
        ["AUC measures ordering, not agreement with outcome frequencies.", "Class weighting changes the fitted posterior unless corrected.", "A small set of prefix features need not be a sufficient state.", "Independent task calibration supports marginal checks, not certainty at every prefix."],
        70,
        """The theory requires conditional probabilities under the runtime information filtration. A fitted score is an estimator, and a feature-restricted model generally estimates a coarser target. High AUC shows ranking discrimination; it does not show that a score of point nine corresponds to ninety percent correctness. Weighted logistic loss changes the population optimum: with class weights a and b, its probability is a times q divided by a times q plus b times one minus q. The original posterior can be recovered algebraically only under that loss-and-population model or calibrated with independent representative labels. The new portable predictor avoids class weighting and uses independent task groups for Platt calibration. Even then, Brier score and reliability bins are finite-sample marginal diagnostics. They do not prove conditional calibration on every prefix or transfer to a changed prompt. Treating the public domain and a few current and past response statistics as a Markov state would require an additional transition-sufficiency assumption; the thesis does not assert it.""",
        [math_source, "research/prefix_stopping_model.py", "research/outputs/semester2/prefix_model_v1/evaluation.json"])
    add(slides, "Repeated checking needs a valid sequential target",
        "Uncertainty guarantees",
        ["A simultaneous upper bound can control false negative-drift stops.", "Its coverage must hold over all decision times for the actual target.", "Conditional probe independence and pre-probe information must be stated.", "Population diagnostics do not certify one task's unseen continuation."],
        65,
        """A sign certificate has a precise form. If upper bounds cover the true conditional drift simultaneously at all admissible decision times, stopping when the upper bound is nonpositive controls stopping on a positive true drift by the coverage error. It does not certify Bellman optimality or zero accuracy loss. A Hoeffding construction requires a named pre-probe sigma-field, bounded independent probes given that field, and a conditional mean target. Once probes are observed, the policy's information changes. A bound on the pre-probe mean must not be relabeled as a bound on the post-probe drift without a comparison argument. Similarly, a nonnegative e-process needs conditional supermartingale increments under its declared null, not just favorable unconditional averages. The source proves the elementary claim and supplies a shared-sign dependence counterexample. The repository's across-task empirical-Bayes or e-process diagnostics can describe a population panel; they cannot automatically issue a safe live certificate for one question.""",
        [math_source],
        references=[
            {"title": "Hoeffding (1963), Probability inequalities for sums of bounded random variables", "url": "https://doi.org/10.1080/01621459.1963.10500830"},
            {"title": "Howard et al. (2021), Time-uniform, nonparametric, nonasymptotic confidence sequences", "url": "https://doi.org/10.1214/20-AOS1991"},
        ])
    add(slides, "The freeze makes claims traceable",
        "Reproducibility and provenance",
        ["Exact bytes and canonical LF content have separate hashes.", "Corpus identity, exclusions, partitions and analysis units are explicit.", "Generated proofs are verified by exact finite-system enumeration.", "A hash records identity; it does not certify truth or representativeness."],
        60,
        """The reproducibility freeze binds each reported number to a selected source and a documented transformation. On this Windows checkout, canonical LF content and exact local file bytes can have different hashes, so the manifest records both and checks the intended rule. That solves line-ending identity; it does not excuse other changes. The mathematical tests compare Bellman values against independently enumerated admissible policies in 4,374 finite reward trees and check stake ordering in another 2,187 bounded systems. These are meaningful verification of identities and implementations, not empirical evidence about language models. The new prefix model has a serialized coefficient order, scaler, calibrators, source hashes, task roles and row-level label reconstruction audit. The original archived corpus remains unchanged. A preserved copy of the baseline freeze used in training prevents a later manifest refresh from obscuring the training inputs. Missing historical environment details remain limitations rather than being replaced by the current workstation's versions.""",
        ["data_manifest_v1.json", "research/tests/test_mathematical_foundations.py", "research/tests/test_prefix_stopping_model.py", "research/outputs/semester2/prefix_model_v1/training_protocol.json"])
    gsm = {int(row["step"]): row for row in boundaries if row["domain"] == "gsm8k"}
    pooled_rows = []
    for step in (2, 4):
        row = gsm[step]
        pooled_rows.append([
            str(step), f"{row['repair_events']} / {row['repair_denominator']}",
            f"{row['corruption_events']} / {row['corruption_denominator']}",
            f"{float(row['net_drift']):+.4f}",
        ])
    add(slides, "Repair and corruption use different denominators",
        "Pooled GSM8K transitions",
        ["19,500 trajectories over 500 task clusters at each shown transition.", "Net drift subtracts the declared 0.05 response cost.", "Negative utility drift can coexist with increasing accuracy."],
        75,
        """This table connects the conditional identity to observable transition counts. At the step-two GSM8K prefix, the current incorrect candidates form the repair denominator and the current correct candidates form the corruption denominator. The counts are 3,145 repairs among 14,818 incorrect candidates and 1,170 corruptions among 4,682 correct candidates. Their difference divided by 19,500 is the accuracy change, and subtracting point zero five gives net utility drift about plus point zero five one three. At step four, the positive accuracy change is about point zero three seven three; after the same cost, net drift is about minus point zero one two seven. Thus the answer accuracy can still improve while continuation is economically unproductive under this utility. These are pooled panel quantities over model configurations and temperatures. They do not reveal which individual task should stop. Uncertainty uses task clusters rather than treating all repeated trajectories as independent questions.""",
        ["research/outputs/thesis_v1/evidence/boundary_domain_step_metrics.csv"],
        table={"headers": ["Prefix t", "Repair / eligible", "Corruption / eligible", "Net utility drift"], "rows": pooled_rows})
    categories, drift_values = [], []
    for cell, steps, label in (
        ("qwen2p5_7b__gsm8k", (4, 5), "7B/GSM"),
        ("qwen2p5_32b__math", (5, 6), "32B/MATH"),
    ):
        for step in steps:
            row = next(row for row in cells if row["cell"] == cell and int(row["step"]) == step)
            categories.append(f"{label}, t={step}")
            drift_values.append(float(row["net_drift"]))
    add(slides, "The observed crossing depends on the cell",
        "Selected model-domain contrasts",
        ["Qwen2.5-7B/GSM8K changes sign between shown prefixes 4 and 5.", "Qwen2.5-32B/MATH changes sign between shown prefixes 5 and 6.", "Each cell has 1,500 trajectories and 500 task clusters."],
        65,
        """The selected-cell contrasts show why a universal step-two or step-three boundary is not supported. In the Qwen2.5-7B GSM8K cell, expected accuracy improvement minus the response cost is positive at the step-four prefix and negative at step five. For Qwen2.5-32B on MATH, the displayed transition is positive at step five and negative at step six. The same declared utility can therefore produce different descriptive crossing locations across model-domain cells. Each cell has 500 questions at three temperatures, giving 1,500 trajectories; those are not 1,500 independent questions. The graph plots stored net drifts for these exact prefixes, not a fitted smooth curve. It should not be extrapolated to other cells, an uninterrupted token-level thought chain, or a policy-optimal per-instance stopping time. The theorem's persistent conditional signs remain an additional assumption.""",
        ["research/outputs/thesis_v1/evidence/boundary_selected_cell_step_metrics.csv"],
        chart={"kind": "column", "categories": categories, "series": [{"name": "Net drift", "values": drift_values}],
               "x_title": "Model / domain and completed prefix", "y_axis_title": "Accuracy change − 0.05 cost",
               "y_axis_min": -0.04, "y_axis_max": 0.04})
    selected_effects = [
        ("N3", "Empirical-Bayes hazards"), ("N2c", "Lagged logistic"),
        ("N2a", "Gradient-boosted probe"), ("N2b", "Isotonic calibration"),
    ]
    effect_rows = []
    for experiment, label in selected_effects:
        row = next(row for row in effects if row["experiment"] == experiment)
        effect_rows.append([label, f"{float(row['mean_controlled_effect']):+.5f}",
                            f"[{float(row['ci_95_low']):+.5f}, {float(row['ci_95_high']):+.5f}]"])
    add(slides, "More expressive estimators did not always help",
        "Matched development comparisons",
        ["Effects are utility differences per trajectory relative to matched controls.", "Intervals resample 52 model-domain cells.", "These analyses do not establish a universally best estimator."],
        70,
        """Matched comparisons are more informative than comparing a new arm to an unrelated aggregate. The empirical-Bayes hazard replacement improves the recorded step utility by about point zero zero seven eight per trajectory relative to its matched local logistic estimator. Lagged features add a smaller positive contrast. Gradient boosting and isotonic calibration show substantial negative utility contrasts in their recorded experiments. Those outcomes should remain in the thesis; they constrain the claim that a richer score or a calibration layer automatically improves stopping. The intervals are percentile resamples of the 52 selected model-domain cells, not independent generation seeds and not uncertainty over every possible language model. The historical plus 593.55 figure for the hazard arm is a summed utility change over 75,965 trajectories. It is not a peer-agreement effect and must not be reassigned to that mechanism. Nor does the isotonic result prove that calibration is inherently harmful: the model, sample, target and downstream threshold jointly determine policy performance.""",
        ["research/outputs/thesis_v1/evidence/algorithm_v2_normalized_effects.csv"],
        table={"headers": ["Matched modification", "Utility / trajectory", "95% cell interval"], "rows": effect_rows})
    add(slides, "A paired precision contrast has a large local effect",
        "Qwen2.5-7B / GSM8K / step two",
        ["BF16: 418 correct out of 1,500; 4-bit: 204 out of 1,500.", "Absolute accuracy difference: 14.27 percentage points.", "Task-cluster interval: [11.13, 17.53] points."],
        70,
        """The numerical precision comparison is a paired local result. BF16 produces 418 correct candidates and four-bit weights 204 at the recorded step-two endpoint, among the same 1,500 task-temperature trajectories. The difference is 214 over 1,500, or 14.27 percentage points. It is an absolute accuracy contrast, not a fourteen percent relative reduction and not a result for every quantized language model. The task-cluster interval is approximately 11.13 to 17.53 points. A separate cap comparison for Mistral-Small-22B on GSM8K finds 454 losses in each of the 256- and 512-token arms with zero discordant loss indicators. That negative result applies to the specified binary policy-loss endpoint. It does not prove that token sequences, timing or all correctness outcomes are identical, and it does not rule out truncation across other cells. Both comparisons retain their original endpoint, unit and paired design.""",
        ["research/outputs/thesis_v1/evidence/algorithm_v2_normalized_effects.csv", "ThesisDocs/chapters/chapter4_empirical.md"],
        chart={"kind": "column", "categories": ["BF16", "4-bit"], "series": [{"name": "Step-two accuracy", "values": [418 / 1500, 204 / 1500]}],
               "x_title": "Weight precision", "y_axis_title": "Correct / 1,500 trajectories", "y_axis_min": 0, "y_axis_max": 0.35})
    gru = next(row for row in summary if row["configuration"] == "Causal GRU")
    rope = next(row for row in summary if row["configuration"] == "Causal RoPE transformer")
    add(slides, "Pooled causal discrimination hides a weak domain",
        "Five-step task-grouped detector evaluation",
        ["Causal GRU micro AUC 0.8743; task-macro 0.8214.", "Domain-macro AUC 0.8102; worst domain GPQA 0.6313.", "RoPE transformer has lower AUC but slightly higher stored step utility."],
        65,
        """The standardized corpus allows a prefix-safe sequence comparison under task-grouped evaluation. The causal GRU's micro AUC is point eight seven four three, but task-macro and domain-macro summaries are lower. The worst-domain GPQA score is about point six three one three. This difference matters for deployment: a pooled score can be driven by easier or larger groups while leaving a weak subgroup. Task-macro AUC is defined only for tasks with both classes; 2,679 of the 2,948 task groups are eligible. The causal RoPE transformer has slightly lower micro AUC and slightly higher recorded step utility than the GRU. Their descriptive fold intervals overlap, so the summaries do not establish a unique architecture winner. These are stored archive predictions. They support retrospective causal-prefix scoring and replay; they are not the new portable model's live AUC, and they do not establish conditional calibration.""",
        ["research/outputs/thesis_v1/evidence/tournament_balanced_summary.csv", "research/outputs/thesis_v1/evidence/tournament_per_domain.csv"],
        chart={"kind": "column", "categories": ["Micro", "Task macro", "Domain macro", "Worst domain"],
               "series": [{"name": "Causal GRU AUC", "values": [float(gru[key]) for key in ("micro_oof_auc", "task_macro_auc", "domain_macro_auc", "worst_domain_auc")]}],
               "x_title": "Aggregation rule", "y_axis_title": "ROC AUC", "y_axis_min": 0.5, "y_axis_max": 1.0},
        table_secondary={"headers": ["Configuration", "Micro AUC", "Step utility"], "rows": [
            ["Causal GRU", f"{float(gru['micro_oof_auc']):.4f}", f"{float(gru['micro_step_utility']):.4f}"],
            ["Causal RoPE transformer", f"{float(rope['micro_oof_auc']):.4f}", f"{float(rope['micro_step_utility']):.4f}"],
        ]})
    add(slides, "The retrospective 0.955 score is not a live guarantee",
        "Information and evaluation audit",
        ["Stacked development AUC: 0.955156; reduced-feature control: 0.943223.", "Bidirectional scoring and centered smoothing use future responses.", "Meta-training is not nested within the scoring outer fold.", "The 0.8743 causal result comes from a different evaluation."],
        90,
        """The most attractive historical number needs the clearest qualification. The stacked hybrid has stored AUC point nine five five one five six, versus point nine four three two two three for its reduced-feature LightGBM control. The task-bootstrap lift interval describes their difference on this development evaluation. It is not an absolute AUC interval. The bidirectional component reads all five saved steps and copies a trajectory score to earlier rows, while centered smoothing also uses later observations. Those inputs are unavailable at an early runtime stop. In addition, upstream meta-training is not confined to each reported outer training fold, so task-grouped scoring does not remove that dependence. The control labeled No Peers retains committee aggregates, preventing a pure peer-effect interpretation. The point nine five five and point eight seven four three scores therefore cannot be presented as before-and-after evidence that removing future information caused a particular AUC drop. They are different evaluations. The portable live stopper consumes neither of these full-sequence scores.""",
        ["ThesisDocs/chapters/chapter4_empirical.md", "ThesisDocs/chapters/chapter6_discussion.md"],
        table={"headers": ["Evidence", "Permitted claim"], "rows": [
            ["Stacked AUC 0.955156", "Retrospective development diagnostic"],
            ["Causal GRU AUC 0.874326", "Task-grouped archive prefix ranking"],
            ["Portable prefix model", "Separate train / calibration / holdout"],
            ["Actual online run", "Measured generated-cost / accuracy trade-off"],
        ]})
    add(slides, "The controller prevents future response calls",
        "Runtime implementation and measured latency",
        ["It owns the prefix, rejects future/out-of-order inputs, and closes permanently on stop.", "Strict JSON telemetry authorizes confidence; fallbacks carry confidence=None.", "Peers require a complete same-step barrier and fully charged computation.", "Saved-prefix benchmark: 18,880 decisions; worst observed 0.7465 ms."],
        80,
        """The causal interface accepts exactly one newly completed Observation. It has no parameter for gold, correctness, future rows or a full-sequence score. The controller owns the accumulated prefix and rejects skipped or repeated steps. After termination, a new response is rejected rather than silently beginning a new run. Strict JSON parsing is important: a legacy fallback can invent a default confidence of fifty, which must not become a trusted observation. The live adapter instead uses None on strict failure. Optional peer votes require the complete declared roster at the current step, a closed barrier before the decision and recorded peer costs. The default experiment uses no peers or external verifier. On saved prefixes from 100 MATH problems over 20 repeats, the combined benchmark makes 18,880 prepared-observation decisions. Median latency is point zero zero seven six milliseconds and the worst observed is point seven four six five. This verifies the under-ten-millisecond engineering target on that benchmark; it excludes model loading, generation, tokenization and peer waits. The learned model has a separate benchmark rather than inheriting this number.""",
        ["research/online_stopping_controller.py", "research/online_generation.py", "research/outputs/semester2/online_stopping_20261002/latency_summary.json"],
        table={"headers": ["Prepared-observation decisions", "Latency"], "rows": [
            ["Median", "0.0076 ms"], ["95th percentile", "0.0209 ms"],
            ["99th percentile", "0.0342 ms"], ["Maximum observed", "0.7465 ms"],
        ]})
    q = learned["probabilities"]["q_current"]
    nxt = learned["probabilities"]["p_next"]
    add(slides, "A portable trained model estimates current and next correctness",
        "Fixed task-disjoint probability fitting",
        ["1,500 archived tasks: 902 train, 322 calibration, 276 evaluation.", "Two unweighted logistic models; train-only scaling and separate Platt calibration.", "21 features use only current/prior Observation fields plus public domain.", "No live GSM8K-test or trap outcomes select the model or threshold."],
        65,
        """The thesis now includes a trained deployable artifact rather than only a confidence heuristic. It fits two probabilities: correctness of the latest nonempty currently available candidate and correctness of the same selector after one more response. Public task identity fixes roles before fitting or parser availability. All 1,500 archived Qwen2.5-0.5B tasks are retained: 902 training, 322 calibration and 276 evaluation. The scaler and base logistic coefficients are fitted only on training tasks; Platt calibrators use only the independent calibration tasks. Evaluation enters neither fit. The feature contract contains current and prior answer churn, response and thought length, observed confidence when strict-valid, fixed text-density summaries, step and public domain. It excludes labels, timestamps and future rows. Archive-only gold regrading aligns reconstructed live-parser candidates with targets: 4,269 of 7,500 candidate strings differ from saved strings, but only 43 correctness labels change. The original corpus is preserved. These held-out scores validate marginal archive prediction, not runtime conditional certainty.""",
        ["research/prefix_stopping_model.py", "research/train_prefix_stopping_model.py", "research/outputs/semester2/prefix_model_v1/training_protocol.json", "research/outputs/semester2/prefix_model_v1/label_reconstruction_audit.csv"],
        table={"headers": ["Held-out target", "Rows", "AUC", "Calibrated Brier"], "rows": [
            ["Selected current correctness", str(q["calibrated"]["rows"]), f"{q['calibrated']['auc']:.4f}", f"{q['calibrated']['brier']:.4f}"],
            ["Selected next correctness", str(nxt["calibrated"]["rows"]), f"{nxt['calibrated']['auc']:.4f}", f"{nxt['calibrated']['brier']:.4f}"],
        ]})
    replay_rows = []
    for key, label in (("never", "Full horizon"), ("fixed_2", "Fixed two"), ("learned_drift", "Learned drift")):
        result = learned["policies"][key]
        replay_rows.append([label, percent(result["accuracy"]), percent(result["completion_token_saving_fraction"]),
                            f"{result['mean_stop_step']:.3f}"])
    add(slides, "The trained drift policy nearly reduces to fixed two",
        "Held-out archive replay and transport limits",
        ["Stop when p̂next − q̂current − 0.05 ≤ 0, with floor 2 / horizon 5.", "275 of 276 held-out tasks stop at two; one stops at three.", "No material policy advantage over fixed-two is demonstrated.", "Archive strict JSON support is zero: live prompt transport is a major limitation."],
        80,
        """The declared policy is myopic: subtract current probability and the fixed response cost from the next-correctness estimate, and stop at the first nonpositive result after the floor. It does not approximate a validated Bellman continuation value. On the 276 held-out archive tasks, 275 stop at step two and one at step three. The learned policy has essentially the same accuracy as fixed two and slightly greater recorded cost. It therefore does not demonstrate added policy value over that simple baseline. Relative to full horizon, its paired accuracy difference is plus 1.45 percentage points with a task-cluster interval from minus 1.45 to plus 4.35 points. Replay completion-token savings are 51.29 percent, with a task-cluster interval about 49.31 to 53.09 percent; those are archived counts, not measured live savings. Archive outputs contain no strict-valid JSON records, so confidence-related features lack training support. Live JSON prompting and charged EOS tokens also differ. A historical in-sample 7B replay saved 54.34 percent while losing 6.33 accuracy points; it remains a separate development diagnostic.""",
        ["research/outputs/semester2/prefix_model_v1/evaluation.json", "research/outputs/semester2/prefix_model_v1/heldout_policy_replay.csv", "research/outputs/thesis_v1/evidence/offline_replay_metrics.csv"],
        table={"headers": ["Archive holdout policy", "Accuracy", "Completion saving", "Mean stop"], "rows": replay_rows})
    live_rows = []
    if live:
        live_rows = [
            ["Selected-answer accuracy", percent(live["baseline_accuracy"]), percent(live["active_accuracy"]), "Pending"],
            ["Generated completion tokens", str(live["baseline_generated_tokens"]), str(live["active_generated_tokens"]), "Pending"],
            ["Completion-token saving", "0%", percent(live["measured_completion_token_savings"]), "Pending"],
            ["Mean stopping step", "5.00", f"{live['mean_active_stop_step']:.2f}", "Pending"],
            ["Shared-prefix identical tasks", "Reference", f"{live['shared_prefix_identical_problems']} / 100", "Pending"],
        ]
    if learned_live:
        if not live_rows:
            raise ValueError("learned live results require the paired baseline ledger")
        live_rows[0][-1] = percent(learned_live["active_accuracy"])
        live_rows[1][-1] = str(learned_live["active_generated_tokens"])
        live_rows[2][-1] = percent(learned_live["measured_completion_token_savings"])
        live_rows[3][-1] = f"{learned_live['mean_active_stop_step']:.2f}"
        live_rows[4][-1] = f"{learned_live['shared_prefix_identical_problems']} / 100"
    add(slides, "Actual generation measures a modest heuristic saving",
        "100-task GSM8K test development panel",
        ["Original heuristic completion saving: 2.94%; accuracy 6 / 100 in both arms.",
         "Reported intervals: savings [0.85%, 5.63%]; accuracy change [−4.29, +4.29] points.",
         "Only 88 / 500 baseline responses are strict-valid JSON (17.6%).",
         "Learned live outcomes remain pending until their final ledger is complete." if not learned_live else "Learned results use the frozen artifact and a separately executed active arm."],
        65,
        """This is actual generation evidence on 100 cached GSM8K-test public tasks, kept separate from the label ledger until grading. The original heuristic produces 21,591 completion tokens versus 22,244 for full horizon, giving a 2.94 percent reduction. Both arms answer six questions correctly. The paired accuracy interval is approximately minus 4.29 to plus 4.29 percentage points, so an observed zero change is not a tight noninferiority conclusion. Only 88 of the 500 baseline responses satisfy the strict JSON contract. The active arm has mean stopping step 4.85: six tasks stop for stable high confidence, and 94 reach the terminal horizon. Shared generated prefixes are identical on 94 of 100 tasks, which limits an exact counterfactual interpretation on the remainder. Prompt work, padded decoder slots and model seconds are reported separately in the ledger. The result is far below a thirty-to-forty-percent heuristic savings target. The trained policy uses a frozen artifact in a separate active generation run; its outcome is inserted only when that run's final metrics exist, with no retuning.""",
        ["research/outputs/semester2/online_stopping_20261002/live_metrics.json", "research/outputs/semester2/online_stopping_20261002/live_paired_results.csv", "research/outputs/semester2/online_stopping_20261002/learned_main/live_metrics.json"],
        table={"headers": ["Actual outcome", "Never", "Heuristic", "Learned"], "rows": live_rows},
        pending_outcomes=[] if learned_live else ["learned_main"],
        outcome_refresh={"source": "research/outputs/semester2/online_stopping_20261002/learned_main/live_metrics.json"})
    trap_rows = []
    for label, result in (("Heuristic", trap), ("Learned", learned_trap)):
        if result is None:
            trap_rows.append([label, "Pending", "Pending", "Pending"])
        else:
            trap_rows.append([label, f"{result['active_correct']} / {result['problems_or_trajectories']}",
                              percent(result["measured_completion_token_savings"]),
                              f"{result['paired_worsened']} worsened / {result['paired_improved']} improved"])
    add(slides, "Trap questions probe the policy's failure modes",
        "Adversarial bank and limits",
        ["20 mathematical prompts with separately derived gold answers.", "Changing bases, dependent sampling, rates, precedence and answer contracts.", "Wrong stable answers and delayed repair can defeat a working controller.", "One small model, one frozen prompt and a development panel limit generalization."],
        65,
        """The adversarial bank is independently specified with 20 public mathematical prompts and separate exact short derivations. It tests traps such as changing percentage bases, harmonic average speed, conditional probability, dependent draws, inclusive endpoint counts and exponent precedence. It is not an exhaustive estimate of worst-case adversarial risk. Classic forms can occur in pretraining even when these project records are newly written. A stable, confident wrong answer is a statistical failure mode of the heuristic, not necessarily a state-machine bug. Likewise, a late repair can defeat the learned myopic drift rule without violating prefix causality. Malformed observations, forged peer receipts and cancellation belong to the separate software test suite. When outcomes are available, the table reports accuracy, actual generated-token saving and paired worsened or improved counts without adjusting the thresholds on those questions. The scope remains Qwen2.5-0.5B under this protocol; it does not establish transfer to larger models or all mathematical tasks.""",
        ["research/adversarial_tasks_v1.jsonl", "research/adversarial_gold_v1.jsonl", "research/outputs/semester2/online_stopping_20261002/adversarial_live/live_metrics.json", "research/outputs/semester2/online_stopping_20261002/learned_adversarial/live_metrics.json"],
        table={"headers": ["Trap policy", "Accuracy", "Completion saving", "Paired changes"], "rows": trap_rows},
        pending_outcomes=[name for name, value in (("adversarial_live", trap), ("learned_adversarial", learned_trap)) if value is None])
    add(slides, "The contribution is a testable stopping framework",
        "Conclusion and next confirmatory study",
        ["Exact conditional drift and finite-horizon optimal-stopping proofs.", "Frozen empirical contrasts retain subgroup and negative results.", "A causal runtime and serialized task-disjoint trained predictor.", "Next: unseen tasks, prespecified accuracy margin, multiple seeds and full costs."],
        65,
        """The thesis contributes a rigorous decision framework, a traceable empirical record and an implementable causal stopping system. The general optimum is the Snell-envelope first-contact rule; a first drift crossing needs persistent conditional signs. The empirical panels show local repair, corruption, cost crossings and estimator contrasts, while preserving negative outcomes and uncertainty at the correct cluster level. The engineering system actually prevents future response generation and keeps labels outside its runtime interface. The trained predictor is serializable and calibrated on separate task groups, but its archive policy largely collapses to fixed two and its live transport has limited support. These are honest limits of the completed research result. A confirmatory extension should freeze an unseen task panel, prompt, fitted model and cost preference, state a meaningful accuracy noninferiority margin, use several generation seeds, and report prompt, completion, peer and timing costs. Defense, committee signatures and institutional deposit remain real external events; a generated package cannot substitute for them. I welcome questions about the mathematics, evidence boundaries and deployment trade-offs.""",
        [math_source, "ThesisDocs/chapters/chapter6_discussion.md", "ThesisDocs/requirements_acceptance_matrix.md"])

    # Final result refresh changes content without changing the 25-slide design.
    if learned_live and learned_main_ci:
        slides[22]["title"] = "Actual generation yields different policy trade-offs"
        token_ci = learned_main_ci["completion_token_savings_cluster_bootstrap_95ci"]
        accuracy_ci = learned_main_ci["accuracy_delta_conservative_exact_95ci"]
        slides[22]["bullets"] = [
            f"Learned completion saving: {percent(learned_live['measured_completion_token_savings'])}; interval [{percent(token_ci[0])}, {percent(token_ci[1])}].",
            f"Learned accuracy: 7 / 100 versus 6 / 100; change interval [{100 * accuracy_ci[0]:+.2f}, {100 * accuracy_ci[1]:+.2f}] points.",
            "All 100 learned tasks stop at two; no registered noninferiority margin.",
            "Baseline strict JSON: 88 / 500; learned shared prefixes: 100 / 100.",
        ]
        slides[22]["speaker_notes"] = """The frozen learned policy now has actual generation results. It produces 9,674 completion tokens versus the prior full-horizon arm's 22,244, a 56.51 percent reduction; the task bootstrap interval is 54.83 to 58.01 percent. It answers seven of 100 tasks correctly versus six for baseline. The conservative paired interval for the one-point improvement is minus 4.27 to plus 6.21 points. No noninferiority margin was registered. Every task stops at two, so the result cannot establish added value over fixed two. Shared prefixes match on all 100 tasks. This learned arm is separately generated, not replay; its baseline is the actually executed prior arm linked by hash. The original heuristic saved 2.94 percent with six correct in each arm and shared-prefix agreement on 94 tasks. Baseline JSON success is only 88 of 500 responses. Absolute accuracy is low. Prompt work, padded slots and model seconds have separate ledgers. These are one-model development outcomes; no label was used to tune the frozen policy."""
        slides[22]["sources"].append("research/outputs/semester2/online_stopping_20261002/learned_main/live_uncertainty.json")
    if learned_trap and learned_trap_ci:
        slides[23]["speaker_notes"] = """The frozen bank contains 20 mathematical traps with independent short derivations: changing percentage bases, dependent draws, harmonic rates, endpoint counting, exponent precedence and answer contracts. The learned arm emits 2,268 completion tokens against 4,736 for full horizon, saving 52.11 percent; descriptive resampling gives 45.86 to 57.17 percent. Both arms answer one question correctly. All 20 learned tasks stop at two, and every shared prefix matches. The conservative iid-reference accuracy-change interval spans minus 19.68 to plus 19.68 points. Because the bank is handpicked, this is not randomized-population adversarial coverage. The heuristic separately saves 3.34 percent with one correct in each arm. A wrong stable answer or a missed late repair can defeat the policy without a software bug. Malformed observations and forged peer receipts belong to the separate state-machine tests. No trap result changes the coefficients, calibrators or threshold. Classic puzzle forms may occur in pretraining; the evidence is robustness on this bank and protocol, not exhaustive worst-case risk."""
        slides[23]["sources"].append("research/outputs/semester2/online_stopping_20261002/learned_adversarial/live_uncertainty.json")
    if controller_latency:
        all_latency = controller_latency["all"]
        slides[19]["bullets"][-1] = (
            f"Saved-MATH benchmark: {all_latency['decisions']:,} decisions; worst observed {all_latency['max_ms']:.4f} ms."
        )
        slides[19]["speaker_notes"] = (
            "The controller accepts one newly completed Observation with no gold, future row or full-sequence score. "
            "It owns the prefix, rejects skipped or repeated steps, and permanently closes after stopping. "
            "Strict JSON is necessary for trusted confidence: a legacy fallback default must remain missing. "
            "A peer panel requires the complete declared same-step roster, a barrier before the decision and charged peer computation. "
            "The default live policy uses no peers or verifier. "
            f"The final saved-MATH benchmark has {controller_latency['distinct_problems']} problems, {controller_latency['repeats']} repeats "
            f"and {all_latency['decisions']:,} combined heuristic/control decisions. "
            f"Median latency is {all_latency['median_ms']:.4f} ms and maximum {all_latency['max_ms']:.4f} ms. "
            "This covers prepared-observation validation, policy features, selection and closure, excluding telemetry extraction, loading, generation and peer waits. "
            "It establishes a worst-observed under-ten-millisecond result on this workload, not a worst-case execution-time proof."
        )
        slides[19]["table"]["rows"] = [
            ["Median (saved MATH)", f"{all_latency['median_ms']:.4f} ms"],
            ["95th percentile (saved MATH)", f"{all_latency['p95_ms']:.4f} ms"],
            ["99th percentile (saved MATH)", f"{all_latency['p99_ms']:.4f} ms"],
            ["Maximum (saved MATH)", f"{all_latency['max_ms']:.4f} ms"],
        ]
    if learned_latency:
        slides[19]["speaker_notes"] += (
            f" Separately, the learned main-panel benchmark makes {learned_latency['decisions']:,} decisions over "
            f"{learned_latency['distinct_problems']} problems and {learned_latency['repeats']} repeats. Its maximum is "
            f"{learned_latency['max_ms']:.4f} ms, including prefix features, both calibrated heads and the controller, "
            "and excluding generation, loading, tokenization and peer waits."
        )
        slides[19]["sources"].append("research/outputs/semester2/online_stopping_20261002/learned_main/learned_latency_summary.json")
        slides[19]["table"]["rows"].append(["Learned maximum (4,000 main decisions)", f"{learned_latency['max_ms']:.4f} ms"])
    # Short chart annotations avoid duplicating the plotted labels in small space.
    slides[14]["bullets"] = ["7B/GSM: crossing 4 → 5.", "32B/MATH: crossing 5 → 6.", "500 tasks per cell; three temperatures."]
    slides[17]["bullets"] = ["GRU micro AUC: 0.8743.", "Worst domain GPQA: 0.6313.", "AUC and policy utility rank differently."]
    captions = {
        3: "Overlapping corpora; counts are not additive.",
        4: "The floor and horizon are declared protocol constraints.",
        5: "The runtime interface excludes gold and future responses; statistical measurability has separate assumptions.",
        10: "Immediate drift is negative at zero, yet optimal continuation value is 0.80.",
        14: "This population utility drift can turn negative while average accuracy still increases.",
        16: "Matched utility contrasts, with 52-cell descriptive bootstrap intervals.",
        19: "Future-step inputs and non-nested meta-fitting preclude a live 0.955 claim.",
        20: "Controller timing excludes LM generation/loading, tokenization and peer waits.",
        21: "Separate task roles; archive calibration does not establish live calibration (0 / 7,500 strict-valid JSON).",
        22: "Learned drift nearly equals fixed two; these archived token counts are not measured live savings.",
        23: "All learned tasks stop at two; added value over fixed two and accuracy noninferiority are not established.",
        24: "Handpicked bank: descriptive robustness only; all learned tasks stop at two.",
    }
    for number, caption in captions.items():
        if len(caption) > 150:
            raise AssertionError("caption too long")
        slides[number - 1]["table_caption"] = caption
    if len(slides) != 25 or sum(slide["seconds"] for slide in slides) != 1800:
        raise AssertionError((len(slides), sum(slide["seconds"] for slide in slides)))
    package = {
        "schema": "thesis-defense-content-v1", "title": slides[0]["title"], "author": "Aditya Bhatt",
        "canvas": {"width": 1280, "height": 720}, "slide_count": 25, "total_seconds": 1800,
        "style": {"background": "#FFFFFF", "ink": "#112437", "accent": "#004B87", "font": "Calibri",
                  "title_min_points": 32, "body_min_points": 17, "cover_title_min_points": 42,
                  "evidence": "editable native PowerPoint tables and charts; no evidence screenshots"},
        "artifact_status": "content complete; PPTX authoring and visual QA are separate",
        "model_artifact_sha256": learned["artifact_sha256"],
        "slides": slides,
    }
    (HERE / "defense_slides.json").write_text(json.dumps(package, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    source, notes, schedule = ["# Defense slide source", "", "Exactly 25 slides. Allocated speaking time: 30 minutes.", ""], ["# Thirty-minute defense speaker notes", ""], [
        "| Slide | Start | Duration | Title |", "|---:|---:|---:|---|",
    ]
    elapsed = 0
    for slide in slides:
        start = f"{elapsed // 60:02d}:{elapsed % 60:02d}"
        elapsed += slide["seconds"]
        end = f"{elapsed // 60:02d}:{elapsed % 60:02d}"
        schedule.append(f"| {slide['number']} | {start} | {slide['seconds']} s | {slide['title']} |")
        source += [f"## Slide {slide['number']}: {slide['title']}", "", slide["kicker"], "",
                   *["- " + bullet for bullet in slide["bullets"]], ""]
        if "equation" in slide:
            source += [slide["equation"], ""]
        if "table" in slide:
            table = slide["table"]
            source += ["| " + " | ".join(table["headers"]) + " |",
                       "|" + "|".join("---" for _ in table["headers"]) + "|"]
            source += ["| " + " | ".join(row) + " |" for row in table["rows"]]
            source += [""]
        if "chart" in slide:
            source += ["Editable chart data: " + json.dumps(slide["chart"], ensure_ascii=False), ""]
        if "table_caption" in slide:
            source += ["Visible evidence qualification: " + slide["table_caption"], ""]
        source += ["Sources: " + "; ".join(slide["sources"]), ""]
        notes += [f"## Slide {slide['number']} • {start}–{end} • {slide['title']}", "", slide["speaker_notes"], "",
                  "Sources: " + "; ".join(slide["sources"]), ""]
        for reference in slide.get("references", []):
            notes += [f"- [{reference['title']}]({reference['url']})"]
        notes += [""]
    (HERE / "defense_slide_source.md").write_text("\n".join(source) + "\n", encoding="utf-8")
    (HERE / "speaker_notes_30_minutes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    (HERE / "timing_plan.md").write_text("\n".join(schedule) + "\n", encoding="utf-8")
    print(json.dumps({"slides": len(slides), "seconds": elapsed, "pending": [
        {"slide": slide["number"], "outcomes": slide["pending_outcomes"]}
        for slide in slides if slide.get("pending_outcomes")
    ]}))


if __name__ == "__main__":
    main()
