# Independent review of result interpretation and explanatory copy

Reviewer: Agent B. Scope: presentation of the user's existing logarithm-prompt run and final explanatory formulas after the UI changes. This file is B's only output for this task. No model generation, source/data changes, training, grading, dependency changes or controller-policy changes were performed by this reviewer.

## Status

**Final verdict: the revised explanation is mathematically faithful to the recorded comparison, with no unresolved issue in this review's scope.** B independently checked the original and subsequent live-run observations, reread the final explanatory source, exercised 13 browser/formula check groups without model generation, and viewed the final actual-run desktop light/dark and mobile dark screenshots. The numerical conclusions below do not depend on the earlier summary's verdict flags. This is an interpretation/UI review, not an answer-quality, calibration or accessibility certification.

## Exact observed run

Read-only access used `http://127.0.0.1:8765/api/status`, then the returned run endpoint. The latest user run at inspection was `716a4be6d70148d6a2fd50847b920364`, status complete, learned policy, `math` domain, response cost 0.05. Its prompt asks for all real solutions of `log_2(x^2-7x+10)+log_2(x^2-7x+12)=-1`.

The complete terminal snapshot was read without modifying it. Its canonical JSON SHA256, calculated with sorted keys and compact separators, was `3055eb007b3bfd419d2bb01986e55f5d18c29b4648bce36d8e4990bf23a5485b`. This identifies B's read-only evidence while leaving the run and research files untouched.

| Quantity | Independent result |
| --- | --- |
| Active STOP boundary | Response 2 of 5 |
| Active selected candidate step | 2 |
| Baseline terminal boundary / selected step | 5 / 5 |
| Completed-response token counts | 35, 33, 33, 34, 34 |
| Active / baseline completion tokens | 68 / 169 |
| Avoided active-arm suffix | 101 tokens; 3 responses |
| Completion-token comparison reduction | 101/169=59.76331360946746%, displayed 59.8% |
| Active / baseline prompt tokens | 483 / 1,772 |
| Physical completion tokens for the whole shared comparison | 169 |

The coordinator's later authorized browser run, `0a1ffd55862447c796ba5b0032b5e61d`, was also read independently from the live API. It used the same logarithm prompt and recorded the same five candidate strings, parse failures, response-token counts, stopping boundary and fitted estimates. Its differing elapsed times do not change the interpretation. B did not initiate this or any other model run.

These figures establish a response-budget/token comparison on this one shared trajectory. The demonstration still executed the full baseline. Three skipped responses are 60% of a five-response budget, while avoided completion tokens are 59.8%; these are different denominators and should not be merged.

## Why the old answer badge is misleading for this run

All five Observation records have `parse_success=false` and trusted `confidence=None`. Independently applying Python's JSON decoder to each raw response produces `Invalid \escape` at character 48. The generated strings contain TeX `\(` and `\)` escapes that are invalid as single JSON escapes. The existing fallback parser consequently retained the entire raw JSON-like string as the candidate, rather than a trustworthy structured `answer` field.

The early selected fallback string and the baseline selected fallback string differ only in the embedded raw confidence value 98 versus 100. Replacing 98 with 100 makes the early string exactly equal to the baseline string. The previous `answers_match=false` therefore reports a whole-output string difference; it does not show a changed mathematical answer, worse reasoning, repair, corruption or overthinking.

The main UI should state **“Answer comparison unavailable: the expected answer format was not parsed.”** Keep the recorded fallback and raw text in technical/output details. Do not repair the JSON, newly extract 25 as the actual selected candidate, change the controller state, or describe raw confidence 98 as trusted telemetry. Such changes would alter the instrument rather than clarify its presentation.

For general runs, check format availability at each arm's **selected candidate step**, rather than at the most recent response or across every response. A valid earlier candidate retained after a blank later response remains eligible for a syntactic answer-text comparison. Conversely, a selected malformed fallback remains malformed even if an earlier response was valid. A syntactically valid candidate still has not been graded for mathematical correctness.

## Independent decision arithmetic

B replayed only the first two recorded Observation objects through the unchanged frozen predictor. The artifact SHA256 is `92fe0af86ac0f204d514a6938d0a29dace2b3cdffa24800cc4e2c85a2d51879f`. The output matches the recorded decision exactly:

| Term | Probability/utility units | Percentage-point equivalent |
| --- | ---: | ---: |
| Current fitted score q |0.0033001717766077128 |0.33001717766077127% |
| Next-response fitted score p |0.006325662440481306 |0.6325662440481306% |
| Predicted correctness gain p-q |0.0030254906638735934 |+0.30254906638735934 percentage points |
| Assigned response cost lambda |0.05 |5.0 percentage points of normalized reward |
| Net estimated value p-q-lambda |-0.04697450933612641 |-4.697450933612641 percentage points of normalized reward |

With correct-answer reward 1 and wrong-answer penalty 0, the UI may put gain and cost on the same equivalent scale. It must say **percentage points**, not relative-percent improvement, when subtracting these numbers. The 0.05 cost is a chosen utility penalty; it is not 5% of tokens, money, energy or an observed 5% accuracy loss.

The rule is `continue if p-q-lambda>0`, otherwise STOP after the minimum floor. Thus a **positive estimated improvement can still lead to STOP** when it is smaller than the assigned continuation cost. This run's fitted gain is about +0.3 points, the cost is 5.0 points and the net is about -4.7 points. No claim that the predicted gain itself is negative is warranted.

The raw model-written confidence 98 is a separate quantity from q≈0.3% and p≈0.6%. Here the raw confidence is not trusted at all because the structured record failed parsing. The fitted scores use the declared prefix features, including the missing/untrusted-confidence indicator; they are not that raw confidence rescaled. The fitted scores are development estimates, not grades or certified probabilities for this new prompt.

## Suggested primary copy

Lead with the observable action:

> Stopped after 2 of 5 responses. The early arm skipped 3 further responses.

Then show the token comparison:

> 68 versus 169 completion tokens — 101 fewer for the early arm (59.8%). The shared comparison actually generated 169 tokens to finish the baseline.

Then state the answer limitation:

> Answer comparison unavailable: the expected answer format was not parsed. No mathematical correctness check was performed.

The decision explanation can read:

> The fitted predictor estimated a +0.3 percentage-point gain from another response. That is below the chosen 5.0-point continuation penalty, leaving a net value of -4.7 points, so the policy stopped.

Keep “Current fitted score 0.3%” and “Next-response fitted score 0.6%” as supporting details, clearly separated from model-written confidence. A short final statement such as **“This shows a budget difference; it does not establish overthinking or answer quality.”** is faithful to this run. The generic “different answers” badge and opaque decimal drift should not lead the result.

## Required interpretation corner cases

| Case | Honest interpretation / check |
| --- | --- |
| Exact zero net value | STOP follows `<=0`; show a tie/zero net, not a negative gain. |
| q=0 or p=0 | Render 0.0%, not missing/undefined. |
| Positive gain below cost | STOP; explain gain minus cost, as in this user run. |
| Positive net continuation | Continue after the floor; do not use the STOP explanation. |
| Later STOP, e.g. response 4 | Report 4/5 and 1 remaining response avoided; use that decision's q,p,cost, not response-two values. |
| Terminal response 5 | “Full five-response budget used; no further responses skipped.” Terminal stopping needs no next-response estimate, so do not invent one or imply a drift crossing. |
| Before minimum response 2 | “Minimum budget not reached.” No fitted head is queried at response 1. |
| Heuristic/no q,p estimates | Explain the actual heuristic/horizon reason; suppress gain-minus-cost probability arithmetic and avoid fabricated zeros. |
| Earlier valid answer retained | Format status follows selected_step, and resource cost follows every executed response through the STOP boundary. |
| Blank selected candidate | No answer-text comparison; show unavailable/blank output without calling two blanks a match. |
| Selected fallback malformed | No structured-answer comparison; retain raw output in details unchanged. |
| Valid selected strings equal/different | Describe normalized text agreement/difference only, with no correctness or semantic-equivalence inference. |
| Cancelled/failed incomplete baseline | Preserve partial output and any real active STOP, but do not show a completed five-response token saving. |
| Very small nonzero net rounding to 0.0 | The explanation's sign must follow the original value. Use “slightly positive/negative” or more precision rather than display an apparent exact tie with the opposite action. |

The final review verified that these explanations use the actual recorded decision values and configured cost rather than hard-coded numbers. The user case's rounded 0.3% and 0.6% labels are summaries; gain and net use full precision before formatting. No model/controller/parser change was needed to make the result understandable.

## Final source and browser checks

The final actual logarithm run named in `ux_v2/verification.json`, `5adf55c8ec28469cba0fd0ec56e98dec`, was independently fetched from its read-only API endpoint. Its canonical snapshot SHA256 is `2480ec0efcff93ac40df7025d56f496616f98e95343b8900f8d2cacc1647a919`. B independently confirmed the five token counts, every parse failure at character 48, missing trusted confidence, the selected strings' sole 98/100 difference, and the exact gain/cost/net arithmetic. The expected adverse model/parser behavior was preserved.

Final production source inspected:

| File | SHA256 |
| --- | --- |
| `static/app.js` | `abe9639f20acc2a7b53f29267886ea72c8bef1704287bf059a42f6ad2f16f147` |
| `static/index.html` | `fc1abebbf21f48715b2b5fb0d4eabe92a137f18f6fa44fe2385cced372a892f4` |
| `static/styles.css` | `a2d14bbfd74563ad4a8e836915583e325ff9a35fbc11d2defa77829fa80ec958` |

The checked functions are `app.js:23` (`points`), `:46` (`readableAnswer`), `:69` (`renderOutcome`), `:211` (`renderAnswer`), `:375` (`showDecision`), `:428` (`applySummary`) and `:466` (`finishRun`). They use actual retained-step metadata, actual supplied scores and configured cost. They suppress fabricated terminal/absent estimates, withhold malformed candidates from the primary answer, retain original output in disclosures, qualify text comparison as ungraded, and distinguish physical comparison work from the active-arm difference. The main explainer now gives the current/next fitted scores and gain-versus-penalty reason in visible prose. The intro says another response must help **enough**, matching the cost-sensitive rule.

B's independent browser check used a fresh installed Chrome instance and intercepted every request in memory. It supplied the current production assets and explicit synthetic API records; all requests were GET, no request reached the live inference service, and no model generation or persistent test artifact was produced. Main cases exercised the actual event-restoration wiring. For seven additional numerical/copy edge cases, an in-memory test-only entry point called the unchanged production `showDecision` function and inspected the DOM. No production file was instrumented or edited.

| Independent check group | Result |
| --- | --- |
| Malformed selected record withheld; comparison unavailable | PASS |
| Earlier valid response retained after a malformed later response | PASS |
| Positive +0.3-point gain below 5.0-point penalty, -4.7 net | PASS |
| Full horizon: both use five, 0.0% reduction, no next-head estimate | PASS |
| Later response-four STOP: one remaining response | PASS |
| q=p=0 are supplied estimates, rendered 0.0% | PASS |
| Exact gain/cost tie: zero net and STOP, including configured cost 0.25 | PASS |
| Tiny positive net: Continue, positive sign retained, rounded tie explained | PASS |
| Tiny negative net: STOP, negative sign retained | PASS |
| Missing learned estimates after the floor: metadata unavailable | PASS |
| Response-one minimum: no estimate invented | PASS |
| Heuristic: actual confidence-rule explanation, no fitted-score arithmetic | PASS |
| Blank selected output: no answer-text comparison | PASS |

All 13 groups passed with zero browser page errors. The tiny-net examples displayed `+0.000010` and `-0.000010` percentage points rather than false signed zeros. C's separate eight-group artifact `ux_tests_results.json` also records passing selected-step, zero/tie, cancellation, collapsed-details and mobile theme checks on the same three static hashes. B reviewed that evidence as corroboration, not as a substitute for independent arithmetic and source inspection.

B viewed `ux_v2/comparison_light.png`, `comparison_dark.png` and `mobile_dark_full.png`. They visibly lead with STOP 2/5 and three skipped responses, show 59.8% as a text-budget difference, state answer comparison unavailable, and preserve access to the literal records. The two arms show 68 versus 169 completion tokens. The mobile explainer separates the 0.3%/0.6% fitted scores from the +0.3-point gain and 5.0-point penalty. No sign, denominator or answer-quality claim in those images conflicts with the recorded run.

## Preservation and review correction log

The UI revision did not change the backend or frozen instrument. B independently compared `runtime.py`, `server.py` and `streaming_generator.py` with their pre-UX entries in `ux_v2/baseline/artifact_manifest.json`: all three are byte-identical. Their SHA256 values remain `2fdcccab2c4af7df137b92ebe28d0a960b74ccbd85ac21466cb8d3487c6a3be5`, `11e2fb02f2ef4a0c2faf1fe190a42ad3da03679bc2a734a4b36c566f7967640f` and `0328d281a6378700ad08b9b18fa692cb3854291d396df9951433cffd5b18c4e1`. B also independently rehashed all 36 original research/manuscript evidence files in the audit baseline; all match. The frozen predictor identity remains the SHA256 recorded above.

Two presentation corners identified during this review were corrected by the coordinator: tiny nonzero point values now retain their sign, with an explicit rounded gain/penalty tie explanation; and learned decisions missing metadata after the floor say estimates are unavailable instead of claiming the minimum is unmet. Exact-tie STOP copy now says the gain does not exceed the penalty. These are explanatory corrections, not policy changes.

B's first ad hoc browser probe read `inner_text` from intentionally collapsed estimate details, so it returned empty strings and failed the probe assertion. The independent probe was corrected to inspect DOM text for numerical checks; the production page did not require a fix. A separate initial all-cases probe assumed every synthetic case was complete, but the expanded fixture set includes a cancelled case; it was narrowed to the intended complete cases. These were reviewer harness assumptions, not product defects. Reading the baseline JSON also required explicit UTF-8 BOM handling on Windows; re-reading with `utf-8-sig` completed the preservation check.

The remaining limitation is substantive and visible: the local model did not provide reliably structured answers for this advanced prompt, no correctness grading was performed, and this run does not establish that continuing worsened an answer. The observed resource-budget result and the predictor's cost-sensitive reason can nevertheless be explained accurately, as the revised UI now does.
