# UX explanation regression review

All eight browser groups passed on the final combined revision at 2026-10-14 04:08 UTC, with zero browser page errors. The existing 25 backend tests also passed (0.838 seconds). No production files or backend tests were changed for this review.

These are explicitly **synthetic UX test fixtures**, not model observations, accuracy results, or a demonstration that early stopping improves an answer. The harness serves the real production assets on an isolated ephemeral loopback server with an inert manager, intercepts every API call, rejects non-GET or unknown API routes, and launches the existing Chrome installation through Playwright in a fresh profile. It generates no model output and performs no fitting or grading. Fixture prompts visibly identify their test status.

| Browser group | Meaning checked | Result |
| --- | --- | --- |
| Unreadable selected record | A selected whole-JSON fragment is withheld from the primary answer, its literal record remains available in collapsed details, and answer comparison is unavailable without a same/different-answer conclusion. | PASS |
| Readable prior selection | A valid answer retained from response two remains readable when the stopping response three is malformed. Its source distinguishes selection from stopping. | PASS |
| Small positive gain | q=.003 and p=.006 display +0.3 percentage points of improvement against 5.0 required, with -4.7 net. The visible prose explains STOP. Response three is full-run-only; early has two cards and baseline five. Completed logs and technical estimates start collapsed. | PASS |
| Five-response horizon | Both branches use five responses, savings are zero, and the terminal explanation supplies no fabricated next-response estimate. | PASS |
| Zero scores and blank selection | Zero-valued estimates are displayed as supplied data. A blank selected candidate remains unreadable even when parsing succeeded. | PASS |
| Exact tie at response four | Gain exactly equal to cost has net zero and STOP. The explanation says the gain does not exceed the penalty; only the full run has response five. | PASS |
| Tiny positive net | A positive net below one-decimal display precision keeps its sign (+1.0e-7 percentage points) and agrees with Continue. The rounded gain/penalty tie is explained as a precision effect. A following cancellation has no completed savings conclusion. | PASS |
| Mobile theme/readability | At 390×844, dark and light choices survive reload, actual surfaces change, key text meets 4.5:1 contrast, the document has no horizontal overflow, and the checked explanation/answer boxes do not overlap. | PASS |

The minimum measured contrast among the four checked elements was 8.10:1 in dark mode and 5.46:1 in light mode. These checks cover the outcome summary, decision explainer, and two primary answers. They are not a full accessibility certification. Screenshots were also inspected for the desktop stopping explanation and mobile dark view; the outcome, timeline, branch answers, and collapsed logs are visibly separated.

Reproduction commands, from the repository root:

```powershell
python -B demo/early_stopping/reviews/ux_tests_browser.py
python -B -m unittest discover -s demo/early_stopping/tests -p test_*.py -v
```

Evidence:

- [Browser harness](ux_tests_browser.py) and [synthetic fixtures](ux_tests_fixtures.py).
- [Machine-readable results](ux_tests_results.json), including every intercepted API request, measurements, static source hashes, and zero-generation declaration.
- [Desktop stopping explanation](ux_tests_small_gain_stop.png), [unreadable selected record](ux_tests_unreadable_selected.png), [mobile dark](ux_tests_mobile_dark.png), and [mobile light](ux_tests_mobile_light.png).

The initial run stopped because the harness read `inner_text` from intentionally collapsed technical details. The harness was corrected to verify the collapsed state, open the actual disclosure control, read its visible values, and close it again. The final run passed; this was not a production defect.

Tested source SHA-256 values (rechecked unchanged after the run):

```text
app.js      abe9639f20acc2a7b53f29267886ea72c8bef1704287bf059a42f6ad2f16f147
index.html  fc1abebbf21f48715b2b5fb0d4eabe92a137f18f6fa44fe2385cced372a892f4
styles.css  a2d14bbfd74563ad4a8e836915583e325ff9a35fbc11d2defa77829fa80ec958
```

This review exercises restoration of realistic recorded event sequences and human explanations. Live generation, current user-prompt behavior, predictor performance, heuristic-policy copy, system-theme changes, assistive technology, and browsers other than installed Chrome are outside this test result. The coordinator owns the separate real-model/browser verification.
