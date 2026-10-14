# Live early-stopping backend review

Reviewed 2026-10-13. Final result: **25 tests pass** (14 runtime/state tests and 11 HTTP/SSE tests). No unresolved backend failure remains in the exercised contract.

Run:

```text
python -B demo/early_stopping/tests/run_review_tests.py
```

The final run completed 25 cases in 0.377 seconds, exit code 0. [The execution record](../tests/backend_test_record.json) records the command, environment scope and SHA-256 of the reviewed backend/test sources; [the log](../tests/backend_test.log) preserves individual results. All generation and prediction fixtures are deterministic. No language model was loaded or executed by this review, and no fitting, training, grading or dependency changes occurred.

## Verified behavior

- Exactly one response stream is generated, with five calls through the baseline horizon. The learned arm receives the identical completed observations through response two, then remains frozen while the baseline continues. Subsequent streaming updates address only the baseline.
- The controller recomputes its drift from current/next probabilities. A supplied fictitious `gain=999` does not change the decision. Next-step inference occurs only at eligible responses two through four; terminal response five bypasses that head.
- Retained earlier answers preserve their selected step while costs charge every completed response through the stopping step. Both a heuristic confidence-drop retention branch and latest-nonempty learned selection are tested.
- Actual unequal completion costs `[11, 13, 29, 31, 47]` produce active/baseline totals of 24/131 and saving 107/131. Prompt totals are independently 205/555. Physical generation is 131 completion tokens, rather than the sum of overlapping logical arms. The model's raw `stop` flag does not bypass the policy.
- Cancellation before generation schedules nothing; cancellation between responses schedules no next response; cancellation inside a blocked baseline call unwinds without another call. Decoder/predictor failures propagate and clear generator callbacks/resources.
- Run events have increasing sequence numbers and one terminal event. Worker failures and cancellation release the busy slot; another run can start. Runner completion is separate from the manager's sole terminal emission.
- HTTP tests use an ephemeral loopback server and injected generator. They verify local-only binding, strict Host/Origin/cross-site rejection, static path whitelisting including encoded traversal attempts, JSON content/type/encoding/size validation, unknown runs, invalid event cursors, overlap rejection, cancellation and failure recovery.
- SSE output equals the retained event history. `Last-Event-ID` resumes after the requested sequence. Disconnecting a stream leaves the run active; reconnecting recovers all events. Cancellation remains an explicit action.

These checks establish the shared-prefix comparison contract. The active-arm savings are the policy comparison against that continuation; the demo still physically generates the full baseline.

## Findings resolved during review

1. Malformed `Origin: http://[` could raise from `urlsplit` before a bounded rejection response. The coordinator added a parse guard; the independent HTTP regression now receives 403 and schedules no work.
2. Early POST rejections closed Windows sockets while small request bodies remained unread. Independent runs intermittently received `WinError 10053` instead of the intended 403/415 response. The coordinator added a bounded small-body rejection drain with a short timeout and connection-close handling. Oversized/unbounded bodies are not drained. The final HTTP regression passes.

Only tests, test logs/records and this note were authored by this reviewer. Runtime/server fixes belong to the coordinator.

## Scope remaining with the coordinator

Actual model streaming, device behavior, browser rendering and end-to-end latency require the coordinator's real-model/browser smoke test. This suite exercises their generator interface and event semantics without claiming physical GPU measurements or answer correctness. Frozen thesis controllers/predictors were reused without modification.
