# Live early-stopping comparison

Type a math prompt, watch local model output arrive, and compare the existing early-stopping policy with a five-response full run.

From the repository root in PowerShell:

```powershell
./demo/early_stopping/start.ps1
```

This starts a hidden local server and opens **http://127.0.0.1:8765** in Chrome when installed, or your default browser otherwise. The first comparison loads the existing local Qwen2.5-0.5B-Instruct snapshot; later comparisons reuse it. No download, training, paid API or research-data write is performed. Requests and recent results stay in server memory, with at most 30 comparisons retained. Server logs contain routes/errors; the app does not intentionally log prompt bodies.

For a visible terminal and direct control:

```powershell
& "$env:LOCALAPPDATA/Programs/Python/Python312/python.exe" -B demo/early_stopping/server.py
```

Stop that server with Ctrl+C. The background launcher's PID/logs are in `%LOCALAPPDATA%/ResearchThesis/early_stopping_demo/`. A process started by the launcher can be stopped with `Stop-Process -Id <its PID>`. To run on CPU, use `start.ps1 -Device cpu` when no demo server is already running. To select another complete local snapshot, invoke `server.py --model-path PATH --device cpu|cuda`; the learned predictor's behavior is only supported for its original model/instrument, so other models are exploratory.

## What the comparison does

- Every response is generated live. Output is streamed while that response is being decoded.
- The early and full arms share exactly the same generated prefix. Both receive the same completed observations until the early policy says STOP. Its answer, prefix and resource counts then freeze. Only the full arm receives later responses, through response five.
- The policy decides **between completed responses**, with a minimum of two. Token streaming makes progress visible; it is not a claim of risk-based stopping inside a response. The common response boundary is valid complete JSON, EOS or 128 new tokens.
- The default learned policy uses the frozen prefix artifact and its existing current/next score formula. The alternative is the existing confidence/stability heuristic. A horizon of five is enforced before learned next-step inference. No reference answers or future observations enter the active controller.
- Completion-token savings equal `(full completion tokens - early completion tokens) / full completion tokens`. All responses including the first, and EOS, are charged. Input tokens are shown separately. Because the demo generates a shared prefix once and continues the full arm, **physical generation for the entire comparison equals the full arm's work**. The displayed saving is the avoided suffix for an early-only run, not a claim that the comparison itself used fewer tokens than the full arm.
- The selected answer is the controller's retained answer, which may come from an earlier response. Matching answers are not a correctness grade. Prompt processing, model loading, and elapsed wall time are not converted into dollars or energy savings.

The fitted probabilities are estimates, not certified confidence levels. The artifact was trained/calibrated on archived math/arithmetic tasks; arbitrary user prompts and other models are outside that empirical validation. On the thesis's learned live panels every task stopped at response two, so the UI demonstrates executable stopping without claiming an adaptive advantage or preserved accuracy.

The server accepts loopback requests only, checks Host/Origin, exposes a fixed static/API route list, and rejects oversized or malformed requests. Closing a browser tab does not cancel a run: reopen the UI to reconnect or use Cancel. Restarting the server discards its in-memory results.

## Reading the results

The result starts with a plain-language summary and a five-response timeline. For example, **stopped after 2 of 5** means the early path skipped three responses while the full path kept going. The percentage below measures generated-text tokens, not accuracy or confidence. Both paths share their starting responses.

The estimated-improvement rule compares two fitted scores: the current retained answer and the answer after one more response. A change from 0.3% to 0.6% is an improvement of **0.3 percentage points**. The default continuation penalty requires more than **5.0 percentage points** of improvement, so that case stops. Low scores can still cause a stop; a stop does not certify that the model solved the question. These predictor scores are separate from confidence claimed by the generating model. Exact ties stop; displayed values are rounded while decisions use full precision.

If a selected response does not match the expected answer format, the primary result says **Answer could not be read reliably**, and answer comparison is unavailable. You can inspect the literal unchanged candidate under **See the retained record**, or expand a response and its original output. The interface does not reinterpret malformed JSON as a verified answer. A correctly formatted earlier candidate remains readable when the controller retains it after a later malformed response. Even readable answers are ungraded.

Use the **Theme** menu for Light, Dark or System. The choice persists in this browser. Completed response logs and technical scores are collapsed by default; the current response opens while text is arriving. The current usability review and real-browser evidence are in [reviews/UX_UPDATE.md](reviews/UX_UPDATE.md). Older reports under `reviews/` record the initial interface at their pinned source hashes.

## Development checks

```powershell
& "$env:LOCALAPPDATA/Programs/Python/Python312/python.exe" -B -m unittest discover -s demo/early_stopping/tests -v
```

Tests inject deterministic generators; they do not run a model or refit the predictor. Real-model/browser verification is recorded separately in `reviews/`.

The isolated usability regressions use synthetic API fixtures and never invoke a model:

```powershell
& "$env:LOCALAPPDATA/Programs/Python/Python312/python.exe" -B demo/early_stopping/reviews/ux_tests_browser.py
```
