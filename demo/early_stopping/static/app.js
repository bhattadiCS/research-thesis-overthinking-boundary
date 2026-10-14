"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const formatNumber = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });
  const armNames = ["early", "baseline"];
  const eventTypes = ["run_started", "status", "response_start", "token", "response_end", "decision", "complete", "cancelled", "error"];
  const examples = {
    shopping: { domain: "gsm8k", prompt: "A shop sells notebooks for $3 each. Maya buys 4 notebooks and pays with $20. How much change should she receive?" },
    equation: { domain: "math", prompt: "Solve for x: 3(x - 2) = 2x + 5." },
    speed: { domain: "gsm8k", prompt: "A train travels 90 miles in 1.5 hours. What is its average speed in miles per hour?" },
    logarithm: { domain: "math", prompt: "Find all real solutions of\n\nlog_2(x^2 - 7x + 10) + log_2(x^2 - 7x + 12) = -1.\n\nGive the complete solution set in exact form, not decimals, with a brief justification." },
  };
  const makeArm = () => ({ cards: new Map(), stopped: false, steps: 0, selectedStep: null, answer: null, completionTokens: null, promptTokens: null, elapsedSeconds: null });
  const state = { server: null, statusFailed: false, checking: false, active: false, starting: false, restoring: false, restoreAttempted: false, viewVersion: 0, cancelling: false, runId: null, lastSeq: -1, source: null, snapshotInFlight: false, maxSteps: 5, stepCost: 0.05, policy: "learned", decision: null, terminal: false, arms: { early: makeArm(), baseline: makeArm() } };

  const selectedPolicy = () => document.querySelector('input[name="policy"]:checked').value;
  const finiteNumber = (value) => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));
  const numberText = (value) => finiteNumber(value) ? formatNumber.format(Number(value)) : "—";
  const timeText = (value) => !finiteNumber(value) ? "—" : Number(value) < 1 ? `${Math.round(Number(value) * 1000)} ms` : `${Number(value).toFixed(1)} s`;
  const percentage = (value) => finiteNumber(value) ? `${(Number(value) * 100).toFixed(1)}%` : "—";
  const announce = (message) => { if (!state.restoring) $("announcer").textContent = message; };
  const points = (value, signed = false) => {
    if (!finiteNumber(value)) return "Unavailable";
    const scaled = Number(value) * 100;
    const text = Math.abs(scaled) > 0 && Math.abs(scaled) < 0.05 ? scaled.toPrecision(2) : scaled.toFixed(1);
    return `${signed && scaled > 0 ? "+" : ""}${text} percentage points`;
  };

  function setupTheme() {
    const selector = $("theme-select");
    if (!selector) return;
    let preferred = "system";
    try { preferred = localStorage.getItem("early-stopping-theme") || "system"; } catch (_) { /* Browser storage may be disabled. */ }
    const apply = (choice) => {
      const theme = ["system", "light", "dark"].includes(choice) ? choice : "system";
      document.documentElement.dataset.theme = theme;
      document.documentElement.style.colorScheme = theme === "system" ? "light dark" : theme;
      selector.value = theme;
      try { localStorage.setItem("early-stopping-theme", theme); } catch (_) { /* Theme still works for this page. */ }
    };
    apply(preferred);
    selector.addEventListener("change", () => apply(selector.value));
  }

  function readableAnswer(arm) {
    const card = arm.cards.get(arm.selectedStep);
    return Boolean(arm.answer !== null && String(arm.answer).trim() && card?.parseSuccess === true);
  }

  function renderTimeline() {
    const timeline = $("run-timeline");
    if (!timeline) return;
    const cutoff = state.arms.early.stopped ? state.arms.early.steps : null;
    for (const node of timeline.querySelectorAll("[data-step]")) {
      const step = Number(node.dataset.step);
      const card = state.arms.baseline.cards.get(step);
      const baselineOnly = cutoff !== null && step > cutoff;
      node.classList.toggle("is-shared", !baselineOnly);
      node.classList.toggle("is-baseline-only", baselineOnly);
      node.classList.toggle("is-stop", cutoff === step && cutoff < state.maxSteps);
      node.classList.toggle("is-complete", Boolean(card?.complete));
      node.classList.toggle("is-active", Boolean(card && !card.complete && !state.terminal));
      const label = node.querySelector(".timeline-status");
      if (label) label.textContent = state.terminal && !card?.complete ? "Not completed" : cutoff === step && cutoff < state.maxSteps ? "Early stop" : baselineOnly ? "Full run only" : card?.complete ? "Both runs" : card && !state.terminal ? "Writing now" : "Waiting";
    }
  }

  function renderOutcome(summary = null) {
    const heading = $("outcome-heading"), explanation = $("outcome-summary"), quality = $("outcome-quality");
    if (!heading || !explanation || !quality) return;
    $("outcome-card").dataset.quality = "pending";
    const early = state.arms.early, full = state.arms.baseline;
    if (summary) {
      const skipped = full.steps - early.steps;
      heading.textContent = skipped > 0 ? `Stopped after ${early.steps} of ${full.steps} responses` : `Both runs used all ${full.steps} responses`;
      explanation.textContent = skipped > 0 ? `Early stopping skipped ${skipped} ${skipped === 1 ? "response" : "responses"}. The full run kept going to show what happened next.` : "The policy did not stop before the limit, so no later responses were skipped.";
      const readable = readableAnswer(early) && readableAnswer(full);
      quality.textContent = !readable ? "Answer comparison unavailable. The model's final output could not be read reliably, so this run does not establish whether continuing helped or hurt the answer." : summary.answers_match ? "Both runs retained the same answer text. The app has not checked whether that answer is mathematically correct." : "The retained answer text changed during the extra responses. The app has not checked which answer is correct, so a change alone does not demonstrate overthinking.";
      $("outcome-card").dataset.quality = readable ? "ungraded" : "unreadable";
    } else if (early.stopped && early.steps < state.maxSteps) {
      heading.textContent = `Early stopping finished at response ${early.steps}`;
      explanation.textContent = `Its answer and token count are now frozen. The full run is continuing toward response ${state.maxSteps}.`;
      quality.textContent = readableAnswer(early) ? "The early answer is ungraded. Wait for the full run before comparing the final results." : "The early answer could not be read reliably. Stopping does not mean that the problem has been solved.";
    } else if (state.active || state.starting) {
      heading.textContent = "The model is generating a shared starting sequence";
      explanation.textContent = "Both sides see the same responses until the early-stopping policy says stop. The first decision is allowed after two responses.";
      quality.textContent = "Watch how much text is generated and which answer each run retains. Correctness is not graded automatically.";
    } else {
      heading.textContent = "What will you compare?";
      explanation.textContent = "One run may stop early; the other continues through five responses. Their shared starting sequence is generated once.";
      quality.textContent = "Less generated text is a budget difference. To demonstrate overthinking, you would also need to establish that later responses made an answer worse.";
    }
    renderTimeline();
  }

  function setRunStatus(message, type = "idle") {
    $("run-status-text").textContent = message;
    $("run-status").dataset.state = type;
  }

  function showRunError(message) {
    $("run-error").textContent = message;
    $("run-error").hidden = false;
  }

  async function fetchJSON(url, options = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 20000);
    try {
      const response = await fetch(url, { ...options, signal: controller.signal, headers: { Accept: "application/json", ...(options.body ? { "Content-Type": "application/json" } : {}), ...options.headers } });
      const contentType = response.headers.get("content-type") || "";
      const data = contentType.includes("json") ? await response.json() : null;
      if (!response.ok) throw new Error(data?.error || data?.message || `The local server returned HTTP ${response.status}.`);
      if (data === null) throw new Error("The local server returned an unexpected response.");
      return data;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("The local server did not respond in time. Try again when it is ready.");
      if (error instanceof TypeError) throw new Error("Could not reach the local server. Check that the demo server is still running.");
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }

  function updateAvailability() {
    const model = state.server?.model;
    const predictor = state.server?.predictor;
    const busyElsewhere = Boolean(state.server?.busy && !state.active && !state.starting && !state.restoring);
    const predictorMissing = selectedPolicy() === "learned" && predictor && !predictor.available;
    let notice = "";
    let path = "";
    if (state.statusFailed) notice = "Could not reach the local server. Keep this page open and make sure the demo server is running.";
    else if (model && !model.available) { notice = "The local model is unavailable. Start the demo server with a valid model folder."; path = model.path || ""; }
    else if (busyElsewhere) notice = "The local model is busy with another run. This comparison will be available when it finishes.";
    else if (predictorMissing) notice = "The learned predictor is unavailable. Choose Confidence heuristic to run without it, or restore the predictor artifact.";
    const noticeElement = $("system-notice");
    noticeElement.replaceChildren();
    if (notice) {
      noticeElement.append(document.createTextNode(notice));
      if (path) { const code = document.createElement("code"); code.textContent = path; noticeElement.append(code); }
    }
    noticeElement.hidden = !notice;
    $("run-button").disabled = state.active || state.starting || state.restoring || !state.server || state.statusFailed || !model?.available || Boolean(predictorMissing) || busyElsewhere;
    $("run-button").classList.toggle("is-running", state.active || state.starting || state.restoring);
    $("run-button").querySelector("span").textContent = state.restoring ? "Restoring…" : state.starting ? "Starting…" : state.active ? "Running…" : "Run comparison";
    $("prompt").readOnly = state.active || state.starting || state.restoring;
    $("domain").disabled = state.active || state.starting || state.restoring;
    document.querySelectorAll('input[name="policy"], .example-chip').forEach((input) => { input.disabled = state.active || state.starting || state.restoring; });
    $("cancel-button").hidden = !state.active || !state.runId;
    $("cancel-button").disabled = state.cancelling;
    $("cancel-button").textContent = state.cancelling ? "Cancelling…" : "Cancel run";
  }

  async function checkStatus() {
    if (state.checking) return;
    state.checking = true;
    try {
      state.server = await fetchJSON("/api/status");
      state.statusFailed = false;
      const model = state.server.model || {};
      $("server-status").dataset.state = !model.available ? "error" : state.server.busy ? "busy" : "ready";
      $("server-status-text").textContent = !model.available ? "Local model unavailable" : state.server.busy ? "Local inference running" : model.loaded ? "Local model ready" : "Local model found · loads on first run";
      const name = model.name || (model.path ? String(model.path).split(/[\\/]/).filter(Boolean).pop() : "Local model");
      $("model-name").textContent = name;
      $("model-name").title = model.path || name;
      $("model-device").textContent = model.device ? String(model.device).toUpperCase() : "Device unknown";
      const predictor = state.server.predictor || {};
      $("predictor-detail").textContent = predictor.available ? `Predictor ${predictor.sha256 ? String(predictor.sha256).slice(0, 10) : "ready"}` : "Predictor unavailable";
      $("predictor-detail").title = predictor.sha256 || "";
      void maybeRestoreRun(state.server.run);
    } catch (_) {
      state.statusFailed = true;
      $("server-status").dataset.state = "error";
      $("server-status-text").textContent = "Local server unavailable";
    } finally {
      state.checking = false;
      updateAvailability();
    }
  }

  function updatePolicyDescription() {
    const learned = selectedPolicy() === "learned";
    $("policy-description").textContent = learned ? "A separate predictor estimates whether another response will improve the answer enough to justify continuing." : "Stops when the model repeats an answer with high reported confidence, or meets another confidence/stability condition. Reported confidence can be wrong.";
    $("continuation-cost").hidden = !learned;
    if (!state.runId) $("early-policy-label").textContent = learned ? "Estimated improvement rule" : "Confidence and stability rule";
    updateAvailability();
  }

  function updatePromptCount() {
    $("prompt-count").textContent = `${formatNumber.format($("prompt").value.length)} / 3,000`;
    $("prompt-error").hidden = true;
    $("prompt").removeAttribute("aria-invalid");
  }

  function setBadge(arm, text, type) {
    $(`${arm}-badge`).textContent = text;
    $(`${arm}-badge`).dataset.state = type;
  }

  function renderMetrics(armName) {
    const arm = state.arms[armName];
    const steps = $(`${armName}-steps`);
    steps.replaceChildren(document.createTextNode(numberText(arm.steps) + " "));
    const limit = document.createElement("span"); limit.textContent = `/ ${state.maxSteps}`; steps.append(limit);
    $(`${armName}-completion`).textContent = numberText(arm.completionTokens);
    $(`${armName}-prompt`).textContent = numberText(arm.promptTokens);
    $(`${armName}-time`).textContent = timeText(arm.elapsedSeconds);
  }

  function renderAnswer(armName, stoppedAt = null) {
    const arm = state.arms[armName];
    const waiting = arm.answer === null;
    const readable = readableAnswer(arm);
    $(`${armName}-answer`).textContent = waiting ? "Waiting for an answer…" : readable ? String(arm.answer) : String(arm.answer).trim() ? "Answer could not be read reliably" : "No answer returned";
    let source = arm.selectedStep ? `Retained from response ${arm.selectedStep}` : waiting ? "Responses will appear below as they are generated." : "The model did not return a nonempty candidate.";
    if (finiteNumber(stoppedAt)) source += ` · ${armName === "early" ? "stopped after" : "generated"} ${stoppedAt} ${Number(stoppedAt) === 1 ? "response" : "responses"}`;
    $(`${armName}-answer-source`).textContent = source;
    $(`${armName}-format-note`).hidden = waiting || readable;
    const literal = $(`${armName}-retained-answer`);
    if (literal) literal.textContent = waiting ? "" : String(arm.answer) || "(empty)";
    const retainedDetails = $(`${armName}-retained-details`);
    if (retainedDetails) retainedDetails.hidden = waiting || readable;
    $(`${armName}-answer`).closest(".selected-answer").classList.toggle("is-unreadable", !waiting && !readable);
  }

  function resetComparison(policy) {
    state.viewVersion += 1;
    if (state.source) state.source.close();
    state.source = null;
    state.runId = null;
    state.lastSeq = -1;
    state.terminal = false;
    state.cancelling = false;
    state.maxSteps = 5;
    state.stepCost = 0.05;
    state.policy = policy;
    state.decision = null;
    state.arms = { early: makeArm(), baseline: makeArm() };
    $("early-policy-label").textContent = policy === "learned" ? "Estimated improvement rule" : "Confidence and stability rule";
    $("result-strip").hidden = true;
    $("restored-note").hidden = true;
    $("run-error").hidden = true;
    $("decision-box").hidden = true;
    for (const arm of armNames) {
      $(`${arm}-responses`).replaceChildren();
      $(`${arm}-answer`).textContent = "—";
      $(`${arm}-answer-source`).textContent = "Waiting for the first candidate.";
      $(`${arm}-format-note`).hidden = true;
      if ($(`${arm}-retained-details`)) $(`${arm}-retained-details`).hidden = true;
      $(`${arm}-panel`).dataset.frozen = "false";
      $(`${arm}-panel`).dataset.active = "false";
      setBadge(arm, "Waiting", "waiting");
      renderMetrics(arm);
    }
    renderOutcome();
  }

  function validArms(event) {
    return (Array.isArray(event.arms) ? event.arms : []).filter((arm) => armNames.includes(arm) && !(arm === "early" && state.arms.early.stopped));
  }

  function ensureCard(armName, step, shared) {
    const arm = state.arms[armName];
    if (arm.cards.has(step)) return arm.cards.get(step);
    const node = $("response-template").content.firstElementChild.cloneNode(true);
    node.querySelector("h4").textContent = `Response ${step}`;
    node.querySelector(".shared-badge").hidden = !shared;
    node.querySelector("pre").setAttribute("aria-label", `${armName === "early" ? "Early stopping" : "Full budget"}, response ${step}, generated text`);
    const card = { node, step, complete: false, tokens: null, promptTokens: null, parseSuccess: null, pre: node.querySelector("pre") };
    const list = $(`${armName}-responses`);
    list.querySelector(".empty-state")?.remove();
    list.append(node);
    arm.cards.set(step, card);
    return card;
  }

  function updateTokenCounts(armName) {
    const arm = state.arms[armName];
    const cards = [...arm.cards.values()];
    if (cards.some((card) => finiteNumber(card.tokens))) arm.completionTokens = cards.reduce((sum, card) => sum + (finiteNumber(card.tokens) ? Number(card.tokens) : 0), 0);
    if (cards.some((card) => finiteNumber(card.promptTokens))) arm.promptTokens = cards.reduce((sum, card) => sum + (finiteNumber(card.promptTokens) ? Number(card.promptTokens) : 0), 0);
    arm.steps = cards.filter((card) => card.complete).length;
    renderMetrics(armName);
  }

  function setLiveText(card, text) {
    const nearBottom = card.pre.scrollHeight - card.pre.clientHeight - card.pre.scrollTop < 32;
    card.pre.textContent = typeof text === "string" ? text : String(text ?? "");
    if (nearBottom) card.pre.scrollTop = card.pre.scrollHeight;
  }

  function responseStart(event) {
    const arms = validArms(event);
    const shared = event.arms?.includes("early") && event.arms?.includes("baseline");
    for (const arm of arms) {
      const card = ensureCard(arm, event.step, shared);
      if (card.node.tagName === "DETAILS") card.node.open = true;
      card.node.querySelector(".raw-response").open = true;
      setBadge(arm, "Generating", "generating");
      $(`${arm}-panel`).dataset.active = "true";
    }
    setRunStatus(`Writing response ${event.step} of ${state.maxSteps}${shared ? " · both runs" : " · full run only"}`, "running");
    announce(`Response ${event.step} is generating${shared ? " for both arms" : " for the full budget"}.`);
    renderOutcome();
  }

  function responseToken(event) {
    for (const arm of validArms(event)) {
      const card = ensureCard(arm, event.step, event.arms?.length === 2);
      // The server sends FULL cumulative decoded text. Never concatenate chunks.
      setLiveText(card, event.text);
      if (finiteNumber(event.generated_tokens)) card.tokens = Number(event.generated_tokens);
      card.node.querySelector(".response-token-count").textContent = finiteNumber(card.tokens) ? `${numberText(card.tokens)} tokens` : "";
      updateTokenCounts(arm);
    }
  }

  function responseEnd(event) {
    const observation = event.observation || {};
    for (const arm of validArms(event)) {
      const card = ensureCard(arm, event.step, event.arms?.length === 2);
      card.complete = true;
      card.parseSuccess = observation.parse_success;
      const unstructured = observation.parse_success === false || observation.parse_success === 0;
      card.node.dataset.unstructured = String(unstructured);
      card.node.querySelector(".response-format-note").hidden = !unstructured;
      if (typeof observation.raw_text === "string") setLiveText(card, observation.raw_text);
      if (finiteNumber(observation.generated_tokens)) card.tokens = Number(observation.generated_tokens);
      if (finiteNumber(observation.prompt_tokens)) card.promptTokens = Number(observation.prompt_tokens);
      card.node.querySelector(".response-state").textContent = "Complete";
      card.node.querySelector(".response-token-count").textContent = finiteNumber(card.tokens) ? `${numberText(card.tokens)} tokens` : "";
      const thought = typeof observation.thought === "string" ? observation.thought.trim() : "";
      const answer = observation.answer === null || observation.answer === undefined ? "" : String(observation.answer).trim();
      const summary = card.node.querySelector(".response-summary");
      summary.hidden = !unstructured && !thought;
      summary.querySelector("p").textContent = unstructured ? "The model did not return the expected answer format. Open Original output to inspect exactly what it wrote." : thought;
      const candidate = card.node.querySelector(".response-candidate");
      candidate.hidden = unstructured || !answer;
      candidate.querySelector("p").textContent = unstructured ? "" : answer;
      const raw = card.node.querySelector(".raw-response");
      raw.querySelector("summary").firstChild.nodeValue = "Original output ";
      // Keep raw output available; parsed fields become the compact completed view.
      raw.open = false;
      if (card.node.tagName === "DETAILS") card.node.open = false;
      const footer = card.node.querySelector(".response-footer");
      const details = [];
      if (finiteNumber(card.promptTokens)) details.push(`${numberText(card.promptTokens)} input tokens`);
      if (finiteNumber(event.step_seconds)) details.push(timeText(event.step_seconds));
      if (unstructured) details.push("Answer format was not recognized");
      if (finiteNumber(observation.confidence)) details.push(`Reported confidence ${Number(observation.confidence).toFixed(0)}/100`);
      footer.textContent = details.join(" · "); footer.hidden = !details.length;
      updateTokenCounts(arm);
      $(`${arm}-panel`).dataset.active = "false";
    }
    renderTimeline();
  }

  function reasonText(decision) {
    const reason = String(decision.reason || "");
    const key = reason.toLowerCase();
    if (key.includes("horizon") || key.includes("max_step") || Number(decision.step) >= state.maxSteps) return "The five-response limit has been reached. No next-response estimate is needed.";
    if (key.includes("floor") || key.includes("min_step") || key.includes("minimum")) return "The policy continues until the two-response minimum is reached.";
    if (key.includes("drift") && finiteNumber(decision.q_current) && finiteNumber(decision.p_next)) {
      return decision.stop ? "The estimated improvement is at or below the continuation penalty." : "The estimated improvement exceeds the continuation penalty.";
    }
    if (key.includes("stable") || key.includes("stability")) return "The parsed answer and reported confidence meet the heuristic’s stability condition.";
    if (key.includes("confidence_drop")) return "Reported confidence fell. The heuristic stops and retains the previous answer.";
    if (key.includes("churn") || key.includes("answer_change") || key.includes("wobble")) return "The answer changed repeatedly. The heuristic stops and retains an already observed answer.";
    if (key.includes("untrusted") || key.includes("incomplete")) return "The response format was not recognized, so the confidence rule cannot use its confidence claim to stop.";
    if (reason.includes(" ") && !reason.includes("_")) return reason;
    return reason ? reason.replace(/_/g, " ").replace(/^./, (letter) => letter.toUpperCase()) + "." : decision.stop ? "The policy selected this response as its stopping point." : "The policy requests another response.";
  }

  function showDecision(decision) {
    state.decision = decision;
    const stop = Boolean(decision.stop);
    $("decision-box").hidden = false;
    $("decision-box").dataset.stop = String(stop);
    $("decision-icon").textContent = stop ? "■" : "→";
    $("decision-title").textContent = Number(decision.step) >= state.maxSteps ? "Response limit reached" : stop ? "Stop here" : "Keep going";
    $("decision-step").textContent = `After response ${decision.step}`;
    $("decision-reason").textContent = reasonText(decision);
    const hasEstimates = Number(decision.step) < state.maxSteps && [decision.q_current, decision.p_next, decision.estimated_drift].every(finiteNumber);
    if ($("estimate-details")) $("estimate-details").hidden = !hasEstimates;
    $("estimate-grid").hidden = !hasEstimates;
    $("estimate-note").hidden = !hasEstimates;
    $("estimate-note").textContent = `These are fitted predictor scores, separate from confidence claimed by the generating model. The demo penalty is ${state.stepCost} reward units, equivalent to ${points(state.stepCost)} of required improvement. Displayed values are rounded; the rule uses full precision. Scores and answer correctness have not been certified for this prompt.`;
    $("estimate-current").textContent = percentage(decision.q_current);
    $("estimate-next").textContent = percentage(decision.p_next);
    $("estimate-drift").textContent = points(decision.estimated_drift, true);
    if ($("estimate-gain")) $("estimate-gain").textContent = hasEstimates ? points(Number(decision.p_next) - Number(decision.q_current), true) : "Unavailable";
    if ($("estimate-required")) $("estimate-required").textContent = hasEstimates ? points(state.stepCost) : "Unavailable";
    if ($("decision-explainer")) {
      if (hasEstimates) {
        const gain = Number(decision.p_next) - Number(decision.q_current);
        const roundedTie = gain !== state.stepCost && (gain * 100).toFixed(1) === (state.stepCost * 100).toFixed(1);
        $("decision-explainer").textContent = `Predictor score: ${percentage(decision.q_current)} now → ${percentage(decision.p_next)} after another response. Expected improvement: ${points(gain, true)}. Continuing requires more than ${points(state.stepCost)}. ${stop ? "The estimated gain does not exceed the penalty, so the rule stops." : "The estimated gain clears that penalty, so the rule keeps going."}${roundedTie ? " The displayed gain and penalty look tied because of rounding; the rule uses full precision." : ""} It can stop even when both scores are low; stopping does not certify the answer.`;
      } else $("decision-explainer").textContent = Number(decision.step) >= state.maxSteps ? "This is the common limit for both runs. No prediction beyond response five is made." : state.policy === "heuristic" ? "This rule uses reported confidence and answer changes; it does not estimate current or next-response correctness." : Number(decision.step) < 2 ? "The two-response minimum must be reached before this rule can decide to stop." : "No current/next predictor estimates were supplied for this decision. The recorded reason is shown above.";
    }
  }

  function policyDecision(event) {
    const name = event.arm;
    if (!armNames.includes(name) || (name === "early" && state.arms.early.stopped)) return;
    const arm = state.arms[name];
    const decision = event.decision || {};
    const totals = event.totals || {};
    if (finiteNumber(totals.steps)) arm.steps = Number(totals.steps);
    if (finiteNumber(totals.completion_tokens)) arm.completionTokens = Number(totals.completion_tokens);
    if (finiteNumber(totals.prompt_tokens)) arm.promptTokens = Number(totals.prompt_tokens);
    if (finiteNumber(event.elapsed_seconds)) arm.elapsedSeconds = Number(event.elapsed_seconds);
    if (Object.hasOwn(decision, "selected_answer")) arm.answer = decision.selected_answer;
    if (Object.hasOwn(decision, "selected_step")) arm.selectedStep = finiteNumber(decision.selected_step) ? Number(decision.selected_step) : null;
    renderMetrics(name);
    renderAnswer(name, decision.stop ? decision.step : null);
    if (name === "early") showDecision(decision);
    if (decision.stop) {
      arm.stopped = true;
      setBadge(name, Number(decision.step) >= state.maxSteps ? "Limit reached" : `Stopped at ${decision.step}`, "stopped");
      $(`${name}-panel`).dataset.frozen = "true";
      $(`${name}-panel`).dataset.active = "false";
      if (name === "early") announce(`Early policy stopped after response ${decision.step}. The full budget continues.`);
    } else setBadge(name, "Continuing", "generating");
    renderOutcome();
  }

  function applySummary(summary, partial = false) {
    if (!summary) return;
    for (const name of armNames) {
      const data = summary[name];
      if (!data) continue;
      const arm = state.arms[name];
      if (Object.hasOwn(data, "answer")) arm.answer = data.answer;
      if (Object.hasOwn(data, "selected_step")) arm.selectedStep = finiteNumber(data.selected_step) ? Number(data.selected_step) : null;
      if (finiteNumber(data.stopped_at_step)) arm.steps = Number(data.stopped_at_step);
      if (finiteNumber(data.completion_tokens)) arm.completionTokens = Number(data.completion_tokens);
      if (finiteNumber(data.prompt_tokens)) arm.promptTokens = Number(data.prompt_tokens);
      if (finiteNumber(data.elapsed_seconds)) arm.elapsedSeconds = Number(data.elapsed_seconds);
      renderMetrics(name);
      renderAnswer(name, data.stopped_at_step);
      if (!partial) {
        arm.stopped = true;
        $(`${name}-panel`).dataset.frozen = "true";
        $(`${name}-panel`).dataset.active = "false";
        setBadge(name, name === "baseline" ? "Complete" : Number(data.stopped_at_step) < state.maxSteps ? `Stopped at ${data.stopped_at_step}` : "Limit reached", "complete");
      }
    }
    if (partial) return;
    let saving = summary.completion_savings;
    let saved = summary.completion_tokens_saved;
    if (!finiteNumber(saving) && Number(summary.baseline?.completion_tokens) > 0 && finiteNumber(summary.early?.completion_tokens)) saving = 1 - Number(summary.early.completion_tokens) / Number(summary.baseline.completion_tokens);
    if (!finiteNumber(saved) && finiteNumber(summary.baseline?.completion_tokens) && finiteNumber(summary.early?.completion_tokens)) saved = Number(summary.baseline.completion_tokens) - Number(summary.early.completion_tokens);
    $("result-savings").textContent = percentage(saving);
    $("result-token-detail").textContent = finiteNumber(saved) ? `${numberText(saved)} of ${numberText(summary.baseline?.completion_tokens)} tokens` : "";
    const comparable = readableAnswer(state.arms.early) && readableAnswer(state.arms.baseline);
    $("answer-match").textContent = !comparable ? "Answer comparison unavailable" : summary.answers_match ? "Same retained answer text · ungraded" : "Retained answer text changed · ungraded";
    $("answer-match").dataset.match = comparable ? String(Boolean(summary.answers_match)) : "unavailable";
    $("answer-match").title = comparable ? "Whitespace/case-normalized string comparison only. This is not mathematical equivalence or a correctness grade." : "At least one retained candidate was not read from the expected response format. The literal record is available under Original retained record.";
    $("result-note").textContent = "This percentage compares generated text, not answer accuracy. Tokens are small pieces of text.";
    $("physical-tokens").textContent = finiteNumber(summary.physical_completion_tokens) ? `The comparison itself generated ${numberText(summary.physical_completion_tokens)} tokens to show the full continuation. An early-only run would use the shorter budget.` : "";
    $("result-strip").hidden = false;
    renderOutcome(summary);
  }

  function finishRun(type, event) {
    state.active = false;
    state.starting = false;
    state.cancelling = false;
    state.terminal = true;
    state.source?.close(); state.source = null;
    applySummary(event.summary, type !== "complete");
    if (type === "complete") {
      setRunStatus("Comparison complete", "complete");
      announce("Comparison complete. Token totals and selected answers are ready.");
    } else {
      const message = event.message || (type === "cancelled" ? "The run was cancelled." : "The run could not finish.");
      setRunStatus(type === "cancelled" ? "Run cancelled · partial results retained" : "Run could not finish", type === "cancelled" ? "idle" : "error");
      if (type === "error") showRunError(message);
      for (const arm of armNames) {
        $(`${arm}-panel`).dataset.active = "false";
        if (!state.arms[arm].stopped) setBadge(arm, type === "cancelled" ? "Cancelled" : "Interrupted", type === "cancelled" ? "waiting" : "error");
        for (const card of state.arms[arm].cards.values()) if (!card.complete) card.node.querySelector(".response-state").textContent = "Interrupted";
      }
      announce(message);
      if ($("outcome-heading")) {
        $("outcome-heading").textContent = type === "cancelled" ? "Comparison cancelled" : "Comparison could not finish";
        $("outcome-summary").textContent = "Partial responses are retained below. A complete full run is needed to calculate the final token difference.";
        $("outcome-quality").textContent = "No final answer-quality or savings conclusion is available for this interrupted run.";
      }
      renderTimeline();
    }
    updateAvailability();
    void checkStatus();
  }

  function handleEvent(event) {
    if (!event || typeof event !== "object" || (event.run_id && event.run_id !== state.runId)) return;
    if (finiteNumber(event.seq)) {
      const seq = Number(event.seq);
      if (seq <= state.lastSeq) return;
      state.lastSeq = seq;
    }
    if (state.terminal) return;
    switch (event.type) {
      case "run_started":
        state.maxSteps = Number(event.max_steps) || 5;
        state.stepCost = finiteNumber(event.step_cost) ? Number(event.step_cost) : 0.05;
        state.policy = event.policy || state.policy;
        $("early-policy-label").textContent = state.policy === "heuristic" ? "Confidence and stability rule" : "Estimated improvement rule";
        for (const arm of armNames) renderMetrics(arm);
        setRunStatus("Preparing local inference", "running");
        renderOutcome();
        break;
      case "status": setRunStatus(event.message || "Local inference running", "running"); break;
      case "response_start": responseStart(event); break;
      case "token": responseToken(event); break;
      case "response_end": responseEnd(event); break;
      case "decision": policyDecision(event); break;
      case "complete": finishRun("complete", event); break;
      case "cancelled": finishRun("cancelled", event); break;
      case "error": finishRun("error", event); break;
      default: break;
    }
  }

  async function checkSnapshot() {
    if (state.snapshotInFlight || !state.runId || !state.active) return;
    const requestedRunId = state.runId;
    state.snapshotInFlight = true;
    try {
      const run = await fetchJSON(`/api/runs/${encodeURIComponent(requestedRunId)}`);
      if (state.runId !== requestedRunId) return;
      if (Array.isArray(run.events)) for (const event of run.events) handleEvent(event);
      if (!state.terminal && ["complete", "completed", "cancelled", "error", "failed"].includes(run.status)) finishRun(["complete", "completed"].includes(run.status) ? "complete" : run.status === "cancelled" ? "cancelled" : "error", { summary: run.summary, message: run.error });
    } catch (_) {
      if (state.active) setRunStatus("Connection interrupted · trying to reconnect", "running");
    } finally { state.snapshotInFlight = false; }
  }

  async function maybeRestoreRun(reference) {
    if (state.restoreAttempted || state.restoring || state.starting || state.active || state.runId || !reference?.id) return;
    state.restoreAttempted = true;
    state.restoring = true;
    const requestedId = String(reference.id);
    const requestedVersion = state.viewVersion;
    updateAvailability();
    setRunStatus("Restoring the latest run from this local server…");
    try {
      const run = await fetchJSON(`/api/runs/${encodeURIComponent(requestedId)}`);
      // Never let a late snapshot replace a newer user-requested comparison.
      if (state.viewVersion !== requestedVersion || state.starting || state.runId || state.active) return;
      const config = run.config || {};
      const policy = ["learned", "heuristic"].includes(config.policy) ? config.policy : "learned";
      resetComparison(policy);
      $("prompt").value = String(run.prompt || "");
      $("domain").value = config.domain === "math" ? "math" : "gsm8k";
      document.querySelectorAll('input[name="policy"]').forEach((input) => { input.checked = input.value === policy; });
      updatePromptCount();
      state.runId = String(run.id || requestedId);
      state.active = true;
      state.starting = false;
      updatePolicyDescription();
      if (Array.isArray(run.events)) for (const event of run.events) handleEvent(event);
      const terminal = ["complete", "completed", "cancelled", "error", "failed"].includes(run.status);
      if (!state.terminal && terminal) finishRun(["complete", "completed"].includes(run.status) ? "complete" : run.status === "cancelled" ? "cancelled" : "error", { summary: run.summary, message: run.error });
      $("restored-note").hidden = false;
      if (state.active) {
        $("restored-note").textContent = "Reconnected to the current live run. Earlier responses were restored from its event history; reopening this page started no new generation.";
        setRunStatus("Reconnected to the current live run", "running");
        connectEvents(state.runId);
      } else {
        $("restored-note").textContent = "Restored a previous run from this local server. These are saved results, not a new live generation. Choose Run comparison to start again.";
        setRunStatus(run.status === "cancelled" ? "Previous cancelled run restored" : ["error", "failed"].includes(run.status) ? "Previous interrupted run restored" : "Previous comparison restored", ["error", "failed"].includes(run.status) ? "error" : "complete");
      }
    } catch (error) {
      if (state.viewVersion === requestedVersion && !state.active) {
        setRunStatus("Could not restore the previous run", "error");
        showRunError(error.message);
      }
    } finally {
      state.restoring = false;
      updateAvailability();
      announce(state.active ? "Reconnected to the current live comparison." : "Previous run restored. The run controls are ready.");
    }
  }

  function connectEvents(runId) {
    const source = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`);
    state.source = source;
    const receive = (message) => {
      if (typeof message.data !== "string") return;
      try { handleEvent(JSON.parse(message.data)); }
      catch (error) { console.error("Could not read a comparison event", error); showRunError("A server event could not be read. Reconnecting to the stored run state."); void checkSnapshot(); }
    };
    source.onmessage = receive;
    for (const type of eventTypes) source.addEventListener(type, receive);
    source.onopen = () => { if (state.active && state.lastSeq < 0) setRunStatus("Connected · waiting for local inference", "running"); };
    source.onerror = () => { if (state.active && state.source === source) { setRunStatus("Connection interrupted · reconnecting", "running"); void checkSnapshot(); } };
  }

  async function runComparison(event) {
    event.preventDefault();
    if (state.active || state.starting || state.restoring) return;
    const prompt = $("prompt").value.trim();
    if (!prompt || prompt.length > 3000) {
      $("prompt-error").textContent = !prompt ? "Enter a problem, or choose one of the examples." : "Keep the problem within 3,000 characters.";
      $("prompt-error").hidden = false;
      $("prompt").setAttribute("aria-invalid", "true");
      $("prompt").focus();
      return;
    }
    if ($("run-button").disabled) return;
    const policy = selectedPolicy();
    const domain = $("domain").value;
    resetComparison(policy);
    state.starting = true;
    updateAvailability();
    renderOutcome();
    setRunStatus("Starting a real local run…", "running");
    try {
      const result = await fetchJSON("/api/runs", { method: "POST", body: JSON.stringify({ prompt, domain, policy, step_cost: 0.05 }) });
      if (!result.run_id) throw new Error("The local server did not return a run ID.");
      state.runId = String(result.run_id);
      state.active = true;
      state.starting = false;
      updateAvailability();
      connectEvents(state.runId);
      // Only a newly requested run scrolls the page, never tokens or restoration.
      $("comparison").scrollIntoView({
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
        block: "start",
      });
    } catch (error) {
      state.starting = false;
      state.active = false;
      setRunStatus("Could not start comparison", "error");
      showRunError(error.message);
      updateAvailability();
      void checkStatus();
    }
  }

  async function cancelRun() {
    if (!state.active || !state.runId || state.cancelling) return;
    state.cancelling = true;
    updateAvailability();
    setRunStatus("Cancellation requested · waiting for the model", "running");
    try { await fetchJSON(`/api/runs/${encodeURIComponent(state.runId)}/cancel`, { method: "POST", body: "{}" }); }
    catch (error) { state.cancelling = false; showRunError(error.message); updateAvailability(); }
  }

  $("run-form").addEventListener("submit", runComparison);
  $("cancel-button").addEventListener("click", cancelRun);
  $("prompt").addEventListener("input", updatePromptCount);
  document.querySelectorAll('input[name="policy"]').forEach((input) => input.addEventListener("change", updatePolicyDescription));
  document.querySelectorAll(".example-chip").forEach((button) => button.addEventListener("click", () => {
    const example = examples[button.dataset.example];
    if (!example || state.active || state.starting) return;
    $("prompt").value = example.prompt;
    $("domain").value = example.domain;
    updatePromptCount();
    $("prompt").focus();
  }));
  document.addEventListener("visibilitychange", () => { if (!document.hidden) { void checkStatus(); if (state.active) void checkSnapshot(); } });
  window.addEventListener("beforeunload", () => state.source?.close());
  setupTheme();
  updatePromptCount();
  updatePolicyDescription();
  renderOutcome();
  void checkStatus();
  setInterval(() => { if (!document.hidden) void checkStatus(); }, 4000);
})();
