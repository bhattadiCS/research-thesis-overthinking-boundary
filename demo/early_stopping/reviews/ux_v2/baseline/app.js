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
  };
  const makeArm = () => ({ cards: new Map(), stopped: false, steps: 0, selectedStep: null, answer: null, completionTokens: null, promptTokens: null, elapsedSeconds: null });
  const state = { server: null, statusFailed: false, checking: false, active: false, starting: false, restoring: false, restoreAttempted: false, viewVersion: 0, cancelling: false, runId: null, lastSeq: -1, source: null, snapshotInFlight: false, maxSteps: 5, policy: "learned", decision: null, terminal: false, arms: { early: makeArm(), baseline: makeArm() } };

  const selectedPolicy = () => document.querySelector('input[name="policy"]:checked').value;
  const finiteNumber = (value) => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));
  const numberText = (value) => finiteNumber(value) ? formatNumber.format(Number(value)) : "—";
  const timeText = (value) => !finiteNumber(value) ? "—" : Number(value) < 1 ? `${Math.round(Number(value) * 1000)} ms` : `${Number(value).toFixed(1)} s`;
  const percentage = (value) => finiteNumber(value) ? `${(Number(value) * 100).toFixed(1)}%` : "—";
  const announce = (message) => { if (!state.restoring) $("announcer").textContent = message; };

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
    $("policy-description").textContent = learned ? "Uses frozen current- and next-answer predictors to estimate whether another response is worth its cost." : "Uses the model’s reported confidence and answer stability. Reported confidence is not a calibrated probability.";
    $("continuation-cost").hidden = !learned;
    if (!state.runId) $("early-policy-label").textContent = learned ? "Learned drift policy" : "Confidence heuristic";
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
    $(`${armName}-answer`).textContent = arm.answer === null ? "—" : String(arm.answer).trim() || "No candidate selected";
    let source = arm.selectedStep ? `Selected from response ${arm.selectedStep}` : "No nonempty candidate selected";
    if (finiteNumber(stoppedAt)) source += ` · ${armName === "early" ? "stopped after" : "generated"} ${stoppedAt} ${Number(stoppedAt) === 1 ? "response" : "responses"}`;
    $(`${armName}-answer-source`).textContent = source;
    const selected = arm.cards.get(arm.selectedStep);
    $(`${armName}-format-note`).hidden = !(selected && (selected.parseSuccess === false || selected.parseSuccess === 0));
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
    state.policy = policy;
    state.decision = null;
    state.arms = { early: makeArm(), baseline: makeArm() };
    $("early-policy-label").textContent = policy === "learned" ? "Learned drift policy" : "Confidence heuristic";
    $("result-strip").hidden = true;
    $("restored-note").hidden = true;
    $("run-error").hidden = true;
    $("decision-box").hidden = true;
    for (const arm of armNames) {
      $(`${arm}-responses`).replaceChildren();
      $(`${arm}-answer`).textContent = "—";
      $(`${arm}-answer-source`).textContent = "Waiting for the first candidate.";
      $(`${arm}-format-note`).hidden = true;
      $(`${arm}-panel`).dataset.frozen = "false";
      $(`${arm}-panel`).dataset.active = "false";
      setBadge(arm, "Waiting", "waiting");
      renderMetrics(arm);
    }
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
      ensureCard(arm, event.step, shared);
      setBadge(arm, "Generating", "generating");
      $(`${arm}-panel`).dataset.active = "true";
    }
    setRunStatus(`Generating response ${event.step} of ${state.maxSteps}${shared ? " · shared prefix" : " · full budget only"}`, "running");
    announce(`Response ${event.step} is generating${shared ? " for both arms" : " for the full budget"}.`);
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
      summary.hidden = !thought;
      summary.querySelector("p").textContent = thought;
      const candidate = card.node.querySelector(".response-candidate");
      candidate.hidden = !answer;
      candidate.querySelector("p").textContent = answer;
      const raw = card.node.querySelector(".raw-response");
      raw.querySelector("summary").firstChild.nodeValue = "Raw response ";
      // Keep raw output available; parsed fields become the compact completed view.
      if (thought || answer) raw.open = false;
      const footer = card.node.querySelector(".response-footer");
      const details = [];
      if (finiteNumber(card.promptTokens)) details.push(`${numberText(card.promptTokens)} input tokens`);
      if (finiteNumber(event.step_seconds)) details.push(timeText(event.step_seconds));
      if (observation.parse_success === false || observation.parse_success === 0) details.push("Strict format not parsed; raw output retained");
      if (finiteNumber(observation.confidence)) details.push(`Reported confidence ${Number(observation.confidence).toFixed(0)}/100`);
      footer.textContent = details.join(" · "); footer.hidden = !details.length;
      updateTokenCounts(arm);
      $(`${arm}-panel`).dataset.active = "false";
    }
  }

  function reasonText(decision) {
    const reason = String(decision.reason || "");
    const key = reason.toLowerCase();
    if (key.includes("horizon") || key.includes("max_step") || Number(decision.step) >= state.maxSteps) return "The five-response limit has been reached. No next-response estimate is needed.";
    if (key.includes("floor") || key.includes("min_step") || key.includes("minimum")) return "The policy continues until the two-response minimum is reached.";
    if (key.includes("drift")) return decision.stop ? "Estimated next-response gain does not cover the 0.05 continuation penalty." : "Estimated next-response gain exceeds the 0.05 continuation penalty.";
    if (key.includes("stable") || key.includes("stability")) return "The parsed answer and reported confidence meet the heuristic’s stability condition.";
    if (key.includes("confidence_drop")) return "The heuristic’s confidence-drop condition was met.";
    if (key.includes("churn") || key.includes("answer_change")) return "The heuristic’s answer-change condition was met.";
    if (reason.includes(" ") && !reason.includes("_")) return reason;
    return reason ? reason.replace(/_/g, " ").replace(/^./, (letter) => letter.toUpperCase()) + "." : decision.stop ? "The policy selected this response as its stopping point." : "The policy requests another response.";
  }

  function showDecision(decision) {
    state.decision = decision;
    const stop = Boolean(decision.stop);
    $("decision-box").hidden = false;
    $("decision-box").dataset.stop = String(stop);
    $("decision-icon").textContent = stop ? "■" : "→";
    $("decision-title").textContent = stop ? "Stop here" : "Continue";
    $("decision-step").textContent = `After response ${decision.step}`;
    $("decision-reason").textContent = reasonText(decision);
    const hasEstimates = Number(decision.step) < state.maxSteps && [decision.q_current, decision.p_next, decision.estimated_drift].some(finiteNumber);
    $("estimate-grid").hidden = !hasEstimates;
    $("estimate-note").hidden = !hasEstimates;
    $("estimate-current").textContent = percentage(decision.q_current);
    $("estimate-next").textContent = percentage(decision.p_next);
    $("estimate-drift").textContent = finiteNumber(decision.estimated_drift) ? `${Number(decision.estimated_drift) >= 0 ? "+" : ""}${Number(decision.estimated_drift).toFixed(4)}` : "—";
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
    $("answer-match").textContent = summary.answers_match ? "Same normalized answer text" : "Different normalized answer text";
    $("answer-match").dataset.match = String(Boolean(summary.answers_match));
    $("answer-match").title = "Whitespace/case-normalized string comparison only. This is not semantic equivalence or a correctness grade.";
    $("result-note").textContent = summary.comparison_note || "The comparison generates the full baseline. Avoided tokens refer to the early arm’s continuation budget.";
    $("physical-tokens").textContent = finiteNumber(summary.physical_completion_tokens) ? `Actually generated for this comparison: ${numberText(summary.physical_completion_tokens)} completion tokens, shared prefix counted once.` : "";
    $("result-strip").hidden = false;
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
        state.policy = event.policy || state.policy;
        $("early-policy-label").textContent = state.policy === "heuristic" ? "Confidence heuristic" : "Learned drift policy";
        for (const arm of armNames) renderMetrics(arm);
        setRunStatus("Preparing local inference", "running");
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
  updatePromptCount();
  updatePolicyDescription();
  void checkStatus();
  setInterval(() => { if (!document.hidden) void checkStatus(); }, 4000);
})();
