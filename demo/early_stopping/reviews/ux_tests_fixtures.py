"""Explicit synthetic UX fixtures: no language-model or grading execution."""
from __future__ import annotations

RUN_ID = "f" * 32
FIXTURE_NOTE = "SYNTHETIC UX TEST FIXTURE. No model generation. Both arms share the same saved test prefix."


def response(step, *, answer="6", parsed=True, raw=None):
    thought = f"Synthetic verification response {step}."
    return dict(step=step, answer=answer, confidence=90 if parsed else None,
                parse_success=parsed, generated_tokens=20 + step, prompt_tokens=100 + step,
                thought=thought, raw_text=raw if raw is not None else f'{{"thought":"{thought}","answer":"{answer}","confidence":90,"stop":false}}',
                auxiliary_tokens=0, observed_ns=0, model_stop_flag=False)


def complete_fixture(name, observations, *, early_stop=2, head_values=None):
    head_values = head_values or {2: (.003, .006)}
    events = []
    def emit(event_type, **fields):
        events.append(dict(seq=len(events) + 1, type=event_type, run_id=RUN_ID, **fields))
    def selected(step):
        choices = [o for o in observations[:step] if o["answer"].strip()]
        return choices[-1] if choices else observations[step - 1]
    def totals(step):
        return dict(steps=step, completion_tokens=sum(o["generated_tokens"] for o in observations[:step]),
                    prompt_tokens=sum(o["prompt_tokens"] for o in observations[:step]))
    def decision(step, *, early):
        chosen = selected(step)
        terminal = step == 5
        stop = terminal or early and step == early_stop
        q, next_probability = head_values.get(step, (None, None)) if early and 2 <= step < 5 else (None, None)
        reason = "terminal_horizon" if terminal else "minimum_two_steps" if step == 1 else ("estimated_drift_nonpositive" if stop else "estimated_drift_positive") if early else "continue"
        return dict(step=step, stop=bool(stop), reason=reason, selected_answer=chosen["answer"],
                    selected_step=chosen["step"], q_current=q, p_next=next_probability,
                    estimated_drift=None if q is None else next_probability - q - .05)
    emit("run_started", prompt=f"SYNTHETIC UX TEST FIXTURE: {name}", policy="learned", domain="math",
         max_steps=5, min_steps=2, step_cost=.05, model_name="UX TEST FIXTURE — no model generation")
    for step, observation in enumerate(observations, 1):
        arms = ["early", "baseline"] if step <= early_stop else ["baseline"]
        emit("response_start", step=step, arms=arms)
        emit("token", step=step, arms=arms, text=observation["raw_text"], generated_tokens=observation["generated_tokens"])
        emit("response_end", step=step, arms=arms, observation=observation, step_seconds=.1)
        emit("decision", arm="baseline", decision=decision(step, early=False), totals=totals(step), elapsed_seconds=step * .1)
        if step <= early_stop:
            emit("decision", arm="early", decision=decision(step, early=True), totals=totals(step), elapsed_seconds=step * .1)
    def final(step):
        chosen = selected(step)
        return dict(answer=chosen["answer"], selected_step=chosen["step"], stopped_at_step=step,
                    reason="terminal_horizon" if step == 5 else "estimated_drift_nonpositive",
                    completion_tokens=totals(step)["completion_tokens"], prompt_tokens=totals(step)["prompt_tokens"], elapsed_seconds=step * .1)
    early, baseline = final(early_stop), final(5)
    saved = baseline["completion_tokens"] - early["completion_tokens"]
    summary = dict(early=early, baseline=baseline, completion_tokens_saved=saved,
                   completion_savings=saved / baseline["completion_tokens"],
                   physical_completion_tokens=baseline["completion_tokens"], physical_prompt_tokens=baseline["prompt_tokens"],
                   answers_match=early["answer"].casefold().strip() == baseline["answer"].casefold().strip(), comparison_note=FIXTURE_NOTE)
    emit("complete", summary=summary)
    return dict(id=RUN_ID, status="complete", prompt=f"SYNTHETIC UX TEST FIXTURE: {name}",
                config=dict(prompt=f"SYNTHETIC UX TEST FIXTURE: {name}", domain="math", policy="learned", step_cost=.05),
                events=events, summary=summary, error=None, last_seq=len(events))


def cases():
    bad = [response(i, answer=f'{{"thought":"unfinished {i}", "answer":', parsed=False,
                    raw=f'{{"thought":"unfinished {i}", "answer":') for i in range(1, 6)]
    prior = [response(i, answer="6") for i in range(1, 6)]
    prior[2] = response(3, answer="", parsed=False, raw='{"thought":"truncated later response')
    prior[4] = response(5, answer="7")
    valid = [response(i, answer="6" if i <= 2 else "7") for i in range(1, 6)]
    full = [response(i, answer="6") for i in range(1, 6)]
    blank = [response(i, answer="") for i in range(1, 6)]
    result = {
        "unreadable_selected": complete_fixture("unreadable whole-JSON selected candidates", bad),
        "prior_readable": complete_fixture("valid prior selection with malformed later response", prior,
                                           early_stop=3, head_values={2: (.003, .5), 3: (.003, .006)}),
        "stop_two": complete_fixture("small positive gain below response cost", valid),
        "full_horizon": complete_fixture("both arms use all five responses", full, early_stop=5,
                                         head_values={2: (.1, .8), 3: (.1, .8), 4: (.1, .8)}),
        "zero_scores_blank": complete_fixture("zero predictor scores and blank selected candidates", blank,
                                              head_values={2: (0, 0)}),
        "exact_tie_four": complete_fixture("gain exactly equals cost after response four", full, early_stop=4,
                                            head_values={2: (.1, .8), 3: (.1, .8), 4: (0, .05)}),
    }
    # A real cancellation between responses two and three leaves the last
    # policy decision as Continue. It must not be mistaken for a completed run.
    tiny = complete_fixture("near-threshold positive gain, then cancellation", full, early_stop=5,
                            head_values={2: (.3, .350000001), 3: (.1, .8), 4: (.1, .8)})
    cutoff = next(i for i, event in enumerate(tiny["events"])
                  if event["type"] == "response_start" and event["step"] == 3)
    tiny["events"] = tiny["events"][:cutoff]
    tiny["events"].append(dict(seq=cutoff + 1, type="cancelled", run_id=RUN_ID,
                               message="Synthetic cancellation between responses two and three.", summary=None))
    tiny.update(status="cancelled", summary=None, last_seq=cutoff + 1)
    result["tiny_positive_cancelled"] = tiny
    return result


def status_for(snapshot):
    return dict(model=dict(name="UX TEST FIXTURE — no model generation", path="synthetic-test-only",
                           available=True, loaded=True, device="cpu"),
                predictor=dict(available=True, sha256="0" * 64), busy=False,
                run=dict(id=snapshot["id"], status=snapshot["status"]))
