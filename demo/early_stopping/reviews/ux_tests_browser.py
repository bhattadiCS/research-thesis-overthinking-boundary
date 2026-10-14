"""Human-facing explanation regressions with explicitly synthetic API fixtures.

Uses existing Playwright/Chrome and the production static HTTP host with an
inert manager. Every API route is intercepted. No model work is possible.
"""
from __future__ import annotations

from pathlib import Path
import datetime
import hashlib
import json
import re
import sys
import threading

from playwright.sync_api import sync_playwright, expect

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT))
from ux_tests_fixtures import cases, status_for
from demo.early_stopping.server import create_server


class InertManager:
    def status(self):
        return dict(model=dict(available=False), predictor=dict(available=False), busy=False, run=None)
    def start(self, *args, **kwargs):
        raise AssertionError("Synthetic UX tests must never start model generation")
    def get(self, *args, **kwargs):
        raise AssertionError("Synthetic UX tests must intercept all API calls")
    def close(self):
        pass


fixtures = cases()
page_errors = []
api_requests = []
checks = []
server = create_server(InertManager(), port=0)
server.RequestHandlerClass.log_message = lambda *args: None
server_thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
server_thread.start()
BASE = f"http://127.0.0.1:{server.server_address[1]}"


def open_fixture(browser, name, *, width=1440, height=1100, completed=True):
    snapshot = fixtures[name]
    context = browser.new_context(viewport=dict(width=width, height=height), device_scale_factor=1,
                                  reduced_motion="reduce", color_scheme="light")
    page = context.new_page()
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    def api(route):
        request = route.request
        api_requests.append(dict(case=name, method=request.method, url=request.url))
        if request.method != "GET":
            raise AssertionError("UX fixture attempted a mutating API request")
        if request.url.endswith("/api/status"):
            body = status_for(snapshot)
        elif request.url.endswith("/api/runs/" + snapshot["id"]):
            body = snapshot
        else:
            raise AssertionError("Unexpected synthetic API route: " + request.url)
        route.fulfill(status=200, content_type="application/json", body=json.dumps(body))
    page.route("**/api/**", api)
    page.goto(BASE, wait_until="domcontentloaded")
    expect(page.locator("#restored-note")).to_be_visible()
    if completed:
        expect(page.locator("#result-strip")).to_be_visible()
    else:
        expect(page.locator("#result-strip")).to_be_hidden()
    return context, page


def record(name, page, **facts):
    checks.append(dict(name=name, passed=True, **facts))
    page.screenshot(path=str(OUT / f"ux_tests_{name}.png"), full_page=True)


def estimate_texts(page, *identifiers):
    details = page.locator("#estimate-details")
    expect(details).to_be_visible()
    assert details.get_attribute("open") is None, "Technical estimates should start collapsed"
    details.locator("summary").click()
    values = {}
    for identifier in identifiers:
        element = page.locator("#" + identifier)
        expect(element).to_be_visible()
        values[identifier] = element.inner_text()
    details.locator("summary").click()
    return values


def contrast(page, selector):
    return page.locator(selector).evaluate("""el => {
      const rgb = text => {
        const numbers = text.match(/[\\d.]+/g).map(Number);
        return [numbers[0]/255, numbers[1]/255, numbers[2]/255, numbers.length > 3 ? numbers[3] : 1];
      };
      const linear = c => c <= .04045 ? c/12.92 : ((c+.055)/1.055)**2.4;
      const luminance = c => .2126*linear(c[0])+.7152*linear(c[1])+.0722*linear(c[2]);
      let node = el, background = null;
      while (node) {
        const value = rgb(getComputedStyle(node).backgroundColor);
        if (value[3] > .99) { background = value; break; }
        node = node.parentElement;
      }
      background = background || [1,1,1,1];
      const color = rgb(getComputedStyle(el).color);
      const foreground = color.slice(0,3).map((c,i) => c*color[3]+background[i]*(1-color[3]));
      const a=luminance(foreground), b=luminance(background);
      return {ratio:(Math.max(a,b)+.05)/(Math.min(a,b)+.05), foreground:color, background,
              font:getComputedStyle(el).fontSize};
    }""")


try:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True, chromium_sandbox=True)
        # Invalid selected whole JSON is retained as raw evidence, never promoted
        # to a readable answer or an answer disagreement.
        context, page = open_fixture(browser, "unreadable_selected")
        for arm in ("early", "baseline"):
            answer = page.locator(f"#{arm}-answer").inner_text()
            assert re.search(r"(could not|cannot|unreadable|no answer|not.*read)", answer, re.I), answer
            assert "{" not in answer and "thought" not in answer, answer
            details = page.locator(f"#{arm}-retained-details")
            expect(details).to_be_visible()
            assert details.get_attribute("open") is None
            details.locator("summary").click()
            expect(page.locator(f"#{arm}-retained-answer")).to_contain_text('{"thought"')
            details.locator("summary").click()
        badge = page.locator("#answer-match").inner_text()
        assert re.search(r"(unavailable|cannot compare|not comparable|no readable)", badge, re.I), badge
        assert not re.search(r"\b(same|different)\b", badge, re.I), badge
        outcome = " ".join(page.locator(selector).inner_text() for selector in
                           ("#outcome-heading", "#outcome-summary", "#outcome-quality"))
        assert not re.search(r"(different.*answer|same.*answer|answers?\s+(?:are\s+|were\s+)?(?:different|match|differ))", outcome, re.I), outcome
        record("unreadable_selected", page, answer_comparison=badge,
               primary_answers_withhold_unreadable_candidates=True, raw_evidence_available=True,
               outcome_avoids_answer_disagreement_claim=outcome)
        context.close()

        # Quality follows the selected response, not the latest malformed row.
        context, page = open_fixture(browser, "prior_readable")
        expect(page.locator("#early-answer")).to_have_text("6")
        expect(page.locator("#early-answer-source")).to_contain_text("2")
        expect(page.locator("#early-answer-source")).to_contain_text("3")
        assert not re.search(r"(could not read|unreadable|unavailable)", page.locator("#early-answer").inner_text(), re.I)
        expect(page.locator("#early-retained-details")).to_be_hidden()
        assert page.locator("#early-responses .response-card").count() == 3
        record("prior_readable", page, selected_step=2, stopped_at=3,
               malformed_response_does_not_taint_readable_selected_answer=True)
        context.close()

        # Correct percentage-point arithmetic must drive a plain-language reason.
        context, page = open_fixture(browser, "stop_two")
        estimates = estimate_texts(page, "estimate-gain", "estimate-required", "estimate-drift")
        gain, required, drift = (estimates[identifier] for identifier in
                                 ("estimate-gain", "estimate-required", "estimate-drift"))
        explanation = page.locator("#decision-reason").inner_text() + " " + page.locator("#decision-explainer").inner_text()
        assert re.search(r"\+?0\.3\b", gain), gain
        assert re.search(r"\b5(?:\.0)?\b", required), required
        assert re.search(r"[-−]4\.7\b", drift), drift
        assert re.search(r"(gain|improv|benefit)", explanation, re.I), explanation
        assert re.search(r"(cost|penalty|needed|required)", explanation, re.I), explanation
        assert re.search(r"(stop|not.*worth|too small|less than|below)", explanation, re.I), explanation
        assert "0.3" in explanation and re.search(r"\b5(?:\.0)?\b", explanation), explanation
        expect(page.locator("#decision-title")).to_contain_text(re.compile("stop", re.I))
        assert page.locator("#early-responses .response-card").count() == 2
        assert page.locator("#baseline-responses .response-card").count() == 5
        assert page.locator(".response-card").evaluate_all("cards => cards.every(card => card.tagName === 'DETAILS' && !card.open)"), "Completed response logs must be collapsed by default"
        timeline = [li.locator(".timeline-status").inner_text() for li in page.locator("#run-timeline li[data-step]").all()]
        assert len(timeline) == 5, timeline
        assert re.search(r"(skip|full.*only|baseline.*only|not.*early)", timeline[2], re.I), timeline
        record("small_gain_stop", page, gain=gain, required=required, net=drift,
               explanation=explanation, response_three=timeline[2], completed_response_logs_collapsed=True)
        context.close()

        # A forced horizon is explained without inventing a sixth-response head.
        context, page = open_fixture(browser, "full_horizon")
        expect(page.locator("#estimate-details")).to_be_hidden()
        expect(page.locator("#result-savings")).to_have_text(re.compile(r"^0(?:\.0+)?%$"))
        assert page.locator("#early-responses .response-card").count() == 5
        assert page.locator("#baseline-responses .response-card").count() == 5
        outcome = page.locator("#outcome-summary").inner_text()
        assert re.search(r"(no.*(earlier|skip|sav)|both.*five|both.*5|full.*budget)", outcome, re.I), outcome
        reason = page.locator("#decision-explainer").inner_text()
        assert re.search(r"(limit|horizon|five|5)", reason, re.I), reason
        assert not re.search(r"(0\.3|5\.0.*(cost|penalty)|expected.*sixth)", reason, re.I), reason
        record("horizon_no_savings", page, outcome=outcome, reason=reason,
               terminal_next_estimates_hidden=True, avoided_tokens=0)
        context.close()

        # Zero scores are supplied estimates, whereas a blank selected candidate
        # is still unreadable even when the response parser succeeded.
        context, page = open_fixture(browser, "zero_scores_blank")
        estimates = estimate_texts(page, "estimate-current", "estimate-next")
        assert estimates["estimate-current"] == "0.0%", estimates
        assert estimates["estimate-next"] == "0.0%", estimates
        expect(page.locator("#early-answer")).to_have_text(re.compile("no answer", re.I))
        expect(page.locator("#answer-match")).to_have_text(re.compile("unavailable", re.I))
        expect(page.locator("#decision-title")).to_have_text(re.compile("stop", re.I))
        record("zero_scores_blank", page, zero_is_supplied_estimate=True,
               blank_parse_success_does_not_establish_readable_answer=True)
        context.close()

        # Equality stops, including at a later nonterminal response. The displayed
        # explanation must say that equality fails the strictly-positive rule.
        context, page = open_fixture(browser, "exact_tie_four")
        expect(page.locator("#decision-title")).to_have_text(re.compile("stop", re.I))
        explanation = page.locator("#decision-explainer").inner_text()
        assert re.search(r"(does not exceed|at or below|not greater|equal)", explanation, re.I), explanation
        assert page.locator("#early-responses .response-card").count() == 4
        expect(page.locator('#run-timeline li[data-step="5"] .timeline-status')).to_have_text(re.compile("full.*only|baseline.*only", re.I))
        net = estimate_texts(page, "estimate-drift")["estimate-drift"]
        assert float(net.split()[0]) == 0
        record("exact_tie_four", page, net=0, strict_positive_continuation=True,
               explanation=explanation, stopped_at=4)
        context.close()

        # A positive net must retain its sign even below one-decimal precision.
        # The rounded gain/penalty tie must not contradict the Continue decision.
        context, page = open_fixture(browser, "tiny_positive_cancelled", completed=False)
        expect(page.locator("#decision-title")).to_have_text(re.compile("keep|continu", re.I))
        net = estimate_texts(page, "estimate-drift")["estimate-drift"]
        assert float(net.split()[0]) > 0, net
        explanation = page.locator("#decision-explainer").inner_text()
        assert re.search(r"(round|precision)", explanation, re.I), explanation
        assert re.search(r"(keep.*going|continu|clears)", explanation, re.I), explanation
        expect(page.locator("#outcome-heading")).to_have_text(re.compile("cancel", re.I))
        assert page.locator("#early-responses .response-card").count() == 2
        record("tiny_positive_continue", page, displayed_net=net,
               rounded_tie_explained=explanation, incomplete_run_has_no_savings=True)
        context.close()

        # Theme preference survives reload; explanations remain readable at a
        # narrow viewport, rather than merely preserving a selected option.
        context, page = open_fixture(browser, "stop_two", width=390, height=844)
        theme_facts = []
        for theme in ("dark", "light"):
            page.locator("#theme-select").select_option(theme)
            expect(page.locator("html")).to_have_attribute("data-theme", theme)
            page.reload(wait_until="domcontentloaded")
            expect(page.locator("#result-strip")).to_be_visible()
            expect(page.locator("html")).to_have_attribute("data-theme", theme)
            assert page.locator("#theme-select").input_value() == theme
            assert page.evaluate("() => document.documentElement.scrollWidth <= innerWidth"), "mobile horizontal overflow"
            ratios = {selector: contrast(page, selector) for selector in ("#outcome-summary", "#decision-explainer", "#early-answer", "#baseline-answer")}
            assert all(item["ratio"] >= 4.5 for item in ratios.values()), ratios
            backdrop = ratios["#outcome-summary"]["background"][:3]
            assert (sum(backdrop) < 1.2 if theme == "dark" else sum(backdrop) > 2), (theme, backdrop)
            bg = page.locator("html").evaluate("el => getComputedStyle(el).backgroundColor")
            bounds = {selector: page.locator(selector).bounding_box() for selector in ("#outcome-summary", "#decision-explainer", "#early-answer")}
            assert all(box and 0 <= box["x"] and box["x"] + box["width"] <= 391 for box in bounds.values()), bounds
            boxes = list(bounds.items())
            for index, (left_name, left) in enumerate(boxes):
                for right_name, right in boxes[index + 1:]:
                    overlap_x = min(left["x"] + left["width"], right["x"] + right["width"]) - max(left["x"], right["x"])
                    overlap_y = min(left["y"] + left["height"], right["y"] + right["height"]) - max(left["y"], right["y"])
                    assert overlap_x <= 1 or overlap_y <= 1, (left_name, right_name, left, right)
            theme_facts.append(dict(theme=theme, root_background=bg, contrast=ratios, bounds=bounds))
            page.screenshot(path=str(OUT / f"ux_tests_mobile_{theme}.png"), full_page=True)
        checks.append(dict(name="theme_persistence_mobile", passed=True, facts=theme_facts))
        context.close()
        assert not page_errors, page_errors
        browser.close()
finally:
    server.shutdown()
    server.server_close()
    server_thread.join(2)
    result = dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  synthetic_fixtures=True, language_model_generations=0, host="isolated ephemeral loopback with inert manager",
                  failure=None if sys.exc_info()[0] is None else str(sys.exc_info()[1]),
                  checks=checks, page_errors=page_errors, intercepted_api_requests=api_requests,
                  static_source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT.parent / "static").iterdir() if p.is_file()})
    (OUT / "ux_tests_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(passed_checks=len(checks), checks=[r["name"] for r in checks], page_errors=page_errors), indent=2))
