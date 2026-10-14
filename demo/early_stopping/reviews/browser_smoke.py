"""Developer end-to-end test against a real local model, in isolated Chrome.

Uses the preinstalled browser/Playwright; no existing browser profile, browser
security settings, research dependencies, frozen evidence or models changed.
"""
from pathlib import Path
import datetime, hashlib, json, time, urllib.request
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
BASE = 'http://127.0.0.1:8765'

def api(path):
    with urllib.request.urlopen(BASE+path, timeout=5) as r:
        return json.load(r)

errors=[]
def wait_ui(page, predicate, timeout=120.):
    # Locator reads avoid introducing eval() into a page with a strict CSP.
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if predicate():
            return
        page.wait_for_timeout(75)
    raise AssertionError('UI state did not arrive: '+page.locator('#run-status-text').inner_text())

with sync_playwright() as p:
    browser = p.chromium.launch(channel='chrome', headless=True, chromium_sandbox=True)
    context = browser.new_context(viewport={'width':1440,'height':1100},device_scale_factor=1)
    page = context.new_page()
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto(BASE, wait_until='domcontentloaded')
    page.locator('#run-button').wait_for(state='visible')
    wait_ui(page,lambda:page.locator('#run-button').is_enabled())
    page.screenshot(path=str(OUT/'desktop_ready.png'),full_page=True)
    page.locator('#prompt').fill('')
    page.locator('#run-button').click()
    assert page.locator('#prompt-error').is_visible()
    page.locator('[data-example="shopping"]').click()
    assert 'notebooks' in page.locator('#prompt').input_value()
    assert page.locator('#domain').input_value() == 'gsm8k'
    # Real cancellation followed by a successful new comparison proves that
    # the model/session can be reused and leaves a completed example to inspect.
    page.locator('#prompt').fill('Compute 7 times 8 and give the number.')
    page.locator('#run-button').click()
    wait_ui(page,lambda:page.locator('#baseline-responses pre').count()>0 and bool(page.locator('#baseline-responses pre').first.inner_text()),120.)
    cancel_id=api('/api/status')['run']['id']
    page.locator('#cancel-button').click()
    wait_ui(page,lambda:page.locator('#run-status-text').inner_text().startswith('Run cancelled'),30.)
    assert not page.locator('#result-strip').is_visible()
    cancelled=api('/api/runs/'+cancel_id)
    assert cancelled['status']=='cancelled' and cancelled['summary'] is None
    wait_ui(page,lambda:page.locator('#run-button').is_enabled())
    page.screenshot(path=str(OUT/'desktop_cancelled.png'),full_page=True)
    page.locator('#prompt').fill('What is 2 + 2? Give the number.')
    page.locator('#run-button').click()
    # Prove visible decoded text appears before the first completed response.
    wait_ui(page,lambda:any(c.locator('pre').inner_text() and 'Generating' in c.locator('.response-state').inner_text() for c in page.locator('#baseline-responses .response-card').all()))
    streamed_text = page.locator('#baseline-responses pre').first.inner_text()
    wait_ui(page,lambda:'Stopped' in page.locator('#early-badge').inner_text() and page.locator('#baseline-responses .response-card').count()>=3)
    frozen_answer = page.locator('#early-answer').inner_text()
    frozen_tokens = page.locator('#early-completion').inner_text()
    frozen_steps = page.locator('#early-steps').inner_text()
    page.screenshot(path=str(OUT/'desktop_during_comparison.png'),full_page=True)
    running_id = api('/api/status')['run']['id']
    page.reload(wait_until='domcontentloaded')
    wait_ui(page,lambda:page.locator('#baseline-responses .response-card').count()>=3,15.)
    assert page.locator('#early-answer').inner_text() == frozen_answer
    assert page.locator('#early-completion').inner_text() == frozen_tokens
    wait_ui(page,lambda:page.locator('#run-status-text').inner_text() in ('Comparison complete','Previous comparison restored'))
    snapshot=api('/api/runs/'+running_id)
    summary=snapshot['summary']
    assert summary is not None and snapshot['status']=='complete'
    assert summary['early']['stopped_at_step'] < 5
    assert page.locator('#early-responses .response-card').count() == summary['early']['stopped_at_step']
    assert page.locator('#baseline-responses .response-card').count() == 5
    assert page.locator('#early-answer').inner_text() == summary['early']['answer'] == frozen_answer
    assert page.locator('#early-completion').inner_text() == str(summary['early']['completion_tokens']) == frozen_tokens
    assert page.locator('#baseline-completion').inner_text() == str(summary['baseline']['completion_tokens'])
    assert page.locator('#result-savings').inner_text() == f"{summary['completion_savings']*100:.1f}%"
    assert page.locator('#result-strip').is_visible()
    assert not page.locator('#estimate-grid').is_hidden()
    wait_ui(page,lambda:page.locator('#run-button').is_enabled())
    page.screenshot(path=str(OUT/'desktop_complete.png'),full_page=True)
    desktop_dom=page.locator('body').inner_text()
    (OUT/'desktop_complete.txt').write_text(desktop_dom,encoding='utf-8')
    # A fresh narrow viewport must reconstruct the completed run without a new
    # generation and keep controls/results within the viewport width.
    mobile=browser.new_context(viewport={'width':390,'height':844},device_scale_factor=1)
    mobile_page=mobile.new_page()
    mobile_page.on('pageerror', lambda e: errors.append(str(e)))
    mobile_page.goto(BASE,wait_until='domcontentloaded')
    wait_ui(mobile_page,lambda:mobile_page.locator('#run-status-text').inner_text()=='Previous comparison restored',15.)
    assert mobile_page.evaluate('() => document.documentElement.scrollWidth <= innerWidth')
    assert mobile_page.locator('#baseline-responses .response-card').count()==5
    mobile_page.screenshot(path=str(OUT/'mobile_complete.png'),full_page=True)
    assert api('/api/status')['run']['id']==running_id
    assert not errors,errors
    report=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                browser='Installed Google Chrome, isolated automated-test context, normal sandbox',
                actual_model_generation=True, run_id=running_id, summary=summary,
                stream_before_response_complete=bool(streamed_text),
                early_answer_and_cost_frozen=True, reload_reconnected=True,
                mobile_reconnected_without_new_generation=True, mobile_no_horizontal_overflow=True,
                real_cancellation_verified=True, cancelled_run_id=cancel_id,
                blank_prompt_validation=True,example_chip_and_domain_verified=True,
                browser_page_errors=errors,
                files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (OUT.parent/'static').iterdir() if f.is_file()})
    (OUT/'browser_smoke.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    browser.close()
