"""UI failure-state checks with explicitly synthetic HTTP fixtures; no model work."""
from pathlib import Path
import json, datetime
from playwright.sync_api import sync_playwright, expect

OUT=Path(__file__).resolve().parent
BASE='http://127.0.0.1:8765'
base_status=dict(model=dict(name='Test fixture',path='missing/local/snapshot',available=True,loaded=False,device='cpu'),
                 predictor=dict(available=True,sha256='0'*64),busy=False,run=None)
errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(channel='chrome',headless=True,chromium_sandbox=True)
    def state_page(status):
        page=browser.new_page(viewport={'width':1280,'height':900})
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.route('**/api/status',lambda route:route.fulfill(status=200,content_type='application/json',body=json.dumps(status)))
        return page
    missing=json.loads(json.dumps(base_status));missing['model']['available']=False
    page=state_page(missing);page.goto(BASE)
    expect(page.locator('#system-notice')).to_contain_text('local model is unavailable')
    expect(page.locator('#run-button')).to_be_disabled()
    page.screenshot(path=str(OUT/'missing_model.png'),full_page=True)
    page.close()
    missing=json.loads(json.dumps(base_status));missing['predictor']['available']=False
    page=state_page(missing);page.goto(BASE)
    expect(page.locator('#system-notice')).to_contain_text('learned predictor is unavailable')
    expect(page.locator('#run-button')).to_be_disabled()
    page.locator('input[value="heuristic"]').check()
    expect(page.locator('#run-button')).to_be_enabled()
    expect(page.locator('#continuation-cost')).to_be_hidden()
    page.close()
    page=state_page(base_status)
    page.route('**/api/runs',lambda route:route.fulfill(status=409,content_type='application/json',body=json.dumps({'error':'A comparison is already running.'})))
    page.goto(BASE)
    expect(page.locator('#run-button')).to_be_enabled()
    page.locator('#prompt').fill('A test fixture question')
    page.locator('#run-button').click()
    expect(page.locator('#run-error')).to_contain_text('already running')
    expect(page.locator('#run-button')).to_be_enabled()
    expect(page.locator('#result-strip')).to_be_hidden()
    page.close()
    page=browser.new_page();page.on('pageerror',lambda error:errors.append(str(error)))
    page.route('**/api/status',lambda route:route.abort())
    page.goto(BASE)
    expect(page.locator('#server-status-text')).to_have_text('Local server unavailable')
    expect(page.locator('#run-button')).to_be_disabled()
    page.close()
    assert not errors,errors
    report=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                synthetic_http_fixtures=True,model_generations=0,
                missing_model_blocks_run_with_explanation=True,
                missing_predictor_allows_heuristic=True,heuristic_hides_learned_cost=True,
                start_error_has_message_and_recovers_controls=True,
                unreachable_server_is_visible=True,page_errors=errors)
    (OUT/'ui_states.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    browser.close()
