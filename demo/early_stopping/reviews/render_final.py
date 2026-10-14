"""Render final presentation against an explicitly restored actual prior run.

Only /api/status's latest-run pointer is overridden inside this isolated test
context. Events/observations/summary are read from the real completed run's API.
The UI labels this saved history; this command creates no new model generation.
"""
from pathlib import Path
import datetime, hashlib, json, urllib.request
from playwright.sync_api import sync_playwright, expect

OUT=Path(__file__).resolve().parent
BASE='http://127.0.0.1:8765'
receipt=json.loads((OUT/'browser_smoke.json').read_text())
run_id=receipt['run_id']
with urllib.request.urlopen(BASE+'/api/status',timeout=5) as r:
    status=json.load(r)
assert not status['busy'],'Wait for the current comparison to finish before rendering saved history.'
status['run']={'id':run_id,'status':'complete'}
errors=[]
with sync_playwright() as p:
    browser=p.chromium.launch(channel='chrome',headless=True,chromium_sandbox=True)
    reports=[]
    for name,width,height in [('desktop',1440,1100),('mobile',390,844)]:
        context=browser.new_context(viewport={'width':width,'height':height},device_scale_factor=1)
        page=context.new_page()
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.route('**/api/status',lambda route:route.fulfill(status=200,content_type='application/json',body=json.dumps(status)))
        page.goto(BASE,wait_until='domcontentloaded')
        expect(page.locator('#run-status-text')).to_have_text('Previous comparison restored')
        expect(page.locator('#restored-note')).to_contain_text('not a new live generation')
        expect(page.locator('#run-button')).to_be_enabled()
        assert page.locator('#early-responses .response-card').count()==receipt['summary']['early']['stopped_at_step']
        assert page.locator('#baseline-responses .response-card').count()==5
        assert page.locator('#early-completion').inner_text()==str(receipt['summary']['early']['completion_tokens'])
        assert page.locator('#baseline-completion').inner_text()==str(receipt['summary']['baseline']['completion_tokens'])
        assert page.evaluate('() => document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(OUT/f'final_{name}.png'),full_page=True)
        if name=='desktop':
            page.locator('#comparison').scroll_into_view_if_needed()
            page.screenshot(path=str(OUT/'comparison_preview.png'),full_page=False)
        cues=page.get_by_text('Unstructured response',exact=True).count()
        assert cues>0,'Incomplete-format responses must have a visible cue.'
        fonts=page.locator('.response-summary p').first.evaluate('(el) => getComputedStyle(el).fontSize')
        assert float(fonts.rstrip('px'))>=12
        reports.append(dict(viewport=name,width=width,height=height,no_horizontal_overflow=True,
                            unstructured_cues=cues,response_summary_font=fonts,
                            real_saved_run_restored=True,counts_match_live_receipt=True))
        context.close()
    assert not errors,errors
    report=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                run_id=run_id,new_model_generations=0,actual_saved_events=True,
                note='Isolated test changes only latest-run pointer; actual prior run served and clearly labeled as saved.',
                viewports=reports,page_errors=errors,
                final_source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (OUT.parent/'static').iterdir() if f.is_file()})
    (OUT/'render_final.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    browser.close()
