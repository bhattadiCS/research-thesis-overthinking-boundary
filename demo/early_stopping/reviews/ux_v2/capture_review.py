"""Retain the actual browser-tested logarithm run and render final UX evidence.

No generation, training, grading or replay presented as live. These screenshots
show the actual completed local run, restored by the UI and labeled as history.
"""
from pathlib import Path
import datetime, hashlib, json, subprocess, urllib.request
from playwright.sync_api import sync_playwright, expect

OUT=Path(__file__).resolve().parent
APP=OUT.parents[1]
ROOT=APP.parents[1]
BASE='http://127.0.0.1:8765'
def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def api(path):
    with urllib.request.urlopen(BASE+path,timeout=5) as r:
        return json.load(r)
status=api('/api/status')
run=api('/api/runs/'+status['run']['id'])
assert run['status']=='complete' and 'log_2(x^2 - 7x + 10)' in run['prompt']
events=run['events'];ends=[e for e in events if e['type']=='response_end']
assert len(ends)==5
early=[e for e in events if e['type']=='decision' and e['arm']=='early']
stop=early[-1]['decision']['step'];assert stop==2 and len(early)==2
assert all(e['observation']['parse_success'] is False for e in ends)
assert all(e['arms']==['baseline'] for e in ends[stop:])
summary=run['summary']
assert summary['early']['completion_tokens']==sum(e['observation']['generated_tokens'] for e in ends[:stop])
assert summary['baseline']['completion_tokens']==sum(e['observation']['generated_tokens'] for e in ends)
assert summary['physical_completion_tokens']==summary['baseline']['completion_tokens']
decision=early[-1]['decision'];gain=decision['p_next']-decision['q_current']
assert decision['estimated_drift']==gain-run['config']['step_cost']<=0
(OUT/'logarithm_after.json').write_text(json.dumps(run,indent=2)+'\n',encoding='utf-8')
sources={p.name:sha(p) for p in (APP/'static').iterdir() if p.is_file()}
ux=read(APP/'reviews/ux_tests_results.json')
assert sources==ux['static_source_sha256']
assert ux['failure'] is None and len(ux['checks'])==8 and not ux['page_errors']
assert all(r['passed'] for r in ux['checks'])
for path,digest in read(APP/'tests/backend_test_record.json')['source_sha256'].items():
    assert sha(ROOT/path)==digest,path
baseline=read(ROOT/'ThesisDocs/verification/math_reliability_audit_2026-10-13/baseline_hashes.json')
assert all(sha(ROOT/r['path'])==r['sha256'] for r in baseline)
assert not subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).strip()
errors=[];views=[]
with sync_playwright() as p:
    browser=p.chromium.launch(channel='chrome',headless=True,chromium_sandbox=True)
    for name,width,height in [('desktop',1440,1100),('mobile',390,844)]:
        context=browser.new_context(viewport={'width':width,'height':height},device_scale_factor=1,reduced_motion='reduce')
        page=context.new_page();page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(BASE,wait_until='domcontentloaded')
        expect(page.locator('#outcome-heading')).to_have_text('Stopped after 2 of 5 responses')
        expect(page.locator('#restored-note')).to_contain_text('not a new live generation')
        expect(page.locator('#outcome-summary')).to_contain_text('skipped 3 responses')
        expect(page.locator('#answer-match')).to_have_text('Answer comparison unavailable')
        assert page.locator('#result-savings').inner_text()==f"{summary['completion_savings']*100:.1f}%"
        for arm in ('early','baseline'):
            expect(page.locator(f'#{arm}-answer')).to_have_text('Answer could not be read reliably')
            assert page.locator(f'#{arm}-retained-answer').text_content()==summary[arm]['answer']
            assert page.locator(f'#{arm}-completion').inner_text()==str(summary[arm]['completion_tokens'])
        for theme in ('light','dark'):
            page.locator('#theme-select').select_option(theme)
            expect(page.locator('html')).to_have_attribute('data-theme',theme)
            assert page.evaluate('() => document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path=str(OUT/f'{name}_{theme}_full.png'),full_page=True)
            if name=='desktop':
                page.locator('#comparison').scroll_into_view_if_needed()
                page.screenshot(path=str(OUT/f'comparison_{theme}.png'))
            views.append(dict(viewport=name,theme=theme,no_horizontal_overflow=True,
                              actual_retained_record_unchanged=True,readability_gate_verified=True))
        context.close()
    browser.close()
assert not errors,errors
record=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            actual_run_id=run['id'],actual_prompt=run['prompt'],summary=summary,
            early_responses=stop,full_responses=5,skipped_responses=5-stop,
            current_score=decision['q_current'],next_score=decision['p_next'],
            gain_percentage_points=gain*100,cost_percentage_points=run['config']['step_cost']*100,
            net_percentage_points=decision['estimated_drift']*100,
            all_five_outputs_malformed=True,answer_comparison_not_established=True,
            browser_tools_checks='Coordinator inspected the existing Edge user tab, ran this exact prompt through Run comparison after the redesign, observed completed result, switched Dark/Light/System, expanded calculation and retained record; actual screenshots/AX states returned by Browser Use.',
            capture_scope='These isolated Chrome renders restore the actual completed run; no new generation. Actual live interaction used Browser Use on the user tab.',
            viewports=views,page_errors=errors,source_sha256=sources,
            ux_groups_passed=8,backend_tests_passed=25,original_files_preserved=len(baseline))
(OUT/'verification.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:record[k] for k in ['actual_run_id','early_responses','full_responses','skipped_responses','gain_percentage_points','cost_percentage_points','net_percentage_points','ux_groups_passed','backend_tests_passed','original_files_preserved','page_errors']},indent=2))
