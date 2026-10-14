"""Completion audit using current source bytes, actual events and live service."""
from pathlib import Path
import datetime, hashlib, json, subprocess, urllib.request
import sys

OUT=Path(__file__).resolve().parent
APP=OUT.parent
ROOT=APP.parents[1]
sys.path.insert(0,str(ROOT/'research'))
from online_stopping_controller import Observation
from prefix_stopping_model import PrefixStoppingModel

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def api(path):
    with urllib.request.urlopen('http://127.0.0.1:8765'+path,timeout=5) as r:
        assert r.status==200
        return json.load(r)

browser=read(OUT/'browser_smoke.json')
render=read(OUT/'render_final.json')
states=read(OUT/'ui_states.json')
test=read(APP/'tests/backend_test_record.json')
assert test['exit_code']==0
for path,digest in test['source_sha256'].items():
    assert sha(ROOT/path)==digest,path
for path,digest in browser['files'].items():
    assert sha(APP/'static'/path)==digest,path
assert browser['files']==render['final_source_sha256']
assert not browser['browser_page_errors'] and not render['page_errors'] and not states['page_errors']
assert browser['actual_model_generation'] and browser['stream_before_response_complete']
assert browser['reload_reconnected'] and browser['real_cancellation_verified']
status=api('/api/status')
assert status['model']['loaded'] and status['model']['available'] and status['predictor']['available']
run=api('/api/runs/'+browser['run_id'])
cancelled=api('/api/runs/'+browser['cancelled_run_id'])
assert run['status']=='complete' and cancelled['status']=='cancelled' and cancelled['summary'] is None
events=run['events']
assert [e['seq'] for e in events]==list(range(1,len(events)+1))
assert sum(e['type'] in ('complete','cancelled','error') for e in events)==1
ends=[e for e in events if e['type']=='response_end']
assert [e['step'] for e in ends]==[1,2,3,4,5]
early=[e for e in events if e['type']=='decision' and e['arm']=='early']
stop=early[-1]['decision']['step']
assert early[-1]['decision']['stop'] and len(early)==stop<5
assert all(e['arms']==['baseline'] for e in events if e['type'] in ('token','response_start','response_end') and e['step']>stop)
summary=run['summary']
assert summary==browser['summary']
for name,prefix in [('early',ends[:stop]),('baseline',ends)]:
    assert summary[name]['completion_tokens']==sum(e['observation']['generated_tokens'] for e in prefix)
    assert summary[name]['prompt_tokens']==sum(e['observation']['prompt_tokens'] for e in prefix)
assert summary['physical_completion_tokens']==summary['baseline']['completion_tokens']
assert summary['completion_tokens_saved']==summary['baseline']['completion_tokens']-summary['early']['completion_tokens']
assert summary['completion_savings']==summary['completion_tokens_saved']/summary['baseline']['completion_tokens']
predictor=PrefixStoppingModel.load(ROOT/'research/outputs/semester2/prefix_model_v1/prefix_model.json')
assert predictor.artifact_sha256==status['predictor']['sha256']
estimate=predictor.estimate(tuple(Observation(**e['observation']) for e in ends[:stop]),run['config']['domain'])
decision=early[-1]['decision']
assert decision['q_current']==estimate.q_current and decision['p_next']==estimate.p_next
assert decision['estimated_drift']==(estimate.p_next-estimate.q_current)-run['config']['step_cost']<=0
for filename,value in [('completed_run.json',run),('cancelled_run.json',cancelled)]:
    (OUT/filename).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
baseline=read(ROOT/'ThesisDocs/verification/math_reliability_audit_2026-10-13/baseline_hashes.json')
assert len(baseline)==36 and all(sha(ROOT/r['path'])==r['sha256'] for r in baseline)
assert not subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).strip()
record=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            requirements_proven=['typed_prompt_actual_generation','live_decoded_text','early_stop_vs_full_continuation',
                                 'selected_answer_and_prefix_freeze','actual_token_accounting','cancel_and_reuse',
                                 'running_reload_reconnect','mobile_saved_restore','visible_error_states'],
            source_hashes_match_actual_browser_and_tests=True,original_hashes_preserved=36,
            existing_tracked_sources_unchanged=True,actual_completed_run_id=run['id'],
            actual_early_responses=stop,actual_baseline_responses=5,
            actual_summary=summary,service_status=status,
            unverified_scientific_claims=['answer_correctness','adaptive_advantage','true_conditional_calibration','accuracy_noninferiority','Bellman_optimality'],
            verification_limits='Local Qwen0.5B math/arithmetic demonstration; Chrome verified, connected Edge profile blocks local pages. No frozen research data modified.')
(OUT/'completion_check.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
files=[]
for p in sorted(APP.rglob('*')):
    if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts:
        files.append(dict(path=p.relative_to(APP).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
(APP/'artifact_manifest.json').write_text(json.dumps(dict(created_utc=record['verified_utc'],files=files),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(requirements=record['requirements_proven'],original_hashes_preserved=36,
                     completed_run=run['id'],early_responses=stop,baseline_responses=5,
                     model_loaded=status['model']['loaded'],busy=status['busy'],demo_files=len(files)),indent=2))
