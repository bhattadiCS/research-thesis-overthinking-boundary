"""A real local-model smoke comparison through the production HTTP/SSE API."""
from pathlib import Path
import datetime, json, time, urllib.request

OUT = Path(__file__).resolve().parent
BASE = 'http://127.0.0.1:8765'
def get(path):
    with urllib.request.urlopen(BASE+path, timeout=10) as r:
        return json.load(r)
payload = dict(prompt='A box holds 12 red balls and 8 blue balls. How many balls are in the box altogether?',
               domain='gsm8k', policy='learned', step_cost=.05)
before = get('/api/status')
req = urllib.request.Request(BASE+'/api/runs', data=json.dumps(payload).encode(),
                             headers={'Content-Type':'application/json'}, method='POST')
with urllib.request.urlopen(req, timeout=10) as r:
    run_id = json.load(r)['run_id']
started = time.perf_counter()
events=[]
arrivals=[]
with urllib.request.urlopen(BASE+f'/api/runs/{run_id}/events', timeout=180) as stream:
    for line in stream:
        if line.startswith(b'data: '):
            event = json.loads(line[6:])
            events.append(event)
            arrivals.append(dict(seq=event['seq'],type=event['type'],step=event.get('step'),seconds=time.perf_counter()-started))
            if event['type'] in ('complete','cancelled','error'):
                break
assert events[-1]['type']=='complete',events[-1]
assert [e['seq'] for e in events] == list(range(1,len(events)+1))
response_ends=[e for e in events if e['type']=='response_end']
assert [e['step'] for e in response_ends] == [1,2,3,4,5]
decisions=[e for e in events if e['type']=='decision' and e['arm']=='early']
stop=next(e['decision']['step'] for e in decisions if e['decision']['stop'])
assert len(decisions)==stop
assert all(e['arms']==['baseline'] for e in response_ends if e['step']>stop)
summary=events[-1]['summary']
assert summary['physical_completion_tokens']==summary['baseline']['completion_tokens']
assert summary['early']['completion_tokens']==sum(e['observation']['generated_tokens'] for e in response_ends if e['step']<=stop)
assert summary['baseline']['completion_tokens']==sum(e['observation']['generated_tokens'] for e in response_ends)
streaming=[]
for step in range(1,6):
    tokens=[e for e in arrivals if e['type']=='token' and e['step']==step]
    end=next(e for e in arrivals if e['type']=='response_end' and e['step']==step)
    assert len(tokens)>1 and tokens[0]['seconds']<end['seconds']
    streaming.append(dict(step=step,token_events=len(tokens),first_token_seconds=tokens[0]['seconds'],response_end_seconds=end['seconds']))
report=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run_id=run_id,
            status_before=before,request=payload,summary=summary,streaming=streaming,
            real_model_generation=True,active_observation_count=stop,baseline_observation_count=5,
            physical_accounting_verified=True,completed_response_counts_verified=True,
            no_correctness_grade=True)
(OUT/'live_smoke_events.json').write_text(json.dumps(events,indent=2)+'\n',encoding='utf-8')
(OUT/'live_smoke.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
