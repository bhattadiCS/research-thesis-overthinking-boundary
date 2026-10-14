"""Replay only already observed ledger events; never grade/generate/train."""
from pathlib import Path
import hashlib,json,sys
sys.dont_write_bytecode=True
out=Path(__file__).resolve().parent;root=out.parents[3];sys.path.insert(0,str(root/'research'))
from online_stopping_controller import Observation,OnlineStoppingController,PolicyConfig
from learned_online_stopping_controller import LearnedPolicy,LearnedOnlineStoppingController
from prefix_stopping_model import PrefixStoppingModel
online=root/'research/outputs/semester2/online_stopping_20261002';result=[]
for folder in [online,online/'adversarial_live',online/'learned_main',online/'learned_adversarial']:
    manifest=json.loads((folder/'live_manifest.json').read_text(encoding='utf-8'));records=[json.loads(s) for s in (folder/'live_generation_results.jsonl').read_text(encoding='utf-8').splitlines()];domains={r['task_id']:r['domain'] for r in (json.loads(s) for s in (folder/'live_public_tasks.jsonl').read_text(encoding='utf-8').splitlines())};mismatches=[];decisions=0
    for r in records:
        cfg=manifest['baseline_policy'] if r['policy']=='never' else manifest['active_policy']
        if r['policy']=='learned_prefix_drift_v1':controller=LearnedOnlineStoppingController(PrefixStoppingModel.load(folder/'frozen_prefix_predictor.json'),domains[r['task_id']],LearnedPolicy(**cfg))
        else:controller=OnlineStoppingController(PolicyConfig(**cfg))
        for observation,expected in zip(r['observations'],r['decisions']):
            assert observation['observed_ns']<=expected['decision_ns']
            # Historical monotonic clocks belong to another boot/session;
            # retain their ordering check, then rebase replay-only timestamps.
            actual=controller.observe(Observation(**{**observation,'observed_ns':0}));decisions+=1
            for field in ['step','stop','reason','selected_answer','selected_step']:
                if getattr(actual,field)!=expected[field]:mismatches.append(dict(task_id=r['task_id'],policy=r['policy'],step=observation['step'],field=field,actual=getattr(actual,field),expected=expected[field]))
        try:controller.observe(Observation(step=len(r['observations'])+1,answer='future',confidence=1,parse_success=True,generated_tokens=1))
        except RuntimeError:pass
        else:raise AssertionError('closed controller accepted future event')
    result.append(dict(folder=folder.relative_to(root).as_posix(),histories=len(records),decisions=decisions,mismatches=mismatches));assert not mismatches
    print(folder.name,len(records),decisions,len(mismatches),flush=True)
p=out/'ancillary_independent.json';a=json.loads(p.read_text(encoding='utf-8'));rows=a['replay_pareto']
for label,r in rows.items():
    r['nondominated']=not any(q['active_correct']>=r['active_correct'] and q['active_completion_tokens']<=r['active_completion_tokens'] and (q['active_correct']>r['active_correct'] or q['active_completion_tokens']<r['active_completion_tokens']) for q in rows.values())
p.write_text(json.dumps(a,indent=2)+'\n',encoding='utf-8')
(out/'runtime_compare.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
