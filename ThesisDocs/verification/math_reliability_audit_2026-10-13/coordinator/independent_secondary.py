from pathlib import Path
import csv,json,collections,math
import numpy as np
from independent_scores import auc_rank
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
def rows(p): return list(csv.DictReader(p.open(encoding='utf-8',newline='')))
results={}
for name,relative,score in [('strict_tabular','strict_tabular_stopping_oof_v1/safe_scalar_lgbm_predictions.csv','oof_q'),('strict_text','strict_text_stopping_oof_v1/tail512_word12_predictions.csv','oof_probability')]:
    r=rows(ROOT/'research/outputs/experiments_v2'/relative)
    results[name]=dict(rows=len(r),tasks=len({x['task_id'] for x in r}),auc=auc_rank([(float(x[score]),int(x['correct'])) for x in r]))
archive=rows(ROOT/'research/outputs/semester2/prefix_model_v1/heldout_policy_replay.csv')
archive=[r for r in archive if r['policy']=='learned_drift'];n=len(archive)
bt=sum(int(r['baseline_completion_tokens']) for r in archive);at=sum(int(r['active_completion_tokens']) for r in archive)
results['heldout_replay']=dict(n=n,stop_counts=dict(collections.Counter(r['stop_step'] for r in archive)),baseline_correct=sum(int(r['baseline_correct']) for r in archive),active_correct=sum(int(r['active_correct']) for r in archive),baseline_tokens=bt,active_tokens=at,saving=1-at/bt,accuracy_delta=sum(int(r['active_correct'])-int(r['baseline_correct']) for r in archive)/n,step_utilities_match=all(abs(float(r['active_utility'])-(int(r['active_correct'])-.05*(int(r['stop_step'])-1)))<1e-12 for r in archive))
for name,p in [('learned_latency',ROOT/'research/outputs/semester2/online_stopping_20261002/learned_main/learned_decision_latency_repeated_prefixes.csv'),('heuristic_latency',ROOT/'research/outputs/semester2/online_stopping_20261002/controller_latency.csv')]:
    if not p.exists():
        results[name]=dict(note='Path unavailable; leave heuristic timing source reproduction to exact Agent C path.')
        continue
    r=rows(p);x=np.array([float(a['latency_ms']) for a in r])
    results[name]=dict(n=len(x),median=float(np.median(x)),p99=float(np.quantile(x,.99)),maximum=float(x.max()))
(OUT/'independent_secondary.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(json.dumps(results,indent=2))
