from pathlib import Path
import csv,json,collections
import numpy as np
from independent_scores import auc_rank
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
records=[]
for p in sorted((ROOT/'research/outputs/experiments_v2').glob('global_*/trace_steps.csv')):
    for r in csv.DictReader(p.open(encoding='utf-8',newline='')):
        records.append((p.parent.name+'::'+r['run_id'],int(r['step']),r['task_id'],r['domain'],int(r['correct'])))
records.sort(key=lambda r:(r[0],r[1]));pred=np.load(ROOT/'research/outputs/experiments_v2/ultimate_oof_predictions.npz')
labels=np.array([r[4] for r in records]);domains=np.array([r[3] for r in records]);tasks=collections.defaultdict(list)
for i,r in enumerate(records):tasks[r[2]].append(i)
mixed={t:indices for t,indices in tasks.items() if len(set(labels[indices]))==2}
results={}
for name in ['gru','transformer','linear','beta','tcn','ssm','fno','moe']:
    scores=pred['q__'+name].ravel();assert len(scores)==len(records)
    domain_scores={d:auc_rank(list(zip(scores[domains==d],labels[domains==d]))) for d in sorted(set(domains))}
    results[name]=dict(micro_auc=auc_rank(list(zip(scores,labels))),domain_auc=domain_scores,domain_macro_auc=sum(domain_scores.values())/4,worst_domain_auc=min(domain_scores.values()),task_macro_auc=sum(auc_rank([(scores[i],labels[i]) for i in ix]) for ix in mixed.values())/len(mixed),mixed_tasks=len(mixed))
(OUT/'independent_tournament.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(json.dumps(results,indent=2))
