"""Independent saved-pair arithmetic; no production analysis module imported."""
import csv, json, hashlib
from pathlib import Path
from fractions import Fraction as F
from scipy.stats import beta
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
BASE=ROOT/'research/outputs/semester2/online_stopping_20261002'
records=[]
def bit(v):
    assert v in ('True','False','0','1')
    return int(v in ('True','1'))
def cp(k,n):
    return (0 if k==0 else float(beta.ppf(.0125,k,n-k+1)),1 if k==n else float(beta.ppf(.9875,k+1,n-k)))
for sub in ('','adversarial_live','learned_main','learned_adversarial'):
    p=BASE/sub/'live_paired_results.csv'
    with p.open(encoding='utf-8',newline='') as h: rows=list(csv.DictReader(h))
    n=len(rows)
    assert n==len(set(r['task_id'] for r in rows))
    b=[bit(r['baseline_correct']) for r in rows]; a=[bit(r['active_correct']) for r in rows]
    plus=sum(x<y for x,y in zip(b,a)); minus=sum(x>y for x,y in zip(b,a))
    bt=sum(int(r['baseline_generated_tokens']) for r in rows); at=sum(int(r['active_generated_tokens']) for r in rows)
    bp=sum(int(r['baseline_prompt_tokens']) for r in rows); ap=sum(int(r['active_prompt_tokens']) for r in rows)
    l1,u1=cp(plus,n);l0,u0=cp(minus,n)
    # Same declared percentile-bootstrap estimand, implemented directly on exact
    # integer token pairs independently of the production uncertainty module.
    rng=np.random.default_rng(804); token=np.array([[int(r['baseline_generated_tokens']),int(r['active_generated_tokens'])] for r in rows],dtype=np.int64)
    vals=[]
    for _ in range(10):
        sums=token[rng.integers(0,n,size=(1000,n))].sum(axis=1)
        vals.extend((sums[:,0]-sums[:,1])/sums[:,0])
    records.append({'panel':sub or 'heuristic_main','source':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'n':n,'unique_tasks':n,'baseline_correct':sum(b),'active_correct':sum(a),'improved':plus,'worsened':minus,'baseline_completion':bt,'active_completion':at,'baseline_prompt':bp,'active_prompt':ap,'saving_exact':str(F(bt-at,bt)),'saving_percent':100*(bt-at)/bt,'prompt_plus_completion_saving_percent':100*(bt+bp-at-ap)/(bt+bp),'accuracy_delta_points':100*F(sum(a)-sum(b),n),'paired_cp_95_points':[100*(l1-u0),100*(u1-l0)],'saving_bootstrap_percent':[float(100*v) for v in np.quantile(vals,[.025,.975])],'stopped_step_counts':{s:sum(r['active_stop_step']==s for r in rows) for s in sorted(set(r['active_stop_step'] for r in rows))},'shared_prefix_identical':sum(bit(r['shared_prefix_identical']) for r in rows),'limits':'Arithmetic relative to saved frozen labels and ledger token counts; does not independently adjudicate generated answers, verify generation events or establish population sampling.'})
    records[-1]['accuracy_delta_points']=float(records[-1]['accuracy_delta_points'])
(OUT/'independent_live_arithmetic.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
