"""Independent row aggregation and binomial-tail inversion, no thesis imports."""
from pathlib import Path
import csv,json,math,hashlib
import numpy as np
from scipy.stats import binom
ROOT=Path(__file__).resolve().parents[4]; OUT=Path(__file__).resolve().parent
BASE=ROOT/'research/outputs/semester2/online_stopping_20261002'
def cp(k,n):
    # Each marginal has error .025; each tail .0125. Invert finite
    # binomial tails by bisection rather than the production beta PPF.
    tail=.0125
    lo=0.;hi=1.
    if k==0: lower=0.
    else:
        for _ in range(75):
            mid=(lo+hi)/2
            if binom.sf(k-1,n,mid)<tail: lo=mid
            else: hi=mid
        lower=(lo+hi)/2
    lo=0.;hi=1.
    if k==n: upper=1.
    else:
        for _ in range(75):
            mid=(lo+hi)/2
            if binom.cdf(k,n,mid)>tail: lo=mid
            else: hi=mid
        upper=(lo+hi)/2
    return lower,upper
def flag(v):
    if v.lower() in ('true','1'): return 1
    if v.lower() in ('false','0'): return 0
    raise ValueError(v)
results=[]
for folder in (BASE,BASE/'adversarial_live',BASE/'learned_main',BASE/'learned_adversarial'):
    p=folder/'live_paired_results.csv'; rows=list(csv.DictReader(p.open(encoding='utf-8')))
    n=len(rows); assert len({r['task_id'] for r in rows})==n
    b=[flag(r['baseline_correct']) for r in rows]; a=[flag(r['active_correct']) for r in rows]
    improved=sum(x==0 and y==1 for x,y in zip(b,a)); worsened=sum(x==1 and y==0 for x,y in zip(b,a))
    lp,up=cp(improved,n); lm,um=cp(worsened,n)
    bt=sum(int(r['baseline_generated_tokens']) for r in rows); at=sum(int(r['active_generated_tokens']) for r in rows)
    saved=json.loads((folder/'live_uncertainty.json').read_text()); metrics=json.loads((folder/'live_metrics.json').read_text())
    tokens=np.array([[int(r['baseline_generated_tokens']),int(r['active_generated_tokens'])] for r in rows],dtype=np.int64)
    indices=np.random.default_rng(saved['bootstrap_seed']).integers(0,n,(saved['bootstrap_draws'],n))
    totals=tokens[indices].sum(axis=1); draws=(totals[:,0]-totals[:,1])/totals[:,0]
    interval=[lp-um,up-lm]; boot=np.quantile(draws,[.025,.975]).tolist()
    result=dict(panel=folder.relative_to(BASE).as_posix(),source=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),n=n,baseline_correct=sum(b),active_correct=sum(a),improved=improved,worsened=worsened,accuracy_delta=(sum(a)-sum(b))/n,baseline_tokens=bt,active_tokens=at,saving=(bt-at)/bt,cp_binomial_inversion=interval,bootstrap=boot,bootstrap_seed=saved['bootstrap_seed'],bootstrap_draws=saved['bootstrap_draws'],cp_difference_from_recorded=max(abs(x-y) for x,y in zip(interval,saved['accuracy_delta_conservative_exact_95ci'])),bootstrap_difference_from_recorded=max(abs(x-y) for x,y in zip(boot,saved['completion_token_savings_cluster_bootstrap_95ci'])),zero_discordance_upper=1-.0125**(1/n),prompt_completion_saving=1-(metrics['active_prompt_tokens']+at)/(metrics['baseline_prompt_tokens']+bt))
    assert result['cp_difference_from_recorded']<1e-12
    assert result['bootstrap_difference_from_recorded']<1e-12
    assert bt==metrics['baseline_generated_tokens'] and at==metrics['active_generated_tokens']
    results.append(result)
panels=list(csv.DictReader((ROOT/'research/outputs/thesis_v1/evidence/boundary_domain_step_metrics.csv').open()))
boundary=[]
for row in panels:
    n=int(float(row['repair_denominator']))+int(float(row['corruption_denominator']))
    revents=int(float(row['repair_events'])); cevents=int(float(row['corruption_events']))
    q=int(float(row['corruption_denominator']))/n
    alpha=revents/int(float(row['repair_denominator'])) if float(row['repair_denominator']) else None
    beta=cevents/int(float(row['corruption_denominator'])) if float(row['corruption_denominator']) else None
    net=(revents-cevents)/n-.05
    assert abs(q-float(row['accuracy']))<1e-12
    assert abs(net-float(row['net_drift']))<1e-12
    boundary.append(dict(domain=row['domain'],step=int(row['step']),n=n,q=q,alpha=alpha,beta=beta,accuracy_gain=(revents-cevents)/n,net=net))
out=dict(live=results,boundary_arithmetic=boundary,precision_contrast_pp=100*(418-204)/1500,token_step_equivalence=.05/.0002,sampling_limit='Arithmetic conditional on frozen grader labels. iid task sampling and population transport not established by these calculations. Bootstrap draws resample observed tasks.')
(OUT/'independent_numerics.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
