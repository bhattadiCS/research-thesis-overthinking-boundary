"""Read-only independent arithmetic from frozen caches and paired raw arms.

Does not import the production recomputation module, fit models or regrade.
"""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
V2=ROOT/'research/outputs/experiments_v2'
MATRIX=ROOT/'research/outputs/experiment_matrix'
TARGET=ROOT/'research/outputs/thesis_v1/evidence/algorithm_v2_normalized_effects.csv'
inputs={}

def bind(path):
    path=Path(path)
    key=path.relative_to(ROOT).as_posix()
    inputs[key]={'sha256':sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    return path

def readjson(path):
    return json.loads(bind(path).read_text(encoding='utf-8'))

def readcsv(path,columns=None):
    return pd.read_csv(bind(path),usecols=columns,low_memory=False)

def interval(numerators,denominators,label):
    # Same declared seed protocol, independently written explicit weighted sums.
    seed=int.from_bytes(sha256(('804|'+label).encode()).digest()[:8],'big')
    rng=np.random.default_rng(seed)
    n=len(numerators)
    indices=rng.integers(n,size=(10000,n))
    # Frequency vectors clarify that clusters, rather than individual rows, resample.
    weights=np.asarray([np.bincount(draw,minlength=n) for draw in indices])
    ratios=(weights@np.asarray(numerators))/(weights@np.asarray(denominators))
    return [float(x) for x in np.quantile(ratios,[.025,.975])]

def check_target(name,result,target):
    row=target.loc[name]
    recomputed=np.asarray([result['mean_effect'],*result['interval']],dtype=float)
    stored=np.asarray([row['mean_controlled_effect'],row['ci_95_low'],row['ci_95_high']],dtype=float)
    result['stored_values']=stored.tolist()
    result['absolute_differences']=abs(recomputed-stored).tolist()
    result['matches_stored_at_1e_minus_12']=bool(np.allclose(recomputed,stored,rtol=0,atol=1e-12))
    result['v7_rounding']=[format(x,'+.5f') for x in recomputed]
    assert result['matches_stored_at_1e_minus_12'],(name,result)

def raw_steps(path):
    frame=readcsv(path,['run_id','task_id','step','correct','utility','expected_answer','temperature','seed'])
    for name in ['step','correct','utility','temperature','seed']:
        frame[name]=pd.to_numeric(frame[name],errors='raise')
    assert not frame.duplicated(['run_id','step']).any()
    assert frame['correct'].isin([0,1]).all()
    assert (frame['step']>=1).all() and (frame['step']%1==0).all()
    assert frame.groupby('run_id')['task_id'].nunique().eq(1).all()
    return frame

def paired_policy(arm):
    frame=readcsv(arm/'detector_comparison_by_run.csv',
                  ['run_id','task_id','detector','stop_step','stop_utility'])
    frame=frame[frame['detector'].isin(['hazard_drift','never_stop'])].copy()
    assert not frame.duplicated(['run_id','detector']).any()
    raw=raw_steps(arm/'trace_steps.csv')
    joined=frame.merge(raw[['run_id','task_id','step','correct']],left_on=['run_id','stop_step'],
                       right_on=['run_id','step'],suffixes=('_detector','_raw'),validate='many_to_one')
    assert len(joined)==len(frame)
    assert joined['task_id_detector'].eq(joined['task_id_raw']).all()
    joined['utility_twentieths']=20*joined['correct']-(joined['step']-1)
    assert np.allclose(joined['stop_utility'],joined['utility_twentieths']/20,rtol=0,atol=1e-12)
    final=raw.groupby('run_id')['step'].max()
    ns=joined[joined.detector=='never_stop'].set_index('run_id')
    assert ns['step'].eq(final.loc[ns.index]).all()
    wide=joined.pivot(index='run_id',columns='detector',values='utility_twentieths')
    wide['loss']=(wide['hazard_drift']<wide['never_stop']).astype(int)
    meta=raw.drop_duplicates('run_id').set_index('run_id')[['task_id','expected_answer','temperature','seed']]
    wide=wide.join(meta,validate='one_to_one')
    assert len(wide)==raw['run_id'].nunique()
    return wide

def main():
    bind(ROOT/'tools/compute_progress_report_review_metrics.py')
    target=readcsv(TARGET).set_index('experiment')
    n14=[readjson(p) for p in sorted((V2/'algov2_cache').glob('*.json'))]
    n23=[readjson(p) for p in sorted((V2/'algov2_cache_n2_n3').glob('*.json'))]
    assert len(n14)==len(n23)==52
    assert [r['cell'] for r in n14]==[r['cell'] for r in n23]
    counts=[int(r['n']) for r in n14]
    assert sum(counts)==75965
    for a,b in zip(n14,n23):
        assert a['mismatch']==0
        assert all(int(b[key]['n'])==a['n'] for key in ['baseline','n2a','n2b','n2c','n3'])
    # N4's one-parameter baseline belongs to its own cache context. It need
    # not equal the separate matched N2/N3 baseline across experiments.
    baseline_differences=[{'cell':a['cell'],'n4_one_parameter':a['n4']['du_1param'],
                          'n23_matched_baseline':b['baseline']['du_1param']}
                         for a,b in zip(n14,n23)
                         if abs(a['n4']['du_1param']-b['baseline']['du_1param'])>1e-10]
    results={}
    for name,key in [('N2a','n2a'),('N2b','n2b'),('N2c','n2c'),('N3','n3'),('N4',None)]:
        if key:
            effect=[r[key]['du_1param']-r['baseline']['du_1param'] for r in n23]
        else:
            effect=[r['n4']['du_2param']-r['n4']['du_1param'] for r in n14]
        result={'effect_total':float(sum(effect)),'mean_effect':float(sum(effect)/sum(counts)),
                'interval':interval(effect,counts,name),'clusters':52,'trajectories':sum(counts),
                'endpoint':'step utility per trajectory','basis':'frozen per-cell cache totals; no refitting'}
        # Independent exact decimal summation of rounded utility twentieths.
        exact_total=sum(round(x*20) for x in effect)/20
        assert abs(exact_total-result['effect_total'])<1e-8
        result['utility_twentieths_total']=int(sum(round(x*20) for x in effect))
        check_target(name,result,target)
        results[name]=result
    # N5 recomputes each binary loss from raw selected label + response count.
    c=paired_policy(MATRIX/'mistral_small_24b_2409__gsm8k')
    t=paired_policy(V2/'p4b_mistral_small_24b_2409__gsm8k_tok512')
    assert set(c.index)==set(t.index) and len(c)==len(t)==1500
    t=t.loc[c.index]
    for key in ['task_id','expected_answer','temperature','seed']:
        assert c[key].astype(str).eq(t[key].astype(str)).all(),key
    assert set(c['temperature'])=={.1,.6,1.0} and c['seed'].eq(7).all()
    assert c.groupby('task_id').size().eq(3).all()
    difference=t['loss']-c['loss']
    task=pd.DataFrame({'task_id':c['task_id'],'difference':difference}).groupby('task_id',sort=False)['difference'].agg(['sum','count'])
    result={'effect_total':int(difference.sum()),'mean_effect':float(difference.mean()),
            'interval':interval(task['sum'],task['count'],'N5'),'clusters':len(task),'trajectories':len(difference),
            'endpoint':'512 minus256 paired loss-risk difference','losses_control':int(c['loss'].sum()),
            'losses_treatment':int(t['loss'].sum()),'discordant_pairs':int((difference!=0).sum()),
            'basis':'raw archived labels and exact integer utility-twentieth reconstruction at stored stop steps'}
    check_target('N5',result,target);results['N5']=result
    # N6 direct raw step-two pairing, including reference/task/temperature/seed equality.
    b=raw_steps(V2/'p8_qwen7b_none'/'trace_steps.csv')
    q=raw_steps(V2/'p8_qwen7b_4bit'/'trace_steps.csv')
    b=b[b.step==2].set_index('run_id');q=q[q.step==2].set_index('run_id')
    assert b.index.is_unique and q.index.is_unique and set(b.index)==set(q.index) and len(b)==1500
    q=q.loc[b.index]
    for key in ['task_id','expected_answer','temperature','seed']:
        assert b[key].astype(str).eq(q[key].astype(str)).all(),key
    assert set(b['temperature'])=={.1,.6,1.0} and b['seed'].eq(7).all()
    assert b.groupby('task_id').size().eq(3).all()
    difference=b['correct']-q['correct']
    task=pd.DataFrame({'task_id':b['task_id'],'difference':difference}).groupby('task_id',sort=False)['difference'].agg(['sum','count'])
    result={'effect_total':int(difference.sum()),'mean_effect':float(difference.mean()),
            'interval':interval(task['sum'],task['count'],'N6'),'clusters':len(task),'trajectories':len(difference),
            'endpoint':'BF16 minus4bit step-two correctness','bf16_correct':int(b['correct'].sum()),
            'fourbit_correct':int(q['correct'].sum()),'discordant_pairs':int((difference!=0).sum()),
            'basis':'raw archived step-two labels; no fitting or regrading'}
    check_target('N6',result,target);results['N6']=result
    for name,result in results.items():
        result['claim_id']='A-NUM-'+name
        result['outcome']='VERIFIED UNDER RECORDED ASSUMPTIONS'
    inventory=json.loads((HERE.parent/'claim_inventory_initial.json').read_text(encoding='utf-8-sig'))
    line_names={208:'N2a',209:'N2b',210:'N2c',211:'N3',212:'N4',213:'N5',214:'N6'}
    coverage=[]
    for row in inventory:
        if row['version']=='v7' and row['line'] in line_names:
            name=line_names[row['line']]
            # Match labels/endpoints to declared cache contrast; numerical cells
            # are fully reproduced, estimator training provenance is separate.
            coverage.append({'inventory_id':row['id'],'path':row['path'],'line':row['line'],
                             'column':row.get('table_column'),'claim':row['claim'],
                             'outcome':'VERIFIED UNDER RECORDED ASSUMPTIONS',
                             'agent_a_claim_id':'A-NUM-'+name,
                             'evidence':'agent_a/numerical_cross_review.json results.'+name,
                             'limit':'Estimator-label provenance is read as the frozen cache/key contract, not independently refit. Archived endpoint labels are not regraded.'})
    (HERE/'numerical_coverage_map.json').write_text(json.dumps(coverage,indent=2)+'\n',encoding='utf-8')
    output={'reviewer':'Agent A','independence':'Production script read for file paths/declared seeds only; no production recompute helper or Agent C result imported.',
            'results':results,'inputs':inputs,'cross_experiment_baseline_differences':baseline_differences,
            'limits':['N2/N3/N4 reproduction begins at frozen per-cell cached policy totals, not raw saved per-trajectory fitted outputs.',
                      'No new fitting/training/collection/generation/regrading; label semantics are not adjudicated by arithmetic.',
                      'Percentile bootstrap reproduces the declared fixed-panel resampling calculation; it is not proof of population coverage.']}
    (HERE/'numerical_cross_review.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'results':results,'hashed_inputs':len(inputs)},indent=2))

if __name__=='__main__':
    main()
