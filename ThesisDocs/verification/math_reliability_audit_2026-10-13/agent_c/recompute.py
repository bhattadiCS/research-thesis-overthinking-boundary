"""Independent read-only arithmetic; writes only beside this script. No fit/regrade."""
from pathlib import Path
import collections, hashlib, json, math, re, sys
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
HASHES={}
def track(p):
    p=Path(p); HASHES[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest(); return p
def read(p): return json.loads(track(p).read_text(encoding='utf-8'))
def csv(p,**kwargs): return pd.read_csv(track(p),**kwargs)
def save(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n',encoding='utf-8')
def compare(name,value,reported,tol=1e-11):
    if not np.allclose(value,reported,atol=tol,rtol=tol,equal_nan=True): raise AssertionError((name,value,reported))
def key(s): return ' '.join(s.casefold().split())
def features(prefix,domain):
    o=prefix[-1]; keys=[key(x['answer']) for x in prefix]; streak=1
    for k in keys[-2::-1]:
        if k!=keys[-1]: break
        streak+=1
    trusted=[x['confidence']/100 for x in prefix if x['parse_success'] and x['confidence'] is not None]
    c=(o['confidence']-50)/50 if o['parse_success'] and o['confidence'] is not None else 0
    prev=prefix[-2] if len(prefix)>1 else None
    delta=(o['confidence']-prev['confidence'])/100 if prev and all(x['parse_success'] and x['confidence'] is not None for x in [o,prev]) else 0
    words=re.findall('[a-z]+',o['thought'].casefold())
    return [o['step'],float(o['parse_success']),float(bool(o['answer'].strip())),c,float(not(o['parse_success'] and o['confidence'] is not None)),delta,np.mean(trusted) if trusted else .5,math.log1p(o['generated_tokens']),math.log1p(sum(x['generated_tokens'] for x in prefix)),float(len(keys)>1 and keys[-1]!=keys[-2]),streak,sum(a!=b for a,b in zip(keys,keys[1:])),math.log1p(len(o['answer'])),math.log1p(len(o['thought'])),sum(x.isdigit() for x in o['thought'])/max(1,len(o['thought'])),sum(x in {'maybe','uncertain','unsure','however','alternatively','doubt'} for x in words)/max(1,len(words)),sum(x in {'check','verify','recheck','calculate','confirm','verification'} for x in words)/max(1,len(words)),float(domain=='gsm8k'),float(domain=='math'),o['step']*c,streak*c]
def score(artifact,name,x,calibrated=True):
    m=artifact['models'][name]; z=(np.asarray(x)-m['mean'])/m['scale']; logits=m['intercept']+z@np.asarray(m['coefficient'])
    if calibrated: logits=logits*m['calibration']['slope']+m['calibration']['intercept']
    return expit(logits)
def metrics(y,p):
    ids=np.minimum((p*10).astype(int),9)
    return dict(rows=len(y),auc=roc_auc_score(y,p),brier=np.mean((y-p)**2),ece_10_equal_width=sum(np.sum(ids==i)/len(y)*abs(np.mean(p[ids==i])-np.mean(y[ids==i])) for i in range(10) if np.any(ids==i)))

def live():
    base=ROOT/'research/outputs/semester2/online_stopping_20261002'; result={}; baselines={}
    for name,folder in [('heuristic_main',base),('heuristic_traps',base/'adversarial_live'),('learned_main',base/'learned_main'),('learned_traps',base/'learned_adversarial')]:
        f=csv(folder/'live_paired_results.csv'); m=read(folder/'live_metrics.json'); u=read(folder/'live_uncertainty.json'); manifest=read(folder/'live_manifest.json')
        records=[json.loads(s) for s in track(folder/'live_generation_results.jsonl').read_text(encoding='utf-8').splitlines()]
        n=len(f); b=f.baseline_correct.to_numpy(int); a=f.active_correct.to_numpy(int); plus=int(((a==1)&(b==0)).sum()); minus=int(((a==0)&(b==1)).sum())
        def cp(k):return [0 if k==0 else beta.ppf(.0125,k,n-k+1),1 if k==n else beta.ppf(.9875,k+1,n-k)]
        ip,wp=cp(plus),cp(minus); interval=[ip[0]-wp[1],ip[1]-wp[0]]
        counts=[int(f.baseline_generated_tokens.sum()),int(f.active_generated_tokens.sum())]; savings=1-counts[1]/counts[0]
        rng=np.random.default_rng(804); arr=f[['baseline_generated_tokens','active_generated_tokens']].to_numpy(np.int64); idx=rng.integers(n,size=(10000,n)); totals=arr[idx].sum(1); boot=np.quantile(1-totals[:,1]/totals[:,0],[.025,.975])
        compare(name+' CP',interval,u['accuracy_delta_conservative_exact_95ci']);compare(name+' savings',savings,m['measured_completion_token_savings']);compare(name+' bootstrap',boot,u['completion_token_savings_cluster_bootstrap_95ci'])
        indexed=f.set_index('task_id'); policies=collections.Counter(); reasons=collections.Counter(); strict=collections.Counter(); policy_counts=collections.Counter(); histories=[]
        task_domains={r['task_id']:r['domain'] for r in (json.loads(s) for s in track(folder/'live_public_tasks.jsonl').read_text(encoding='utf-8').splitlines())}
        artifact=read(folder/'frozen_prefix_predictor.json') if name.startswith('learned') else None
        for r in records:
            policy=r['policy']; isbase=policy=='never'; arm='baseline' if isbase else 'active'; row=indexed.loc[r['task_id']]; obs=r['observations']; decisions=r['decisions']
            assert len(obs)==r['stopped_at_step']==len(decisions)
            assert [o['step'] for o in obs]==list(range(1,len(obs)+1))
            assert not any(d['stop'] for d in decisions[:-1]) and decisions[-1]['stop']
            assert sum(o['generated_tokens'] for o in obs)==r['generated_tokens']==int(row[f'{arm}_generated_tokens'])
            assert sum(o['prompt_tokens'] for o in obs)==r['prompt_tokens']==int(row[f'{arm}_prompt_tokens'])
            assert r['answer']==str(row[f'{arm}_answer'])
            assert all(not d['stop'] for d in decisions if d['step']<2)
            policies[policy]+=1; policy_counts[(arm,len(obs))]+=1; reasons[(arm,r['reason'])]+=1; strict[arm]+=sum(o['parse_success'] for o in obs)
            if isbase: histories.append(r)
            if artifact is not None and not isbase:
                for i,d in enumerate(decisions):
                    if d.get('q_current') is None: continue
                    x=features(obs[:i+1],task_domains[r['task_id']])
                    q=float(score(artifact,'q_current',x)); p=float(score(artifact,'p_next',x)); gain=p-q-.05
                    compare(name+' q',q,d['q_current']);compare(name+' next',p,d['p_next']);compare(name+' gain',gain,d['estimated_drift']);assert d['stop']==(gain<=0)
        baselines[name]=histories
        batches=csv(folder/'live_batch_metrics.csv'); ledger={}
        for p,g in batches.groupby('policy',sort=False):
            ledger[p]={c:float(g[c].sum()) for c in ['prompt_tokens','generated_tokens','padded_prefill_token_slots','decode_token_slots','model_seconds','tokenize_seconds']}
        totalprompt=[int(f.baseline_prompt_tokens.sum()),int(f.active_prompt_tokens.sum())]
        result[name]=dict(tasks=n,baseline_correct=int(b.sum()),active_correct=int(a.sum()),improvements=plus,worsenings=minus,completion_tokens=counts,completion_savings=savings,accuracy_interval=interval,token_bootstrap_interval=boot.tolist(),prompt_tokens=totalprompt,total_token_savings=1-(sum([counts[1],totalprompt[1]]))/sum([counts[0],totalprompt[0]]),stop_counts={str(k):v for k,v in policy_counts.items()},termination_reasons={str(k):v for k,v in reasons.items()},strict_json=dict(strict),batch_totals=ledger,shared_prefix_matches=int(f.shared_prefix_identical.sum()))
        print('live',name,n,counts,flush=True)
    assert baselines['heuristic_main']==baselines['learned_main'];assert baselines['heuristic_traps']==baselines['learned_traps']
    latency=csv(base/'learned_main/learned_decision_latency_repeated_prefixes.csv'); print('latency columns',latency.columns.tolist(),flush=True)
    col='latency_ms';v=latency[col].to_numpy();result['learned_latency']=dict(decisions=len(v),median=np.median(v),p99=np.quantile(v,.99),maximum=max(v),columns=latency.columns.tolist())
    save('live_independent.json',result)

def prefix():
    folder=ROOT/'research/outputs/semester2/prefix_model_v1'; artifact=read(folder/'prefix_model.json'); f=csv(folder/'heldout_prefix_predictions.csv'); evaluation=read(folder/'evaluation.json'); split=csv(folder/'task_split.csv');audit=csv(folder/'label_reconstruction_audit.csv',keep_default_na=False)
    assert len(artifact['feature_names'])==21;assert split.task_key.is_unique
    def role(k):
        x=int.from_bytes(hashlib.sha256(('prefix-model-v1-task-split-fixed-20261002::'+k).encode()).digest()[:8],'big')/2**64
        return 'train' if x<.6 else 'calibration' if x<.8 else 'evaluation'
    assert (split.task_key.map(role)==split.split).all(); assert audit.groupby('task_key').split.nunique().max()==1
    public=ROOT/'research/outputs/semester2/online_stopping_20261002/live_public_tasks.jsonl'; trap=ROOT/'research/adversarial_tasks_v1.jsonl'
    excluded={json.loads(s)['task_id'] for p in [public,trap] for s in track(p).read_text(encoding='utf-8').splitlines()}
    assert not any(k.split('::',1)[1] in excluded for k in split.task_key)
    targets=[]
    for tid,g in audit.groupby('trajectory_id',sort=False):
        g=g.sort_values('step'); latest=0
        for r in g.itertuples():
            if r.reconstructed_candidate.strip():latest=int(r.reconstructed_correct)
            targets.append((tid,int(r.step),latest))
    targets=pd.DataFrame(targets,columns=['trajectory_id','step','derived_q']);targets['derived_next']=targets.groupby('trajectory_id',sort=False).derived_q.shift(-1)
    joined=f.merge(targets,on=['trajectory_id','step'],validate='one_to_one');compare('current targets',joined.q_target,joined.derived_q);compare('next targets',joined.next_target,joined.derived_next)
    results={'task_split_counts':split.split.value_counts().to_dict(),'rows_in_label_version':len(audit),'candidate_changes':int(audit.candidate_disagrees.sum()),'label_changes':int(audit.label_disagrees.sum()),'strict_json_rows':int(audit.strict_json.sum()),'public_exclusions':len(excluded),'heads':{}}
    x=f[artifact['feature_names']].to_numpy(float)
    for head,target in [('q_current','q_target'),('p_next','next_target')]:
        mask=f[target].notna().to_numpy(); p=score(artifact,head,x[mask]);raw=score(artifact,head,x[mask],False)
        compare(head+' scoring',p,f.loc[mask,head]); compare(head+' raw scoring',raw,f.loc[mask,head+'_raw']);ms=metrics(f.loc[mask,target].to_numpy(int),p)
        for k,v in ms.items():compare(head+' '+k,v,evaluation['probabilities'][head]['calibrated'][k])
        results['heads'][head]=dict(calibrated=ms,raw=metrics(f.loc[mask,target].to_numpy(int),raw),maximum_score_error=float(np.max(np.abs(p-f.loc[mask,head]))),serialized=artifact['models'][head])
    replay=csv(folder/'heldout_policy_replay.csv');policies={}
    for name,g in replay.groupby('policy',sort=False):
        compare(name+' active utility',g.active_utility,g.active_correct-.05*(g.stop_step-1));compare(name+' baseline utility',g.baseline_utility,g.baseline_correct-.05*4)
        gs=g.groupby('task_key',sort=True).agg(n=('stop_step','size'),d=('accuracy_difference','sum'),active=('active_completion_tokens','sum'),baseline=('baseline_completion_tokens','sum')).to_numpy(float)
        rng=np.random.default_rng(20261002);idx=rng.integers(len(gs),size=(2000,len(gs))); sums=gs[idx].sum(1)
        acc=np.quantile(sums[:,1]/sums[:,0],[.025,.975]);tok=np.quantile(1-sums[:,2]/sums[:,3],[.025,.975]);s=1-g.active_completion_tokens.sum()/g.baseline_completion_tokens.sum()
        e=evaluation['policies'][name];compare(name+' saving',s,e['completion_token_saving_fraction']);compare(name+' accuracy interval',acc,e['task_cluster_95_percentile_intervals']['accuracy_difference']);compare(name+' token interval',tok,e['task_cluster_95_percentile_intervals']['completion_token_saving_fraction'])
        policies[name]=dict(n=len(g),tasks=g.task_key.nunique(),accuracy=g.active_correct.mean(),baseline_accuracy=g.baseline_correct.mean(),stop_counts=g.stop_step.value_counts().to_dict(),completion_saving=s,accuracy_ci=acc.tolist(),token_ci=tok.tolist())
        if name=='learned_drift':
            inferred=f.assign(gain=f.p_next-f.q_current-.05).query('step>=2 and step<5 and gain<=0').groupby('trajectory_id',sort=False).step.min();expected=g.trajectory_id.map(inferred).fillna(5);compare('replayed learned stops',g.stop_step,expected)
    results['replay']=policies;save('prefix_independent.json',results);print('prefix',results['task_split_counts'],results['heads']['q_current']['calibrated'],flush=True)

def tournament():
    root=ROOT/'research/outputs/experiments_v2';frames=[];census=[]
    for p in sorted(root.glob('global_*/trace_steps.csv')):
        f=csv(p,usecols=lambda c:c in ['run_id','task_id','step','correct','domain','raw_generation_tokens','k2_agreement','k2_raw_generation_tokens']);f['trajectory_id']=p.parent.name+'::'+f.run_id;f['source_cell']=p.parent.name;frames.append(f)
        census.append(dict(cell=p.parent.name,rows=len(f),runs=f.run_id.nunique(),tasks=f.task_id.nunique(),domain=f.domain.iloc[0]))
    f=pd.concat(frames).sort_values(['trajectory_id','step'],kind='stable').reset_index(drop=True); assert not f.duplicated(['trajectory_id','step']).any(); assert (f.groupby('trajectory_id').size()==5).all()
    metadata=f.groupby('trajectory_id',sort=False).first(); y=f.correct.to_numpy(int).reshape(-1,5);tasks=metadata.task_id.to_numpy(); domains=metadata.domain.to_numpy();npz=np.load(track(root/'ultimate_oof_predictions.npz'));q=npz['q__gru'];assert q.shape==y.shape
    scores={}
    rawtokens=pd.to_numeric(f.raw_generation_tokens,errors='coerce').fillna(0).to_numpy(np.float32).reshape(-1,5);k2=pd.to_numeric(f.k2_raw_generation_tokens,errors='coerce').fillna(0).to_numpy(np.float32).reshape(-1,5);agreement=pd.to_numeric(f.k2_agreement,errors='coerce').to_numpy().reshape(-1,5)
    display={'gru':'Causal GRU','transformer':'Causal RoPE transformer','linear':'Task-grouped linear baseline','beta':'Truncated Beta mixture','tcn':'Causal residual TCN','ssm':'Selective SSM','fno':'Causal Fourier neural operator','moe':'Five-expert causal MoE','moe_hysteresis':'MoE plus hysteresis'};summary=csv(ROOT/'research/outputs/thesis_v1/evidence/tournament_balanced_summary.csv').set_index('configuration')
    for name in ['gru','transformer','linear','beta','tcn','ssm','fno','moe','moe_hysteresis']:
        source='moe' if name=='moe_hysteresis' else name;q=npz['q__'+source];micro=roc_auc_score(y.ravel(),q.ravel());ds={d:roc_auc_score(y[domains==d].ravel(),q[domains==d].ravel()) for d in np.unique(domains)};taskaucs=[]
        for t in np.unique(tasks):
            mask=tasks==t; yy=y[mask].ravel()
            if len(np.unique(yy))==2: taskaucs.append(roc_auc_score(yy,q[mask].ravel()))
        q64=q.astype(float);mu=(1-q64)*npz['repair__'+source].astype(float)-q64*npz['corruption__'+source].astype(float)-.05;eligible=mu<=0;eligible[:,0]=False;trigger=np.zeros_like(eligible)
        if name=='moe_hysteresis':
            trigger=(q64>.1)&(q64<.9)&np.isin(agreement,[0,1]);trigger[:,0]=False;eligible&=~(trigger&(agreement==0))
        stop=np.where(eligible.any(1),eligible.argmax(1),4);index=np.arange(len(y));correct=y[index,stop];stepu=correct-.05*stop;stoppedtokens=np.cumsum(rawtokens,axis=1,dtype=np.float32)[index,stop]+(np.cumsum(k2*trigger,axis=1,dtype=np.float32)[index,stop] if name=='moe_hysteresis' else 0);tokenu=correct-.0002*stoppedtokens
        scores[name]=dict(micro_auc=micro,domain_auc=ds,domain_macro=np.mean(list(ds.values())),worst_domain=min(ds.values()),task_macro=np.mean(taskaucs),valid_task_auc_groups=len(taskaucs),step_utility=float(stepu.mean()),token_utility=float(tokenu.mean()))
        ref=summary.loc[display[name]]
        for ours,stored in [('micro_auc','micro_oof_auc'),('domain_macro','domain_macro_auc'),('worst_domain','worst_domain_auc'),('task_macro','task_macro_auc'),('step_utility','micro_step_utility')]:compare(name+' '+ours,scores[name][ours],ref[stored])
        compare(name+' token utility',scores[name]['token_utility'],ref.micro_token_utility,tol=1e-7)
    # Strict stored row-level OOF outputs are independently joined to original labels.
    strict={}
    for name,folder,filename,col in [('tabular','strict_tabular_stopping_oof_v1','safe_scalar_lgbm_predictions.csv','oof_q'),('text','strict_text_stopping_oof_v1','tail512_word12_predictions.csv','oof_probability')]:
        g=csv(root/folder/filename);j=g.merge(f[['trajectory_id','step','correct']],on=['trajectory_id','step'],suffixes=('_pred','_source'),validate='one_to_one');assert len(j)==len(f);assert (j.correct_pred==j.correct_source).all()
        strict[name]=dict(rows=len(g),auc=roc_auc_score(g.correct,g[col]),tasks=g.task_id.nunique())
    result=dict(rows=len(f),trajectories=len(metadata),tasks=metadata.task_id.nunique(),cells=len(census),by_domain=f.groupby('domain').agg(rows=('step','size'),trajectories=('trajectory_id','nunique'),tasks=('task_id','nunique')).to_dict('index'),census=census,scores=scores,strict=strict)
    save('tournament_independent.json',result);print('tournament',len(f),len(metadata),result['tasks'],scores['gru'],flush=True)

def matrix():
    matrixroot=ROOT/'research/outputs/experiment_matrix';parts=[];counts=[];verdicts=collections.Counter();categories=collections.Counter();endpoint_errors=0
    guards=['step','correct','confidence','hidden_l2_shift','hidden_cosine_shift','hidden_norm','entropy_mean','utility','elapsed_seconds','seed','k2_agreement','k2_raw_generation_tokens','answer_span_mean_logprob','answer_span_min_logprob','answer_span_mean_entropy','answer_span_std_entropy']
    for p in sorted(matrixroot.glob('*/trace_steps.csv')):
        if not (p.parent/'detector_comparison_by_run.csv').exists():continue
        f=csv(p,usecols=lambda c:c in guards+['run_id','task_id','answer','answer_normalized'],low_memory=False);raw=len(f);rawruns=f.run_id.nunique();bad=pd.Series(False,index=f.index)
        for c in guards:
            if c in f:
                num=pd.to_numeric(f[c],errors='coerce');bad|=num.isna()&f[c].notna()
                if c in ['step','correct']:bad|=num.isna()
        badruns=set(f.loc[bad,'run_id']); f=f.loc[~f.run_id.isin(badruns)].copy();f['step']=pd.to_numeric(f.step).astype(int);f['correct']=pd.to_numeric(f.correct).astype(int);f['cell']=p.parent.name;f['domain']=p.parent.name.rsplit('__',1)[1];f=f.sort_values(['run_id','step'],kind='stable')
        compare('step cost source '+p.parent.name,pd.to_numeric(f.utility),f.correct-.05*(f.step-1))
        d=csv(p.parent/'detector_comparison_by_run.csv');d=d[d.detector.isin(['hazard_drift','never_stop'])];assert not d.duplicated(['run_id','detector']).any();du=d.pivot(index='run_id',columns='detector',values='stop_utility');st=d.pivot(index='run_id',columns='detector',values='stop_step');assert len(du)==f.run_id.nunique()
        diff=du.hazard_drift-du.never_stop;verdicts.update(np.where(diff>0,'win',np.where(diff<0,'loss','tie')))
        lookup=f.set_index(['run_id','step']);hdlabels=du.hazard_drift+.05*(st.hazard_drift-1);nslabels=du.never_stop+.05*(st.never_stop-1)
        observed=lookup.loc[list(zip(du.index,st.hazard_drift.astype(int))),'correct'].to_numpy();compare('stopped labels '+p.parent.name,hdlabels,observed)
        observedfinal=lookup.loc[list(zip(du.index,st.never_stop.astype(int))),'correct'].to_numpy();compare('final labels '+p.parent.name,nslabels,observedfinal)
        losses=diff[diff<0].index;assert np.allclose(hdlabels.loc[losses],0) and np.allclose(nslabels.loc[losses],1)
        for run,g in f[f.run_id.isin(losses)].groupby('run_id',sort=False):
            stop=int(st.loc[run,'hazard_drift']);ans=g.loc[g.step==stop,'answer_normalized'].iloc[0];earlier=g[(g.step>=2)&(g.step<stop)&(g.correct==1)]
            if pd.isna(ans) or not str(ans).strip():cat='B_stopped_on_empty_answer'
            elif len(earlier):cat='C_passed_earlier_correct'
            elif int(g.loc[g.step==1,'correct'].iloc[0])==1:cat='D_step1_only_correct'
            elif int(g.loc[g.step==stop+1,'correct'].iloc[0])==1:cat='E_next_step_repair'
            else:cat='F_late_repair'
            categories[cat]+=1
        counts.append(dict(cell=p.parent.name,raw_rows=raw,sanitized_rows=len(f),raw_runs=rawruns,sanitized_runs=f.run_id.nunique(),removed_runs=len(badruns),tasks=f.task_id.nunique()))
        f['next_correct']=f.groupby('run_id',sort=False).correct.shift(-1);parts.append(f[['cell','domain','run_id','task_id','step','correct','next_correct']]);print('matrix cell',p.parent.name,len(f),flush=True)
    f=pd.concat(parts,ignore_index=True); result=dict(raw_rows=sum(c['raw_rows'] for c in counts),sanitized_rows=len(f),sanitized_runs=sum(c['sanitized_runs'] for c in counts),cells=len(counts),task_ids=f.task_id.nunique(),census=counts,verdicts=dict(verdicts),categories=dict(categories))
    f=f[f.next_correct.notna()].copy();f['repair']=((f.correct==0)&(f.next_correct==1)).astype(int);f['corruption']=((f.correct==1)&(f.next_correct==0)).astype(int);f['drift']=f.next_correct-f.correct-.05
    tables=[]
    for filename,idcols in [('boundary_domain_step_metrics.csv',['domain','step']),('boundary_selected_cell_step_metrics.csv',['cell','step'])]:
        ref=csv(ROOT/'research/outputs/thesis_v1/evidence'/filename)
        for _,r in ref.iterrows():
            g=f
            for col in idcols:g=g[g[col]==r[col]]
            n=len(g);wrong=int((g.correct==0).sum());right=int(g.correct.sum());rep=int(g.repair.sum());cor=int(g.corruption.sum())
            row=dict(trajectories=n,task_clusters=g.task_id.nunique(),accuracy=right/n,repair_events=rep,repair_denominator=wrong,repair_hazard=rep/wrong,corruption_events=cor,corruption_denominator=right,corruption_hazard=cor/right,net_drift=(rep-cor)/n-.05)
            for k,v in row.items():compare(filename+' '+str(tuple(r[c] for c in idcols))+' '+k,v,r[k])
            clustered=g.groupby('task_id',sort=False).agg(n=('correct','size'),q=('correct','sum'),repair=('repair','sum'),corruption=('corruption','sum'),drift=('drift','sum'));a=clustered.to_numpy(float);label=('domain|'+r.domain if idcols[0]=='domain' else 'cell|'+r.cell)+'|'+str(int(r.step))
            for metric,suffix,num,den in [('accuracy','q',a[:,1],a[:,0]),('repair','repair',a[:,2],a[:,0]-a[:,1]),('corruption','corruption',a[:,3],a[:,1]),('net_drift','drift',a[:,4],a[:,0])]:
                seed=int.from_bytes(hashlib.sha256(('804|'+label+'|'+suffix).encode()).digest()[:8],'big');rng=np.random.default_rng(seed);draws=[]
                for _ in range(10):
                    indices=rng.integers(len(a),size=(1000,len(a)));d=den[indices].sum(1);z=num[indices].sum(1);draws.extend(z[d!=0]/d[d!=0])
                interval=np.quantile(draws,[.025,.975]);lo=metric+'_ci_low';hi=metric+'_ci_high';compare(label+' '+metric+' bootstrap',interval,[r[lo],r[hi]]);row[lo]=float(interval[0]);row[hi]=float(interval[1])
            row.update({k:r[k] for k in idcols});tables.append(row)
    result['all_transition_point_estimates']=tables;save('matrix_independent.json',result);print('matrix totals', {k:v for k,v in result.items() if k not in ['census','all_transition_point_estimates']},flush=True)

def prefix_sources():
    sys.path.insert(0,str(ROOT/'research'))
    from online_generation import complete_json_object
    from real_trace_experiments import parse_generation
    folder=ROOT/'research/outputs/semester2/prefix_model_v1';artifact=read(folder/'prefix_model.json');audit=csv(folder/'label_reconstruction_audit.csv',keep_default_na=False).set_index(['trajectory_id','step']); predictions=csv(folder/'heldout_prefix_predictions.csv').set_index(['trajectory_id','step'],drop=False);protocol=read(folder/'training_protocol.json');rows=[];mismatches=[]
    for source in protocol['sources']:
        p=ROOT/source['path'];sourcef=csv(p,usecols=['run_id','task_id','domain','step','raw_generation_tokens','raw_text','correct','answer'],keep_default_na=False)
        for run,g in sourcef.groupby('run_id',sort=True):
            g=g.sort_values('step');tid=p.parent.name+'::'+run;obs=[]
            for r in g.itertuples():
                strict=complete_json_object(r.raw_text);parsed=strict or parse_generation(r.raw_text,'number' if r.domain=='gsm8k' else 'math','minimal_json');a=audit.loc[(tid,r.step)]
                if str(parsed.get('answer','')).strip()!=a.reconstructed_candidate:mismatches.append(dict(trajectory_id=tid,step=r.step,current_parser_candidate=str(parsed.get('answer','')).strip(),frozen_candidate=a.reconstructed_candidate,split=a.split))
                compare('archive saved label',int(r.correct),int(a.saved_correct))
                o=dict(step=int(r.step),answer=a.reconstructed_candidate,thought=str(parsed.get('thought','')),confidence=float(strict['confidence']) if strict else None,parse_success=strict is not None,generated_tokens=int(r.raw_generation_tokens));obs.append(o);x=features(obs,r.domain)
                rows.append(dict(trajectory_id=tid,split=a.split,**dict(zip(artifact['feature_names'],x))))
                if a.split=='evaluation':compare('raw-source prefix features',x,predictions.loc[(tid,r.step),artifact['feature_names']].to_numpy(float))
    frame=pd.DataFrame(rows);scaler=[]
    for head in ['q_current','p_next']:
        g=frame[(frame.split=='train')&((frame.step<=4) if head=='p_next' else True)];x=g[artifact['feature_names']].to_numpy(float);mean=x.mean(0);std=x.std(0);std[std==0]=1
        compare(head+' scaler mean',mean,artifact['models'][head]['mean']);compare(head+' scaler scale',std,artifact['models'][head]['scale']);scaler.append(dict(head=head,train_rows=len(g),maximum_mean_difference=float(max(abs(mean-artifact['models'][head]['mean']))),maximum_scale_difference=float(max(abs(std-artifact['models'][head]['scale'])))))
    save('prefix_source_independent.json',dict(rows=len(frame),candidate_parser_mismatches=mismatches,scalers=scaler,heldout_features_checked=len(predictions),features=len(artifact['feature_names']),source_use='No references read or regrading/fit executed; frozen reconstructed labels retained.'))
    print('source prefix features/scalers',scaler,'current/frozen parser differences',len(mismatches),flush=True)

def controls():
    v2=ROOT/'research/outputs/experiments_v2';a=[read(p) for p in sorted((v2/'algov2_cache').glob('*.json'))];b=[read(p) for p in sorted((v2/'algov2_cache_n2_n3').glob('*.json'))];assert [x['cell'] for x in a]==[x['cell'] for x in b]
    counts=np.array([r['n'] for r in a]);n=counts.sum();ref=csv(ROOT/'research/outputs/thesis_v1/evidence/algorithm_v2_normalized_effects.csv').set_index('experiment');result={}
    for name in ['N2a','N2b','N2c','N3','N4']:
        effect=np.array([r[name.lower()]['du_1param']-r['baseline']['du_1param'] for r in b]) if name!='N4' else np.array([r['n4']['du_2param']-r['n4']['du_1param'] for r in a]);seed=int.from_bytes(hashlib.sha256(('804|'+name).encode()).digest()[:8],'big');rng=np.random.default_rng(seed);idx=rng.integers(len(a),size=(10000,len(a)));boot=np.quantile(effect[idx].sum(1)/counts[idx].sum(1),[.025,.975]);r=ref.loc[name]
        compare(name+' total',effect.sum(),r.controlled_effect_total);compare(name+' mean',effect.sum()/n,r.mean_controlled_effect);compare(name+' CI',boot,[r.ci_95_low,r.ci_95_high]);result[name]=dict(cells=len(a),trajectories=int(n),total=float(effect.sum()),per_trajectory=float(effect.sum()/n),interval=boot.tolist(),cache_effects=effect.tolist())
    for name in ['N1 LOCO','N1 LOMO']:
        r=ref.loc[name];compare(name+' saved normalization',r.controlled_effect_total/r.n_units,r.mean_controlled_effect);result[name]=dict(reconstructed_normalization=r.controlled_effect_total/r.n_units,outcome='NOT ESTABLISHED',gap='Cache lacks historical per-heldout Ridge predictions; no prohibited refit performed. Saved-total normalization only verified.')
    def panel(folder):
        d=csv(folder/'detector_comparison_by_run.csv');p=d[d.detector.isin(['hazard_drift','never_stop'])].pivot(index='run_id',columns='detector',values='stop_utility');p['loss']=(p.hazard_drift<p.never_stop).astype(int);return p
    p=panel(ROOT/'research/outputs/experiment_matrix/mistral_small_24b_2409__gsm8k');q=panel(v2/'p4b_mistral_small_24b_2409__gsm8k_tok512');joined=p.join(q,lsuffix='_256',rsuffix='_512',validate='one_to_one');assert len(joined)==1500;result['N5']=dict(pairs=len(joined),losses_256=int(joined.loss_256.sum()),losses_512=int(joined.loss_512.sum()),discordances=int((joined.loss_256!=joined.loss_512).sum()),contrast=float((joined.loss_512-joined.loss_256).mean()),interval=[0.,0.])
    def step2(folder):
        f=csv(folder/'trace_steps.csv',usecols=['run_id','task_id','step','correct']);return f[f.step==2].set_index('run_id')
    p=step2(v2/'p8_qwen7b_none');q=step2(v2/'p8_qwen7b_4bit');j=p.join(q,lsuffix='_bf16',rsuffix='_4bit',validate='one_to_one');assert (j.task_id_bf16==j.task_id_4bit).all();j['diff']=j.correct_bf16-j.correct_4bit;g=j.groupby('task_id_bf16',sort=False).agg(n=('diff','size'),d=('diff','sum')).to_numpy(float);seed=int.from_bytes(hashlib.sha256(b'804|N6').digest()[:8],'big');rng=np.random.default_rng(seed);idx=rng.integers(len(g),size=(10000,len(g)));s=g[idx].sum(1);ci=np.quantile(s[:,1]/s[:,0],[.025,.975]);r=ref.loc['N6'];compare('N6 interval',ci,[r.ci_95_low,r.ci_95_high]);result['N6']=dict(pairs=len(j),tasks=len(g),BF16_correct=int(j.correct_bf16.sum()),fourbit_correct=int(j.correct_4bit.sum()),contrast=j['diff'].mean(),interval=ci.tolist())
    save('controls_independent.json',result);print('controls', {k: v.get('per_trajectory',v.get('contrast',v.get('outcome'))) for k,v in result.items()},flush=True)

def ancillary():
    v2=ROOT/'research/outputs/experiments_v2';result={}
    for label,folder in [('anonymous','committee_oof_peer_dynamics_v3'),('fixed13','committee_oof_peer_dynamics_fixed13_v2')]:
        root=v2/folder; a=csv(root/'anonymous_minimal/peer_dynamics_predictions.csv');b=csv(root/'anonymous_minimal_baseline/peer_dynamics_predictions.csv');keys=['task_id','trajectory_id','step'];j=a.merge(b,on=keys,suffixes=('_a','_b'),validate='one_to_one').sort_values(keys,kind='stable').reset_index(drop=True);assert len(j)==len(a)==len(b);assert (j.correct_a==j.correct_b).all() and (j.fold_a==j.fold_b).all()
        assert j.groupby('task_id').fold_a.nunique().max()==1
        tasks,codes=np.unique(j.task_id,return_inverse=True);y=j.correct_a.to_numpy(int);pa=j.oof_probability_a.to_numpy();pb=j.oof_probability_b.to_numpy();scores=[]
        # Weighted Mann-Whitney AUC, tie groups sorted once. Equivalent to
        # concatenating repeated whole-task blocks; avoids repeated sorting.
        prepared=[]
        for p in [pa,pb]:
            order=np.argsort(p,kind='stable');s=p[order];starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1];prepared.append((order,starts))
        rng=np.random.default_rng(20260725)
        for _ in range(2000):
            multiplicity=np.bincount(rng.integers(len(tasks),size=len(tasks)),minlength=len(tasks));w=multiplicity[codes];aucs=[]
            for order,starts in prepared:
                pos=np.add.reduceat((w*y)[order],starts);neg=np.add.reduceat((w*(1-y))[order],starts);aucs.append(float(np.sum(pos*(np.cumsum(neg)-.5*neg))/(pos.sum()*neg.sum())))
            scores.append(aucs[0]-aucs[1])
        ca=roc_auc_score(y,pa);cb=roc_auc_score(y,pb);ci=np.quantile(scores,[.025,.975]);report=read(root/'peer_dynamics_ablation_report.json');ref=report['paired_deltas']['anonymous_minimal_minus_anonymous_minimal_baseline'];compare(label+' peer delta',ca-cb,ref['observed_delta_auc']);compare(label+' peer CI',ci,ref['task_cluster_bootstrap_delta_auc_95_ci'])
        result[label]=dict(rows=len(j),tasks=len(tasks),auc=ca,baseline_auc=cb,delta=ca-cb,interval=ci.tolist(),matched_folds=True,bootstrap_draws=2000)
        print('peer',label,result[label],flush=True)
    for label,path,col in [('selected_no_timing',v2/'selected_answer_oof_v4_no_batch_timing/selected_answer_oof_predictions.csv','selected_answer_lgbm_oof'),('selected_dynamics',v2/'selected_answer_causal_dynamics_oof_v3/medium_capacity_predictions.csv','oof_probability')]:
        f=csv(path);assert not f.duplicated(['task_id','step']).any();result[label]=dict(rows=len(f),tasks=f.task_id.nunique(),auc=roc_auc_score(f.selected_correct,f[col]),brier=float(np.mean((f.selected_correct-f[col])**2)))
    online=ROOT/'research/outputs/semester2/online_stopping_20261002';f=csv(online/'latency_100_math_problems.csv'); lat={}
    for label,g in [('all',f)]+list(f.groupby('policy',sort=False)):
        v=g.latency_ms.to_numpy();lat[label]=dict(decisions=len(v),median=float(np.median(v)),p99=float(np.quantile(v,.99)),maximum=float(v.max()))
    result['heuristic_latency']=lat
    pareto={}
    for p in sorted(online.glob('replay_*_paired.csv')):
        g=csv(p);pareto[p.name]=dict(tasks=g.task_id.nunique(),trajectories=len(g),baseline_correct=int(g.baseline_correct.sum()),active_correct=int(g.active_correct.sum()),baseline_completion_tokens=int(g.baseline_generated_tokens.sum()),active_completion_tokens=int(g.active_generated_tokens.sum()),saving=1-g.active_generated_tokens.sum()/g.baseline_generated_tokens.sum(),mean_stop=float(g.active_stop_step.mean()))
    result['replay_pareto']=pareto
    provenance=[]
    for folder in [online,online/'adversarial_live',online/'learned_main',online/'learned_adversarial']:
        manifest=read(folder/'live_manifest.json')
        for name,expected in manifest['code_sha256'].items():
            p=track(folder/'locked_code'/name);raw=p.read_bytes();actual=hashlib.sha256(raw).hexdigest();lf=hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest();assert expected in [actual,lf];provenance.append(dict(folder=folder.relative_to(ROOT).as_posix(),file=name,expected=expected,raw=actual,canonical_lf=lf))
    result['locked_code_bindings']=provenance
    stack=read(v2/'blackwell_5day_tournament_v1/blackwell_tournament_report.json');result['stacked_diagnostic']=dict(reported_auc=stack['stacked_meta_ensemble_auc'],reported_control_auc=stack['control_baseline_auc'],derived_difference=stack['stacked_meta_ensemble_auc']-stack['control_baseline_auc'],outcome='NOT ESTABLISHED',gap='Only aggregate report retained in recorded output directory; no per-row stack/control probabilities found there. No model refit performed. Aggregate subtraction verified; individual AUC, Brier and 10000 bootstrap lifts not independently reproduced.')
    save('ancillary_independent.json',result)

def metadata():
    result={};rosters={}
    for group,folder,pattern in [('matrix',ROOT/'research/outputs/experiment_matrix','*/metadata.json'),('standardized',ROOT/'research/outputs/experiments_v2','global_*/metadata.json')]:
        rows=[];roster={}
        for p in sorted(folder.glob(pattern)):
            if group=='matrix' and not (p.parent/'detector_comparison_by_run.csv').exists():continue
            m=read(p);roster[m['model']['alias']]=m['model'];rows.append(dict(cell=p.parent.name,model=m['model']['alias'],temperatures=m.get('temperatures'),seeds=m.get('seeds'),max_steps=m.get('max_steps'),completion_cap=m.get('max_new_tokens'),shuffle_seed=m.get('dataset_shuffle_seed'),requested_split=m.get('dataset_split'),task_source=m.get('task_source'),task_count=len(m.get('tasks',[])),task_id_prefixes=sorted(set(t['task_id'].split('_')[0]+'_'+t['task_id'].split('_')[1] for t in m.get('tasks',[])))))
        result[group]=rows;rosters[group]=roster
    result['rosters']=rosters;save('metadata_independent.json',result);print('metadata rosters',{k:len(v) for k,v in rosters.items()},flush=True)

if __name__=='__main__':
    try:
        action=sys.argv[1];globals()[action]()
    finally:save('hashes_'+sys.argv[1]+'.json',HASHES)
