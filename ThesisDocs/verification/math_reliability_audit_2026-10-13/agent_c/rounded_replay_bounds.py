"""Bound what the retained four-decimal fitted coefficients can certify.

No model fit or regenerated outputs. Evaluates all coefficients consistent
with the historical report's decimal rounding using deterministic intervals.
"""
from pathlib import Path
import hashlib,json,re,subprocess
import numpy as np,pandas as pd
from scipy.special import expit
out=Path(__file__).resolve().parent;root=out.parents[3]
p=root/'research/outputs/real_traces_bf16_ladder/active_stopping_simulation_report.md';text=p.read_text(encoding='utf-8');weights={}
for name in ['q_t','alpha','beta']:
 row=next(s for s in text.splitlines() if s.startswith('| '+name+' |'));weights[name]=np.array([float(s.strip()) for s in row.split('|')[2:-1]])
source=root/'research/outputs/real_traces_bf16_ladder/qwen2p5_7b/trace_steps.csv';features=['entropy_mean','confidence','hidden_l2_shift','answer_changed'];f=pd.read_csv(source,usecols=['run_id','step','correct','raw_generation_tokens']+features).sort_values(['run_id','step'],kind='stable');assert f[features].notna().all().all()
x=np.column_stack([np.ones(len(f)),f[features].to_numpy(float)]);radius=.00005*np.abs(x).sum(1);heads={}
for k,w in weights.items():z=x@w;heads[k]=(expit(z-radius),expit(z+radius),expit(z))
ql,qu,q=heads['q_t'];al,au,a=heads['alpha'];bl,bu,b=heads['beta'];f['mu_lower']=(1-qu)*al-qu*bu-.05;f['mu_upper']=(1-ql)*au-ql*bl-.05;f['mu_rounded']=(1-q)*a-q*b-.05
ambiguous=[];never_tokens=never_correct=rounded_tokens=rounded_correct=0;tokenlo=tokenhi=correctlo=correcthi=0;certified=0
for rid,g in f.groupby('run_id',sort=False):
 g=g.reset_index(drop=True);cost=g.raw_generation_tokens.cumsum().to_numpy(int);ys=g.correct.to_numpy(int);last=len(g)-1;possible=[];rounded=last;firstamb=[]
 for i in range(1,last):
  if g.mu_rounded.iloc[i]<=0 and rounded==last:rounded=i
  if g.mu_upper.iloc[i]<=0:possible.append(i);break
  if g.mu_lower.iloc[i]<=0:possible.append(i);firstamb.append(i)
 else:possible.append(last)
 # For consistency, terminal is always forced if no earlier rounded crossing.
 never_tokens+=int(cost[-1]);never_correct+=int(ys[-1]);rounded_tokens+=int(cost[rounded]);rounded_correct+=int(ys[rounded]);tokenlo+=int(min(cost[possible]));tokenhi+=int(max(cost[possible]));correctlo+=int(min(ys[possible]));correcthi+=int(max(ys[possible]))
 if len(possible)==1:certified+=1
 else:ambiguous.append(dict(run_id=rid,possible_steps=[i+1 for i in possible],bounds=[dict(step=i+1,lower=float(g.mu_lower.iloc[i]),upper=float(g.mu_upper.iloc[i]),rounded=float(g.mu_rounded.iloc[i])) for i in possible if i!=last]))
result=dict(scope='Conditional on retained four-decimal coefficients being standard rounding of the originally applied coefficients and the saved raw source features being their exact inputs. Historical source fits raw features with no scaler; feature order is complete. No model fitting or output generation.',report_path=p.relative_to(root).as_posix(),report_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_path=source.relative_to(root).as_posix(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rounding_half_width=.00005,features=features,weights={k:v.tolist() for k,v in weights.items()},rows=len(f),trajectories=f.run_id.nunique(),baseline=dict(tokens=never_tokens,correct=never_correct),rounded_coefficients=dict(tokens=rounded_tokens,correct=rounded_correct),certified_stop_trajectories=certified,ambiguous_stop_trajectories=len(ambiguous),conservative_possible_aggregate=dict(token_total_lower=tokenlo,token_total_upper=tokenhi,correct_count_lower=correctlo,correct_count_upper=correcthi),ambiguous=ambiguous)
result['conservative_outer_aggregate_bounds']=result.pop('conservative_possible_aggregate')
result['aggregate_bound_scope']='Conservative outer bounds formed from separate trajectory possibilities. Shared coefficient vectors couple all trajectories; endpoints are not asserted jointly attainable. 97 trajectories are unclassified by this conservative calculation.'
(out/'rounded_replay_bounds.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print({k:v for k,v in result.items() if k not in ['ambiguous','weights','scope']})
