"""Read-only content preservation and restored-fact checks for thesis v7."""
from pathlib import Path
import hashlib, json, re

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'ThesisDocs/verification/content_audit_v7_2026-10-08'
def read(path): return json.loads((ROOT/path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def text(path): return (ROOT/path).read_text(encoding='utf-8')
checks=[]
def check(name,condition,details=None): checks.append({'name':name,'passed':bool(condition),'details':details})
base=read('ThesisDocs/verification/content_audit_v7_2026-10-08/protected_baseline.json')
check('286 previously published/frozen files retain exact bytes',len(base['files'])==286 and all(sha(e['path'])==e['sha256'] for e in base['files']))
for name in ['chapter1_introduction.md','chapter2_mathematics.md','chapter5_discussion.md']:
 check(name+' unchanged from v6',sha('ThesisDocs/concise_v7/'+name)==sha('ThesisDocs/concise_v6/'+name))
new=text('ThesisDocs/Masters_Thesis_Formal_v7.md');old=text('ThesisDocs/Masters_Thesis_Formal_v6.md')
for name in ['chapter3_methods.md','chapter4_results.md']:
 a=text('ThesisDocs/concise_v6/'+name);b=text('ThesisDocs/concise_v7/'+name)
 # Check every original paragraph except the two explicitly expanded paragraphs.
 changes=['The thirteen-model matrix spans','Accuracy is the mean grader label'] if name.startswith('chapter3') else []
 parts=[p for p in a.split('\n\n') if p.strip() and not any(p.startswith(x) for x in changes)]
 check(name+' all other original paragraphs retained',all(p in b for p in parts),len(parts))
def expressions(s): return [a or b for a,b in re.findall(r'\$\$(.*?)\$\$|(?<!\\)\$(.*?)(?<!\\)\$',s,re.S)]
a=expressions(old);b=iter(expressions(new))
check('all 70 ordered v6 math expressions retained',len(a)==70 and all(any(x==y for y in b) for x in a),{'v6':len(a),'v7':len(expressions(new))})
def tables(s): return [x for x in re.findall(r'(?m)^\|[^\n]*\|(?:\n\|[^\n]*\|)+',s)]
check('all three scientific table bodies unchanged',len(tables(new))==3 and tables(new)==tables(old))
check('all three figure paths and captions unchanged',re.findall(r'!\[[^\]]+\]\([^\)]+\)',new)==re.findall(r'!\[[^\]]+\]\([^\)]+\)',old) and re.findall(r'\*\*Figure[^\n]+',new)==re.findall(r'\*\*Figure[^\n]+',old))
models=read('research/outputs/semester2/prefix_model_v1/prefix_model.json')
evaluation=read('research/outputs/semester2/prefix_model_v1/evaluation.json')
check('portable feature count and explicit missing-confidence feature',len(models['feature_names'])==21 and 'confidence_missing' in models['feature_names'] and '21 features' in new,models['feature_names'])
head_rows=[]
for target in ('q_current','p_next'):
 d=evaluation['probabilities'][target]['calibrated']
 values={key:f'{d[key]:.4f}' for key in ('auc','brier','ece_10_equal_width')}
 check(target+' restored held-out metrics equal saved evaluation',all(v in new for v in values.values()) and f"{d['rows']:,} rows" in new,{'target':target,'rows':d['rows'],**values})
 head_rows.append({'target':target,'rows':d['rows'],**values})
manifests=[]
for folder in ['','adversarial_live/','learned_main/','learned_adversarial/']:
 path='research/outputs/semester2/online_stopping_20261002/'+folder+'live_manifest.json';m=read(path)
 expected={'batch_size':32,'max_new_tokens':128,'seed':20261002,'temperature':0.0,'do_sample':False}
 check('restored generation contract '+folder,all(m['generation'][k]==v for k,v in expected.items()) and m['model_snapshot']=='7ae557604adf67be50417f59c2c2f167def9a775',expected)
 manifests.append({'path':path,'sha256':sha(path),'generation':m['generation'],'model_snapshot':m['model_snapshot']})
policy=read(manifests[0]['path'])['active_policy']
check('heuristic parameters match the executed manifest',policy['confidence_threshold']==90 and policy['stable_steps']==2 and policy['confidence_drop']==15 and policy['wobble_changes']==2 and policy['min_peers']==0,policy)
check('token/trajectory notation explicitly defined',all(s in new for s in ['Index $i$ denotes a trajectory','$L_{i,s}$ its recorded completion tokens','total token accounting charges it']))
peer=read('tmp/pdfs/content_audit_v7_2026-10-08/peer_structure.json')
peer_summary=[]
for item in peer:
 toc='\n'.join(p['text'] for p in item['toc'])
 peer_summary.append({'file':item['file'],'pages':item['pages'],'toc_physical_pages':[p['physical_page'] for p in item['toc']],'has_intro':bool(re.search('Introduction',toc,re.I)),'has_theory_or_method':bool(re.search('Method|Preliminar|Theor|Framework',toc,re.I)),'has_experiments_or_results':bool(re.search('Experiment|Result|Empirical|Computational Implementation',toc,re.I))})
check('four thesis structures inspected',len(peer_summary)==4 and all(p['has_intro'] and p['has_theory_or_method'] and p['has_experiments_or_results'] for p in peer_summary),peer_summary)
result={'schema':'thesis-content-restoration-audit-v7','baseline_commit':base['commit'],'core_comparison':'v6 primary thesis','full_support_comparison':'v5 extended report','checks':checks,'restored_head_metrics':head_rows,'executed_live_manifest_bindings':manifests,'evaluation_path':'research/outputs/semester2/prefix_model_v1/evaluation.json','evaluation_sha256':sha('research/outputs/semester2/prefix_model_v1/evaluation.json'),'audit_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'all_pass':all(c['passed'] for c in checks),'scope':'Exact prior paragraph, math-expression, table, figure and source-value comparisons; preserved v5 contains further supporting content not printed in v7. This does not claim a lossless 83-to-34-page rewrite.'}
(OUT/'content_preservation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'all_pass':result['all_pass'],'failed':[c['name'] for c in checks if not c['passed']],'checks':len(checks),'head_metrics':head_rows}))
if not result['all_pass']: raise SystemExit(1)
