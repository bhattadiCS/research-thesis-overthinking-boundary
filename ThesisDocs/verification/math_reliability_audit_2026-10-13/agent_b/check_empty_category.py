"""Independent read-only source-row reconstruction of the frozen B predicate."""
from pathlib import Path
import ast, hashlib, json
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
REPORT=ROOT/'research/reports/thesis_failure_audit_v1'
MATRIX=ROOT/'research/outputs/experiment_matrix'
classify_path=ROOT/'research/classify_losses.py'
trace_path=ROOT/'research/trace_analysis.py'
ctext=classify_path.read_text(encoding='utf-8')
tt=trace_path.read_text(encoding='utf-8')
# Execute only the production pure sanitizer's definition, never module main,
# training, grading, collection, report-generation or import side effects.
tree=ast.parse(tt)
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_sanitize_step_frame')
code=ast.get_source_segment(tt,node)
namespace={'pd':pd}
exec(compile(code,str(trace_path),'exec'),namespace)
sanitize=namespace['_sanitize_step_frame']
columns=next(ast.literal_eval(n.value) for n in ast.parse(ctext).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TRACE_USECOLS' for t in n.targets))
losses=pd.read_csv(REPORT/'loss_classification.csv',keep_default_na=False,dtype=str)
per=pd.read_csv(REPORT/'per_run_verdicts.csv',keep_default_na=False,dtype=str)
losses['stop_step_int']=pd.to_numeric(losses.hd_step).astype(int)
assert not losses.duplicated(['cell','run_id']).any()
verdict_losses=per.loc[per.verdict=='loss']
assert set(zip(losses.cell,losses.run_id))==set(zip(verdict_losses.cell,verdict_losses.run_id))
assert len(losses)==5735 and len(per)==75965
records=[]; audit=[]; source_hashes={}
for cell,wanted in losses.groupby('cell',sort=True):
    path=MATRIX/cell/'trace_steps.csv'
    source_hashes[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    # Default reading reproduces the frozen classifier's NA semantics and
    # production eligibility. Literal reading is separate and never normalizes
    # raw/normalized answer strings, including 'None', 'N/A', and empty values.
    default=pd.read_csv(path,usecols=lambda c:c in columns,low_memory=False)
    default['_csv_record']=default.index+2
    default,dropped=sanitize(default)
    duplicate_count=int(default.duplicated(['run_id','step']).sum())
    default=default.drop_duplicates(['run_id','step'],keep='first')
    raw=pd.read_csv(path,keep_default_na=False,dtype=str,usecols=['run_id','step','answer','answer_normalized'],low_memory=False)
    raw['_csv_record']=raw.index+2
    original_count=len(raw)
    target=default.merge(wanted[['run_id','stop_step_int','category','stop_answer','stop_answer_normalized']],left_on=['run_id','step'],right_on=['run_id','stop_step_int'],how='inner',validate='one_to_one',suffixes=('_source','_archived'))
    assert len(target)==len(wanted),(cell,len(target),len(wanted))
    literal=raw.set_index('_csv_record')
    for _,r in target.iterrows():
        source=literal.loc[int(r['_csv_record'])]
        assert source['run_id']==r['run_id']
        norm=r['answer_normalized']
        predicate=bool(pd.isna(norm) or str(norm).strip()=='')
        assert predicate==(r['category']=='B_stopped_on_empty_answer'),(cell,r['run_id'],predicate,r['category'])
        if predicate:
            records.append({'cell':cell,'run_id':r['run_id'],'step':int(r['step']),'source':path.relative_to(ROOT).as_posix(),'csv_record_number_including_header':int(r['_csv_record']),'literal_answer':source['answer'],'literal_answer_normalized':source['answer_normalized'],'literal_answer_nonempty':bool(source['answer'].strip()),'literal_normalized_nonempty':bool(source['answer_normalized'].strip()),'default_normalized_is_na':bool(pd.isna(norm)),'saved_category':r['category'],'saved_stop_answer_literal':r['stop_answer'],'saved_stop_answer_normalized_literal':r['stop_answer_normalized']})
    audit.append({'cell':cell,'raw_rows':original_count,'sanitizer_dropped_rows':dropped,'sanitized_duplicate_keys':duplicate_count,'loss_stop_rows_joined':len(target),'category_B':sum(r['cell']==cell for r in records)})

assert len(records)==196
result={'method':'All5735 frozen losses joined by source-qualified(cell,run_id,stop_step) to original trace rows; exact production sanitizer executed in isolation on default-NA parse, raw fields independently read with keep_default_na=False,dtype=str; no grader, learner or production report main executed.','source_loss_count':len(losses),'source_verdict_count':len(per),'category_B_reconstructed':len(records),'classification_B_count':int((losses.category=='B_stopped_on_empty_answer').sum()),'source_join_missing':0,'source_predicate_category_mismatches':0,'literal_raw_answer_nonempty':sum(r['literal_answer_nonempty'] for r in records),'literal_raw_answer_empty_or_whitespace':sum(not r['literal_answer_nonempty'] for r in records),'literal_normalized_nonempty':sum(r['literal_normalized_nonempty'] for r in records),'all196_normalized_literals_empty':all(r['literal_answer_normalized']=='' for r in records),'nonempty_literal_answer_value_counts':dict(pd.Series([r['literal_answer'] for r in records if r['literal_answer_nonempty']]).value_counts().items()),'raw_literal_disagrees_with_saved_stop_answer':sum(r['literal_answer']!=r['saved_stop_answer_literal'] for r in records),'records':records,'cell_audit':audit,'source_sha256':{**source_hashes,**{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [REPORT/'loss_classification.csv',REPORT/'per_run_verdicts.csv',classify_path,trace_path,ROOT/'tools/summarize_failure_audit.py']}}}
(OUT/'empty_category_independent.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['records','cell_audit','source_sha256','nonempty_literal_answer_value_counts']},indent=2))
