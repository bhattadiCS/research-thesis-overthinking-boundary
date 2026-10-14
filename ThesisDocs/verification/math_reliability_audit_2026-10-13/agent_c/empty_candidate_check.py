from pathlib import Path
import json,hashlib
import pandas as pd
out=Path(__file__).resolve().parent;root=out.parents[3]
f=pd.read_csv(root/'research/reports/thesis_failure_audit_v1/loss_classification.csv',keep_default_na=False)
f=f[f.category=='B_stopped_on_empty_answer'];rows=[];hashes={}
for cell,g in f.groupby('cell',sort=False):
    p=root/'research/outputs/experiment_matrix'/cell/'trace_steps.csv';hashes[p.relative_to(root).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    source=pd.read_csv(p,usecols=['run_id','step','answer','answer_normalized'],keep_default_na=False)
    source['step']=pd.to_numeric(source.step,errors='coerce');source=source[source.step.notna()]
    j=g[['run_id','hd_step','stop_answer']].merge(source,left_on=['run_id','hd_step'],right_on=['run_id','step'],validate='one_to_one')
    for r in j.itertuples():
        if str(r.answer).strip():rows.append(dict(cell=cell,run_id=r.run_id,step=r.step,raw_answer=str(r.answer),raw_normalized_field=str(r.answer_normalized),saved_classification_raw_answer=str(r.stop_answer)))
result=dict(category_B=len(f),nonempty_raw_candidates=len(rows),examples=rows,source_hashes=hashes)
(out/'empty_candidate_check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(len(f),len(rows))
