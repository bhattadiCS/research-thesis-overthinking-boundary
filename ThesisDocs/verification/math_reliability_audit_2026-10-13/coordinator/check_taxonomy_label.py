"""Independent literal-field check of the proposed normalized-answer label."""
from pathlib import Path
import csv,json,collections
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
rows=list(csv.DictReader((ROOT/'research/reports/thesis_failure_audit_v1/loss_classification.csv').open(encoding='utf-8',newline='')))
selected=[r for r in rows if r['category']=='B_stopped_on_empty_answer']
checks=[]
for cell in sorted({r['cell'] for r in selected}):
    targets={(r['run_id'],int(float(r['hd_step']))):r for r in selected if r['cell']==cell}
    path=ROOT/'research/outputs/experiment_matrix'/cell/'trace_steps.csv'
    for r in csv.DictReader(path.open(encoding='utf-8',newline='')):
        try: k=(r['run_id'],int(float(r['step'])))
        except (ValueError,TypeError):continue
        if k in targets:
            checks.append(dict(cell=cell,run_id=k[0],step=k[1],answer=r.get('answer',''),normalized=r.get('answer_normalized',''),classification_answer=targets[k].get('stop_answer','')))
assert len(checks)==196
nonempty=[r for r in checks if r['answer'].strip()]
assert len(nonempty)==108
out=dict(cases=len(checks),nonempty_literal_raw_answers=len(nonempty),normalized_field_literals=dict(collections.Counter(r['normalized'] for r in checks)),examples=nonempty[:5],outcome='REQUIRES CORRECTION OR ADDITIONAL ASSUMPTIONS',category='presentation issue',correction='Missing normalized answer at the stop',effect='No counts, shares, labels or utility values change.')
(OUT/'taxonomy_label_check.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
