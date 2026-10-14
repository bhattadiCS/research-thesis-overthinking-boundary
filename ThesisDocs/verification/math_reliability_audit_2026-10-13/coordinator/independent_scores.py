"""Second review of held-out probability metrics without model/sklearn imports."""
from pathlib import Path
import json,csv,math
ROOT=Path(__file__).resolve().parents[4]; OUT=Path(__file__).resolve().parent
BASE=ROOT/'research/outputs/semester2/prefix_model_v1'
rows=list(csv.DictReader((BASE/'heldout_prefix_predictions.csv').open()))
model=json.loads((BASE/'prefix_model.json').read_text())
def sigmoid(z): return 1/(1+math.exp(-z))
def auc_rank(pairs):
    # Count success-vs-failure score orderings using grouped exact ties.
    pairs=sorted(pairs); negatives=0; numerator=0.; positive_total=sum(y for _,y in pairs)
    i=0
    while i<len(pairs):
        j=i+1
        while j<len(pairs) and pairs[j][0]==pairs[i][0]: j+=1
        pos=sum(y for _,y in pairs[i:j]); neg=j-i-pos
        numerator+=pos*(negatives+.5*neg); negatives+=neg; i=j
    return numerator/(positive_total*negatives)
results={}
for name,target in [('q_current','q_target'),('p_next','next_target')]:
    head=model['models'][name] if 'models' in model else model['heads'][name]
    pairs=[]; errors=[]
    for row in rows:
        if not row[target]: continue
        z=head['intercept']+sum((float(row[f])-m)/sd*w for f,m,sd,w in zip(model['feature_names'],head['mean'],head['scale'],head['coefficient']))
        cal=head['calibration']; score=sigmoid(cal['intercept']+cal['slope']*z)
        errors.append(abs(score-float(row[name]))); pairs.append((score,int(float(row[target]))))
    ece=0.
    for k in range(10):
        group=[(p,y) for p,y in pairs if min(int(p*10),9)==k]
        if group: ece+=abs(sum(p-y for p,y in group))/len(pairs)
    result=dict(rows=len(pairs),auc=auc_rank(pairs),brier=sum((p-y)**2 for p,y in pairs)/len(pairs),ece_10_equal_width=ece,maximum_score_error=max(errors))
    assert result['maximum_score_error']<1e-12
    results[name]=result
(OUT/'independent_scores.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(json.dumps(results,indent=2))
