from pathlib import Path
import csv,json,collections,hashlib
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
def rows(p): return csv.DictReader(p.open(encoding='utf-8',newline=''))
summary={};hashes={}
for name,base,pattern in [('matrix',ROOT/'research/outputs/experiment_matrix','*/trace_steps.csv'),('standardized',ROOT/'research/outputs/experiments_v2','global_*/trace_steps.csv')]:
    tasks=set();runs=set();counts=collections.Counter();cells=[];raw=0;eligible=0
    for p in sorted(base.glob(pattern)):
        if name=='matrix' and not (p.parent/'detector_comparison_by_run.csv').exists(): continue
        cells.append(p.parent.name);hashes[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        for r in rows(p):
            raw+=1;tasks.add(r['task_id']);runs.add((p.parent.name,r['run_id']));counts[(p.parent.name,r['run_id'])]+=1
        if name=='matrix':
            seen={r['run_id'] for r in rows(p.parent/'detector_comparison_by_run.csv') if r['detector']=='hazard_drift'}
            eligible+=len(seen)
    summary[name]=dict(cells=len(cells),raw_rows=raw,raw_distinct_runs=len(runs),task_ids=len(tasks),saved_hazard_eligible_runs=eligible if name=='matrix' else None,all_five_responses=all(n==5 for n in counts.values()) if name=='standardized' else None)
assert summary['standardized']['raw_rows']==144440 and summary['standardized']['raw_distinct_runs']==28888 and summary['standardized']['task_ids']==2948
assert summary['matrix']['raw_rows']==798770 and summary['matrix']['saved_hazard_eligible_runs']==75965
(OUT/'independent_census.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');(OUT/'census_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
