"""Read-only reconstruction with historical parser; no grading/model fitting."""
from pathlib import Path
import hashlib,importlib.util,json,subprocess,sys
import pandas as pd
out=Path(__file__).resolve().parent;root=out.parents[3];sys.path.insert(0,str(root/'research'))
revision='09225c95c676ab3cae1c4b9586d74f7f803f519f'
raw=subprocess.check_output(['git','show',revision+':research/real_trace_experiments.py'],cwd=root)
p=out/'archived_parser_09225.py';p.write_bytes(raw)
spec=importlib.util.spec_from_file_location('audit_historical_parser',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
audit=pd.read_csv(root/'research/outputs/semester2/prefix_model_v1/label_reconstruction_audit.csv',keep_default_na=False).set_index(['trajectory_id','step']);changes=[];count=0
for domain in ['gsm8k','math']:
    p=root/'research/outputs/experiments_v2'/('global_qwen2p5_0p5b_'+domain)/'trace_steps.csv'
    f=pd.read_csv(p,usecols=['run_id','step','raw_text'],keep_default_na=False)
    for r in f.itertuples():
        tid=p.parent.name+'::'+r.run_id;a=audit.loc[(tid,r.step)];candidate=str(m.parse_generation(r.raw_text,'number' if domain=='gsm8k' else 'math','minimal_json')['answer']).strip();count+=1
        if candidate!=a.reconstructed_candidate:changes.append(dict(trajectory_id=tid,step=r.step,historical_parser_candidate=candidate,frozen_candidate=a.reconstructed_candidate))
result=dict(revision=revision,source_sha256=hashlib.sha256(raw).hexdigest(),rows=count,mismatches=changes,scope='Parser reconstruction only; no references read, labels regraded, or coefficients refit.')
(out/'historical_parser_comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(count,'historical parser mismatches',len(changes))
