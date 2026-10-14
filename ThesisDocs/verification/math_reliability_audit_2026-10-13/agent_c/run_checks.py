"""Run existing relevant tests with logs and scratch constrained to audit dir."""
from pathlib import Path
import hashlib,json,os,platform,subprocess,sys,time
out=Path(__file__).resolve().parent; root=out.parents[3];scratch=out/'test_tmp';scratch.mkdir(exist_ok=True)
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',TMP=str(scratch),TEMP=str(scratch))
tests=['test_mathematical_foundations.py','test_live_uncertainty.py','test_online_controller.py','test_learned_online_controller.py','test_prefix_stopping_model.py','test_online_replay_accounting.py','test_live_ledger_audit.py']
results=[]
for name in tests:
    path=root/'research/tests'/name
    cmd=[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider','--basetemp',str(scratch/name.removesuffix('.py')),str(path)] if name in ['test_live_ledger_audit.py'] else [sys.executable,'-B',str(path)]
    start=time.perf_counter();p=subprocess.run(cmd,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (out/(name.removesuffix('.py')+'.log')).write_text(p.stdout,encoding='utf-8')
    results.append(dict(command=cmd,exit_code=p.returncode,elapsed_seconds=time.perf_counter()-start,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),log=name.removesuffix('.py')+'.log'))
    print(name,p.returncode,p.stdout.splitlines()[-2:],flush=True)
(out/'tests.json').write_text(json.dumps(dict(python=sys.executable,version=platform.python_version(),environment_notes='Existing environment; no dependencies changed; bytecode disabled and temp/logs in audit directory.',results=results),indent=2)+'\n',encoding='utf-8')
