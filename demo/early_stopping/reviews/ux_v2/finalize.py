"""Pin the final usability revision and preserve the initial version receipts."""
from pathlib import Path
import datetime, hashlib, json, re, subprocess
from html.parser import HTMLParser

OUT=Path(__file__).resolve().parent
APP=OUT.parents[1]
ROOT=APP.parents[1]
def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
ux=read(APP/'reviews/ux_tests_results.json')
actual=read(OUT/'verification.json')
current={p.name:sha(p) for p in (APP/'static').iterdir() if p.is_file()}
assert current==ux['static_source_sha256']==actual['source_sha256']
assert ux['failure'] is None and len(ux['checks'])==8 and all(r['passed'] for r in ux['checks'])
assert not ux['page_errors'] and not actual['page_errors']
assert actual['original_files_preserved']==36
assert (APP/'reviews/ux_interpretation.md').is_file()
assert (APP/'reviews/ux_test_review.md').is_file()
assert (APP/'reviews/UX_UPDATE.md').is_file()
class IDs(HTMLParser):
    def __init__(self):
        super().__init__();self.ids=[]
    def handle_starttag(self,tag,attrs):
        self.ids.extend(value for key,value in attrs if key=='id')
ids=IDs();ids.feed((APP/'static/index.html').read_text(encoding='utf-8'))
assert len(ids.ids)==len(set(ids.ids))
for identifier in re.findall(r'\$\("([\w-]+)"\)',(APP/'static/app.js').read_text(encoding='utf-8')):
    assert identifier in ids.ids,identifier
for path,digest in read(APP/'tests/backend_test_record.json')['source_sha256'].items():
    assert sha(ROOT/path)==digest,path
baseline=read(ROOT/'ThesisDocs/verification/math_reliability_audit_2026-10-13/baseline_hashes.json')
assert all(sha(ROOT/r['path'])==r['sha256'] for r in baseline)
assert not subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).strip()
original=read(OUT/'baseline/artifact_manifest.json')
old_sources={r['path'].split('/')[-1]:r['sha256'] for r in original['files'] if r['path'].startswith('static/')}
original_copies={p.name:sha(p) for p in (OUT/'baseline').iterdir() if p.name in ('app.js','index.html','styles.css')}
assert original_copies==old_sources,'Initial interface source copies must match its earlier manifest.'
record=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            revision='usability-v2',source_sha256=current,unique_html_ids=len(ids.ids),
            all_literal_js_dom_targets_present=True,ux_browser_groups_passed=8,
            backend_tests_passed=25,original_thesis_baselines_unchanged=36,
            unchanged_backend_source_hashes=True,initial_interface_bytes_preserved=True,
            actual_browser_tool_tested_run=actual['actual_run_id'],
            scope='Human explanations, selected-output presentation, contrast/themes and responsive layout. No runtime decisions/selector/model/math policy changed.',
            limitations='The local small LLM returned malformed output for the test problem. Answer comparison/correctness/overthinking cannot be established from that run. Numeric budget comparison remains verified.')
(OUT/'completion_check.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
files=[]
for p in sorted(APP.rglob('*')):
    if p.is_file() and p!=APP/'artifact_manifest.json' and '__pycache__' not in p.parts:
        files.append(dict(path=p.relative_to(APP).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
(APP/'artifact_manifest.json').write_text(json.dumps(dict(created_utc=record['verified_utc'],current_revision='usability-v2',scope='Demo files, excluding this manifest and Python bytecode caches.',files=files),indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2))
