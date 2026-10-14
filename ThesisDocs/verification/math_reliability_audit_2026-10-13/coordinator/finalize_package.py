"""Check audit accounting and freeze a hash inventory; never edit originals."""
from pathlib import Path
import json, hashlib, collections, subprocess, datetime, csv, re

OUT = Path(__file__).resolve().parent
AUDIT = OUT.parent
ROOT = OUT.parents[3]
def read(p):
    return json.loads((AUDIT/p).read_text(encoding='utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

ledger = read('claim_ledger.json')
initial = read('claim_inventory_initial.json')
ids = {r['id'] for r in ledger}
assert len(ids) == len(ledger) == 1348
assert ids == {r['id'] for r in initial}
nodes = {r['id'] for r in read('dependency_nodes.json')}
allowed = {'VERIFIED UNDER RECORDED ASSUMPTIONS', 'INCORRECT',
           'REQUIRES CORRECTION OR ADDITIONAL ASSUMPTIONS', 'NOT ESTABLISHED'}
for unit in ledger:
    assert unit['outcome'] in allowed
    assert unit['reviewer']
    assert (ROOT/unit['path']).is_file()
    assert all(d in nodes or d in ids for d in unit['dependencies'])
    assert all(d in ids for d in unit['duplicates'])
    if unit['outcome'] == 'VERIFIED UNDER RECORDED ASSUMPTIONS':
        assert unit['evidence']
    if unit['outcome'] == 'NOT ESTABLISHED':
        assert unit['remaining_evidence']
with (AUDIT/'claim_ledger.csv').open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
assert [r['id'] for r in rows] == [r['id'] for r in ledger]
assert all(r['parent'] in ids for r in read('sentence_claim_ledger.json'))
for edge in read('dependency_edges.json'):
    assert edge['claim'] in ids and edge['depends_on'] in ids | nodes
assert len([r for r in ledger if r['kind']=='equation']) == 330
assert len(read('unresolved_claims.json')) == 39
assert all(r['exit_code'] == 0 for r in read('agent_c/tests.json')['results'])
assert len(read('agent_c/tests.json')['results']) == 7
baseline = read('baseline_hashes.json')
assert len(baseline) == 36
assert all(sha(ROOT/r['path']) == r['sha256'] for r in baseline)
work = AUDIT/'corrected_extended_v5_math_audit1'
pdf = work/'Masters_Thesis_Formal_v5_math_audit1_Aditya_Bhatt.pdf'
md = work/'ThesisDocs/Masters_Thesis_Formal_v5_math_audit1.md'
assert sha(pdf) == '5b6b47b64ff1c94c836ef255b84cb5dc7f72ec31bd7b985bc18e72182ff0d373'
assert sha(md) == '5c2181dae627b4407cebf93f8084a1adad9f4a5584134cc89610f349b7fd3654'
assert sha(pdf) == read('agent_a/corrected_final_copy_review.json')['final_pdf_sha256']
assert not read('corrected_extended_v5_math_audit1/final_verification.json')['pending_visual_review']
original = (ROOT/'ThesisDocs/Masters_Thesis_Formal_v5.md').read_text(encoding='utf-8')
assert md.read_text(encoding='utf-8') == original.replace('No extracted candidate at the stop','Missing normalized answer at the stop').replace('Table 7 applies mutually exclusive priority rules, with empty stops first.','Table 7 prioritizes missing normalized answers under the archived CSV convention; raw text can be present.')
assert not subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).strip()
head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert head == '5304575ce8d421673ee85e3429a128fbddeee697'
# Check relative local Markdown links in the entry point/report. Remote primary
# sources are documented separately, not mistaken for local artifacts.
for name in ['README.md','AUDIT_REPORT.md']:
    text = (AUDIT/name).read_text(encoding='utf-8')
    for target in re.findall(r'\]\(([^)]+)\)', text):
        if not re.match(r'https?://', target) and target not in ('package_validation.json', 'artifact_manifest.json'):
            assert (AUDIT/target).exists(), (name,target)
validation = dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_units=len(ledger), math_expressions=330, sentences=1197,
    outcomes=dict(collections.Counter(r['outcome'] for r in ledger)),
    scientific_unresolved=39, every_initial_unit_accounted_for=True,
    unique_ids_and_dependency_targets_valid=True, csv_json_ids_agree=True,
    test_suites_succeeded=7, existing_test_cases_passed=87,
    original_hashes_preserved=36, git_head=head, tracked_diff_empty=True,
    final_pdf_sha256=sha(pdf), final_manuscript_sha256=sha(md),
    final_pdf_independently_reviewed=True, source_change_count=2,
    blanket_all_mathematics_verdict=False)
(AUDIT/'package_validation.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8')
files=[]
for p in sorted(AUDIT.rglob('*')):
    if not p.is_file() or p.name=='artifact_manifest.json' or '__pycache__' in p.parts:
        continue
    files.append(dict(path=p.relative_to(AUDIT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    hash_algorithm='SHA-256 of raw bytes', scope='Audit directory, excluding this self-referential manifest and Python bytecode caches.',
    file_count=len(files), total_bytes=sum(r['bytes'] for r in files), files=files)
(AUDIT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(validation=validation,manifest_files=len(files),total_bytes=manifest['total_bytes']),indent=2))
