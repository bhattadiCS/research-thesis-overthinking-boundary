from pathlib import Path
import shutil,json,hashlib,difflib
ROOT=Path(__file__).resolve().parents[4];AUDIT=Path(__file__).resolve().parents[1]
WORK=AUDIT/'corrected_extended_v5_math_audit1';DOCS=WORK/'ThesisDocs';DOCS.mkdir(parents=True,exist_ok=True)
for rel in ['chapters','references.md','appendices.md','images/thesis_v2','images/thesis_v4','formal/source_integrity_v4.json']:
    src=ROOT/'ThesisDocs'/rel;dst=DOCS/rel;dst.parent.mkdir(parents=True,exist_ok=True)
    if src.is_dir():shutil.copytree(src,dst,dirs_exist_ok=True)
    else:shutil.copy2(src,dst)
p=DOCS/'chapters/chapter4_empirical.md';s=p.read_text(encoding='utf-8')
old='Table 7 applies mutually exclusive priority rules, with empty stops first.'
new='Table 7 prioritizes missing normalized answers under the archived CSV convention; raw text can be present.'
assert s.count(old)==1;s=s.replace(old,new);p.write_text(s,encoding='utf-8',newline='\n')
builder=(ROOT/'tools/build_master_thesis.py').read_text(encoding='utf-8')
edits=[('ROOT = Path(__file__).resolve().parents[1]',f'ROOT = Path({str(ROOT)!r})'),('DOCS = ROOT / "ThesisDocs"',f'DOCS = Path({str(DOCS)!r})'),('choices=["v5"], default="v5"','choices=["v5_math_audit1"], default="v5_math_audit1"'),('WORK = ROOT / "tmp/pdfs/formal_thesis" / args.document_version / args.edition',f'WORK = Path({str(WORK/"build")!r}) / args.edition'),('    sync_theory()','    # Audit working copy preserves the independently reviewed copied theory.'),('    prefix_evaluation = read_json(PREFIX_MODEL / "evaluation.json")','    failures["categories"][0]["description"] = "Missing normalized answer at the stop"\n    prefix_evaluation = read_json(PREFIX_MODEL / "evaluation.json")')]
for old,new in edits:
    assert builder.count(old)==1,(old,builder.count(old));builder=builder.replace(old,new)
(WORK/'build_extended_v5_math_audit1.py').write_text(builder,encoding='utf-8',newline='\n')
diff=list(difflib.unified_diff((ROOT/'ThesisDocs/chapters/chapter4_empirical.md').read_text(encoding='utf-8').splitlines(True),s.splitlines(True),fromfile='published/chapter4_empirical.md',tofile='working/chapter4_empirical.md'))
(WORK/'source_correction.diff').write_text(''.join(diff),encoding='utf-8')
record={'version':'v5_math_audit1','purpose':'Unpublished mathematical reliability audit working copy; primary v7 and published v5 unchanged.','semantic_changes':[{'source':'working builder evidence_inserts FAILURE_TABLE','before':'No extracted candidate at the stop','after':'Missing normalized answer at the stop'},{'source':'ThesisDocs/chapters/chapter4_empirical.md','before':'Table 7 applies mutually exclusive priority rules, with empty stops first.','after':'Table 7 prioritizes missing normalized answers under the archived CSV convention; raw text can be present.'}],'numerical_changes':False,'theorem_changes':False,'frozen_data_changes':False,'builder_original_sha256':hashlib.sha256((ROOT/'tools/build_master_thesis.py').read_bytes()).hexdigest(),'builder_working_sha256':hashlib.sha256((WORK/'build_extended_v5_math_audit1.py').read_bytes()).hexdigest()}
(WORK/'revision_manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(WORK)
