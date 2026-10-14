"""Sequential final-byte review after candidate1's identified layout defect."""
from pathlib import Path
from hashlib import sha256
import difflib,json,re
import pandas as pd
import fitz

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
WORK=HERE.parent/'corrected_extended_v5_math_audit1'
EXPECTED='5b6b47b64ff1c94c836ef255b84cb5dc7f72ec31bd7b985bc18e72182ff0d373'
MD_EXPECTED='5c2181dae627b4407cebf93f8084a1adad9f4a5584134cc89610f349b7fd3654'

def h(path):return sha256(Path(path).read_bytes()).hexdigest()

def main():
    old=ROOT/'ThesisDocs/Masters_Thesis_Formal_v5.md'
    new=WORK/'ThesisDocs/Masters_Thesis_Formal_v5_math_audit1.md'
    pdf=WORK/'Masters_Thesis_Formal_v5_math_audit1_Aditya_Bhatt.pdf'
    raw=pdf.read_bytes();digest=sha256(raw).hexdigest()
    assert digest==EXPECTED and h(new)==MD_EXPECTED
    (HERE/('reviewed_revision_'+digest[:12]+'.pdf')).write_bytes(raw)
    a=old.read_text(encoding='utf-8');b=new.read_text(encoding='utf-8')
    diff=''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile=str(old),tofile=str(new)))
    (HERE/'corrected_final_compiled.diff').write_text(diff,encoding='utf-8')
    edits=[]
    for tag,i,j,k,l in difflib.SequenceMatcher(a=a.splitlines(),b=b.splitlines()).get_opcodes():
        if tag!='equal':edits.append({'operation':tag,'original_lines':[i+1,j],'corrected_lines':[k+1,l],'original':a.splitlines()[i:j],'corrected':b.splitlines()[k:l]})
    assert len(edits)==2
    assert 'Table 7 prioritizes missing normalized answers under the archived CSV convention; raw text can be present.' in b
    pattern=re.compile(r'\$\$(.*?)\$\$|(?<!\\)\$(?!\$)(.*?)(?<!\\)\$(?!\$)',re.S)
    tex=lambda s:[m.group(1) if m.group(1) is not None else m.group(2) for m in pattern.finditer(s)]
    ta,tb=tex(a),tex(b)
    assert len(ta)==len(tb)==257 and ta==tb
    cells=lambda s:[x.strip() for line in s.splitlines() if line.lstrip().startswith('|') for x in line.split('|')[1:-1] if re.search(r'\d',x)]
    ca,cb=cells(a),cells(b)
    assert len(ca)==len(cb)==277 and ca==cb
    # Reconcile the full stored first-active predicate, in addition to candidate1's
    # independent raw field check on every one of its196 selected stops.
    loss=ROOT/'research/reports/thesis_failure_audit_v1/loss_classification.csv'
    frame=pd.read_csv(loss,low_memory=False)
    predicate=frame['stop_answer_normalized'].isna() | frame['stop_answer_normalized'].astype(str).str.strip().eq('')
    assert len(frame)==5735 and int(predicate.sum())==196
    assert predicate.eq(frame['category'].eq('B_stopped_on_empty_answer')).all()
    initial=json.loads((HERE/'corrected_working_copy_review.json').read_text(encoding='utf-8'))
    assert h(loss)==initial['taxonomy_input_sha256'][loss.relative_to(ROOT).as_posix()]
    baseline=json.loads((HERE.parent/'baseline_hashes.json').read_text(encoding='utf-8-sig'))
    preserved=[]
    for row in baseline:
        path=row['path']
        if path.startswith('ThesisDocs/concise_v7/') or path in ['ThesisDocs/Masters_Thesis_Formal_v7.md','tools/build_concise_thesis_v7.py','output/pdf/Masters_Thesis_Formal_v7_Aditya_Bhatt.pdf','ThesisDocs/Masters_Thesis_Formal_v5.md','output/pdf/Masters_Thesis_Formal_v5_Aditya_Bhatt.pdf']:
            now=h(ROOT/path)
            assert now==row['sha256'],path
            preserved.append({'path':path,'sha256':now,'baseline_matches':True})
    render=HERE/('revision_final_render_'+digest[:8]);render.mkdir(exist_ok=True)
    doc=fitz.open(stream=raw,filetype='pdf');assert len(doc)==83
    pages=[]
    for n in [50,51,52]:
        page=doc[n-1];png=render/('page_%03d.png'%n)
        page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(png)
        pages.append({'pdf_page_1_based':n,'body_label':page.get_label(),'image':png.relative_to(HERE).as_posix(),'image_sha256':h(png),'text':page.get_text()})
    assert 'accuracy preservation.' in pages[1]['text']
    assert pages[2]['text'].startswith('Chapter 5 Online stopping')
    result={'reviewer':'Agent A','final_pdf_sha256':digest,'final_compiled_sha256':h(new),'final_pages':len(doc),
            'original_compiled_sha256':h(old),'exact_edits':edits,'tex_occurrences_each':257,'all_tex_identical':True,
            'digit_containing_table_cells_each':277,'all_numeric_table_cells_identical':True,
            'full_taxonomy_predicate_rows_checked':5735,'missing_normalized_predicate_count':196,'full_predicate_equals_category_B':True,
            'raw_field_evidence':'agent_a/corrected_working_copy_review.json:191 blank plus5 literal n/a;108 nonempty raw answers, all196 unique raw stop keys independently checked.',
            'archived_taxonomy_sha256_unchanged':h(loss),'preserved_baseline_files':preserved,'rendered_pages':pages,
            'changed_claim_outcome':'VERIFIED UNDER RECORDED ASSUMPTIONS',
            'visual_outcome':'PENDING ACTUAL IMAGE REVIEW','visual_scope':'Final PDFpages50-52/body43-45, not complete83-page visual review.',
            'unresolved_endpoint_status_unchanged':['N1 LOCO/LOMO underlying held-out threshold predictions','historical stacked pipeline endpoint','in-sample replay endpoint']}
    (HERE/'corrected_final_copy_review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['exact_edits','preserved_baseline_files','rendered_pages']},indent=2))

if __name__=='__main__':main()
