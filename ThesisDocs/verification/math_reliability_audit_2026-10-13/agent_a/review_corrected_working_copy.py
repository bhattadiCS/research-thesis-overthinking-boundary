"""Read-only independent final-byte/source/predicate review of audit working copy."""
from pathlib import Path
from hashlib import sha256
from collections import Counter
import csv,difflib,json,re,io
import fitz
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
WORK=HERE.parent/'corrected_extended_v5_math_audit1'
EXPECTED='bba3e78c1d6edcc2a38e7f3cd2159382ca5f6a34320413ac42ea08f0775b5559'

def h(path):return sha256(Path(path).read_bytes()).hexdigest()

def main():
    old=ROOT/'ThesisDocs/Masters_Thesis_Formal_v5.md'
    new=WORK/'ThesisDocs/Masters_Thesis_Formal_v5_math_audit1.md'
    final=WORK/'Masters_Thesis_Formal_v5_math_audit1_Aditya_Bhatt.pdf'
    pdfbytes=final.read_bytes();digest=sha256(pdfbytes).hexdigest()
    assert digest==EXPECTED,digest
    # Preserve exact reviewed bytes in own directory if a later layout repair rebuilds.
    snapshot=HERE/('reviewed_revision_'+digest[:12]+'.pdf')
    snapshot.write_bytes(pdfbytes)
    a=old.read_text(encoding='utf-8');b=new.read_text(encoding='utf-8')
    diff=''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile=str(old),tofile=str(new)))
    (HERE/'corrected_working_compiled.diff').write_text(diff,encoding='utf-8')
    edits=[]
    for tag,i,j,k,l in difflib.SequenceMatcher(a=a.splitlines(),b=b.splitlines()).get_opcodes():
        if tag!='equal':edits.append({'operation':tag,'old_start_line':i+1,'old_end_line':j,'new_start_line':k+1,'new_end_line':l,'old':a.splitlines()[i:j],'new':b.splitlines()[k:l]})
    assert len(edits)==2
    pattern=re.compile(r'\$\$(.*?)\$\$|(?<!\\)\$(?!\$)(.*?)(?<!\\)\$(?!\$)',re.S)
    tex=lambda s:[m.group(1) if m.group(1) is not None else m.group(2) for m in pattern.finditer(s)]
    aa,bb=tex(a),tex(b)
    assert len(aa)==len(bb)==257 and aa==bb
    numeric_cells=lambda s:[cell.strip() for line in s.splitlines() if line.lstrip().startswith('|') for cell in line.split('|')[1:-1] if re.search(r'\d',cell)]
    ca,cb=numeric_cells(a),numeric_cells(b)
    assert len(ca)==len(cb)==277 and ca==cb
    taxonomy=ROOT/'research/reports/thesis_failure_audit_v1/loss_classification.csv'
    with taxonomy.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==5735
    cats=Counter(r['category'] for r in rows)
    assert dict(cats)=={'B_stopped_on_empty_answer':196,'C_passed_earlier_correct':460,'D_step1_only_correct':600,'E_next_step_repair':1706,'F_late_repair':2773}
    assert 'verdict_v2_regrade_mcqfix' not in rows[0]
    selected=[r for r in rows if r['category']=='B_stopped_on_empty_answer']
    matched=[];inputs={taxonomy.relative_to(ROOT).as_posix():h(taxonomy)}
    for cell in sorted({r['cell'] for r in selected}):
        targets={(r['run_id'],int(float(r['hd_step']))) for r in selected if r['cell']==cell}
        path=ROOT/'research/outputs/experiment_matrix'/cell/'trace_steps.csv'
        inputs[path.relative_to(ROOT).as_posix()]=h(path)
        with path.open(encoding='utf-8',newline='') as f:
            for r in csv.DictReader(f):
                try:key=(r['run_id'],int(float(r['step'])))
                except (ValueError,TypeError):continue
                if key in targets:
                    matched.append({'cell':cell,'run_id':key[0],'step':key[1],
                                    'answer':r.get('answer',''),'answer_normalized':r.get('answer_normalized','')})
    assert len(matched)==196
    assert len({(r['cell'],r['run_id'],r['step']) for r in matched})==196
    literals=Counter(r['answer_normalized'] for r in matched)
    assert dict(literals)=={'':191,'n/a':5},dict(literals)
    nonempty=sum(bool(r['answer'].strip()) for r in matched)
    assert nonempty==108
    # Verify the actual default pandas CSV semantics invoked by the classifier.
    check=pd.read_csv(io.StringIO('answer_normalized\n""\n"n/a"\n'))
    assert len(check)==2 and check['answer_normalized'].isna().all()
    render=HERE/('revision_final_render_'+digest[:8]);render.mkdir(exist_ok=True)
    doc=fitz.open(stream=pdfbytes,filetype='pdf')
    pages=[]
    for n in [50,51,52,53]:
        page=doc[n-1];png=render/('page_%03d.png'%n)
        page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(png)
        pages.append({'pdf_page_1_based':n,'body_label':page.get_label(),'image':png.relative_to(HERE).as_posix(),'image_sha256':h(png),'text':page.get_text()})
    result={'reviewer':'Agent A','pdf_sha256':digest,'expected_pdf_sha256_matches':True,'pdf_pages':len(doc),
            'compiled_original_sha256':h(old),'compiled_corrected_sha256':h(new),'edits':edits,
            'tex_occurrences_original':len(aa),'tex_occurrences_new':len(bb),'all_tex_occurrences_byte_text_identical':aa==bb,
            'digit_containing_table_cells_original':len(ca),'digit_containing_table_cells_new':len(cb),'all_digit_containing_table_cells_identical':ca==cb,
            'taxonomy_counts':dict(cats),'missing_normalized_literal_counts':dict(literals),'nonempty_raw_extracted_answer_count':nonempty,
            'category_b_share_percent':196/5735*100,'taxonomy_input_sha256':inputs,'rendered_pages':pages,
            'visual_outcome':'REQUIRES CORRECTION OR ADDITIONAL ASSUMPTIONS',
            'visual_finding':'PDF page52/body45 contains only orphan continuation line accuracy preservation. plus folio; page51/body44 ends the same paragraph after two lines.',
            'changed_taxonomy_claim_outcome':'VERIFIED UNDER RECORDED ASSUMPTIONS',
            'dependencies':'Archived labels, stored stops, default CSV missing-value semantics and first active taxonomy predicate; no regrading or count reassignment.',
            'unresolved_endpoints_unchanged':['N1 LOCO/LOMO underlying held-out threshold predictions','historical stacked pipeline endpoint','in-sample replay endpoint'],
            'scope':'Source exact comparison plus read-only predicate check and final-byte renders of pages50-53; not complete84-page visual review.'}
    (HERE/'corrected_working_copy_review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['taxonomy_input_sha256','rendered_pages','edits']},indent=2))

if __name__=='__main__':main()
