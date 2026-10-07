"""Build the concise v6 thesis while preserving the extended v5 document.

Uses the reviewed KaTeX/Chrome/ReportLab authoring components and frozen data.
No model execution, training or historical artifact writes are performed.
"""
from __future__ import annotations
import argparse, hashlib, json, re, subprocess
from pathlib import Path
import fitz
import pandas as pd
from reportlab.pdfgen import canvas
import build_master_thesis as shared

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'ThesisDocs'
SOURCE=DOCS/'concise_v6'
ABSTRACT=(
    "Additional response revisions can repair an answer, corrupt it, or consume computation without sufficient gain. "
    "This thesis studies stopping at complete-response boundaries using graded correctness and explicit computation cost. "
    "It derives repair-corruption drift, specializes finite-horizon optimal stopping, proves a sufficient persistence condition for a myopic rule, and gives a delayed-repair counterexample. "
    "Frozen experiments distinguish a variable-horizon matrix from a standardized corpus of 144,440 rows, 28,888 trajectories and 2,948 tasks. "
    "Matched estimator, token-cap and precision comparisons show protocol-specific trade-offs, including cases where greater estimator capacity or calibration reduces utility. "
    "An executed prefix controller prevents future generation. On 100 GSM8K questions its learned arm saves 56.51 percent of completion tokens with seven correct answers versus six at the full horizon; on twenty traps it saves 52.11 percent with one correct answer in each arm. "
    "Every learned live task stops at the two-response floor, so useful adaptation and accuracy noninferiority remain unestablished. "
    "The results support cost-sensitive stopping experiments with joint reporting of answer quality, physical cost and decision-time information."
)
CHAPTERS=['chapter1_introduction.md','chapter2_mathematics.md','chapter3_methods.md','chapter4_results.md','chapter5_discussion.md']
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def inserts():
    old=shared.evidence_inserts(False,preserve_figures=True)
    corpus=re.sub(r'Table 2\.', 'Table 1.',old['CORPUS_TABLE'],count=1)
    controls=pd.read_csv(ROOT/'research/outputs/thesis_v1/evidence/algorithm_v2_normalized_effects.csv')
    labels={'N1 LOCO':'Threshold, held-out cell','N1 LOMO':'Threshold, held-out model','N2a':'Gradient boosting','N2b':'Isotonic calibration','N2c':'Lagged logistic','N3':'Empirical-Bayes hazards','N4':'Step-two churn','N5':'512 versus 256 tokens','N6':'BF16 versus 4-bit'}
    rows=[]
    for _,r in controls.iterrows():
        unit='Step utility' if r.effect_unit=='step utility per trajectory' else 'Loss risk' if r.effect_unit=='loss-risk difference' else 'Accuracy'
        rows.append([labels[r.experiment],unit,f'{r.mean_controlled_effect:+.5f}',f'[{r.ci_95_low:+.5f}, {r.ci_95_high:+.5f}]'])
    controlled=shared.table(['Matched contrast','Endpoint','Mean effect','95% interval'],rows,
      'Table 2. Controlled development contrasts. Estimator intervals resample 52 cells; token-cap/precision intervals resample 500 task clusters. Utility means are per trajectory; accuracy and risk differences are proportions.')
    rows=[]
    for label,folder in [('GSM8K, heuristic',shared.ONLINE),('Traps, heuristic',shared.ADVERSARIAL),('GSM8K, learned',shared.ONLINE/'learned_main'),('Traps, learned',shared.ONLINE/'learned_adversarial')]:
        m=read(folder/'live_metrics.json');u=read(folder/'live_uncertainty.json');n=m['problems_or_trajectories']
        lo,hi=u['accuracy_delta_conservative_exact_95ci'];slo,shi=u['completion_token_savings_cluster_bootstrap_95ci']
        rows.append([label,str(n),f"{100*m['baseline_accuracy']:.0f}% / {100*m['active_accuracy']:.0f}%",
            f"{m['baseline_generated_tokens']:,} / {m['active_generated_tokens']:,}",
            f"{100*m['measured_completion_token_savings']:.2f}%<br>[{100*slo:.2f}, {100*shi:.2f}]",
            f'[{100*lo:+.2f}, {100*hi:+.2f}]',f"{m['shared_prefix_identical_problems']}/{n}"])
    live=shared.table(['Panel / policy','n','Accuracy (%)<br>full / stopped','Tokens<br>full / stopped','Saving (%)<br>[95% CI]','Change (pp)<br>[95% CI]','Prefix<br>match'],rows,
      'Table 3. Actual paired generation. Tokens count completions. Accuracy intervals use the conservative iid-reference calculation; saving intervals bootstrap paired tasks. Traps are handpicked. Learned rows reuse the previously generated baseline, rather than adding independent baseline collections.')
    return {'CORPUS_TABLE':corpus,'CONTROLLED_TABLE':controlled,'LIVE_TABLE':live,
      'DELAYED_REPAIR_FIGURE':'![Delayed repair stopping counterexample](images/thesis_v4/delayed_repair_tree.png)\n\n**Figure 1. Delayed repair defeats a myopic rule.** The horizon-two example permits stopping at zero. Its first negative drift precedes a later repair; the true persistence condition fails. This theoretical floor differs from the live floor of two.',
      'POPULATION_FIGURE':'![Population accuracy and net continuation gain](images/thesis_v2/population_transitions.png)\n\n**Figure 2. Population accuracy and continuation gain.** Bands are task-cluster bootstrap intervals. The horizontal line marks zero net gain, not zero accuracy. Panel crossings do not identify an optimal action for every prefix.',
      'ACTUAL_COST_FIGURE':'![Actual accuracy and completion-token costs](images/thesis_v2/actual_live_pareto.png)\n\n**Figure 3. Actual paired answer quality and completion cost.** Learned points stop at two on every task and provide no demonstrated adaptation beyond that fixed budget. Point estimates omit intervals, reported in Table 3.'}

def print_part(source,name,work,runtime,chrome,left):
    content=shared.html_math(source,runtime,name)
    if name=='part_4':
        assert content.count('<table>')==2
        live_start=content.rfind('<table>')
        content=content[:live_start]+content[live_start:].replace('<table>','<table class="live-results"><colgroup>'+''.join(f'<col style="width:{width}%">' for width in [16,6,13,19,15,19,12])+'</colgroup>',1)
    bibliography=name=='part_6'
    typography='font-size:11pt;line-height:1.35;' if bibliography else 'font-size:12pt;line-height:2;'
    spacing='8pt' if bibliography else '12pt'
    keep_references='p{break-inside:avoid;}' if bibliography else ''
    css=f'''@page{{size:letter;margin:1in 1in 1.35in {left/72:g}in;}}
    body{{font-family:Arial,sans-serif;{typography}color:#111;margin:0;}}
    p{{margin:0 0 {spacing};orphans:2;widows:2;}}h1{{font-size:18pt;line-height:1.35;margin:0 0 25pt;break-after:avoid;}}h2{{font-size:14pt;line-height:1.4;margin:20pt 0 10pt;break-after:avoid;}}
    table{{font-size:10.1pt;line-height:1.35;border-collapse:collapse;width:100%;margin:12pt 0 18pt;break-inside:avoid;}}.table-block,.figure-block,.math-intro,.proof-ending{{break-inside:avoid;}}
    th{{text-align:left;border-bottom:1pt solid #333;}}td{{border-bottom:.4pt solid #bbb;}}td,th{{padding:6pt 4pt;vertical-align:top;overflow-wrap:anywhere;}}tr{{break-inside:avoid;}}thead{{display:table-header-group;}}
    .live-results{{table-layout:fixed;}}.live-results td:nth-child(2),.live-results td:nth-child(7){{white-space:nowrap;}}
    code{{font-size:10.1pt;overflow-wrap:anywhere;}}img{{width:100%;height:auto;break-inside:avoid;}}a{{color:#222;overflow-wrap:anywhere;text-decoration:none;}}
    .katex{{font-size:1.03em;}}.katex-display{{margin:14pt 0;line-height:1.2;break-inside:avoid;}}.math-inline-short{{white-space:nowrap;}}.math-punctuation{{font-family:Arial,sans-serif;font-size:12pt;}}{keep_references}'''
    src=work/f'{name}.html';pdf=work/f'{name}.pdf';src.write_text(f'<!doctype html><html><head><meta charset="utf-8"><base href="{DOCS.as_uri()}/"><link rel="stylesheet" href="{(runtime/"node_modules/katex/dist/katex.min.css").as_uri()}"><style>{css}</style></head><body>{content}</body></html>',encoding='utf-8',newline='\n')
    subprocess.run([str(chrome),'--headless','--disable-gpu','--no-pdf-header-footer','--allow-file-access-from-files',f'--user-data-dir={work/"chrome_profile"}',f'--print-to-pdf={pdf}','--run-all-compositor-stages-before-draw','--virtual-time-budget=2000',src.as_uri()],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW,timeout=90)
    assert pdf.exists();return pdf

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--edition',choices=['digital','print'],default='digital');p.add_argument('--submission-date',default='October 2026');p.add_argument('--chrome',type=Path,default=Path('C:/Program Files/Google/Chrome/Application/chrome.exe'));a=p.parse_args()
    work=ROOT/'tmp/pdfs/compact_v6'/a.edition;work.mkdir(parents=True,exist_ok=True);runtime=ROOT/'tmp/thesis_pdf_runtime';left=108 if a.edition=='print' else 72
    shared.WORK=work;shared.LEFT=left;shared.ABSTRACT=ABSTRACT;shared.SUBMISSION_DATE=a.submission_date
    paths=[SOURCE/name for name in CHAPTERS]+[DOCS/'references.md'];keys=re.findall(r'^\[([A-Za-z0-9]+)\]',paths[-1].read_text(encoding='utf-8'),re.M);refs={key:str(i+1) for i,key in enumerate(keys)};add=inserts();sources=[]
    for path in paths:
        s=path.read_text(encoding='utf-8')
        for key,v in add.items():s=s.replace(f'[[{key}]]',v)
        assert not re.search(r'\[\[[A-Z_]+\]\]',s),path
        citation_text=re.sub(r'\$\$(.*?)\$\$|(?<!\\)\$(.*?)(?<!\\)\$|`[^`]+`','',s,flags=re.S)
        unknown=set(re.findall(r'\[([A-Za-z][A-Za-z0-9]+)\](?!\()',citation_text))-set(refs);assert not unknown,(path,unknown)
        for key,v in refs.items():s=s.replace(f'[{key}]',f'[{v}]')
        sources.append(s.translate(str.maketrans({c:'-' for c in '\u2010\u2011\u2012\u2013\u2014\u2015'})))
    fronttext=f'# {shared.TITLE}\n\nby\n\nAditya Bhatt\n\nA thesis submitted to Johns Hopkins University in conformity with the requirements for the degree of Master of Science\n\nBaltimore, Maryland\n\n{a.submission_date}\n\n# Abstract\n\n{ABSTRACT}\n\nResearch adviser: Zerotti Woods\n\nSecond reader: Moustapha Pemy'
    compiled='\n\n'.join([fronttext]+sources);(DOCS/'Masters_Thesis_Formal_v6.md').write_text(compiled,encoding='utf-8',newline='\n')
    parts=[print_part(s,f'part_{i+1}',work,runtime,a.chrome,left) for i,s in enumerate(sources)];body=fitz.open();lengths=[]
    for path in parts:
        with fitz.open(path) as part:lengths.append(len(part));body.insert_pdf(part)
    def locate(label):
        for i,page in enumerate(body):
            if re.sub(r'\s+','',label) in re.sub(r'\s+','',page.get_text()):return i+1
        raise ValueError(f'Heading/caption not found: {label}')
    figures=shared.caption_titles(sources,'Figure');tables=shared.caption_titles(sources,'Table');fp=[locate(f'Figure {i}.') for i in range(1,len(figures)+1)];tp=[locate(f'Table {i}.') for i in range(1,len(tables)+1)];titles=[s.splitlines()[0].lstrip('# ') for s in sources];entries=[(len(m[1]),m[2],locate(m[2])) for s in sources for m in re.finditer(r'^(#{1,3}) ([^\n]+)$',s,re.M)]
    front,fe=shared.front_matter(lengths,titles,fp,tp,[],figures,tables,entries);final=fitz.open(front);fc=len(final);final.insert_pdf(body)
    folio=work/'folios.pdf';c=canvas.Canvas(str(folio),pagesize=(612,792),initialFontName='ThesisArial')
    for i in range(len(final)):
        if i:c.setFont('ThesisArial',10);c.drawCentredString(306,75,shared.roman(i+1) if i<fc else str(i-fc+1))
        c.showPage()
    c.save()
    with fitz.open(folio) as overlay:
        for i in range(1,len(final)):final[i].show_pdf_page(final[i].rect,overlay,i)
    for page in final:page.clean_contents(sanitize=True)
    final.set_toc([[1,title,page] for title,page in fe]+[[level,title,fc+page] for level,title,page in entries]);final.set_metadata({'title':shared.TITLE,'author':'Aditya Bhatt','subject':'Concise Master of Science thesis, Applied and Computational Mathematics'});final.xref_set_key(final.pdf_catalog(),'Lang','(en-US)');final.set_page_labels([{'startpage':0,'style':'r','firstpagenum':1},{'startpage':fc,'style':'D','firstpagenum':1}])
    out=work/'source.pdf';shared.inspect_pdf(final);final.save(out,garbage=4,deflate=True)
    with fitz.open(out) as saved:audit=shared.inspect_pdf(saved)
    record={'schema':'concise-formal-thesis-build-v6','edition':a.edition,'output':out.relative_to(ROOT).as_posix(),'sha256':sha(out),'main_builder_sha256':sha(__file__),'shared_renderer_sha256':sha(shared.__file__),'source_files':{p.relative_to(ROOT).as_posix():sha(p) for p in paths},'canonical_math_source_sha256':sha(ROOT/'research/mathematical_foundations.md'),'data_manifest_sha256':sha(ROOT/'data_manifest_post_review_v1.json'),'audit':audit,'word_count':len(re.findall(r"\b[\w'-]+\b",compiled)),'abstract_words':len(ABSTRACT.split()),'front_matter_pages':fc,'submission_month_year':a.submission_date,'left_margin_inches':left/72,'other_minimum_margins_inches':1,'main_text_font_points':12,'main_text_line_height':2,'bibliography_font_points':11,'bibliography_line_height':1.35,'contents_body_entries':[{'level':l,'title':t,'body_page':i} for l,t,i in entries],'chapter_page_counts':dict(zip(titles,lengths)),'figure_titles':figures,'table_titles':tables,'figure_body_pages':fp,'table_body_pages':tp,'extended_document':'output/pdf/Masters_Thesis_Formal_v5_Aditya_Bhatt.pdf','target_total_pages':[25,35],'visual_review':'required before release'}
    (DOCS/f'formal/build_manifest_{a.edition}_v6.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n');print(json.dumps({'pages':len(final),'words':record['word_count'],'front':fc,'parts':record['chapter_page_counts'],'output':str(out)}))

if __name__=='__main__':main()
