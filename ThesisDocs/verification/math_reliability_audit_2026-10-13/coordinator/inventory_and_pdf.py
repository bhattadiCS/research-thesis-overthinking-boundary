"""Read-only exhaustive source units, source bindings, and fresh PDF renders."""
from pathlib import Path
import re, json, hashlib, csv, importlib.util, sys
import fitz
from PIL import Image, ImageOps, ImageDraw

ROOT=Path(__file__).resolve().parents[4]
AUDIT=Path(__file__).resolve().parents[1]
OUT=AUDIT/'coordinator'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s): return re.sub(r'\s+',' ',s).strip()
units=[]
math_re=re.compile(r'\$\$(.*?)\$\$|(?<![\\$])\$(?!\$)(.*?)(?<!\\)\$',re.S)
for ver in ('v7','v5'):
    p=ROOT/f'ThesisDocs/Masters_Thesis_Formal_{ver}.md'
    s=p.read_text(encoding='utf-8'); section='Frontmatter'
    for m in re.finditer(r'(?m)^#{1,3} .+$|(?:[^\n]+(?:\n(?!\n)[^\n]+)*)',s):
        block=m.group(); line=s[:m.start()].count('\n')+1
        if block.startswith('#'): section=block.lstrip('# '); continue
        if section=='References' or section=='Frontmatter': continue
        # Every substantive paragraph and table cell is retained, even nonnumeric
        # qualifiers: this conservative superset avoids keyword-only sampling.
        rows=block.splitlines() if block.startswith('|') else [block]
        for j,row in enumerate(rows):
            if re.fullmatch(r'[| :\-]+',row): continue
            texts=[c.strip() for c in row.strip('|').split('|')] if row.startswith('|') else [row]
            for col,claim in enumerate(texts):
                kind='table_cell' if row.startswith('|') else 'caption' if re.match(r'\*\*(Figure|Table)',claim) else 'figure' if claim.startswith('![') else 'paragraph'
                uid=f'{ver.upper()}-{len([x for x in units if x["version"]==ver])+1:04d}'
                units.append(dict(id=uid,version=ver,path=p.relative_to(ROOT).as_posix(),line=line+j if row.startswith('|') else line,end_line=line+block.count('\n'),section=section,kind=kind,table_column=col+1 if row.startswith('|') else '',claim=claim,dependencies=[],duplicates=[],reviewer='A/B' if section.startswith('Chapter 2') else 'C/coordinator',outcome='NOT ESTABLISHED',evidence='Awaiting explicit claim evidence reconciliation. Paragraph units include all assertions within the quoted text.'))
        # Each expression has its own identity and parent dependency; exact
        # duplicates and repeated inline expressions remain separate locations.
        for em in math_re.finditer(block):
            claim=em.group(1) if em.group(1) is not None else em.group(2)
            parent=next(x['id'] for x in reversed(units) if x['version']==ver and x['kind']!='equation')
            uid=f'{ver.upper()}-{len([x for x in units if x["version"]==ver])+1:04d}'
            units.append(dict(id=uid,version=ver,path=p.relative_to(ROOT).as_posix(),line=line+block[:em.start()].count('\n'),end_line=line+block[:em.end()].count('\n'),section=section,kind='equation',table_column='',claim=claim,dependencies=[parent],duplicates=[],reviewer='A/B' if section.startswith('Chapter 2') else 'C/coordinator',outcome='NOT ESTABLISHED',evidence='Awaiting explicit claim evidence reconciliation.'))
for u in units:
    u['duplicates']=[v['id'] for v in units if v['id']!=u['id'] and v['version']!=u['version'] and norm(v['claim'])==norm(u['claim'])]
(AUDIT/'claim_inventory_initial.json').write_text(json.dumps(units,indent=2,ensure_ascii=False),encoding='utf-8')

# Check receipt source bindings independently, never take their verdict as proof.
bindings=[]
for ver in ('v5','v7'):
    receipt=json.loads((ROOT/f'ThesisDocs/formal/source_integrity_{ver}.json').read_text(encoding='utf-8-sig'))
    if 'source_bindings' in receipt:
        source_bindings=receipt['source_bindings']
    else:
        source_bindings=json.loads((ROOT/f'ThesisDocs/formal/build_manifest_digital_{ver}.json').read_text(encoding='utf-8-sig'))['source_files']
        source_bindings[receipt['compiled_manuscript']]=receipt['compiled_sha256']
    for path,expected in source_bindings.items():
        p=ROOT/path; actual=sha(p) if p.exists() else None
        bindings.append(dict(version=ver,path=path,expected=expected,actual=actual,match=expected==actual))
(OUT/'source_binding_check.json').write_text(json.dumps(bindings,indent=2),encoding='utf-8')

# Reconstruct compilation without running either mutating main() workflow.
sys.path.insert(0,str(ROOT/'tools'))
import build_master_thesis as v5
import build_concise_thesis_v7 as v7
refs=re.findall(r'^\[([A-Za-z0-9]+)\]',(ROOT/'ThesisDocs/references.md').read_text(encoding='utf-8'),re.M)
compile_checks=[]
for ver,builder,paths,inserts in [('v7',v7,[ROOT/'ThesisDocs/concise_v7'/n for n in v7.CHAPTERS]+[ROOT/'ThesisDocs/references.md'],v7.inserts()),('v5',v5,[ROOT/'ThesisDocs/chapters'/n for n in v5.CHAPTERS]+[ROOT/'ThesisDocs/references.md',ROOT/'ThesisDocs/appendices.md'],v5.evidence_inserts(False,preserve_figures=True))]:
    rendered=[]
    for path in paths:
        s=path.read_text(encoding='utf-8')
        for key,value in inserts.items(): s=s.replace(f'[[{key}]]',value)
        for i,key in enumerate(refs,1): s=s.replace(f'[{key}]',f'[{i}]')
        if ver=='v5': s=s.replace('images/thesis_v1/','images/thesis_v2/')
        s=s.translate(str.maketrans({c:'-' for c in '\u2010\u2011\u2012\u2013\u2014\u2015'}))
        rendered.append(s)
    actual=(ROOT/f'ThesisDocs/Masters_Thesis_Formal_{ver}.md').read_text(encoding='utf-8')
    body='\n\n'.join(rendered)
    compile_checks.append(dict(version=ver,compiled_body_matches=actual.endswith(body),builder_abstract_in_compiled=builder.ABSTRACT in actual,body_sha256=hashlib.sha256(body.encode()).hexdigest(),source_paths=[p.relative_to(ROOT).as_posix() for p in paths]))
(OUT/'compilation_check.json').write_text(json.dumps(compile_checks,indent=2),encoding='utf-8')

renders=OUT/'renders'; renders.mkdir(exist_ok=True)
comparison=[]; page_index=[]
for ver in ('v7','v5'):
    pub=fitz.open(ROOT/f'output/pdf/Masters_Thesis_Formal_{ver}_Aditya_Bhatt.pdf'); raw=fitz.open(ROOT/f'ThesisDocs/formal/raw_sources/{ver}/digital.pdf')
    for i,page in enumerate(pub):
        pix=page.get_pixmap(dpi=110,alpha=False); path=renders/f'{ver}_page_{i+1:03d}.png'; pix.save(path)
        a=page.get_pixmap(dpi=72,alpha=False); b=raw[i].get_pixmap(dpi=72,alpha=False)
        comparison.append(dict(version=ver,page=i+1,raster_equal=a.samples==b.samples,text_equal=page.get_text()==raw[i].get_text()))
        page_index.append(dict(version=ver,page=i+1,headings=[x for x in page.get_text().splitlines() if re.match(r'^(Chapter |[1-6]\.[1-9]|Appendix |Table \d|Figure \d|Theorem |Proposition |Corollary )',x)]))
    for start in range(0,len(pub),4):
        canvas=Image.new('RGB',(1200,1620),'#ddd'); draw=ImageDraw.Draw(canvas)
        for k,i in enumerate(range(start,min(start+4,len(pub)))):
            im=Image.open(renders/f'{ver}_page_{i+1:03d}.png'); im.thumbnail((594,785))
            x=(k%2)*600; y=(k//2)*810; canvas.paste(im,(x,y+20)); draw.text((x+5,y+4),f'{ver} PDF page {i+1}',fill='black')
        canvas.save(renders/f'{ver}_sheet_{start//4+1:02d}.png')
(OUT/'pdf_raw_comparison.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
(OUT/'pdf_page_index.json').write_text(json.dumps(page_index,indent=2),encoding='utf-8')
print(json.dumps(dict(units=len(units),counts={ver:sum(u['version']==ver for u in units) for ver in ('v7','v5')},bindings_mismatched=[x for x in bindings if not x['match']],compilation=compile_checks,pdf_pages=len(comparison),raw_differences=[x for x in comparison if not x['raster_equal'] or not x['text_equal']]),indent=2))
