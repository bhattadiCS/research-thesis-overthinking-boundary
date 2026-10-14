from pathlib import Path
import json,hashlib,re,difflib
import fitz
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[4];AUDIT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
WORK=AUDIT/'corrected_extended_v5_math_audit1'
old=(ROOT/'ThesisDocs/Masters_Thesis_Formal_v5.md').read_text(encoding='utf-8')
p=WORK/'ThesisDocs/Masters_Thesis_Formal_v5_math_audit1.md';new=p.read_text(encoding='utf-8')
old_phrase='Table 7 applies mutually exclusive priority rules, with empty stops first.'
new_phrase='Table 7 prioritizes missing normalized answers under the archived CSV convention; raw text can be present.'
expected=old.replace('No extracted candidate at the stop','Missing normalized answer at the stop').replace(old_phrase,new_phrase)
assert new==expected,'Unexpected manuscript content change'
math=re.compile(r'\$\$(.+?)\$\$|(?<!\\)\$([^$]+?)(?<!\\)\$',re.S)
assert [m.group() for m in math.finditer(old)]==[m.group() for m in math.finditer(new)]
assert re.findall(r'(?<!\w)[+-]?\d+(?:\.\d+)?',old)==re.findall(r'(?<!\w)[+-]?\d+(?:\.\d+)?',new)
(WORK/'compiled_correction.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='published_v5',tofile='v5_math_audit1')),encoding='utf-8')
pdf=WORK/'Masters_Thesis_Formal_v5_math_audit1_Aditya_Bhatt.pdf';doc=fitz.open(pdf)
render=WORK/'final_renders';render.mkdir(exist_ok=True)
index=[]
for i,page in enumerate(doc):
    page.get_pixmap(dpi=110,alpha=False).save(render/f'page_{i+1:03d}.png')
    index.append({'page':i+1,'headings':[s for s in page.get_text().splitlines() if re.match(r'^(Chapter |Table \d|Figure \d|Appendix |[1-6]\.[1-9])',s)]})
for start in range(0,len(doc),4):
    canvas=Image.new('RGB',(1200,1620),'#ddd');draw=ImageDraw.Draw(canvas)
    for k,i in enumerate(range(start,min(start+4,len(doc)))):
        im=Image.open(render/f'page_{i+1:03d}.png');im.thumbnail((594,785));x=(k%2)*600;y=(k//2)*810
        canvas.paste(im,(x,y+20));draw.text((x+5,y+4),f'v5_math_audit1 PDF page {i+1}',fill='black')
    canvas.save(render/f'sheet_{start//4+1:02d}.png')
record=dict(source=p.relative_to(ROOT).as_posix(),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),pdf=pdf.relative_to(ROOT).as_posix(),pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),pages=len(doc),exact_content_changes_only_two=True,all_257_tex_occurrences_byte_identical=True,all_numeric_literals_identical=True,affected_table_page=[i+1 for i,page in enumerate(doc) if 'Missing normalized answer at the stop' in page.get_text()],pending_visual_review=True,page_index=index)
(WORK/'final_verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in record.items() if k!='page_index'},indent=2))
