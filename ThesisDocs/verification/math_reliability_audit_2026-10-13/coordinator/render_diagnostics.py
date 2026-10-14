from pathlib import Path
import json,hashlib,subprocess,re,collections
import fitz,numpy as np
ROOT=Path(__file__).resolve().parents[4]; OUT=Path(__file__).resolve().parent
units=json.loads((OUT.parent/'claim_inventory_initial.json').read_text(encoding='utf-8'))
diagnostics={"inventory_counts":{f'{v}/{k}':n for (v,k),n in collections.Counter((u['version'],u['kind']) for u in units).items()},'pdf_differences':[],'katex':[]}
for ver in ('v7','v5'):
    pub=fitz.open(ROOT/f'output/pdf/Masters_Thesis_Formal_{ver}_Aditya_Bhatt.pdf');raw=fitz.open(ROOT/f'ThesisDocs/formal/raw_sources/{ver}/digital.pdf')
    for i,(a,b) in enumerate(zip(pub,raw),1):
        p=a.get_pixmap(dpi=72,alpha=False);q=b.get_pixmap(dpi=72,alpha=False)
        x=np.frombuffer(p.samples,dtype=np.uint8).astype(int);y=np.frombuffer(q.samples,dtype=np.uint8).astype(int)
        d=abs(x-y)
        if d.max():
            mask=np.any(d.reshape(p.height,p.width,3),axis=2);ys,xs=np.where(mask)
            diagnostics['pdf_differences'].append(dict(version=ver,page=i,max_channel_difference=int(d.max()),changed_pixels=int(mask.sum()),fraction_changed=float(mask.mean()),mean_abs_channel_difference=float(d.mean()),pixel_bbox=[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())],text_identical=a.get_text()==b.get_text()))
    expressions=[dict(tex=u['claim'].strip(),display=u['end_line']>u['line'] or '\\begin' in u['claim']) for u in units if u['version']==ver and u['kind']=='equation']
    src=OUT/f'{ver}_expressions.json';dst=OUT/f'{ver}_katex_rendered.json';src.write_text(json.dumps(expressions),encoding='utf-8')
    run=subprocess.run(['node',str(ROOT/'tools/render_thesis_math.mjs'),str(ROOT/'tmp/thesis_pdf_runtime'),str(src),str(dst)],capture_output=True,text=True)
    diagnostics['katex'].append(dict(version=ver,expressions=len(expressions),returncode=run.returncode,stderr=run.stderr))
(OUT/'render_diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8');print(json.dumps(diagnostics,indent=2))
