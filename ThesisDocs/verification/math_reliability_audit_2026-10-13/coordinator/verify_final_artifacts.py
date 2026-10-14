from pathlib import Path
import json,hashlib,collections
import fitz,numpy as np
ROOT=Path(__file__).resolve().parents[4];AUDIT=Path(__file__).resolve().parents[1];WORK=AUDIT/'corrected_extended_v5_math_audit1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
baseline=json.loads((AUDIT/'baseline_hashes.json').read_text(encoding='utf-8-sig'))
unchanged=[dict(path=r['path'],expected=r['sha256'],actual=sha(ROOT/r['path']),match=sha(ROOT/r['path'])==r['sha256']) for r in baseline]
assert all(r['match'] for r in unchanged)
old=fitz.open(ROOT/'output/pdf/Masters_Thesis_Formal_v5_Aditya_Bhatt.pdf');new=fitz.open(WORK/'Masters_Thesis_Formal_v5_math_audit1_Aditya_Bhatt.pdf')
assert len(old)==len(new)==83
comparison=[]
for i,(a,b) in enumerate(zip(old,new),1):
    x=a.get_pixmap(dpi=110,alpha=False);y=b.get_pixmap(dpi=110,alpha=False)
    x=np.frombuffer(x.samples,dtype=np.uint8).astype(int);y=np.frombuffer(y.samples,dtype=np.uint8).astype(int);d=abs(x-y)
    comparison.append(dict(page=i,text_identical=a.get_text()==b.get_text(),raster_max_channel_difference=int(d.max()),changed_channels_over_one=int((d>1).sum())))
changed=[r for r in comparison if not r['text_identical'] or r['changed_channels_over_one']]
assert [r['page'] for r in changed]==[51],changed
assert 'Missing normalized answer at the stop' in new[50].get_text()
assert 'accuracy preservation.' in new[50].get_text()
assert 'Chapter 5' in new[51].get_text()
verification=json.loads((WORK/'final_verification.json').read_text())
verification.update(pending_visual_review=False,visual_review=dict(coordinator_original_pages_viewed={'v7':list(range(1,34)),'v5':list(range(1,84))},coordinator_final_changed_sheets_viewed=[12,13,14],independent_reviewer='agent_a/corrected_final_copy_review.md',unchanged_final_pages='82 pages have identical text and rendered110dpi RGB values within one channel level of the83-page original already visually inspected; only final page51 materially changes.',changed_page=51,candidate1_rejected='84-page candidate has orphan accuracy-preservation line on page52; preserved separately, superseded by this83-page final.'))
(WORK/'final_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
(AUDIT/'coordinator/final_page_comparison.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
(AUDIT/'baseline_integrity_final.json').write_text(json.dumps(unchanged,indent=2),encoding='utf-8')
summary=dict(baseline_files=len(unchanged),all_original_hashes_unchanged=True,final_pdf_sha256=sha(WORK/'Masters_Thesis_Formal_v5_math_audit1_Aditya_Bhatt.pdf'),pages=83,changed_pages=[r['page'] for r in changed],no_original_version_promoted_or_overwritten=True)
(AUDIT/'coordinator/final_integrity_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
