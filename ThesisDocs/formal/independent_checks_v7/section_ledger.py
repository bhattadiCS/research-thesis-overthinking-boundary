"""Account for every v5 scientific section without claiming lossless compression."""
from pathlib import Path
import re,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'ThesisDocs/verification/content_audit_v7_2026-10-08'
locations={
 '1.1':'1.1','1.2':'1.1','1.3':'1.1','1.4':'1.2','1.5':'5.2','1.6':'1.2','1.7':'1.3',
 '2.1':'2.1','2.2':'2.2','2.3':'2.3','2.4':'2.4','2.5':'2.1 and 5.1','2.6':'5.1 summary; full derivations in v5','2.7':'1.2 and 5.1 scope; full certificates in v5','2.8':'1.2 and 5.1',
 '3.1':'3.1 and Table 1','3.2':'3.1 restored roster; full ID table in v5','3.3':'3.1','3.4':'3.1 and 3.3 restored settings/floor','3.5':'3.1, 3.3 and 5.1; detailed parser cases in v5','3.6':'3.2','3.7':'3.2 restored notation/accounting','3.8':'5.2; historical environment detail in v5',
 '4.1':'4.1 and Figure 2; full step table in v5','4.2':'4.1; full cell tables in v5','4.3':'4.2 and Table 2','4.4':'4.2 and Table 2','4.5':'4.2; full architecture summaries in v5','4.6':'4.2','4.7':'4.2; full failure taxonomy in v5','4.8':'5.1 and 5.2',
 '5.1':'3.3','5.2':'3.3 restored heuristic settings','5.3':'3.3 and 3.4 restored runtime contract','5.4':'4.3 learned latency; additional heuristic benchmark in v5','5.5':'4.3 and Table 3','5.6':'3.4 and 4.3','5.7':'3.3 and 4.3 restored feature/head validation detail','5.8':'4.3, Table 3 and Figure 3','5.9':'4.3 and Figure 3; complete replay/Pareto table in v5',
 '6.1':'5.1','6.2':'3.1 and 5.1','6.3':'2.1, 2.4 and 5.1','6.4':'3.2 and 5.1','6.5':'4.3 and 5.1','6.6':'3.3, 4.3 and 5.1','6.7':'3.4 and 5.1','6.8':'5.2',
 'Appendix A':'5.2 pinned evidence map; full source/provenance catalogue in v5',
 'Appendix B':'Claim qualifications retained across Chapters 2-5; full scope table in v5',
 'Appendix C':'3.4 exact paired-interval argument',
 'Appendix D':'4.2/5.2 qualification; complete peer/selected-answer data in v5',
 'Appendix E':'2.4 Figure 1 and 3.3 runtime flow; other diagrams in v5',
}
source=(ROOT/'ThesisDocs/Masters_Thesis_Formal_v5.md').read_text(encoding='utf-8')
rows=[]
for line in source.splitlines():
 if line.startswith('## '):
  title=line[3:];key=title.split()[0];assert key in locations,key
  rows.append({'v5_section':key,'v5_title':title,'v7_or_support_location':locations[key],'preserved_support':'ThesisDocs/Masters_Thesis_Formal_v5.md'})
 elif line.startswith('# Appendix '):
  title=line[2:];key=' '.join(title.split()[:2]);assert key in locations,key
  rows.append({'v5_section':key,'v5_title':title,'v7_or_support_location':locations[key],'preserved_support':'ThesisDocs/Masters_Thesis_Formal_v5.md'})
 elif line.startswith('### Exact counterexample'):
  rows.append({'v5_section':'2.4 counterexample','v5_title':line[4:],'v7_or_support_location':'2.4 and Figure 1' if '1:' in line else 'Preserved v5 Section 2.4; supporting adaptive-information counterexample','preserved_support':'ThesisDocs/Masters_Thesis_Formal_v5.md'})
assert len(rows)==len(locations)+2
result={'schema':'v5-to-v7-scientific-section-ledger','v5_sha256':hashlib.sha256((ROOT/'ThesisDocs/Masters_Thesis_Formal_v5.md').read_bytes()).hexdigest(),'v7_sha256':hashlib.sha256((ROOT/'ThesisDocs/Masters_Thesis_Formal_v7.md').read_bytes()).hexdigest(),'sections_accounted_for':len(rows),'unmapped_scientific_sections':[],'rows':rows,'interpretation':'All supporting content remains in the preserved extended report. Mapped summaries are not assertions that every original derivation, figure or numeric table is printed in v7.'}
(OUT/'section_ledger.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
comparison=json.loads((ROOT/'ThesisDocs/verification/concise_v6_2026-10-07/jhu_thesis_lengths.json').read_text())
peer=json.loads((ROOT/'ThesisDocs/verification/content_audit_v7_2026-10-08/content_preservation.json').read_text())
checks=next(x['details'] for x in peer['checks'] if x['name']=='four thesis structures inspected')
record={'schema':'jhu-peer-structure-review-v7','as_of':'2026-10-08','scope':'Four actual MSc thesis covers/PDFs and contents sections; convenience sample, not a universal template or quality ranking.','items':[dict(item,structure_check=next(c for c in checks if c['file']==file)) for item,file in zip(comparison['items'],['byerly.pdf','galinkin.pdf','baeder.pdf','columbus.pdf'])],'common_functions':'Introductory motivation/prior work, mathematical or methodological development, empirical evaluation, and conclusion/endmatter. Topic-specific headings and depth differ.','redistribution':'No third-party PDFs or full contents text included.'}
(OUT/'peer_structure_review.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'mapped_v5_scientific_sections_and_appendices':len(rows),'unmapped':0,'peer_theses':len(record['items'])}))
