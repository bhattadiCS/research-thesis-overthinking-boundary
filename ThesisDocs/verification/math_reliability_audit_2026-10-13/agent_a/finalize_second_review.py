"""Preserve the initial ledger, then incorporate independently recorded B review."""
import csv
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
for ext in ('json','csv'):
    initial=HERE/f'claim_ledger_initial.{ext}'
    if not initial.exists():
        initial.write_bytes((HERE/f'claim_ledger.{ext}').read_bytes())
initial_coverage=HERE/'coverage_map_initial.json'
if not initial_coverage.exists():
    initial_coverage.write_bytes((HERE/'coverage_map.json').read_bytes())

groups={
    'B-XA-01':['A-014','A-015'],
    'B-XA-02':['A-017','A-018','A-019','A-020','A-021'],
    'B-XA-03':['A-028','A-029','A-030'],
    'B-XA-04':['A-033','A-034','A-035','A-036'],
    'B-XA-05':['A-038','A-039'],
    'B-XA-06':['A-051','A-052','A-053','A-054'],
}
lookup={claim:group for group,claims in groups.items() for claim in claims}
rows=json.loads((HERE/'claim_ledger.json').read_text(encoding='utf-8'))
for r in rows:
    base=r['claim_id'].removesuffix('-D')
    if base in lookup:
        group=lookup[base]
        r['second_review']='Agent B: agent_b/cross_review_a.md '+group
        r['second_review_outcome']='VERIFIED UNDER RECORDED ASSUMPTIONS'
        r['outcome']='VERIFIED UNDER RECORDED ASSUMPTIONS'
        r['stage']='reconciled after independently recorded first and second reviews'
        r['evidence']+='; independent second review agent_b/cross_review_a.md '+group
(HERE/'claim_ledger.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
fields=list(rows[0])
fields+=sorted(set().union(*(r.keys() for r in rows))-set(fields))
with (HERE/'claim_ledger.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    writer.writerows({k:';'.join(v) if isinstance(v,list) else v for k,v in r.items()} for r in rows)
assert not any(r['second_review']=='PENDING' for r in rows)
print(json.dumps({'second_reviewed_distinct_claims':len(lookup),'central_pending':0},indent=2))
