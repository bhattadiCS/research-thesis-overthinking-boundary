from pathlib import Path
import json,collections
A=Path(__file__).resolve().parents[1]
u=json.loads((A/'claim_ledger.json').read_text(encoding='utf-8'))
for x in json.loads((A/'agent_c/ledger.json').read_text(encoding='utf-8')):print(x['id'],x['title'])
print('UNRESOLVED TABLE CELLS')
for x in u:
    if x['outcome']=='NOT ESTABLISHED' and x['kind']=='table_cell':print(x['id'],x['line'],x['claim'])
print('UNRESOLVED SCIENTIFIC PARAGRAPHS')
for x in u:
    if x['outcome']=='NOT ESTABLISHED' and x['kind'] in ['paragraph','caption','figure'] and x['scope']!='ancillary_metadata_or_roadmap':print(x['id'],x['line'],x['claim'].replace('\n',' ')[:500])
