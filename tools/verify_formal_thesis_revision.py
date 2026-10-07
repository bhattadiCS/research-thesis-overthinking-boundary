"""Compare the concise thesis with its exact v4 publication baseline."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'ThesisDocs/formal/source_snapshots/v4_publication_baseline'
OUT = ROOT / 'ThesisDocs/verification/editorial_revision_v5_2026-10-06'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def maths(source):
    # PowerShell dollar variables in reproduction commands are code, not TeX.
    source = re.sub(r'```[^\n]*\n.*?```|`[^`\n]+`', '', source, flags=re.S)
    return [(m[1] is not None, re.sub(r'\s+', ' ', m[1] if m[1] is not None else m[2]).strip())
            for m in re.finditer(r'\$\$(.+?)\$\$|(?<!\\)\$([^$]+?)(?<!\\)\$', source, re.S)]

def scientific_tables(source):
    result = {}
    for m in re.finditer(r'^\*\*Table (\d+)\.[^\n]+\n\n((?:\|[^\n]*\n)+)', source, re.M):
        result[int(m[1])] = m[2].strip()
    return result

old = (ROOT / 'ThesisDocs/Masters_Thesis_Formal_v4.md').read_text(encoding='utf-8')
new = (ROOT / 'ThesisDocs/Masters_Thesis_Formal_v5.md').read_text(encoding='utf-8')
old_tables, new_tables = scientific_tables(old), scientific_tables(new)
assert set(old_tables) == set(range(1, 19)), sorted(old_tables)
assert set(new_tables) == set(range(1, 18)), sorted(new_tables)
table_checks = []
for number in range(1, 17):
    assert old_tables[number] == new_tables[number], f'Scientific table {number} changed'
    table_checks.append({'v4_table': number, 'v5_table': number, 'body_equal': True})
assert old_tables[18] == new_tables[17], 'Additional prediction table changed'
table_checks.append({'v4_table': 18, 'v5_table': 17, 'body_equal': True})
supp = ROOT / 'ThesisDocs/formal/supplements/v5/reproduction_and_history.txt'
assert old_tables[17] in supp.read_text(encoding='utf-8'), 'Dated status table not retained in supplement'
theory_old = (BASE / 'ThesisDocs/chapters/chapter2_theory.md').read_text(encoding='utf-8')
theory_new = (ROOT / 'ThesisDocs/chapters/chapter2_theory.md').read_text(encoding='utf-8')
assert theory_new == theory_old.replace('Appendix F illustrates', 'Appendix E illustrates'), 'Theory changed beyond appendix link'
expr_checks = []
for path in ['ThesisDocs/chapters/chapter2_theory.md', 'ThesisDocs/chapters/chapter3_methodology.md', 'ThesisDocs/chapters/chapter5_online.md', 'ThesisDocs/appendices.md']:
    first = maths((BASE / path).read_text(encoding='utf-8'))
    second = maths((ROOT / path).read_text(encoding='utf-8'))
    assert first == second, f'Equation sequence changed: {path}'
    expr_checks.append({'path': path, 'expression_count': len(first), 'ordered_tex_equal': True})
refs = ROOT / 'ThesisDocs/references.md'
assert refs.read_bytes() == (BASE / 'ThesisDocs/references.md').read_bytes()
reference_numbers = set(re.findall(r'^\[(\d+)\]', new, re.M))
assert reference_numbers == set(map(str, range(1, 26))), reference_numbers
for filename in ['chapter1_intro.md', 'chapter3_methodology.md', 'chapter4_empirical.md', 'chapter5_online.md', 'chapter6_discussion.md']:
    path = 'ThesisDocs/chapters/' + filename
    # Bibliographic keys survive the edit even if a repeated occurrence is removed.
    old_keys = set(re.findall(r'\[([A-Z][A-Za-z0-9]+)\](?!\()', (BASE / path).read_text(encoding='utf-8')))
    new_keys = set(re.findall(r'\[([A-Z][A-Za-z0-9]+)\](?!\()', (ROOT / path).read_text(encoding='utf-8')))
    assert old_keys <= new_keys, (path, sorted(old_keys-new_keys))
protected = json.loads((BASE / 'protected_artifacts.json').read_text())
assert all(sha(ROOT / row['path']) == row['sha256'] for row in protected)
words = lambda s: len(re.findall(r"\b[\w'-]+\b", s))
result = {
    'schema': 'editorial-semantic-preservation-v5', 'status': 'verified',
    'baseline_commit': json.loads((BASE / 'manifest.json').read_text())['baseline_commit'],
    'v4_words': words(old), 'v5_words': words(new), 'reduction_words': words(old)-words(new),
    'reduction_percent': 100*(words(old)-words(new))/words(old),
    'scientific_table_data': table_checks, 'status_table_preserved_in_electronic_supplement': True,
    'ordered_math_expressions': expr_checks, 'chapter2_only_change': 'Appendix F cross-reference becomes Appendix E',
    'references_byte_equal': True, 'all_25_bibliography_entries_retained': True,
    'protected_historical_files': len(protected), 'protected_historical_hash_mismatches': [],
    'scope': 'Editorial/source equality checks complement human semantic and visual review; no new research or model generation.',
    'source_hashes': {path: sha(ROOT / path) for path in ['ThesisDocs/Masters_Thesis_Formal_v5.md', 'ThesisDocs/appendices.md', 'ThesisDocs/references.md', 'tools/build_master_thesis.py'] + ['ThesisDocs/chapters/' + name for name in ['chapter1_intro.md', 'chapter2_theory.md', 'chapter3_methodology.md', 'chapter4_empirical.md', 'chapter5_online.md', 'chapter6_discussion.md']]},
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'semantic_preservation.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
print(json.dumps({k: result[k] for k in ['status', 'v4_words', 'v5_words', 'reduction_words', 'reduction_percent', 'protected_historical_files']}, indent=2))
