"""Read-only v5 inventory. All outputs stay beside this script."""
from pathlib import Path
import csv, hashlib, json, re
from fractions import Fraction as F
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
V5 = 'ThesisDocs/Masters_Thesis_Formal_v5.md'
text = (ROOT/V5).read_text(encoding='utf-8')
lines = text.splitlines()
VER = 'VERIFIED UNDER RECORDED ASSUMPTIONS'
NE = 'NOT ESTABLISHED'

groups = [
('B-CH2-01',5,16,'Probability space, causal candidates and binary grading endpoint','Binary measurable grading predicate; finite horizon.','D1'),
('B-CH2-02',18,35,'Decision information and deterministic-reference caveat','Nested statistical filtration; nondegenerate posterior needs latent uncertainty or coarser admissible information.','D1'),
('B-CH2-03',37,45,'Causal answer selector and finite candidate maximum','Finite nonempty observable candidate set with measurable enumeration/selection.','D1'),
('B-CH2-04',47,65,'Admissible stopping and cumulative costs','Deterministic m,N; adapted, nonnegative, nondecreasing integrable K; lambda >= 0.','D1'),
('B-CH2-05',67,76,'Lemma 1 observable reward reduction','B-CH2-01, B-CH2-04.','D2'),
('B-CH2-06',78,82,'Potential trace consistency','Stopped prefixes have the potential-trace law; changed prompt/selector/generation requires changed law.','D1'),
('B-CH2-07',86,105,'Joint transitions, conditional hazards and null-state conventions','Binary target; same conditional joint law; versions in [0,1].','D3'),
('B-CH2-08',107,126,'Theorem 2 conditional drift identity','B-CH2-05, B-CH2-07; nested filtration; integrability.','D3'),
('B-CH2-09',128,132,'Doob decomposition; q need not be martingale','B-CH2-08; finite sum of integrable drifts.','D3'),
('B-CH2-10',134,150,'Same-sample transition identity and aggregation cautions','Common nonempty transition panel; null denominators assigned arbitrary hazard.','D4'),
('B-CH2-11',152,175,'Affine stake reward/drift and cost normalization','Fixed v,p >= 0 with w=v+p>0; same target, law, filtration and K for comparisons.','D5'),
('B-CH2-12',179,221,'Theorem 3 Snell envelope, minimum supermartingale and first contact','Finite horizon; integrable adapted G; admissible stop times; earliest ties.','D6'),
('B-CH2-13',223,233,'Continuation advantage and last-transition equality','B-CH2-08, B-CH2-12; t<N.','D6'),
('B-CH2-14',235,263,'Corollary 3.1 stake monotonicity','Fixed filtration, candidates, potential law, floor/horizon and nondecreasing K; w2>=w1>0.','D7'),
('B-CH2-15',267,300,'Theorem 4 random first crossing with pathwise persistence','Conditional drift remains <=0 after random first crossing, almost surely on full potential paths.','D8'),
('B-CH2-16',302,326,'Corollary 4.1 structural drift monotonicity','Pathwise q nondecreasing, alpha nonincreasing, beta nondecreasing; constant or nondecreasing predictable cost.','D8'),
('B-CH2-17',328,341,'Exact delayed-repair counterexample','Latent reference, placeholder never equal to reference, generator learns reference at terminal time; K=t/10.','D9'),
('B-CH2-18',343,363,'Exact population-average/adaptive counterexample','Specified four-atom reference/branch law; observed branch before time-1 action.','D9'),
('B-CH2-19',367,381,'Feature-conditioned hazard identity; feature sigma fields need not be nested','Z_t measurable at t; all feature-conditioned quantities share same conditioning.','D10'),
('B-CH2-20',383,397,'State sufficiency, leakage and prefix-stable transforms','Compressed state must determine conditional future law; training labels permitted offline only.','D10'),
('B-CH2-21',401,406,'AUC invariance and marginal-calibration counterexample','Strictly increasing common score transformation; balanced population with history separating labels.','D11'),
('B-CH2-22',408,433,'Proposition 5 weighted log loss, inverse and drift-sign example','p in (0,1), positive fixed weights; unconstrained population minimization; no assertion finite fitted model reaches optimum.','D11'),
('B-CH2-23',435,474,'Proposition 6 drift perturbation and exact rectangle extrema','All true/fitted probabilities in [0,1]; deterministic error bounds or simultaneous covered intervals.','D12'),
('B-CH2-24',478,500,'Proposition 7 simultaneous upper-bound first-crossing certificate','U_t observable; probability any true mu exceeds U <= delta; floor/horizon fixed; 0<delta<1.','D13'),
('B-CH2-25',502,522,'Conditional Hoeffding and summable failure allocation','Pre-probe measurable target and count n_t>=1; conditionally independent bounded probes with common conditional mean; count fixed before probes.','D14'),
('B-CH2-26',524,537,'Probe information/cost and target-alignment limitations','Post-probe target equals pre-probe target or justified comparison; labels available; applicable concentration hypotheses.','D14'),
('B-CH2-27',539,555,'Elementary e-process, fixed mixtures and Ville bound','Adapted X_i in [-1,1]; conditional mean >=0; fixed eta in [0,1); deterministic convex mixture.','D15'),
('B-CH2-28',557,570,'Shared-sign counterexample and step-level diagnostic scope','One fair common sign versus sequential conditional-mean null; actual trace_analysis.py paths separately inspected.','D15'),
('B-CH2-29',574,584,'Table 1 assumption/interpretation mapping','Each row is a scope qualification, not an empirically established theorem premise.','D16'),
('B-CH2-30',586,607,'Oracle upper bound, floor restriction, accuracy/utility and empirical scope','Same potential trace and admissible candidate selector; positive cost for utility distinction.','D16'),
]
ledger=[]
def add(id,kind,path,start,end,claim,outcome,assumptions,evidence,dependencies='',source_mapping=''):
    ledger.append(dict(id=id,kind=kind,path=path,line_start=start,line_end=end,claim=claim,outcome=outcome,assumptions=assumptions,evidence=evidence,dependencies=dependencies,source_mapping=source_mapping))

src2='ThesisDocs/chapters/chapter2_theory.md'
for id,lo,hi,claim,assum,evid in groups:
    add(id,'consequential_claim',src2,lo,hi,claim,VER,assum,'review.md#'+evid,source_mapping=f'{V5}:{lo+93}-{hi+93}')

# Normalize exactly the builder's citation/image/dash substitutions, never run it.
refs=re.findall(r'^\[([A-Za-z0-9]+)\]',(ROOT/'ThesisDocs/references.md').read_text(encoding='utf-8'),re.M)
def norm(s):
    for i,k in enumerate(refs,1): s=s.replace(f'[{k}]',f'[{i}]')
    s=s.replace('images/thesis_v1/','images/thesis_v2/')
    return s.translate(str.maketrans({c:'-' for c in '\u2010\u2011\u2012\u2013\u2014\u2015'}))
source_files=[f'ThesisDocs/chapters/chapter{i}_{name}.md' for i,name in [(1,'intro'),(2,'theory'),(3,'methodology'),(4,'empirical'),(5,'online'),(6,'discussion')]]+['ThesisDocs/appendices.md']
source_blocks=[]
for path in source_files:
    s=(ROOT/path).read_text(encoding='utf-8')
    for match in re.finditer(r'(?m)^\S[^\n]*(?:\n(?!\s*$).*)*',s):
        value=match[0].strip(); lo=s.count('\n',0,match.start())+1
        if not value.startswith('#') and not value.startswith('[['):
            source_blocks.append((norm(value),path,lo,lo+value.count('\n')))
def mapping(value):
    v=value.strip()
    for b,p,lo,hi in source_blocks:
        if v in b:
            return f'{p}:{lo}-{hi}'
    return 'Generated evidence insert: tools/build_master_thesis.py:151-280; exact frozen dependency requires Agent C/coordinator reconciliation.'
def group_for(line):
    sl=line-93
    if 98<=line<=700:
        for id,lo,hi,*_ in groups:
            if lo<=sl<=hi: return id
    return ''

# Every TeX occurrence, including repeat mentions, is retained separately.
for i,m in enumerate(re.finditer(r'\$\$(.+?)\$\$|(?<!\\)\$([^$]+?)(?<!\\)\$',text,re.S),1):
    lo=text.count('\n',0,m.start())+1; hi=text.count('\n',0,m.end())+1
    claim=(m[1] or m[2]).strip(); group=group_for(lo)
    outcome=VER if group else NE
    evid='review.md#'+next((g[-1] for g in groups if g[0]==group),'D17')
    if lo==1038: outcome=VER; evid='review.md#D17'
    if 1290<=lo<=1302: outcome=VER; evid='review.md#D18'
    if lo==791: outcome=VER; evid='review.md#D17'
    add(f'B-M{i:03}', 'math_expression',V5,lo,hi,claim,outcome,'Inherit recorded assumptions from dependency; inline symbols are definitions/mentions, not independent theorems.',evid,group,mapping(m[0]))

# Exhaustive nonbibliographic paragraph/caption inventory; empirical and code
# assertions remain explicitly NOT ESTABLISHED by B unless analytically derived.
pid=0
theory_scope_ranges=[(5,25),(21,25)]
for m in re.finditer(r'(?m)^\S[^\n]*(?:\n(?!\s*$).*)*',text):
    val=m[0].strip(); lo=text.count('\n',0,m.start())+1; hi=lo+val.count('\n')
    if val.startswith(('#','|','![','$$')) or lo<15 or 1200<=lo<=1248: continue
    if val.startswith(('by','Aditya Bhatt','A thesis submitted','Baltimore','October','Research adviser:','Second reader:')): continue
    pid+=1; group=group_for(lo)
    outcome=VER if group else NE
    note='B independently derived the analytical claim. No assertion that its empirical premise is satisfied.' if group else 'Empirical/source/implementation claim or mixed paragraph: raw source-row/code verification requires Agent C/coordinator; inventory does not assert proof.'
    # Appendix C consists of an independently derived conditional method.
    if 1290<=lo<=1302: outcome=VER; note='D18 conditional paired-binomial derivation; actual sampling/population validity remains explicitly limited.'
    add(f'B-P{pid:03}','paragraph_or_caption',V5,lo,hi,val,outcome,note,'review.md',group,mapping(val))

# Every cell of every table, including repeated quantities and nonnumeric scope
# rows, is separately retained. Header/format rows are inventory metadata only.
tid=0; table=''; headers=[]
for ln,line in enumerate(lines,1):
    mt=re.match(r'\*\*Table (\d+)\.',line)
    if mt: table=mt[1]; headers=[]
    if not line.startswith('|'): continue
    cells=[s.strip() for s in line.strip().strip('|').split('|')]
    if all(re.fullmatch(r':?-+:?',c) for c in cells): continue
    if not headers: headers=cells; continue
    for col,value in enumerate(cells,1):
        tid+=1
        theory_table=table in ('1','16')
        add(f'B-T{tid:03}','table_cell',V5,ln,ln,f'Table {table}; row {cells[0]}; {headers[col-1] if col<=len(headers) else col}: {value}',VER if theory_table else NE,'Scope mapping' if theory_table else 'Frozen source rows, exact denominator, row-level arithmetic and uncertainty design require C/coordinator reconciliation.','review.md#D16' if theory_table else 'review.md#D17',group_for(ln),mapping(line))

# External supporting label mathematics is independently reconstructed.
taskfile='research/adversarial_tasks_v1.jsonl'; goldfile='research/adversarial_gold_v1.jsonl'
tasks=[json.loads(s) for s in (ROOT/taskfile).read_text().splitlines()]
gold=[json.loads(s) for s in (ROOT/goldfile).read_text().splitlines()]
answers=[F(5,100),F(5),F(47),F(120)/(F(60,30)+F(60,60)),F(100)*F(120,100)*F(80,100),F(20,100)*100,F(9,10)*110,F(1,11),F(10*9,2),F(2),F(7),F(-9),F(9),F(3,5)*F(2,4),1/(F(1,3)+F(1,6)),F(43-17+1),F(10,40)*100,F(100%7),F(2*10+3*20,5),F(120+15)]
traps=[]
for i,(t,g,a) in enumerate(zip(tasks,gold,answers),1):
    assert t['task_id']==g['task_id'] and a==F(g['expected_answer'])
    traps.append({'task_id':t['task_id'],'answer':str(a),'derivation':g['derivation'],'verified_from_prompt':True})
    add(f'B-G{i:02}','supporting_exact_reference',goldfile,i,i,g['task_id']+': '+g['derivation'],VER,'Interpret each exact public prompt literally; machines and pumps constant independent rates, equal trip distances, fair independent dice, specified sampling without replacement.','review.md#D19',source_mapping=f'{taskfile}:{i}; {V5}:1024')

add('B-R01','scope_absence',V5,37,37,'Regret is defined relative to a comparator; no quantitative regret bound occurs anywhere in v5 or current mathematical_foundations.md.',VER,'This is a document-scope observation; no new theorem or empirical regret guarantee is inferred.','review.md#D16')

integrity=json.loads((ROOT/'ThesisDocs/formal/source_integrity_v5.json').read_text())
checks=[]
for p,h in integrity['source_bindings'].items():
    actual=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
    checks.append(dict(path=p,expected=h,actual=actual,match=actual==h))
for p in ['output/pdf/Masters_Thesis_Formal_v5_Aditya_Bhatt.pdf','research/mathematical_foundations.md','research/tests/test_mathematical_foundations.py','research/tests/test_live_uncertainty.py','tools/analyze_live_stopping_uncertainty.py',taskfile,goldfile]:
    checks.append(dict(path=p,actual=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()))
(OUT/'hashes.json').write_text(json.dumps(checks,indent=2)+'\n')
(OUT/'trap_derivations.json').write_text(json.dumps(traps,indent=2)+'\n')
(OUT/'claims.json').write_text(json.dumps(ledger,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
with (OUT/'claims.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
summary={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'claims':len(ledger),'math_occurrences':sum(x['kind']=='math_expression' for x in ledger),'paragraphs':pid,'table_cells':tid,'source_bindings_match':all(c.get('match',True) for c in checks),'outcomes':{s:sum(x['outcome']==s for x in ledger) for s in [VER,NE]},'inventory_is_not_exhaustive_verification':True}
(OUT/'inventory_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
