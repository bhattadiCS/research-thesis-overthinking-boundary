"""Explicit reconciliation to coordinator's conservative paragraph/expression inventory."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT_AUDIT=HERE.parent
inventory=json.loads((ROOT_AUDIT/'claim_inventory_initial.json').read_text(encoding='utf-8-sig'))
ledger=json.loads((HERE/'claim_ledger.json').read_text(encoding='utf-8-sig'))
byline={}
for r in ledger:
    if r['path']=='ThesisDocs/Masters_Thesis_Formal_v7.md':
        byline.setdefault(r['line'],[]).append(r)

V='VERIFIED UNDER RECORDED ASSUMPTIONS'
N='NOT ESTABLISHED'
supplement={
29:(['A-026','A-038'],'Averaged saved compute can exceed an accuracy loss (e.g. accuracy -.1, cost saving .2 gives utility +.1); delayed repair gives regret. Questions themselves are not asserted results.'),
37:(['A-021','A-029','A-035','A-038'],'D1-D5 reconstruct the stated mathematical specialization. Contribution novelty/bibliographic comparison and empirical contribution remain coordinator/C scope.'),
39:(['A-008','A-050','A-065'],'Calibration and fixed-time intervals do not supply conditional or time-uniform guarantees. Actual fitted method is empirical scope.'),
74:(['A-017'],'The time range t<N is the hazard-definition domain.'),
92:(['A-028'],'Ferguson primary text accessed; finite recursion independently derived in D3. Book metadata is coordinator scope.'),
114:(['A-038'],'Underlying delayed-repair mathematics is derived in D5; this image has not been visually inspected by A.'),
116:(['A-038','A-039'],'Caption mathematical values, persistence failure and distinct theoretical/live floor are checked by D5.'),
190:(['A-027'],'Population averages cannot establish a per-prefix conditional action. Counts/interval construction require C.'),
194:(['A-027'],'Zero gain differs from zero accuracy; panel crossings do not determine every conditional action. Plotted bands/image require C/coordinator.'),
200:(['A-043','A-047'],'Utility per trajectory is not accuracy percentage points, and ranking/calibration improvement is not an optimality theorem. Actual contrast signs require C.'),
218:(['A-047'],'Ranking summaries alone cannot establish unique model optimality or calibrated runtime probabilities; AUC inputs/numbers require C.'),
230:(['A-051','A-052','A-053','A-055'],'Conditional mathematical coverage is valid; table sampling/token/pair provenance requires C.'),
239:(['A-016','A-055'],'Prefix disagreement limits literal same-prefix continuation interpretation; fixed traps lack randomized adversarial-population coverage. Counts require C.'),
241:(['A-045','A-046','A-066'],'Completion savings do not establish system savings. Counts/timing require C.'),
243:([],'Conditional displayed arithmetic .9369<10 is correct; latency measurement and component inclusion require C.'),
249:(['A-062'],'Uniform two-response action provides no exhibited adaptive allocation over fixed two. Uniformity/figure values require C/coordinator.'),
251:(['A-062'],'Nearly constant stop budgets provide little exhibited adaptation; all percentages/intervals/causal executable provenance require C.'),
258:(['A-021','A-029','A-035','A-047','A-062'],'Binary transitions and conditional stopping argument verified; controlled/live empirical conclusions require C.'),
260:(['A-003','A-004'],'Binary grader endpoint does not imply rationale validity or universal semantic accuracy. Finite regression cases do not prove all labels.'),
264:(['A-041','A-055'],'Task holdout cannot logically exclude pretraining overlap; repeated data-dependent selection can invalidate confirmation. Actual panel/seed/design facts require C.'),
266:(['A-046','A-066'],'Total executed-resource cost and possible peer-cost excess are conditional deductions; archived fleet evidence claims require C.'),
270:(['A-067'],'Hash identity is distinct from scientific provenance/labels/sampling. Actual manifest/source coverage requires coordinator/C.'),
272:(['A-029','A-035','A-038','A-062'],'Conditional mathematics verified; broad empirical implementation/conclusion premises require C.'),
}
# These paragraph units mix independently checked deductions with empirical assertions.
mixed={37,39,70,88,118,147,149,153,159,161,169,181,188,190,194,196,200,216,218,220,224,228,230,239,241,243,245,249,251,258,264,266,270,272}
rows=[]
for item in inventory:
    if item['version']!='v7':
        continue
    line=item['line']
    if not (52<=line<=118 or line in byline or line in supplement):
        continue
    mapped=byline.get(line,[])
    ids=[r['claim_id'].removesuffix('-D') for r in mapped]
    note='Exact compiled-line mapping to independently recorded source claim(s).'
    if line in supplement:
        extra,note=supplement[line]
        ids=list(dict.fromkeys(ids+extra))
    # If a whole paragraph contains data claims, retain unresolved full-unit status.
    empirical_pending=(line in mixed and item['kind'] not in ('equation',))
    visual_pending=(line==114 and item['kind']=='figure')
    pending_second=any(r['second_review']=='PENDING' for r in mapped)
    outcome=N if empirical_pending or visual_pending or pending_second else V
    rows.append(dict(inventory_id=item['id'],path=item['path'],line=line,kind=item['kind'],
                     outcome=outcome,deductive_outcome=V if not visual_pending else N,
                     agent_a_claim_ids=ids,reviewer='Agent A',
                     evidence='agent_a/review.md D1-D7; agent_a/cross_review_b.md; '+note,
                     remaining_evidence=('Rendered image correspondence remains coordinator scope.' if visual_pending else
                         'Whole paragraph includes empirical/implementation/bibliographic premises pending C/coordinator; favorable outcome applies only to explicitly mapped mathematical deductions.' if empirical_pending else
                         'Required independent central second review pending Agent B.' if pending_second else ''),
                     second_review_pending=pending_second))

core=[r for r in inventory if r['version']=='v7' and 48<=r['line']<=118]
covered={r['inventory_id'] for r in rows}
assert all(r['id'] in covered for r in core), [r['id'] for r in core if r['id'] not in covered]
(HERE/'coverage_map.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')

cross=[
('AXB1',['B-CH2-11','B-CH2-14'],'ThesisDocs/chapters/chapter2_theory.md',152,263,'Affine stakes and monotonic earliest optimal stops','w>0; common law/filtration/candidates/cost/floor/horizon; nondecreasing costs'),
('AXB2',['B-CH2-22'],'ThesisDocs/chapters/chapter2_theory.md',408,433,'Weighted population log-loss optimum and drift-flip example','p in (0,1); positive weights; unrestricted probability prediction; not a finite-fit guarantee'),
('AXB3',['B-CH2-23'],'ThesisDocs/chapters/chapter2_theory.md',435,474,'Drift perturbation inequality and exact rectangle extrema','probabilities in [0,1]; nonnegative error tolerances; simultaneous intervals if coverage interpreted'),
('AXB4',['B-CH2-24'],'ThesisDocs/chapters/chapter2_theory.md',478,500,'First-upper-bound-crossing one-step sign guarantee','observable bounds; simultaneous coverage; finite horizon; 0<delta<1'),
('AXB5',['B-CH2-25','B-CH2-26'],'ThesisDocs/chapters/chapter2_theory.md',502,537,'Conditional Hoeffding, summable failure budget and pre/post-probe distinction','positive count fixed given pre-probe information; conditional independence/boundedness/common mean; correct target'),
('AXB6',['B-CH2-27','B-CH2-28'],'ThesisDocs/chapters/chapter2_theory.md',539,570,'Product e-process, fixed mixtures, supremum bound and dependence witness','adapted bounded observations; conditional-mean null; fixed eta and fixed mixture'),
('AXB7',['B-M246','B-M247','B-M248','B-M249','B-M250','B-M251','B-M252','B-M253','B-M254','B-M255','B-M256','B-M257','B-P243','B-P244','B-P245','B-P246'],'ThesisDocs/appendices.md',31,39,'Paired exact binomial/union-bound interval and zero-discordance endpoint','iid binary task pairs; exact marginal CP intervals; no within-pair independence needed'),
]
crossrows=[dict(cross_review_id=i,agent_b_claim_ids=b,path=p,line_start=l,line_end=u,claim=c,
                outcome=V,evidence='agent_a/cross_review_b.md '+i,assumption_conditions=a,
                reviewer='Agent A',reconciliation='Independent derivations agree with B; no manuscript correction justified.')
           for i,b,p,l,u,c,a in cross]
(HERE/'cross_review_b_map.json').write_text(json.dumps(crossrows,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'coverage_units':len(rows),'v7_ch2_units':len(core),'cross_reviewed_bound_groups':len(crossrows)},indent=2))
