"""Map B evidence to every coordinator V5 inventory unit, conservatively."""
import json, hashlib
from pathlib import Path
from collections import Counter
OUT=Path(__file__).resolve().parent
AUDIT=OUT.parent
ROOT=OUT.parents[3]
V='VERIFIED UNDER RECORDED ASSUMPTIONS'
N='NOT ESTABLISHED'
original=json.loads((AUDIT/'claim_inventory_initial.json').read_text(encoding='utf-8'))
own=json.loads((OUT/'claims.json').read_text(encoding='utf-8'))
groups=[x for x in own if x['kind']=='consequential_claim']
compiled='ThesisDocs/Masters_Thesis_Formal_v5.md'

# Exact all-assertion analytical units. Mixed empirical paragraphs are retained
# below as pending, even where a mathematical subclaim has been verified.
deductive={
31:('D17',['B-R01'],'Accuracy and cost utility may order policies differently.'),
35:('D4',['B-CH2-10'],'Binary transition definitions.'),
37:('D16',['B-R01'],'Comparator-relative regret definition and possible forms.'),
41:('D9',['B-CH2-18'],'Population curve does not identify per-prefix optimal action.'),
53:('D2',['B-CH2-05','B-CH2-12','B-CH2-15','B-CH2-23'],'Chapter 2 theoretical contribution/scope, directly checked in full.'),
85:('D14',['B-CH2-25','B-CH2-27'],'Sequential coverage differs from marginal calibration and ordinary pointwise bootstrap intervals.'),
767:('D5',['B-CH2-04','B-CH2-11'],'Sunk versus incremental cost.'),
783:('D10',['B-CH2-20'],'Nested outer-fold fitting requirement; historical numerical identifier is a cross-reference only.'),
785:('D17',[],'Question and cell bootstraps have different conditional descriptive units; neither replicates generation seeds.'),
791:('D17',[],'Declared step/token utility definitions, 250-token conversion, common cost cancellation.'),
793:('D17',[],'Ratio of totals and its baseline-token weighting; total-computation accounting.'),
810:('D4',['B-CH2-10'],'Exact same-panel label transition identity.'),
833:('D8',['B-CH2-15','B-CH2-18'],'Mean/final crossings do not prove optimal causal stopping.'),
994:('D17',[],'Paired endpoint and completion-token ratio definitions; energy and time are distinct.'),
1032:('D18',[],'Paired CP/union-bound method with explicit iid-reference limitation.'),
1038:('D17',['B-CH2-08','B-CH2-11'],'Two-head fitted drift formula is the algebraic one-step target under common conditioning/selector.'),
1121:('D1',['B-CH2-06'],'Replay validity conditional on potential-prefix consistency; replay cannot measure live overhead.'),
1123:('D17',[],'Pareto dominance definition and scalarization/constraint requirement for choice.'),
1133:('D17',[],'Panel-specific empirical Pareto set is not a population guarantee.'),
1140:('D9',['B-CH2-08','B-CH2-12','B-CH2-18'],'Binary transitions and multi-step continuation differ from a population crossing.'),
1142:('D17',[],'Fixed budgets can save computation without correctness recognition; detector costs belong in objective.'),
1146:('D1',['B-CH2-01'],'Binary grader agreement does not imply mathematical understanding; label errors change transition counts.'),
1148:('D16',['B-CH2-30'],'Finite parser tests do not universally adjudicate labels; domain-sensitive normalization caveat.'),
1154:('D11',['B-CH2-07','B-CH2-22'],'Conditional estimands and reweighted probability target.'),
1156:('D16',['B-CH2-12','B-CH2-30'],'Conditional theorem and finite checks do not certify deployment law.'),
1158:('D14',['B-CH2-25','B-CH2-27'],'Task-pair comparison intervals do not supply trajectory-level sequential coverage.'),
1164:('D10',['B-CH2-20'],'Repeated configuration selection compromises untouched evaluation status.'),
1172:('D17',[],'Completion tokens cover one computation component; time/energy need measurement.'),
1174:('D1',['B-CH2-04'],'Executed peers incur costs and can exceed a single-generator comparison.'),
1180:('D9',['B-CH2-17','B-CH2-18'],'Stable wrong answers and delayed repairs are possible counterexamples to heuristic intuitions.'),
1188:('D18',[],'Prespecified margin and paired uncertainty required for the stated noninferiority claim; equality insufficient.'),
1190:('D10',['B-CH2-19','B-CH2-20'],'Correctness scalar need not summarize future law; full history avoids compression premise.'),
1294:('D18',[],'Discordance definitions and accuracy difference.'),
1296:('D18',[],'Marginal binomial CP intervals plus union bound; count independence unnecessary.'),
1298:('D18',[],'Subtraction of simultaneous limits.'),
1300:('D18',[],'Zero-discordance CP endpoints and finite uncertainty; existing finite enumeration run confirms checked cases.'),
1302:('D18',[],'Handpicked-bank reference interpretation and descriptive paired resampling.'),
}
mixed={
765:('D16','Floor restricts the policy class; data claim about step-one-only cases still needs C.'),
795:('D11','AUC/Brier definitions and absence of conditional guarantees verified; actual evaluated summaries remain C.'),
819:('D17','Table count arithmetic yields net +.05128205; raw transition panel and denominators require C.'),
821:('D17','Table count arithmetic yields gain .03733333 and net -.01266667; raw rows require C.'),
823:('D17','Task-cluster interval interpretation conditional on design; actual 500/19500 counts and bootstrap require C.'),
855:('D17','593.55/75965=.0078134667; exact matched artifact contrast requires C.'),
861:('D17','No paired discordances makes the resampling difference identically zero; actual paired identities require C.'),
863:('D17','(418-204)/1500=.14266667; raw precision labels/pairs and task interval require C.'),
883:('D17','Ranking and utility are different objectives; actual scores, paired uncertainty and model comparison require C.'),
885:('D11','Within-task AUC undefined without both classes; actual eligible task counts require C.'),
897:('D17','.955156-.943223=.011933; stored score arrays and bootstrap require C.'),
903:('D17','Positive bootstrap fraction alone is not a conventional p-value or independent generation replication; draw signs require C.'),
907:('D17','Saving arithmetic from displayed totals verifies; raw old replay/provenance require C.'),
909:('D17','Verdict counts sum to75965 and displayed percentages round correctly; raw pairs/partition require C.'),
921:('D17','Strict utility loss with nondecreasing costs implies stop incorrect/final correct; actual taxonomy/priority counts require C.'),
948:('D9','Delayed repair refutes general myopic optimum; actual heuristic decision paths require C.'),
1036:('D17','Direct next-correctness target and algebra verified; actual trained heads/selector alignment require C.'),
1040:('D17','Formula is myopic and no persistence is established; actual frozen policy parameters/terminal path require C.'),
1099:('D17','Independent saved pairs reproduce all numerical endpoints; uniform time2 with matching potential prefixes is a fixed-budget realization. Actual generation/prefix event checks require C.'),
1101:('D18','Independent saved pairs reproduce endpoints; actual generated-prefix matching requires C.'),
1103:('D17','Independent saved pairs reproduce prompt-plus-completion ratios; model times/slots and event-level totals require C.'),
1131:('D17','All learned CSV stop flags are2 and give no extra adaptation on the realized panel; actual events/figure geometry require coordinator/C.'),
1150:('D1','A different label definition changes estimand; actual regrade receipts/unchanged labels require C.'),
1162:('D17','Task holdout cannot rule out pretraining exposure; actual seed/settings evidence require C.'),
1166:('D17','Resampling/training seeds do not create new model generations; actual historical analyses require C.'),
1176:('D17','Controller latency does not imply total response-time equivalence; actual timed scope requires C.'),
1182:('D19','All twenty exact trap keys independently solved and verified; software/generation evidence requires C.'),
1184:('D1','Causal earlier-answer selection changes endpoint/action space; actual retention firing requires C.'),
1196:('D17','Theoretical scope and live-pair arithmetic verified; mixed concluding empirical/code claims await C.'),
}
live_table_lines=set(range(978,991))|set(range(1010,1023))|set(range(1065,1078))|set(range(1083,1096))
coverage=[]
for c in original:
    if c['version']!='v5':continue
    line=c['line']; kind=c['kind']; outcome=N; evid=[]; bids=[]; partial=[]
    assumptions='No favorable empirical inference from a prior audit or a displayed number.'
    missing='Agent C/coordinator source-row, executed-source, metadata, figure or bibliography reconciliation as applicable; independent semantic labels and population transfer remain distinct.'
    scope='empirical_or_other_unresolved'
    for g in groups:
        if g['line_start']+93<=line<=g['line_end']+93:
            outcome=V; bids=[g['id']];evid=['agent_b/review.md#'+g['evidence'].split('#')[-1]]
            assumptions=g['assumptions'];missing='Empirical satisfaction of theorem premises is not asserted.';scope='deductive_or_definition';break
    # Actual runtime-interface claims in this mixed paragraph are not proven by
    # mathematical filtration definitions.
    if line==111 and kind=='paragraph':
        outcome=N;partial=['D1: information model and causal measurability are valid assumptions.'];missing='Actual runtime interface/source conformance requires Agent C.';scope='mixed'
    if line in deductive:
        d,ids,claim=deductive[line];outcome=V;evid=['agent_b/review.md#'+d];bids+=ids
        assumptions=claim+' See explicit assumptions in the cited derivation.';missing='No empirical deployment or sampling-premise establishment is inferred.';scope='deductive_or_definition'
    if line in mixed:
        d,claim=mixed[line];outcome=N;evid=['agent_b/review.md#'+d];partial=[claim];scope='mixed';missing=claim
    if kind=='table_cell' and line in live_table_lines:
        outcome=V;evid=['agent_b/independent_live_arithmetic.json','agent_b/verify_numerics.py','agent_b/review.md#D18'];scope='saved_pair_numerical_identity'
        assumptions='Identity relative to exact saved binary labels, token counts, stop/shared-prefix flags; CP intervals use iid task-pair reference model and trap intervals are descriptive/reference only; bootstrap is 10000 draws seed804.'
        missing='This does not adjudicate generated answers or verify events/prefix text; those remain C. No per-instance or adversarial-population coverage is asserted.'
    if 1277<=line<=1286 and kind=='table_cell':
        # This table states evidence requirements; quantities occurring in a
        # claim name are not reverified merely by the requirement mapping.
        outcome=N;scope='mixed';partial=['D16: evidence requirement/scope mapping is valid conditionally.'];evid=['agent_b/review.md#D16'];missing='Numerical or implementation assertion in the named claim still requires C/coordinator.'
    # Add exact own occurrence IDs. Do not select another same-line theorem's
    # result to manufacture a favorable outcome.
    kmap={'paragraph':'paragraph_or_caption','caption':'paragraph_or_caption','equation':'math_expression','table_cell':'table_cell'}
    ids=[b['id'] for b in own if b['path']==compiled and b['kind']==kmap.get(kind,'') and b['line_start']==line]
    bids=list(dict.fromkeys(bids+ids))
    coverage.append({'claim_id':c['id'],'version':'v5','path':c['path'],'line':line,'end_line':c['end_line'],'kind':kind,'outcome':outcome,'scope':scope,'b_claim_ids':bids,'evidence':evid or ['agent_b/claims.json'],'assumptions_or_interpretation':assumptions,'verified_subclaims':partial,'remaining_evidence':missing,'reviewer':'Agent B','no_blanket_empirical_pass':True})
(OUT/'coverage_map.json').write_text(json.dumps(coverage,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
summary={'mapped_v5_units':len(coverage),'outcomes':dict(Counter(c['outcome'] for c in coverage)),'v5_chapter2_units':sum(98<=c['line']<=700 for c in coverage),'v5_chapter2_unresolved':[c['claim_id'] for c in coverage if 98<=c['line']<=700 and c['outcome']==N],'complete_inventory_mapping_not_complete_verification':True}
(OUT/'coverage_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
maps={'B-XA-01':['A-014','A-015'],'B-XA-02':['A-017','A-018','A-019','A-020','A-021'],'B-XA-03':['A-028','A-029','A-030'],'B-XA-04':['A-033','A-034','A-035','A-036'],'B-XA-05':['A-038','A-039'],'B-XA-06':['A-051','A-052','A-053','A-054']}
(OUT/'cross_review_a_map.json').write_text(json.dumps([{'b_cross_id':b,'a_claim_ids':a,'outcome':V,'evidence':'agent_b/cross_review_a.md#'+b,'assumptions':'Recorded concise-v7 finite-horizon binary reward model; specific theorem/pair-sampling premise in the cited derivation.'} for b,a in maps.items()],indent=2)+'\n')
print(json.dumps(summary))
