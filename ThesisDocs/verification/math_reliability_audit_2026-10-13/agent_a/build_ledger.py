"""Serialize the independently recorded claim inventory; writes only agent_a."""
import csv
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
V='VERIFIED UNDER RECORDED ASSUMPTIONS'
N='NOT ESTABLISHED'
rows=[]

def add(ch,line,compiled,claim,deps,evidence,assumptions,central=False,outcome=V,category='valid conditional result'):
    claim_id=f'A-{len(rows)//2+1:03}'
    source=f'ThesisDocs/concise_v7/chapter{ch}'
    names={1:'_introduction.md',2:'_mathematics.md',3:'_methods.md',4:'_results.md',5:'_discussion.md'}
    source+=names[ch]
    record=dict(claim_id=claim_id,path=source,line=line,claim=claim,dependencies=deps.split(';') if deps else [],
                outcome=N if central else outcome,initial_derivation_outcome=outcome,category=category,
                evidence=evidence,assumption_conditions=assumptions,reviewer='Agent A',
                second_review='PENDING' if central else 'not required for this noncentral claim',
                duplicate_of=None,stage='independent initial review, before exchange')
    rows.append(record)
    duplicate=dict(record,claim_id=claim_id+'-D',path='ThesisDocs/Masters_Thesis_Formal_v7.md',line=compiled,
                   duplicate_of=claim_id,dependencies=[claim_id])
    rows.append(duplicate)

add(2,5,52,'Finite deterministic integer horizon, floor, probability space and increasing observation filtration define the model.','','review.md recorded assumptions','N finite integer >=0; 0<=m<=N; H_t measurable')
add(2,5,52,'Specified measurable causal answer selector is adapted to the declared filtration.','A-001','review.md recorded assumptions','a_t measurable; depends only on H_0,...,H_t')
add(2,7,54,'Binary graded correctness C_t and conditional belief q_t=E[C_t|F_t] are well defined.','A-001;A-002','review.md D1-D2','measurable binary grader/reference/candidate; conditional-expectation versions')
add(2,9,56,'Changing selector or grader changes the endpoint; comparisons fix them.','A-002;A-003','Definitional endpoint identity','fixed selector/grader for each comparison')
add(2,9,56,'Future labels/outputs are excluded from runtime inputs; completed observations enter after costs.','A-001','Model information contract, not empirical execution verification','F_t is restricted as stated; empirical implementation checked separately')
add(2,11,58,'Deterministic measurable reference from fully observed task can make C_t observable and q_t=C_t.','A-002;A-003','review.md recorded assumptions','task/reference and selector/grader F_t-measurable')
add(2,11,58,'Under that full-information model nondegenerate posterior uncertainty needs latent reference or a coarser policy filtration.','A-006','Conditional expectation of a measurable variable equals itself','claim is about declared statistical information, not computational tractability')
add(2,11,58,'A fitted prefix score alone does not establish posterior given all task semantics.','A-003;A-006','review.md D7 calibrated constant-score counterexample','no unproved calibration/transport assumption')
add(2,13,60,'Admissible decisions are F_t stopping times bounded between m and N.','A-001','review.md D1,D3,D4','declared feasible policy class')
add(2,13,60,'Nonnegative adapted nondecreasing integrable cumulative cost ensures stopped-cost integrability.','A-009','0<=K_tau<=K_N; review.md assumptions','E[K_N]<infinity')
add(2,15,62,'Objective maximizes expected selected graded reward minus cumulative executed cost.','A-003;A-009;A-010','Definition; finite integrability proof','specified endpoint and feasible costs')
add(2,17,64,'G_t=q_t-K_t and c_t=E[K_{t+1}-K_t|F_t] are integrable adapted quantities.','A-003;A-010','review.md D2-D3','t<N for incremental cost')
add(2,17,64,'Common incurred cost is sunk for conditional continuation but remains in total cost.','A-012','Subtract F_t-measurable K_t from every conditional continuation payoff','the same incurred cost across actions at this prefix')
add(2,19,66,'Lemma1: E[C_tau-K_tau]=E[G_tau] for every admissible random stop.','A-003;A-009;A-010;A-012','review.md D1','finite horizon; stopping-event measurability; integrability',central=True)
add(2,21,68,'Lemma1 proof conditions on each random stopping event then sums.','A-014','review.md D1 checks every proof step','no observability or martingale assumption on C/q',central=True)
add(2,23,70,'Full-trajectory replay requires stopping to preserve preceding-prefix law.','A-001;A-009','Potential-outcome/prefix compatibility qualification','actual equality/law correspondence must be evidenced, not assumed from deterministic sampling')
add(2,29,76,'Conditional repair/corruption hazards are probabilities in [0,1].','A-003','review.md D2 constructs ratios of conditional joint probabilities','t<N; arbitrary [0,1] versions on zero-at-risk sets')
add(2,31,78,'Zero-at-risk hazard versions do not change corresponding weighted hazard.','A-017','review.md D2: r<=1-q and s<=q','finite probability versions; no division on zero sets')
add(2,33,80,'Next-step correctness conditional mean equals (1-q)alpha+q(1-beta).','A-003;A-017;A-018','review.md D2 binary indicator decomposition','no Markov/independence/stationarity assumption',central=True)
add(2,35,82,'E[q_{t+1}|F_t]=E[C_{t+1}|F_t] by tower property.','A-001;A-003','review.md D2','nested filtrations; bounded labels',central=True)
add(2,37,84,'Exact drift: mu=(1-q)alpha-q beta-c.','A-012;A-019;A-020','review.md D2','binary reward; expected cost increment; t<N',central=True)
add(2,39,86,'Conditional empirical hazards divide events by correct at-risk denominators.','A-017','Finite-panel event-count definitions','same eligible common transition panel; denominator positive or recorded undefined convention')
add(2,39,86,'Repair joint frequency minus corruption joint frequency is accuracy change.','A-019','review.md D2 sum of C_next-C_now','common panel and identical labels/selection endpoints')
add(2,39,86,'Higher conditional corruption hazard can coexist with more repairs.','A-022;A-023','Example 3145/14818<1170/4682 but 3145>1170','different at-risk population sizes')
add(2,41,88,'Historical .05 penalty is a utility convention and not a physical monetary/energy conversion.','A-012','Definition plus absence of a conversion model','empirical convention to be checked against sources by Agent C')
add(2,41,88,'Positive accuracy change can coexist with negative net utility gain.','A-021','Any gain in (0,c) gives negative drift; review.md D7','positive cost; common endpoint')
add(2,41,88,'Population accuracy/estimated drift are distinct from conditional prefix decisions.','A-003;A-021','Conditional expectation vs its population average; D4,D5','no conditional homogeneity assumption')
add(2,47,94,'Finite Snell recursion V_N=G_N; V_t=max(G_t,E[V_{t+1}|F_t]).','A-009;A-012','review.md D3','finite horizon and integrable adapted rewards',central=True)
add(2,49,96,'Theorem1: first permitted Snell equality is optimal and attains E[V_m].','A-014;A-028','review.md D3 complete backward construction','ties stop immediately; finite termination; admissible unrestricted stopping times',central=True)
add(2,51,98,'Theorem1 proof establishes upper bound, attaining policy, measurability, termination and integrability.','A-028;A-029','review.md D3 step-by-step proof','no positivity of cost, Markov property or uniqueness needed',central=True)
add(2,53,100,'Bellman continuation uses later information and is not in general a q-only threshold.','A-028','review.md D5 two regimes with q=.5 and opposite future values','future laws/costs can depend on observed regime')
add(2,53,100,'Fitted one-step model is not the theorem multi-step conditional law.','A-021;A-028','D4-D5 delayed repair and conditional-law distinction','unless additional structure and correctness assumptions independently established')
add(2,57,104,'First nonpositive true-drift time, capped at N, is a permitted stopping time.','A-009;A-021','review.md D4','mu adapted; empty crossing set assigned infinity; deterministic floor')
add(2,59,106,'Persistence is the almost-sure condition 1{tau_mu<=t}mu_t<=0 for every t<N.','A-033','review.md D4 sign analysis','a structural assumption, not inferred from panel curves')
add(2,61,108,'Theorem2: persistent true drift implies myopic stopping optimality.','A-014;A-021;A-033;A-034','review.md D4 stopped telescoping proof','including random crossings, earlier/later competing stops, ties and empty crossing',central=True)
add(2,63,110,'Theorem2 proof: continuation strictly better before crossing; residual reward is supermartingale after crossing.','A-028;A-034;A-035','review.md D4 constructs G_t-G_{t wedge rho} and direct proof','finite conditional sampling; integrable stopped increments',central=True)
add(2,65,112,'Observed curves/fitted heads do not establish pathwise conditional persistence.','A-027;A-034','Marginal curves do not imply conditional sign restrictions','no unreported empirical persistence theorem')
add(2,65,112,'Delayed repair example: G=(0,-.10,.80), mu=(-.10,.90), myopic=0, optimal=2.','A-021;A-028;A-033','review.md D5; finite_tree_check.py exact rational checks','N=2,m=0; deterministic labels/costs',central=True)
add(2,69,118,'Different example floor establishes mathematical failure, not deployed failure frequency.','A-038','review.md D5','the deployed m=2 contract differs')
add(2,69,118,'Supporting bounds/counterexamples imply no unconditional optimality or instance safety for fitted controller.','A-035;A-038','Theorem conditions are substantive; no guarantee follows without them','supporting extended bound validity independently cross-reviewed later')
add(3,19,147,'Rows and trajectories sharing a task need not be independent; task folds retain question rows together.','','Clustered sampling logic','frozen fold/row assignments are empirical Agent C scope')
add(3,21,149,'Bootstrap draws recycle observations and do not create independent generations.','','Definition of bootstrap resampling; review.md D7','independence must come from original sampling process')
add(3,23,151,'Step utility and token utility use penalties .05(tau-1) and .0002 sum L.','A-011','Definition; sunk mandatory response cancels in paired step comparison','empirical convention correspondence remains Agent C scope')
add(3,23,151,'250 completion tokens have the same stated penalty as one response increment.','A-043','Independently computed .05/.0002=250','stated utility units; not physical equivalence')
add(3,23,151,'Ratio-of-total completion savings baseline-length weights per-question savings.','A-043','review.md D7 weighted-average identity','positive baseline completion counts/positive aggregate denominator')
add(3,23,151,'Stopped response index alone cannot establish physical savings.','A-045','Different response lengths/probe costs can yield same stop index','physical claim needs actual resource ledger')
add(3,25,153,'ROC-AUC concerns ranking; Brier concerns probability error; neither alone validates live policy.','A-008','review.md D7 monotone-score and calibration counterexamples','outcome/sampling/transport assumptions required')
add(3,31,159,'Two-response floor can exclude a first-only correct candidate; floor is a protocol choice.','A-009','Sequence C=(1,0,0) with latest-answer selector','runtime correspondence is empirical, feasible-action consequence is mathematical')
add(3,33,161,'Fitted rule stops at nonpositive p_hat_next-q_hat-.05 and is myopic rather than Bellman.','A-021;A-028;A-032','Definition plus D5 delayed repair','actual implementation checked separately; no assertion fitted heads are true means')
add(3,41,169,'Archive calibration does not establish transport to live prompts/contracts.','A-008;A-047','Marginal calibration counterexample and changed conditional law','transport needs additional assumptions/evidence')
add(3,49,177,'For iid pairs, accuracy delta=pi_I-pi_W; marginal discordance counts are binomial.','','review.md D6 indicator identity','iid pairs; binary labels; I/W within-pair independence not needed',central=True)
add(3,49,177,'Two exact marginal 97.5% CP intervals have joint coverage at least95%.','A-051','review.md D6 union bound .025+.025=.05','each marginal interval has >=.975 coverage; iid task-pair reference model',central=True)
add(3,51,179,'Subtracting simultaneous discordance bounds gives delta in [L_I-U_W,U_I-L_W].','A-051;A-052','review.md D6 deterministic interval subtraction','same simultaneous coverage event',central=True)
add(3,53,181,'Zero discordance CP upper marginal bound is 1-.0125^(1/n), nonzero at finite n.','A-052','review.md D6 exact binomial inversion; independent n=100/20 arithmetic','n>=1; equal-tail two-sided97.5% interval',central=True)
add(3,53,181,'Handpicked challenge bank iid-reference bounds do not yield randomized adversarial population coverage.','A-051;A-052','Sampling model mismatch; review.md D6','no claimed random sampling from an adversarial population')
add(4,5,188,'Displayed panel counts imply q=.2401, alpha=.21224<beta=.24989 and net gain=.0513.','A-022;A-023;A-024','Independent arithmetic: 4682/19500 and (3145-1170)/19500-.05','conditional on displayed counts; source-row reproduction belongs to Agent C')
add(4,11,196,'Hindsight last crossing need not be adapted or universally optimal.','A-027;A-038','Later signs may require future observations; D5','a deterministic prespecified time is adapted; the retrospectively selected pathwise crossing generally is not')
add(4,19,216,'Identical paired loss indicators yield zero mean contrast and degenerate empirical bootstrap.','','All paired differences zero implies every resample mean zero','conditional on claim of identical indicators; source-row truth Agent C scope')
add(4,19,216,'Displayed BF16/4bit counts give a14.27 percentage-point difference.','','Independent arithmetic:100*(418-204)/1500=14.2666667','conditional on displayed counts; source-row reproduction Agent C scope')
add(4,23,220,'Positive bootstrap lifts do not repair future-feature or nonnested-training leakage.','A-042;A-047','Resampling cannot alter score construction/information contract','actual leakage evidence Agent C scope')
add(4,27,224,'Binary utility loss under nondecreasing saved costs implies stopped incorrect/final correct.','A-003;A-010','C_stop-C_final+K_final-K_stop<0 forces C_stop-C_final=-1','same binary labels, cumulative cost nondecreasing; empirical all-loss census Agent C scope')
add(4,31,228,'Uniform response-two policy shows no demonstrated adaptation beyond fixed-two budget.','A-048;A-049','Identical selected budget under stated uniform stops','actual uniformity and choice/score overhead require Agent C evidence')
add(4,41,245,'Rows remain task dependent; terminal responses omitted from next-step target.','A-041','Repeated questions share sampling unit; no next response at horizon','actual row/censoring/target contract Agent C scope')
add(5,9,262,'Class-balanced losses can target reweighted rather than original probabilities.','','review.md D7 weighted log-loss minimizer','binary population risk; positive class weights; flexible model minimizer')
add(5,9,262,'Question-level intervals do not imply time-uniform trajectory certificates.','A-052;A-053','review.md D7 repeated-event union distinction','time-uniform claim requires valid sequential control')
add(5,13,266,'Charging thirteen peer generators can make shorter target response more expensive.','A-010;A-046','Sum of positive peer costs can exceed saved target cost','actual acquisitions/cost values empirical; statement is possibility')
add(5,17,270,'Hashes establish artifact identity rather than semantic correctness or representativeness.','','A hash comparison tests byte identity, not provenance/sampling hypotheses','cryptographic collision assumption for identity; no scientific validity inferred')

(HERE/'claim_ledger.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
with (HERE/'claim_ledger.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
    writer.writerows({k:';'.join(v) if isinstance(v,list) else v for k,v in r.items()} for r in rows)
print(json.dumps({'records':len(rows),'distinct_claims':len(rows)//2,'central_claims_pending_second_review':sum(r['second_review']=='PENDING' for r in rows)//2},indent=2))
