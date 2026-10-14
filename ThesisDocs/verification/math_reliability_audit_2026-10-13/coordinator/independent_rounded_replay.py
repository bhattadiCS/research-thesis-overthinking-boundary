"""Independent standard-library evaluation of retained rounded replay heads.

No research module imports, parameter fitting, generation or regrading.
Interval extrema are OUTER bounds; shared coefficients couple all decisions.
"""
from pathlib import Path
import csv, math, json, hashlib, subprocess

OUT = Path(__file__).resolve().parent
AUDIT = OUT.parent
ROOT = OUT.parents[3]
source = ROOT / 'research/outputs/real_traces_bf16_ladder/qwen2p5_7b/trace_steps.csv'
report = ROOT / 'research/outputs/real_traces_bf16_ladder/active_stopping_simulation_report.md'
historical = subprocess.check_output(['git', 'show', '5955600539f6a9a27b417872b86d3da184128935:research/run_active_stopping.py'], cwd=ROOT)
assert b'if step >= 2 and mu_t <= 0:' in historical
assert b'clf.intercept_[0]:.4f' in historical
features = ['entropy_mean', 'confidence', 'hidden_l2_shift', 'answer_changed']
weights = {}
for line in report.read_text(encoding='utf-8').splitlines():
    cells = [s.strip() for s in line.strip('|').split('|')]
    if cells[0] in ('q_t', 'alpha', 'beta'):
        weights[cells[0]] = [float(s) for s in cells[1:]]

def logistic(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)

def heads(row):
    x = [1.] + [float(row[k]) for k in features]
    radius = .00005 * math.fsum(abs(v) for v in x)
    return {key: tuple(logistic(math.fsum(v * w for v, w in zip(x, coef)) + offset)
                       for offset in (-radius, 0., radius))
            for key, coef in weights.items()}

groups = {}
with source.open(encoding='utf-8-sig', newline='') as f:
    for row in csv.DictReader(f):
        groups.setdefault(row['run_id'], []).append(row)
totals = dict(baseline_tokens=0, baseline_correct=0, midpoint_tokens=0,
              midpoint_correct=0, token_outer_lower=0, token_outer_upper=0,
              correct_outer_lower=0, correct_outer_upper=0, interval_certified=0,
              interval_ambiguous=0)
for rows in groups.values():
    rows.sort(key=lambda r: int(r['step']))
    assert [int(r['step']) for r in rows] == list(range(1, 11))
    accumulated = []
    total = 0
    drifts = []
    labels = []
    for row in rows:
        total += int(row['raw_generation_tokens'])
        accumulated.append(total)
        labels.append(int(row['correct']))
        h = heads(row)
        ql, q, qu = h['q_t']; al, a, au = h['alpha']; bl, b, bu = h['beta']
        drifts.append(((1-qu)*al-qu*bu-.05,
                       (1-q)*a-q*b-.05,
                       (1-ql)*au-ql*bl-.05))
    point = next((i for i in range(1, 9) if drifts[i][1] <= 0), 9)
    possible = []
    for i in range(1, 10):
        if i == 9:
            possible.append(i)
            break
        if drifts[i][0] <= 0:
            possible.append(i)
        if drifts[i][2] <= 0:
            break
    totals['baseline_tokens'] += total
    totals['baseline_correct'] += labels[-1]
    totals['midpoint_tokens'] += accumulated[point]
    totals['midpoint_correct'] += labels[point]
    for metric, values in [('token', accumulated), ('correct', labels)]:
        totals[metric+'_outer_lower'] += min(values[i] for i in possible)
        totals[metric+'_outer_upper'] += max(values[i] for i in possible)
    totals['interval_certified' if len(possible) == 1 else 'interval_ambiguous'] += 1

peer = json.loads((AUDIT/'agent_c/rounded_replay_bounds.json').read_text())
expected = dict(baseline_tokens=827804, baseline_correct=1058,
                midpoint_tokens=377774, midpoint_correct=963,
                token_outer_lower=374891, token_outer_upper=380891,
                correct_outer_lower=958, correct_outer_upper=964,
                interval_certified=1403, interval_ambiguous=97)
assert totals == expected, totals
assert peer['rounded_coefficients']['tokens'] == totals['midpoint_tokens']
outer = peer['conservative_outer_aggregate_bounds']
assert outer['token_total_lower'] == totals['token_outer_lower']
result = dict(trajectories=len(groups), rows=sum(map(len, groups.values())),
              totals=totals, independent_agreement=True,
              assumptions='Reported coefficients are nearest-four-decimal versions of applied heads; historical raw unscaled features and step-two floor define the replay.',
              derivation='Triangle inequality bounds each logit by 0.00005*sum(abs(augmented features)); sigmoid is increasing. Drift decreases in q and beta, increases in alpha on [0,1]^3. Enumerate all interval-compatible first crossings, with forced horizon.',
              limits='Outer bounds can include jointly unattainable extrema because coefficients are shared across heads/rows/times. An interval-ambiguous decision is not proof of an actually different historical decision. No exact historical stopped total certified; no contradiction shown.',
              source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
              historical_source_sha256=hashlib.sha256(historical).hexdigest())
(OUT/'independent_rounded_replay.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result, indent=2))
