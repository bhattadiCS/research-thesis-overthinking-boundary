"""Independent exact finite-tree checks; supplemental examples, not theorem proof.

No repository inputs are changed. A node reward is q-K. The q values are
realizable as conditional means of unobserved Bernoulli correctness variables.
All branches are observed, have probability 1/2, and can be used by the policy.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json

OUT = Path(__file__).resolve().parent


def policies(node, horizon):
    t, j = node
    if t == horizon:
        return [{node}]
    result = [{node}]
    for left in policies((t + 1, 2*j), horizon):
        for right in policies((t + 1, 2*j + 1), horizon):
            result.append(left | right)
    return result


def value(stops, reward, start=(0, 0)):
    if start in stops:
        return reward[start]
    t, j = start
    return (value(stops, reward, (t+1, 2*j))
            + value(stops, reward, (t+1, 2*j+1))) / 2


def check(q, cost, floor, horizon=2):
    reward = {(t,j): q[t,j]-t*cost for t,j in q}
    snell = {k:v for k,v in reward.items() if k[0] == horizon}
    drift = {}
    for t in reversed(range(horizon)):
        for j in range(2**t):
            continuation = (snell[t+1,2*j] + snell[t+1,2*j+1])/2
            snell[t,j] = max(reward[t,j], continuation)
            drift[t,j] = (reward[t+1,2*j] + reward[t+1,2*j+1])/2-reward[t,j]
    earliest, myopic = set(), set()
    persistence = True
    crossings = []
    for bits in product((0,1), repeat=horizon):
        path, j = [], 0
        for t in range(horizon+1):
            path.append((t,j))
            if t < horizon:
                j = 2*j + bits[t]
        permitted = path[floor:]
        sv = next(k for k in permitted if snell[k] == reward[k])
        sm = next((k for k in permitted if k[0] < horizon and drift[k] <= 0), path[-1])
        earliest.add(sv)
        myopic.add(sm)
        crossings.append(sm[0])
        for k in path[sm[0]:-1]:
            persistence &= drift[k] <= 0
    permitted_policies = [{(horizon,j) for j in range(2**horizon)}] if floor == horizon else None
    if permitted_policies is None:
        components = [policies((floor,j), horizon) for j in range(2**floor)]
        permitted_policies = [set().union(*parts) for parts in product(*components)]
    best = max(value(p,reward) for p in permitted_policies)
    sv_value, sm_value = value(earliest,reward), value(myopic,reward)
    expected_snell = sum(snell[floor,j] for j in range(2**floor))/2**floor
    assert best == sv_value == expected_snell
    if persistence:
        assert sm_value == best
    return persistence, len(set(crossings)) > 1, sm_value < best


def main():
    nodes = [(t,j) for t in range(3) for j in range(2**t)]
    counts = {"trees_floors_checked":0, "persistent_cases":0,
              "persistent_random_crossing_cases":0, "myopic_suboptimal_cases":0}
    for vals in product((F(0),F(1,2),F(1)), repeat=len(nodes)):
        q = dict(zip(nodes,vals))
        for cost in (F(0),F(1,10),F(1,2)):
            for floor in (0,1,2):
                persistent, random_crossing, suboptimal = check(q,cost,floor)
                counts["trees_floors_checked"] += 1
                counts["persistent_cases"] += persistent
                counts["persistent_random_crossing_cases"] += persistent and random_crossing
                counts["myopic_suboptimal_cases"] += suboptimal
    # N=0, m=N and ties terminate immediately.
    for q in (F(0),F(1,2),F(1)):
        assert max([q]) == q
    # Manuscript delayed repair, no observation uncertainty.
    delayed_rewards = [F(0),F(-1,10),F(4,5)]
    assert delayed_rewards[1]-delayed_rewards[0] == F(-1,10)
    assert delayed_rewards[2]-delayed_rewards[1] == F(9,10)
    assert max(delayed_rewards) == F(4,5)
    result = {"arithmetic":"fractions.Fraction", "horizon":2,
              "q_grid":["0","1/2","1"], "per_increment_costs":["0","1/10","1/2"],
              "floors":[0,1,2], "policy_scope":"all adapted deterministic stopping policies on each binary tree",
              "checks":counts, "snell_mismatches":0,"persistence_theorem_mismatches":0,
              "boundary_checks":["N=0", "m=N", "zero cost", "equality ties", "random crossing", "delayed repair"],
              "limitation":"Finite enumeration supplements the independent deductive proof; it does not prove the general theorem."}
    (OUT/'finite_tree_results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
