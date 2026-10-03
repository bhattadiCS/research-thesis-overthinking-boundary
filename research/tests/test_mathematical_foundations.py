#!/usr/bin/env python
"""Exact finite-system verification of mathematical_foundations.md.

These checks complement the proofs. They use no fitted research models,
external packages, Monte Carlo, or offline oracle inside a causal decision.
Bellman values are compared with independent exhaustive policy evaluation.

Run: python research/tests/test_mathematical_foundations.py
"""
from __future__ import annotations

import itertools
import unittest
from fractions import Fraction as F


ZERO, HALF, ONE = F(0), F(1, 2), F(1)
GRID = (ZERO, HALF, ONE)
NODES = ("", "0", "1", "00", "01", "10", "11")
INTERNAL = ("", "0", "1")
LEAVES = ("00", "01", "10", "11")


def drift(q: F, alpha: F, beta: F, cost: F) -> F:
    return (ONE - q) * alpha - q * beta - cost


def snell(rewards: dict[str, F], floor: int = 0) -> dict[str, F]:
    """Backward recursion on a fair binary tree, with a forced prefix."""
    values = {node: rewards[node] for node in LEAVES}
    for node in ("0", "1", ""):
        continuation = (values[node + "0"] + values[node + "1"]) / 2
        values[node] = (
            continuation if len(node) < floor
            else max(rewards[node], continuation)
        )
    return values


def policy_value(
    rewards: dict[str, F], decisions: dict[str, bool], floor: int = 0,
) -> F:
    """Forward evaluation, integrating over the four observable leaf paths."""
    result = ZERO
    for leaf in LEAVES:
        stop_node = leaf
        for depth in range(floor, 2):
            prefix = leaf[:depth]
            if decisions[prefix]:
                stop_node = prefix
                break
        result += rewards[stop_node] / 4
    return result


def exhaustive_values(rewards: dict[str, F], floor: int = 0) -> list[F]:
    return [
        policy_value(rewards, dict(zip(INTERNAL, choices)), floor)
        for choices in itertools.product((False, True), repeat=3)
    ]


def latent_answer_space() -> list[tuple[tuple[int, int, int], F]]:
    """Eight atoms: observable bits at t=1,2 and a latent binary reference."""
    success_probs = (F(1, 3), F(3, 4), F(2, 5))
    atoms = []
    for bits in itertools.product((0, 1), repeat=3):
        probability = ONE
        for bit, p in zip(bits, success_probs):
            probability *= p if bit else ONE - p
        atoms.append((bits, probability))
    return atoms


def history(bits: tuple[int, int, int], time: int) -> tuple[int, ...]:
    return bits[:time]


def correctness(bits: tuple[int, int, int], time: int) -> int:
    answer = 0 if time == 0 else bits[time - 1]
    return int(answer == bits[2])


def conditional(
    atoms: list[tuple[tuple[int, int, int], F]],
    time: int,
    key: tuple[int, ...],
    value,
) -> F:
    selected = [(bits, p) for bits, p in atoms if history(bits, time) == key]
    return sum((p * value(bits) for bits, p in selected), ZERO) / sum(
        (p for _, p in selected), ZERO,
    )


class MathematicalFoundationsTests(unittest.TestCase):
    def test_transition_count_identity_including_zero_denominators(self):
        checked = 0
        for n00, n01, n10, n11 in itertools.product(range(4), repeat=4):
            n = n00 + n01 + n10 + n11
            if not n:
                continue
            n0, n1 = n00 + n01, n10 + n11
            q = F(n1, n)
            alpha = F(n01, n0) if n0 else ZERO
            beta = F(n10, n1) if n1 else ZERO
            self.assertEqual(drift(q, alpha, beta, ZERO), F(n01 - n10, n))
            # The arbitrary hazard on a null state never changes the result.
            if not n0:
                self.assertEqual(drift(q, ONE, beta, ZERO), F(n01 - n10, n))
            if not n1:
                self.assertEqual(drift(q, alpha, ONE, ZERO), F(n01 - n10, n))
            checked += 1
        self.assertEqual(checked, 255)

    def test_conditional_hazard_identity_and_tower_on_latent_answer_space(self):
        atoms = latent_answer_space()
        self.assertEqual(sum((p for _, p in atoms), ZERO), ONE)
        posteriors = {
            (time, key): conditional(atoms, time, key, lambda b: correctness(b, time))
            for time in range(3)
            for key in itertools.product((0, 1), repeat=time)
        }
        for time in range(2):
            for key in itertools.product((0, 1), repeat=time):
                q = posteriors[time, key]
                repair = conditional(
                    atoms, time, key,
                    lambda b: int(correctness(b, time) == 0 and correctness(b, time + 1) == 1),
                )
                corruption = conditional(
                    atoms, time, key,
                    lambda b: int(correctness(b, time) == 1 and correctness(b, time + 1) == 0),
                )
                alpha = repair / (ONE - q) if q != ONE else ZERO
                beta = corruption / q if q else ZERO
                expected_posterior_change = conditional(
                    atoms, time, key,
                    lambda b: posteriors[time + 1, history(b, time + 1)] - q,
                )
                expected_correctness_change = conditional(
                    atoms, time, key,
                    lambda b: correctness(b, time + 1) - correctness(b, time),
                )
                self.assertEqual(expected_posterior_change, expected_correctness_change)
                self.assertEqual(drift(q, alpha, beta, ZERO), expected_correctness_change)

    def test_reward_reduction_and_doob_telescope_for_all_causal_policies(self):
        atoms = latent_answer_space()
        cost = F(1, 10)
        posterior = {
            (time, key): conditional(atoms, time, key, lambda b: correctness(b, time))
            for time in range(3)
            for key in itertools.product((0, 1), repeat=time)
        }
        rewards = {(t, key): q - cost * t for (t, key), q in posterior.items()}
        mus = {
            (t, key): conditional(
                atoms, t, key,
                lambda b: rewards[t + 1, history(b, t + 1)] - rewards[t, key],
            )
            for t in range(2)
            for key in itertools.product((0, 1), repeat=t)
        }
        for choices in itertools.product((False, True), repeat=3):
            decisions = dict(zip(((), (0,), (1,)), choices))
            actual, observable, telescoped = ZERO, ZERO, rewards[0, ()]
            for bits, p in atoms:
                tau = 2
                for t in range(2):
                    if decisions[history(bits, t)]:
                        tau = t
                        break
                actual += p * (correctness(bits, tau) - cost * tau)
                observable += p * rewards[tau, history(bits, tau)]
                telescoped += p * sum(
                    (mus[t, history(bits, t)] for t in range(tau)), ZERO,
                )
            self.assertEqual(actual, observable)
            self.assertEqual(observable, telescoped)
        # Verify conditional zero mean of every martingale increment directly.
        for t in range(2):
            for key in itertools.product((0, 1), repeat=t):
                self.assertEqual(
                    conditional(
                        atoms, t, key,
                        lambda b: rewards[t + 1, history(b, t + 1)]
                        - rewards[t, key] - mus[t, key],
                    ),
                    ZERO,
                )

    def test_snell_matches_exhaustive_policies_in_4374_finite_systems(self):
        checked = 0
        # General bounded adapted rewards; Snell's theorem also covers this
        # class beyond the exact-answer interpretation of the other tests.
        for probabilities in itertools.product(GRID, repeat=len(NODES)):
            q = dict(zip(NODES, probabilities))
            for cost in (ZERO, F(1, 4)):
                rewards = {node: value - cost * len(node) for node, value in q.items()}
                values = snell(rewards)
                self.assertEqual(values[""], max(exhaustive_values(rewards)))
                contact_policy = {
                    node: values[node] == rewards[node] for node in INTERNAL
                }
                self.assertEqual(policy_value(rewards, contact_policy), values[""])
                for node in INTERNAL:
                    self.assertGreaterEqual(values[node], rewards[node])
                    self.assertGreaterEqual(
                        values[node], (values[node + "0"] + values[node + "1"]) / 2,
                    )
                checked += 1
        self.assertEqual(checked, 4374)

    def test_pathwise_nonincreasing_drift_makes_sign_rule_optimal(self):
        qualifying = 0
        cost = F(1, 4)
        for probabilities in itertools.product(GRID, repeat=len(NODES)):
            q = dict(zip(NODES, probabilities))
            rewards = {node: value - cost * len(node) for node, value in q.items()}
            mus = {
                node: (rewards[node + "0"] + rewards[node + "1"]) / 2 - rewards[node]
                for node in INTERNAL
            }
            if any(mus[node] > mus[""] for node in ("0", "1")):
                continue
            sign_policy = {node: mus[node] <= ZERO for node in INTERNAL}
            self.assertEqual(policy_value(rewards, sign_policy), max(exhaustive_values(rewards)))
            qualifying += 1
        self.assertGreater(qualifying, 100)

    def test_structural_probability_monotonicity_implies_drift_monotonicity(self):
        checked = 0
        for q0, q1, a0, a1, b0, b1 in itertools.product(GRID, repeat=6):
            if q1 < q0 or a1 > a0 or b1 < b0:
                continue
            mu0 = drift(q0, a0, b0, F(1, 10))
            mu1 = drift(q1, a1, b1, F(1, 10))
            expanded = (
                (ONE - q1) * (a1 - a0) + a0 * (q0 - q1)
                - q1 * (b1 - b0) - b0 * (q1 - q0)
            )
            self.assertEqual(mu1 - mu0, expanded)
            self.assertLessEqual(mu1, mu0)
            checked += 1
        self.assertEqual(checked, 216)

    def test_delayed_repair_and_population_mean_counterexamples(self):
        cost = F(1, 10)
        delayed_q = {node: ONE if len(node) == 2 else ZERO for node in NODES}
        rewards = {node: q - cost * len(node) for node, q in delayed_q.items()}
        self.assertEqual((rewards["0"] + rewards["1"]) / 2 - rewards[""], -cost)
        self.assertEqual(snell(rewards)[""], F(4, 5))
        self.assertEqual(policy_value(rewards, {n: True for n in INTERNAL}), ZERO)

        q = {"": HALF, "0": F(2, 5), "1": F(3, 5),
             "00": ONE, "01": ONE, "10": ZERO, "11": ZERO}
        rewards = {node: value - cost * len(node) for node, value in q.items()}
        mu0 = (rewards["0"] + rewards["1"]) / 2 - rewards[""]
        mu1_mean = sum(
            ((rewards[n + "0"] + rewards[n + "1"]) / 2 - rewards[n] for n in ("0", "1")),
            ZERO,
        ) / 2
        self.assertEqual(mu0, -cost)
        self.assertEqual(mu1_mean, -cost)
        adaptive_policy = {"": False, "0": False, "1": True}
        self.assertEqual(policy_value(rewards, adaptive_policy), F(13, 20))
        self.assertEqual(snell(rewards)[""], F(13, 20))
        self.assertGreater(F(13, 20), rewards[""])
        # A common latent exact-answer reference realizes these posteriors.
        joint = {("A", 1): F(1, 5), ("A", 0): F(3, 10),
                 ("B", 1): F(3, 10), ("B", 0): F(1, 5)}
        self.assertEqual(joint["A", 1] + joint["B", 1], HALF)
        self.assertEqual(joint["A", 1] / (joint["A", 0] + joint["A", 1]), F(2, 5))
        self.assertEqual(joint["B", 1] / (joint["B", 0] + joint["B", 1]), F(3, 5))

    def test_floor_restricts_optimum_and_hindsight_is_an_upper_bound(self):
        q = {"": ONE, "0": ZERO, "1": ZERO,
             "00": ONE, "01": ZERO, "10": ZERO, "11": ONE}
        rewards = {node: value - F(1, 10) * len(node) for node, value in q.items()}
        unconstrained = max(exhaustive_values(rewards))
        for floor in (0, 1, 2):
            optimum = max(exhaustive_values(rewards, floor))
            self.assertEqual(snell(rewards, floor)[""], optimum)
            self.assertLessEqual(optimum, unconstrained)
            hindsight = sum(
                (max(rewards[leaf[:depth]] for depth in range(floor, 3)) for leaf in LEAVES),
                ZERO,
            ) / 4
            self.assertGreaterEqual(hindsight, optimum)

    def test_larger_affine_stakes_delay_earliest_optimal_stop_with_fixed_costs(self):
        for probabilities in itertools.product(GRID, repeat=len(NODES)):
            q = dict(zip(NODES, probabilities))
            stops = []
            for stakes in (HALF, ONE, F(2)):
                rewards = {
                    node: stakes * value - F(1, 4) * len(node)
                    for node, value in q.items()
                }
                values = snell(rewards)
                path_stops = {}
                for leaf in LEAVES:
                    path_stops[leaf] = 2
                    for t in range(2):
                        node = leaf[:t]
                        if rewards[node] == values[node]:
                            path_stops[leaf] = t
                            break
                stops.append(path_stops)
            for earlier, later in zip(stops, stops[1:]):
                for leaf in LEAVES:
                    self.assertLessEqual(earlier[leaf], later[leaf])

    def test_hazard_uncertainty_rectangles_and_perturbation_bound(self):
        endpoints = (ZERO, F(1, 3), F(2, 3), ONE)
        intervals = [(low, high) for low in endpoints for high in endpoints if low <= high]
        checked = 0
        for (q_l, q_u), (a_l, a_u), (b_l, b_u) in itertools.product(intervals, repeat=3):
            c_l, c_u = F(1, 20), F(1, 10)
            corners = [
                drift(q, a, b, c)
                for q, a, b, c in itertools.product(
                    (q_l, q_u), (a_l, a_u), (b_l, b_u), (c_l, c_u),
                )
            ]
            self.assertEqual(max(corners), drift(q_l, a_u, b_l, c_l))
            self.assertEqual(min(corners), drift(q_u, a_l, b_u, c_u))
            checked += 1
        self.assertEqual(checked, 1000)
        for q, a, b, q_hat, a_hat, b_hat in itertools.product(GRID, repeat=6):
            error = abs(drift(q_hat, a_hat, b_hat, F(1, 10)) - drift(q, a, b, F(1, 20)))
            bound = (
                2 * abs(q_hat - q) + (ONE - q_hat) * abs(a_hat - a)
                + q_hat * abs(b_hat - b) + F(1, 20)
            )
            self.assertLessEqual(error, bound)

    def test_class_weight_posterior_preserves_ranking_but_changes_drift_sign(self):
        w1, w0 = F(5), F(5, 9)
        probabilities = [F(i, 20) for i in range(1, 20)]
        transformed = []
        for p in probabilities:
            r = w1 * p / (w1 * p + w0 * (ONE - p))
            reconstructed = w0 * r / (w1 * (ONE - r) + w0 * r)
            self.assertEqual(p, reconstructed)
            transformed.append(r)
        self.assertEqual(sorted(transformed), transformed)
        self.assertEqual(len(set(transformed)), len(transformed))
        p = F(1, 10)
        weighted = w1 * p / (w1 * p + w0 * (ONE - p))
        self.assertEqual(weighted, HALF)
        self.assertEqual(drift(p, F(1, 50), F(1, 200), F(1, 100)), F(3, 400))
        self.assertEqual(drift(weighted, F(1, 50), F(1, 200), F(1, 100)), -F(1, 400))

    def test_e_process_has_conditional_null_and_valid_finite_crossing_bound(self):
        rates = (F(1, 4), HALF, F(3, 4))

        def mixture(path: tuple[int, ...]) -> F:
            components = []
            for eta in rates:
                product = ONE
                for x in path:
                    product *= ONE - eta * x
                components.append(product)
            return sum(components, ZERO) / len(rates)

        # Conditional positive-sign probability depends on observed history
        # and is always >= 1/2, so this is a genuinely adapted null.
        def positive_probability(prefix: tuple[int, ...]) -> F:
            return F(3, 4) if prefix and prefix[-1] == 1 else HALF

        for depth in range(7):
            for prefix in itertools.product((-1, 1), repeat=depth):
                p = positive_probability(prefix)
                conditional_next = (
                    p * mixture(prefix + (1,)) + (ONE - p) * mixture(prefix + (-1,))
                )
                self.assertLessEqual(conditional_next, mixture(prefix))
        delta = F(1, 4)
        crossed_probability = ZERO
        for path in itertools.product((-1, 1), repeat=10):
            probability = ONE
            for i, x in enumerate(path):
                p = positive_probability(path[:i])
                probability *= p if x == 1 else ONE - p
            if any(mixture(path[:i]) >= ONE / delta for i in range(11)):
                crossed_probability += probability
        self.assertLessEqual(crossed_probability, delta)
        # Bounded zero marginal means without the conditional null are
        # insufficient: one common fair sign creates perfect dependence.
        shared_negative_product = F(3, 2) ** 10
        self.assertGreater(shared_negative_product, F(20))
        self.assertGreater(HALF, F(1, 20))


if __name__ == "__main__":
    unittest.main(verbosity=2)
