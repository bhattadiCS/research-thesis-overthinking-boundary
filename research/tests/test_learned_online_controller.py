"""Runtime tests for serialized prefix-head integration and real cancellation."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))

from learned_online_stopping_controller import LearnedOnlineStoppingController, LearnedPolicy
from online_generation import BatchMetrics
from online_stopping_controller import Observation, PolicyConfig, PublicTask
from run_online_stopping_evaluation import collect_policy_batch
from run_learned_online_stopping import collect_batch


class FakePredictor:
    def __init__(self, q: float = .8, p_next: float = .7, *, bad_hash: bool = False, by_domain: bool = False) -> None:
        self.q, self.p_next, self.bad_hash, self.by_domain = q, p_next, bad_hash, by_domain
        self.calls = []

    def estimate(self, prefix, domain):
        if len(prefix) == 5:
            raise AssertionError("next-step inference was requested at terminal horizon")
        self.calls.append((len(prefix), domain))
        q, p_next = (.1, .7) if self.by_domain and domain == "math" else (self.q, self.p_next)
        return SimpleNamespace(q_current=q, p_next=p_next, gain=9999, artifact_sha256=("b" if self.bad_hash else "a") * 64, feature_contract="fake-prefix-contract")


def obs(step):
    return Observation(step, "25", 95, True, 7, prompt_tokens=11)


class FakeBatchDecoder:
    def __init__(self):
        self.metrics, self.calls, self.cancelled = [], [], False

    def generate_batch(self, tasks, histories, step, cancellations):
        if any(token.cancelled for token in cancellations):
            raise AssertionError("cancelled task entered a future generation batch")
        if any([o.step for o in history] != list(range(1, step)) for history in histories):
            raise AssertionError("generator received anything beyond the prefix")
        self.calls.append((step, tuple(t.task_id for t in tasks)))
        self.metrics.append(BatchMetrics(len(tasks), len(tasks) * 11, len(tasks) * 7, len(tasks) * 11, len(tasks) * 7, .001, .0001))
        return [obs(step) for _ in tasks]

    def cancel(self):
        self.cancelled = True


class LearnedControllerTests(unittest.TestCase):
    def test_batch_collectors_abort_misaligned_results_without_scheduling_future_work(self):
        tasks = [PublicTask("one", "12+13"), PublicTask("two", "13+12")]
        for learned in (False, True):
            for difference in (-1, 1):
                class MisalignedDecoder(FakeBatchDecoder):
                    def generate_batch(self, tasks, histories, step, cancellations):
                        self.tokens = list(cancellations)
                        result = super().generate_batch(tasks, histories, step, cancellations)
                        return result[:-1] if difference < 0 else result + [obs(step)]

                generator = MisalignedDecoder()
                with self.subTest(learned=learned, difference=difference), self.assertRaises(ValueError):
                    if learned:
                        collect_batch(tasks, FakePredictor(), LearnedPolicy("a" * 64), generator, 0, [])
                    else:
                        collect_policy_batch(tasks, PolicyConfig(), generator, 0, [])
                self.assertEqual(len(generator.calls), 1)
                self.assertTrue(generator.cancelled)
                self.assertTrue(all(token.cancelled for token in generator.tokens))

    def test_batch_collectors_cancel_all_tasks_when_generation_fails(self):
        tasks = [PublicTask("one", "12+13")]
        for learned in (False, True):
            class FailingDecoder(FakeBatchDecoder):
                def generate_batch(self, tasks, histories, step, cancellations):
                    self.tokens = list(cancellations)
                    raise InterruptedError("test generation aborted")

            generator = FailingDecoder()
            with self.subTest(learned=learned), self.assertRaises(InterruptedError):
                if learned:
                    collect_batch(tasks, FakePredictor(), LearnedPolicy("a" * 64), generator, 0, [])
                else:
                    collect_policy_batch(tasks, PolicyConfig(), generator, 0, [])
            self.assertTrue(generator.cancelled)
            self.assertTrue(all(token.cancelled for token in generator.tokens))

    def test_policy_rejects_noninteger_horizons_and_overflowing_utility_scale(self):
        for name in ("min_steps", "max_steps"):
            for value in (2.5, math.nan, math.inf, True):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    LearnedPolicy("a" * 64, **{name: value})
        with self.assertRaises(ValueError):
            LearnedPolicy("a" * 64, reward_value=1e308, wrong_penalty=1e308)
        with self.assertRaises(ValueError):
            LearnedPolicy("a" * 64, reward_value=1e308, step_cost=1e308)

    def test_predictor_not_queried_before_minimum_two_steps(self):
        predictor = FakePredictor()
        controller = LearnedOnlineStoppingController(predictor, "gsm8k", LearnedPolicy("a" * 64))
        self.assertFalse(controller.observe(obs(1)).stop)
        self.assertEqual(predictor.calls, [])
        decision = controller.observe(obs(2))
        self.assertTrue(decision.stop)
        self.assertAlmostEqual(decision.estimated_drift, -.15)
        self.assertEqual(predictor.calls, [(2, "gsm8k")])

    def test_runtime_recomputes_gain_instead_of_trusting_predictor_stop_score(self):
        controller = LearnedOnlineStoppingController(FakePredictor(q=.8, p_next=.82), "gsm8k", LearnedPolicy("a" * 64))
        controller.observe(obs(1))
        decision = controller.observe(obs(2))
        self.assertTrue(decision.stop)  # supplied fake gain=9999 is ignored
        self.assertAlmostEqual(decision.estimated_drift, -.03)

    def test_terminal_horizon_precedes_next_step_head(self):
        predictor = FakePredictor(q=.1, p_next=.8)
        controller = LearnedOnlineStoppingController(predictor, "math", LearnedPolicy("a" * 64))
        for step in range(1, 5):
            self.assertFalse(controller.observe(obs(step)).stop)
        final = controller.observe(obs(5))
        self.assertTrue(final.stop)
        self.assertEqual(final.reason, "terminal_horizon")
        self.assertIsNone(final.p_next)
        self.assertEqual([step for step, domain in predictor.calls], [2, 3, 4])

    def test_stopped_task_is_removed_from_future_actual_generation_batches(self):
        generator = FakeBatchDecoder()
        predictor = FakePredictor(by_domain=True)
        tasks = [PublicTask("stop_at_2", "12+13", domain="gsm8k"), PublicTask("terminal", "12+13", domain="math")]
        batches = []
        results = collect_batch(tasks, predictor, LearnedPolicy("a" * 64), generator, 0, batches)
        self.assertEqual(generator.calls, [(1, ("stop_at_2", "terminal")), (2, ("stop_at_2", "terminal")), (3, ("terminal",)), (4, ("terminal",)), (5, ("terminal",))])
        self.assertEqual([r["stopped_at_step"] for r in results], [2, 5])
        self.assertEqual([r["generated_tokens"] for r in results], [14, 35])
        self.assertEqual([r["prompt_tokens"] for r in results], [22, 55])
        self.assertTrue(generator.cancelled)

    def test_changed_predictor_artifact_is_rejected(self):
        controller = LearnedOnlineStoppingController(FakePredictor(bad_hash=True), "math", LearnedPolicy("a" * 64))
        controller.observe(obs(1))
        with self.assertRaises(ValueError):
            controller.observe(obs(2))

    def test_nonfinite_or_out_of_range_probabilities_are_rejected(self):
        for q, p in ((math.nan, .5), (.5, math.inf), (-.1, .5), (.5, 1.1)):
            controller = LearnedOnlineStoppingController(FakePredictor(q, p), "math", LearnedPolicy("a" * 64))
            controller.observe(obs(1))
            with self.assertRaises(ValueError):
                controller.observe(obs(2))

    def test_reward_penalty_scale_and_step_cost_used(self):
        controller = LearnedOnlineStoppingController(FakePredictor(.6, .65), "math", LearnedPolicy("a" * 64, reward_value=1, wrong_penalty=2, step_cost=.1))
        controller.observe(obs(1))
        decision = controller.observe(obs(2))
        self.assertFalse(decision.stop)
        self.assertAlmostEqual(decision.estimated_drift, .05)


if __name__ == "__main__":
    unittest.main()
