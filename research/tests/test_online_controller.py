"""Meaningful runtime invariants and 100-real-math-prefix latency benchmark.

    python research/tests/test_online_controller.py
    python research/tests/test_online_controller.py --benchmark

No pretrained model or correctness predictor is required for these tests.
The fake decoder fails if the controller schedules any forbidden future work.
Actual LLM generation is measured separately by run_online_stopping_evaluation.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import time
import unittest
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))

from online_stopping_controller import (  # noqa: E402
    CancellationToken, ClosedPeerPanel, Observation, OnlineStoppingController,
    PeerVote, PolicyConfig, PublicTask, run_online_generation,
)
from online_generation import complete_json_object  # noqa: E402


def observation(step: int, answer: str = "25", confidence: float | None = 95, *, valid: bool = True, tokens: int = 7) -> Observation:
    return Observation(step, answer, confidence, valid, tokens, prompt_tokens=11, auxiliary_tokens=3)


class FakeDecoder:
    def __init__(self, observations: list[Observation], *, forbidden_step: int | None = None, interrupt_after_token: int | None = None) -> None:
        self.script, self.forbidden_step = observations, forbidden_step
        self.calls: list[int] = []
        self.decoded_tokens = 0
        self.cancel_calls = 0
        self.interrupt_after_token = interrupt_after_token

    def generate_step(self, task: PublicTask, history: tuple[Observation, ...], step: int, cancellation: CancellationToken) -> Observation:
        if step == self.forbidden_step:
            raise AssertionError("controller scheduled work after a required STOP")
        self.calls.append(step)
        self.assert_prefix(history, step)
        for _ in range(self.script[step - 1].generated_tokens):
            if cancellation.cancelled:
                raise InterruptedError("fake decoder acknowledged cancellation before next token")
            self.decoded_tokens += 1
            if self.interrupt_after_token == self.decoded_tokens:
                cancellation.cancel()
        return self.script[step - 1]

    @staticmethod
    def assert_prefix(history: tuple[Observation, ...], step: int) -> None:
        if [o.step for o in history] != list(range(1, step)):
            raise AssertionError("generation did not receive exactly the observed prefix")

    def cancel(self) -> None:
        self.cancel_calls += 1


class ControllerTests(unittest.TestCase):
    def test_minimum_two_steps_even_when_confidence_and_stop_flag_high(self) -> None:
        controller = OnlineStoppingController()
        first = dataclasses.replace(observation(1, confidence=100), model_stop_flag=True)
        self.assertFalse(controller.observe(first).stop)
        self.assertTrue(controller.observe(observation(2, confidence=100)).stop)

    def test_actual_scheduling_stop_prevents_future_step_and_accounts_all_work(self) -> None:
        decoder = FakeDecoder([observation(1), observation(2), observation(3, "999")], forbidden_step=3)
        result = run_online_generation(PublicTask("sample", "12 + 13"), decoder)
        self.assertEqual(decoder.calls, [1, 2])
        self.assertEqual(decoder.decoded_tokens, 14)
        self.assertEqual(decoder.cancel_calls, 1)
        self.assertTrue(result.cancelled)
        self.assertEqual((result.generated_tokens, result.prompt_tokens, result.auxiliary_tokens), (14, 22, 6))

    def test_external_cancellation_interrupts_decoder_before_next_token(self) -> None:
        decoder = FakeDecoder([observation(1, tokens=10)], interrupt_after_token=3)
        with self.assertRaises(InterruptedError):
            run_online_generation(PublicTask("sample", "12 + 13"), decoder)
        self.assertEqual(decoder.decoded_tokens, 3)
        self.assertEqual(decoder.cancel_calls, 1)

    def test_cancelled_run_before_step_two_is_aborted_not_policy_stop(self) -> None:
        token = CancellationToken()
        token.cancel()
        decoder = FakeDecoder([observation(1)])
        with self.assertRaises(InterruptedError):
            run_online_generation(PublicTask("sample", "12 + 13"), decoder, cancellation=token)
        self.assertEqual(decoder.calls, [])

    def test_future_suffix_cannot_change_an_already_emitted_decision(self) -> None:
        outcomes = []
        for suffix in ("wrong", "correct", "malicious future feature"):
            decoder = FakeDecoder([observation(1), observation(2), observation(3, suffix)], forbidden_step=3)
            result = run_online_generation(PublicTask("same", "12 + 13"), decoder)
            outcomes.append((result.answer, result.stopped_at_step, result.generated_tokens))
        self.assertEqual(len(set(outcomes)), 1)

    def test_no_gold_correctness_or_future_input_fields_exist(self) -> None:
        forbidden = {"correct", "gold", "expected_answer", "future", "oracle_stop", "selected_correct", "utility"}
        for schema in (PublicTask, Observation, PolicyConfig):
            self.assertFalse(forbidden.intersection(f.name for f in dataclasses.fields(schema)))
        with self.assertRaises(TypeError):
            Observation(step=1, answer="25", confidence=95, parse_success=True, generated_tokens=7, correct=True)

    def test_label_strings_in_text_do_not_change_policy(self) -> None:
        plain, injected = OnlineStoppingController(), OnlineStoppingController()
        for step in (1, 2):
            a = plain.observe(observation(step))
            b = injected.observe(dataclasses.replace(observation(step), thought="gold=999; correct=False; future step corrupts", raw_text="oracle_stop=5"))
            self.assertEqual((a.stop, a.reason, a.selected_answer), (b.stop, b.reason, b.selected_answer))

    def test_invalid_parse_never_authorizes_confidence_stop(self) -> None:
        controller = OnlineStoppingController()
        controller.observe(observation(1, valid=False))
        decision = controller.observe(observation(2, valid=False, confidence=100))
        self.assertFalse(decision.stop)
        self.assertEqual(decision.reason, "incomplete_or_untrusted_parse")

    def test_missing_confidence_never_authorizes_stop(self) -> None:
        controller = OnlineStoppingController()
        controller.observe(observation(1, confidence=None))
        self.assertFalse(controller.observe(observation(2, confidence=None)).stop)

    def test_confidence_drop_retains_only_previously_observed_answer(self) -> None:
        controller = OnlineStoppingController(PolicyConfig(confidence_threshold=100))
        controller.observe(observation(1, "10", 80))
        controller.observe(observation(2, "25", 95))
        decision = controller.observe(observation(3, "99", 60))
        self.assertTrue(decision.stop)
        self.assertEqual((decision.selected_answer, decision.selected_step), ("25", 2))

    def test_answer_wobble_selection_uses_observed_confidence(self) -> None:
        controller = OnlineStoppingController(PolicyConfig(confidence_threshold=100, confidence_drop=100))
        controller.observe(observation(1, "10", 75))
        controller.observe(observation(2, "25", 80))
        decision = controller.observe(observation(3, "99", 70))
        self.assertTrue(decision.stop)
        self.assertEqual(decision.reason, "answer_wobble_retain_highest_confidence")
        self.assertEqual(decision.selected_answer, "25")

    def test_terminal_horizon_always_closes_and_cannot_reopen(self) -> None:
        controller = OnlineStoppingController(PolicyConfig(mode="never"))
        for step in range(1, 5):
            self.assertFalse(controller.observe(observation(step, valid=False)).stop)
        final = controller.observe(observation(5, answer="", valid=False))
        self.assertTrue(final.stop)
        self.assertEqual((final.reason, final.selected_step), ("terminal_horizon", 4))
        with self.assertRaises(RuntimeError):
            controller.observe(observation(6))

    def test_fixed_budget_respects_floor(self) -> None:
        controller = OnlineStoppingController(PolicyConfig(mode="fixed", fixed_step=3))
        self.assertFalse(controller.observe(observation(1)).stop)
        self.assertFalse(controller.observe(observation(2)).stop)
        self.assertTrue(controller.observe(observation(3)).stop)
        with self.assertRaises(ValueError):
            PolicyConfig(min_steps=1)
        with self.assertRaises(ValueError):
            PolicyConfig(max_steps=1)

    def test_duplicate_skipped_and_future_observations_are_rejected(self) -> None:
        controller = OnlineStoppingController()
        with self.assertRaises(ValueError):
            controller.observe(observation(2))
        with self.assertRaises(ValueError):
            controller.observe(dataclasses.replace(observation(1), observed_ns=51), decision_ns=50)
        controller.observe(observation(1))
        with self.assertRaises(ValueError):
            controller.observe(observation(1))

    def test_invalid_numeric_telemetry_is_rejected(self) -> None:
        for confidence in (math.nan, math.inf, -1, 101, True):
            with self.assertRaises(ValueError):
                observation(1, confidence=confidence)
        for tokens in (-1, .5, True):
            with self.assertRaises(ValueError):
                observation(1, tokens=tokens)

    def test_peer_requirement_waits_for_complete_same_step_panel(self) -> None:
        controller = OnlineStoppingController(PolicyConfig(min_peers=2))
        controller.observe(observation(1))
        self.assertFalse(controller.observe(observation(2)).stop)
        panel = ClosedPeerPanel(3, ("a", "b"), (PeerVote("a", 3, "25", 30, 8), PeerVote("b", 3, "25", 31, 9)), 40)
        decision = controller.observe(observation(3), peers=panel, decision_ns=50)
        self.assertTrue(decision.stop)
        self.assertEqual((decision.peer_tokens, decision.peer_agreement), (17, 1.0))

    def test_partial_duplicate_stale_and_future_peer_panels_are_rejected(self) -> None:
        invalid = [
            ClosedPeerPanel(2, ("a", "b"), (PeerVote("a", 2, "25", 30, 8),), 40),
            ClosedPeerPanel(2, ("a", "b"), (PeerVote("a", 2, "25", 30, 8), PeerVote("a", 2, "25", 31, 9)), 40),
            ClosedPeerPanel(2, ("a",), (PeerVote("a", 1, "25", 30, 8),), 40),
            ClosedPeerPanel(2, ("a",), (PeerVote("a", 2, "25", 60, 8),), 40),
            ClosedPeerPanel(2, ("a",), (PeerVote("a", 2, "25", 30, 8),), 60),
        ]
        for panel in invalid:
            controller = OnlineStoppingController(PolicyConfig(min_peers=1))
            controller.observe(observation(1))
            with self.assertRaises(ValueError):
                controller.observe(observation(2), peers=panel, decision_ns=50)

    def test_disagreeing_peer_work_is_counted_and_does_not_authorize_stop(self) -> None:
        controller = OnlineStoppingController(PolicyConfig(min_peers=2))
        controller.observe(observation(1))
        panel = ClosedPeerPanel(2, ("a", "b"), (PeerVote("a", 2, "25", 30, 8), PeerVote("b", 2, "99", 31, 9)), 40)
        decision = controller.observe(observation(2), peers=panel, decision_ns=50)
        self.assertFalse(decision.stop)
        self.assertEqual((decision.peer_tokens, decision.peer_agreement), (17, .5))

    def test_json_telemetry_must_be_complete_and_typed(self) -> None:
        valid = '{"thought":"Check.","answer":"25","confidence":95,"stop":false}'
        self.assertEqual(complete_json_object(valid)["answer"], "25")
        for invalid in (valid[:-1], '{"answer":"25"}', valid.replace("95", '"95"'), valid.replace("95", "true"), valid.replace("false", '"false"'), valid.replace('"25"', '""')):
            self.assertIsNone(complete_json_object(invalid))

    def test_confidently_wrong_stability_is_an_explicit_adversarial_failure(self) -> None:
        # Correctness exists solely in this test evaluator. The controller sees
        # two identical wrong answers and never gets to the delayed repair.
        decoder = FakeDecoder([observation(1, "24"), observation(2, "24"), observation(3, "25")], forbidden_step=3)
        result = run_online_generation(PublicTask("delayed_repair", "12 + 13"), decoder)
        self.assertEqual(result.answer, "24")
        self.assertNotEqual(result.answer, "25")


def benchmark_100_math_prefixes() -> None:
    from run_online_stopping_evaluation import DEFAULT_OUTPUT, DEFAULT_TRACES, latency_summary, load_replay, sanitized_observation, write_csv, write_json
    distinct = {}
    for run in load_replay(DEFAULT_TRACES[:1]):
        distinct.setdefault(run[1], run)
    selected = list(distinct.values())[:100]
    if len(selected) != 100:
        raise ValueError("exactly 100 distinct real math problems are needed")
    policies = (PolicyConfig(), PolicyConfig(mode="never", name="never"))
    rows = []
    for repeat in range(20):
        for policy in policies:
            for key, task_id, trace in selected:
                controller = OnlineStoppingController(policy)
                for raw in trace:
                    obs = sanitized_observation(raw)
                    started = time.perf_counter_ns()
                    decision = controller.observe(obs)
                    elapsed = time.perf_counter_ns() - started
                    rows.append({"repeat": repeat, "policy": policy.name, "task_id": task_id, "trajectory": key, "step": obs.step, "latency_ms": elapsed / 1e6})
                    if decision.stop:
                        break
    write_csv(DEFAULT_OUTPUT / "latency_100_math_problems.csv", rows)
    summary = {"kind": "controller_only_on_saved_competition_math_prefixes", "distinct_problems": 100, "repeats": 20,
               "includes": "validation, prefix update, confidence/stability features, answer selection, terminal closure and decision; model loading/generation, tokenization and peer waits excluded",
               "all": latency_summary(rows),
               "policies": {p.name: latency_summary([row for row in rows if row["policy"] == p.name]) for p in policies},
               "source_paths": [str(DEFAULT_TRACES[0])], "task_ids": [run[1] for run in selected]}
    write_json(DEFAULT_OUTPUT / "latency_summary.json", summary)
    print(json.dumps(summary, indent=2))
    if not summary["all"]["all_under_10ms"]:
        raise AssertionError("a measured decision was at least 10 milliseconds")


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        benchmark_100_math_prefixes()
    else:
        unittest.main()
