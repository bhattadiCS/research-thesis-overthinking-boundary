"""Backend scheduling tests with deterministic completed responses."""
from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import sys
import threading
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fakes import ScriptedGenerator, ScriptedPredictor, observation
from online_stopping_controller import CancellationToken
from demo.early_stopping.runtime import ComparisonRunner, RunConfig, RunState


class RuntimeSchedulingTests(unittest.TestCase):
    def run_fixture(self, generator=None, predictor=None, *, policy="learned", emit=None):
        generator = generator or ScriptedGenerator()
        predictor = predictor or ScriptedPredictor()
        events = []
        def collect(kind, **fields):
            events.append((kind, fields))
            if emit:
                emit(kind, **fields)
        result = ComparisonRunner(generator, predictor=predictor).run(
            RunConfig(prompt="Compute 12 + 13.", policy=policy), collect, CancellationToken(),
        )
        return result, generator, predictor, events

    def test_exact_shared_prefix_active_freezes_and_baseline_reaches_horizon(self):
        generator = ScriptedGenerator([observation(i, answer) for i, answer in enumerate(["A", "A", "B", "C", "D"], 1)])
        result, generator, predictor, events = self.run_fixture(generator=generator)
        self.assertEqual([step for step, _, _ in generator.calls], [1, 2, 3, 4, 5])
        self.assertEqual([len(prefix) for prefix, _ in predictor.calls], [2])
        self.assertIs(predictor.calls[0][0][0], generator.observations[0])
        self.assertIs(predictor.calls[0][0][1], generator.observations[1])
        self.assertGreaterEqual(generator.cancel_count, 1)
        self.assertEqual(result["early"]["answer"], "A")
        self.assertEqual(result["early"]["selected_step"], 2)
        self.assertEqual(result["early"]["stopped_at_step"], 2)
        self.assertEqual(result["baseline"]["answer"], "D")
        self.assertEqual(result["baseline"]["stopped_at_step"], 5)
        self.assertFalse(result["answers_match"])
        early_decisions = [data for kind, data in events if kind == "decision" and data["arm"] == "early"]
        self.assertEqual([data["decision"]["step"] for data in early_decisions], [1, 2])
        self.assertAlmostEqual(early_decisions[-1]["decision"]["estimated_drift"], -.15)
        endings = [data for kind, data in events if kind == "response_end"]
        self.assertEqual([data["arms"] for data in endings], [["early", "baseline"]] * 2 + [["baseline"]] * 3)
        self.assertFalse(any(kind in {"complete", "cancelled", "error"} for kind, _ in events), "manager owns terminal events")

    def test_token_accounting_uses_actual_unequal_costs_without_double_counting(self):
        result, generator, _, _ = self.run_fixture()
        self.assertEqual(result["early"]["completion_tokens"], 24)
        self.assertEqual(result["baseline"]["completion_tokens"], 131)
        self.assertEqual(result["early"]["prompt_tokens"], 205)
        self.assertEqual(result["baseline"]["prompt_tokens"], 555)
        self.assertEqual(result["completion_tokens_saved"], 107)
        self.assertAlmostEqual(result["completion_savings"], 107 / 131)
        self.assertEqual(result["physical_completion_tokens"], 131)
        self.assertEqual(result["physical_prompt_tokens"], 555)
        self.assertEqual(len(generator.calls), 5)

    def test_retaining_prior_answer_still_charges_the_completed_prefix(self):
        observations = [observation(1, "A", 80), observation(2, "B", 95), observation(3, "C", 70),
                        observation(4, "D", 95), observation(5, "E", 95)]
        result, _, predictor, _ = self.run_fixture(ScriptedGenerator(observations), policy="heuristic")
        self.assertEqual(predictor.calls, [])
        self.assertEqual(result["early"]["answer"], "B")
        self.assertEqual(result["early"]["selected_step"], 2)
        self.assertEqual(result["early"]["stopped_at_step"], 3)
        self.assertEqual(result["early"]["completion_tokens"], 53)
        self.assertEqual(result["early"]["prompt_tokens"], 314)
        self.assertEqual(result["early"]["reason"], "confidence_drop_retain_previous")

    def test_learned_latest_nonempty_answer_can_precede_the_stop(self):
        observations = [observation(1, "A"), observation(2, "", None, parsed=False)] + [observation(i, "B") for i in range(3, 6)]
        result, _, _, _ = self.run_fixture(ScriptedGenerator(observations))
        self.assertEqual((result["early"]["answer"], result["early"]["selected_step"], result["early"]["stopped_at_step"]), ("A", 1, 2))
        self.assertEqual(result["early"]["completion_tokens"], 24)

    def test_raw_model_stop_flag_does_not_override_the_controller(self):
        observations = [replace(observation(i), model_stop_flag=True) for i in range(1, 6)]
        result, generator, _, _ = self.run_fixture(ScriptedGenerator(observations), ScriptedPredictor(q=.1, next_probability=.9))
        self.assertEqual(result["early"]["stopped_at_step"], 5)
        self.assertEqual([call[0] for call in generator.calls], [1, 2, 3, 4, 5])

    def test_stream_updates_target_only_open_arms_and_callback_is_cleared(self):
        class StreamingFake(ScriptedGenerator):
            callback = None
            def set_callback(self, callback):
                self.callback = callback
            def generate_step(self, task, history, step, cancellation):
                self.callback("partial", 1)
                self.callback(f"completed {step}", self.observations[step - 1].generated_tokens)
                return super().generate_step(task, history, step, cancellation)
        generator = StreamingFake()
        _, _, _, events = self.run_fixture(generator)
        tokens = [data for kind, data in events if kind == "token"]
        self.assertEqual(len(tokens), 10)
        self.assertTrue(all(data["arms"] == ["early", "baseline"] for data in tokens if data["step"] <= 2))
        self.assertTrue(all(data["arms"] == ["baseline"] for data in tokens if data["step"] > 2))
        self.assertEqual([data["text"] for data in tokens[:2]], ["partial", "completed 1"])
        self.assertIsNone(generator.callback)

    def test_invalid_settings_are_rejected_before_generation(self):
        cases = [{"prompt": ""}, {"prompt": None}, {"prompt": " "}, {"prompt": "x" * 3001},
                 {"domain": "arc"}, {"policy": "unknown"}, {"step_cost": -.1}, {"step_cost": True},
                 {"step_cost": float("nan")}, {"step_cost": float("inf")}, {"step_cost": 1.01}]
        for settings in cases:
            generator = ScriptedGenerator()
            with self.subTest(settings=settings), self.assertRaises((TypeError, ValueError)):
                config = RunConfig(**{"prompt": "12 + 13", **settings})
                ComparisonRunner(generator, predictor=ScriptedPredictor()).run(config, lambda *args, **kwargs: None, CancellationToken())
            self.assertEqual(generator.calls, [])

    def test_horizon_is_terminal_before_next_head_inference(self):
        predictor = ScriptedPredictor(q=.1, next_probability=.9)
        _, generator, predictor, _ = self.run_fixture(predictor=predictor)
        self.assertEqual([step for step, _, _ in generator.calls], [1, 2, 3, 4, 5])
        self.assertEqual([len(prefix) for prefix, _ in predictor.calls], [2, 3, 4])

    def test_precancelled_run_schedules_no_model_work(self):
        generator = ScriptedGenerator()
        cancellation = CancellationToken()
        cancellation.cancel()
        with self.assertRaises(InterruptedError):
            ComparisonRunner(generator, predictor=ScriptedPredictor()).run(
                RunConfig(prompt="Compute 12 + 13."), lambda *args, **kwargs: None, cancellation,
            )
        self.assertEqual(generator.calls, [])

    def test_cancellation_during_baseline_generation_stops_future_scheduling(self):
        generator = ScriptedGenerator(block_at=3)
        cancellation = CancellationToken()
        failures = []
        def execute():
            try:
                ComparisonRunner(generator, predictor=ScriptedPredictor()).run(
                    RunConfig(prompt="Compute 12 + 13."), lambda *args, **kwargs: None, cancellation,
                )
            except BaseException as error:
                failures.append(error)
        worker = threading.Thread(target=execute, daemon=True)
        worker.start()
        self.assertTrue(generator.entered.wait(2), "third baseline call never entered")
        cancellation.cancel()
        worker.join(2)
        self.assertFalse(worker.is_alive(), "decoder cancellation did not unwind")
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], InterruptedError)
        self.assertEqual([step for step, _, _ in generator.calls], [1, 2, 3])
        self.assertGreaterEqual(generator.cancel_count, 1)


    def test_lifecycle_has_monotone_sequences_and_exactly_one_terminal(self):
        state = RunState(RunConfig(prompt="12 + 13"))
        state.emit("run_started")
        state.emit("decision", arm="early", decision={"step": 2, "stop": True})
        events, terminal = state.events_since(0, timeout=0)
        self.assertFalse(terminal)
        self.assertEqual([event["seq"] for event in events], [1, 2])
        state.emit("complete", summary={"marker": "final"})
        state.emit("token", text="late")
        state.emit("error", message="late error")
        events, terminal = state.events_since(2, timeout=0)
        self.assertTrue(terminal)
        self.assertEqual([event["type"] for event in events], ["complete"])
        self.assertEqual(state.snapshot()["summary"], {"marker": "final"})
        self.assertEqual(state.snapshot()["status"], "complete")

    def test_cancellation_between_steps_does_not_request_next_response(self):
        generator = ScriptedGenerator()
        cancellation = CancellationToken()
        def emit(*args, **kwargs):
            if generator.calls:
                cancellation.cancel()
        with self.assertRaises(InterruptedError):
            ComparisonRunner(generator, predictor=ScriptedPredictor()).run(
                RunConfig(prompt="Compute 12 + 13."), emit, cancellation,
            )
        self.assertEqual([step for step, _, _ in generator.calls], [1])

    def test_decoder_exception_propagates_and_cancels_backend(self):
        generator = ScriptedGenerator(fail_at=3)
        with self.assertRaisesRegex(RuntimeError, "deterministic decoder failure"):
            self.run_fixture(generator=generator)
        self.assertEqual([step for step, _, _ in generator.calls], [1, 2, 3])
        self.assertGreaterEqual(generator.cancel_count, 1)

    def test_predictor_error_does_not_generate_unapproved_future_responses(self):
        generator = ScriptedGenerator()
        with self.assertRaises(ValueError):
            self.run_fixture(generator=generator, predictor=ScriptedPredictor(q=float("nan")))
        self.assertEqual([step for step, _, _ in generator.calls], [1, 2])
        self.assertGreaterEqual(generator.cancel_count, 1)


if __name__ == "__main__":
    unittest.main()
