"""Regression tests for fair terminal selection and honest missing costs."""

from __future__ import annotations

import json
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_online_stopping_evaluation import paired_metrics, replay, sanitized_observation


class GuardedRow(dict):
    """An observable adapter must never touch labels or future/oracle fields."""

    forbidden = {"correct", "expected_answer", "utility", "oracle_stop", "future_correct"}

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError(f"runtime adapter accessed forbidden field {key}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.forbidden:
            raise AssertionError(f"runtime adapter accessed forbidden field {key}")
        return super().get(key, default)


class ReplayAccountingTests(unittest.TestCase):
    def test_never_baseline_uses_identical_latest_nonempty_terminal_selector(self):
        trace = []
        for step in range(1, 6):
            answer = "24" if step < 4 else "25" if step == 4 else ""
            trace.append({"step": str(step), "answer": answer, "answer_normalized": answer,
                          "confidence": "100", "parse_success": "0" if step == 5 else "1",
                          "raw_generation_tokens": "10", "correct": "1" if step == 4 else "0"})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                replay([("synthetic::run", "question", trace)], [], output)
            metrics = json.loads((output / "replay_metrics.json").read_text())
            never = next(p for p in metrics["policies"] if p["policy"] == "never")
        self.assertEqual((never["baseline_correct"], never["active_correct"]), (1, 1))
        self.assertEqual(never["accuracy_delta"], 0)
        self.assertEqual(never["measured_completion_token_savings"], 0)
        self.assertIsNone(never["total_prompt_completion_auxiliary_token_savings"])

    def test_unknown_prompt_cost_is_missing_not_free(self):
        row = {"baseline_correct": True, "active_correct": True, "baseline_generated_tokens": 10,
               "active_generated_tokens": 5, "active_stop_step": 2}
        unknown = paired_metrics([row])
        self.assertIsNone(unknown["baseline_prompt_tokens"])
        self.assertIsNone(unknown["total_prompt_completion_auxiliary_token_savings"])
        recorded_zero = paired_metrics([row | {"baseline_prompt_tokens": 0, "active_prompt_tokens": 0}])
        self.assertEqual(recorded_zero["total_prompt_completion_auxiliary_token_savings"], .5)

    def test_runtime_adapter_does_not_read_gold_or_future_columns(self):
        row = GuardedRow(step="1", answer="25", confidence="95", parse_success="1", raw_generation_tokens="7",
                         correct="1", expected_answer="25", utility="999", oracle_stop="1", future_correct="0")
        observation = sanitized_observation(row)
        self.assertEqual((observation.answer, observation.generated_tokens), ("25", 7))


if __name__ == "__main__":
    unittest.main()
