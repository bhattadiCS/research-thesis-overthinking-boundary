"""Information-contract and numeric artifact tests for the trained prefix model."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH))
from online_stopping_controller import Observation
from prefix_stopping_model import (
    ARTIFACT_SCHEMA, FEATURE_CONTRACT, FEATURE_NAMES, SUPPORTED_DOMAINS,
    PrefixStoppingModel, prefix_features,
)


def observation(step, answer="12", confidence=80, **extra):
    return Observation(step, answer, confidence, True, 20 + step, thought="Check 3 times 4 equals 12.", **extra)


def synthetic_artifact():
    models = {}
    for name, intercept in (("q_current", -0.7), ("p_next", -0.2)):
        models[name] = {
            "mean": [0.2] * len(FEATURE_NAMES), "scale": [2.0] * len(FEATURE_NAMES),
            "coefficient": [(index - 10) / 100 for index in range(len(FEATURE_NAMES))],
            "intercept": intercept,
            "calibration": {"slope": 0.9, "intercept": 0.1},
        }
    return {
        "schema": ARTIFACT_SCHEMA, "feature_contract": FEATURE_CONTRACT,
        "feature_names": list(FEATURE_NAMES), "domains": list(SUPPORTED_DOMAINS),
        "min_steps": 2, "max_steps": 5, "step_cost": 0.05, "models": models,
    }


class PrefixContractTests(unittest.TestCase):
    def test_only_actual_prefix_fields_affect_features(self):
        prefix = (observation(1), observation(2, answer="13"))
        features = prefix_features(prefix, "gsm8k")
        # Timing, unconsumed telemetry, and scheduling costs are not predictors.
        revised = tuple(replace(o, raw_text="future gold=999", observed_ns=987654,
                                prompt_tokens=999, auxiliary_tokens=321, model_stop_flag=True) for o in prefix)
        self.assertEqual(features, prefix_features(revised, "gsm8k"))
        future_a = prefix + (observation(3, answer="CORRECT"),)
        future_b = prefix + (observation(3, answer="WRONG", confidence=1),)
        self.assertEqual(prefix_features(future_a[:2], "gsm8k"), prefix_features(future_b[:2], "gsm8k"))

    def test_answer_churn_and_streak_use_observed_answers(self):
        prefix = (observation(1, "  A B "), observation(2, "a   b"), observation(3, "17"))
        values = dict(zip(FEATURE_NAMES, prefix_features(prefix, "math")))
        self.assertEqual(values["answer_changed"], 1)
        self.assertEqual(values["prefix_answer_changes"], 1)
        self.assertEqual(values["answer_streak"], 1)
        self.assertEqual(dict(zip(FEATURE_NAMES, prefix_features(prefix[:2], "math")))["answer_streak"], 2)

    def test_fallback_confidence_is_missing(self):
        a = replace(observation(1), parse_success=False, confidence=100)
        b = replace(a, confidence=None)
        self.assertEqual(prefix_features((a,), "gsm8k"), prefix_features((b,), "gsm8k"))
        features = dict(zip(FEATURE_NAMES, prefix_features((b,), "gsm8k")))
        self.assertEqual(features["confidence_missing"], 1)
        self.assertEqual(features["confidence_centered"], 0)

    def test_rejects_labeled_rows_future_or_nonconsecutive_inputs(self):
        for prefix, exception in (
            ([{"step": 1, "correct": 1, "expected_answer": "12"}], TypeError),
            ([], ValueError), ([observation(2)], ValueError),
            ([observation(1), observation(3)], ValueError),
            ([observation(step) for step in range(1, 7)], ValueError),
        ):
            with self.assertRaises(exception):
                prefix_features(prefix, "gsm8k")
        with self.assertRaises(ValueError):
            prefix_features((observation(1),), "gpqa")

    def test_independent_linear_probability_calculation(self):
        artifact = synthetic_artifact()
        model = PrefixStoppingModel(artifact)
        prefix = (observation(1), observation(2, "13", 90))
        features = prefix_features(prefix, "gsm8k")
        estimate = model.estimate(prefix, "gsm8k")
        for name, actual in (("q_current", estimate.q_current), ("p_next", estimate.p_next)):
            component = artifact["models"][name]
            raw = component["intercept"]
            for index in range(len(features)):
                raw += component["coefficient"][index] * (features[index] - component["mean"][index]) / component["scale"][index]
            calibrated = component["calibration"]["slope"] * raw + component["calibration"]["intercept"]
            expected = 1 / (1 + math.exp(-calibrated))
            self.assertAlmostEqual(actual, expected, places=14)
        self.assertAlmostEqual(estimate.gain, estimate.p_next - estimate.q_current - 0.05, places=14)

    def test_invalid_serialization_fails_closed(self):
        mutations = (
            lambda a: a["feature_names"].reverse(),
            lambda a: a["models"]["q_current"]["coefficient"].pop(),
            lambda a: a["models"]["p_next"]["scale"].__setitem__(0, 0),
            lambda a: a["models"]["p_next"].__setitem__("intercept", float("inf")),
            lambda a: a["models"]["q_current"]["mean"].__setitem__(0, float("nan")),
            lambda a: a.__setitem__("max_steps", 6),
            lambda a: a.__setitem__("step_cost", -1),
        )
        for mutate in mutations:
            artifact = copy.deepcopy(synthetic_artifact())
            mutate(artifact)
            with self.assertRaises(ValueError):
                PrefixStoppingModel(artifact)

    def test_terminal_has_no_next_step_estimate(self):
        model = PrefixStoppingModel(synthetic_artifact())
        prefix = tuple(observation(step) for step in range(1, 6))
        self.assertTrue(0 <= model.score_current(prefix, "math") <= 1)
        with self.assertRaises(ValueError):
            model.estimate(prefix, "math")

    def test_extreme_finite_logits_emit_probabilities(self):
        artifact = synthetic_artifact()
        artifact["models"]["q_current"]["intercept"] = 1e6
        artifact["models"]["p_next"]["intercept"] = -1e6
        estimate = PrefixStoppingModel(artifact).estimate((observation(1), observation(2)), "math")
        self.assertEqual(estimate.q_current, 1)
        self.assertEqual(estimate.p_next, 0)


class FrozenArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = RESEARCH / "outputs/semester2/prefix_model_v1"
        path = cls.output / "prefix_model.json"
        if not path.is_file():
            raise unittest.SkipTest("train the frozen artifact before artifact verification")
        cls.model = PrefixStoppingModel.load(path)

    def test_file_byte_hash_matches_evaluation_ledger(self):
        raw = (self.output / "prefix_model.json").read_bytes()
        evaluation = json.loads((self.output / "evaluation.json").read_text())
        self.assertEqual(self.model.artifact_sha256, hashlib.sha256(raw).hexdigest())
        self.assertEqual(self.model.artifact_sha256, evaluation["artifact_sha256"])

    def test_training_freeze_copy_preserves_protocol_identity(self):
        protocol = json.loads((self.output / "training_protocol.json").read_text())
        raw = (self.output / "data_freeze_at_training.json").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), protocol["data_freeze"]["sha256"])

    def test_roles_are_task_disjoint_and_live_ids_excluded(self):
        import csv
        with (self.output / "task_split.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        groups = {role: {r["task_key"] for r in rows if r["split"] == role}
                  for role in ("train", "calibration", "evaluation")}
        self.assertTrue(all(groups.values()))
        self.assertFalse(groups["train"] & groups["calibration"])
        self.assertFalse(groups["train"] & groups["evaluation"])
        self.assertFalse(groups["calibration"] & groups["evaluation"])
        all_tasks = set().union(*groups.values())
        for path in (
            RESEARCH / "outputs/semester2/online_stopping_20261002/live_public_tasks.jsonl",
            RESEARCH / "adversarial_tasks_v1.jsonl",
        ):
            if path.is_file():
                for line in path.read_text().splitlines():
                    record = json.loads(line)
                    self.assertNotIn(f"{record['domain']}::{record['task_id']}", all_tasks)
        protocol = json.loads((self.output / "training_protocol.json").read_text())
        self.assertEqual(protocol["task_split_sha256"], hashlib.sha256((self.output / "task_split.csv").read_bytes()).hexdigest())
        self.assertEqual(self.model.artifact["protocol_sha256"], hashlib.sha256((self.output / "training_protocol.json").read_bytes()).hexdigest())

    def test_portable_scores_match_independent_vector_expression(self):
        import numpy as np
        import pandas as pd
        frame = pd.read_csv(self.output / "heldout_prefix_predictions.csv")
        x = frame.loc[:, FEATURE_NAMES].to_numpy(float)
        artifact = self.model.artifact
        for component_name in ("q_current", "p_next"):
            component = artifact["models"][component_name]
            logits = ((x - np.asarray(component["mean"])) / np.asarray(component["scale"])) @ np.asarray(component["coefficient"]) + component["intercept"]
            logits = logits * component["calibration"]["slope"] + component["calibration"]["intercept"]
            expected = 1 / (1 + np.exp(-logits))
            stored = frame[component_name].to_numpy(float)
            mask = np.isfinite(stored)
            self.assertLess(float(np.abs(expected[mask] - stored[mask]).max()), 1e-12)
            self.assertTrue(all(0 <= self.model._score(component_name, tuple(row)) <= 1 for row in x[:100]))

    def test_reconstruction_audit_covers_every_complete_row(self):
        import pandas as pd
        protocol = json.loads((self.output / "training_protocol.json").read_text())
        audit = pd.read_csv(self.output / "label_reconstruction_audit.csv", keep_default_na=False)
        census = protocol["eligibility_census"]
        self.assertEqual(len(audit), census["included_step_rows"])
        self.assertEqual(audit.trajectory_id.nunique(), census["included_trajectories"])
        self.assertEqual(int(audit.candidate_disagrees.sum()), census["reconstructed_candidate_disagreement_rows"])
        self.assertEqual(int(audit.label_disagrees.sum()), census["reconstructed_label_disagreement_rows"])
        self.assertEqual(int(audit.strict_json.sum()), census["strict_json_rows"])


class TrainingLabelTests(unittest.TestCase):
    def test_archive_regrading_and_candidate_carry_forward(self):
        import pandas as pd
        from train_prefix_stopping_model import load_examples
        answers = ("12", "", "13", "", "12")
        raw = [
            json.dumps({"thought": "Check answer.", "answer": answer, "confidence": 80, "stop": False})
            if answer else "" for answer in answers
        ]
        rows = [{
            "run_id": "synthetic_run", "task_id": "synthetic_task", "domain": "gsm8k",
            "step": step, "answer": answer if answer else "99", "correct": int(answer == "12"),
            "expected_answer": "12", "raw_generation_tokens": 20, "raw_text": raw[step - 1],
        } for step, answer in enumerate(answers, 1)]
        with tempfile.TemporaryDirectory(prefix="prefix-model-test-") as directory:
            path = Path(directory) / "trace_steps.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            frame, trajectories, census, audit, exclusions = load_examples([path], set())
            self.assertEqual(frame.q_target.tolist(), [1, 1, 0, 0, 1])
            self.assertEqual(frame.next_target.iloc[:4].tolist(), [1, 0, 0, 1])
            self.assertEqual(trajectories[0]["selected_steps"], [1, 1, 3, 3, 5])
            self.assertEqual(census["included_trajectories"], 1)
            self.assertEqual(len(audit), 5)
            self.assertEqual(int(audit.candidate_disagrees.sum()), 2)
            self.assertTrue(exclusions.empty)
            # Later answers and labels may change targets, but never earlier features.
            changed = pd.DataFrame(rows)
            changed.loc[changed.step >= 3, "expected_answer"] = "12"
            changed.loc[changed.step >= 3, "raw_text"] = ""
            changed.to_csv(path, index=False)
            alternate = load_examples([path], set())[0]
            self.assertEqual(frame.loc[frame.step <= 2, FEATURE_NAMES].to_numpy().tolist(),
                             alternate.loc[alternate.step <= 2, FEATURE_NAMES].to_numpy().tolist())


if __name__ == "__main__":
    unittest.main()
