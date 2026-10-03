"""Check finite-sample coverage and independent paired-result auditing."""
from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("live_uncertainty", ROOT / "tools/analyze_live_stopping_uncertainty.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PairedUncertaintyTests(unittest.TestCase):
    def test_binomial_interval_rejects_invalid_counts_and_error_rates(self):
        for k, n, error in ((-1, 4, .05), (5, 4, .05), (1.5, 4, .05),
                            (1, 4.5, .05), (0, 0, .05), (True, 4, .05),
                            (1, 4, 0), (1, 4, 1), (1, 4, 1.5), (1, 4, float("nan"))):
            with self.subTest(k=k, n=n, error=error), self.assertRaises(ValueError):
                MODULE.clopper_pearson(k, n, error)

    def test_exact_coverage_over_multinomial_outcomes(self):
        # Enumerate every possible (improvement,worsening,concordance) outcome.
        # This checks coverage of the final difference interval, including the
        # dependence between the two discordance categories, rather than only
        # comparing the implementation with another beta quantile routine.
        for n in (1, 2, 5, 10, 15):
            bounds = [MODULE.clopper_pearson(k, n, .025) for k in range(n + 1)]
            for plus_tenths in range(11):
                for minus_tenths in range(11 - plus_tenths):
                    plus, minus = plus_tenths / 10, minus_tenths / 10
                    same = max(0.0, 1 - plus - minus)
                    truth = plus - minus
                    coverage = 0.0
                    for improved in range(n + 1):
                        for worsened in range(n - improved + 1):
                            concordant = n - improved - worsened
                            probability = (math.comb(n, improved) * math.comb(n - improved, worsened)
                                           * plus ** improved * minus ** worsened * same ** concordant)
                            lo = bounds[improved][0] - bounds[worsened][1]
                            hi = bounds[improved][1] - bounds[worsened][0]
                            if lo - 1e-12 <= truth <= hi + 1e-12:
                                coverage += probability
                    self.assertGreaterEqual(coverage, .95 - 1e-12, (n, plus, minus, coverage))

    def fixture(self, folder: Path):
        frame = pd.DataFrame({"task_id": ["a", "b", "c", "d"],
                              "baseline_correct": [0, 1, 1, 0], "active_correct": [1, 0, 1, 0],
                              "baseline_generated_tokens": [100, 100, 100, 100],
                              "active_generated_tokens": [50, 100, 100, 50]})
        frame.to_csv(folder / "live_paired_results.csv", index=False)
        result = {"problems_or_trajectories": 4, "paired_improved": 1, "paired_worsened": 1,
                  "baseline_correct": 2, "active_correct": 2,
                  "baseline_generated_tokens": 400, "active_generated_tokens": 300}
        (folder / "live_metrics.json").write_text(json.dumps(result), encoding="utf-8")
        return frame, result

    def test_aggregate_audit_and_reproducible_task_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.fixture(folder)
            first = MODULE.analyze(folder, 1000, 804)
            second = MODULE.analyze(folder, 1000, 804)
            self.assertEqual(first, second)
            self.assertEqual(first["accuracy_delta"], 0)
            self.assertEqual(first["completion_token_savings"], .25)
            lo, hi = first["accuracy_delta_conservative_exact_95ci"]
            self.assertLessEqual(lo, 0)
            self.assertGreaterEqual(hi, 0)

    def test_rejects_aggregate_report_inconsistent_with_individual_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            _, reported = self.fixture(folder)
            reported["active_correct"] = 3
            (folder / "live_metrics.json").write_text(json.dumps(reported), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagrees"):
                MODULE.analyze(folder, 100, 804)

    def test_rejects_duplicate_tasks_and_fractional_token_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            frame, _ = self.fixture(folder)
            frame.loc[1, "task_id"] = "a"
            frame.to_csv(folder / "live_paired_results.csv", index=False)
            with self.assertRaisesRegex(ValueError, "unique task"):
                MODULE.analyze(folder, 100, 804)
            frame, _ = self.fixture(folder)
            frame["active_generated_tokens"] = frame.active_generated_tokens.astype(float)
            frame.loc[0, "active_generated_tokens"] = 50.5
            frame.to_csv(folder / "live_paired_results.csv", index=False)
            with self.assertRaisesRegex(ValueError, "nonnegative integers"):
                MODULE.analyze(folder, 100, 804)

    def test_rejects_missing_or_blank_task_identifiers(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for identifier in (None, "", "  ", "\t"):
                with self.subTest(identifier=identifier):
                    frame, _ = self.fixture(folder)
                    frame.loc[0, "task_id"] = identifier
                    frame.to_csv(folder / "live_paired_results.csv", index=False)
                    with self.assertRaisesRegex(ValueError, "unique task"):
                        MODULE.analyze(folder, 100, 804)

    def test_exact_token_totals_and_small_savings_above_float_integer_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for baseline, active in ((2**53 + 1, 2**53), (2**54, 2**54 - 1),
                                     (2**64 + 1, 2**64)):
                with self.subTest(baseline=baseline, active=active):
                    pd.DataFrame({"task_id": ["NA"], "baseline_correct": [1], "active_correct": [1],
                                  "baseline_generated_tokens": [baseline],
                                  "active_generated_tokens": [active]}).to_csv(
                                      folder / "live_paired_results.csv", index=False)
                    reported = {"problems_or_trajectories": 1, "paired_improved": 0, "paired_worsened": 0,
                                "baseline_correct": 1, "active_correct": 1,
                                "baseline_generated_tokens": baseline, "active_generated_tokens": active}
                    (folder / "live_metrics.json").write_text(json.dumps(reported), encoding="utf-8")
                    result = MODULE.analyze(folder, 100, 804)
                    self.assertEqual(result["counts"], reported)
                    expected = (baseline - active) / baseline
                    self.assertEqual(result["completion_token_savings"], expected)
                    self.assertEqual(result["completion_token_savings_cluster_bootstrap_95ci"], [expected, expected])

    def test_direct_analysis_rejects_invalid_bootstrap_draw_count(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.fixture(folder)
            for draws in (0, -1, 1.5, True):
                with self.subTest(draws=draws), self.assertRaisesRegex(ValueError, "draws"):
                    MODULE.analyze(folder, draws, 804)

    def test_token_exponent_budget_precedes_integer_materialization(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            frame, _ = self.fixture(folder)
            frame["active_generated_tokens"] = frame.active_generated_tokens.astype(str)
            frame.loc[0, "active_generated_tokens"] = "1e1000000000"
            frame.to_csv(folder / "live_paired_results.csv", index=False)
            with self.assertRaisesRegex(ValueError, "nonnegative integers"):
                MODULE.analyze(folder, 100, 804)


if __name__ == "__main__":
    unittest.main()
