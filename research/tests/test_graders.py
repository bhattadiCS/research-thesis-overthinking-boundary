#!/usr/bin/env python
"""Regression tests for the answer graders (audit fixes).

Run:  python research/tests/test_graders.py
Covers the three confirmed grader defects from the deep code audit plus the
sound cases that must keep passing. Imports real_trace_experiments via importlib
(registered in sys.modules so its @dataclass resolves).
"""
from __future__ import annotations

import importlib.util
import contextlib
import io
import sys
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("rte", ROOT / "research" / "real_trace_experiments.py")
rte = importlib.util.module_from_spec(spec)
sys.modules["rte"] = rte
spec.loader.exec_module(rte)

norm_num = lambda s: rte.normalize_answer(s, "number")
norm_int = lambda s: rte.normalize_answer(s, "int")
norm_math = rte.normalize_math_answer
norm_mcq = lambda s: rte.normalize_answer(s, "mcq")
math_eq = rte.math_answers_equivalent

# (callable, input, expected, label)
CASES = [
    # --- extract_numeric_candidate: answer is the LAST number, not a leading fraction ---
    (norm_num, "3/4 of the class, so 18 students", "18", "fraction-before-int -> int wins"),
    (norm_num, "the ratio is 2/3 giving 240", "240", "ratio then total -> total wins"),
    (norm_num, "18", "18", "bare int"),
    (norm_num, "the answer is 3/4", "3/4", "genuine fraction answer preserved"),
    (norm_num, "1,000 dollars", "1000", "thousands comma"),
    (norm_num, "42 cans", "42", "trailing unit"),
    # --- extract_word_fraction_values: bare ordinals are NOT fractions ---
    (norm_num, "a third", "1/3", "'a third' = 1/3 (explicit 'a' numerator) stays correct"),
    (norm_num, "the third option", "the third option", "bare ordinal 'third' is NOT read as 1/3"),
    (norm_num, "one half is 12", "12", "explicit word-fraction not mistaken for trailing answer"),
    # --- MATH _strip_latex_math: do not collapse equations to RHS ---
    (norm_math, "5x-7y+11z+4=0", "5x-7y+11z+4=0", "plane equation kept whole (not '0')"),
    (norm_math, "x = 5", "5", "single-var assignment peeled"),
    (norm_math, "n=12", "12", "single-var assignment peeled (no spaces)"),
    (norm_math, r"\boxed{\frac{1}{2}}", "1/2", "boxed fraction"),
    # --- MCQ (GPQA / ARC): extract the choice letter ---
    (norm_mcq, "B", "B", "bare letter"),
    (norm_mcq, "(D)", "D", "parenthesized letter"),
    (norm_mcq, "Answer: A", "A", "answer-marker letter"),
    (norm_mcq, "The answer is C.", "C", "answer-is letter"),
    (norm_mcq, "B. Mitochondria are the powerhouse", "B", "letter then option text"),
    (norm_mcq, "I think it's option D because ...", "D", "option-marker in prose"),
    (norm_mcq, "Mitochondria", "", "free-text option (no letter) -> empty"),
    (norm_num, ".5", "1/2", "leading decimal point"),
    (norm_num, "-2.5e-3", "-1/400", "negative scientific notation"),
    (norm_num, "1e3", "1000", "scientific exponent is not the final answer"),
    (norm_num, "1.5/2", "3/4", "decimal fraction numerator"),
    (norm_num, "-1.5/-2", "3/4", "signed decimal fraction"),
    (norm_num, r"\boxed{\frac{3}{4}}", "3/4", "nested boxed numeric fraction"),
    (norm_num, "3/4 of 20 gives .5", "1/2", "last numeric position with a leading decimal"),
    (norm_mcq, "Blue", "", "word prefix is not a choice letter"),
    (norm_mcq, "Cannot determine", "", "abstention is not option C"),
    (norm_mcq, "Answer is complicated", "", "answer cue must precede a complete letter token"),
    (norm_mcq, "Choice: banana", "", "option word is not its initial letter"),
    (norm_mcq, "A/B", "", "slash-delimited ambiguity is not a single choice"),
    (norm_mcq, "B2", "", "alphanumeric token is not a choice"),
    (norm_mcq, r"\boxed{B}", "B", "boxed MCQ remains supported"),
    (norm_math, "1(2)", "1(2)", "implicit product must not concatenate digits"),
    (norm_math, "(1)(2)", "(1)(2)", "adjacent groups must not concatenate digits"),
    (norm_mcq, "(A)/(B)", "", "parenthesized slash alternatives are ambiguous"),
    (norm_mcq, "(A)/B", "", "mixed slash alternatives are ambiguous"),
    (norm_mcq, "Answer: (A)/(B)", "", "answer cue does not disambiguate alternatives"),
    (norm_int, "7.0", "7", "integer-valued decimal retains its whole value"),
    (norm_int, "1e3", "1000", "integer-valued scientific notation"),
    (norm_int, "-2e3", "-2000", "negative integer-valued scientific notation"),
    (norm_int, "2.5", "5/2", "noninteger decimal is not reduced to its last digit"),
    (norm_int, "the answer is 3/2", "3/2", "noninteger fraction is not reduced to its denominator"),
    (norm_int, "42 cans", "42", "integer with units remains supported"),
    (norm_int, "12", "12", "bare integer remains supported"),
]

EQ_CASES = [
    ("18", "18", True),
    (r"\frac{1}{2}", "0.5", True),
    ("x^2+1", "1+x^2", True),
    ("5x-7y+11z+4=0", "0", False, "plane must NOT equal 0"),
    ("3", "4", False),
    # text/unit-wrapped numeric answers (model right, was a false negative)
    ("Savings: 550 gallons", "550", True, "number wrapped in words/units"),
    ("1.25 miles", "1.25", True, "decimal with unit"),
    ("c=33", "33", True, "single-var assignment"),
    # guard: verbose wrong answer with a different number stays wrong
    ("Hybrid: 250 gallons saved", "550", False, "different number -> not a false positive"),
    (r'"answer": "3"', r"\frac{7}{2}", False, "nested-json wrong answer stays wrong"),
    ("1+2", "2", False, "arithmetic is not its trailing numeric token"),
    ("1+2", "3", True),
    ("2*3", "3", False),
    ("2*3", "6", True),
    ("sqrt(2)", "2", False),
    ("cos(0)", "0", False),
    ("cos(0)", "1", True),
    ("1(2)", "12", False),
    ("1(2)", "2", True),
    ("(1)(2)", "12", False),
    ("(1)(2)", "2", True),
    ("7.0/2", "0", False),
    ("7.0/2", "3.5", True),
    ("-2.5e-3", "-1/400", True),
    (".5", "1/2", True),
    ("1e3", "3", False),
    ("1e3", "1000", True),
    (r"\sqrt{4}", "2", True),
    (r"\sqrt{2}", "2", False),
    ("1/2 gallons", "0.5", True),
    ("1e3 miles", "1000", True),
    ("-2.5e-3 meters", "-1/400", True),
    ("1e400", "2e400", False, "large exact numeric answers do not crash float conversion"),
    ("sqrt 2", "2", False),
    ("sqrt 4", "2", True),
    ("cos 0", "0", False),
    ("cos 0", "1", True),
    ("log 1", "1", False),
    ("log 1", "0", True),
    ("pi 2", "2", False),
    ("x 2", "2", False),
    ("3!!", "3", True),
    ("factorial2(3)", "3", True),
    ("9007199254740993", "9007199254740992", False),
    ("100000000000000000001", "100000000000000000000", False),
    ("100000000000000000001 gallons", "100000000000000000000", False),
    ("0.0000005", "0", True, "exact comparison retains the declared absolute tolerance"),
    ("0.000001", "0", False, "absolute tolerance remains strict"),
]


class GraderRegressionTests(unittest.TestCase):
    """Keep historical cases and focused current-API regressions discoverable."""

    def test_symbolic_candidate_cannot_execute_python_calls(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            equivalent = math_eq("__import__('builtins').print('GRADER_EVAL_MARKER')", "x")
        self.assertFalse(equivalent)
        self.assertEqual(stream.getvalue(), "")

    def test_symbolic_gold_cannot_execute_python_calls(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            equivalent = math_eq("x", "__import__('builtins').print('GRADER_EVAL_MARKER')")
        self.assertFalse(equivalent)
        self.assertEqual(stream.getvalue(), "")

    def test_huge_exponent_is_rejected_before_fraction_construction(self):
        with mock.patch.object(rte, "Fraction", side_effect=AssertionError("Must not construct a huge number")):
            self.assertEqual(rte._canonical_fraction("1e1000000000"), "1e1000000000")
            self.assertIsNone(rte._as_number("1e1000000000"))

    def test_symbolic_arithmetic_and_function_budgets(self):
        # Only safe representative values are constructed. The outer recursive
        # powers/factorials must be rejected before expensive materialization.
        for expression in ("1e1000000000", "2**1000000000", "2**(2**20)",
                           "factorial(201)", "factorial(factorial(10))", "(x+1)**1000"):
            with self.subTest(expression=expression), self.assertRaisesRegex(ValueError, "budget"):
                rte._safe_symbolic_expression(expression)
        self.assertEqual(rte._safe_symbolic_expression("2**10"), 1024)


def _normalization_test(case):
    def test(self):
        fn, value, expected, label = case
        self.assertEqual(fn(value), expected, label)
    return test


def _equivalence_test(case):
    def test(self):
        a, b, expected = case[:3]
        self.assertEqual(math_eq(a, b), expected, case[3] if len(case) > 3 else f"{a} vs {b}")
    return test


for _index, _case in enumerate(CASES, 1):
    setattr(GraderRegressionTests, f"test_normalization_{_index:02}", _normalization_test(_case))
for _index, _case in enumerate(EQ_CASES, 1):
    setattr(GraderRegressionTests, f"test_equivalence_{_index:02}", _equivalence_test(_case))

def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(GraderRegressionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
