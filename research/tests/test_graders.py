#!/usr/bin/env python
"""Regression tests for the answer graders (audit fixes).

Run:  python research/tests/test_graders.py
Covers the three confirmed grader defects from the deep code audit plus the
sound cases that must keep passing. Imports real_trace_experiments via importlib
(registered in sys.modules so its @dataclass resolves).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("rte", ROOT / "research" / "real_trace_experiments.py")
rte = importlib.util.module_from_spec(spec)
sys.modules["rte"] = rte
spec.loader.exec_module(rte)

norm_num = lambda s: rte.normalize_answer(s, "number")
norm_math = rte.normalize_math_answer
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
]

def main() -> int:
    failures = 0
    for fn, inp, expected, label in CASES:
        got = fn(inp)
        ok = got == expected
        failures += not ok
        print(f"  {'OK' if ok else 'XX'} {label}: {inp!r} -> {got!r} (want {expected!r})")
    print("  --- math equivalence ---")
    for case in EQ_CASES:
        a, b, want = case[0], case[1], case[2]
        label = case[3] if len(case) > 3 else f"{a} vs {b}"
        got = math_eq(a, b)
        ok = got == want
        failures += not ok
        print(f"  {'OK' if ok else 'XX'} {label}: math_eq({a!r},{b!r})={got} (want {want})")
    total = len(CASES) + len(EQ_CASES)
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
