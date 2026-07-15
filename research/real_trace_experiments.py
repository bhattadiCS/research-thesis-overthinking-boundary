from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import logging
import math
import random
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

TORCHVISION_STUB_ROOT = Path(__file__).resolve().parent / "_vendor" / "torchvision_stub"
if TORCHVISION_STUB_ROOT.exists() and importlib.util.find_spec("torchvision") is None:
    sys.path.insert(0, str(TORCHVISION_STUB_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None


def ensure_bitsandbytes_params4bit_compatibility() -> None:
    try:
        from bitsandbytes.nn.modules import Params4bit
    except Exception:
        return

    current_new = Params4bit.__new__
    if getattr(current_new, "_hf_compat_patched", False):
        return

    try:
        signature = inspect.signature(current_new)
    except (TypeError, ValueError):
        return

    if "_is_hf_initialized" in signature.parameters:
        return

    def compat_new(cls, data=None, *args, **kwargs):
        kwargs.pop("_is_hf_initialized", None)
        return current_new(cls, data, *args, **kwargs)

    compat_new._hf_compat_patched = True
    Params4bit.__new__ = compat_new
    logging.info(
        "Applied Params4bit compatibility shim for Accelerate/Transformers quantized dispatch."
    )


def ensure_accelerate_meta_state_dict_compatibility() -> None:
    try:
        import accelerate.hooks as accelerate_hooks
        import accelerate.big_modeling as accelerate_big_modeling
    except Exception:
        return

    current_hook = accelerate_hooks.attach_execution_device_hook
    if getattr(current_hook, "_hf_compat_patched", False):
        return

    add_hook_to_module = accelerate_hooks.add_hook_to_module
    align_devices_hook = accelerate_hooks.AlignDevicesHook

    def compat_attach_execution_device_hook(
        module: torch.nn.Module,
        execution_device: int | str | torch.device,
        skip_keys: str | list[str] | None = None,
        preload_module_classes: list[str] | None = None,
        tied_params_map: dict[int, dict[torch.device, torch.Tensor]] | None = None,
    ) -> None:
        if not hasattr(module, "_hf_hook"):
            try:
                has_state = len(module.state_dict()) > 0
            except RuntimeError as exc:
                if "Tensor.item() cannot be called on meta tensors" not in str(exc):
                    raise
                has_state = any(value is not None for value in module._parameters.values()) or any(
                    value is not None for value in module._buffers.values()
                )
            if has_state:
                add_hook_to_module(
                    module,
                    align_devices_hook(
                        execution_device,
                        skip_keys=skip_keys,
                        tied_params_map=tied_params_map,
                    ),
                )

        if preload_module_classes is not None and module.__class__.__name__ in preload_module_classes:
            return

        for child in module.children():
            compat_attach_execution_device_hook(
                child,
                execution_device,
                skip_keys=skip_keys,
                preload_module_classes=preload_module_classes,
                tied_params_map=tied_params_map,
            )

    compat_attach_execution_device_hook._hf_compat_patched = True
    accelerate_hooks.attach_execution_device_hook = compat_attach_execution_device_hook
    if hasattr(accelerate_big_modeling, "attach_execution_device_hook"):
        accelerate_big_modeling.attach_execution_device_hook = compat_attach_execution_device_hook
    logging.info(
        "Applied Accelerate meta state_dict compatibility shim for quantized dispatch."
    )


def resolve_low_vram_device_map(
    model_spec: ModelSpec,
    actual_device: str,
    target_precision: str,
    device_map: str | None,
) -> str | dict[str, int | str] | None:
    if device_map is not None:
        return device_map
    if actual_device != "cuda" or target_precision != "4bit":
        return None
    # For small models (≤ 8B params), force single-GPU loading.
    # device_map="auto" estimates memory using the unquantized model size,
    # which can cause accelerate to offload 4-bit layers to CPU as meta
    # tensors -- bitsandbytes then crashes with "Cannot copy out of meta
    # tensor; no data!" when generate() tries to move them back to CUDA.
    try:
        num_params = float(re.findall(r"[\d.]+", model_spec.parameter_count)[0])
        if num_params <= 8:
            return {"": 0}
    except (IndexError, ValueError):
        pass
    return None


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "real_traces"
STEP_COST = 0.05

class IOManager:
    def __init__(self, max_threads: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_threads)
        self.pending_futures = []

    def save_npz_async(self, path: Path, data_dict: dict[str, np.ndarray]) -> None:
        future = self.executor.submit(np.savez_compressed, path, **data_dict)
        self.pending_futures.append(future)
        self._cleanup()

    def _cleanup(self) -> None:
        self.pending_futures = [f for f in self.pending_futures if not f.done()]

    def wait_all(self) -> None:
        for f in self.pending_futures:
            f.result()
        self.pending_futures = []

    def shutdown(self) -> None:
        self.wait_all()
        self.executor.shutdown()

SYSTEM_PROMPT = (
    "You are participating in a research protocol about incremental reasoning. "
    "Each step should be brief and update the current best answer rather than re-deriving the full solution. "
    "If you already know the answer, do one short verification step and still return the requested format."
)
DAY_NAMES = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]
NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
FRACTION_DENOMINATORS = {
    "half": 2,
    "halves": 2,
    "third": 3,
    "thirds": 3,
    "quarter": 4,
    "quarters": 4,
    "fourth": 4,
    "fourths": 4,
    "fifth": 5,
    "fifths": 5,
    "sixth": 6,
    "sixths": 6,
    "seventh": 7,
    "sevenths": 7,
    "eighth": 8,
    "eighths": 8,
    "ninth": 9,
    "ninths": 9,
    "tenth": 10,
    "tenths": 10,
}
PLURAL_FRACTION_WORDS = {
    "halves",
    "thirds",
    "quarters",
    "fourths",
    "fifths",
    "sixths",
    "sevenths",
    "eighths",
    "ninths",
    "tenths",
}
ANSWER_CUE_PHRASES = (
    "answer",
    "final",
    "therefore",
    "thus",
    "hence",
    "so",
    "equals",
    "equal to",
    "gives",
    "will be",
    "would be",
    "should be",
    "left with",
    "remaining",
)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    domain: str
    difficulty: str
    prompt: str
    answer_type: str
    expected_answer: str
    notes: str
    source: str = "builtin"
    source_index: int = -1


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    hf_name: str
    family: str
    parameter_count: str


BUILTIN_TASKS = [
    TaskSpec(
        task_id="calendar_offset_easy",
        domain="calendar",
        difficulty="easy",
        prompt=(
            "If today is Tuesday, what day of the week will it be 14 days later? "
            "Answer with only the weekday."
        ),
        answer_type="day",
        expected_answer="tuesday",
        notes="Very easy task where extra verification can become redundant.",
    ),
    TaskSpec(
        task_id="fraction_remaining",
        domain="arithmetic",
        difficulty="medium",
        prompt=(
            "A tank is half full. Then one quarter of the whole tank is drained. "
            "What fraction of the tank is full now?"
        ),
        answer_type="fraction",
        expected_answer="1/4",
        notes="Easy fraction task with exact verification.",
    ),
    TaskSpec(
        task_id="linear_equation",
        domain="algebra",
        difficulty="medium",
        prompt="Solve 2x + 5 = 19. Answer with only the value of x.",
        answer_type="int",
        expected_answer="7",
        notes="Simple algebra task that should be solvable by small instruct models.",
    ),
    TaskSpec(
        task_id="python_trace",
        domain="code",
        difficulty="medium",
        prompt=(
            "What does this Python code print?\n"
            "total = 0\n"
            "for i in range(1, 5):\n"
            "    total += i\n"
            "print(total)\n"
            "Answer with only the printed value."
        ),
        answer_type="int",
        expected_answer="10",
        notes="Straightforward code tracing benchmark.",
    ),
]


MODEL_CATALOG = {
    "deepseek_r1_distill_1p5b": ModelSpec(
        alias="deepseek_r1_distill_1p5b",
        hf_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        family="DeepSeek-R1 distill",
        parameter_count="1.5B",
    ),
    "deepseek_r1_distill_7b": ModelSpec(
        alias="deepseek_r1_distill_7b",
        hf_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        family="DeepSeek-R1 distill",
        parameter_count="7B",
    ),
    "qwen2p5_0p5b": ModelSpec(
        alias="qwen2p5_0p5b",
        hf_name="Qwen/Qwen2.5-0.5B-Instruct",
        family="Qwen2.5 instruct",
        parameter_count="0.5B",
    ),
    "qwen2p5_3b": ModelSpec(
        alias="qwen2p5_3b",
        hf_name="Qwen/Qwen2.5-3B-Instruct",
        family="Qwen2.5 instruct",
        parameter_count="3B",
    ),
    "qwen2p5_7b": ModelSpec(
        alias="qwen2p5_7b",
        hf_name="Qwen/Qwen2.5-7B-Instruct",
        family="Qwen2.5 instruct",
        parameter_count="7B",
    ),
    "qwen2p5_14b": ModelSpec(
        alias="qwen2p5_14b",
        hf_name="Qwen/Qwen2.5-14B-Instruct",
        family="Qwen2.5 instruct",
        parameter_count="14B",
    ),
    "qwen2p5_32b": ModelSpec(
        alias="qwen2p5_32b",
        hf_name="Qwen/Qwen2.5-32B-Instruct",
        family="Qwen2.5 instruct",
        parameter_count="32B",
    ),
    "mistral_7b_instruct_v0p3": ModelSpec(
        alias="mistral_7b_instruct_v0p3",
        hf_name="mistralai/Mistral-7B-Instruct-v0.3",
        family="Mistral instruct",
        parameter_count="7B",
    ),
    "gemma_4_31b_it": ModelSpec(
        alias="gemma_4_31b_it",
        hf_name="google/gemma-4-31B-it",
        family="Gemma 4",
        parameter_count="31B",
    ),
    "gemma_4_26b_moe_it": ModelSpec(
        alias="gemma_4_26b_moe_it",
        hf_name="google/gemma-4-26B-A4B-it",
        family="Gemma 4 MoE",
        parameter_count="26B",
    ),
    "gemma_4_e4b_it": ModelSpec(
        alias="gemma_4_e4b_it",
        hf_name="principled-intelligence/gemma-4-E4B-it-text-only",
        family="Gemma 4 Edge (text-only)",
        parameter_count="4B",
    ),
    "qwen_3p5_9b": ModelSpec(
        alias="qwen_3p5_9b",
        hf_name="Qwen/Qwen3.5-9B",
        family="Qwen3.5 instruct",
        parameter_count="9B",
    ),
    "qwen_3p5_35b_moe_it": ModelSpec(
        alias="qwen_3p5_35b_moe_it",
        hf_name="Qwen/Qwen3.5-35B-A3B",
        family="Qwen3.5 MoE",
        parameter_count="35B",
    ),
    "qwen_3p5_35b_gptq_int4": ModelSpec(
        alias="qwen_3p5_35b_gptq_int4",
        hf_name="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
        family="Qwen3.5 MoE GPTQ",
        parameter_count="35B",
    ),
    "qwen_3p5_35b_awq": ModelSpec(
        alias="qwen_3p5_35b_awq",
        hf_name="QuantTrio/Qwen3.5-35B-A3B-AWQ",
        family="Qwen3.5 MoE AWQ",
        parameter_count="35B",
    ),
    "llama_3p1_8b_instruct": ModelSpec(
        alias="llama_3p1_8b_instruct",
        hf_name="meta-llama/Llama-3.1-8B-Instruct",
        family="Llama 3.1",
        parameter_count="8B",
    ),
    "phi_4_mini_instruct": ModelSpec(
        alias="phi_4_mini_instruct",
        hf_name="microsoft/Phi-4-mini-instruct",
        family="Phi 4",
        parameter_count="4B",
    ),
    "internlm3_8b_instruct": ModelSpec(
        alias="internlm3_8b_instruct",
        hf_name="internlm/InternLM3-8B-Instruct",
        family="InternLM3",
        parameter_count="8B",
    ),
    "yi_1p5_9b_chat": ModelSpec(
        alias="yi_1p5_9b_chat",
        hf_name="01-ai/Yi-1.5-9B-Chat",
        family="Yi 1.5",
        parameter_count="9B",
    ),
    "mistral_small_24b_2409": ModelSpec(
        # Text-only Mistral-Small (22B, MistralForCausalLM). Replaces the
        # Mistral-Small-3.1-24B-2503 entry, which is a vision-language model
        # (Mistral3ForConditionalGeneration) that this text-only harness cannot
        # load. Gated -- needs HF_TOKEN + accepting the license on the HF page.
        alias="mistral_small_24b_2409",
        hf_name="mistralai/Mistral-Small-Instruct-2409",
        family="Mistral Small",
        parameter_count="22B",
    ),
}


def sanitize_split_name(dataset_split: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", dataset_split.lower()).strip("_") or "split"


def extract_word_fraction_values(text: str) -> list[str]:
    values: list[str] = []
    pattern = re.compile(
        r"\b(?:(?P<numerator>a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+)?"
        r"(?P<denominator>half|halves|third|thirds|quarter|quarters|fourth|fourths|fifth|fifths|sixth|sixths|seventh|sevenths|eighth|eighths|ninth|ninths|tenth|tenths)\b",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        # AUDIT FIX: require an EXPLICIT numerator. Previously a bare singular
        # ordinal ("third", "quarter") was read as 1/3, 1/4 via an implicit
        # "one", turning ordinary words into fraction answers and creating
        # false negatives on GSM8K.
        if match.group("numerator") is None:
            continue
        numerator_token = match.group("numerator").lower()
        denominator_token = match.group("denominator").lower()
        numerator = NUMBER_WORDS.get(numerator_token)
        denominator = FRACTION_DENOMINATORS.get(denominator_token)
        if numerator is None or denominator is None:
            continue
        values.append(str(Fraction(numerator, denominator)))
    return values


def _canonical_fraction(value: str) -> str:
    try:
        normalized = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return value
    return str(normalized.numerator) if normalized.denominator == 1 else str(normalized)


def extract_numeric_candidate(text: str) -> str:
    # AUDIT FIX: the final answer is the numeric token that appears LAST by
    # position, regardless of whether it is a fraction or an integer/decimal.
    # The previous implementation scanned fractions first and returned a leading
    # fraction even when the real answer was the trailing integer, deflating
    # GSM8K correctness on ratio-flavoured problems.
    stripped = text.strip().lower().replace(",", "")
    if not stripped:
        return ""

    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", stripped)
    if boxed_matches:
        stripped = boxed_matches[-1].strip().lower().replace(",", "")

    candidates: list[tuple[int, str]] = []
    fraction_spans: list[tuple[int, int]] = []
    for match in re.finditer(r"-?\d+\s*/\s*-?\d+", stripped):
        fraction_spans.append((match.start(), match.end()))
        candidates.append((match.start(), _canonical_fraction(match.group().replace(" ", ""))))
    for match in re.finditer(r"-?\d+(?:\.\d+)?", stripped):
        if any(start <= match.start() < end for start, end in fraction_spans):
            continue  # already covered by a fraction match
        candidates.append((match.start(), _canonical_fraction(match.group())))

    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[-1][1]

    word_fraction_matches = extract_word_fraction_values(stripped)
    if word_fraction_matches:
        return word_fraction_matches[-1]

    return ""


def _last_boxed_content(s: str) -> str | None:
    """Return the content of the last \\boxed{...} with balanced braces (handles
    nesting like \\boxed{\\frac{1}{2}}, which a non-greedy regex truncates)."""
    key = "\\boxed{"
    idx = s.rfind(key)
    if idx == -1:
        return None
    i = idx + len(key)
    depth = 1
    out: list[str] = []
    while i < len(s) and depth > 0:
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out)


def _strip_latex_math(text: str) -> str:
    """Reduce a LaTeX/competition-math answer to a compact comparable form."""
    s = str(text).strip()
    # AUDIT FIX: only peel a leading single-variable assignment ("x = ...").
    # The previous "split on last =" collapsed equations/systems/lines to a
    # single token (e.g. "5x-7y+11z+4=0" -> "0"), accepting "0" as correct for
    # a plane. Keep the full expression for everything else.
    if (s.count("=") == 1 and re.match(r"^\s*[A-Za-z]\w*\s*=", s)
            and not any(op in s for op in ("<=", ">=", "==", "!="))):
        s = s.split("=", 1)[1].strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    boxed_content = _last_boxed_content(s)
    if boxed_content is not None:
        s = boxed_content
    s = re.sub(r"\\(?:text|mathrm|mbox|mathbf|mathit|operatorname)\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\frac\s*(\d)\s*(\d)", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\s*(\w)", r"sqrt(\1)", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/").replace("\\pi", "pi")
    for token in ("\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\ ", "\\$",
                  "\\(", "\\)", "\\[", "\\]", "^{\\circ}", "^\\circ", "\\circ", "\\%", "%", "$"):
        s = s.replace(token, "")
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # 1,000 -> 1000
    return s.replace(" ", "").rstrip(".").strip()


def normalize_math_answer(raw_answer: str) -> str:
    if raw_answer is None:
        return ""
    s = _strip_latex_math(raw_answer)
    if s == "":
        return ""
    bare = s.replace("(", "").replace(")", "")
    if re.fullmatch(r"-?\d+(\.\d+)?", bare) or re.fullmatch(r"-?\d+/-?\d+", bare):
        try:
            frac = Fraction(bare).limit_denominator(10**6)
            return str(frac.numerator) if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"
        except (ValueError, ZeroDivisionError):
            pass
    return s.lower()


def _as_number(text: str | None) -> float | None:
    if not text:
        return None
    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        try:
            return float(text)
        except (ValueError, TypeError):
            return None


def math_answers_equivalent(candidate: str, expected: str) -> bool:
    norm_candidate = normalize_math_answer(candidate)
    norm_expected = normalize_math_answer(expected)
    if not norm_expected:
        return False
    if norm_candidate == norm_expected:
        return True
    try:
        if abs(float(norm_candidate) - float(norm_expected)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    # VALIDATION FIX: when the gold answer is purely numeric, the model often
    # gives the right number wrapped in words/units ("Savings: 550 gallons",
    # "1.25 miles"). Compare the candidate's trailing numeric token by value.
    # Gated on a numeric gold answer so a verbose wrong answer that merely
    # mentions a different number cannot become a false positive.
    exp_num = _as_number(norm_expected)
    candidate_str = str(candidate)
    # Only number-grab from PROSE-wrapped numbers, never from symbolic answers
    # (equations/expressions: '=', '^', '\', or digit-letter adjacency like "5r"),
    # so "5x-7y+11z+4=0" is never reduced to its trailing "0".
    looks_symbolic = bool(re.search(r"[=^\\]|\d[a-zA-Z]|[a-zA-Z]\d", candidate_str))
    if exp_num is not None and not looks_symbolic:
        cand_num = _as_number(extract_numeric_candidate(candidate_str))
        if cand_num is not None and abs(cand_num - exp_num) < 1e-6:
            return True
    try:  # symbolic fallback; sympy ships as an indirect torch dependency
        import sympy
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations, implicit_multiplication_application,
        )
        transforms = standard_transformations + (implicit_multiplication_application,)
        def _sympy_ready(expr: str) -> str:
            # Digit-letter adjacency like '0x^2' tokenizes as a broken hex
            # literal BEFORE sympy's implicit-multiplication transform runs
            # (TokenError; rigor_audit/01 §4c) -- make the product explicit,
            # leaving scientific notation (2e3) alone.
            return re.sub(r"(\d)(?![eE][0-9])([a-zA-Z(])", r"\1*\2", expr.replace("^", "**"))

        expr_a = parse_expr(_sympy_ready(norm_candidate), transformations=transforms)
        expr_b = parse_expr(_sympy_ready(norm_expected), transformations=transforms)
        if sympy.simplify(expr_a - expr_b) == 0:
            return True
    except Exception:  # noqa: BLE001 -- unparseable expressions just fall through
        pass
    return False


_MCQ_LETTERS = "ABCDEFGH"


def normalize_mcq_answer(raw_answer: str) -> str:
    """Extract the single multiple-choice letter (A-H) from a model answer.
    Handles 'B', '(C)', 'Answer: D', 'The answer is A', boxed letters, etc."""
    if raw_answer is None:
        return ""
    s = str(raw_answer).strip()
    boxed = _last_boxed_content(s)
    if boxed is not None:
        s = boxed
    s = s.upper()
    for pattern in (r"(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:|=)?\s*\(?([A-H])\)?",
                    r"\(([A-H])\)",
                    # '/' guards: a bare letter adjacent to a slash is part of a
                    # token like 'N/A', not a choice (rigor_audit/01 §4a).
                    r"(?<![A-Z/])([A-H])(?![A-Z/])"):
        matches = re.findall(pattern, s)
        if matches:
            return matches[-1]
    return s[:1] if s[:1] in _MCQ_LETTERS else ""


def _format_mcq_prompt(question: str, labeled_options: list[tuple[str, str]]) -> str:
    lines = [question.strip(), "", "Options:"]
    lines += [f"{label}. {text}" for label, text in labeled_options]
    lines += ["", "Respond with the letter of the single correct option."]
    return "\n".join(lines)


def normalize_answer(raw_answer: str, answer_type: str) -> str:
    if answer_type == "math":
        return normalize_math_answer(raw_answer)
    if answer_type == "mcq":
        return normalize_mcq_answer(raw_answer)
    text = raw_answer.strip().lower()
    text = re.sub(r"\s+", " ", text)
    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed_matches:
        text = boxed_matches[-1].strip().lower()
    if not text:
        return ""

    if answer_type == "day":
        for day_name in DAY_NAMES:
            if day_name in text:
                return day_name
        return text.split()[0]

    if answer_type == "int":
        match = re.findall(r"-?\d+", text)
        return match[-1] if match else text

    if answer_type in {"fraction", "number"}:
        numeric_candidate = extract_numeric_candidate(text)
        return numeric_candidate or text

    return text


def verify_answer(task: TaskSpec, candidate_answer: str) -> bool:
    if task.answer_type == "math":
        return math_answers_equivalent(candidate_answer, task.expected_answer)
    if task.answer_type == "mcq":
        gold = normalize_mcq_answer(task.expected_answer)
        return gold != "" and normalize_mcq_answer(candidate_answer) == gold
    return normalize_answer(candidate_answer, task.answer_type) == normalize_answer(task.expected_answer, task.answer_type)


def utility(correct: bool, step: int) -> float:
    return float(correct) - STEP_COST * (step - 1)


def lexical_overlap(current_text: str, prior_text: str) -> float:
    current_tokens = set(re.findall(r"[a-z0-9]+", current_text.lower()))
    prior_tokens = set(re.findall(r"[a-z0-9]+", prior_text.lower()))
    if not current_tokens or not prior_tokens:
        return 0.0
    return len(current_tokens & prior_tokens) / len(current_tokens | prior_tokens)


def conversation_prompt(task: TaskSpec, history: list[dict[str, Any]], step: int, max_steps: int, prompt_mode: str) -> str:
    if history:
        rendered_history = "\n".join(
            (
                f"Step {item['step']}: thought={item['thought']} | answer={item['answer']} | "
                f"confidence={item['confidence']} | stop={item['model_stop_flag']}"
            )
            for item in history
        )
    else:
        rendered_history = "No previous steps."

    protocol_text = (
        f"Research protocol: you are at incremental reasoning step {step} of {max_steps}. "
        "Even if you already know the answer, continue with one short reasoning or verification step. "
        "Do not repeat the full solution. Update the answer field every time.\n\n"
        f"Previous steps:\n{rendered_history}\n\n"
    )

    if prompt_mode == "structured_four_line":
        format_instr = (
            "Return exactly four lines:\n"
            "THOUGHT: <one short sentence>\n"
            "ANSWER: <current best final answer only>\n"
            "CONFIDENCE: <integer 0-100>\n"
            "STOP: <yes or no>"
        )
    elif prompt_mode == "minimal_json":
        format_instr = (
            "Return a single JSON object with the following keys: 'thought' (string), 'answer' (string), "
            "'confidence' (integer 0-100), and 'stop' (boolean)."
        )
    else:
        if task.answer_type in {"int", "fraction", "number"}:
            format_instr = "Provide only the final numeric answer."
        else:
            format_instr = "Provide only the final answer."

    return (
        f"Task id: {task.task_id}\n"
        f"Domain: {task.domain}\n"
        f"Difficulty band: {task.difficulty}\n"
        f"Task: {task.prompt}\n\n"
        f"{protocol_text if prompt_mode != 'answer_only' else ''}"
        f"{format_instr}"
    )


def render_prompt(tokenizer: Any, user_prompt: str, system_prompt_mode: str) -> str:
    if system_prompt_mode == "default":
        sys_msg = SYSTEM_PROMPT
    elif system_prompt_mode == "short":
        sys_msg = "You are a logical reasoning assistant."
    else:
        sys_msg = ""

    messages = []
    if sys_msg:
        messages.append({"role": "system", "content": sys_msg})
    messages.append({"role": "user", "content": user_prompt})

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if sys_msg:
        return f"System: {sys_msg}\nUser: {user_prompt}\nAssistant:"
    return f"User: {user_prompt}\nAssistant:"


def extract_typed_answer(text: str, answer_type: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""

    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", stripped)
    if boxed_matches:
        stripped = boxed_matches[-1].strip()

    if answer_type == "math":
        return stripped.strip()

    if answer_type == "mcq":
        return normalize_mcq_answer(stripped)

    lowered = stripped.lower()
    if answer_type == "day":
        day_matches = re.findall(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", lowered)
        return day_matches[-1] if day_matches else ""

    if answer_type == "int":
        int_matches = re.findall(r"-?\d+", stripped)
        return int_matches[-1] if int_matches else ""

    if answer_type in {"fraction", "number"}:
        return extract_numeric_candidate(stripped)

    return stripped


def split_answer_segments(text: str) -> list[str]:
    return [
        segment.strip(" \t\r\n-:;,")
        for segment in re.split(r"(?:\n+|(?<=[.!?])\s+)", text)
        if segment.strip()
    ]


def segment_has_answer_cue(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in ANSWER_CUE_PHRASES)


def extract_answer(raw_text: str, answer_type: str) -> tuple[str, str]:
    cleaned = raw_text.strip()
    if not cleaned:
        return "", "fallback"

    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", cleaned)
    if boxed_matches:
        candidate = extract_typed_answer(boxed_matches[-1], answer_type)
        if candidate:
            return candidate, "boxed"

    regions: list[tuple[str, str]] = []
    think_parts = re.split(r"</think>", cleaned, flags=re.IGNORECASE)
    if len(think_parts) > 1 and think_parts[-1].strip():
        regions.append(("after_think", think_parts[-1].strip()))
    regions.append(("full_text", cleaned))

    for region_name, region_text in regions:
        for segment in reversed(split_answer_segments(region_text)):
            candidate = extract_typed_answer(segment, answer_type)
            if candidate and segment_has_answer_cue(segment):
                return candidate, f"{region_name}_cue_segment"

    for region_name, region_text in regions:
        candidate = extract_typed_answer(region_text, answer_type)
        if candidate:
            return candidate, region_name

    return "", "fallback"


def parse_generation(raw_text: str, answer_type: str, prompt_mode: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```\w*\n", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n```$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    result = {
        "thought": "",
        "answer": "",
        "confidence": 50,
        "stop": False,
        "parse_success": 0,
        "output_format_type": "fallback_freeform",
        "answer_extraction_source": "fallback",
        "stop_extraction_source": "default",
        "confidence_extraction_source": "default",
    }

    if prompt_mode == "structured_four_line":
        line_thought = re.search(r"^THOUGHT\s*:\s*(.*)$", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        line_answer = re.search(r"^ANSWER\s*:\s*(.*)$", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        line_confidence = re.search(r"^CONFIDENCE\s*:\s*(-?\d+(?:\.\d+)?)", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        line_stop = re.search(r"^STOP\s*:\s*(yes|no|true|false)", cleaned, flags=re.IGNORECASE | re.MULTILINE)

        if line_thought and line_answer and line_confidence and line_stop:
            result["parse_success"] = 1
            result["output_format_type"] = "structured_exact"
        elif line_thought or line_answer or line_confidence or line_stop:
            result["output_format_type"] = "structured_partial"

        if line_thought:
            result["thought"] = line_thought.group(1).strip()
        if line_answer:
            result["answer"] = line_answer.group(1).strip()
            result["answer_extraction_source"] = "ANSWER_line"
        if line_confidence:
            result["confidence"] = int(float(line_confidence.group(1)))
            result["confidence_extraction_source"] = "CONFIDENCE_line"
        if line_stop:
            result["stop"] = line_stop.group(1).lower() in {"yes", "true"}
            result["stop_extraction_source"] = "STOP_line"

    elif prompt_mode == "minimal_json":
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                parsed = json.loads(candidate)
                result["thought"] = str(parsed.get("thought", "")).strip()
                result["answer"] = str(parsed.get("answer", "")).strip()
                result["confidence"] = int(float(parsed.get("confidence", 50)))
                result["stop"] = bool(parsed.get("stop", False))
                result["parse_success"] = 1
                result["output_format_type"] = "json_exact"
                result["answer_extraction_source"] = "json_field"
                result["confidence_extraction_source"] = "json_field"
                result["stop_extraction_source"] = "json_field"
            except (json.JSONDecodeError, TypeError, ValueError):
                result["output_format_type"] = "json_partial"

    if not result["answer"]:
        extracted_answer, extraction_source = extract_answer(cleaned, answer_type)
        if extracted_answer:
            result["answer"] = extracted_answer
            result["answer_extraction_source"] = extraction_source

    if not result["thought"]:
        think_match = re.search(r"<think>(.*?)</think>", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if think_match:
            result["thought"] = think_match.group(1).strip()
        else:
            lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
            if lines:
                result["thought"] = lines[0]

    return result


def extract_gsm8k_reference_answer(answer_text: str) -> str:
    match = re.search(r"####\s*([^\n]+)", answer_text)
    if match:
        return normalize_answer(match.group(1), "number")
    return normalize_answer(answer_text, "number")


def load_gsm8k_tasks(max_tasks: int, dataset_split: str, shuffle_seed: int | None) -> list[TaskSpec]:
    if load_dataset is None:
        raise ImportError("datasets is required for --task-source gsm8k. Install it with pip install datasets evaluate bitsandbytes tqdm.")

    try:
        dataset = load_dataset("openai/gsm8k", "main", split=dataset_split)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download or load the GSM8K dataset from HuggingFace. "
            f"Please check your internet connection on the VM and verify that you can access HuggingFace. "
            f"Error details: {exc}"
        ) from exc
    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed)
    if max_tasks > 0 and len(dataset) > max_tasks:
        dataset = dataset.select(range(max_tasks))

    split_name = sanitize_split_name(dataset_split)
    tasks: list[TaskSpec] = []
    for index, example in enumerate(dataset):
        question = str(example["question"]).strip()
        expected_answer = extract_gsm8k_reference_answer(str(example["answer"]))
        short_hash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
        tasks.append(
            TaskSpec(
                task_id=f"gsm8k_{split_name}_{index:05d}_{short_hash}",
                domain="gsm8k",
                difficulty="grade_school_math",
                prompt=question,
                answer_type="number",
                expected_answer=expected_answer,
                notes=f"gsm8k main split={dataset_split}",
                source="gsm8k",
                source_index=index,
            )
        )
    return tasks


def load_math_tasks(max_tasks: int, dataset_split: str, shuffle_seed: int | None) -> list[TaskSpec]:
    """Competition MATH. Uses HuggingFaceH4/MATH-500 (500 problems, openly
    available); falls back to lighteval/MATH. Answers are graded with
    math_answers_equivalent (LaTeX-normalized + numeric + sympy fallback)."""
    if load_dataset is None:
        raise ImportError("datasets is required for --task-source math. Install it with pip install datasets.")
    split = dataset_split if dataset_split and dataset_split not in {"train", "main"} else "test"
    # AUDIT FIX: load ONLY the pinned corpus and FAIL HARD if unavailable. The
    # previous silent fallback to lighteval/MATH (~5000 problems) could swap the
    # whole corpus on a transient error, giving different models different
    # problems and silently breaking cross-model comparability.
    repo = "HuggingFaceH4/MATH-500"
    try:
        dataset = load_dataset(repo, split=split)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load the pinned MATH dataset {repo} (split={split}). Failing hard rather than "
            f"silently substituting a different corpus. Error details: {exc}"
        ) from exc
    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed)
    if max_tasks > 0 and len(dataset) > max_tasks:
        dataset = dataset.select(range(max_tasks))

    split_name = sanitize_split_name(split)
    tasks: list[TaskSpec] = []
    for index, example in enumerate(dataset):
        problem = str(example.get("problem", "")).strip()
        answer = example.get("answer")
        if answer is None or str(answer).strip() == "":
            boxed = re.findall(r"\\boxed\{(.+?)\}", str(example.get("solution", "")))
            answer = boxed[-1] if boxed else str(example.get("solution", ""))
        level = str(example.get("level", "")).strip()
        subject = str(example.get("subject", example.get("type", ""))).strip()
        short_hash = hashlib.md5(problem.encode("utf-8")).hexdigest()[:8]
        tasks.append(
            TaskSpec(
                task_id=f"math_{split_name}_{index:05d}_{short_hash}",
                domain="math",
                difficulty=(f"MATH {level}".strip() or "competition_math"),
                prompt=problem,
                answer_type="math",
                expected_answer=str(answer).strip(),
                notes=f"MATH repo={repo} split={split} subject={subject} level={level}",
                source="math",
                source_index=index,
            )
        )
    return tasks


def load_arc_tasks(max_tasks: int, dataset_split: str, shuffle_seed: int | None) -> list[TaskSpec]:
    """ARC-Challenge (allenai/ai2_arc, public). Multiple choice; all options are
    relabeled A,B,C(,D,E) and the gold answerKey is mapped to the new letter."""
    if load_dataset is None:
        raise ImportError("datasets is required for --task-source arc.")
    split = dataset_split if dataset_split and dataset_split not in {"train", "main"} else "test"
    try:
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    except Exception as exc:
        raise RuntimeError(f"Failed to load allenai/ai2_arc ARC-Challenge (split={split}). Error: {exc}") from exc
    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed)
    if max_tasks > 0 and len(dataset) > max_tasks:
        dataset = dataset.select(range(max_tasks))
    split_name = sanitize_split_name(split)
    tasks: list[TaskSpec] = []
    for index, example in enumerate(dataset):
        question = str(example["question"]).strip()
        texts = list(example["choices"]["text"])
        labels = [str(label) for label in example["choices"]["label"]]
        new_labels = [chr(ord("A") + j) for j in range(len(texts))]
        gold = dict(zip(labels, new_labels)).get(str(example["answerKey"]).strip(), "")
        if not gold or not texts:
            continue
        short_hash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
        tasks.append(TaskSpec(
            task_id=f"arc_{split_name}_{index:05d}_{short_hash}", domain="arc",
            difficulty="arc_challenge", prompt=_format_mcq_prompt(question, list(zip(new_labels, texts))),
            answer_type="mcq", expected_answer=gold, notes=f"ARC-Challenge split={split}",
            source="arc", source_index=index))
    return tasks


def load_gpqa_tasks(max_tasks: int, dataset_split: str, shuffle_seed: int | None) -> list[TaskSpec]:
    """GPQA main (Idavidrein/gpqa, GATED -- needs HF_TOKEN + accepted terms).
    The correct + 3 incorrect answers are deterministically shuffled per item."""
    if load_dataset is None:
        raise ImportError("datasets is required for --task-source gpqa.")
    try:
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
    except Exception as exc:
        raise RuntimeError(
            "Failed to load Idavidrein/gpqa (gpqa_main). It is GATED: accept the terms on the dataset "
            f"page and set HF_TOKEN. Error: {exc}") from exc
    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed)
    if max_tasks > 0 and len(dataset) > max_tasks:
        dataset = dataset.select(range(max_tasks))
    new_labels = ["A", "B", "C", "D"]
    tasks: list[TaskSpec] = []
    for index, example in enumerate(dataset):
        question = str(example["Question"]).strip()
        options = [str(example["Correct Answer"]).strip(),
                   str(example["Incorrect Answer 1"]).strip(),
                   str(example["Incorrect Answer 2"]).strip(),
                   str(example["Incorrect Answer 3"]).strip()]
        rng = random.Random((shuffle_seed if shuffle_seed is not None else 0) * 100003 + index)
        order = [0, 1, 2, 3]
        rng.shuffle(order)
        gold = new_labels[order.index(0)]  # where the correct option (index 0) landed
        labeled = list(zip(new_labels, [options[k] for k in order]))
        short_hash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
        tasks.append(TaskSpec(
            task_id=f"gpqa_main_{index:05d}_{short_hash}", domain="gpqa", difficulty="gpqa_main",
            prompt=_format_mcq_prompt(question, labeled), answer_type="mcq", expected_answer=gold,
            notes="GPQA main", source="gpqa", source_index=index))
    return tasks


def load_tasks(task_source: str, max_tasks: int, dataset_split: str, shuffle_seed: int | None) -> list[TaskSpec]:
    if task_source == "builtin":
        return BUILTIN_TASKS[:max_tasks]
    if task_source == "gsm8k":
        return load_gsm8k_tasks(max_tasks=max_tasks, dataset_split=dataset_split, shuffle_seed=shuffle_seed)
    if task_source == "math":
        return load_math_tasks(max_tasks=max_tasks, dataset_split=dataset_split, shuffle_seed=shuffle_seed)
    if task_source == "arc":
        return load_arc_tasks(max_tasks=max_tasks, dataset_split=dataset_split, shuffle_seed=shuffle_seed)
    if task_source == "gpqa":
        return load_gpqa_tasks(max_tasks=max_tasks, dataset_split=dataset_split, shuffle_seed=shuffle_seed)
    raise ValueError(f"Unsupported task source: {task_source}")


def run_id_for(model_alias: str, task: TaskSpec, temperature: float, seed: int) -> str:
    return f"{model_alias}__{task.task_id}__temp{temperature:.2f}__seed{seed}"


def chunked(items: list[TaskSpec], size: int) -> Iterable[list[TaskSpec]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def output_paths(output_dir: Path, is_baseline: bool) -> dict[str, Path]:
    steps_name = "baseline_steps.csv" if is_baseline else "trace_steps.csv"
    runs_name = "baseline_runs.csv" if is_baseline else "trace_runs.csv"
    batch_metrics_name = "baseline_batch_metrics.csv" if is_baseline else "batch_metrics.csv"
    return {
        "steps": output_dir / steps_name,
        "runs": output_dir / runs_name,
        "batch_metrics": output_dir / batch_metrics_name,
        "hazard": output_dir / "hazard_by_step.csv",
        "pilot": output_dir / "pilot_summary.csv",
        "metadata": output_dir / "metadata.json",
    }


def expected_steps_per_run(is_baseline: bool, max_steps: int) -> int:
    return 1 if is_baseline else max_steps


def _escape_control_chars(frame: pd.DataFrame) -> pd.DataFrame:
    # Control characters (\r, \f from decoded LaTeX like \rho, \frac) inside
    # text fields break CSV row framing when writes are interrupted
    # (rigor_audit/01 Axis 5a root cause). Tabs and newlines survive pandas
    # quoting; everything else in C0 gets replaced.
    pattern = re.compile(r"[\x00-\x08\x0b-\x1f]")
    for column in frame.columns:
        if frame[column].dtype == object or pd.api.types.is_string_dtype(frame[column]):
            frame[column] = frame[column].map(
                lambda value: pattern.sub(" ", value) if isinstance(value, str) else value
            )
    return frame


def append_records(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    frame = _escape_control_chars(pd.DataFrame(rows))
    if path.exists():
        existing_columns = pd.read_csv(path, nrows=0).columns.tolist()
        for column in existing_columns:
            if column not in frame.columns:
                frame[column] = np.nan
        extra_columns = [column for column in frame.columns if column not in existing_columns]
        frame = frame[existing_columns + extra_columns]
        frame.to_csv(path, mode="a", header=False, index=False)
        return
    frame.to_csv(path, index=False)


def max_nan(values: Iterable[float]) -> float:
    finite_values = [float(value) for value in values if not math.isnan(float(value))]
    if not finite_values:
        return float("nan")
    return max(finite_values)


def backup_for_reconciliation(path: Path) -> Path:
    backup_path = path.with_name(f"{path.stem}.pre_reconcile{path.suffix}")
    if path.exists() and not backup_path.exists():
        shutil.copy2(path, backup_path)
    return backup_path


def summarize_run_rows(run_rows: pd.DataFrame) -> dict[str, Any]:
    ordered = run_rows.sort_values("step").drop_duplicates(subset=["step"], keep="last").reset_index(drop=True)
    first_row = ordered.iloc[0]
    final_row = ordered.iloc[-1]
    first_correct = ordered.loc[ordered["correct"] == 1, "step"]
    first_model_stop = ordered.loc[ordered["model_stop_flag"] == 1, "step"]
    oracle_row = ordered.loc[ordered["utility"].idxmax()]

    return {
        "run_id": str(first_row["run_id"]),
        "model_alias": str(first_row["model_alias"]),
        "model_name": str(first_row["model_name"]),
        "task_id": str(first_row["task_id"]),
        "domain": str(first_row["domain"]),
        "difficulty": str(first_row["difficulty"]),
        "task_source": str(first_row["task_source"]),
        "task_source_index": int(first_row["task_source_index"]),
        "temperature": float(first_row["temperature"]),
        "seed": int(first_row["seed"]),
        "prompt_mode": str(first_row["prompt_mode"]),
        "system_prompt_mode": str(first_row["system_prompt_mode"]),
        "is_baseline": int(first_row["is_baseline"]),
        "ever_correct": int(ordered["correct"].max()),
        "correct_at_step_1": int(first_row["correct"]),
        "oracle_stop": int(oracle_row["step"]),
        "first_correct_step": int(first_correct.iloc[0]) if not first_correct.empty else -1,
        "first_model_stop_step": int(first_model_stop.iloc[0]) if not first_model_stop.empty else -1,
        "revision_count": int(ordered["answer_changed"].sum()),
        "best_utility": float(ordered["utility"].max()),
        "final_correct": int(final_row["correct"]),
        "device": str(first_row["device"]),
    }


def _drop_truncated_rows(frame: pd.DataFrame, key_cols: list[str]) -> tuple[pd.DataFrame, int]:
    """Drop rows whose key identifier columns are missing/unparseable -- the
    signature of an append-only CSV line truncated by a hard kill mid-write
    (e.g. the autostart watchdog SIGKILLing a hung run). Such a half-row would
    otherwise poison numeric columns (object dtype) or blow up int(step) on
    resume, crashing the whole cell. The affected run simply lacks a complete
    row set and gets re-collected, which is exactly what we want."""
    if frame.empty:
        return frame, 0
    before = len(frame)
    out = frame
    for col in key_cols:
        if col not in out.columns:
            continue
        out = out[out[col].notna()]
        if col == "step":  # must be a clean integer; garbled value -> drop the row
            out = out[pd.to_numeric(out[col], errors="coerce").notna()]
    out = out.reset_index(drop=True)
    return out, before - len(out)


def reconcile_existing_outputs(
    output_dir: Path,
    hidden_dir: Path,
    is_baseline: bool,
    max_steps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str], dict[str, Any]]:
    paths = output_paths(output_dir, is_baseline)
    expected_steps = expected_steps_per_run(is_baseline, max_steps)
    expected_step_sequence = list(range(1, expected_steps + 1))

    steps_frame = pd.read_csv(paths["steps"]) if paths["steps"].exists() else pd.DataFrame()
    runs_frame = pd.read_csv(paths["runs"]) if paths["runs"].exists() else pd.DataFrame()
    steps_frame, truncated_step_rows = _drop_truncated_rows(steps_frame, ["run_id", "step"])
    runs_frame, truncated_run_rows = _drop_truncated_rows(runs_frame, ["run_id"])
    if truncated_step_rows or truncated_run_rows:
        print(
            f"[reconcile] dropped {truncated_step_rows} truncated step row(s) and "
            f"{truncated_run_rows} truncated run row(s) from a prior hard kill; "
            "affected runs will be re-collected.",
            flush=True,
        )
    hidden_run_ids = {path.stem for path in hidden_dir.glob("*.npz")} if hidden_dir.exists() else set()

    duplicate_step_rows = 0
    duplicate_run_rows = 0
    if not steps_frame.empty:
        steps_frame = steps_frame.reset_index(drop=True)
        steps_frame["_row_order"] = np.arange(len(steps_frame))
        duplicate_step_rows = int(steps_frame.duplicated(subset=["run_id", "step"], keep=False).sum())
        steps_frame = steps_frame.sort_values("_row_order").drop_duplicates(subset=["run_id", "step"], keep="last")
    if not runs_frame.empty:
        runs_frame = runs_frame.reset_index(drop=True)
        runs_frame["_row_order"] = np.arange(len(runs_frame))
        duplicate_run_rows = int(runs_frame.duplicated(subset=["run_id"], keep=False).sum())
        runs_frame = runs_frame.sort_values("_row_order").drop_duplicates(subset=["run_id"], keep="last")

    completed_from_steps: set[str] = set()
    incomplete_step_run_ids: list[str] = []
    reconstructed_runs: list[dict[str, Any]] = []
    existing_run_ids = set(runs_frame["run_id"].astype(str)) if not runs_frame.empty else set()

    if not steps_frame.empty:
        for run_id, group in steps_frame.groupby("run_id", sort=False):
            run_id_str = str(run_id)
            ordered = group.sort_values("step")
            observed_steps = [int(value) for value in ordered["step"].tolist()]
            is_complete = observed_steps == expected_step_sequence
            if is_complete:
                completed_from_steps.add(run_id_str)
                if run_id_str not in existing_run_ids:
                    reconstructed_runs.append(summarize_run_rows(ordered))
            else:
                incomplete_step_run_ids.append(run_id_str)

    runs_without_complete_steps = sorted(existing_run_ids - completed_from_steps)
    hidden_without_complete_steps = sorted(hidden_run_ids - completed_from_steps)
    completed_run_ids = set(completed_from_steps)

    if not steps_frame.empty:
        sanitized_steps = steps_frame[steps_frame["run_id"].astype(str).isin(completed_run_ids)].copy()
        sanitized_steps = sanitized_steps.sort_values(["run_id", "step", "_row_order"])
        sanitized_steps = sanitized_steps.drop(columns=["_row_order"], errors="ignore")
    else:
        sanitized_steps = steps_frame

    sanitized_runs_frames: list[pd.DataFrame] = []
    if not runs_frame.empty:
        sanitized_runs_frames.append(runs_frame[runs_frame["run_id"].astype(str).isin(completed_run_ids)].copy())
    if reconstructed_runs:
        sanitized_runs_frames.append(pd.DataFrame(reconstructed_runs))
    if sanitized_runs_frames:
        sanitized_runs = pd.concat(sanitized_runs_frames, ignore_index=True, sort=False)
        sanitized_runs["task_source_index"] = pd.to_numeric(sanitized_runs["task_source_index"], errors="coerce").fillna(-1).astype(int)
        sanitized_runs["temperature"] = pd.to_numeric(sanitized_runs["temperature"], errors="coerce")
        sanitized_runs["seed"] = pd.to_numeric(sanitized_runs["seed"], errors="coerce").fillna(-1).astype(int)
        sanitized_runs = sanitized_runs.sort_values(["temperature", "seed", "task_source_index", "run_id"])
        sanitized_runs = sanitized_runs.drop(columns=["_row_order"], errors="ignore")
    else:
        sanitized_runs = pd.DataFrame(columns=runs_frame.columns.drop("_row_order", errors="ignore"))

    anomalies_detected = any(
        [
            duplicate_step_rows,
            duplicate_run_rows,
            reconstructed_runs,
            incomplete_step_run_ids,
            runs_without_complete_steps,
            hidden_without_complete_steps,
        ]
    )
    reconciliation_report = {
        "expected_steps_per_run": expected_steps,
        "completed_run_count": len(completed_run_ids),
        "hidden_state_file_count": len(hidden_run_ids),
        "duplicate_step_rows_removed": duplicate_step_rows,
        "duplicate_run_rows_removed": duplicate_run_rows,
        "reconstructed_run_summaries": len(reconstructed_runs),
        "incomplete_step_run_ids": incomplete_step_run_ids,
        "runs_without_complete_steps": runs_without_complete_steps,
        "hidden_without_complete_steps": hidden_without_complete_steps,
        "anomalies_detected": bool(anomalies_detected),
    }
    if anomalies_detected:
        if paths["steps"].exists():
            backup_for_reconciliation(paths["steps"])
        if paths["runs"].exists():
            backup_for_reconciliation(paths["runs"])
        sanitized_steps.to_csv(paths["steps"], index=False)
        sanitized_runs.to_csv(paths["runs"], index=False)
        with (output_dir / "checkpoint_reconciliation.json").open("w", encoding="utf-8") as handle:
            json.dump(reconciliation_report, handle, indent=2)

    return sanitized_steps.to_dict("records"), sanitized_runs.to_dict("records"), completed_run_ids, reconciliation_report


def load_existing_outputs(output_dir: Path, is_baseline: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    paths = output_paths(output_dir, is_baseline)
    if paths["steps"].exists():
        existing_steps = pd.read_csv(paths["steps"]).to_dict("records")
    else:
        existing_steps = []
    if paths["runs"].exists():
        existing_runs = pd.read_csv(paths["runs"]).to_dict("records")
    else:
        existing_runs = []
    completed_run_ids = {str(row["run_id"]) for row in existing_runs}
    return existing_steps, existing_runs, completed_run_ids


def load_model(
    model_spec: ModelSpec,
    device: str,
    quantization: str,
    device_map: str | None,
    attn_implementation: str,
    model_path_override: str | None = None,
    offload_folder: str | None = None,
) -> tuple[Any, Any, str, str]:
    actual_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model_source = model_path_override or model_spec.hf_name
    
    # Precision Auto-Tuning for L4
    target_precision = quantization
    if quantization == "auto" and actual_device == "cuda":
        # Extract numeric parameter count
        try:
            num_params = float(re.findall(r"[\d.]+", model_spec.parameter_count)[0])
            # If < 12B, use bfloat16 for speed on L4 24GB.
            target_precision = "bf16" if num_params < 12 else "4bit"
        except:
            target_precision = "4bit"
            
    backend = "transformers+torch(cpu)"
    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    compute_dtype = torch.bfloat16 if actual_device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    dtype_label = "bfloat16" if compute_dtype == torch.bfloat16 else "float16"
    logging.info("Precision auto-tuning: %s detected for %s model.", dtype_label, model_spec.parameter_count)
    
    if actual_device == "cuda":
        load_kwargs["torch_dtype"] = compute_dtype
        # Flash Attention 2 Enforcement
        if attn_implementation == "auto" or attn_implementation == "flash_attention_2":
            load_kwargs["attn_implementation"] = "flash_attention_2"
            logging.info("Flash Attention 2 explicitly enabled.")
        else:
            load_kwargs["attn_implementation"] = attn_implementation
            logging.info("Attention implementation: %s (native FA2 kernels via SDPA).", attn_implementation)

    _MULTIMODAL_FAMILIES = ["gemma 4", "llama 4", "qwen3.5"]
    _TEXT_ONLY_OVERRIDE = "text-only" in model_spec.family.lower()
    is_multimodal = (
        not _TEXT_ONLY_OVERRIDE
        and any(f in model_spec.family.lower() for f in _MULTIMODAL_FAMILIES)
    )
    if is_multimodal:
        from transformers import AutoModelForMultimodalLM
        model_class = AutoModelForMultimodalLM
    else:
        model_class = AutoModelForCausalLM
    
    # Loading handle (Processor for multimodal, Tokenizer for vanilla)
    if is_multimodal:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_source, trust_remote_code=True)
        # Use the inner tokenizer for text encoding; it also has apply_chat_template
        tokenizer = getattr(processor, "tokenizer", processor)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    resolved_device_map = resolve_low_vram_device_map(
        model_spec=model_spec,
        actual_device=actual_device,
        target_precision=target_precision,
        device_map=device_map,
    )
    if resolved_device_map is not None and resolved_device_map != device_map:
        logging.info("Using explicit low-VRAM device map for %s: %s", model_spec.alias, resolved_device_map)

    try:
        if actual_device == "cuda" and target_precision in {"8bit", "4bit"}:
            if BitsAndBytesConfig is None:
                raise ImportError("bitsandbytes support is unavailable in the installed transformers stack.")
            if target_precision == "4bit":
                ensure_bitsandbytes_params4bit_compatibility()
                ensure_accelerate_meta_state_dict_compatibility()
            if target_precision == "8bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                )
            load_kwargs["device_map"] = resolved_device_map or "auto"
            if offload_folder:
                load_kwargs["offload_folder"] = offload_folder
                load_kwargs["offload_state_dict"] = True
            try:
                model = model_class.from_pretrained(model_source, **load_kwargs)
            except ValueError as ve:
                if "flash_attention_2" in str(ve):
                    logging.warning("Flash Attention 2 not supported for this model. Falling back to SDPA.")
                    load_kwargs["attn_implementation"] = "sdpa"
                    model = model_class.from_pretrained(model_source, **load_kwargs)
                else:
                    raise
            backend = f"transformers+torch(cuda-{target_precision})"
        elif actual_device == "cuda" and resolved_device_map:
            load_kwargs["device_map"] = resolved_device_map
            if offload_folder:
                load_kwargs["offload_folder"] = offload_folder
                load_kwargs["offload_state_dict"] = True
            model = model_class.from_pretrained(model_source, **load_kwargs)
            backend = f"transformers+torch(cuda-device-map={resolved_device_map})"
        else:
            model = model_class.from_pretrained(model_source, **load_kwargs)
            model.to(actual_device)
            backend = f"transformers+torch({actual_device})"
    except Exception as exc:
        logging.warning("Primary model load failed for %s: %s", model_source, exc)
        if (
            actual_device == "cuda"
            and target_precision == "4bit"
            and "Some modules are dispatched on the CPU or the disk" in str(exc)
        ):
            logging.warning(
                "4-bit loading requested CPU offload but %s still does not fit cleanly on this GPU. "
                "This usually means the card is too small for the model family even with quantization.",
                model_source,
            )
        if actual_device == "cuda":
            raise
        fallback_kwargs = {"trust_remote_code": True}
        model = AutoModelForCausalLM.from_pretrained(model_source, **fallback_kwargs)
        model.to("cpu")
        actual_device = "cpu"
        backend = "transformers+torch(cpu-fallback)"

    model.eval()
    return model, tokenizer, actual_device, backend


def trim_completion_ids(completion_ids: torch.Tensor, pad_token_id: int | None, eos_token_id: int | None) -> torch.Tensor:
    trimmed = completion_ids
    trim_token_ids = {token_id for token_id in (pad_token_id, eos_token_id) if token_id is not None}
    while len(trimmed) > 0 and int(trimmed[-1].item()) in trim_token_ids:
        trimmed = trimmed[:-1]
    return trimmed


def apply_output_head_with_model_postprocessing(model: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    output_embeddings = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if output_embeddings is None:
        raise AttributeError("Model does not expose output embeddings for low-memory scoring.")

    logits = output_embeddings(hidden_states)
    config = getattr(model, "config", None)
    if config is None:
        return logits

    text_config_getter = getattr(config, "get_text_config", None)
    text_config = text_config_getter() if callable(text_config_getter) else config
    final_logit_softcapping = getattr(text_config, "final_logit_softcapping", None)
    if final_logit_softcapping is not None:
        logits = logits / final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * final_logit_softcapping
    return logits


def low_memory_scoring_pass(
    model: Any,
    generated_ids: torch.Tensor,
    generated_attention_mask: torch.Tensor,
    prompt_width: int,
    completion_width: int,
    output_hidden_states: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...] | None]:
    base_model = getattr(model, "model", None)
    if base_model is None:
        base_model = getattr(model, "base_model", None)
    if base_model is None:
        raise AttributeError("Model does not expose a base model for low-memory scoring.")

    base_outputs = base_model(
        input_ids=generated_ids,
        attention_mask=generated_attention_mask,
        use_cache=False,
        return_dict=True,
        output_hidden_states=output_hidden_states,
    )
    last_hidden_state = getattr(base_outputs, "last_hidden_state", None)
    if last_hidden_state is None:
        raise AttributeError("Base model did not return last_hidden_state for low-memory scoring.")

    scoring_start = prompt_width - 1
    scoring_stop = scoring_start + completion_width
    scoring_hidden_states = last_hidden_state[:, scoring_start:scoring_stop, :]
    scoring_logits = apply_output_head_with_model_postprocessing(model, scoring_hidden_states)
    
    all_hidden_states = getattr(base_outputs, "hidden_states", None)
    return scoring_logits, last_hidden_state, all_hidden_states


_PROJECTION_MATRIX_CACHE = {}

def project_hidden_features(hidden_vector: Any, target_dim: int = 64, seed: int = 42) -> np.ndarray:
    """Projects a hidden state vector/tensor to a lower dimension using a cached CPU projection matrix."""
    if isinstance(hidden_vector, torch.Tensor):
        hidden_vector = hidden_vector.float().cpu().numpy()
        
    D = hidden_vector.shape[0]
    cache_key = (D, target_dim, seed)
    
    if cache_key not in _PROJECTION_MATRIX_CACHE:
        rng = np.random.RandomState(seed)
        _PROJECTION_MATRIX_CACHE[cache_key] = rng.normal(size=(D, target_dim)) / np.sqrt(target_dim)
        
    proj_matrix = _PROJECTION_MATRIX_CACHE[cache_key]
    return np.dot(hidden_vector, proj_matrix)


def find_answer_token_indices(completion_ids: torch.Tensor, raw_text: str, answer_str: str, prompt_mode: str, tokenizer: Any) -> list[int]:
    """Finds the indices of tokens in completion_ids that correspond to the answer_str."""
    if not answer_str:
        return []
        
    char_start = -1
    if prompt_mode == "structured_four_line":
        # Search for ANSWER: <answer_str>
        match = re.search(r"ANSWER\s*:\s*" + re.escape(answer_str), raw_text, re.IGNORECASE)
        if match:
            char_start = match.end(0) - len(answer_str)
    elif prompt_mode == "minimal_json":
        # Search for "answer"\s*:\s*"<answer_str>"
        match = re.search(r'"answer"\s*:\s*"' + re.escape(answer_str) + r'"', raw_text)
        if match:
            char_start = match.end(0) - len(answer_str) - 1 # exclude closing quote
            
    if char_start == -1:
        char_start = raw_text.rfind(answer_str)
        
    if char_start == -1:
        return []
        
    char_end = char_start + len(answer_str)
    
    token_indices = []
    for idx in range(len(completion_ids)):
        prefix_text_prev = tokenizer.decode(completion_ids[:idx], skip_special_tokens=True)
        prefix_text_curr = tokenizer.decode(completion_ids[:idx+1], skip_special_tokens=True)
        tok_start = len(prefix_text_prev)
        tok_end = len(prefix_text_curr)
        # Check overlap
        if max(tok_start, char_start) < min(tok_end, char_end):
            token_indices.append(idx)
    return token_indices


def generate_batch_with_diagnostics(
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    actual_device: str,
    temperature: float,
    max_new_tokens: int,
    answer_types: list[str] | None = None,
    prompt_mode: str = "structured_four_line",
    enable_extended_observables: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if actual_device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    started_at = time.perf_counter()
    encode_started_at = time.perf_counter()
    encoded = tokenizer(
        prompt_texts,
        padding=True,
        pad_to_multiple_of=8 if actual_device == "cuda" else None,
        return_tensors="pt",
    )
    if actual_device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    encode_seconds = time.perf_counter() - encode_started_at

    prompt_tokens = int(encoded["attention_mask"].sum().item())
    input_ids = encoded["input_ids"].to(actual_device)
    attention_mask = encoded["attention_mask"].to(actual_device)
    prompt_width = input_ids.shape[1]

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.02,
    }
    if temperature > 0:
        generation_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.95})
    else:
        generation_kwargs.update({"do_sample": False})

    with torch.inference_mode():
        generation_started_at = time.perf_counter()
        generated_ids = model.generate(**generation_kwargs)
        if actual_device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - generation_started_at

        forward_started_at = time.perf_counter()
        completion_width = generated_ids.shape[1] - prompt_width
        generated_attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((attention_mask.shape[0], completion_width), dtype=attention_mask.dtype, device=attention_mask.device),
            ],
            dim=1,
        )
        try:
            scoring_logits, final_hidden_states, all_hidden_states = low_memory_scoring_pass(
                model=model,
                generated_ids=generated_ids,
                generated_attention_mask=generated_attention_mask,
                prompt_width=prompt_width,
                completion_width=completion_width,
                output_hidden_states=enable_extended_observables,
            )
        except Exception as exc:
            logging.warning("Low-memory scoring path failed; falling back to wrapper forward: %s", exc)
            forward_outputs = model(
                generated_ids,
                attention_mask=generated_attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            scoring_logits = forward_outputs.logits[:, prompt_width - 1 : prompt_width - 1 + completion_width, :]
            final_hidden_states = forward_outputs.hidden_states[-1]
            all_hidden_states = forward_outputs.hidden_states
        if actual_device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - forward_started_at

    # Postprocess on CPU to avoid expanding completion logits into softmax buffers on a constrained GPU.
    generated_ids_cpu = generated_ids.detach().cpu()
    scoring_logits = scoring_logits.float().cpu()
    final_hidden_states = final_hidden_states.float().cpu()
    
    # Perform GPU-side mean pooling BEFORE clearing the cache.
    # This reduces the CPU copy size from 1 GB (entire layer) to 1 MB (mean pooled vector).
    L1_pooled_list = []
    L2_pooled_list = []
    if enable_extended_observables and all_hidden_states is not None:
        num_layers = len(all_hidden_states) - 1
        L1_idx = num_layers // 3
        L2_idx = 2 * num_layers // 3
        L1_layer = all_hidden_states[L1_idx + 1]
        L2_layer = all_hidden_states[L2_idx + 1]
        
        batch_size = scoring_logits.shape[0]
        for index in range(batch_size):
            raw_completion_ids = generated_ids_cpu[index, prompt_width:]
            completion_ids = trim_completion_ids(raw_completion_ids, tokenizer.pad_token_id, tokenizer.eos_token_id)
            generated_length = int(completion_ids.shape[0])
            
            if generated_length > 0:
                L1_pooled = L1_layer[index, prompt_width : prompt_width + generated_length, :].mean(dim=0).float().cpu().numpy()
                L2_pooled = L2_layer[index, prompt_width : prompt_width + generated_length, :].mean(dim=0).float().cpu().numpy()
            else:
                L1_pooled = np.zeros((model.config.hidden_size,), dtype=np.float32)
                L2_pooled = np.zeros((model.config.hidden_size,), dtype=np.float32)
                
            L1_pooled_list.append(L1_pooled)
            L2_pooled_list.append(L2_pooled)
            
    if actual_device == "cuda":
        del input_ids
        del attention_mask
        del generated_ids
        del generated_attention_mask
        if "all_hidden_states" in locals():
            del all_hidden_states
        release_cuda_memory()

    results: list[dict[str, Any]] = []
    postprocess_started_at = time.perf_counter()
    batch_size = scoring_logits.shape[0]
    for index in range(batch_size):
        raw_completion_ids = generated_ids_cpu[index, prompt_width:]
        completion_ids = trim_completion_ids(raw_completion_ids, tokenizer.pad_token_id, tokenizer.eos_token_id)
        generated_length = int(completion_ids.shape[0])
        raw_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        # Default extended observables
        answer_span_mean_logprob = float("nan")
        answer_span_min_logprob = float("nan")
        answer_span_mean_entropy = float("nan")
        answer_span_std_entropy = 0.0
        mid_hidden_1_proj_str = ""
        mid_hidden_2_proj_str = ""

        if generated_length > 0:
            completion_scoring_logits = scoring_logits[index, :generated_length]
            log_probs = torch.log_softmax(completion_scoring_logits, dim=-1)
            probs = torch.softmax(completion_scoring_logits, dim=-1)
            token_logprobs = log_probs.gather(1, completion_ids.unsqueeze(1)).squeeze(1)
            token_entropies = -(probs * log_probs).sum(dim=-1)
            pooled_hidden = final_hidden_states[index, prompt_width : prompt_width + generated_length, :].mean(dim=0).float().cpu().numpy()
            
            if enable_extended_observables and answer_types is not None:
                # N8a: Answer-span token diagnostics
                ans_type = answer_types[index]
                parsed = parse_generation(raw_text, ans_type, prompt_mode)
                ans_str = parsed["answer"]
                ans_tokens = find_answer_token_indices(completion_ids, raw_text, ans_str, prompt_mode, tokenizer)
                
                if ans_tokens:
                    ans_logprobs = token_logprobs[ans_tokens]
                    ans_entropies = token_entropies[ans_tokens]
                    
                    answer_span_mean_logprob = float(ans_logprobs.mean().item())
                    answer_span_min_logprob = float(ans_logprobs.min().item())
                    answer_span_mean_entropy = float(ans_entropies.mean().item())
                    answer_span_std_entropy = float(ans_entropies.std().item()) if len(ans_entropies) > 1 else 0.0
                
                # N8b: Pooled mid-layer hidden states projection
                if L1_pooled_list and L2_pooled_list:
                    L1_tensor = L1_pooled_list[index]
                    L2_tensor = L2_pooled_list[index]
                    
                    L1_proj = project_hidden_features(L1_tensor, target_dim=64, seed=42)
                    L2_proj = project_hidden_features(L2_tensor, target_dim=64, seed=42)
                    
                    mid_hidden_1_proj_str = ",".join(f"{x:.6f}" for x in L1_proj)
                    mid_hidden_2_proj_str = ",".join(f"{x:.6f}" for x in L2_proj)
        else:
            token_logprobs = torch.empty(0)
            token_entropies = torch.empty(0)
            pooled_hidden = np.zeros((model.config.hidden_size,), dtype=np.float32)

        results.append(
            {
                "raw_text": raw_text,
                "generated_tokens": generated_length,
                "mean_token_logprob": float(token_logprobs.mean().item()) if len(token_logprobs) else float("nan"),
                "mean_entropy": float(token_entropies.mean().item()) if len(token_entropies) else float("nan"),
                "entropy_std": float(token_entropies.std().item()) if len(token_entropies) > 1 else 0.0,
                "pooled_hidden": pooled_hidden,
                "answer_span_mean_logprob": answer_span_mean_logprob,
                "answer_span_min_logprob": answer_span_min_logprob,
                "answer_span_mean_entropy": answer_span_mean_entropy,
                "answer_span_std_entropy": answer_span_std_entropy,
                "mid_hidden_1_proj": mid_hidden_1_proj_str,
                "mid_hidden_2_proj": mid_hidden_2_proj_str,
            }
        )
    if actual_device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    postprocess_seconds = time.perf_counter() - postprocess_started_at
    elapsed_seconds = time.perf_counter() - started_at
    generated_tokens = int(sum(item["generated_tokens"] for item in results))
    batch_metrics = {
        "batch_size": len(prompt_texts),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_seconds": elapsed_seconds,
        "tokenize_seconds": encode_seconds,
        "generation_seconds": generation_seconds,
        "forward_seconds": forward_seconds,
        "postprocess_seconds": postprocess_seconds,
        "gpu_memory_allocated_gb": gpu_memory_allocated_gb(actual_device),
        "gpu_memory_reserved_gb": gpu_memory_reserved_gb(actual_device),
        "gpu_max_memory_allocated_gb": gpu_max_memory_allocated_gb(actual_device),
        "gpu_max_memory_reserved_gb": gpu_max_memory_reserved_gb(actual_device),
        "split_count": 1,
        "oom_retry_count": 0,
    }
    return results, batch_metrics


def release_cuda_memory() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()


def safe_generate_batch_with_diagnostics(
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    actual_device: str,
    temperature: float,
    max_new_tokens: int,
    allow_single_retry: bool = True,
    answer_types: list[str] | None = None,
    prompt_mode: str = "structured_four_line",
    enable_extended_observables: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    oom_occurred = False
    try:
        return generate_batch_with_diagnostics(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            actual_device=actual_device,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            answer_types=answer_types,
            prompt_mode=prompt_mode,
            enable_extended_observables=enable_extended_observables,
        )
    except RuntimeError as exc:
        if actual_device != "cuda" or "out of memory" not in str(exc).lower():
            raise
        oom_occurred = True

    if oom_occurred:
        import gc
        release_cuda_memory()
        gc.collect()

        if len(prompt_texts) == 1:
            if not allow_single_retry:
                raise RuntimeError("CUDA out of memory for single prompt; retry disabled or already attempted.")
            logging.warning("CUDA OOM for single prompt. Retrying once after cache release.")
            retry_results, retry_metrics = safe_generate_batch_with_diagnostics(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=prompt_texts,
                actual_device=actual_device,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                allow_single_retry=False,
                answer_types=answer_types,
                prompt_mode=prompt_mode,
                enable_extended_observables=enable_extended_observables,
            )
            retry_metrics["oom_retry_count"] = int(retry_metrics.get("oom_retry_count", 0)) + 1
            return retry_results, retry_metrics

        logging.warning("CUDA OOM for microbatch size %d. Retrying with smaller splits.", len(prompt_texts))
        midpoint = max(1, len(prompt_texts) // 2)
        first_results, first_metrics = safe_generate_batch_with_diagnostics(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=prompt_texts[:midpoint],
            actual_device=actual_device,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            allow_single_retry=allow_single_retry,
            answer_types=answer_types[:midpoint] if answer_types else None,
            prompt_mode=prompt_mode,
            enable_extended_observables=enable_extended_observables,
        )
        release_cuda_memory()
        gc.collect()

        second_results, second_metrics = safe_generate_batch_with_diagnostics(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=prompt_texts[midpoint:],
            actual_device=actual_device,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            allow_single_retry=allow_single_retry,
            answer_types=answer_types[midpoint:] if answer_types else None,
            prompt_mode=prompt_mode,
            enable_extended_observables=enable_extended_observables,
        )
        merged_metrics = {
            "batch_size": int(first_metrics["batch_size"]) + int(second_metrics["batch_size"]),
            "prompt_tokens": int(first_metrics["prompt_tokens"]) + int(second_metrics["prompt_tokens"]),
            "generated_tokens": int(first_metrics["generated_tokens"]) + int(second_metrics["generated_tokens"]),
            "total_seconds": float(first_metrics["total_seconds"]) + float(second_metrics["total_seconds"]),
            "tokenize_seconds": float(first_metrics["tokenize_seconds"]) + float(second_metrics["tokenize_seconds"]),
            "generation_seconds": float(first_metrics["generation_seconds"]) + float(second_metrics["generation_seconds"]),
            "forward_seconds": float(first_metrics["forward_seconds"]) + float(second_metrics["forward_seconds"]),
            "postprocess_seconds": float(first_metrics["postprocess_seconds"]) + float(second_metrics["postprocess_seconds"]),
            "gpu_memory_allocated_gb": max_nan([
                float(first_metrics["gpu_memory_allocated_gb"]),
                float(second_metrics["gpu_memory_allocated_gb"]),
            ]),
            "gpu_memory_reserved_gb": max_nan([
                float(first_metrics["gpu_memory_reserved_gb"]),
                float(second_metrics["gpu_memory_reserved_gb"]),
            ]),
            "gpu_max_memory_allocated_gb": max_nan([
                float(first_metrics["gpu_max_memory_allocated_gb"]),
                float(second_metrics["gpu_max_memory_allocated_gb"]),
            ]),
            "gpu_max_memory_reserved_gb": max_nan([
                float(first_metrics["gpu_max_memory_reserved_gb"]),
                float(second_metrics["gpu_max_memory_reserved_gb"]),
            ]),
            "split_count": int(first_metrics.get("split_count", 1)) + int(second_metrics.get("split_count", 1)),
            "oom_retry_count": int(first_metrics.get("oom_retry_count", 0)) + int(second_metrics.get("oom_retry_count", 0)) + 1,
        }
        return first_results + second_results, merged_metrics



def gpu_memory_allocated_gb(actual_device: str) -> float:
    if actual_device != "cuda" or not torch.cuda.is_available():
        return float("nan")
    return torch.cuda.memory_allocated() / (1024**3)


def gpu_memory_reserved_gb(actual_device: str) -> float:
    if actual_device != "cuda" or not torch.cuda.is_available():
        return float("nan")
    return torch.cuda.memory_reserved() / (1024**3)


def gpu_max_memory_allocated_gb(actual_device: str) -> float:
    if actual_device != "cuda" or not torch.cuda.is_available():
        return float("nan")
    return torch.cuda.max_memory_allocated() / (1024**3)


def gpu_max_memory_reserved_gb(actual_device: str) -> float:
    if actual_device != "cuda" or not torch.cuda.is_available():
        return float("nan")
    return torch.cuda.max_memory_reserved() / (1024**3)


def first_pending_run(
    model_spec: ModelSpec,
    tasks: list[TaskSpec],
    temperatures: list[float],
    seeds: list[int],
    completed_run_ids: set[str],
) -> dict[str, Any] | None:
    for temperature in temperatures:
        for seed in seeds:
            for task in tasks:
                run_id = run_id_for(model_spec.alias, task, temperature, seed)
                if run_id in completed_run_ids:
                    continue
                return {
                    "run_id": run_id,
                    "task_id": task.task_id,
                    "task_source_index": task.source_index,
                    "temperature": temperature,
                    "seed": seed,
                }
    return None


def write_runtime_metadata(
    *,
    metadata_path: Path,
    model_spec: ModelSpec,
    backend: str,
    actual_device: str,
    quantization: str,
    device_map: str | None,
    attn_implementation: str,
    temperatures: list[float],
    seeds: list[int],
    max_steps: int,
    max_new_tokens: int,
    max_tasks: int,
    task_source: str,
    dataset_split: str,
    dataset_shuffle_seed: int,
    batch_size: int,
    prompt_mode: str,
    system_prompt_mode: str,
    resume_enabled: bool,
    completed_run_ids: set[str],
    pending_run_count: int,
    next_pending_run: dict[str, Any] | None,
    reconciliation_report: dict[str, Any],
    tasks: list[TaskSpec],
    model_source: str | None,
) -> None:
    metadata = {
        "model": asdict(model_spec),
        "backend": backend,
        "model_source": model_source or model_spec.hf_name,
        "device": actual_device,
        "quantization": quantization,
        "device_map": device_map,
        "attn_implementation": attn_implementation,
        "temperatures": temperatures,
        "seeds": seeds,
        "max_steps": max_steps,
        "max_new_tokens": max_new_tokens,
        "max_tasks": max_tasks,
        "task_source": task_source,
        "dataset_split": dataset_split,
        "dataset_shuffle_seed": dataset_shuffle_seed,
        "batch_size": batch_size,
        "step_cost": STEP_COST,
        "prompt_mode": prompt_mode,
        "system_prompt_mode": system_prompt_mode,
        "resume_enabled": resume_enabled,
        "completed_run_count": len(completed_run_ids),
        "pending_run_count": pending_run_count,
        "next_pending_run": next_pending_run,
        "checkpoint_reconciliation": reconciliation_report,
        "tasks": [asdict(task) for task in tasks],
        "hidden_state_accessible": True,
        "token_logprobs_accessible": True,
        "reasoning_traces_accessible": True,
        "batch_metrics_accessible": True,
        "runtime_limit_notes": "Batch-first GSM8K runner with checkpoint reconciliation, append-only CSV outputs, incremental metadata snapshots, and automatic microbatch splitting on CUDA OOM.",
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def run_batch_traces(
    model: Any,
    tokenizer: Any,
    model_spec: ModelSpec,
    tasks: list[TaskSpec],
    actual_device: str,
    temperature: float,
    seed: int,
    max_steps: int,
    max_new_tokens: int,
    prompt_mode: str,
    system_prompt_mode: str,
    is_baseline: bool,
    hidden_dir: Path,
    io_manager: IOManager | None = None,
    enable_k2_agreement: bool = False,
    enable_extended_observables: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not tasks:
        return [], [], []

    actual_prompt_mode = "answer_only" if is_baseline else prompt_mode
    steps_to_run = 1 if is_baseline else max_steps
    batch_metric_rows: list[dict[str, Any]] = []
    contexts = [
        {
            "task": task,
            "run_id": run_id_for(model_spec.alias, task, temperature, seed),
            "history": [],
            "hidden_vectors": [],
            "rows": [],
        }
        for task in tasks
    ]

    for step in range(1, steps_to_run + 1):
        prompts = [
            render_prompt(
                tokenizer,
                conversation_prompt(
                    task=context["task"],
                    history=context["history"],
                    step=step,
                    max_steps=max_steps,
                    prompt_mode=actual_prompt_mode,
                ),
                system_prompt_mode,
            )
            for context in contexts
        ]

        # First path generation (main trajectory)
        generated_batch, batch_metrics = safe_generate_batch_with_diagnostics(
            model=model,
            tokenizer=tokenizer,
            prompt_texts=prompts,
            actual_device=actual_device,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            answer_types=[context["task"].answer_type for context in contexts],
            prompt_mode=actual_prompt_mode,
            enable_extended_observables=enable_extended_observables,
        )
        
        # Second path generation (N7: k=2 agreement)
        k2_batch = None
        if enable_k2_agreement and not is_baseline:
            k2_batch, _ = safe_generate_batch_with_diagnostics(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=prompts,
                actual_device=actual_device,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                answer_types=None,
                prompt_mode=actual_prompt_mode,
                enable_extended_observables=False,
            )

        batch_elapsed_seconds = float(batch_metrics["total_seconds"])
        batch_tokens = sum(item["generated_tokens"] for item in generated_batch)
        if k2_batch is not None:
            batch_tokens += sum(item["generated_tokens"] for item in k2_batch)
            
        batch_examples_per_second = len(contexts) / max(batch_elapsed_seconds, 1e-6)
        batch_tokens_per_second = batch_tokens / max(batch_elapsed_seconds, 1e-6)
        batch_metric_rows.append(
            {
                "phase": "generate",
                "temperature": temperature,
                "seed": seed,
                "step": step,
                "requested_batch_size": len(contexts),
                "realized_batch_size": int(batch_metrics["batch_size"]),
                "prompt_tokens": int(batch_metrics["prompt_tokens"]),
                "generated_tokens": batch_tokens,
                "wall_clock_seconds": batch_elapsed_seconds,
                "examples_per_second": batch_examples_per_second,
                "tokens_per_second": batch_tokens_per_second,
                "tokenize_seconds": float(batch_metrics["tokenize_seconds"]),
                "generation_seconds": float(batch_metrics["generation_seconds"]),
                "forward_seconds": float(batch_metrics["forward_seconds"]),
                "postprocess_seconds": float(batch_metrics["postprocess_seconds"]),
                "hidden_state_write_seconds": 0.0,
                "gpu_memory_allocated_gb": float(batch_metrics["gpu_memory_allocated_gb"]),
                "gpu_memory_reserved_gb": float(batch_metrics["gpu_memory_reserved_gb"]),
                "gpu_max_memory_allocated_gb": float(batch_metrics["gpu_max_memory_allocated_gb"]),
                "gpu_max_memory_reserved_gb": float(batch_metrics["gpu_max_memory_reserved_gb"]),
                "split_count": int(batch_metrics.get("split_count", 1)),
                "oom_retry_count": int(batch_metrics.get("oom_retry_count", 0)),
            }
        )
        logging.info(
            "temp=%.2f seed=%d step=%d/%d | batch=%d | generated_tokens=%d | ex_s=%.2f | tok_s=%.2f | gpu_peak_alloc_gb=%.2f | gpu_peak_reserved_gb=%.2f | splits=%d | oom_retries=%d",
            temperature,
            seed,
            step,
            steps_to_run,
            len(contexts),
            batch_tokens,
            batch_examples_per_second,
            batch_tokens_per_second,
            float(batch_metrics["gpu_max_memory_allocated_gb"]),
            float(batch_metrics["gpu_max_memory_reserved_gb"]),
            int(batch_metrics.get("split_count", 1)),
            int(batch_metrics.get("oom_retry_count", 0)),
        )

        for idx, (context, generated) in enumerate(zip(contexts, generated_batch, strict=True)):
            history = context["history"]
            parsed = parse_generation(generated["raw_text"], context["task"].answer_type, actual_prompt_mode)
            normalized_answer = normalize_answer(parsed["answer"], context["task"].answer_type)
            is_correct = verify_answer(context["task"], parsed["answer"])
            prior_answer = history[-1]["answer_normalized"] if history else ""
            answer_changed = bool(history) and normalized_answer != prior_answer and normalized_answer != ""
            prior_thought = " ".join(item["thought"] for item in history[-2:])

            hidden_vector = generated["pooled_hidden"]
            if context["hidden_vectors"]:
                previous_hidden = context["hidden_vectors"][-1]
                denominator = float(np.linalg.norm(hidden_vector) * np.linalg.norm(previous_hidden))
                hidden_cosine_shift = 1.0 - float(np.dot(hidden_vector, previous_hidden) / denominator) if denominator else 0.0
                hidden_l2_shift = float(np.linalg.norm(hidden_vector - previous_hidden))
            else:
                hidden_cosine_shift = 0.0
                hidden_l2_shift = 0.0
            context["hidden_vectors"].append(hidden_vector)

            hit_max_new_tokens = int(generated["generated_tokens"] == max_new_tokens)
            truncated_output_suspected = int(hit_max_new_tokens and not parsed["parse_success"])
            tokens_per_second = generated["generated_tokens"] / max(batch_elapsed_seconds, 1e-6)

            # N7: k=2 agreement extraction
            k2_agreement = -1
            k2_tokens = 0
            if k2_batch is not None:
                k2_gen = k2_batch[idx]
                parsed_k2 = parse_generation(k2_gen["raw_text"], context["task"].answer_type, actual_prompt_mode)
                normalized_k2 = normalize_answer(parsed_k2["answer"], context["task"].answer_type)
                if context["task"].answer_type == "math":
                    k2_agreement = int(math_answers_equivalent(normalized_answer, normalized_k2))
                else:
                    k2_agreement = int(normalized_answer == normalized_k2)
                k2_tokens = k2_gen["generated_tokens"]

            row = {
                "run_id": context["run_id"],
                "model_alias": model_spec.alias,
                "model_name": model_spec.hf_name,
                "task_id": context["task"].task_id,
                "domain": context["task"].domain,
                "difficulty": context["task"].difficulty,
                "task_source": context["task"].source,
                "task_source_index": context["task"].source_index,
                "expected_answer": context["task"].expected_answer,
                "step": step,
                "thought": parsed["thought"],
                "answer": parsed["answer"],
                "answer_normalized": normalized_answer,
                "correct": int(is_correct),
                "confidence": int(max(0, min(100, parsed["confidence"]))),
                "model_stop_flag": int(bool(parsed["stop"])),
                "answer_changed": int(answer_changed),
                "thought_token_count": int(len(re.findall(r"\w+", parsed["thought"]))),
                "raw_generation_tokens": generated["generated_tokens"],
                "mean_token_logprob": generated["mean_token_logprob"],
                "entropy_mean": generated["mean_entropy"],
                "entropy_std": generated["entropy_std"],
                "hidden_norm": float(np.linalg.norm(hidden_vector)),
                "hidden_l2_shift": hidden_l2_shift,
                "hidden_cosine_shift": hidden_cosine_shift,
                "lexical_echo": lexical_overlap(parsed["thought"], prior_thought),
                "verbose_confidence_proxy": float(parsed["confidence"]) / 100.0 + 0.01 * generated["generated_tokens"],
                "utility": utility(correct=is_correct, step=step),
                "elapsed_seconds": batch_elapsed_seconds,
                "tokens_per_second": tokens_per_second,
                "gpu_memory_allocated_gb": gpu_memory_allocated_gb(actual_device),
                "seed": seed,
                "temperature": temperature,
                "device": actual_device,
                "prompt_mode": actual_prompt_mode,
                "system_prompt_mode": system_prompt_mode,
                "is_baseline": int(is_baseline),
                "parse_success": parsed["parse_success"],
                "output_format_type": parsed["output_format_type"],
                "answer_extraction_source": parsed["answer_extraction_source"],
                "stop_extraction_source": parsed["stop_extraction_source"],
                "confidence_extraction_source": parsed["confidence_extraction_source"],
                "hit_max_new_tokens": hit_max_new_tokens,
                "truncated_output_suspected": truncated_output_suspected,
                "raw_text_length_chars": len(generated["raw_text"]),
                "raw_text_length_tokens": generated["generated_tokens"],
                "raw_text": generated["raw_text"],
                # N7 Columns
                "k2_agreement": k2_agreement,
                "k2_raw_generation_tokens": k2_tokens,
                # N8 Columns
                "answer_span_mean_logprob": generated.get("answer_span_mean_logprob", float("nan")),
                "answer_span_min_logprob": generated.get("answer_span_min_logprob", float("nan")),
                "answer_span_mean_entropy": generated.get("answer_span_mean_entropy", float("nan")),
                "answer_span_std_entropy": generated.get("answer_span_std_entropy", 0.0),
                "mid_hidden_1_proj": generated.get("mid_hidden_1_proj", ""),
                "mid_hidden_2_proj": generated.get("mid_hidden_2_proj", ""),
            }
            history.append(row)
            context["rows"].append(row)

    hidden_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    all_runs: list[dict[str, Any]] = []
    hidden_state_write_seconds = 0.0
    for context in contexts:
        run_rows = context["rows"]
        hidden_vectors = context["hidden_vectors"]
        if not run_rows:
            continue

        hidden_write_started_at = time.perf_counter()
        npz_path = hidden_dir / f"{context['run_id']}.npz"
        if io_manager:
            io_manager.save_npz_async(npz_path, {"hidden_states": np.stack(hidden_vectors)})
            logging.info("IOManager: Background saving started for %s", npz_path.name)
        else:
            np.savez_compressed(npz_path, hidden_states=np.stack(hidden_vectors))
        hidden_state_write_seconds += time.perf_counter() - hidden_write_started_at
        logging.info("batch_metrics: hidden_state_write_seconds = %.4f", hidden_state_write_seconds)
        first_correct_step = next((row["step"] for row in run_rows if row["correct"] == 1), -1)
        first_model_stop_step = next((row["step"] for row in run_rows if row["model_stop_flag"] == 1), -1)
        oracle_stop = max(run_rows, key=lambda row: row["utility"])["step"]
        revision_count = int(sum(row["answer_changed"] for row in run_rows))

        run_summary = {
            "run_id": context["run_id"],
            "model_alias": model_spec.alias,
            "model_name": model_spec.hf_name,
            "task_id": context["task"].task_id,
            "domain": context["task"].domain,
            "difficulty": context["task"].difficulty,
            "task_source": context["task"].source,
            "task_source_index": context["task"].source_index,
            "temperature": temperature,
            "seed": seed,
            "prompt_mode": actual_prompt_mode,
            "system_prompt_mode": system_prompt_mode,
            "is_baseline": int(is_baseline),
            "ever_correct": int(first_correct_step != -1),
            "correct_at_step_1": int(run_rows[0]["correct"] == 1),
            "oracle_stop": oracle_stop,
            "first_correct_step": first_correct_step if first_correct_step != -1 else max_steps,
            "first_model_stop_step": first_model_stop_step if first_model_stop_step != -1 else max_steps,
            "revision_count": revision_count,
            "best_utility": max(row["utility"] for row in run_rows),
            "final_correct": run_rows[-1]["correct"],
            "device": actual_device,
        }
        all_rows.extend(run_rows)
        all_runs.append(run_summary)
        logging.info(
            "completed %s | ever_correct=%d | first_correct=%s | final_correct=%d | revisions=%d",
            context["run_id"],
            run_summary["ever_correct"],
            run_summary["first_correct_step"],
            run_summary["final_correct"],
            revision_count,
        )

    batch_metric_rows.append(
        {
            "phase": "hidden_state_write",
            "temperature": temperature,
            "seed": seed,
            "step": 0,
            "requested_batch_size": len(contexts),
            "realized_batch_size": len(contexts),
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "wall_clock_seconds": hidden_state_write_seconds,
            "examples_per_second": float("nan"),
            "tokens_per_second": float("nan"),
            "tokenize_seconds": 0.0,
            "generation_seconds": 0.0,
            "forward_seconds": 0.0,
            "postprocess_seconds": 0.0,
            "hidden_state_write_seconds": hidden_state_write_seconds,
            "gpu_memory_allocated_gb": gpu_memory_allocated_gb(actual_device),
            "gpu_memory_reserved_gb": gpu_memory_reserved_gb(actual_device),
            "gpu_max_memory_allocated_gb": gpu_max_memory_allocated_gb(actual_device),
            "gpu_max_memory_reserved_gb": gpu_max_memory_reserved_gb(actual_device),
            "split_count": 0,
            "oom_retry_count": 0,
        }
    )

    return all_rows, all_runs, batch_metric_rows


def _safe_mean(series: pd.Series) -> float:
    """Mean that tolerates a string-contaminated numeric column: coerce
    non-numeric entries to NaN and average the rest. AUDIT/STABILITY FIX -- a
    single stray string in a summary column was doing str+float inside pandas
    .mean() and crashing a cell's end-of-run summary, throwing away hours of
    completed collection (e.g. yi x math failed after 5 h)."""
    return float(pd.to_numeric(series, errors="coerce").mean())


def summarize_transitions(step_frame: pd.DataFrame) -> pd.DataFrame:
    ordered = step_frame.sort_values(["run_id", "step"]).copy()
    ordered["next_correct"] = ordered.groupby("run_id")["correct"].shift(-1)
    ordered = ordered.dropna(subset=["next_correct"])
    ordered["next_correct"] = ordered["next_correct"].astype(int)
    ordered["repair"] = ((ordered["correct"] == 0) & (ordered["next_correct"] == 1)).astype(int)
    ordered["corruption"] = ((ordered["correct"] == 1) & (ordered["next_correct"] == 0)).astype(int)
    rows: list[dict[str, Any]] = []
    for step, group in ordered.groupby("step"):
        wrong_group = group[group["correct"] == 0]
        correct_group = group[group["correct"] == 1]
        n_repairs = wrong_group["repair"].sum()
        n_corruptions = correct_group["corruption"].sum()
        repair_rate = _safe_mean(wrong_group["repair"]) if len(wrong_group) else float("nan")
        corruption_rate = _safe_mean(correct_group["corruption"]) if len(correct_group) else float("nan")
        q_t = _safe_mean(group["correct"])
        if (n_repairs + n_corruptions) >= 3 and not math.isnan(repair_rate) and not math.isnan(corruption_rate):
            hazard_mu = (1.0 - q_t) * repair_rate - q_t * corruption_rate - STEP_COST
        else:
            hazard_mu = float("nan")
        rows.append(
            {
                "step": int(step),
                "q_t": q_t,
                "n_repairs": int(n_repairs),
                "n_corruptions": int(n_corruptions),
                "repair_rate": repair_rate,
                "corruption_rate": corruption_rate,
                "hazard_mu": hazard_mu,
                "entropy_mean": _safe_mean(group["entropy_mean"]),
                "confidence_mean": _safe_mean(group["confidence"]),
                "answer_changed_rate": _safe_mean(group["answer_changed"]),
                "hidden_shift_mean": _safe_mean(group["hidden_l2_shift"]),
                "n_transitions": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def build_pilot_summary(
    step_frame: pd.DataFrame,
    run_frame: pd.DataFrame,
    transition_frame: pd.DataFrame,
    model_spec: ModelSpec,
    backend: str,
    actual_device: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_alias": model_spec.alias,
                "model_name": model_spec.hf_name,
                "family": model_spec.family,
                "parameter_count": model_spec.parameter_count,
                "n_runs": int(len(run_frame)),
                "n_tasks": int(step_frame["task_id"].nunique()),
                "mean_oracle_stop": _safe_mean(run_frame["oracle_stop"]),
                "mean_first_correct_step": _safe_mean(run_frame["first_correct_step"]),
                "mean_model_stop_step": _safe_mean(run_frame["first_model_stop_step"]),
                "mean_revision_count": _safe_mean(run_frame["revision_count"]),
                "mean_entropy": _safe_mean(step_frame["entropy_mean"]),
                "mean_hidden_shift": _safe_mean(step_frame["hidden_l2_shift"]),
                "repair_rate_overall": _safe_mean(transition_frame["repair_rate"]) if not transition_frame.empty else float("nan"),
                "corruption_rate_overall": _safe_mean(transition_frame["corruption_rate"]) if not transition_frame.empty else float("nan"),
                "runs_ever_correct": int(run_frame["ever_correct"].sum()),
                "backend": backend,
                "device": actual_device,
            }
        ]
    )


def infer_existing_runtime_context(paths: dict[str, Path], requested_device: str) -> tuple[str, str]:
    backend = "transformers+torch(uninitialized)"
    if requested_device == "auto":
        actual_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        actual_device = requested_device

    metadata_path = paths["metadata"]
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            backend = str(metadata.get("backend", backend))
            actual_device = str(metadata.get("device", actual_device))
            return backend, actual_device
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            logging.warning("Failed to read existing metadata from %s for runtime context recovery.", metadata_path)

    pilot_path = paths["pilot"]
    if pilot_path.exists():
        try:
            pilot_summary = pd.read_csv(pilot_path)
            if not pilot_summary.empty:
                backend = str(pilot_summary.iloc[0].get("backend", backend))
                actual_device = str(pilot_summary.iloc[0].get("device", actual_device))
        except (OSError, ValueError, pd.errors.EmptyDataError):
            logging.warning("Failed to read existing pilot summary from %s for runtime context recovery.", pilot_path)

    return backend, actual_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-trace reasoning experiments on builtin tasks or GSM8K.")
    parser.add_argument("--model", default="deepseek_r1_distill_1p5b", choices=sorted(MODEL_CATALOG.keys()))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--quantization", default="none", choices=["none", "8bit", "4bit"])
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--model-path-override", default=None)
    parser.add_argument("--attn-implementation", default="auto", choices=["auto", "sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.6])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-tasks", type=int, default=len(BUILTIN_TASKS))
    parser.add_argument("--task-source", default="builtin", choices=["builtin", "gsm8k", "math", "arc", "gpqa"])
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-shuffle-seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-mode", default="structured_four_line", choices=["structured_four_line", "minimal_json", "answer_only"])
    parser.add_argument("--system-prompt-mode", default="default", choices=["default", "short", "none"])
    parser.add_argument("--run-baseline", action="store_true", help="Run competence baseline instead of iterative reasoning")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--io-threads", type=int, default=4, help="Number of background threads for IO operations")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--enable-k2-agreement", action="store_true", help="Enable k=2 self-consistency agreement recording (N7)")
    parser.add_argument("--enable-extended-observables", action="store_true", help="Enable answer-span metrics and mid-layer hidden projections (N8)")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir = output_dir / ("hidden_states_baseline" if args.run_baseline else "hidden_states")
    paths = output_paths(output_dir, args.run_baseline)
    model_spec = MODEL_CATALOG[args.model]

    tasks = load_tasks(
        task_source=args.task_source,
        max_tasks=args.max_tasks,
        dataset_split=args.dataset_split,
        shuffle_seed=args.dataset_shuffle_seed,
    )
    logging.info("Loaded %d tasks from %s.", len(tasks), args.task_source)

    existing_steps, existing_runs, completed_run_ids, reconciliation_report = reconcile_existing_outputs(
        output_dir=output_dir,
        hidden_dir=hidden_dir,
        is_baseline=args.run_baseline,
        max_steps=args.max_steps,
    )
    if reconciliation_report["anomalies_detected"]:
        logging.warning("Checkpoint reconciliation adjusted on-disk artifacts: %s", reconciliation_report)
    if args.resume and completed_run_ids:
        logging.info("Resuming with %d completed runs already on disk.", len(completed_run_ids))

    total_requested_runs = len(tasks) * len(args.temperatures) * len(args.seeds)
    pending_requested_runs = 0
    for temperature in args.temperatures:
        for seed in args.seeds:
            for task in tasks:
                run_id = run_id_for(model_spec.alias, task, temperature, seed)
                if not args.resume or run_id not in completed_run_ids:
                    pending_requested_runs += 1
    logging.info("Requested runs=%d | pending runs=%d", total_requested_runs, pending_requested_runs)
    next_pending_run = first_pending_run(
        model_spec=model_spec,
        tasks=tasks,
        temperatures=args.temperatures,
        seeds=args.seeds,
        completed_run_ids=completed_run_ids if args.resume else set(),
    )
    if next_pending_run is not None:
        logging.info(
            "Next pending run: task_index=%s | temperature=%.2f | seed=%d | run_id=%s",
            next_pending_run["task_source_index"],
            next_pending_run["temperature"],
            next_pending_run["seed"],
            next_pending_run["run_id"],
        )

    if pending_requested_runs == 0 and existing_runs:
        logging.info("No pending runs detected. Rebuilding summaries from existing artifacts.")
    elif pending_requested_runs == 0:
        logging.info("No runs to execute.")

    model = None
    tokenizer = None
    backend, actual_device = infer_existing_runtime_context(paths, args.device)
    if pending_requested_runs > 0:
        offload_folder = output_dir / "model_offload"
        offload_folder.mkdir(parents=True, exist_ok=True)
        model, tokenizer, actual_device, backend = load_model(
            model_spec=model_spec,
            device=args.device,
            quantization=args.quantization,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
            model_path_override=args.model_path_override,
            offload_folder=str(offload_folder),
        )
        logging.info("Model loaded: %s | backend=%s | device=%s", model_spec.hf_name, backend, actual_device)

    io_manager = IOManager(max_threads=args.io_threads)
    try:
        write_runtime_metadata(
            metadata_path=paths["metadata"],
            model_spec=model_spec,
            backend=backend,
            actual_device=actual_device,
            quantization=args.quantization,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
            temperatures=args.temperatures,
            seeds=args.seeds,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            max_tasks=args.max_tasks,
            task_source=args.task_source,
            dataset_split=args.dataset_split,
            dataset_shuffle_seed=args.dataset_shuffle_seed,
            batch_size=args.batch_size,
            prompt_mode=args.prompt_mode,
            system_prompt_mode=args.system_prompt_mode,
            resume_enabled=args.resume,
            completed_run_ids=completed_run_ids,
            pending_run_count=pending_requested_runs,
            next_pending_run=next_pending_run,
            reconciliation_report=reconciliation_report,
            tasks=tasks,
            model_source=args.model_path_override,
        )

        all_rows: list[dict[str, Any]] = list(existing_steps)
        all_runs: list[dict[str, Any]] = list(existing_runs)

        if model is not None and tokenizer is not None:
            for temperature in args.temperatures:
                for seed in args.seeds:
                    pending_tasks = [
                        task
                        for task in tasks
                        if not args.resume or run_id_for(model_spec.alias, task, temperature, seed) not in completed_run_ids
                    ]
                    if not pending_tasks:
                        logging.info("Skipping temp=%.2f seed=%d because all runs already exist.", temperature, seed)
                        continue

                    set_seed(seed)
                    batches = list(chunked(pending_tasks, max(1, args.batch_size)))
                    progress = tqdm(batches, desc=f"temp={temperature:.2f} seed={seed}", unit="batch")
                    for task_batch in progress:
                        batch_rows, batch_runs, batch_metrics = run_batch_traces(
                            model=model,
                            tokenizer=tokenizer,
                            model_spec=model_spec,
                            tasks=task_batch,
                            actual_device=actual_device,
                            temperature=temperature,
                            seed=seed,
                            max_steps=args.max_steps,
                            max_new_tokens=args.max_new_tokens,
                            prompt_mode=args.prompt_mode,
                            system_prompt_mode=args.system_prompt_mode,
                            is_baseline=args.run_baseline,
                            hidden_dir=hidden_dir,
                            io_manager=io_manager,
                            enable_k2_agreement=args.enable_k2_agreement,
                            enable_extended_observables=args.enable_extended_observables,
                        )
                        append_records(paths["steps"], batch_rows)
                        append_records(paths["runs"], batch_runs)
                        append_records(paths["batch_metrics"], batch_metrics)
                        all_rows.extend(batch_rows)
                        all_runs.extend(batch_runs)
                        completed_run_ids.update(run["run_id"] for run in batch_runs)
                        next_pending_run = first_pending_run(
                            model_spec=model_spec,
                            tasks=tasks,
                            temperatures=args.temperatures,
                            seeds=args.seeds,
                            completed_run_ids=completed_run_ids if args.resume else set(),
                        )
                        write_runtime_metadata(
                            metadata_path=paths["metadata"],
                            model_spec=model_spec,
                            backend=backend,
                            actual_device=actual_device,
                            quantization=args.quantization,
                            device_map=args.device_map,
                            attn_implementation=args.attn_implementation,
                            temperatures=args.temperatures,
                            seeds=args.seeds,
                            max_steps=args.max_steps,
                            max_new_tokens=args.max_new_tokens,
                            max_tasks=args.max_tasks,
                            task_source=args.task_source,
                            dataset_split=args.dataset_split,
                            dataset_shuffle_seed=args.dataset_shuffle_seed,
                            batch_size=args.batch_size,
                            prompt_mode=args.prompt_mode,
                            system_prompt_mode=args.system_prompt_mode,
                            resume_enabled=args.resume,
                            completed_run_ids=completed_run_ids,
                            pending_run_count=max(total_requested_runs - len(completed_run_ids), 0),
                            next_pending_run=next_pending_run,
                            reconciliation_report=reconciliation_report,
                            tasks=tasks,
                            model_source=args.model_path_override,
                        )
                        progress.set_postfix(
                            runs=len(batch_runs),
                            correct=sum(int(run["ever_correct"]) for run in batch_runs),
                        )

        if not all_rows or not all_runs:
            raise RuntimeError("No trace data is available to summarize. The experiment produced no rows.")

        if args.run_baseline:
            logging.info("Wrote baseline artifacts to: %s", output_dir)
            return

        step_frame = pd.DataFrame(all_rows)
        run_frame = pd.DataFrame(all_runs)
        transition_frame = summarize_transitions(step_frame)
        pilot_summary = build_pilot_summary(step_frame, run_frame, transition_frame, model_spec, backend, actual_device)
    finally:
        if io_manager:
            io_manager.shutdown()

    transition_frame.to_csv(paths["hazard"], index=False)
    pilot_summary.to_csv(paths["pilot"], index=False)

    next_pending_run = first_pending_run(
        model_spec=model_spec,
        tasks=tasks,
        temperatures=args.temperatures,
        seeds=args.seeds,
        completed_run_ids=completed_run_ids if args.resume else set(),
    )
    write_runtime_metadata(
        metadata_path=paths["metadata"],
        model_spec=model_spec,
        backend=backend,
        actual_device=actual_device,
        quantization=args.quantization,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        temperatures=args.temperatures,
        seeds=args.seeds,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        max_tasks=args.max_tasks,
        task_source=args.task_source,
        dataset_split=args.dataset_split,
        dataset_shuffle_seed=args.dataset_shuffle_seed,
        batch_size=args.batch_size,
        prompt_mode=args.prompt_mode,
        system_prompt_mode=args.system_prompt_mode,
        resume_enabled=args.resume,
        completed_run_ids=completed_run_ids,
        pending_run_count=max(total_requested_runs - len(completed_run_ids), 0),
        next_pending_run=next_pending_run,
        reconciliation_report=reconciliation_report,
        tasks=tasks,
        model_source=args.model_path_override,
    )

    for _, row in pilot_summary.iterrows():
        print(
            f"{row['model_alias']}: runs={int(row['n_runs'])}, "
            f"ever_correct={int(row['runs_ever_correct'])}, "
            f"oracle={row['mean_oracle_stop']:.2f}, first_correct={row['mean_first_correct_step']:.2f}, "
            f"repair={row['repair_rate_overall']:.3f}, corruption={row['corruption_rate_overall']:.3f}, "
            f"device={row['device']}"
        )
    print(f"Wrote real-trace artifacts to: {output_dir}")


if __name__ == "__main__":
    main()