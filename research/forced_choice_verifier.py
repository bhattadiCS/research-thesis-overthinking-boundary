#!/usr/bin/env python3
"""Pure, auditable primitives for a four-option causal verifier.

The historical panel cannot resolve GPQA through agreement alone: the correct
answer is often present but not the plurality answer, and sometimes is absent
from the panel entirely.  A verifier must therefore inspect the frozen public
question/options and be able to select any legal option.  This module contains
only the deterministic and scoring-facing pieces needed by a prospective
collector.  It never accepts gold labels, task identifiers, model aliases, or
raw candidate answers as verifier inputs.

``score_forced_choice_options`` computes the exact causal continuation
likelihood for each frozen option string.  It intentionally fails closed if a
tokenizer cannot preserve the prompt prefix when appending a continuation;
silently scoring a differently tokenized prompt would invalidate the ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


OPTION_LABELS: tuple[str, ...] = ("A", "B", "C", "D")
PROMPT_OPTIONS_ONLY_V1 = "prompt_options_only_v1"
ANONYMOUS_RATIONALE_V1 = "anonymous_rationale_v1"
SUPPORTED_VARIANTS = (PROMPT_OPTIONS_ONLY_V1, ANONYMOUS_RATIONALE_V1)
SCORE_METHOD = "exact_causal_continuation_logprob_v1"
TOKEN_ACCOUNTING_CONTRACT = "four_independent_continuations_v1"


class ForcedChoiceVerifierError(ValueError):
    """Raised for a malformed public prompt or scoring contract."""


@dataclass(frozen=True)
class ForcedChoiceScores:
    """JSON-safe result of exact four-way continuation scoring."""

    option_logprobs: dict[str, float]
    option_posteriors: dict[str, float]
    argmax_option: str
    top1_margin: float
    entropy: float
    base_prompt_tokens: int
    option_scoring_tokens: dict[str, int]
    score_method: str = SCORE_METHOD
    token_accounting_contract: str = TOKEN_ACCOUNTING_CONTRACT

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def verifier_template_sha256(variant: str) -> str:
    """Return the hash of the fixed template for a supported prompt variant."""

    if variant == PROMPT_OPTIONS_ONLY_V1:
        template = _prompt_options_template()
    elif variant == ANONYMOUS_RATIONALE_V1:
        template = _anonymous_rationale_template()
    else:
        raise ForcedChoiceVerifierError(f"Unsupported verifier variant {variant!r}.")
    return sha256_text(template)


def frozen_option_continuations() -> dict[str, str]:
    """Frozen strings scored after the verifier's ``Final option:`` prompt."""

    # The leading space is intentional and belongs in the manifest.  It avoids
    # accidental word-piece attachment to the colon in most causal tokenizers.
    return {label: f" {label}" for label in OPTION_LABELS}


def parse_four_option_mcq_prompt(public_prompt: str) -> dict[str, str]:
    """Parse exactly A--D options from the frozen public MCQ prompt.

    The trace collectors render multiple-choice prompts as one option per line
    (``A. ...`` through ``D. ...``).  We accept ``.``, ``)``, or ``:`` after
    the label but require all four labels exactly once and in canonical order.
    Any ambiguity is a protocol error, not something a verifier should guess.
    """

    if not isinstance(public_prompt, str) or not public_prompt.strip():
        raise ForcedChoiceVerifierError("Public MCQ prompt must be a non-empty string.")
    matches = list(re.finditer(r"(?mi)^\s*([A-D])\s*[\.)\:]\s*", public_prompt))
    labels = [match.group(1).upper() for match in matches]
    if labels != list(OPTION_LABELS):
        raise ForcedChoiceVerifierError(
            "Public MCQ prompt must contain exactly one line-start A., B., C., and D. option in that order; "
            f"observed {labels!r}."
        )
    options: dict[str, str] = {}
    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(public_prompt)
        text = public_prompt[match.end() : next_start]
        # The standard public prompt ends with a response instruction after
        # option D.  It is not part of the D option and must not be re-injected
        # as evidence when the verifier prompt is rendered.
        text = re.split(r"(?mi)^\s*respond\s+with\s+the\s+letter\b", text, maxsplit=1)[0]
        text = text.strip()
        if not text:
            raise ForcedChoiceVerifierError(f"Option {OPTION_LABELS[index]!r} is empty.")
        options[OPTION_LABELS[index]] = text
    return options


def _question_prefix(public_prompt: str) -> str:
    first = re.search(r"(?mi)^\s*A\s*[\.)\:]\s*", public_prompt)
    if first is None:
        raise ForcedChoiceVerifierError("Cannot identify the first option in the public MCQ prompt.")
    question = public_prompt[: first.start()]
    question = re.sub(r"(?mi)^\s*options\s*:\s*$", "", question).strip()
    if not question:
        raise ForcedChoiceVerifierError("Public MCQ prompt has no question text before option A.")
    return question


def _prompt_options_template() -> str:
    return (
        "You are an independent multiple-choice verifier. Solve the frozen public question using only "
        "the question and four options below. Do not assume a panel consensus is correct.\n\n"
        "QUESTION:\n{question}\n\nOPTIONS:\n{options}\n\n"
        "Return no explanation. The final option must be exactly one of A, B, C, or D.\n"
        "Final option:"
    )


def _anonymous_rationale_template() -> str:
    return (
        "You are an independent multiple-choice verifier. Solve the frozen public question using the "
        "question, four options, and anonymized current-barrier rationales below. Rationales may be wrong; "
        "do not infer model identity or use consensus as a correctness label.\n\n"
        "QUESTION:\n{question}\n\nOPTIONS:\n{options}\n\n"
        "ANONYMIZED RATIONALES:\n{rationales}\n\n"
        "Return no explanation. The final option must be exactly one of A, B, C, or D.\n"
        "Final option:"
    )


def _normalize_rationale(value: str, max_chars: int) -> str:
    if not isinstance(value, str):
        raise ForcedChoiceVerifierError("A rationale must be a string.")
    if max_chars < 1:
        raise ForcedChoiceVerifierError("rationale_max_chars must be positive.")
    compact = re.sub(r"\s+", " ", value).strip()
    return compact[:max_chars]


def anonymize_rationales(
    thoughts: Sequence[str],
    *,
    seed_material: str,
    max_chars: int,
) -> list[str]:
    """Deterministically shuffle text-only rationales and return ``R01: ...``.

    Callers must pass only event ``thought`` strings.  This function does not
    accept model aliases, answers, confidences, or any label-bearing field.
    Empty thoughts are dropped so their count cannot encode an ordering slot.
    """

    if not isinstance(seed_material, str) or not seed_material:
        raise ForcedChoiceVerifierError("Anonymization requires non-empty deterministic seed material.")
    normalized = [_normalize_rationale(value, max_chars) for value in thoughts]
    normalized = [value for value in normalized if value]
    if not normalized:
        raise ForcedChoiceVerifierError("Anonymous-rationale verifier requires at least one non-empty thought.")
    seed = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "big")
    shuffled = list(normalized)
    random.Random(seed).shuffle(shuffled)
    return [f"R{index:02d}: {value}" for index, value in enumerate(shuffled, start=1)]


def render_verifier_prompt(
    public_prompt: str,
    *,
    variant: str,
    thoughts: Sequence[str] | None = None,
    seed_material: str | None = None,
    rationale_max_chars: int = 768,
) -> str:
    """Render a frozen public-prompt verifier request without labels.

    A caller receives an exception if the source prompt is not exactly a
    four-option A--D MCQ.  This makes GPQA/ARC scope explicit and prevents a
    hidden conversion of arbitrary free-form tasks into a bogus choice task.
    """

    options = parse_four_option_mcq_prompt(public_prompt)
    question = _question_prefix(public_prompt)
    option_text = "\n".join(f"{label}. {options[label]}" for label in OPTION_LABELS)
    if variant == PROMPT_OPTIONS_ONLY_V1:
        if thoughts not in (None, (), []):
            raise ForcedChoiceVerifierError("prompt_options_only_v1 must not receive panel rationale text.")
        return _prompt_options_template().format(question=question, options=option_text)
    if variant == ANONYMOUS_RATIONALE_V1:
        if thoughts is None or seed_material is None:
            raise ForcedChoiceVerifierError("anonymous_rationale_v1 requires thoughts and seed_material.")
        rationales = "\n".join(
            anonymize_rationales(thoughts, seed_material=seed_material, max_chars=rationale_max_chars)
        )
        return _anonymous_rationale_template().format(
            question=question,
            options=option_text,
            rationales=rationales,
        )
    raise ForcedChoiceVerifierError(f"Unsupported verifier variant {variant!r}.")


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    raw_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else getattr(encoded, "input_ids", None)
    if raw_ids is None:
        raise ForcedChoiceVerifierError("Tokenizer did not return input_ids.")
    if raw_ids and isinstance(raw_ids[0], list):
        if len(raw_ids) != 1:
            raise ForcedChoiceVerifierError("Tokenizer unexpectedly returned a batch for a scalar prompt.")
        raw_ids = raw_ids[0]
    ids = [int(value) for value in raw_ids]
    if not ids:
        raise ForcedChoiceVerifierError("Tokenizer produced no tokens for a verifier prompt.")
    return ids


def _continuation_ids(tokenizer: Any, prompt: str, continuation: str) -> tuple[list[int], list[int]]:
    base_ids = _token_ids(tokenizer, prompt)
    full_ids = _token_ids(tokenizer, prompt + continuation)
    if full_ids[: len(base_ids)] != base_ids or len(full_ids) <= len(base_ids):
        raise ForcedChoiceVerifierError(
            "Tokenizer changed the frozen prompt prefix while appending a choice continuation. "
            "Use a prompt/continuation boundary with prefix-stable tokenization."
        )
    return base_ids, full_ids[len(base_ids) :]


def _sequence_start_from_mask(attention_mask: Any, row_index: int, expected_length: int) -> int:
    """Return the first real-token position for one padded scorer row.

    The production loader intentionally uses left padding for causal models,
    but exact continuation scoring is also useful as a standalone primitive.
    Reading the attention mask rather than assuming a padding side makes the
    target-token alignment correct for both left- and right-padded tokenizers.
    """

    active = attention_mask[row_index].detach().to("cpu").nonzero(as_tuple=False).flatten().tolist()
    if len(active) != expected_length or not active:
        raise ForcedChoiceVerifierError(
            "Tokenizer attention mask length does not match the independently tokenized verifier sequence."
        )
    first = int(active[0])
    if active != list(range(first, first + len(active))):
        raise ForcedChoiceVerifierError("Tokenizer attention mask must contain one contiguous token span per sequence.")
    return first


def posterior_from_logprobs(option_logprobs: Mapping[str, float]) -> tuple[dict[str, float], str, float, float]:
    """Convert exact option log likelihoods to a normalized posterior summary."""

    if set(option_logprobs) != set(OPTION_LABELS):
        raise ForcedChoiceVerifierError("Option log probabilities must cover exactly A, B, C, and D.")
    values = {label: float(option_logprobs[label]) for label in OPTION_LABELS}
    if not all(math.isfinite(value) for value in values.values()):
        raise ForcedChoiceVerifierError("Option log probabilities must be finite.")
    maximum = max(values.values())
    unnormalized = {label: math.exp(value - maximum) for label, value in values.items()}
    denominator = sum(unnormalized.values())
    posteriors = {label: unnormalized[label] / denominator for label in OPTION_LABELS}
    ranked = sorted(OPTION_LABELS, key=lambda label: (-posteriors[label], label))
    argmax = ranked[0]
    margin = posteriors[ranked[0]] - posteriors[ranked[1]]
    entropy = -sum(value * math.log(value) for value in posteriors.values() if value > 0.0)
    return posteriors, argmax, margin, entropy


def score_forced_choice_options(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    device: Any | None = None,
    continuations: Mapping[str, str] | None = None,
    microbatch_size: int = 4,
) -> ForcedChoiceScores:
    """Score four options by exact causal continuation log likelihood.

    The four choices are evaluated as independent prompt+continuation sequences,
    which is why a collector must account for four prompt forwards.  The helper
    neither samples text nor reads labels.  ``device`` should be the resolved
    input device returned by the model loader; using it explicitly is necessary
    for models with non-default placement.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - only relevant to a broken runtime.
        raise RuntimeError("PyTorch is required to score forced-choice options.") from exc
    if not isinstance(prompt, str) or not prompt.strip():
        raise ForcedChoiceVerifierError("Verifier scoring prompt must be non-empty.")
    if microbatch_size < 1:
        raise ForcedChoiceVerifierError("microbatch_size must be positive.")
    frozen = dict(frozen_option_continuations() if continuations is None else continuations)
    if set(frozen) != set(OPTION_LABELS) or not all(isinstance(frozen[key], str) and frozen[key] for key in OPTION_LABELS):
        raise ForcedChoiceVerifierError("Frozen continuations must be non-empty strings for exactly A--D.")

    base_ids: list[int] | None = None
    full_texts: list[str] = []
    full_ids: list[list[int]] = []
    continuation_ids: list[list[int]] = []
    for label in OPTION_LABELS:
        candidate_base, suffix = _continuation_ids(tokenizer, prompt, frozen[label])
        if base_ids is None:
            base_ids = candidate_base
        elif base_ids != candidate_base:
            raise ForcedChoiceVerifierError("Tokenizer produced inconsistent base prompt tokens across options.")
        full_texts.append(prompt + frozen[label])
        full_ids.append(candidate_base + suffix)
        continuation_ids.append(suffix)
    assert base_ids is not None

    if device is None:
        try:
            device = next(model.parameters()).device
        except (AttributeError, StopIteration) as exc:
            raise ForcedChoiceVerifierError("Pass an explicit input device for a parameterless/dispatched model.") from exc
    input_device = torch.device(device)
    logprob_values: dict[str, float] = {}
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        for start in range(0, len(OPTION_LABELS), microbatch_size):
            stop = min(start + microbatch_size, len(OPTION_LABELS))
            labels = OPTION_LABELS[start:stop]
            texts = full_texts[start:stop]
            encoded = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt")
            if not isinstance(encoded, Mapping):
                raise ForcedChoiceVerifierError("Tokenizer must return a mapping for batched verifier scoring.")
            input_ids = encoded["input_ids"].to(input_device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            else:
                attention_mask = attention_mask.to(input_device)
            with torch.inference_mode():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                logits = outputs.logits.float()
            for local_index, label in enumerate(labels):
                option_index = OPTION_LABELS.index(label)
                sequence_ids = full_ids[option_index]
                suffix_ids = continuation_ids[option_index]
                sequence_start = _sequence_start_from_mask(attention_mask, local_index, len(sequence_ids))
                total = 0.0
                prompt_length = len(base_ids)
                for suffix_index, target_id in enumerate(suffix_ids):
                    target_position = sequence_start + prompt_length + suffix_index
                    logit_position = target_position - 1
                    if logit_position < sequence_start:
                        raise ForcedChoiceVerifierError("A continuation must have at least one prompt token before it.")
                    row = torch.log_softmax(logits[local_index, logit_position], dim=-1)
                    total += float(row[int(target_id)].item())
                logprob_values[label] = total
    finally:
        if was_training:
            model.train()

    posteriors, argmax, margin, entropy = posterior_from_logprobs(logprob_values)
    return ForcedChoiceScores(
        option_logprobs={label: float(logprob_values[label]) for label in OPTION_LABELS},
        option_posteriors=posteriors,
        argmax_option=argmax,
        top1_margin=float(margin),
        entropy=float(entropy),
        base_prompt_tokens=len(base_ids),
        option_scoring_tokens={label: len(continuation_ids[index]) for index, label in enumerate(OPTION_LABELS)},
    )


def score_forced_choice_prompt_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    device: Any | None = None,
    continuations: Mapping[str, str] | None = None,
    sequence_microbatch_size: int = 32,
) -> list[ForcedChoiceScores]:
    """Batch exact A--D scoring across independent public prompts.

    ``sequence_microbatch_size`` counts complete ``prompt + option`` sequences,
    not tasks.  Thus a task batch of eight produces 32 causal sequences.  This
    is the throughput-oriented companion to ``score_forced_choice_options``:
    it keeps each prompt's four independent forward costs explicit while making
    large-GPU verification practical.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required to score forced-choice options.") from exc
    if not prompts:
        return []
    if sequence_microbatch_size < len(OPTION_LABELS):
        raise ForcedChoiceVerifierError("sequence_microbatch_size must hold all four options for at least one prompt.")
    frozen = dict(frozen_option_continuations() if continuations is None else continuations)
    if set(frozen) != set(OPTION_LABELS) or not all(isinstance(frozen[label], str) and frozen[label] for label in OPTION_LABELS):
        raise ForcedChoiceVerifierError("Frozen continuations must be non-empty strings for exactly A--D.")
    if device is None:
        try:
            device = next(model.parameters()).device
        except (AttributeError, StopIteration) as exc:
            raise ForcedChoiceVerifierError("Pass an explicit input device for a parameterless/dispatched model.") from exc
    input_device = torch.device(device)

    records: list[tuple[int, str, str, list[int], list[int], int]] = []
    prompt_lengths: list[int] = []
    for prompt_index, prompt in enumerate(prompts):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ForcedChoiceVerifierError("Every verifier scoring prompt must be non-empty.")
        base_ids: list[int] | None = None
        for label in OPTION_LABELS:
            candidate_base, suffix = _continuation_ids(tokenizer, prompt, frozen[label])
            if base_ids is None:
                base_ids = candidate_base
            elif candidate_base != base_ids:
                raise ForcedChoiceVerifierError("Tokenizer produced inconsistent base prompt tokens across options.")
            records.append((prompt_index, label, prompt + frozen[label], candidate_base + suffix, suffix, len(candidate_base)))
        assert base_ids is not None
        prompt_lengths.append(len(base_ids))

    logprobs: list[dict[str, float]] = [{label: 0.0 for label in OPTION_LABELS} for _ in prompts]
    suffix_lengths: list[dict[str, int]] = [{label: 0 for label in OPTION_LABELS} for _ in prompts]
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        for start in range(0, len(records), sequence_microbatch_size):
            chunk = records[start : start + sequence_microbatch_size]
            texts = [item[2] for item in chunk]
            encoded = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt")
            if not isinstance(encoded, Mapping):
                raise ForcedChoiceVerifierError("Tokenizer must return a mapping for batched verifier scoring.")
            input_ids = encoded["input_ids"].to(input_device)
            attention_mask = encoded.get("attention_mask")
            attention_mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask.to(input_device)
            with torch.inference_mode():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                logits = outputs.logits.float()
            for local_index, (prompt_index, label, _text, full_ids, suffix_ids, prompt_length) in enumerate(chunk):
                sequence_start = _sequence_start_from_mask(attention_mask, local_index, len(full_ids))
                total = 0.0
                for suffix_index, target_id in enumerate(suffix_ids):
                    target_position = sequence_start + prompt_length + suffix_index
                    logit_position = target_position - 1
                    if logit_position < sequence_start:
                        raise ForcedChoiceVerifierError("A continuation must have at least one prompt token before it.")
                    row = torch.log_softmax(logits[local_index, logit_position], dim=-1)
                    total += float(row[int(target_id)].item())
                logprobs[prompt_index][label] = total
                suffix_lengths[prompt_index][label] = len(suffix_ids)
    finally:
        if was_training:
            model.train()

    output: list[ForcedChoiceScores] = []
    for prompt_index in range(len(prompts)):
        posteriors, argmax, margin, entropy = posterior_from_logprobs(logprobs[prompt_index])
        output.append(
            ForcedChoiceScores(
                option_logprobs={label: float(logprobs[prompt_index][label]) for label in OPTION_LABELS},
                option_posteriors=posteriors,
                argmax_option=argmax,
                top1_margin=float(margin),
                entropy=float(entropy),
                base_prompt_tokens=prompt_lengths[prompt_index],
                option_scoring_tokens={label: int(suffix_lengths[prompt_index][label]) for label in OPTION_LABELS},
            )
        )
    return output


def _self_test() -> None:
    prompt = (
        "Which statement is correct?\n\nOptions:\n"
        "A. First possibility\nB. Second possibility\nC. Third possibility\nD. Fourth possibility\n\n"
        "Respond with the letter of the single correct option."
    )
    assert parse_four_option_mcq_prompt(prompt) == {
        "A": "First possibility",
        "B": "Second possibility",
        "C": "Third possibility",
        "D": "Fourth possibility",
    }
    try:
        parse_four_option_mcq_prompt("Question\nA. one\nC. three\nD. four")
    except ForcedChoiceVerifierError:
        pass
    else:  # pragma: no cover - assertion failure intentionally signals a bad parser.
        raise AssertionError("Missing B option was accepted.")
    first = anonymize_rationales(["  reason one ", "reason two"], seed_material="barrier|variant", max_chars=32)
    second = anonymize_rationales(["  reason one ", "reason two"], seed_material="barrier|variant", max_chars=32)
    assert first == second and all(row.startswith("R") for row in first)
    rendered = render_verifier_prompt(prompt, variant=PROMPT_OPTIONS_ONLY_V1)
    assert "Final option:" in rendered and "First possibility" in rendered
    anonymous = render_verifier_prompt(
        prompt,
        variant=ANONYMOUS_RATIONALE_V1,
        thoughts=["reason one", "reason two"],
        seed_material="barrier|variant",
        rationale_max_chars=32,
    )
    assert "ANONYMIZED RATIONALES:" in anonymous and "R01:" in anonymous
    posteriors, argmax, margin, entropy = posterior_from_logprobs({"A": -2.0, "B": 0.0, "C": -1.0, "D": -3.0})
    assert argmax == "B" and margin > 0.0 and entropy > 0.0
    assert abs(sum(posteriors.values()) - 1.0) < 1e-12
    assert verifier_template_sha256(PROMPT_OPTIONS_ONLY_V1) != verifier_template_sha256(ANONYMOUS_RATIONALE_V1)

    # Exercise the batched scorer against independently scored prompts under
    # both padding conventions.  The real model loader uses left padding, but
    # this protects the standalone primitive from silently misaligning an
    # option token when a caller supplies a right-padded tokenizer.
    import torch
    from types import SimpleNamespace

    class ToyTokenizer:
        def __init__(self, padding_side: str) -> None:
            self.padding_side = padding_side

        @staticmethod
        def _encode(text: str) -> list[int]:
            return [1 + (ord(character) % 127) for character in text]

        def __call__(self, texts: str | Sequence[str], **kwargs: Any) -> dict[str, Any]:
            values = [texts] if isinstance(texts, str) else list(texts)
            encoded = [self._encode(str(value)) for value in values]
            if kwargs.get("return_tensors") != "pt":
                return {"input_ids": encoded[0] if isinstance(texts, str) else encoded}
            width = max(len(row) for row in encoded)
            ids: list[list[int]] = []
            masks: list[list[int]] = []
            for row in encoded:
                padding = width - len(row)
                if self.padding_side == "left":
                    ids.append([0] * padding + row)
                    masks.append([0] * padding + [1] * len(row))
                else:
                    ids.append(row + [0] * padding)
                    masks.append([1] * len(row) + [0] * padding)
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, *, input_ids: Any, attention_mask: Any, use_cache: bool) -> Any:
            del attention_mask, use_cache
            vocabulary = torch.arange(128, device=input_ids.device, dtype=torch.float32)
            centers = (input_ids.unsqueeze(-1) + 17) % 128
            logits = -(vocabulary.view(1, 1, -1) - centers.float()).abs() + self.anchor * 0.0
            return SimpleNamespace(logits=logits)

    toy_prompts = ["Short verifier prompt.", "A much longer verifier prompt with deliberately different padding."]
    for padding_side in ("left", "right"):
        toy_tokenizer = ToyTokenizer(padding_side)
        toy_model = ToyModel()
        individual = [
            score_forced_choice_options(toy_model, toy_tokenizer, item, device="cpu", microbatch_size=2)
            for item in toy_prompts
        ]
        grouped = score_forced_choice_prompt_batch(
            toy_model,
            toy_tokenizer,
            toy_prompts,
            device="cpu",
            sequence_microbatch_size=4,
        )
        for expected, observed in zip(individual, grouped, strict=True):
            assert expected.argmax_option == observed.argmax_option
            assert expected.option_scoring_tokens == observed.option_scoring_tokens
            for label in OPTION_LABELS:
                assert abs(expected.option_logprobs[label] - observed.option_logprobs[label]) < 1e-6
                assert abs(expected.option_posteriors[label] - observed.option_posteriors[label]) < 1e-8
    print("forced_choice_verifier_self_test=PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure utilities for an auditable A--D forced-choice verifier.")
    parser.add_argument("--self-test", action="store_true", help="Run parser/template/posterior tests without a model or GPU.")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    parser.error("Choose --self-test, or import this module from a prospective collector.")


if __name__ == "__main__":
    main()
