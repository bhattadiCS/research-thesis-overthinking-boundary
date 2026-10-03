"""Local Hugging Face adapter for actual prefix-only online generation.

Each generation call produces one incremental reasoning step.  A JSON boundary
criterion ends that call when the required object is complete.  The controller
then either schedules the next step or removes the task permanently.  The same
boundary criterion is used by never-stop and active policies.  External aborts
are checked after every decoded token, including inside a running HF call.
Imports/model loading are intentionally separate from the lightweight policy.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from online_stopping_controller import CancellationToken, Observation, PublicTask


def complete_json_object(text: str) -> dict[str, Any] | None:
    """Strict telemetry parser; fallback defaults must not authorize stopping."""

    for offset, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or set(("thought", "answer", "confidence", "stop")) - value.keys():
            continue
        if not isinstance(value["thought"], str) or not isinstance(value["answer"], str) or not value["answer"].strip():
            continue
        if isinstance(value["confidence"], bool) or not isinstance(value["confidence"], int) or not 0 <= value["confidence"] <= 100:
            continue
        if not isinstance(value["stop"], bool):
            continue
        return value
    return None


def reasoning_prompt(task: PublicTask, history: Sequence[Observation], step: int, max_steps: int) -> str:
    prefix = "\n".join(
        f"Step {o.step}: thought={o.thought} | answer={o.answer} | confidence={o.confidence}"
        for o in history
    ) or "No previous steps."
    return (
        f"Task: {task.prompt}\n\n"
        f"You are at incremental reasoning step {step} of {max_steps}. "
        "Do one short reasoning or verification step and update your current answer. "
        "Use one short sentence for thought; do not repeat a full solution.\n"
        f"Previous steps:\n{prefix}\n\n"
        "Return exactly one JSON object, with a numeric integer confidence from 0 to 100 and a boolean stop. "
        'Format example: {"thought":"A brief verification.","answer":"25","confidence":90,"stop":false}\n'
        "The answer field must contain only the current final answer."
    )


@dataclass(frozen=True)
class BatchMetrics:
    batch_size: int
    prompt_tokens: int
    generated_tokens: int
    padded_prefill_token_slots: int
    decode_token_slots: int
    model_seconds: float
    tokenize_seconds: float


class HuggingFaceStepGenerator:
    """No downloads, services, diagnostics pass, or hidden verifier work.

    ``model_path`` must be an existing complete local snapshot. ``dtype`` uses
    float16 on CUDA and float32 on CPU. No quantization is performed. Report the
    small-model results as development evidence, not confirmation of a 13-model
    committee or the retrospective ensemble.
    """

    def __init__(self, model_path: str | Path, *, device: str = "cuda", max_steps: int = 5, max_new_tokens: int = 128) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        path = Path(model_path).resolve()
        if not path.is_dir() or not (path / "config.json").is_file():
            raise FileNotFoundError("a complete existing local model snapshot is required")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.model_path, self.device = path, device
        self.max_steps, self.max_new_tokens = max_steps, max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path), local_files_only=True,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            attn_implementation="sdpa",
        ).to(device).eval()
        self._inflight: tuple[CancellationToken, ...] = ()
        self.metrics: list[BatchMetrics] = []

    def cancel(self) -> None:
        for token in self._inflight:
            token.cancel()

    def generate_step(self, task: PublicTask, history: Sequence[Observation], step: int, cancellation: CancellationToken) -> Observation:
        return self.generate_batch([task], [history], step, [cancellation])[0]

    def generate_batch(
        self, tasks: Sequence[PublicTask], histories: Sequence[Sequence[Observation]], step: int,
        cancellations: Sequence[CancellationToken],
    ) -> list[Observation]:
        from transformers import StoppingCriteria, StoppingCriteriaList
        from real_trace_experiments import parse_generation

        if not tasks or len(tasks) != len(histories) or len(tasks) != len(cancellations):
            raise ValueError("a nonempty aligned generation batch is required")
        if any(token.cancelled for token in cancellations):
            raise InterruptedError("cannot schedule a cancelled generation")
        torch, tokenizer = self.torch, self.tokenizer
        started = time.perf_counter()
        prompts = [tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are a careful mathematical reasoning assistant. Follow the requested JSON format."},
             {"role": "user", "content": reasoning_prompt(task, history, step, self.max_steps)}],
            tokenize=False, add_generation_prompt=True,
        ) for task, history in zip(tasks, histories)]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        prompt_counts = encoded["attention_mask"].sum(dim=1).tolist()
        width = encoded["input_ids"].shape[1]
        tokenize_seconds = time.perf_counter() - started
        stopped_lengths: list[int | None] = [None] * len(tasks)

        class CompletedStepOrCancelled(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
                done = []
                for i, row in enumerate(input_ids):
                    length = int(row.shape[0] - width)
                    completed = stopped_lengths[i] is not None
                    if not completed:
                        is_eos = int(row[-1]) == tokenizer.eos_token_id
                        completed = cancellations[i].cancelled or is_eos or complete_json_object(
                            tokenizer.decode(row[width:], skip_special_tokens=True)
                        ) is not None
                        if completed:
                            stopped_lengths[i] = length
                    done.append(completed)
                return torch.tensor(done, device=input_ids.device, dtype=torch.bool)

        self._inflight = tuple(cancellations)
        if self.device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded, max_new_tokens=self.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([CompletedStepOrCancelled()]),
                    use_cache=True,
                )
            if self.device == "cuda":
                torch.cuda.synchronize()
            model_seconds = time.perf_counter() - started
        finally:
            self._inflight = ()
        if any(token.cancelled for token in cancellations):
            raise InterruptedError("generation aborted during token decoding")
        completion_width = int(generated.shape[1] - width)
        observations = []
        for i, task in enumerate(tasks):
            length = stopped_lengths[i] if stopped_lengths[i] is not None else completion_width
            completion = generated[i, width:width + length]
            raw_text = tokenizer.decode(completion, skip_special_tokens=True)
            strict = complete_json_object(raw_text)
            parsed = strict or parse_generation(raw_text, task.answer_type, "minimal_json")
            observations.append(Observation(
                step=step, answer=str(parsed.get("answer", "")).strip(),
                confidence=float(strict["confidence"]) if strict else None,
                parse_success=strict is not None, generated_tokens=int(length),
                prompt_tokens=int(prompt_counts[i]), observed_ns=time.monotonic_ns(),
                thought=str(parsed.get("thought", "")), raw_text=raw_text,
                model_stop_flag=bool(parsed.get("stop", False)),
            ))
        self.metrics.append(BatchMetrics(
            len(tasks), int(sum(prompt_counts)), sum(o.generated_tokens for o in observations),
            int(width * len(tasks)), int(completion_width * len(tasks)), model_seconds, tokenize_seconds,
        ))
        return observations
