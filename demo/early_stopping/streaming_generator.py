"""Token updates around the repository's existing incremental HF adapter.

This demo extension leaves the frozen research adapter and model bytes intact.
The stopping policy still acts only after each completed incremental response.
"""
from __future__ import annotations

import time
from typing import Callable, Sequence

from online_generation import HuggingFaceStepGenerator, complete_json_object, reasoning_prompt
from online_stopping_controller import CancellationToken, Observation, PublicTask


class StreamingStepGenerator(HuggingFaceStepGenerator):
    def set_callback(self, callback: Callable[[str, int], None] | None) -> None:
        self._callback = callback

    def generate_step(self, task: PublicTask, history: Sequence[Observation], step: int,
                      cancellation: CancellationToken) -> Observation:
        from transformers import StoppingCriteria, StoppingCriteriaList
        from real_trace_experiments import parse_generation

        if cancellation.cancelled:
            raise InterruptedError('Comparison cancelled')
        torch, tokenizer = self.torch, self.tokenizer
        text = tokenizer.apply_chat_template([
            {'role': 'system', 'content': 'You are a careful mathematical reasoning assistant. Follow the requested JSON format.'},
            {'role': 'user', 'content': reasoning_prompt(task, history, step, self.max_steps)},
        ], tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(text, return_tensors='pt').to(self.device)
        width = int(encoded['input_ids'].shape[1])
        callback = getattr(self, '_callback', None)

        class CompletedStepOrCancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                completion = input_ids[0, width:]
                raw = tokenizer.decode(completion, skip_special_tokens=True)
                if callback is not None:
                    callback(raw, int(completion.shape[0]))
                done = (cancellation.cancelled or int(input_ids[0, -1]) == tokenizer.eos_token_id
                        or complete_json_object(raw) is not None)
                return torch.tensor([done], device=input_ids.device, dtype=torch.bool)

        self._inflight = (cancellation,)
        try:
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded, max_new_tokens=self.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([CompletedStepOrCancelled()]),
                    use_cache=True,
                )
            if cancellation.cancelled:
                raise InterruptedError('Comparison cancelled during generation')
        finally:
            self._inflight = ()
        completion = generated[0, width:]
        raw = tokenizer.decode(completion, skip_special_tokens=True)
        if callback is not None:
            callback(raw, int(completion.shape[0]))
        strict = complete_json_object(raw)
        parsed = strict or parse_generation(raw, task.answer_type, 'minimal_json')
        return Observation(
            step=step, answer=str(parsed.get('answer', '')).strip(),
            confidence=float(strict['confidence']) if strict else None,
            parse_success=strict is not None, generated_tokens=int(completion.shape[0]),
            prompt_tokens=int(encoded['attention_mask'].sum().item()),
            observed_ns=time.monotonic_ns(), thought=str(parsed.get('thought', '')),
            raw_text=raw, model_stop_flag=bool(parsed.get('stop', False)),
        )
