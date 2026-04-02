# Non-Qwen Family Selection

## Objective

Select the strongest genuinely non-Qwen family that is practical on the available 22.5 GB NVIDIA L4 under the existing hidden-state and token-level instrumentation.

## Candidate Audit

| Priority | HF model id | Family | Params | Access result | Chat template | Hidden states and token-level outputs | Quantization path | L4 fit with instrumentation | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `google/gemma-2-9b-it` | Gemma 2 instruct | 9B | blocked | yes on model card | expected yes via `AutoModelForCausalLM`, but runtime access blocked | model card documents `bitsandbytes` 4-bit and 8-bit usage | not benchmarked because the runtime cannot access weights without accepting the gated license | Real runtime test returned a gated-repo `401` when fetching `config.json`. |
| 2 | `meta-llama/Llama-3.1-8B-Instruct` | Llama 3.1 instruct | 8B | blocked | yes on model card | expected yes via `AutoModelForCausalLM`, but runtime access blocked | model card documents Transformers support and quantized ecosystem variants | not benchmarked because the runtime cannot access weights without accepting the gated license | Real runtime test returned a gated-repo `401` when fetching `config.json`. |
| 3 | `mistralai/Mistral-7B-Instruct-v0.3` | Mistral instruct | 7B | accessible | yes | yes: runtime config/tokenizer load succeeded, architecture is `MistralForCausalLM`, and the harness already extracts hidden states plus token logprobs from standard causal-LM outputs | supported through the repo's existing `4bit` and `8bit` bitsandbytes loader plus documented BF16 Transformers path | pending direct L4 benchmark, but selected as the first feasible candidate | Open Apache-2.0 model, standard Transformers chat-template support, and no access-gating blocker in this environment. |

## Real Runtime Evidence

The configured Colab environment attempted direct `AutoConfig.from_pretrained()` and `AutoTokenizer.from_pretrained()` access for all three candidates.

- `google/gemma-2-9b-it`: failed with `OSError` and gated-repo `401`.
- `meta-llama/Llama-3.1-8B-Instruct`: failed with `OSError` and gated-repo `401`.
- `mistralai/Mistral-7B-Instruct-v0.3`: succeeded; `model_type=mistral`, `architectures=['MistralForCausalLM']`, `hidden_size=4096`, `chat_template=True`.

## Selected Model

- Selected alias: `mistral_7b_instruct_v0p3`
- Exact HF model id: `mistralai/Mistral-7B-Instruct-v0.3`
- Family: Mistral instruct
- Parameter count: 7B
- Hidden-state access: yes through `output_hidden_states=True`
- Token-level outputs: yes through standard logits, log-softmax scoring, and decoded completions
- Prompt compatibility: yes through `tokenizer.apply_chat_template`
- Quantization path: compatible with the current repo loader for `4bit`, `8bit`, and unquantized execution
- Final benchmarked L4 configuration: unquantized (`none`) + `sdpa` + batch size `4`
- Benchmark evidence: full precision at batch size `4` delivered the best stable example throughput among the tested configurations, while `flash_attention_2` was unavailable and 8-bit underperformed

## Decision

Mistral 7B instruct is the chosen non-Qwen family for this autonomous cycle because it is the first candidate in the priority order that is both genuinely non-Qwen and actually downloadable in the live Colab runtime. Gemma and Llama remain preferred families in principle, but they are blocked here by access gating rather than by a harness defect.