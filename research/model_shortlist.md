# Model Shortlist

## Environment Audit

- OS: Windows
- Python environment: local virtualenv
- Active backend for executed pilots: `transformers + torch(cpu)`
- GPU present: GTX 1650 4 GB
- Practical blocker: the installed PyTorch build in this workspace is CPU-only, so larger reasoning models are runtime-limited even when weights are downloadable.

## Actual Execution Status

| Model | Family role | Params | Backend | Hidden states | Logprobs | Trace access | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen2.5-0.5B-Instruct` | Small executable control | 0.5B | `transformers+torch(cpu)` | Yes | Yes | Yes, via controlled incremental protocol | Completed | Produced real traces and diagnostics, but stayed in a low-skill regime. |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Preferred reasoning-family distill | 1.5B | `transformers+torch(cpu)` | Yes | Yes | Yes, via the same harness | Downloaded and initialized; meaningful pilot blocked | CPU generation was too slow for a useful local pilot. |

## Planned But Not Run

| Model | Reason to care | Local status |
| --- | --- | --- |
| `Qwen/Qwen2.5-1.5B-Instruct` | Stronger open Qwen control than the 0.5B model | Likely feasible only as a very small CPU pilot; not run in this iteration. |
| `meta-llama/Llama-3.2-1B-Instruct` or best accessible Llama substitute | Needed for cross-family control | Not run; likely limited by access friction plus CPU runtime. |
| `QwQ-32B` | Preferred alternative reasoning family | Infeasible on this workstation. |
| `Llama 3.3 70B Instruct` | Preferred large-family control | Infeasible on this workstation. |
| `Gemma 3`, `Mistral Small`, `Phi` reasoning variants | Medium instrumentation targets | Some may be runnable with more engineering, but not within the current CPU-only envelope. |

## Current Recommendation

The harness is ready. The next materially better empirical step is not more tuning of 0.5B controls; it is running either the DeepSeek 1.5B distill or a 1B-1.5B Qwen/Llama-family model in an environment with CUDA-enabled PyTorch or a quantized inference backend.
