# Frontier Phase 2 Execution Status — 2026-04-04

## Summary

The verified L4 software baseline is now enforced by the runner, and two frontier-family smoke validations succeeded in the live runtime:

- `gemma_4_e4b_it` (`google/gemma-4-E4B-it`)
- `qwen_3p5_9b` (`Qwen/Qwen3.5-9B`)

The full Phase 2 frontier mission could **not** be completed on this L4 because the remaining frontier targets failed live feasibility checks:

- `qwen_3p5_35b_moe_it` downloaded successfully but did not fit in 4-bit NF4 on the L4.
- Official Llama 4 instruct repos are gated in this runtime, and public Scout mirrors are too large for the L4 memory envelope.

## Environment Baseline

Verified in the live runtime after upgrading dependencies:

| Package | Version |
| --- | --- |
| `torch` | `2.10.0+cu128` |
| `transformers` | `5.5.0` |
| `accelerate` | `1.13.0` |
| `bitsandbytes` | `0.49.2` |
| `evaluate` | `0.4.6` |

GPU baseline:

- GPU: `NVIDIA L4`
- VRAM: `23.66 GB`
- CUDA: `12.8`
- BF16: `True`
- Flash SDP: `True`

## Code and Config Changes Applied

1. `tools/run_colab_experiment.py`
   - Enforces minimum package versions instead of only checking for module presence.
   - Fails fast when `--skip-install` is used against an invalid baseline.
   - Adds explicit public aliases for `qwen_3p5_9b` and `llama_4_scout_17b_it`.

2. `requirements-colab.txt`
   - Updated to the audited minimums.
   - Removed the stale standalone `flash-attn` requirement.

3. `research/real_trace_experiments.py`
   - Corrected the Qwen 3.5 base alias to the live public `Qwen/Qwen3.5-9B` target.
   - Added a public Llama 4 Scout alias for runtime probing.

## Successful Smoke Runs

### 1. Gemma 4 Edge 4B

Command:

```bash
python3 tools/run_colab_experiment.py \
  --model gemma_4_e4b_it \
  --smoke-only \
  --smoke-output-dir research/outputs/real_traces_colab_smoke_gemma_4_e4b_it \
  --io-threads 4 \
  --attn-implementation sdpa \
  --skip-install --skip-simulator
```

Observed runtime indicators:

- Precision log present: `bfloat16 detected for 4B model`
- Attention log present: `sdpa (native FA2 kernels via SDPA)`
- Async I/O log present
- `hidden_state_write_seconds`: `0.0006`, `0.0003`

Integrity checks:

- `gemma_4_e4b_it__calendar_offset_easy__temp0.10__seed7.npz`: shape `(2, 2560)`, NaN `False`, Inf `False`, L2 shift `30.54`
- `gemma_4_e4b_it__fraction_remaining__temp0.10__seed7.npz`: shape `(2, 2560)`, NaN `False`, Inf `False`, L2 shift `18.90`

Artifacts:

- `research/outputs/real_traces_colab_smoke_gemma_4_e4b_it`
- `research/outputs/real_traces_colab_smoke_gemma_4_e4b_it.zip`

### 2. Qwen 3.5 Public Base (9B)

Command:

```bash
python3 tools/run_colab_experiment.py \
  --model qwen_3p5_9b \
  --smoke-only \
  --smoke-output-dir research/outputs/real_traces_colab_smoke_qwen_3p5_9b \
  --io-threads 4 \
  --attn-implementation sdpa \
  --skip-install --skip-simulator
```

Observed runtime indicators:

- Precision log present: `bfloat16 detected for 9B model`
- Attention log present: `sdpa (native FA2 kernels via SDPA)`
- Async I/O log present
- `hidden_state_write_seconds`: `0.0006`, `0.0002`

Integrity checks:

- `qwen_3p5_9b__calendar_offset_easy__temp0.10__seed7.npz`: shape `(2, 4096)`, NaN `False`, Inf `False`, L2 shift `45.39`
- `qwen_3p5_9b__fraction_remaining__temp0.10__seed7.npz`: shape `(2, 4096)`, NaN `False`, Inf `False`, L2 shift `18.35`

Artifacts:

- `research/outputs/real_traces_colab_smoke_qwen_3p5_9b`
- `research/outputs/real_traces_colab_smoke_qwen_3p5_9b.zip`

## Blockers Verified in Live Runtime

### 1. Qwen 3.5 35B A3B Does Not Fit the L4 in NF4

Command attempted:

```bash
python3 tools/run_colab_experiment.py \
  --model qwen_3p5_35b_moe_it \
  --quantization 4bit \
  --smoke-only \
  --smoke-max-tasks 1 \
  --smoke-output-dir research/outputs/real_traces_colab_smoke_qwen_3p5_35b_moe_it \
  --io-threads 4 \
  --attn-implementation sdpa \
  --skip-install --skip-simulator
```

Observed result:

- The model downloaded successfully.
- Total downloaded weight size reached approximately `71.9 GB`.
- Transformers then failed during 4-bit load with:

```text
ValueError: Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the quantized model.
```

Interpretation:

- The `Qwen/Qwen3.5-35B-A3B` target exceeded the L4 NF4 memory envelope in the live stack.
- No smoke artifacts were produced.

### 2. Llama 4 Public and Official Targets Are Not Viable on This L4

Access probes:

- `meta-llama/Llama-4-Scout-17B-16E-Instruct`: gated `401` in this runtime.
- `chutesai/Llama-4-Scout-17B-16E-Instruct`: public, config and processor load succeed.
- `unsloth/Llama-4-Scout-17B-16E-Instruct`: public, config and processor load succeed.
- `unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit`: public, but config validation failed under `transformers 5.5.0`.

Size probes via `HfApi.model_info(..., files_metadata=True)`:

- `chutesai/Llama-4-Scout-17B-16E-Instruct`: approximately `217.32 GB`
- `unsloth/Llama-4-Scout-17B-16E-Instruct`: approximately `217.32 GB`
- `unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit`: approximately `60.44 GB`

Interpretation:

- There is no live public `Llama-4-8B-Instruct` target to match the original plan.
- The available public Scout mirrors are too large for a single L4, even before runtime overhead.
- The public pre-quantized Scout variant does not currently pass config validation in the verified Transformers stack.

## Operational Conclusion

The current workspace is ready for frontier-family experiments that fit the L4, but the exact Phase 2 mission in its original form is blocked by model availability and memory feasibility:

- `Gemma 4` is validated for smoke and has a verified 31B audit path.
- `Qwen 3.5` is validated at the public 9B base target, but the 35B A3B model does not fit the L4 in NF4.
- `Llama 4` cannot be completed on this L4 using currently accessible public repos.

## Required Next Steps

At least one of the following is required before the full frontier validation can honestly be claimed complete:

1. A larger GPU class for Qwen 35B and Llama 4 Scout.
2. A smaller public Llama 4 target that actually exists and is compatible with the current harness.
3. Authenticated access to an official Llama 4 repo plus hardware sized for the real weight footprint.
4. A revised Phase 2 protocol that treats `Qwen/Qwen3.5-9B` and a non-Llama fallback as the frontier-family coverage set.