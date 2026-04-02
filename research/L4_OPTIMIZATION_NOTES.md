# L4 Optimization Notes

## Hardware

- Runtime: Google Colab single NVIDIA L4
- Reported capacity: 23034 MiB total VRAM
- Benchmark target model: `mistralai/Mistral-7B-Instruct-v0.3`
- Protocol slice for optimization: GSM8K train, shuffle seed `17`, temperature `0.1`, seed `7`, `max_steps=2`, `max_new_tokens=256`

## Benchmark Artifacts

- Backend scan: `research/outputs/benchmark_mistral7b_l4_attn_scan`
- 4-bit batch ladder: `research/outputs/benchmark_mistral7b_l4_4bit_ladder`
- Full-precision batch ladder: `research/outputs/benchmark_mistral7b_l4_fp_ladder`

## Backend Scan

The first pass compared `4bit`, `8bit`, and full precision at batch size `2` across `sdpa`, `auto`, and `flash_attention_2`.

- `flash_attention_2` was not usable in this runtime because the `flash_attn` package is not installed. After removing silent CPU fallback, all `flash_attention_2` attempts failed fast and were correctly recorded as failures.
- `4bit` with `sdpa` and `auto` were nearly identical at batch size `2`; `sdpa` held a slight edge and was kept for the 4-bit ladder.
- `8bit` was materially slower than both `4bit` and full precision at the same batch size while also showing much lower mean GPU utilization, so it was screened out before a deeper batch ladder.
- Full precision with `sdpa` outperformed both quantized alternatives at batch size `2`, so it advanced to the full-precision ladder.

## Batch Ladder Results

### 4-bit `sdpa`

| Batch size | Mean examples/s | Mean tokens/s | Peak reserved VRAM | Mean GPU util | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| 2 | 0.1121 | 11.81 | 13.52 GB | 88.22% | stable |
| 4 | 0.1862 | 19.65 | 13.43 GB | 82.66% | stable |
| 6 | 0.2346 | 25.95 | 13.42 GB | 80.35% | best 4-bit point |
| 8 | 0.2224 | 25.03 | 13.43 GB | 80.43% | stable but slower than batch 6 |

### Full precision `sdpa`

| Batch size | Mean examples/s | Mean tokens/s | Peak reserved VRAM | Mean GPU util | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 0.1447 | 15.14 | 14.09 GB | 86.48% | stable |
| 2 | 0.2106 | 22.43 | 14.46 GB | 82.89% | stable |
| 4 | 0.3497 | 35.61 | 15.27 GB | 75.97% | best overall |
| 6 | 0.3192 | 38.23 | 16.25 GB | 76.09% | stable but worse by examples/s |

### 8-bit `sdpa`

The backend scan result at batch size `2` was enough to reject 8-bit from the final ladder.

- Mean examples/s: `0.0695`
- Mean tokens/s: `7.86`
- Peak reserved VRAM: `13.24 GB`
- Mean GPU utilization: `28.21%`
- Assessment: slower than both 4-bit and full precision at the same batch size, with noticeably lower GPU utilization, so it was not the best stable throughput path.

## Bottleneck Diagnosis

Successful configurations were compute-bound.

- Model generation time dominated every successful run.
- Hidden-state extraction added a smaller but visible cost.
- CPU preprocessing and hidden-state write time were negligible relative to generation time.
- The chosen configuration therefore did not leave performance on the table because of disk writes or tokenization overhead.

## Final Runtime Choice

- Model: `mistral_7b_instruct_v0p3`
- Quantization: `none`
- Attention backend: `sdpa`
- Batch size: `4`
- Device: `cuda`
- Prompt mode: `minimal_json`
- System prompt mode: `default`

This configuration is the best stable throughput path observed on the L4 for the matched harness. It does not consume the entire 22.5 GB card, but that is the correct outcome here: moving from batch size `4` to `6` increased memory usage while reducing per-example throughput, so additional VRAM consumption did not improve the objective that matters for the 300-task run.

## Reproducible Full-Run Command

```bash
python3 research/real_trace_experiments.py \
  --model mistral_7b_instruct_v0p3 \
  --device cuda \
  --quantization none \
  --attn-implementation sdpa \
  --max-tasks 300 \
  --max-steps 10 \
  --max-new-tokens 256 \
  --task-source gsm8k \
  --dataset-split train \
  --dataset-shuffle-seed 17 \
  --batch-size 4 \
  --prompt-mode minimal_json \
  --system-prompt-mode default \
  --output-dir research/outputs/real_traces_l4_mistral_7b
```