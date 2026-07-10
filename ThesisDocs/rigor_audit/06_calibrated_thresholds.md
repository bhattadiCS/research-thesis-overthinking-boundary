# Calibrated Per-Cell Threshold Table — P3d Productization

**Date:** 2026-07-10 · **Generator:** `research/calibrate_cell_thresholds.py` (committed, re-runnable) · **Table:** `research/outputs/experiment_matrix/_aggregate/calibrated_cell_thresholds.csv` (52 cells)
**Validation:** recomputed stops matched the recorded `hazard_drift` stops for all runs (**0 mismatches across 52 cells**); Σ`oof_dU` = **+1,545.85**, agreeing with the committed P3d census ([05_offline_experiment_results.md](05_offline_experiment_results.md)) exactly.

## How to read / use the table (leakage recipe — do not skip)

- **`oof_dU`** (per cell) is the honest, out-of-fold performance estimate: fold thresholds were fit on GroupKFold(5)-by-`task_id` train folds and evaluated on held-out folds. **Quote these.**
- **`delta_full_DEPLOY`** is the deployment value: the offset re-fit on all of the cell's runs after the procedure was validated out-of-fold (standard practice). Its `cal_full_*` companion columns are in-sample by construction — **never** cite them as expected gains.
- The stop rule with calibration is: stop at the first step ≥ 2 with `mu_hat ≤ delta_full_DEPLOY` for the cell.
- **Recalibrate from scratch** after any change to graders, features, the probe/hazard models, λ, or the trace corpus. Never tune δ on runs you then use to claim a gain.

## Global result

| Metric | Baseline | OOF-calibrated |
|---|---|---|
| Net utility vs baseline | — | **+1,545.85** (~44% of the 3,543-utility loss mass) |
| Losses | 5,735 (7.55%) | 6,191 (8.15%) |

The calibration is **utility-optimal, not win-rate-optimal**: it trades ~456 net additional small losses (mostly early-stop ties/wins becoming narrow losses) for large step-cost savings plus 631 dissolved losses. Any advisor-facing use must state both columns together.

## Structure of the fitted thresholds — the thesis-relevant finding

δ mean **+0.080**, median **+0.122**: on 45 of 52 cells the uncalibrated policy **over-thinks** relative to the cell's own cost structure (positive offset = stop sooner). The **7 negative-δ cells (want more patience) are almost exactly the late-boundary cells** from the cross-family story: `llama_3p1_8b_instruct__gsm8k`, `mistral_small_24b_2409__gsm8k`, `qwen2p5_14b__gsm8k`, `qwen2p5_32b__gsm8k`, `yi_1p5_9b_chat__gsm8k`, `qwen2p5_32b__math`, plus `phi_4_mini_instruct__arc`. The calibration independently rediscovers the overthinking-boundary structure: patience pays only where a genuine late boundary exists.

## Notable cells

| Cell | δ (deploy) | OOF ΔU | Loss% base → calibrated |
|---|---|---|---|
| `mistral_small_24b_2409__math` | +0.150 | **+145.00** | 17.00 → 27.13 (utility↑, win-rate↓ — extreme divergence) |
| `mistral_small_24b_2409__gsm8k` | **−0.145** | **+122.65** | **30.27 → 14.00** (improves both metrics — the audit's worst cell) |
| `qwen2p5_3b__math` | +0.150 | +107.80 | 5.47 → 8.20 |
| `yi_1p5_9b_chat__math` | +0.150 | +107.80 | 4.00 → 5.54 |
| worst 3 cells (`deepseek_7b__arc`, `llama__arc`, `deepseek_7b__gsm8k`) | +0.02…+0.08 | −7.75…−14.65 | calibration slightly hurts; consider δ=0 for these |

Caveats: 10/52 cells have fold-δ spans > 0.05 (calibration unstable there — treat their deploy values as low-confidence); 3 cells have negative OOF ΔU (a per-cell deployment rule may floor δ at 0 where OOF ΔU ≤ 0, at the cost of re-pre-registering that variant).
