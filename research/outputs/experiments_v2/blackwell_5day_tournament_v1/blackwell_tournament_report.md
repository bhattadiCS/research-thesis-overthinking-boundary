# Blackwell GPU Tournament & Scientific Verification Report

**Date:** July 22, 2026  
**Hardware:** NVIDIA RTX PRO 6000 Blackwell Server Edition (98GB VRAM)  
**Pipeline:** Deep Hybrid MoE Probe (PyTorch AMP bfloat16 + Transformer with RoPE + BiGRU + TCN) + Physics + Bootstrap Validation  
**Total Trajectory Rows:** 144,440 (2,948 Task Groups)  
**Feature Space:** 225 Input Dimensions

---

## 🏆 Tournament Metric Summary

| Model / Architecture | OOF ROC-AUC | Brier Loss | 95% Confidence Interval |
| :--- | ---: | ---: | :--- |
| **Control Baseline (No Peers)** | **0.943223** | 0.0892 | Baseline Reference |
| **PyTorch Deep Hybrid MoE Probe** | **0.936430** | 0.0815 | Sequence Neural Probe |
| **STACKED HYBRID META-ENSEMBLE** | **0.955156** | **0.076396** | Bootstrap lift CI in the JSON: **[0.010384, 0.013480]** |

---

## 📈 10,000-Draw Multi-Seed Bootstrap Significance

- **Mean Delta AUC Lift:** `+0.011937`
- **95% Bootstrap Confidence Interval:** `[0.010384, 0.013480]`
- **Empirical bootstrap proportion $\Pr^*(\Delta > 0)$:** **`100.00%`**

The persisted interval is for the task-clustered AUC lift over the control, not a standalone confidence interval for the absolute stacked AUC. The reported proportion is an empirical bootstrap proportion, not a conventional hypothesis-test $p$-value.

