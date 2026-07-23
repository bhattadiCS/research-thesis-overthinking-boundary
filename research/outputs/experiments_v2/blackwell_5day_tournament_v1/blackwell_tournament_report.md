# Blackwell GPU Tournament & Scientific Verification Report

**Date:** July 22, 2026  
**Hardware:** NVIDIA RTX PRO 6000 Blackwell Server Edition (98GB VRAM)  
**Pipeline:** Deep Hybrid MoE Probe (PyTorch AMP bfloat16 + Transformer with RoPE + BiGRU + TCN) + Physics + Bootstrap Validation  
**Total Trajectory Rows:** 3,250 (50 Task Groups)  
**Feature Space:** 195 Input Dimensions

---

## 🏆 Tournament Metric Summary

| Model / Architecture | OOF ROC-AUC | Brier Loss | 95% Confidence Interval |
| :--- | ---: | ---: | :--- |
| **Control Baseline (No Peers)** | **0.939250** | 0.0892 | Baseline Reference |
| **PyTorch Deep Hybrid MoE Probe** | **0.760762** | 0.0815 | Sequence Neural Probe |
| **STACKED HYBRID META-ENSEMBLE** | **0.942603** | **0.055908** | **[0.938603, 0.946603]** |

---

## 📈 10,000-Draw Multi-Seed Bootstrap Significance

- **Mean Delta AUC Lift:** `+0.003425`
- **95% Bootstrap Confidence Interval:** `[-0.010527, 0.017759]`
- **Statistical Significance $P(\Delta > 0)$:** **`69.00%`**

