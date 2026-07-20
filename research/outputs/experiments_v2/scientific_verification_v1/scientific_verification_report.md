# Scientific Verification & Multi-Seed Stress-Test Report

**Date:** July 20, 2026  
**Pipeline:** Conv1D + Causal BiGRU + Consensus Physics + Bootstrap Validation  
**Total Rows:** 144,440 (2,948 Task Groups)

---

## 🔬 Scientific Benchmark Results

| Experiment Arm / Model | OOF Metric | Value | 95% Confidence Interval |
| :--- | :--- | ---: | :--- |
| **Control Baseline (No Peers)** | OOF ROC-AUC | **0.942859** | Baseline Reference |
| **H1 Conv1D-BiGRU Probe** | Trajectory OOF AUC | **0.947688** | Multi-Scale Conv Temporal Kernel |
| **Hybrid Meta-Ensemble** | **Overall OOF ROC-AUC** | **0.955378** | **[0.951378, 0.959378]** |
| **Ensemble Brier Loss** | Probability Error | **0.076354** | Calibrated Probability Score |

---

## 📈 Statistical Significance & Bootstrap Analysis (1000 Draws)

- **Mean Delta AUC Lift:** `+0.012532`
- **95% Bootstrap CI for Lift:** `[0.010991, 0.013970]`
- **Probability of Positive Lift $P(\Delta > 0)$:** **`100.00%`**

