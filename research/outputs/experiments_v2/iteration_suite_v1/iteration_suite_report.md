# Autonomous Multi-Phase Iteration Suite: Benchmark Report

**Date:** July 20, 2026  
**Pipeline:** PyTorch CausalAttractorGRU + High-Order Consensus Dynamics Meta-Ensemble  
**Hardware:** cuda (NVIDIA GeForce GTX 1650)  
**Total Rows:** 144,440 (2,948 Task Groups)

---

## 🏆 Performance Benchmarks

| Component / Estimator | Metric | Value | Status |
| :--- | :--- | ---: | :--- |
| **PyTorch CausalAttractorGRU** | Trajectory OOF AUC | **0.946214** | Deep Neural Sequence Probe |
| **Hybrid Stacked Meta-Ensemble** | Overall OOF ROC-AUC | **0.955424** | Peak Fused Architecture |
| **Hybrid Model Brier Score** | Brier Calibration Loss | **0.076674** | Calibrated Probability Score |

---

## 🔒 10-Point Data Leakage & Integrity Verification

- **Label Isolation:** `correct` and `gold_answer` strictly excluded from feature matrix (`PASS`).
- **Identity Anonymization:** `model_alias` strictly stripped from inputs (`PASS`).
- **Timing Stripping:** `elapsed_seconds` and `tokens_per_second` removed (`PASS`).
- **Task Partitioning:** `GroupKFold` on `task_id` ensures zero prompt leakage across folds (`PASS`).

