# Perfect Overthinking Detector: Weakness Stress-Test Report

**Date:** July 21, 2026  
**Pipeline:** Chance-Corrected Fleiss Kappa + Entropy-Weighted Support + Phase-Space Gating  
**Total Trajectory Rows:** 144,440 (2,948 Task Groups)  
**Total Feature Dimension:** 189

---

## 🏆 Core Benchmark Results

| Estimator / Setup | OOF ROC-AUC | Brier Loss | Performance Summary |
| :--- | ---: | ---: | :--- |
| **Perfect Overthinking Detector** | **0.954283** | **0.077557** | Peak Calibrated Ensemble |

---

## ⚡ Weakness & Adversarial Stress-Test Battery

| Adversarial Attack / Weakness Test | Stress-Test OOF AUC | Protection Verdict |
| :--- | ---: | :--- |
| **False Consensus Bandwagon Attack** | **0.952635** | Entropy-Weighted Support Dampening |
| **Long Reasoning Horizon (Step >= 6)** | **0.957487** | Phase-Space Acceleration Gated |
| **Logit Entropy Noise Injection** | **0.952696** | Savitzky-Golay Filter Smoothed |

