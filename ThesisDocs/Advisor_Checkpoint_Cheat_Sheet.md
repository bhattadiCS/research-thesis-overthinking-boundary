# Advisor Checkpoint: 5-Minute Presentation Cheat Sheet
*Use this cheat sheet to deliver a concise, high-impact 5-minute update to Dr. Woods, leaving 10 minutes for discussion.*

---

## ⏱️ 5-Minute Talk Script (Step-by-Step)

### Step 1: Progress & Engineering Breakthroughs (Time: 60s)
* **What to say:** 
  > *"Since our July 1 meeting, we have formalized our entire framework to strictly follow the scientific method, completed our global sweep, and accelerated our experiments on our new RTX Blackwell GPU."*
* **The Highlights (Bullet Points):**
  * **3.5x Sweep Speedup:** Optimized batch sizes and restricted max tasks to 500 per cell, reducing runtimes from 22 hours to under 8 hours.
  * **System Hardening:** Wrapped math evaluation engines in 5-second timers to prevent hangs on garbage LLM output, and hardened trace saving against VM restarts.
  * **Global Sweep Complete:** Logged **19,948 reasoning paths** (147,740 individual steps) across 52 experimental cells.

### Step 2: The Variables Map & Scientific Rigor (Time: 60s)
* **What to say:** 
  > *"We mapped our experiment to a strict variable architecture to isolate confounding factors. The key takeaway here is our finding on **Quantization Invariance**."*
* **The Highlights (Bullet Points):**
  * **Knobs We Turn (Inputs):** Step budget, Temperature (randomness), Model Scale, Quantization (compression), and Model Family.
  * **What We Measure (Outputs):** Correctness ($q_t$), Overthinking Drift ($D$), Detector AUC, Latency, and Utility.
  * **Quantization Invariance (Key Finding):** Compressing models from 16-bit to 4-bit weights lowers accuracy slightly, but preserves the *shape* of the hidden state trajectories. Our stopping detectors transfer from uncompressed to compressed models with zero loss in prediction AUC (~0.86).

### Step 3: The Breakthrough Results (Time: 120s)
* **What to say:** 
  > *"We ran a cross-validation tournament over our 19,948 reasoning paths to evaluate different stopping methods. Our main contribution is showing that sequence-based models outperform static baselines."*
* **The Highlights (Bullet Points):**
  * Show him the **Tournament Table**:
    * *Baseline (Linear Logistic Regression):* **0.7380 AUC**
    * *LSTM (Sequence Model):* **0.8714 AUC** (a **+13.3%** boost!)
    * *Gated Self-Consistency (Traditional consensus):* **0.8686 AUC** (but requires generating 5-10 parallel paths, making it extremely expensive).
  * **Why sequence models win:** Overthinking is a gradual process of logical drift. A static linear probe looks only at the current step snapshot (like guessing a movie plot from a single frame). The LSTM reads the entire history of step transitions, making it highly accurate in predicting when a model is about to make a mistake on a single run.

---

## 🙋‍♂️ Expected Questions from Dr. Woods & Quick Answers

Use these direct, simple answers to handle his queries confidently:

### Q1: Why do sequence models (LSTM/GRU) beat static linear classifiers?
* **Answer:** *"Overthinking is not a sudden, random event; it is a gradual build-up of logical drift. A linear classifier only sees the current step's snapshot. The LSTM tracks the history of steps, mapping the model's confidence trajectory over time to detect when it starts to wander off course."*

### Q2: Why does weight compression (quantization) not break the stopping detector?
* **Answer:** *"Quantization introduces background noise that lowers a model's base accuracy, but it does not change how the model reasons. The relative trajectory shape of the model's hidden states remains intact. Because our LSTM classifies the shape of the trajectory rather than the raw weight values, it generalizes perfectly across compression levels."*

### Q3: Why does model scale affect overthinking?
* **Answer:** *"Tiny models (0.5B) make random errors and drift quickly due to lack of capacity. Large models (32B) have high reasoning stability, but when they do overthink, they generate highly logical-sounding, systematic justifications for their wrong answers. That is why a sequence model is required to catch them—it isolates the structural pattern of the reasoning path."*

### Q4: How is this better than the standard industry method (Self-Consistency / voting)?
* **Answer:** *"Voting works, but it requires generating 5 to 10 full reasoning paths and comparing them, which is 5x to 10x more expensive. Our LSTM stops the model midway on a **single reasoning path**, saving massive compute costs while achieving higher accuracy."*

---

## 📌 Interactive Slide Links for Presentation
* 📊 **[Interactive Thesis Dashboard](file:///C:/Aditya_Data/Personal/ResearchThesis/ThesisDocs/thesis_presentation_dashboard.html)** (Toggle variables live to show ceteris paribus effects).
* 📄 **[Detailed Progress Report](file:///C:/Aditya_Data/Personal/ResearchThesis/ThesisDocs/July_16_Progress.md)** (Contains full variable breakdowns, Mermaid charts, and generated matplotlib plots).
