# Semester 2 Kickoff: Thesis Progress & Roadmap
## Discussion Guide for Informal Advisor Sync (Fall 2026)

**Student:** Aditya Bhatt ([abhatt25@jh.edu](mailto:abhatt25@jh.edu))  
**Research Adviser:** Dr. Zerotti Woods (Johns Hopkins APL / JHU ACM)  
**Second Reader:** Dr. Moustapha Pemy (Towson University / JHU ACM)  
**Degree:** M.S. in Applied and Computational Mathematics (JHU EN.625.804 Thesis)  
**Meeting Style:** Informal checkup & planning sync to kick off the final semester  
**Target Defense Date:** Late November / Early December 2026  
**GitHub Repository:** [`bhattadiCS/research-thesis-overthinking-boundary`](https://github.com/bhattadiCS/research-thesis-overthinking-boundary) (branch `main`)

---

## 🎯 60-Second Elevator Pitch for Dr. Woods

> *"Dr. Woods, over Semester 1 we finished all the heavy experimental runs:
> - We collected and analyzed over **75,000 reasoning steps** across 13 different open-source AI models (Llama, Mistral, Qwen, DeepSeek) on math and reasoning benchmarks.
> - We confirmed that models reach peak accuracy early (at **step 2 or 3**), and that continuing beyond that point causes **overthinking drift**—the model second-guesses itself and changes right answers to wrong ones.
> - We trained a classifier on an NVIDIA Blackwell GPU that tells apart correct and incorrect reasoning paths with **95.5% reliability (0.955 ROC-AUC)**.
>
> For this final semester, we do not need more large compute sweeps. We want to **pull the writing timeline forward immediately** so we have plenty of time for revisions:
> 1. **Start writing right now:** Because our Semester 1 data and experiments are already documented, we can draft Chapters 1–4 this month and deliver **Draft v1.0 by October 9**.
> 2. **Build a live 'stop' button in parallel:** In Weeks 1–3, we turn our detector into a real-time tool that stops models during generation, saving 30% to 40% on compute.
> 3. **4-Week Revision Window:** Submitting Draft v1 by October 9 gives you and Dr. Pemy a full month for feedback, so our late-November defense will be polished and stress-free."*

---

## 📊 Visual Walkthrough: The Core Findings in Plain English

### Figure 1: The Core Problem — Why Thinking Longer Hurts
When AI reasoning models think for too long, they often find the right answer early on, and then ruin it by overthinking.

| (a) Overthinking Drift (Accuracy by Step) | (b) Net Score (Accuracy minus Token Cost) |
| :---: | :---: |
| ![Overthinking Drift](images/overthinking_drift_by_step.png) | ![Stopping Utility](images/stopping_utility_by_step.png) |
| *Accuracy peaks at steps 2–3, then drops by up to 15% as models overthink.* | *Every extra step costs compute ($\lambda = 0.05$). Stopping at the peak saves tokens and protects accuracy.* |

**The Intuition:**
- Every step of thinking costs computing power (for example, 5% of the total budget per step).
- **If the model gets the right answer at Step 2 and stops:** It gets full credit with minimal compute used.
- **If the model keeps rambling to Step 5 and changes to the wrong answer:** It wasted 5 steps of compute and gets zero credit!
- Our goal is to stop at the peak—saving cost and locking in the correct answer.

---

### Figure 2: The Mechanism — What Happens Inside Each Step?

```mermaid
flowchart LR
    subgraph Early["Early Steps (Steps 1 to 2)"]
        direction TB
        E1["Model catches arithmetic slip-ups"]
        E2["Fixing mistakes happens often"]
        E3["Result: Accuracy goes UP"]
        E4["Action: Keep thinking! ✅"]
        E1 --> E3
        E2 --> E3
        E3 --> E4
    end

    subgraph SweetSpot["The Sweet Spot (Step 2 to 3)"]
        direction TB
        S1["Model hits the correct answer"]
        S2["Accuracy peaks here (75% to 80%)"]
        S3["Action: Best time to stop! 🎯"]
        S1 --> S2
        S2 --> S3
    end

    subgraph Late["Late Steps (Steps 4 to 5)"]
        direction TB
        L1["Model doubts its own correct answer"]
        L2["Changes right answers to wrong ones"]
        L3["Result: Accuracy drops & tokens wasted"]
        L4["Action: Stop immediately! 🛑"]
        L1 --> L3
        L2 --> L3
        L3 --> L4
    end

    Early --> SweetSpot --> Late
```

**The Balance in Plain English:**
At every step, two forces compete:
1. **The Fixing Force:** The chance that the model notices and corrects an earlier mistake.
2. **The Breaking Force:** The chance that the model doubts itself and ruins a good answer.

- In **Steps 1 and 2**, the fixing force is strong: the model checks its work and catches early slips.
- In **Steps 4 and 5**, the breaking force takes over: almost no new fixes happen, and the model starts second-guessing itself.
- **The Rule:** Stop the moment the risk of breaking a good answer is bigger than the chance of fixing a bad one!

---

### Figure 3: How Did We Get to 0.955 AUC? Is It Real?

#### (a) Performance Comparison Across Detector Architectures
![OOF AUC Comparison](images/oof_auc_comparison.png)
*Combining lightweight tree models with a deep sequence model boosted detection reliability to 0.955 ROC-AUC.*

#### (b) How the Detector Works
```mermaid
flowchart TD
    A["75,000+ Reasoning Traces<br/>(13 open-source models, 4 benchmarks)"] --> B["Extract 225 Clues per Step<br/>(Answer consistency, changes between steps, peer agreement)"]
    B --> C1["Tree Classifier (LightGBM)<br/>Score: 0.943 AUC"]
    B --> C2["Deep Neural Network (MoE Probe)<br/>Score: 0.936 AUC"]
    C1 --> D["Stacked Combination<br/>(Blends tree logic with deep sequence learning)"]
    C2 --> D
    D --> E["Final Combined Score: 0.955 AUC<br/>(Identifies correct answers 95.5% of the time)"]
```

#### Plain-English Answers to Key Questions:

1. **What does 0.955 AUC mean?**
   - It is a **ranking score**, not raw accuracy.
   - If you hand our detector two reasoning attempts—one right and one wrong—our detector correctly ranks the right one above the wrong one **95.5 out of 100 times**.

2. **Is it real, or did it just memorize the questions?**
   - **It is real.** We tested across 144,440 examples using 5-fold cross-validation where entire question groups were held out.
   - The detector was never evaluated on questions it had seen during training.

3. **The one honest nuance to explain to Dr. Woods:**
   - The 0.955 score was measured **retrospectively** (looking at traces after all 5 steps were generated).
   - Our main engineering milestone for this semester (Weeks 1–2) is to make this work **live during generation**, looking only at the steps written so far so we can stop the model in real time.

---

### Figure 4: Model Size and Precision Findings

| (a) Model Size vs Overthinking | (b) Full Precision vs 4-Bit Compression |
| :---: | :---: |
| ![Model Scale Accuracy Drift](images/model_scale_accuracy_drift.png) | ![Quantization Generalization](images/quantization_generalization.png) |
| *Bigger models (14B, 32B) keep reasoning effectively longer, while smaller models (0.5B, 3B) start overthinking earlier.* | *Heavily compressed 4-bit models lose 14.3% accuracy in early steps compared to full-precision BF16.* |

---

### Figure 5: What Worked vs What Failed (Honest Scientific Method)

Real scientific rigor means testing hypotheses and honestly reporting what worked and what failed:

| What We Tested | What Happened | Practical Takeaway |
| :--- | :--- | :--- |
| **Peer Agreement** (Do other models get the same answer?) | +593 points in utility | **Huge Win ✅:** When multiple models agree on an answer, it is almost certainly right. |
| **Step History** (Tracking answer changes between steps) | +252 points in utility | **Win ✅:** When a model flips its answer back and forth, it is a strong signal of confusion. |
| **Precision Impact** (Full BF16 vs 4-bit compression) | 4-bit dropped accuracy by 14.3% | **Confirmed ✅:** Heavy model compression hurts step-by-step reasoning significantly. |
| **Minimum Step Floor** (Forcing at least 2 steps) | Stopping at Step 1 was terrible; Step 2+ worked | **Confirmed ✅:** Models need at least one revision step to catch simple arithmetic slips. |
| **Complex Deep Trees** (Non-linear gradient boosted trees) | Lost 2,785 points | **Failed ❌:** Deep models memorized the training data; simpler linear blending worked much better. |
| **Probability Calibration** (Isotonic calibration) | Lost 3,142 points | **Failed ❌:** Complex calibration broke on edge cases; raw ensemble probabilities were safer. |
| **Bigger Token Limits** (Increasing cap from 256 to 512 tokens) | 0.0% change in accuracy | **Disproved ❌:** Models were not running out of token space; they were genuinely overthinking. |

#### Where Did the Remaining 7.5% Errors Come From?
We inspected every single mistake made by our detector:
- **0% were grading bugs.** Our grading suite passed all 30/30 unit tests.
- **52% were early errors:** The model was wrong on both Step 1 and Step 2. Because our safety rule requires at least 2 steps, it stopped before the model could recover.
- **48% were last-second fixes:** The model was wrong on Steps 1, 2, 3, and 4, and only fixed it on Step 5. No real-time system can reliably predict that a model wrong for 4 steps will suddenly get it right at the end.

---

## 📅 Semester 2 Accelerated Roadmap (Draft v1 by October 9)

By pulling the writing schedule forward to run in parallel with engineering, we submit **Draft v1.0 by October 9** and unlock a **full 4-week revision period** with Dr. Woods and Dr. Pemy.

```mermaid
flowchart LR
    P1["Phase 1: Weeks 1 to 3<br/><b>Write Ch 1-4 & Build Live Tool</b><br/>Intro, Background, Setup & Math while building online stopper"] --> P2["Phase 2: Weeks 4 to 5<br/><b>Finish Draft v1.0</b><br/>Ch 5-6 Results & Discussion<br/><b>SUBMIT DRAFT v1 BY OCT 9</b> 🎯"]
    P2 --> P3["Phase 3: Weeks 6 to 9<br/><b>4-Week Committee Revision Window</b><br/>Two full feedback cycles with Dr. Woods + conference paper prep"]
    P3 --> P4["Phase 4: Weeks 10 to 14<br/><b>Rehearse, Defend & Graduate</b><br/>Mock defenses, oral defense in late Nov, library archival"]
```

### Accelerated Week-by-Week Milestones

| Week | Target Dates | Engineering & Code Goals | Pulled-Forward Writing Deliverables |
| :---: | :--- | :--- | :--- |
| **W1** | Sep 9 – Sep 13 | Lock dataset hashes and package code cleanly. | **Draft Chapter 1 (Introduction & Motivation) and Chapter 3 (Experimental Setup & 13 Models).** |
| **W2** | Sep 14 – Sep 20 | Build `online_stopping_controller.py` to stop models live. | **Draft Chapter 4 (Empirical Evidence of Overthinking & Scaling Drift across 4 Benchmarks).** |
| **W3** | Sep 21 – Sep 27 | Measure live token savings vs accuracy trade-offs. | **Draft Chapter 2 (Mathematical Formulation & Stopping Theory).** |
| **W4** | Sep 28 – Oct 4 | Test against tricky adversarial questions and traps. | **Draft Chapter 5 (Live Online Stopping Results & Pareto Analysis) and Chapter 6 (Discussion & Limitations).** |
| **W5** | Oct 5 – Oct 11 | Compile complete manuscript, figures, and bibliography. | **SUBMIT COMPLETE THESIS DRAFT v1.0 TO DR. WOODS & COMMITTEE (OCTOBER 9 TARGET)! 🚀** |
| **W6** | Oct 12 – Oct 18 | Address initial committee feedback on structure and proofs. | **Committee Revision Cycle 1** (incorporate high-level advisor feedback). |
| **W7** | Oct 19 – Oct 25 | Polish text and format figures; produce Draft v1.1. | Format core findings into a 25-page paper for conference/journal submission. |
| **W8** | Oct 26 – Nov 1 | Fine-tune mathematical proofs and text clarity. | **Committee Revision Cycle 2** (line-by-line advisor polish; produce Draft v2.0). |
| **W9** | Nov 2 – Nov 8 | Finalize conference submission package. | Build 25-slide defense deck (30-minute presentation). |
| **W10** | Nov 9 – Nov 15 | Practice presentation timing and transitions. | **Mock Defense #1** (recorded practice talk with peers/lab group). |
| **W11** | Nov 16 – Nov 22 | Adviser Q&A preparation; test tough defense questions. | **Mock Defense #2** (adviser dry run); confirm defense logistics & announcement. |
| **W12** | Nov 23 – Nov 29 | **PUBLIC ORAL DEFENSE (30-min presentation + 30-min Q&A).** | Committee evaluation and signature approval. |
| **W13** | Nov 30 – Dec 6 | Final formatting checks for JHU Sheridan Libraries (PDF/A). | **Submit final thesis to JHU ETD library repository.** |
| **W14** | Dec 7 – Dec 11 | Submit signed completion paperwork to the JHU Registrar. | **Degree clearance and final grade recorded.** |

---

### Weekly Time Commitment (15–20 Hours/Week)

| Day | Focus | Description |
| :--- | :---: | :--- |
| **Monday** | 2 hrs | Weekly Planning & Theory Check |
| **Tue / Wed** | 6 hrs | Deep Computational & Writing Blocks |
| **Thursday** | 4 hrs | Data Analysis, Figures & Code Audit |
| **Friday** | 2 hrs | Committee Check-in & Feedback Log |
| **Weekend** | 4 hrs | Proofreading & Manuscript Drafting |

---

## 💬 4 Quick Discussion Questions for Dr. Woods

```mermaid
flowchart TD
    Q1["1. Are 75,000 reasoning traces enough, or do you want any more data runs?"]
    Q2["2. For our live stopping tool, do you prefer saving max compute (save 40% tokens) or max accuracy (stay within 0.5% of peak)?"]
    Q3["3. For Chapter 2 (Math Theory), what level of proof detail do you and Dr. Pemy want to see?"]
    Q4["4. Does October 9 work well for receiving Draft v1.0, giving us a full month for revisions before the defense?"]
    Q1 --> Q2 --> Q3 --> Q4
```

1. **Data Completeness:** Confirm that our 75,965 traces across 13 models are more than enough data so we can focus 100% on writing and live testing.
2. **Stopping Rule Preference:** In production, do we want a **Conservative rule** (keep accuracy drop below 0.5%) or an **Efficiency rule** (cut computing costs by 40%+)?
3. **Math Depth:** Dr. Woods and Dr. Pemy's expectations for proof rigor in Chapter 2.
4. **Accelerated Calendar:** Propose the **October 9 delivery date for Draft v1.0** (two weeks early!) to give the committee **4 full weeks of revision time** before the late-November defense.

---

## 📁 Key File Quick Reference
- **Master Research Report:** [`ThesisDocs/Thesis_Semester1_Research_Report_Fall_2026.md`](file:///C:/Aditya_Data/Personal/ResearchThesis/ThesisDocs/Thesis_Semester1_Research_Report_Fall_2026.md)
- **Scientific Rigor Audit:** [`ThesisDocs/rigor_audit/00_EXECUTIVE_SUMMARY.md`](file:///C:/Aditya_Data/Personal/ResearchThesis/ThesisDocs/rigor_audit/00_EXECUTIVE_SUMMARY.md)
- **Unit Tests (30/30 passing):** [`research/tests/test_graders.py`](file:///C:/Aditya_Data/Personal/ResearchThesis/research/tests/test_graders.py)
- **Blackwell GPU Tournament Results:** [`research/outputs/experiments_v2/blackwell_5day_tournament_v1/blackwell_tournament_report.md`](file:///C:/Aditya_Data/Personal/ResearchThesis/research/outputs/experiments_v2/blackwell_5day_tournament_v1/blackwell_tournament_report.md)
