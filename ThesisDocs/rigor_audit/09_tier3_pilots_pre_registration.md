# Tier-3 Pilots Pre-Registration: k=2 Self-Consistency (N7) and Extended Observables (N8)

**Date:** 2026-07-12 · **Status:** Pre-registered · **Focus:** Enriching the Observation Filtration

---

## 1. Context & Motivation

Having completed Tier-1 and Tier-2, we established that Empirical Bayes Hazard Shrinkage (N3) and rolling history windows (N2c) yield significant utility gains, but the standard 10 step-level observables are beginning to hit a mathematical performance ceiling. 

To dissolve the remaining missed late-correction losses (specifically the unpredictable E/F category failures where corrections happen late), we must **enrich the model's observation filtration**. This document pre-registers the pilot experiments for **N7** (Self-Consistency Agreement) and **N8** (Answer-Span & Mid-Layer activation features).

---

## 2. Experimental Variables & Instrumentation

We have instrumented the real-trace generation engine in `research/real_trace_experiments.py` to record the following new columns per step:

### N7: Self-Consistency agreement ($k=2$)
* **`k2_agreement`**: (Binary $\{0, 1\}$) At step $t$, we generate a second independent continuation from the same step prompt. We parse the answer from both paths. `k2_agreement = 1` if the normalized answers match, `0` otherwise.
* **`k2_raw_generation_tokens`**: (Integer) The number of tokens generated in the second independent path, enabling **honest token-cost accounting**.

### N8a: Answer-Span Diagnostics
* **`answer_span_mean_logprob`** / **`answer_span_min_logprob`**: Mean and minimum token logprob restricted strictly to the tokens making up the extracted answer span (rather than averaging over the entire thought).
* **`answer_span_mean_entropy`** / **`answer_span_std_entropy`**: Mean and standard deviation of token-level entropy restricted strictly to the answer span.

### N8b: Pooled Mid-Layer Projections
* **`mid_hidden_1_proj`** / **`mid_hidden_2_proj`**: Deterministic random projections (from $D$ dims to $64$ dims using a fixed random seed) of the mean-pooled hidden states at layer $L_1 = \text{num\_layers} // 3$ and layer $L_2 = 2 \cdot \text{num\_layers} // 3$. This stores high-dimensional mid-layer activations as a compact string inside the step CSV.

---

## 3. Pre-Registered Hypotheses & Success Criteria

### Hypothesis N7 (Self-Consistency)
A second independent continuation acts as a direct proxy for the model's belief confidence. High agreement should correlate with correct steps, while disagreement highlights unstable overthinking states.
* **Metric 1 (Uncharged)**: Training an offline OOF probe using $k=2$ agreement yields a utility increase of $\ge \mathbf{+100}$ points over the baseline.
* **Metric 2 (Honest Cost-Charged — Success Metric)**: Adding the token costs of the second path to the utility equation (i.e. charging $\lambda \cdot \text{k2\_tokens}$ as part of the step penalty) still yields a net utility gain of $\ge \mathbf{+30}$ points.

### Hypothesis N8 (Answer-Span & Mid-layer Features)
Restricting logits to the final answer span removes the noise of the raw chain-of-thought tokens. Simultaneously, mid-layer hidden states capture representation stability that final logits miss.
* **Success Metric**: Training an offline OOF probe with the N8 features (baseline + N8a + N8b) outperforms the baseline probe by $\ge \mathbf{+50}$ expected utility points.

---

## 4. Pilot Scope

We will run the pilot evaluation on **2 cells** under identical temperature and seed settings to maintain control:
1. **Model**: `Qwen2.5-7B-Instruct`
2. **Tasks**: `GSM8K` (500 tasks, split `train`) and `MATH` (500 tasks, split `train`)
3. **Temperature**: `0.6`
4. **Seed**: `7`

---

## 5. Execution Commands for the Nvidia Box

Run the following commands in the remote GPU container to execute the pilots and collect the enriched trace steps:

```bash
# 1. Pull the instrumented code changes
git pull

# 2. Run the GSM8K Pilot (500 runs)
python research/real_trace_experiments.py \
    --model qwen2p5_7b \
    --task-source gsm8k \
    --max-tasks 500 \
    --temperatures 0.6 \
    --seeds 7 \
    --enable-k2-agreement \
    --enable-extended-observables \
    --output-dir research/outputs/experiments_v2/tier3_pilot_gsm8k

# 3. Run the MATH Pilot (500 runs)
python research/real_trace_experiments.py \
    --model qwen2p5_7b \
    --task-source math \
    --max-tasks 500 \
    --temperatures 0.6 \
    --seeds 7 \
    --enable-k2-agreement \
    --enable-extended-observables \
    --output-dir research/outputs/experiments_v2/tier3_pilot_math
```

*(Note: These runs will create updated `trace_steps.csv` files under their respective output directories containing the new N7 and N8 telemetry columns.)*
