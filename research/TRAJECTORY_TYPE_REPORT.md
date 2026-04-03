# Correctness-Conditional Trajectory Analysis

## Overview

The macro continuation value metric assumes a single forward path for all model solutions. However, observation indicates reasoning loops diverge into path trajectories:
1. **Persistent Correct**: Found exact match token immediately and remained static.
2. **Corruption**: Transferred from Correct to Incorrect via model hallucination or over-thinking.
3. **Repair**: Transferred safely from Incorrect to Correct.
4. **Persistent Wrong**: Stagnantly failed the prompt constraint across the 10-step limitation.

Phase C partitioned trace instances by path behaviors mapping the macro phenomena to specific temporal patterns.

## Trajectory Distribution Details

| Model | Repair | Corruption | Persistent Correct | Persistent Wrong |
|---|---|---|---|---|
| DeepSeek 1.5B | 45.3% | 23.0% | 0.7% | 31.0% |
| Mistral 7B | 15.3% | 20.9% | 9.3% | 54.4% |
| Qwen 0.5B | 1.9% | 0.7% | 6.4% | 91.0% |
| Qwen 7B | 47.0% | 31.3% | 5.1% | 16.6% |

### Feature Correlations by Path Type

**A Critical Capability Metric:** 
The percentage of "repair" pathways scales linearly with intrinsic problem-solving aptitude on the dataset format. Qwen 0.5B lacks all baseline contextual reasoning, and therefore possesses merely 1.9% repair viability. In stark comparison, Qwen 7B converts 47% of instances using long-sequence repairs.

**Stopping Implications for Overthinking:**
Mistral 7B maintains a surprisingly high `Corruption` density relative to its `Repair` paths ($20.9\% > 15.3\%$), reinforcing the difficulty stratification discovery: Mistral frequently un-solves successful instances prior to repairing faulty ones, heavily penalizing its aggregate trajectory value and enforcing an earlier stopping threshold.
