# Chapter 1 Introduction and related work

## 1.1 Problem and research questions

Additional language-model reasoning can repair an answer, replace a correct answer, or consume computation without sufficient improvement. A useful stopping policy must balance answer quality against the cost of continuing. This thesis studies that trade-off at complete-response boundaries: a generator proposes an answer, receives a revision request, and may produce another candidate. These response-and-revision trajectories differ from tokens or latent thoughts within one uninterrupted reasoning process.

A final answer conceals earlier opportunities to stop and later opportunities to repair. Accordingly, the observational unit is an ordered trajectory, and a runtime decision uses only its available prefix. Correctness means agreement with the versioned reference grader. A repair changes an incorrect candidate to a correct one; a corruption makes the reverse transition. Neither label establishes that the preceding reasoning is mathematically valid.

The study asks three connected questions. How do repair and corruption determine the value of another response? When does a one-step drift rule agree with optimal stopping? Do controlled experiments and an actually executed controller support useful accuracy-cost trade-offs? Answer corruption, unproductive computation and policy regret are distinct outcomes: a policy can improve utility while losing accuracy, or stop before a valuable repair.

## 1.2 Prior work and contribution

Chain-of-thought prompting makes intermediate reasoning explicit [Wei2022]. Verifier training and process supervision evaluate completed answers or intermediate steps [Cobbe2021], [Lightman2023]. Self-consistency aggregates sampled solutions, while Adaptive-Consistency adjusts their sampling budget [Wang2022], [Aggarwal2023]. Successive revision instead conditions later responses on earlier work; its transition law, answer selector and acquisition costs need not match independent-path sampling.

Adaptive Computation Time and PonderNet learn internal halting decisions [Graves2016], [Banino2021]. CALM exits intermediate network layers while continuing token generation [Schuster2022]. The overthinking literature and answer-convergence methods address excessive reasoning and termination within a chain of thought [Chen2024], [Liu2025]. REFRAIN uses a redundancy discriminator and an adaptive controller [Sun2026]; OS-Pruner applies accuracy-cost optimal stopping at paragraph boundaries [Ehab2026]. These are substantial antecedents. This thesis studies complete-response revision under its own generation, selection, grading and cost conventions, rather than claiming to originate learned halting or optimal stopping.

Finite-horizon stopping theory distinguishes immediate gain from conditional continuation value [Ferguson], [Peskir2006]. The mathematical contribution here is its explicit specialization to graded response trajectories: the repair-corruption identity, a sufficient persistence condition for a myopic rule, and counterexamples when that condition fails. The empirical contribution is a set of matched contrasts, including negative findings, and a measured runtime implementation. Prior methods were not reproduced under a shared protocol, so comparative superiority is untested.

The fitted models use sigmoid calibration [Platt1999], [Guo2017]. Calibration on archived prefixes does not establish conditional probabilities under a changed live prompt or stopping-induced distribution. Likewise, bounded-observation concentration and confidence sequences require their stated assumptions [Hoeffding1963], [Howard2021]; an arbitrary confidence threshold or repeatedly consulted bootstrap interval does not inherit those guarantees.

The main result is deliberately limited. Continuation value varies across recorded model-domain panels, and improvements in detector ranking need not improve stopping utility. The live prototype avoids future generation, but its learned rule stops at the two-response floor on every evaluated task. Those savings establish a reduced budget on the observed panels; they do not establish useful adaptation or accuracy noninferiority.

## 1.3 Organization and supporting material

Chapter 2 develops the mathematical argument. Chapter 3 defines the data, methods and evaluation contracts; Chapter 4 reports results; Chapter 5 discusses their implications. The preserved extended v5 manuscript supplies full proofs, calibration and uncertainty bounds, complete rosters, additional analyses and reproduction details. This concise thesis retains the argument and evidence needed for its stated conclusions; it performs no new collection or training.
