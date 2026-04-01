# Answers to Open Questions

## Question 1
Yes. Across 8 runs covering 8 tasks, the model started with step-1 competence $q_1=0.000$, reached correctness at least once in 2 runs, and exhibited measurable repair and corruption rates of 0.125 and 0.500 respectively. That moves the experiment out of the low-skill regime and into a regime where continuation hazards can actually be estimated.

## Question 2
Yes, with caveats. The global correctness probe achieved mean Brier 0.1112 and mean AUC 1.0000. The strongest correctness-side observable in this run was reasoning length (thought_token_count, coeff=1.142). That is enough evidence to say that verifier-lite observables carry signal about $q_t$, although the calibration quality should still be stress-tested on a second model family.

## Question 3
Partially yes. The data contains both repairs and corruptions, and the hazard-based stop rule achieved mean oracle gap 0.0437 with false-late rate 0.750. The empirical-Bernstein detector achieved mean oracle gap 0.0375. So the hazards are learnable well enough to support a practical stop rule, but the current stop rules are still conservative rather than oracle-close.

## Question 5
Within the currently completed L4 run, the most stable observable looks like reasoning length (thought_token_count, coeff=1.142). On the corruption side, the strongest positive hazard signal was token entropy (entropy_mean, coeff=0.293). That makes hidden-state drift, entropy, and verbosity-derived signals the main candidates, with cross-family stability still needing the 7B follow-up before it can be called settled.

## Question 6
In the current traces, corruption appears earliest through token entropy (entropy_mean, coeff=0.293). The drift estimate crosses zero at step 2, while the never-stop policy retains a mean oracle gap of 0.2125. That pattern is more consistent with a corruption cascade than with harmless extra verification.
