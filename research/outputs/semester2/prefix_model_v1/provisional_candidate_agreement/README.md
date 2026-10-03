# Provisional model preserved for the audit trail

This first artifact required all five reconstructed live-parser candidates to agree with saved candidates. It retained only 355 of 1,500 trajectories and thereby selected a panel partly on later parser outcomes. That eligibility rule was rejected on methodological grounds before any live learned generation.

The deployed artifact is the parent directory's prefix_model.json, byte SHA-256 92fe0af86ac0f204d514a6938d0a29dace2b3cdffa24800cc4e2c85a2d51879f. It retains all 1,500 complete archived trajectories, reconstructs candidates with the actual live parser, and regrades them offline against archived gold under the existing verify_answer function. The original archived data are untouched. Features, task-hash roles, hyperparameters, calibration method, cost and stopping rule were unchanged.

This directory's AUC, calibration and replay scores are superseded. Its model was not selected for deployment. No live benchmark or trap outcome was used to make the correction.
