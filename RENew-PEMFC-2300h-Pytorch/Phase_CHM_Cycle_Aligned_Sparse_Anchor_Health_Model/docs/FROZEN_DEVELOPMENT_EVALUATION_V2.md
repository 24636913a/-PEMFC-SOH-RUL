# Frozen CHM Development evaluation — Final V2

`PREREGISTERED_CHM_PROTOCOL_FINAL_V2.yaml` is the sole executable source for
this evaluation. There are six outer leave-one-anchor-out folds: 300, 600, 850,
1150, 1500, and 1750 h. Each held-out target is excluded from fitting, profile
selection, early stopping, and baseline calibration.

Only A BALANCED `(1, 1, 0.1)`, B ANCHOR_EMPHASIS `(3, 1, 0.1)`, and C
OBSERVATION_EMPHASIS `(1, 3, 0.1)` are eligible. All three loss components are
means. Five inner folds × three profiles × six outer folds × three seeds gives
exactly 270 inner fits; there are 18 outer deterministic refits. A tie within
`1e-8` resolves A, then B, then C.

Primary Development qualification has exactly five gates: 6/6 legal LOAO
predictions; CHM MAE lower than B0; CHM MAE no worse than the better B1/B2
within `1e-6`; zero complete-cycle monotonic violations; and at least 5/6 LOAO
95% intervals covering their held-out targets. The gate comparison uses only
the six true outer-heldout predictions of each baseline.

S1 is a `PRESPECIFIED_SECONDARY_ROBUSTNESS_DIAGNOSTIC`, not a sixth gate. After
the deterministic primary Development experiment, it trains with the already
selected profile and anchors 300/850/1750 only. Its Spearman, median trajectory
difference, and terminal difference thresholds are diagnostic references. An
`S1_ROBUSTNESS_CONCERN` cannot reverse an otherwise passed primary qualification
and cannot automatically open Test.
