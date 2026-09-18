# Strict causal calibration audit

Train OOF is the sole calibration source. Per fold, each prediction point uses inverse target multiplicity weight so every target timestamp contributes a total weight of one. The fixed chronological Train-internal split uses an 80% early fitting region and a 20% late-origin diagnostic region, with completed-target purge: Fold1 1194/1194, Fold2 1525/1525 and Fold3 1865/1865 are the maximum fitting target row/minimum holdout origin row. Thus no calibration label crosses into the held-out origin future.

The only planned calibrator was logistic calibration on logit(p), C=1.0, eps=1e-6. Its fitted coefficients are recorded but marked `NOT_DEPLOYED_INTERNAL_FAIL`; neither Validation labels nor metrics were used to alter it. Evidence: calibration CSVs.
