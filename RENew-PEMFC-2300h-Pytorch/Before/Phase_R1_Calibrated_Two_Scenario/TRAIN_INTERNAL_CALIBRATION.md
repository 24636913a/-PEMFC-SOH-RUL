# Train-internal calibration result

**CALIBRATION_INTERNAL_FAIL.** In every fold, both weighted Brier and Log Loss increased on the fixed purged late-20% diagnostic:

- Fold1: Brier 0.005956→0.010404; Log Loss 0.025963→0.037094.
- Fold2: Brier 0.004072→0.006902; Log Loss 0.016108→0.024470.
- Fold3: Brier 0.005746→0.007728; Log Loss 0.020797→0.028247.

This was a preregistered fail-closed condition. CTS-P2 Validation inference, probability diagnostics on Validation, physical scenario construction, tail audit, bootstrap and Gate A were **NOT_RUN**. No alternative calibration method, C value or threshold was tried. Evidence: `outputs/calibration/calibration_internal_holdout.csv`.
