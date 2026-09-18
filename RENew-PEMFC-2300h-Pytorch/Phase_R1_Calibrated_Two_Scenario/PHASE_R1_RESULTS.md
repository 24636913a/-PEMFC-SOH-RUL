# Phase R1 — Calibrated Two-Scenario Future Conditions

**Status: CALIBRATION_INTERNAL_FAIL.** This is an exploratory Development confirmation attempt. Test remained closed.

1. R0 reporting corrections are documented in `R0_CORRECTION_AUDIT.md`.
2. Frozen P0/P1/P2/RS reproduce.
3. P2 p_raw, q_plus and q_hard reconstruct exactly.
4. Train OOF calibration is strict causal with inverse target-multiplicity weighting and completed-target purge.
5. The pre-registered logistic calibration fails every Train-internal late-20% diagnostic: all three folds worsen both Brier and Log Loss.
6. Therefore CTS-P2, Uncalibrated-CTS, scenarios, Validation metrics, tails, regimes, bootstrap and Gate A are **NOT_RUN**.
7. No claim is made that R0 Oracle improvement was achieved in deployable prediction, that TimeXer exceeds the LSTM reference, or that SOH/RUL was predicted.
8. Test Firewall is PASS: causal feature prefix ends at 1819.5 h and the actual terminal raw row read before filtering is 1819.49 h; no Test rows/predictions/metrics were accessed.

The correct scientific action is to stop without calibration-method substitution or Validation-driven repair.
