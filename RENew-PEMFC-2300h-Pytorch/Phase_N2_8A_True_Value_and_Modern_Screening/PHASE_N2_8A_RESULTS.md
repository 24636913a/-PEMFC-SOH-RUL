# Phase N2.8A Results — True Predictive-Value Audit and Modern Screening

## Primary Development-only ΔV metrics (seed 42)

| Model | Mean R²ΔV | Mean RMSEΔV (mV) | Mean MAEΔV (mV) | Mean R²V |
|---|---|---|---|---|
| LSTM | 0.9974 | 9.006 | 4.790 | 0.99476 |
| TimesNet | 0.9959 | 9.895 | 3.546 | 0.99166 |
| TimeXer | 0.9653 | 36.170 | 18.968 | 0.92875 |

`R²ΔV`, `RMSEΔV`, and `MAEΔV` compare `y_true_abs − V_t` directly with `y_pred_abs − V_t`; R²V remains a secondary reconstructed-voltage statistic.

## Q1–Q7: predictive value and integrity

- **Q1 LSTM:** R²ΔV=0.9974, RMSEΔV=9.006 mV, MAEΔV=4.790 mV.
- **Q2 TimesNet:** R²ΔV=0.9959, RMSEΔV=9.895 mV, MAEΔV=3.546 mV.
- **Q3:** the per-fold R²V − R²ΔV difference is in `anchor_contribution_audit.csv`. The absolute-voltage result includes the known origin anchor, while R²ΔV directly scores future change; no percentage-accuracy interpretation is used.
- **Q4:** anchor contribution is reported for LSTM, TimesNet, and Ridge with both R² values and their difference.
- **Q5:** label permutation collapsed: R²ΔV range [-0.0418, 0.0103], far below the 0.5 leakage-warning threshold.
- **Q6:** 48-origin temporal misalignment collapsed: R²ΔV range [-0.2990, 0.0705].
- **Q7:** all 14 feature rows PASS: no future-voltage/current/condition access, centered window, negative target shift, or future backfill.

## Q8–Q16: high-fidelity screening

- **Q8 TimeXer:** R²ΔV=0.9653, RMSEΔV=36.170 mV, MAEΔV=18.968 mV. It is over 20% worse than TimesNet on mean RMSEΔV and has no Fold3 improvement, so it is directly stopped.
- **Q9 TimeMixer++:** BLOCKED, not scored. The named anonymous official source returns HTTP 410. The authors' accessible `kwuking/TimeMixer` commit is old TimeMixer and lacks MRTI/TID/MCM/MRM; no surrogate was used.
- **Q10:** among evaluated modern candidates, LSTM has the highest mean R²ΔV (0.9974); among the specified new-model comparison incumbents, TimesNet is the reference.
- **Q11/Q12 Fold3:** LSTM: R²ΔV=0.9943, RMSEΔV=14.651 mV; TimesNet: R²ΔV=0.9892, RMSEΔV=20.230 mV; TimeXer: R²ΔV=0.9640, RMSEΔV=36.961 mV. TimeMixer++ has no valid Fold3 result.
- **Q13 tail errors:** Fold3 P95 rank: TimesNet=14.132 mV, LSTM=16.722 mV, TimeXer=53.016 mV. Fold3 P99 rank: LSTM=53.260 mV, TimesNet=64.487 mV, TimeXer=188.157 mV.
- **Q14 lifecycle:** Fold3/Fold1 RMSEΔV rank: TimeXer=1.167, LSTM=2.360, TimesNet=7.879. This ratio is contextual, not a replacement for primary ΔV accuracy.
- **Q15 TimeXer three-seed:** NO; it failed the seed42 direct-stop condition.
- **Q16 TimeMixer++ three-seed:** NO; a verified official implementation is not accessible.

## Q17–Q18: frozen decision

- **Q17 selected backbone:** `UNRESOLVED`. The previously frozen high-fidelity TimesNet remains the incumbent reference, but N2.8A cannot make a final three-way model decision without legitimate TimeMixer++ execution.
- **Q18 next phase:** **NO-GO**. Obtain an author-confirmed, versioned, accessible TimeMixer++ source first; then rerun only the missing high-fidelity screen under this frozen Development protocol. Do not access Test.

## Firewall

`test_target_rows_read=0`, `test_predictions_constructed=false`, `test_metrics_computed=false`, and `test_feature_diagnostics_computed=false`.
