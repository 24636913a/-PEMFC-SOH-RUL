# TimeXer PEMFC adaptation audit

## Fixed official source

- Repository: `https://github.com/thuml/TimeXer`.
- Vendor commit: `76011909357972bd55a27adba2e1be994d81b327`.
- Retained path: official TimeXer `EnEmbedding`, `DataEmbedding_inverted`, encoder self/cross attention, and `FlattenHead` with the frozen N2.8A small configuration.

## Allowed role-only adapters

| Variant | Endogenous input | Exogenous input | Future covariates | Additional context |
|---|---|---|---|---|
| TimeXer-PEMFC-A | `cell_voltage_mean` only, passed as TimeXer MS-mode final channel | remaining thirteen F2 variables from the same 48 historical bins only | none | none |
| TimeXer-PEMFC-Life | exactly as PEMFC-A | exactly as PEMFC-A | none | one origin-known `OpHrs2` scalar, standardized on each training fold and repeated only across its historical 48-step input representation |

`TimeXer-PEMFC-Life` does not build a future age path and does not supply future operating conditions. It is diagnostic only. Both rows use the same chronological windows, exact F2, train-only scaler, MSE, AdamW, early stopping, and seed as the frozen comparison. No LSTM, added attention, filtering, physics loss, or hyperparameter search is present.

## Result-specific stop audit

- Original frozen TimeXer mean RMSE_delta: 36.170 mV.
- PEMFC-A mean RMSE_delta: 36.170 mV; Fold3: 36.961 mV.
- PEMFC-Life mean RMSE_delta: 38.007 mV; Fold3: 40.597 mV.
- Final status: **PERMANENT STOP: best adaptation mean RMSE_delta > 25 mV**.

Novelty cannot override unacceptable predictive performance.
