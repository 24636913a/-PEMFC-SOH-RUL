# Phase CMRF — H3/H6 origin-conditioned P1S-retention forecasting

This phase is a Development-only, causal forecasting study.  It evaluates two
independent raw-cycle horizons: H3 (secondary) and H6 (the preregistered
primary horizon).  The target is raw, causally available P1S retention,
`DeltaR_H = R_target - R_origin`; predictions are reconstructed as
`R_origin + predicted_DeltaR_H`.

Inputs are limited to a legal 48 x 14 Raw F2 history ending at the observed
origin P1S measurement.  The target P1S measurement, target timing, target
conditions, operating age, B2 calibration, smoothed labels, Test data, and
future covariates are excluded from all model inputs.

Run from this directory with:

```powershell
python scripts/run_cmrf_h3_h6.py
```

The script is fail-closed: the mixed causal-feature CSV is read only through
`nrows=3007` and filtered to the 3005-row Development prefix.  The raw-cycle
catalog is an existing Development-only artifact and is checked to remain
strictly below the Development boundary before it is used for physical-cycle
continuity.
