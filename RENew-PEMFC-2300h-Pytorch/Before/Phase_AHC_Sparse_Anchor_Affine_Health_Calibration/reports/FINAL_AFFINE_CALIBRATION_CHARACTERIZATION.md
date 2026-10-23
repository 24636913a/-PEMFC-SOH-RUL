# Phase AHC final characterization

## Established Development evidence

B2 LOAO reproduced exactly (MAE 0.000476869906820), outperforming raw B0 and CHM in the historical frozen LOAO experiment.

## New posthoc checks

All-six candidate: `Y = 1.198811493832 R + -0.209036246312`. Chronological n=3 B2 MAE/RMSE/bias: 0.000764687034 / 0.000864347098 / 0.000073731440; raw B0 n=3 MAE: 0.013021458879. Parameter, function, deletion, chronological, and support-sensitivity tables are retained in `outputs/`.

## Limitations

This is posthoc small-n Development characterization. R and Y may share an age trend; the six anchors cannot identify an independent health mapping or causal mechanism. Only three median-source anchors are inside joint common support. No validated SOH, Test generalization, RUL, or cross-stack claim is made.

## Decision

`AFFINE_CALIBRATION_DEVELOPMENT_HYPOTHESIS_SUPPORTED`. Test opening is not automatic; at most it is a manual-review candidate.
