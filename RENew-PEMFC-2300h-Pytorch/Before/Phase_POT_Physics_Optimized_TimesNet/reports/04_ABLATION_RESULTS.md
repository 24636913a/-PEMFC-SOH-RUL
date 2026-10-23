# Phase POT — Formal Ablation Results

Primary metric: three-fold equal forecast-query-weighted RMSE, reported as the mean across the three formal seeds. Overlapping forecast queries are not treated as independent observations.

| Model | RMSE (mV) | MAE (mV) | R² | Endpoint RMSE (mV) | P99 abs. error (mV) | Target-time-equal RMSE (mV) |
|---|---:|---:|---:|---:|---:|---:|
| M1 Raw F2 | 13.860 | 4.091 | 0.9930 | 27.861 | 46.048 | 25.197 |
| M2 EMD | 12.955 | 3.952 | 0.9941 | 26.928 | 38.983 | 24.014 |
| M3 Optuna Raw F2 | 13.644 | 4.065 | 0.9934 | 28.534 | 42.598 | 24.799 |
| M4 Optuna EMD | 14.070 | 4.184 | 0.9931 | 32.153 | 44.619 | 26.229 |

M4 vs M2 preregistered gates: RMSE≤0.99×M2 = `False`; ≥2 paired seed wins = `False` (1/3); Fold 3 RMSE≤1.03×M2 = `False`; P99≤1.05×M2 = `False`; target-time-equal RMSE≤1.05×M2 = `False`.

Therefore, the frozen M4 advance gate is **not passed**. M3's numerical improvement over M1 is descriptive; the registered advance decision was based on M4 versus M2 and is not retroactively changed.
