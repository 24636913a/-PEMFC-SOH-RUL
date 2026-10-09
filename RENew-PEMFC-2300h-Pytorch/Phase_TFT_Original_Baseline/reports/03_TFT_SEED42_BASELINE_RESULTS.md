# Canonical TFT Seed-42 Development baseline

The preregistered formal score is the three-fold equal forecast-query-weighted RMSE: **26.158 mV**. The equal-fold MAE is **12.108 mV**.

| Fold | Forecast queries | RMSE (mV) | MAE (mV) | R² | 12 h endpoint RMSE (mV) | Selected inner epoch |
|---|---:|---:|---:|---:|---:|---:|
| fold_1 | 5,592 | 26.656 | 13.132 | 0.9813 | 24.209 | 246 |
| fold_2 | 5,880 | 18.365 | 9.857 | 0.9911 | 26.515 | 297 |
| fold_3 | 11,184 | 33.453 | 13.335 | 0.9705 | 33.583 | 289 |

This establishes a canonical full-TFT baseline under the frozen strict-history protocol. It is a Seed-42 baseline, not a multi-seed stability claim. No Test metric was computed.
