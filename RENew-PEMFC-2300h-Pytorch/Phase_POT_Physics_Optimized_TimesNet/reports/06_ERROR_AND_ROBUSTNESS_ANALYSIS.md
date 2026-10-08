# Phase POT — Error and Robustness Analysis

The primary aggregation is equal-fold and forecast-query weighted. Target-time-equal RMSE is a mandatory sensitivity metric, not a replacement selection metric. The formal M4/M2 ratios were: RMSE 1.0861, Fold 3 RMSE 1.0376, P99 1.1446, and target-time-equal RMSE 1.0922.

Seed-level M2 and M4 equal-fold RMSE values (mV):

| Seed | M2 EMD | M4 Optuna EMD |
|---:|---:|---:|
| 42 | 13.335 | 14.834 |
| 2024 | 13.130 | 15.051 |
| 3407 | 12.401 | 12.326 |

Only 1/3 paired seeds improved for M4. The extreme-error plot is explicitly a post-hoc descriptive diagnostic (`2383`), not an outlier-removal rule or a new selection criterion. All legal validation queries, including the historical extreme targets, remained in the formal metrics.
