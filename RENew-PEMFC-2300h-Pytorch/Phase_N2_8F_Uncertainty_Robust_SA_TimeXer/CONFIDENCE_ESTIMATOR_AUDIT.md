# Confidence estimator audit

The confidence target is per-horizon standardized future-exogenous Ridge error. Training supervision consists only of completed causal OOF residuals. Validation true residuals are diagnostic-only. A confidence gate is valid only when the mean validation Spearman and top/bottom actual-error ratio clear their frozen thresholds.

| fold | validation_spearman | top_to_bottom_actual_error_ratio | criterion_spearman | criterion_ratio |
|---|---|---|---|---|
| fold_1 | 0.081422 | 1.165467 | 0.100000 | 1.100000 |
| fold_2 | 0.389520 | 1.948284 | 0.100000 | 1.100000 |
| fold_3 | 0.094102 | 1.200617 | 0.100000 | 1.100000 |

Final status: **VALID**.
