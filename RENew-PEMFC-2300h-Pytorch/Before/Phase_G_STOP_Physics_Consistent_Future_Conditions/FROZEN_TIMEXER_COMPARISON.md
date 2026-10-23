# Frozen SA-TimeXer Comparison

The same N2.8F M0 weights, history tensors, validation origins, targets, and scoring code are used. Only future-condition tensors change.

| Scheme | R²_delta | RMSE (mV) | MAE (mV) | Fold1 | Fold2 | Fold3 | pooled P95 | pooled P99 | 9.5 h | 12 h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | 0.983989 | 24.538 | 12.393 | 21.884 | 23.846 | 27.885 | 33.426 | 129.652 | 26.074 | 33.528 |
| P1 | 0.984111 | 24.443 | 12.318 | 21.729 | 23.828 | 27.772 | 32.753 | 129.643 | 26.017 | 33.520 |
| P2 | 0.986918 | 21.347 | 10.382 | 17.671 | 15.987 | 30.382 | 26.166 | 68.630 | 19.864 | 35.475 |

P0 reproduces 24.538245 mV within 0.001 mV. P1 improves only 0.39% and fails Gate B. P2 improves mean RMSE by 13.01%, improves two folds, reduces P99 by 47.07%, reduces 9.5 h RMSE by 23.82%, and removes measured violations, so it passes Gate B. Fold3 worsens from 27.885 to 30.382 mV and 12 h worsens from 33.528 to 35.475 mV; the gain is not uniform.

The paired moving-block bootstrap for P2 versus P0 has mean RMSE difference −1.070 mV with 95% interval [−4.879, 2.530] mV and probability of improvement 0.703. It describes uncertainty and does not override the preregistered point-estimate gate.
