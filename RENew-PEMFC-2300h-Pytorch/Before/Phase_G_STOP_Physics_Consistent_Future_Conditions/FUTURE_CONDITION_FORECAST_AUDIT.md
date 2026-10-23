# Future-Condition Forecast Audit

All forecasts are causal expanding-window predictions with target-completion purge and Train-only scaling. P0 and P2 use the same legal OOF origins. P1 is a deterministic minimal projection of P0.

| Scheme | q MAE | q RMSE | mixed F1 | current-mean MAE (A) | current-std MAE (A) | current-p2p MAE (A) | violating trajectories |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | 0.024810 | 0.041624 | 0.510011 | 4.887 | 2.787 | 14.287 | 100.0% |
| P1 | 0.017061 | 0.038242 | 0.510011 | 4.887 | 2.787 | 14.287 | 0.0% |
| P2 | 0.001743 | 0.022112 | 0.990499 | 2.042 | 2.022 | 14.287 | 0.0% |

q=1 Precision/Recall/F1 are **NOT_ESTIMABLE**, not zero. P2 sharply improves q and current mean prediction, while peak-to-peak is unchanged because it remains the independent Ridge forecast. P1/P2 eliminate measured range and current-density identity violations. P0 flags every trajectory because unconstrained q leaves `[0,1]` and/or current-density identities beyond tolerance.
