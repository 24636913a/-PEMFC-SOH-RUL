# Phase 3A — causal slow-degradation / transient decomposition

## Question

Phase 3A asks which FC2 voltage component is predictably encoded in legal
history. It evaluates the causal decomposition

```text
U(t) = D(t) + R(t)
D(t) = alpha * U(t) + (1 - alpha) * D(t-1)
R(t) = U(t) - D(t)
```

with `D(0) = U(0)` and no centred smoother, future interpolation, or future
operating condition.

## Development-only protocol

Only FC2's first 60% hourly prefix is available. The final 40% test partition
is never constructed, fitted, predicted, or scored. Phase 2b's three fixed
expanding folds remain unchanged: 0–35/35–40%, 0–45/45–50%, and 0–55/55–60%.

The only predeclared EWMA scales are:

| tau (h) | alpha = 1-exp(-1/tau) |
|---:|---:|
| 3 | 0.283469 |
| 6 | 0.153518 |
| 12 | 0.079956 |

T1 predicts `D(t+h)-D(t)` using only `[D(t-14), ..., D(t)]`. It compares
slow persistence (zero delta), slow linear trend, and the same one-layer
hidden-64 residual MIMO-LSTM.

T2 is evaluated only for the validation-selected tau. It predicts `R(t+h)`
with: (A) R history only, and (B) R history plus legal historical
`J_mean/I_mean/I_std/I_p2p`. It compares zero transient and transient
persistence baselines.

T0 is not retrained: the saved Phase-2b R1 raw-residual validation forecasts
are read solely as a comparison reference.

## Metrics and tau selection

Every fold and horizon reports MAE, RMSE, R2, NRMSE=`RMSE/std(y_true)`, and
the relevant analytical-baseline skills. Tau ranking first retains scales
within 3% of the lowest mean T1 NRMSE, then maximizes mean Skill vs Slow
Linear, minimizes fold-mean NRMSE standard deviation, and uses tau=6 for a
near tie with tau=12. The report separately records whether the selected tau
actually has stable positive Slow-Linear skill; selection is not proof of
success.

## Stop boundary

Phase 3A ends after its development-only report. It does not evaluate the
final test, train T0 again, use Full-26 or OCM, add uncertainty, extend the
horizon, run other seeds, or start RUL work.
