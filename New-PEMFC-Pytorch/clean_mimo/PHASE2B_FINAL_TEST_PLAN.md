# Frozen R3 final-test protocol

## Authorization and single-run boundary

This is the one authorized final evaluation following Phase-2b validation-only
selection. It evaluates **only** R3 `state_load` on the original fixed 40%
chronological FC2 test partition. The runner refuses to overwrite its output
and writes a completion marker after success; it must not be re-run for model
selection.

## Locked configuration

- FC2, 1-hour causal aggregation, chronological 50% / 10% / 40% train /
  validation / test, split-contained windows.
- Input R3: `Utot_mean`, `U1_mean`–`U5_mean`, `J_mean`, `I_mean`, `I_std`,
  `I_p2p`.
- 15-hour history, four direct prediction horizons.
- Residual label `Utot(t+h)-Utot(t)` and restoration with the observed origin
  `Utot(t)` anchor.
- One-layer LSTM, hidden size 64, Linear 4-output head, MSE, Adam, seed 42,
  original learning rate, batch size, weight decay, and validation early stop.
- Input and residual scalers fit only training windows/residuals.

## Comparators and reports

The final test reports Persistence, 15-hour Linear Trend, and R3 Residual
MIMO-LSTM. For h=1–4 and their arithmetic mean, it writes MAE, RMSE, R²,
`Skill_P = 1 - RMSE_model/RMSE_persistence`, and
`Skill_L = 1 - RMSE_model/RMSE_linear`.

No other feature set, model architecture, seed, ensemble, horizon, OCM,
decomposition, or hyperparameter search is evaluated in this stage.
