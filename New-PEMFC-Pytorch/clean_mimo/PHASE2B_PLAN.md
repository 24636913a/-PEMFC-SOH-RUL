# Phase 2b — FC2 information-predictability validation

## Purpose and scope

Phase 2b determines whether **legal history at the prediction origin** contains
incremental information for FC2 residual forecasting.  It does not introduce a
new architecture, change the target definition, or evaluate the held-out final
40% test partition.

The fixed target remains:

```text
delta_U[h] = Utot(t+h) - Utot(t), h = 1..4
Uhat(t+h) = observed Utot(t) + delta_Uhat(t+h)
```

The anchor is the final observed `Utot_mean` in the 15-hour input history.

## Frozen development protocol

Only FC2 and seed 42 are used.  The first 60% of its hourly life is the
development region.  Three predeclared chronological folds are evaluated:

| Fold | Train rows | Validation rows |
|---|---|---|
| 1 | 0%–35% | 35%–40% |
| 2 | 0%–45% | 45%–50% |
| 3 | 0%–55% | 55%–60% |

Each fold constructs 15-hour-to-4-hour MIMO windows *after* its boundaries are
set.  A scaler fits only flattened training windows/residual labels from that
fold.  No validation window crosses a fold boundary.  No test window is
constructed, scored, exported, or used to choose a feature set.

The LSTM stays one layer, hidden size 64, MSE, Adam, batch size 64, learning
rate `1e-3`, weight decay `1e-5`, and validation early stopping.

## Predeclared feature ladder

| ID | Causal input history |
|---|---|
| R1 `utot_only` | `Utot_mean` |
| R2 `voltage_state` | `Utot_mean`, `U1_mean`–`U5_mean` |
| R3 `state_load` | R2 plus `J_mean`, `I_mean`, `I_std`, `I_p2p` |

All entries use observed historical hourly values only.  No full-26 run,
relative-input run, elapsed time, feature search, or hyperparameter search is
included in Phase 2b.

## Metrics and selection rule

For every fold, feature set, and horizon, Phase 2b reports MAE, RMSE, R², and:

```text
Skill_P = 1 - RMSE_model / RMSE_persistence
Skill_L = 1 - RMSE_model / RMSE_linear_trend
```

The selected feature set is the one with the lowest unweighted mean residual
MIMO validation RMSE over all 3 folds × 4 horizons.  Exact ties are broken by
higher mean `Skill_L`, then the predefined R1/R2/R3 order.  This decision is
derived from validation artifacts only; final-test evaluation is deliberately
out of scope for this run.

## Stop boundary

After the FC2 Phase-2b R1/R2/R3 validation report is produced, work stops for
review.  It does not run the frozen final test, OCM, causal transient
decomposition, horizon expansion, multiple seeds, uncertainty estimation,
true-future RUL, or external validation.
