# Phase N2.8C results

## Firewall and frozen task

All work used only the development prefix of the frozen 0.5 h / 48-step / 24-step / F2 / DeltaV task. Test target rows read: 0. No Test forecast, metric, feature diagnostic, or model-selection result was constructed.

## Q1: Can missing future exogenous context explain TimeXer failure?

Oracle future exogenous context did not reduce development RMSEΔV below 20 mV. Missing forecastable future exogenous input is not a sufficient explanation for TimeXer failure on this task.

## Q2–Q5: Future exogenous forecast quality

| model | variable | rmse_mean | mae_mean | r2_mean | folds |
|---|---|---|---|---|---|
| CycleRepeat | actual_current_A_mean | 16.547223 | 7.087253 | 0.939916 | 3 |
| CycleRepeat | actual_current_A_peak_to_peak | 37.523090 | 13.435149 | 0.875229 | 3 |
| CycleRepeat | actual_current_A_std | 8.593059 | 4.548795 | 0.922856 | 3 |
| CycleRepeat | actual_current_density_A_cm2_mean | 0.059097 | 0.025308 | 0.939914 | 3 |
| CycleRepeat | actual_current_density_A_cm2_peak_to_peak | 0.134002 | 0.047984 | 0.875245 | 3 |
| CycleRepeat | actual_current_density_A_cm2_std | 0.030696 | 0.016252 | 0.922825 | 3 |
| CycleRepeat | load_transition_count | 3.428554 | 1.823375 | 0.892219 | 3 |
| CycleRepeat | operation_point_transitions | 1.373616 | 0.999521 | 0.953821 | 3 |
| CycleRepeat | zero_current_fraction | 0.071914 | 0.010585 | 0.871502 | 3 |
| RidgeMultiHorizon | actual_current_A_mean | 7.827206 | 4.886696 | 0.986534 | 3 |
| RidgeMultiHorizon | actual_current_A_peak_to_peak | 19.889278 | 14.286609 | 0.964880 | 3 |
| RidgeMultiHorizon | actual_current_A_std | 4.109457 | 2.786785 | 0.982360 | 3 |
| RidgeMultiHorizon | actual_current_density_A_cm2_mean | 0.027952 | 0.017452 | 0.986535 | 3 |
| RidgeMultiHorizon | actual_current_density_A_cm2_peak_to_peak | 0.071032 | 0.051027 | 0.964881 | 3 |
| RidgeMultiHorizon | actual_current_density_A_cm2_std | 0.014683 | 0.009958 | 0.982345 | 3 |
| RidgeMultiHorizon | load_transition_count | 1.452778 | 1.070663 | 0.980603 | 3 |
| RidgeMultiHorizon | operation_point_transitions | 0.624813 | 0.490745 | 0.990479 | 3 |
| RidgeMultiHorizon | zero_current_fraction | 0.041624 | 0.024810 | 0.958304 | 3 |

Best mean R²: `RidgeMultiHorizon` / `operation_point_transitions` = 0.990479. Worst mean R²: `CycleRepeat` / `zero_current_fraction` = 0.871502.

## Q6–Q11: TimeXer comparison and exogenous gain

| model | eligibility | mean_r2_delta | mean_rmse_delta_mV | mean_mae_delta_mV | fold3_rmse_delta_mV | mean_p95_mV | mean_p99_mV | status |
|---|---|---|---|---|---|---|---|---|
| TimeXer-Original | reference_only | 0.965271 | 36.170404 | 18.967869 | 36.961407 | 53.684400 | 163.132030 | complete |
| TimeXer-OracleFutureExo | DIAGNOSTIC_ORACLE_ONLY | 0.985608 | 23.267980 | 14.898761 | 20.948443 | 40.711613 | 92.257565 | complete |
| TimeXer-CycleExo | FORMAL_CANDIDATE |  |  |  |  |  |  | not_run_oracle_terminal_stop |
| TimeXer-RidgeExo | FORMAL_CANDIDATE |  |  |  |  |  |  | not_run_oracle_terminal_stop |

| model | eligibility | mean_rmse_delta_mV | gain_pct | status |
|---|---|---|---|---|
| TimeXer-CycleExo | FORMAL_CANDIDATE |  |  | not_run_oracle_terminal_stop |
| TimeXer-RidgeExo | FORMAL_CANDIDATE |  |  | not_run_oracle_terminal_stop |
| TimeXer-OracleFutureExo | DIAGNOSTIC_ORACLE_ONLY | 23.267980 | 35.671218 | complete |

Oracle is `DIAGNOSTIC_ORACLE_ONLY` and never participates in formal ranking or three-seed promotion.

## Q12–Q13: Exogenous error versus voltage error

| model | fold | n_origins | pearson | spearman | status |
|---|---|---|---|---|---|
| not_run | all | 0 |  |  | not_run_oracle_terminal_stop |

| model | fold | stratum | n_origins | rmse_delta_mV | mae_delta_mV | p95_mV | p99_mV |
|---|---|---|---|---|---|---|---|
| not_run | all | not_run_oracle_terminal_stop | 0 |  |  |  |  |

## Q14–Q16: Fold3 and key horizons

Fold3 information is in the comparison table. At 9.5 h:

| model | horizon_hours | r2_delta | rmse_delta_mV | mae_delta_mV |
|---|---|---|---|---|
| TimeXer-OracleFutureExo | 9.500000 | 0.776450 | 18.902867 | 10.352966 |
| TimeXer-Original | 9.500000 | 0.901716 | 13.680394 | 5.895074 |

At 9.5 h, Oracle future exogenous context **does not help**: RMSEDeltaV rises from 13.680394 mV to 18.902867 mV.

At 12 h:

| model | horizon_hours | r2_delta | rmse_delta_mV | mae_delta_mV |
|---|---|---|---|---|
| TimeXer-OracleFutureExo | 12.000000 | 0.984784 | 24.703736 | 16.059920 |
| TimeXer-Original | 12.000000 | 0.952364 | 43.992350 | 22.742624 |

At 12 h, Oracle improves RMSEDeltaV from 43.992350 mV to 24.703736 mV, but that local benefit does not overcome the three-fold mean terminal gate. Oracle Fold3 RMSEDeltaV also improves from 36.961407 mV to 20.948443 mV (43.32% lower), while remaining a diagnostic-only result and above the 20 mV mean threshold.

## Q17–Q20: Pre-registered decision

- Oracle RMSEDeltaV = 23.267980 mV. Oracle terminal gate (>20 mV): `True`.
- Formal structural recovery (<20 mV): `False`.
- Three-seed threshold (<=11.9 mV) reached: `False`.
- Final TimeXer status: **TERMINAL_STOP**.
- Next-phase recommendation: **Lifecycle-Aware Robust TimesNet; do not continue the TimeXer route.**

## Frozen artifacts

- `PHASE_N2_8C_DECISION.yaml`
- `TIMEXER_FUTURE_EXO_ADAPTER_AUDIT.md`
- development-only CSV artifacts under `outputs/`
