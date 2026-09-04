# Phase 3A results — FC2 causal decomposition

All quantities in this report come from the first 60% development region and the three expanding validation folds. No final-test row, prediction, or score was used.

## T1 tau validation and selection

T1 input is causal slow-state D history only; target is Delta D(t+h). NRMSE is RMSE divided by the standard deviation of the corresponding target horizon.

| tau_hours | mean_NRMSE | mean_Skill_SlowPersistence | mean_Skill_SlowLinear | fold_mean_NRMSE_std | positive_SlowLinear_folds | within_3pct_best_NRMSE | validation_selected_tau |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3.000000 | 1.101383 | 0.047711 | 0.257519 | 0.209945 | 3.000000 | True | False |
| 6.000000 | 1.113142 | 0.066225 | 0.269378 | 0.250081 | 3.000000 | True | True |
| 12.000000 | 1.110286 | 0.107020 | 0.261260 | 0.278381 | 3.000000 | True | False |

Validation-selected tau: **6 h**.

## T1 neural summary versus saved T0 reference

| source | mean_RMSE | mean_NRMSE |
|---|---:|---:|
| Phase2b_saved_R1_raw_residual | 0.022587 | 1.046101 |

| tau_hours | MAE | RMSE | R2 | NRMSE | Skill_slow_persistence | Skill_slow_linear_trend |
|---:|---:|---:|---:|---:|---:|---:|
| 3.000000 | 0.004218 | 0.008509 | -0.250351 | 1.101383 | 0.047711 | 0.257519 |
| 6.000000 | 0.002841 | 0.005419 | -0.289473 | 1.113142 | 0.066225 | 0.269378 |
| 12.000000 | 0.001878 | 0.003325 | -0.297176 | 1.110286 | 0.107020 | 0.261260 |

### Selected-tau T1 stability by fold

| fold | MAE | RMSE | R2 | NRMSE | Skill_slow_persistence | Skill_slow_linear_trend |
|---|---:|---:|---:|---:|---:|---:|
| fold_1 | 0.003351 | 0.004466 | -0.987704 | 1.400596 | 0.114129 | 0.077132 |
| fold_2 | 0.004076 | 0.010538 | 0.013414 | 0.993252 | 0.026644 | 0.191089 |
| fold_3 | 0.001097 | 0.001252 | 0.105871 | 0.945578 | 0.057903 | 0.539912 |

## T2 transient forecasting at selected tau

T2-A uses R history only. T2-B adds legal historical J_mean/I_mean/I_std/I_p2p. The relevant skills are versus zero transient and transient persistence.

| variant | tau_hours | MAE | RMSE | R2 | NRMSE | Skill_zero_transient | Skill_transient_persistence |
|---|---:|---:|---:|---:|---:|---:|---:|
| T2A_R_history_only | 6.000000 | 0.005148 | 0.016145 | 0.254706 | 0.837110 | 0.264835 | -0.102796 |
| T2B_R_history_plus_load | 6.000000 | 0.004952 | 0.015969 | 0.313514 | 0.790765 | 0.307907 | -0.006915 |

### T2 stability by fold

| variant | fold | MAE | RMSE | R2 | NRMSE | Skill_zero_transient | Skill_transient_persistence |
|---|---|---:|---:|---:|---:|---:|---:|
| T2A_R_history_only | fold_1 | 0.003954 | 0.005571 | 0.310221 | 0.797910 | 0.497928 | -0.427850 |
| T2A_R_history_only | fold_2 | 0.009781 | 0.040760 | -0.027854 | 1.013828 | -0.006445 | 0.295651 |
| T2A_R_history_only | fold_3 | 0.001708 | 0.002103 | 0.481751 | 0.699593 | 0.303022 | -0.176188 |
| T2B_R_history_plus_load | fold_1 | 0.003795 | 0.005392 | 0.348830 | 0.772167 | 0.514264 | -0.371048 |
| T2B_R_history_plus_load | fold_2 | 0.009647 | 0.040751 | -0.027400 | 1.013606 | -0.006225 | 0.295802 |
| T2B_R_history_plus_load | fold_3 | 0.001414 | 0.001763 | 0.619111 | 0.586522 | 0.415681 | 0.054502 |

## Development-only reading

Selected T1 NRMSE is `1.113142` versus saved T0 NRMSE `1.046101`. T1 has positive Slow-Linear skill in `3/3` folds. T2-B lowers mean RMSE by `+1.09%` versus T2-A, which is a weak development-only load increment rather than a stable final conclusion.

Phase 3A does not meet the predeclared decomposition-success criteria: T1 NRMSE does not improve on saved T0; T2 is not harder than T1 in NRMSE. EWMA smoothing alone has not established a more forecastable degradation target.

## Stop boundary

No final test, OCM, additional decomposition variant, horizon extension, multi-seed run, uncertainty method, RUL experiment, or external validation was performed.
