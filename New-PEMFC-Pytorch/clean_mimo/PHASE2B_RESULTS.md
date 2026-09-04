# Phase-2b results — FC2 expanding validation feature ladder

This report contains development-only validation results. No final-test window was constructed, evaluated, or used for selection.

## Frozen protocol

- FC2, seed 42, causal hourly aggregation, N=15, H=4, residual MIMO-LSTM (one-layer LSTM, hidden 64) with train-only fold scalers.
- Folds: 0–35/35–40%, 0–45/45–50%, and 0–55/55–60% train/validation boundaries; the final 40% remains inaccessible to this runner.
- R1=`Utot_mean`; R2=R1 plus U1–U5; R3=R2 plus J_mean/I_mean/I_std/I_p2p. Linear Trend is the primary analytical comparison.

## Validation-only feature selection

Selection rule: lowest unweighted residual-MIMO RMSE over 3 folds × 4 horizons; exact ties use higher Skill_L then fixed R1/R2/R3 order.

| validation_rank | feature_set | mean_validation_MAE | mean_validation_RMSE | mean_validation_R2 | mean_Skill_P | mean_Skill_L | predeclared_order |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1.000000 | R3_state_load | 0.009040 | 0.022446 | 0.309148 | 0.013861 | 0.128328 | 2.000000 |
| 2.000000 | R1_utot_only | 0.009287 | 0.022587 | 0.285658 | 0.044692 | 0.148573 | 0.000000 |
| 3.000000 | R2_voltage_state | 0.010180 | 0.022825 | 0.290413 | -0.038619 | 0.096760 | 1.000000 |

Validation-selected feature set: **R3_state_load**.

## Residual MIMO average across h=1–4 by fold

| fold | feature_set | mean_h1_h4_MAE | mean_h1_h4_RMSE | mean_h1_h4_R2 | mean_h1_h4_Skill_P | mean_h1_h4_Skill_L |
|---|---|---:|---:|---:|---:|---:|
| fold_1 | R1_utot_only | 0.005017 | 0.006773 | 0.894385 | 0.043759 | 0.131365 |
| fold_1 | R2_voltage_state | 0.005784 | 0.007747 | 0.871910 | -0.187459 | -0.027196 |
| fold_1 | R3_state_load | 0.005397 | 0.007364 | 0.880158 | -0.075084 | 0.041486 |
| fold_2 | R1_utot_only | 0.020916 | 0.058525 | -0.582146 | 0.083834 | -0.164063 |
| fold_2 | R2_voltage_state | 0.022766 | 0.058284 | -0.569378 | 0.087597 | -0.159164 |
| fold_2 | R3_state_load | 0.019849 | 0.057575 | -0.531212 | 0.098744 | -0.145399 |
| fold_3 | R1_utot_only | 0.001926 | 0.002464 | 0.544736 | 0.006482 | 0.478417 |
| fold_3 | R2_voltage_state | 0.001990 | 0.002442 | 0.568708 | -0.015993 | 0.476641 |
| fold_3 | R3_state_load | 0.001875 | 0.002397 | 0.578499 | 0.017923 | 0.488898 |

For the validation-selected set, mean Skill_L is positive in `2/3` folds. The rank is a validation-only decision rule, not evidence by itself that condition information has a stable increment over the linear baseline.

## Residual MIMO by horizon across folds

`Skill_P = 1 - RMSE_model/RMSE_persistence`; `Skill_L = 1 - RMSE_model/RMSE_linear`. Values below are fold mean/std/min/max; they are not final-test estimates.

| feature_set | horizon | MAE_mean | MAE_std | MAE_min | MAE_max | RMSE_mean | RMSE_std | RMSE_min | RMSE_max | R2_mean | R2_std | R2_min | R2_max | Skill_P_mean | Skill_P_std | Skill_P_min | Skill_P_max | Skill_L_mean | Skill_L_std | Skill_L_min | Skill_L_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R1_utot_only | 1.000000 | 0.006192 | 0.008317 | 0.000862 | 0.015775 | 0.020414 | 0.032108 | 0.001121 | 0.057479 | 0.453004 | 0.864172 | -0.544171 | 0.983606 | 0.084575 | 0.063831 | 0.012448 | 0.133779 | 0.329715 | 0.441973 | -0.171067 | 0.665273 |
| R1_utot_only | 2.000000 | 0.008260 | 0.009619 | 0.001598 | 0.019288 | 0.021756 | 0.031178 | 0.002069 | 0.057703 | 0.371861 | 0.799336 | -0.543126 | 0.934411 | 0.046841 | 0.044546 | 0.006684 | 0.094757 | 0.176939 | 0.331497 | -0.163239 | 0.499013 |
| R1_utot_only | 3.000000 | 0.010227 | 0.011056 | 0.002279 | 0.022854 | 0.023061 | 0.030893 | 0.002957 | 0.058632 | 0.245626 | 0.750704 | -0.580614 | 0.885797 | 0.047822 | 0.042542 | -0.001140 | 0.075755 | 0.101131 | 0.280307 | -0.159946 | 0.397352 |
| R1_utot_only | 4.000000 | 0.012467 | 0.011853 | 0.002964 | 0.025749 | 0.025117 | 0.030694 | 0.003709 | 0.060284 | 0.072142 | 0.717710 | -0.660671 | 0.773728 | -0.000470 | 0.062429 | -0.066677 | 0.057329 | -0.013493 | 0.318401 | -0.230509 | 0.352031 |
| R2_voltage_state | 1.000000 | 0.006880 | 0.007917 | 0.001123 | 0.015908 | 0.020934 | 0.031208 | 0.001303 | 0.056920 | 0.442601 | 0.829230 | -0.514296 | 0.950781 | -0.177587 | 0.309601 | -0.500909 | 0.116172 | 0.192025 | 0.389646 | -0.159683 | 0.610881 |
| R2_voltage_state | 2.000000 | 0.008876 | 0.010210 | 0.001603 | 0.020548 | 0.021943 | 0.030817 | 0.002115 | 0.057446 | 0.365711 | 0.781802 | -0.529407 | 0.914694 | -0.004193 | 0.097822 | -0.095872 | 0.098790 | 0.137253 | 0.326468 | -0.158056 | 0.487826 |
| R2_voltage_state | 3.000000 | 0.011292 | 0.011990 | 0.002291 | 0.024903 | 0.023389 | 0.030330 | 0.002835 | 0.058224 | 0.251512 | 0.724127 | -0.558652 | 0.835722 | 0.001810 | 0.104851 | -0.116787 | 0.082199 | 0.050017 | 0.322647 | -0.151860 | 0.422129 |
| R2_voltage_state | 4.000000 | 0.013671 | 0.014146 | 0.002942 | 0.029703 | 0.025033 | 0.030985 | 0.003516 | 0.060546 | 0.101830 | 0.735167 | -0.675159 | 0.786445 | 0.025495 | 0.053583 | -0.036269 | 0.059529 | 0.007746 | 0.327649 | -0.195431 | 0.385728 |
| R3_state_load | 1.000000 | 0.005999 | 0.007702 | 0.000839 | 0.014852 | 0.020665 | 0.031971 | 0.001154 | 0.057562 | 0.446958 | 0.862724 | -0.548626 | 0.974775 | 0.004936 | 0.092313 | -0.074500 | 0.106210 | 0.285356 | 0.421018 | -0.172755 | 0.655325 |
| R3_state_load | 2.000000 | 0.008042 | 0.008709 | 0.001589 | 0.017948 | 0.021792 | 0.030434 | 0.002151 | 0.056849 | 0.372005 | 0.760499 | -0.497759 | 0.911729 | -0.013072 | 0.112738 | -0.114755 | 0.108163 | 0.133116 | 0.317930 | -0.146012 | 0.479188 |
| R3_state_load | 3.000000 | 0.009901 | 0.009956 | 0.002199 | 0.021144 | 0.022876 | 0.029781 | 0.002782 | 0.057091 | 0.282221 | 0.698635 | -0.498612 | 0.848180 | 0.028223 | 0.090632 | -0.073607 | 0.100049 | 0.075575 | 0.310729 | -0.129457 | 0.433088 |
| R3_state_load | 4.000000 | 0.012218 | 0.011782 | 0.002871 | 0.025452 | 0.024450 | 0.029985 | 0.003503 | 0.058798 | 0.135409 | 0.685194 | -0.579849 | 0.785947 | 0.035357 | 0.063683 | -0.037475 | 0.080555 | 0.019266 | 0.320897 | -0.196822 | 0.387990 |

## Stop boundary

No final-test evaluation, Full-26 run, relative-input run, OCM, causal decomposition, horizon expansion, multiple seeds, uncertainty experiment, or RUL work was performed.
