# Frozen R3 final-test results — FC2

This report comes from the single authorized final 40% test evaluation after R3 was selected by Phase-2b validation only. No other feature set or model was evaluated in this final-test stage.

## Locked configuration

R3 State+Load (`Utot`, U1–U5, J_mean, I_mean, I_std, I_p2p); 1 h; N=15; H=4; residual target; one-layer hidden-64 LSTM; MSE/Adam; seed 42; original train/validation/test and scaler rules.

## Final-test metrics

`Skill_P = 1 - RMSE_model/RMSE_persistence`; `Skill_L = 1 - RMSE_model/RMSE_linear`.

| model | horizon | MAE | RMSE | R2 | Skill_P | Skill_L |
|---|---|---:|---:|---:|---:|---:|
| persistence | 1 | 0.005061 | 0.040793 | -0.066284 | 0.000000 | -0.215423 |
| persistence | 2 | 0.006136 | 0.041005 | -0.074816 | 0.000000 | -0.178308 |
| persistence | 3 | 0.007138 | 0.041261 | -0.085651 | 0.000000 | -0.140683 |
| persistence | 4 | 0.008056 | 0.041528 | -0.097188 | 0.000000 | -0.103003 |
| linear_trend | 1 | 0.008213 | 0.033562 | 0.278198 | 0.177241 | 0.000000 |
| linear_trend | 2 | 0.009665 | 0.034800 | 0.225866 | 0.151326 | 0.000000 |
| linear_trend | 3 | 0.011088 | 0.036172 | 0.165627 | 0.123332 | 0.000000 |
| linear_trend | 4 | 0.012446 | 0.037650 | 0.098164 | 0.093384 | 0.000000 |
| R3_residual_mimo_lstm | 1 | 0.006186 | 0.039010 | 0.024866 | 0.043696 | -0.162313 |
| R3_residual_mimo_lstm | 2 | 0.007912 | 0.039280 | 0.013703 | 0.042063 | -0.128745 |
| R3_residual_mimo_lstm | 3 | 0.009106 | 0.039049 | 0.027627 | 0.053607 | -0.079535 |
| R3_residual_mimo_lstm | 4 | 0.012096 | 0.040578 | -0.047561 | 0.022877 | -0.077770 |
| persistence | mean_h1_h4 | 0.006598 | 0.041147 | -0.080985 | 0.000000 | -0.159354 |
| linear_trend | mean_h1_h4 | 0.010353 | 0.035546 | 0.191964 | 0.136321 | 0.000000 |
| R3_residual_mimo_lstm | mean_h1_h4 | 0.008825 | 0.039479 | 0.004658 | 0.040561 | -0.112090 |

## Frozen-test interpretation

R3 mean RMSE relative to Linear Trend: `-11.07%`. R3 has positive Skill_L at `0/4` horizons.

R3 is materially weaker than Linear Trend overall; the next scientific branch is causal slow-degradation versus reversible-transient decomposition, not a more complex sequence model.

## Integrity record

The result artifact records `test_evaluation_count = 1`. The completion marker blocks a repeat run; Phase-1, Phase-2, and Phase-2b validation outputs were not modified.
