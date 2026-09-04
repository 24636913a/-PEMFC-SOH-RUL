# Phase-2 seed-42 results — Persistence-Anchored Residual MIMO

This report is generated from completed Phase-2 artifacts.  No model was selected by ranking test results, no Phase-1 output was modified, and no multi-seed, feature-ladder, relative-input, OCM, or RUL experiment was run.

## Configuration

- FC1 and FC2: original causal 1 h aggregation, N=15, H=4, 50/10/40 split-contained windows, seed 42.
- Inputs: R4 `full_26`, `absolute_inputs`; no elapsed operating time.
- Neural models: same one-layer 64-hidden LSTM, Linear four-output head, MSE, Adam, unchanged Phase-1 learning rate/batch size/weight decay/early stopping.
- Residual label: `Utot(t+h)-Utot(t)`; its independent scaler is fitted only on flattened training residuals. Prediction is restored with the observed origin Utot anchor.

## Test metrics: FC1

| model | horizon | MAE | RMSE | R2 | persistence_RMSE | RMSE_improvement_vs_persistence | MSE_skill_vs_persistence |
|---|---|---:|---:|---:|---:|---:|---:|
| persistence | 1 | 0.000974 | 0.002566 | 0.954700 | 0.002566 | 0.000000 | 0.000000 |
| persistence | 2 | 0.001694 | 0.004320 | 0.871350 | 0.004320 | 0.000000 | 0.000000 |
| persistence | 3 | 0.002383 | 0.005573 | 0.785362 | 0.005573 | 0.000000 | 0.000000 |
| persistence | 4 | 0.003025 | 0.006626 | 0.696297 | 0.006626 | 0.000000 | 0.000000 |
| persistence | mean_h1_h4 | 0.002019 | 0.004771 | 0.826927 | 0.004771 | 0.000000 | 0.000000 |
| linear_trend | 1 | 0.002990 | 0.005788 | 0.769572 | 0.002566 | -1.255370 | -4.086692 |
| linear_trend | 2 | 0.003831 | 0.007300 | 0.632589 | 0.004320 | -0.689941 | -1.855900 |
| linear_trend | 3 | 0.004687 | 0.008732 | 0.473094 | 0.005573 | -0.566801 | -1.454864 |
| linear_trend | 4 | 0.005542 | 0.010106 | 0.293459 | 0.006626 | -0.525261 | -1.326423 |
| linear_trend | mean_h1_h4 | 0.004263 | 0.007981 | 0.542178 | 0.004771 | -0.759343 | -2.180970 |
| absolute_mimo_lstm | 1 | 0.039891 | 0.042850 | -11.631421 | 0.002566 | -15.698440 | -277.837908 |
| absolute_mimo_lstm | 2 | 0.041790 | 0.044113 | -12.417295 | 0.004320 | -9.212400 | -103.293119 |
| absolute_mimo_lstm | 3 | 0.041991 | 0.044402 | -12.623821 | 0.005573 | -6.967032 | -62.473591 |
| absolute_mimo_lstm | 4 | 0.044809 | 0.047451 | -14.576717 | 0.006626 | -6.161656 | -50.289318 |
| absolute_mimo_lstm | mean_h1_h4 | 0.042120 | 0.044704 | -12.812314 | 0.004771 | -9.509882 | -123.473484 |
| residual_mimo_lstm | 1 | 0.001292 | 0.002696 | 0.950010 | 0.002566 | -0.050488 | -0.103525 |
| residual_mimo_lstm | 2 | 0.001977 | 0.004381 | 0.867658 | 0.004320 | -0.014249 | -0.028701 |
| residual_mimo_lstm | 3 | 0.003397 | 0.006000 | 0.751240 | 0.005573 | -0.076557 | -0.158975 |
| residual_mimo_lstm | 4 | 0.003824 | 0.006956 | 0.665221 | 0.006626 | -0.049917 | -0.102325 |
| residual_mimo_lstm | mean_h1_h4 | 0.002623 | 0.005008 | 0.808532 | 0.004771 | -0.047803 | -0.098381 |

## Test metrics: FC2

| model | horizon | MAE | RMSE | R2 | persistence_RMSE | RMSE_improvement_vs_persistence | MSE_skill_vs_persistence |
|---|---|---:|---:|---:|---:|---:|---:|
| persistence | 1 | 0.005061 | 0.040793 | -0.066284 | 0.040793 | 0.000000 | 0.000000 |
| persistence | 2 | 0.006136 | 0.041005 | -0.074816 | 0.041005 | 0.000000 | 0.000000 |
| persistence | 3 | 0.007138 | 0.041261 | -0.085651 | 0.041261 | 0.000000 | 0.000000 |
| persistence | 4 | 0.008056 | 0.041528 | -0.097188 | 0.041528 | 0.000000 | 0.000000 |
| persistence | mean_h1_h4 | 0.006598 | 0.041147 | -0.080985 | 0.041147 | 0.000000 | 0.000000 |
| linear_trend | 1 | 0.008213 | 0.033562 | 0.278198 | 0.040793 | 0.177241 | 0.323067 |
| linear_trend | 2 | 0.009665 | 0.034800 | 0.225866 | 0.041005 | 0.151326 | 0.279752 |
| linear_trend | 3 | 0.011088 | 0.036172 | 0.165627 | 0.041261 | 0.123332 | 0.231454 |
| linear_trend | 4 | 0.012446 | 0.037650 | 0.098164 | 0.041528 | 0.093384 | 0.178048 |
| linear_trend | mean_h1_h4 | 0.010353 | 0.035546 | 0.191964 | 0.041147 | 0.136321 | 0.253080 |
| absolute_mimo_lstm | 1 | 0.033152 | 0.047502 | -0.445908 | 0.040793 | -0.164485 | -0.356026 |
| absolute_mimo_lstm | 2 | 0.032956 | 0.047075 | -0.416557 | 0.041005 | -0.148021 | -0.317953 |
| absolute_mimo_lstm | 3 | 0.028438 | 0.043527 | -0.208184 | 0.041261 | -0.054925 | -0.112867 |
| absolute_mimo_lstm | 4 | 0.029300 | 0.044242 | -0.245249 | 0.041528 | -0.065338 | -0.134945 |
| absolute_mimo_lstm | mean_h1_h4 | 0.030962 | 0.045586 | -0.328975 | 0.041147 | -0.108192 | -0.230448 |
| residual_mimo_lstm | 1 | 0.006968 | 0.040388 | -0.045247 | 0.040793 | 0.009914 | 0.019729 |
| residual_mimo_lstm | 2 | 0.007240 | 0.040340 | -0.040249 | 0.041005 | 0.016212 | 0.032161 |
| residual_mimo_lstm | 3 | 0.007743 | 0.040710 | -0.056822 | 0.041261 | 0.013366 | 0.026554 |
| residual_mimo_lstm | 4 | 0.009188 | 0.040804 | -0.059268 | 0.041528 | 0.017433 | 0.034561 |
| residual_mimo_lstm | mean_h1_h4 | 0.007785 | 0.040561 | -0.050397 | 0.041147 | 0.014231 | 0.028251 |

## Validation diagnostic metrics

Validation files are recorded for future validation-only formulation decisions; no feature set was chosen from the test table.

### FC1 validation

| model | horizon | MAE | RMSE | R2 | persistence_RMSE | RMSE_improvement_vs_persistence | MSE_skill_vs_persistence |
|---|---|---:|---:|---:|---:|---:|---:|
| persistence | 1 | 0.000920 | 0.002049 | 0.922691 | 0.002049 | 0.000000 | 0.000000 |
| persistence | 2 | 0.001667 | 0.003450 | 0.775255 | 0.003450 | 0.000000 | 0.000000 |
| persistence | 3 | 0.002433 | 0.004590 | 0.591767 | 0.004590 | 0.000000 | 0.000000 |
| persistence | 4 | 0.003130 | 0.005550 | 0.392070 | 0.005550 | 0.000000 | 0.000000 |
| persistence | mean_h1_h4 | 0.002038 | 0.003910 | 0.670446 | 0.003910 | 0.000000 | 0.000000 |
| linear_trend | 1 | 0.003163 | 0.005145 | 0.512447 | 0.002049 | -1.511289 | -5.306571 |
| linear_trend | 2 | 0.004111 | 0.006513 | 0.199120 | 0.003450 | -0.887725 | -2.563504 |
| linear_trend | 3 | 0.005087 | 0.007840 | -0.191180 | 0.004590 | -0.708184 | -1.917891 |
| linear_trend | 4 | 0.006051 | 0.009122 | -0.642390 | 0.005550 | -0.643658 | -1.701610 |
| linear_trend | mean_h1_h4 | 0.004603 | 0.007155 | -0.030501 | 0.003910 | -0.937714 | -2.872394 |
| absolute_mimo_lstm | 1 | 0.011967 | 0.013820 | -2.517655 | 0.002049 | -5.745469 | -44.501347 |
| absolute_mimo_lstm | 2 | 0.012584 | 0.014004 | -2.702044 | 0.003450 | -3.058595 | -15.472190 |
| absolute_mimo_lstm | 3 | 0.013192 | 0.014604 | -3.133052 | 0.004590 | -2.181862 | -9.124243 |
| absolute_mimo_lstm | 4 | 0.012970 | 0.014219 | -2.990829 | 0.005550 | -1.562152 | -5.564621 |
| absolute_mimo_lstm | mean_h1_h4 | 0.012679 | 0.014161 | -2.835895 | 0.003910 | -3.137019 | -18.665600 |
| residual_mimo_lstm | 1 | 0.000994 | 0.002060 | 0.921854 | 0.002049 | -0.005399 | -0.010827 |
| residual_mimo_lstm | 2 | 0.001508 | 0.003329 | 0.790852 | 0.003450 | 0.035323 | 0.069398 |
| residual_mimo_lstm | 3 | 0.002490 | 0.004443 | 0.617431 | 0.004590 | 0.031943 | 0.062866 |
| residual_mimo_lstm | 4 | 0.002906 | 0.005207 | 0.464759 | 0.005550 | 0.061686 | 0.119568 |
| residual_mimo_lstm | mean_h1_h4 | 0.001975 | 0.003760 | 0.698724 | 0.003910 | 0.030888 | 0.060251 |

### FC2 validation

| model | horizon | MAE | RMSE | R2 | persistence_RMSE | RMSE_improvement_vs_persistence | MSE_skill_vs_persistence |
|---|---|---:|---:|---:|---:|---:|---:|
| persistence | 1 | 0.001074 | 0.001517 | 0.959689 | 0.001517 | 0.000000 | 0.000000 |
| persistence | 2 | 0.001975 | 0.002774 | 0.860136 | 0.002774 | 0.000000 | 0.000000 |
| persistence | 3 | 0.002812 | 0.003850 | 0.720283 | 0.003850 | 0.000000 | 0.000000 |
| persistence | 4 | 0.003523 | 0.004736 | 0.561722 | 0.004736 | 0.000000 | 0.000000 |
| persistence | mean_h1_h4 | 0.002346 | 0.003219 | 0.775457 | 0.003219 | 0.000000 | 0.000000 |
| linear_trend | 1 | 0.003054 | 0.003933 | 0.728929 | 0.001517 | -1.593155 | -5.724453 |
| linear_trend | 2 | 0.003820 | 0.004945 | 0.555641 | 0.002774 | -0.782436 | -2.177079 |
| linear_trend | 3 | 0.004594 | 0.005914 | 0.339979 | 0.003850 | -0.536101 | -1.359605 |
| linear_trend | 4 | 0.005369 | 0.006825 | 0.089835 | 0.004736 | -0.441070 | -1.076683 |
| linear_trend | mean_h1_h4 | 0.004209 | 0.005404 | 0.428596 | 0.003219 | -0.838190 | -2.584455 |
| absolute_mimo_lstm | 1 | 0.005839 | 0.006910 | 0.163127 | 0.001517 | -3.556349 | -19.760312 |
| absolute_mimo_lstm | 2 | 0.007565 | 0.008799 | -0.406934 | 0.002774 | -2.171639 | -9.059296 |
| absolute_mimo_lstm | 3 | 0.005023 | 0.006203 | 0.273902 | 0.003850 | -0.611158 | -1.595831 |
| absolute_mimo_lstm | 4 | 0.006040 | 0.007307 | -0.043333 | 0.004736 | -0.542895 | -1.380525 |
| absolute_mimo_lstm | mean_h1_h4 | 0.006117 | 0.007305 | -0.003309 | 0.003219 | -1.720510 | -7.948991 |
| residual_mimo_lstm | 1 | 0.001877 | 0.002131 | 0.920393 | 0.001517 | -0.405281 | -0.974814 |
| residual_mimo_lstm | 2 | 0.002081 | 0.003045 | 0.831566 | 0.002774 | -0.097392 | -0.204270 |
| residual_mimo_lstm | 3 | 0.002716 | 0.003761 | 0.733058 | 0.003850 | 0.023103 | 0.045672 |
| residual_mimo_lstm | 4 | 0.003430 | 0.004486 | 0.606710 | 0.004736 | 0.052714 | 0.102649 |
| residual_mimo_lstm | mean_h1_h4 | 0.002526 | 0.003356 | 0.772932 | 0.003219 | -0.106714 | -0.257691 |

## Absolute-level distribution diagnostic

| Stack | Split | target mean (V) | target min (V) | target max (V) |
|---|---|---:|---:|---:|
| FC1 | train | 3.299532 | 3.258781 | 3.360202 |
| FC1 | validation | 3.249583 | 3.232798 | 3.261538 |
| FC1 | test | 3.227634 | 3.193433 | 3.264319 |
| FC2 | train | 3.234709 | 3.132269 | 3.462324 |
| FC2 | validation | 3.216620 | 3.203233 | 3.231084 |
| FC2 | test | 3.185298 | 3.127302 | 3.688911 |

## Decision path

**Current path: B.** FC1 residual recovers relative to absolute MIMO, while FC2 residual remains weaker than Linear trend.

The decision is a seed-42 diagnostic, not a claim of statistical stability.  The prescribed next experiment must be confirmed before any Phase-2b expanding validation, R1–R4 feature-ladder run, relative-input run, or OCM work.

## FC2 event diagnostics

Completed: `outputs_phase2\FC2\seed_42\FC2_EVENT_DIAGNOSTICS.md`. Events were retained and analyzed descriptively; they were not deleted or used as model inputs.
