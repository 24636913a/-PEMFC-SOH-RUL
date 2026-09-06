# Phase 3D results — prediction-origin relative history diagnostic

All results are from the frozen front 60% development prefix and the three chronological folds only. The final 40% partition was not accessed.

## Frozen protocol

FC1/FC2; N=15; H=24; Utot-only residual MIMO-LSTM; one-layer hidden-64 encoder; seed 42; unchanged optimization and early stopping. Phase-3C-R's Nmax=48 legal origin schedule is retained, while both models see only the final 15 observations. The sole model input difference is absolute history versus prediction-origin-relative history.

## Residual-MIMO validation metrics

Values are means and standard deviations across all fold–horizon cells; the per-horizon table below supplies the direct three-fold mean/std at each h.

| fc | representation | MAE_mean | MAE_std | RMSE_mean | RMSE_std | R2_mean | R2_std | Skill_Persistence_mean | Skill_Persistence_std | Skill_Linear_mean | Skill_Linear_std | Skill_Best_mean | Skill_Best_std |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FC1 | absolute | 0.005839 | 0.003300 | 0.006873 | 0.003535 | -0.577925 | 1.008451 | 0.085602 | 0.186508 | 0.521073 | 0.129310 | 0.085602 | 0.186508 |
| FC1 | relative | 0.005463 | 0.002690 | 0.006995 | 0.003424 | -0.805491 | 1.480967 | 0.080845 | 0.100521 | 0.519170 | 0.113351 | 0.080845 | 0.100521 |
| FC2 | absolute | 0.007806 | 0.006030 | 0.011575 | 0.012313 | -0.912049 | 1.419657 | 0.066273 | 0.152116 | 0.352278 | 0.199786 | 0.064333 | 0.152319 |
| FC2 | relative | 0.007174 | 0.004945 | 0.011005 | 0.011998 | -0.770237 | 1.411706 | 0.109657 | 0.117165 | 0.384164 | 0.175535 | 0.107961 | 0.116246 |

## Forecast-skill summary

`MeanSkill_Best_all` averages h=1–24. `LateSkill_Best_h13_h24` averages h=13–24. Stable-positive means Skill_Best > 0 in every fold; crossover is the first of three consecutive stable-positive horizons.

| fc | representation | mean_RMSE | mean_RMSE_std_across_folds | MeanSkill_Best_all | LateSkill_Best_h13_h24 | stable_positive_horizon_count | stable_positive_horizons | crossover_h_c |
|---|---|---:|---:|---:|---:|---:|---|---:|
| FC1 | absolute | 0.006873 | 0.001469 | 0.085602 | 0.175161 | 20.000000 | 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 | 5.000000 |
| FC1 | relative | 0.006995 | 0.001599 | 0.080845 | 0.141706 | 12.000000 | 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 | 9.000000 |
| FC2 | absolute | 0.011575 | 0.009369 | 0.064333 | 0.068062 | 7.000000 | 3, 4, 5, 6, 7, 8, 21 | 3.000000 |
| FC2 | relative | 0.011005 | 0.008891 | 0.107961 | 0.132767 | 17.000000 | 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 | 3.000000 |

## Representation-level distribution-shift audit

Each value is the mean of 15 raw-voltage Wasserstein distances between train and validation input distributions at identical lags. `Delta_W_relative_minus_absolute_V < 0` indicates that prediction-origin normalization reduces this measured train-validation shift.

### Per-fold mean distance

| fc | fold | absolute | relative | Delta_W_relative_minus_absolute_V |
|---|---|---:|---:|---:|
| FC1 | fold_1 | 0.039393 | 0.003507 | -0.035887 |
| FC1 | fold_2 | 0.038773 | 0.002192 | -0.036581 |
| FC1 | fold_3 | 0.054862 | 0.003213 | -0.051649 |
| FC2 | fold_1 | 0.030607 | 0.004837 | -0.025769 |
| FC2 | fold_2 | 0.046555 | 0.005888 | -0.040667 |
| FC2 | fold_3 | 0.027756 | 0.005563 | -0.022193 |

### Across-fold summary

| fc | representation | mean_Wasserstein_V_across_folds | std_Wasserstein_V_across_folds | min_Wasserstein_V_across_folds | max_Wasserstein_V_across_folds |
|---|---|---:|---:|---:|---:|
| FC1 | absolute | 0.044343 | 0.009115 | 0.038773 | 0.054862 |
| FC1 | relative | 0.002970 | 0.000690 | 0.002192 | 0.003507 |
| FC2 | absolute | 0.034972 | 0.010131 | 0.027756 | 0.046555 |
| FC2 | relative | 0.005430 | 0.000538 | 0.004837 | 0.005888 |

## Paired 24-hour moving-block bootstrap — late h=13–24

`Delta_RMSE = RMSE(relative) − RMSE(absolute)`; a negative result favors relative history. Intervals are 95% quantiles from 2,000 paired moving-block resamples of fixed origins (24-hour blocks within each fold).

| fc | Delta_RMSE_mV | CI95_low_mV | CI95_high_mV | relative_better_probability | CI_excludes_zero |
|---|---:|---:|---:|---:|---:|
| FC1 | 0.112954 | -0.521849 | 0.624591 | 0.334000 | False |
| FC2 | -0.898878 | -1.798033 | -0.361730 | 1.000000 | True |

## Predeclared decision checks

A representation can be considered for a later robustness study only if at least two directional criteria improve without a material reversal of the third; paired intervals remain the evidence for whether the late difference is robust.

| fc | Delta_MeanSkill_relative_minus_absolute | Delta_LateSkill_relative_minus_absolute | Delta_stable_positive_horizons | relative_crossover_not_later | criteria_1_mean_skill_improved | criteria_2_late_skill_improved | criteria_3_stability_or_crossover_improved | late_Delta_RMSE_mV | late_CI95_low_mV | late_CI95_high_mV | late_CI_excludes_zero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FC1 | -0.004757 | -0.033456 | -8.000000 | False | False | False | False | 0.112954 | -0.521849 | 0.624591 | False |
| FC2 | 0.043628 | 0.064705 | 10.000000 | True | True | True | True | -0.898878 | -1.798033 | -0.361730 | True |

## Observed outcome

- **FC1:** relative history reduced the mean lag-wise Wasserstein shift in every fold and does not meet the predeclared evidence threshold for retaining relative history.
- **FC2:** relative history reduced the mean lag-wise Wasserstein shift in every fold and meets the predeclared directional criteria and has a late paired interval favoring relative history.

The result is stack-dependent under this one-seed development protocol: reduced marginal representation shift alone is not sufficient to guarantee a forecasting gain. No multi-seed confirmation, final evaluation, or backbone freeze is implied by this phase.

## Per-horizon three-fold mean/std

| fc | representation | horizon | RMSE_mean | RMSE_std | Skill_Best_mean | Skill_Best_std | positive_folds |
|---|---|---:|---:|---:|---:|---:|---:|
| FC1 | absolute | 1.000000 | 0.002185 | 0.000879 | -0.249454 | 0.271437 | 1.000000 |
| FC1 | absolute | 2.000000 | 0.003762 | 0.001132 | -0.328929 | 0.521180 | 0.000000 |
| FC1 | absolute | 3.000000 | 0.004548 | 0.002250 | -0.053788 | 0.051076 | 1.000000 |
| FC1 | absolute | 4.000000 | 0.005262 | 0.002663 | 0.004570 | 0.031958 | 2.000000 |
| FC1 | absolute | 5.000000 | 0.005804 | 0.002917 | 0.046285 | 0.008971 | 3.000000 |
| FC1 | absolute | 6.000000 | 0.006470 | 0.003247 | 0.052079 | 0.016408 | 3.000000 |
| FC1 | absolute | 7.000000 | 0.006999 | 0.003512 | 0.058085 | 0.025323 | 3.000000 |
| FC1 | absolute | 8.000000 | 0.007347 | 0.003653 | 0.060885 | 0.020318 | 3.000000 |
| FC1 | absolute | 9.000000 | 0.007436 | 0.003702 | 0.080093 | 0.054726 | 3.000000 |
| FC1 | absolute | 10.000000 | 0.007828 | 0.003897 | 0.065659 | 0.023733 | 3.000000 |
| FC1 | absolute | 11.000000 | 0.007693 | 0.003981 | 0.103175 | 0.058001 | 3.000000 |
| FC1 | absolute | 12.000000 | 0.007762 | 0.004064 | 0.113854 | 0.056493 | 3.000000 |
| FC1 | absolute | 13.000000 | 0.007820 | 0.004188 | 0.121593 | 0.098721 | 3.000000 |
| FC1 | absolute | 14.000000 | 0.007847 | 0.004355 | 0.139917 | 0.099190 | 3.000000 |
| FC1 | absolute | 15.000000 | 0.007959 | 0.004457 | 0.139948 | 0.101186 | 3.000000 |
| FC1 | absolute | 16.000000 | 0.007927 | 0.004572 | 0.152157 | 0.115193 | 3.000000 |
| FC1 | absolute | 17.000000 | 0.007824 | 0.004644 | 0.166986 | 0.129336 | 3.000000 |
| FC1 | absolute | 18.000000 | 0.007712 | 0.004669 | 0.177859 | 0.145358 | 3.000000 |
| FC1 | absolute | 19.000000 | 0.007652 | 0.004691 | 0.183022 | 0.148634 | 3.000000 |
| FC1 | absolute | 20.000000 | 0.007568 | 0.004584 | 0.183896 | 0.166818 | 3.000000 |
| FC1 | absolute | 21.000000 | 0.007299 | 0.004585 | 0.217181 | 0.172096 | 3.000000 |
| FC1 | absolute | 22.000000 | 0.007443 | 0.004616 | 0.201894 | 0.162832 | 3.000000 |
| FC1 | absolute | 23.000000 | 0.007470 | 0.004569 | 0.199726 | 0.172923 | 3.000000 |
| FC1 | absolute | 24.000000 | 0.007339 | 0.004333 | 0.217751 | 0.152170 | 3.000000 |
| FC1 | relative | 1.000000 | 0.001932 | 0.000948 | -0.059973 | 0.126537 | 2.000000 |
| FC1 | relative | 2.000000 | 0.003300 | 0.001758 | -0.013357 | 0.016175 | 0.000000 |
| FC1 | relative | 3.000000 | 0.004385 | 0.002350 | 0.002738 | 0.021065 | 1.000000 |
| FC1 | relative | 4.000000 | 0.005344 | 0.002638 | -0.020482 | 0.023971 | 1.000000 |
| FC1 | relative | 5.000000 | 0.006185 | 0.003244 | -0.002718 | 0.045500 | 1.000000 |
| FC1 | relative | 6.000000 | 0.006779 | 0.003260 | 0.000970 | 0.019545 | 1.000000 |
| FC1 | relative | 7.000000 | 0.007201 | 0.003340 | 0.020445 | 0.033591 | 2.000000 |
| FC1 | relative | 8.000000 | 0.007580 | 0.003246 | 0.008034 | 0.062883 | 1.000000 |
| FC1 | relative | 9.000000 | 0.007664 | 0.003478 | 0.044628 | 0.026771 | 3.000000 |
| FC1 | relative | 10.000000 | 0.007704 | 0.003457 | 0.067128 | 0.027318 | 3.000000 |
| FC1 | relative | 11.000000 | 0.007718 | 0.003370 | 0.081571 | 0.035795 | 3.000000 |
| FC1 | relative | 12.000000 | 0.007552 | 0.003109 | 0.110834 | 0.055357 | 3.000000 |
| FC1 | relative | 13.000000 | 0.007735 | 0.003537 | 0.118449 | 0.015492 | 3.000000 |
| FC1 | relative | 14.000000 | 0.007743 | 0.003469 | 0.129658 | 0.026639 | 3.000000 |
| FC1 | relative | 15.000000 | 0.007664 | 0.002913 | 0.132807 | 0.070553 | 3.000000 |
| FC1 | relative | 16.000000 | 0.007727 | 0.003467 | 0.147529 | 0.026143 | 3.000000 |
| FC1 | relative | 17.000000 | 0.007701 | 0.003674 | 0.157819 | 0.021825 | 3.000000 |
| FC1 | relative | 18.000000 | 0.007871 | 0.004039 | 0.145565 | 0.037863 | 3.000000 |
| FC1 | relative | 19.000000 | 0.007839 | 0.004131 | 0.148014 | 0.053540 | 3.000000 |
| FC1 | relative | 20.000000 | 0.007884 | 0.004584 | 0.150602 | 0.103844 | 3.000000 |
| FC1 | relative | 21.000000 | 0.008070 | 0.004969 | 0.136279 | 0.145139 | 2.000000 |
| FC1 | relative | 22.000000 | 0.008055 | 0.005195 | 0.144833 | 0.177038 | 2.000000 |
| FC1 | relative | 23.000000 | 0.008015 | 0.005097 | 0.150857 | 0.181199 | 2.000000 |
| FC1 | relative | 24.000000 | 0.008230 | 0.005396 | 0.138055 | 0.217531 | 2.000000 |
| FC2 | absolute | 1.000000 | 0.001725 | 0.000123 | -0.229566 | 0.234986 | 1.000000 |
| FC2 | absolute | 2.000000 | 0.003018 | 0.000957 | -0.203699 | 0.429809 | 2.000000 |
| FC2 | absolute | 3.000000 | 0.003267 | 0.000932 | 0.059242 | 0.015017 | 3.000000 |
| FC2 | absolute | 4.000000 | 0.003797 | 0.001234 | 0.102753 | 0.030846 | 3.000000 |
| FC2 | absolute | 5.000000 | 0.004265 | 0.001361 | 0.121555 | 0.023011 | 3.000000 |
| FC2 | absolute | 6.000000 | 0.004624 | 0.001517 | 0.144904 | 0.062156 | 3.000000 |
| FC2 | absolute | 7.000000 | 0.004954 | 0.001506 | 0.152118 | 0.087126 | 3.000000 |
| FC2 | absolute | 8.000000 | 0.005333 | 0.001265 | 0.141755 | 0.131575 | 3.000000 |
| FC2 | absolute | 9.000000 | 0.006286 | 0.001064 | 0.063273 | 0.081779 | 2.000000 |
| FC2 | absolute | 10.000000 | 0.006109 | 0.000863 | 0.137291 | 0.207651 | 2.000000 |
| FC2 | absolute | 11.000000 | 0.006823 | 0.000423 | 0.110782 | 0.184223 | 2.000000 |
| FC2 | absolute | 12.000000 | 0.007277 | 0.000725 | 0.126851 | 0.184315 | 2.000000 |
| FC2 | absolute | 13.000000 | 0.008909 | 0.003093 | 0.054528 | 0.092746 | 2.000000 |
| FC2 | absolute | 14.000000 | 0.009334 | 0.003147 | 0.064021 | 0.138669 | 2.000000 |
| FC2 | absolute | 15.000000 | 0.009152 | 0.002973 | 0.129275 | 0.141681 | 2.000000 |
| FC2 | absolute | 16.000000 | 0.010440 | 0.004648 | 0.082202 | 0.126700 | 2.000000 |
| FC2 | absolute | 17.000000 | 0.011484 | 0.006040 | 0.065023 | 0.128230 | 2.000000 |
| FC2 | absolute | 18.000000 | 0.012873 | 0.007849 | 0.052689 | 0.090315 | 2.000000 |
| FC2 | absolute | 19.000000 | 0.013852 | 0.008375 | 0.073542 | 0.101425 | 2.000000 |
| FC2 | absolute | 20.000000 | 0.027970 | 0.019301 | 0.052088 | 0.089605 | 2.000000 |
| FC2 | absolute | 21.000000 | 0.027818 | 0.019576 | 0.079749 | 0.079298 | 3.000000 |
| FC2 | absolute | 22.000000 | 0.028806 | 0.019778 | 0.056807 | 0.075027 | 2.000000 |
| FC2 | absolute | 23.000000 | 0.030119 | 0.020233 | 0.029535 | 0.084958 | 1.000000 |
| FC2 | absolute | 24.000000 | 0.029576 | 0.020820 | 0.077285 | 0.087208 | 2.000000 |
| FC2 | relative | 1.000000 | 0.001812 | 0.000535 | -0.248848 | 0.157638 | 0.000000 |
| FC2 | relative | 2.000000 | 0.002535 | 0.000768 | 0.017950 | 0.061739 | 1.000000 |
| FC2 | relative | 3.000000 | 0.003274 | 0.001169 | 0.069332 | 0.054193 | 3.000000 |
| FC2 | relative | 4.000000 | 0.003898 | 0.001519 | 0.092916 | 0.065716 | 3.000000 |
| FC2 | relative | 5.000000 | 0.004418 | 0.001749 | 0.106637 | 0.066789 | 3.000000 |
| FC2 | relative | 6.000000 | 0.004823 | 0.001817 | 0.120987 | 0.059426 | 3.000000 |
| FC2 | relative | 7.000000 | 0.005135 | 0.001795 | 0.133203 | 0.069144 | 3.000000 |
| FC2 | relative | 8.000000 | 0.005403 | 0.001511 | 0.140676 | 0.097088 | 3.000000 |
| FC2 | relative | 9.000000 | 0.005748 | 0.001255 | 0.143814 | 0.122431 | 3.000000 |
| FC2 | relative | 10.000000 | 0.006197 | 0.001037 | 0.138440 | 0.125670 | 3.000000 |
| FC2 | relative | 11.000000 | 0.006681 | 0.000959 | 0.139105 | 0.134196 | 3.000000 |
| FC2 | relative | 12.000000 | 0.007238 | 0.001200 | 0.143657 | 0.123850 | 3.000000 |
| FC2 | relative | 13.000000 | 0.007895 | 0.001821 | 0.143051 | 0.115481 | 3.000000 |
| FC2 | relative | 14.000000 | 0.008554 | 0.002519 | 0.138422 | 0.107313 | 3.000000 |
| FC2 | relative | 15.000000 | 0.008979 | 0.003242 | 0.155497 | 0.107377 | 3.000000 |
| FC2 | relative | 16.000000 | 0.009701 | 0.003943 | 0.142558 | 0.098180 | 3.000000 |
| FC2 | relative | 17.000000 | 0.010358 | 0.005000 | 0.150773 | 0.082430 | 3.000000 |
| FC2 | relative | 18.000000 | 0.011357 | 0.005954 | 0.142742 | 0.068656 | 3.000000 |
| FC2 | relative | 19.000000 | 0.012279 | 0.006932 | 0.168860 | 0.059882 | 3.000000 |
| FC2 | relative | 20.000000 | 0.026236 | 0.020009 | 0.120890 | 0.114869 | 2.000000 |
| FC2 | relative | 21.000000 | 0.027034 | 0.019937 | 0.102813 | 0.104622 | 2.000000 |
| FC2 | relative | 22.000000 | 0.027399 | 0.020454 | 0.117170 | 0.115047 | 2.000000 |
| FC2 | relative | 23.000000 | 0.028181 | 0.020700 | 0.102929 | 0.104338 | 2.000000 |
| FC2 | relative | 24.000000 | 0.028984 | 0.021304 | 0.107500 | 0.117821 | 2.000000 |

## Stop boundary

All per-run and paired representation audits must pass before interpretation. No final test, additional representation, new encoder, multi-seed robustness, probability model, or RUL experiment was run.
