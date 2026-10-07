# Causal same-cycle health-state estimation

## Protocol incident and repair

An earlier execution was invalidated because a whole-file integrity hash read beyond the frozen Development prefix of `causal_features_30min.csv`. No Test labels were shown to have been accessed. The whole-file causal hash was removed, a deterministic Development-prefix-only fingerprint and regression firewall tests were introduced, and every reported CSHSE result below was regenerated from a clean Python process. The prior invalidated artifacts are preserved only under `runtime_invalidated/` and are not used here.

## 1. Corrected estimand

TSHE and CSTL ended every 24 h history before the target cycle began, so they tested previous-operation-to-next-cycle health prediction. CSHSE preregistered the newest legal 48-bin history ending before the target P1S measurement and allowed that history to enter the target cycle. The estimand would be current P1S-B2 calibrated health-proxy estimation, not independently validated true SOH.

## 2. Frozen P1S measurement reconstruction and leakage control

The PAA `middle_50` raw-selection implementation was replayed for each frozen target. Every cycle's median-of-dwell voltage matched the immutable PAA table within the frozen tolerance. The reported measurement start/end is the envelope of actual raw observations used across its multiple contributing dwells. A 0.5 h guard precedes the earliest selected observation; every selected 30 min input bin ends at or before this cutoff. Therefore target P1S measurement observations are excluded.

| cycle_interval_id | p1s_dwell_count | p1s_raw_sample_count | p1s_measurement_start_ophrs2_h | p1s_measurement_end_ophrs2_h | P1S_voltage_difference_V | voltage_match_pass |
|---|---|---|---|---|---|---|
| COMPLETE_004 | 24 | 3099 | 105.97748000 | 114.66496000 | 0.00000000 | True |
| COMPLETE_005 | 24 | 3100 | 115.50023000 | 124.18799000 | 0.00000000 | True |
| COMPLETE_006 | 24 | 3100 | 125.02326000 | 133.71185000 | 0.00000000 | True |
| COMPLETE_007 | 24 | 3098 | 134.54684000 | 143.23404000 | 0.00000000 | True |
| COMPLETE_008 | 24 | 3096 | 144.06905000 | 152.75706000 | 0.00000000 | True |
| COMPLETE_009 | 24 | 3098 | 153.59207000 | 162.28149000 | 0.00000000 | True |
| COMPLETE_010 | 24 | 3098 | 163.11649000 | 171.80591000 | 0.00000000 | True |
| COMPLETE_011 | 24 | 3098 | 172.64090000 | 181.32838000 | 0.00000000 | True |
| COMPLETE_012 | 24 | 3102 | 182.16365000 | 190.85252000 | 0.00000000 | True |
| COMPLETE_013 | 24 | 3100 | 191.68752000 | 200.37639000 | 0.00000000 | True |
| COMPLETE_014 | 24 | 3101 | 201.21165000 | 210.39996000 | 0.00000000 | True |
| COMPLETE_018 | 24 | 3098 | 239.80154000 | 248.49014000 | 0.00000000 | True |
| COMPLETE_019 | 24 | 3098 | 249.32513000 | 258.01096000 | 0.00000000 | True |
| COMPLETE_020 | 24 | 3098 | 258.84622000 | 267.53342000 | 0.00000000 | True |
| COMPLETE_021 | 24 | 3098 | 268.36841000 | 277.05505000 | 0.00000000 | True |
| COMPLETE_022 | 24 | 3099 | 277.89005000 | 286.57919000 | 0.00000000 | True |
| COMPLETE_023 | 24 | 3099 | 287.41418000 | 296.10028000 | 0.00000000 | True |
| COMPLETE_024 | 24 | 3097 | 296.93527000 | 305.62274000 | 0.00000000 | True |
| COMPLETE_029 | 24 | 3100 | 360.22900000 | 368.91534000 | 0.00000000 | True |
| COMPLETE_030 | 24 | 3100 | 369.75037000 | 378.44006000 | 0.00000000 | True |
| COMPLETE_031 | 24 | 3099 | 379.27533000 | 387.96143000 | 0.00000000 | True |
| COMPLETE_032 | 24 | 3098 | 388.79669000 | 397.48416000 | 0.00000000 | True |
| COMPLETE_033 | 24 | 3099 | 398.31943000 | 407.00803000 | 0.00000000 | True |
| COMPLETE_034 | 24 | 3099 | 407.84302000 | 416.53049000 | 0.00000000 | True |
| COMPLETE_035 | 24 | 3097 | 417.36551000 | 426.05270000 | 0.00000000 | True |
| COMPLETE_036 | 24 | 3098 | 426.88797000 | 435.57571000 | 0.00000000 | True |
| COMPLETE_040 | 24 | 3100 | 465.24786000 | 473.93701000 | 0.00000000 | True |
| COMPLETE_041 | 24 | 3099 | 474.77200000 | 483.46060000 | 0.00000000 | True |
| COMPLETE_042 | 24 | 3099 | 484.29559000 | 492.98251000 | 0.00000000 | True |
| COMPLETE_043 | 24 | 3099 | 493.81750000 | 502.50552000 | 0.00000000 | True |
| COMPLETE_044 | 24 | 3098 | 503.34082000 | 512.02661000 | 0.00000000 | True |
| COMPLETE_045 | 24 | 3098 | 512.86163000 | 521.54962000 | 0.00000000 | True |
| COMPLETE_046 | 24 | 3101 | 522.38495000 | 531.07098000 | 0.00000000 | True |
| COMPLETE_047 | 24 | 3101 | 531.90631000 | 540.59406000 | 0.00000000 | True |
| COMPLETE_048 | 24 | 3097 | 541.42877000 | 550.11792000 | 0.00000000 | True |
| COMPLETE_049 | 24 | 3099 | 550.95288000 | 559.63953000 | 0.00000000 | True |
| COMPLETE_055 | 24 | 3099 | 657.89148000 | 666.58093000 | 0.00000000 | True |
| COMPLETE_056 | 24 | 3099 | 667.41589000 | 676.10229000 | 0.00000000 | True |
| COMPLETE_057 | 24 | 3099 | 676.93726000 | 685.62421000 | 0.00000000 | True |
| COMPLETE_058 | 24 | 3100 | 686.45947000 | 695.14886000 | 0.00000000 | True |
| COMPLETE_059 | 24 | 3100 | 695.98419000 | 704.67163000 | 0.00000000 | True |
| COMPLETE_060 | 24 | 3101 | 705.50665000 | 714.19635000 | 0.00000000 | True |
| COMPLETE_061 | 24 | 3102 | 715.03162000 | 723.71881000 | 0.00000000 | True |
| COMPLETE_062 | 24 | 3100 | 724.55408000 | 733.24213000 | 0.00000000 | True |
| COMPLETE_063 | 24 | 3099 | 734.07739000 | 742.76428000 | 0.00000000 | True |
| COMPLETE_064 | 24 | 3099 | 743.59930000 | 752.28955000 | 0.00000000 | True |
| COMPLETE_065 | 24 | 3098 | 753.12457000 | 761.81232000 | 0.00000000 | True |
| COMPLETE_066 | 24 | 3099 | 762.64758000 | 771.33618000 | 0.00000000 | True |
| COMPLETE_067 | 24 | 3102 | 772.17145000 | 780.85864000 | 0.00000000 | True |
| COMPLETE_068 | 24 | 3100 | 781.69391000 | 790.38165000 | 0.00000000 | True |
| COMPLETE_069 | 24 | 3100 | 791.21667000 | 799.90692000 | 0.00000000 | True |
| COMPLETE_070 | 23 | 2906 | 800.74219000 | 809.63745000 | 0.00000000 | True |
| COMPLETE_074 | 24 | 3098 | 839.04266000 | 847.73065000 | 0.00000000 | True |
| COMPLETE_079 | 24 | 3097 | 914.91663000 | 923.60687000 | 0.00000000 | True |
| COMPLETE_080 | 24 | 3099 | 924.44214000 | 933.13184000 | 0.00000000 | True |
| COMPLETE_081 | 24 | 3101 | 933.96710000 | 943.01599000 | 0.00000000 | True |
| COMPLETE_085 | 24 | 3100 | 972.42200000 | 981.47278000 | 0.00000000 | True |
| COMPLETE_089 | 24 | 3098 | 1010.88000000 | 1019.57000000 | 0.00000000 | True |
| COMPLETE_090 | 24 | 3099 | 1020.40000000 | 1029.09000000 | 0.00000000 | True |
| COMPLETE_091 | 24 | 3101 | 1029.93000000 | 1038.62000000 | 0.00000000 | True |
| COMPLETE_092 | 24 | 3099 | 1039.45000000 | 1048.14000000 | 0.00000000 | True |
| COMPLETE_093 | 24 | 3099 | 1048.98000000 | 1057.67000000 | 0.00000000 | True |
| COMPLETE_094 | 24 | 3098 | 1058.50000000 | 1067.19000000 | 0.00000000 | True |
| COMPLETE_095 | 24 | 3100 | 1068.03000000 | 1076.71000000 | 0.00000000 | True |
| COMPLETE_096 | 24 | 3099 | 1077.55000000 | 1086.24000000 | 0.00000000 | True |
| COMPLETE_097 | 24 | 3099 | 1087.07000000 | 1095.76000000 | 0.00000000 | True |
| COMPLETE_098 | 24 | 3098 | 1096.60000000 | 1105.29000000 | 0.00000000 | True |
| COMPLETE_099 | 24 | 3097 | 1106.12000000 | 1114.81000000 | 0.00000000 | True |
| COMPLETE_100 | 24 | 3097 | 1115.65000000 | 1124.33000000 | 0.00000000 | True |
| COMPLETE_105 | 24 | 3099 | 1257.67000000 | 1266.36000000 | 0.00000000 | True |
| COMPLETE_106 | 24 | 3099 | 1267.19000000 | 1275.88000000 | 0.00000000 | True |
| COMPLETE_107 | 24 | 3099 | 1276.72000000 | 1285.40000000 | 0.00000000 | True |
| COMPLETE_108 | 24 | 3101 | 1286.24000000 | 1294.93000000 | 0.00000000 | True |
| COMPLETE_109 | 24 | 3098 | 1295.76000000 | 1304.45000000 | 0.00000000 | True |
| COMPLETE_110 | 24 | 3101 | 1305.29000000 | 1313.98000000 | 0.00000000 | True |
| COMPLETE_111 | 24 | 3100 | 1314.81000000 | 1323.50000000 | 0.00000000 | True |
| COMPLETE_112 | 24 | 3051 | 1324.34000000 | 1333.03000000 | 0.00000000 | True |
| COMPLETE_113 | 24 | 3099 | 1333.86000000 | 1342.55000000 | 0.00000000 | True |
| COMPLETE_114 | 24 | 3099 | 1343.39000000 | 1352.07000000 | 0.00000000 | True |
| COMPLETE_115 | 24 | 3099 | 1352.91000000 | 1361.60000000 | 0.00000000 | True |
| COMPLETE_116 | 24 | 3099 | 1362.43000000 | 1371.12000000 | 0.00000000 | True |
| COMPLETE_117 | 24 | 3100 | 1371.96000000 | 1380.65000000 | 0.00000000 | True |
| COMPLETE_118 | 24 | 3099 | 1381.48000000 | 1390.17000000 | 0.00000000 | True |
| COMPLETE_121 | 24 | 3097 | 1410.05000000 | 1418.74000000 | 0.00000000 | True |
| COMPLETE_122 | 24 | 3099 | 1419.58000000 | 1428.27000000 | 0.00000000 | True |
| COMPLETE_123 | 24 | 3098 | 1429.10000000 | 1437.79000000 | 0.00000000 | True |
| COMPLETE_124 | 24 | 3099 | 1438.63000000 | 1447.32000000 | 0.00000000 | True |
| COMPLETE_125 | 24 | 3097 | 1448.15000000 | 1456.84000000 | 0.00000000 | True |
| COMPLETE_130 | 24 | 3100 | 1514.66000000 | 1523.35000000 | 0.00000000 | True |
| COMPLETE_131 | 24 | 3099 | 1524.18000000 | 1532.87000000 | 0.00000000 | True |
| COMPLETE_132 | 24 | 3099 | 1533.70000000 | 1542.39000000 | 0.00000000 | True |
| COMPLETE_133 | 24 | 3096 | 1543.23000000 | 1551.92000000 | 0.00000000 | True |
| COMPLETE_138 | 24 | 3098 | 1589.80000000 | 1598.49000000 | 0.00000000 | True |
| COMPLETE_139 | 24 | 3098 | 1599.32000000 | 1608.01000000 | 0.00000000 | True |
| COMPLETE_140 | 24 | 3100 | 1608.85000000 | 1617.54000000 | 0.00000000 | True |
| COMPLETE_141 | 24 | 3099 | 1618.37000000 | 1627.06000000 | 0.00000000 | True |
| COMPLETE_142 | 24 | 3100 | 1627.89000000 | 1636.58000000 | 0.00000000 | True |
| COMPLETE_143 | 24 | 3100 | 1637.42000000 | 1646.11000000 | 0.00000000 | True |
| COMPLETE_144 | 24 | 3099 | 1646.94000000 | 1655.63000000 | 0.00000000 | True |
| COMPLETE_145 | 24 | 3100 | 1656.47000000 | 1665.16000000 | 0.00000000 | True |
| COMPLETE_146 | 24 | 3097 | 1665.99000000 | 1674.68000000 | 0.00000000 | True |
| COMPLETE_147 | 24 | 3099 | 1675.52000000 | 1684.20000000 | 0.00000000 | True |
| COMPLETE_148 | 24 | 3099 | 1685.04000000 | 1693.73000000 | 0.00000000 | True |
| COMPLETE_149 | 24 | 3101 | 1694.56000000 | 1703.25000000 | 0.00000000 | True |
| COMPLETE_150 | 24 | 3098 | 1704.09000000 | 1712.78000000 | 0.00000000 | True |

## 3. Same-cycle information gained

    | quantity | count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|---|
| history_extension_h | 105.000000 | -0.380952 | 0.213980 | -0.500000 | -0.500000 | -0.500000 | -0.500000 | 0.000000 |

Mean available-history extension is **-0.381 h**. `same_cycle_history_used` is true for 0/105 legal source events. The frozen PAA timing places the primary P1S measurement immediately after cycle start. With the mandatory 0.5 h guard, no legal input reaches the target cycle. Therefore `FAIL__FROZEN_P1S_MEASUREMENT_PRECEDES_TARGET_CYCLE_DYNAMIC_HISTORY`: the computed metrics below are a guard-constrained pre-cycle descriptive replay, not evidence for the intended same-cycle current-state estimand.

## 4. Frozen representation and probe

The nine Final Backbone Screening DeltaV TimesNet checkpoints remained in evaluation mode with zero parameter updates. The CSTL checkpoint-scaler and sampled prediction-consistency replay were reused unchanged. The pre-output-projection `[B,72,32]` latent was temporally mean pooled and mapped by a chronological Ridge probe. The encoder never receives operating age, P1S measurements, P1S retention, or the proxy target.

## 5. Development results

| fold | pretrained_seed | n_train | n_validation | selected_ridge_alpha | MAE | RMSE | R2 | Spearman |
|---|---|---|---|---|---|---|---|---|
| fold_1 | 42 | 56 | 13 | 0.010000 | 0.002964 | 0.003362 | -4.936149 | 0.802198 |
| fold_1 | 2024 | 56 | 13 | 100.000000 | 0.007589 | 0.007743 | -30.482314 | 0.373626 |
| fold_1 | 3407 | 56 | 13 | 1.000000 | 0.004015 | 0.004671 | -10.459968 | -0.137363 |
| fold_2 | 42 | 69 | 13 | 0.000001 | 0.002290 | 0.003054 | -87.829846 | -0.560440 |
| fold_2 | 2024 | 69 | 13 | 0.000001 | 0.001429 | 0.002062 | -39.485960 | 0.703297 |
| fold_2 | 3407 | 69 | 13 | 0.000001 | 0.002281 | 0.002706 | -68.742848 | -0.428571 |
| fold_3 | 42 | 82 | 23 | 0.010000 | 0.003370 | 0.004274 | -44.725209 | -0.152174 |
| fold_3 | 2024 | 82 | 23 | 0.010000 | 0.002215 | 0.002608 | -16.022024 | 0.387352 |
| fold_3 | 3407 | 82 | 23 | 0.010000 | 0.003016 | 0.003825 | -35.618413 | -0.373518 |

## 6. Simple baselines

| fold | Age_MAE | LastVoltage_MAE | WindowVoltage_MAE | Age_R2 | LastVoltage_R2 | WindowVoltage_R2 |
|---|---|---|---|---|---|---|
| fold_1 | 0.002469 | 0.009041 | 0.000586 | -2.497263 | -43.409106 | 0.752847 |
| fold_2 | 0.006077 | 0.005836 | 0.000312 | -351.464928 | -326.214427 | -0.119693 |
| fold_3 | 0.005099 | 0.005614 | 0.000369 | -65.809074 | -78.799110 | 0.525988 |

## 7. Target spread and Fold 3

| fold | n_validation | target_mean | target_std | target_min | target_max | target_range | TimesNet_mean_RMSE | RMSE_to_target_std |
|---|---|---|---|---|---|---|---|---|
| fold_1 | 13 | 0.972585 | 0.001380 | 0.971446 | 0.976728 | 0.005283 | 0.005259 | 3.810863 |
| fold_2 | 13 | 0.974272 | 0.000324 | 0.973840 | 0.974937 | 0.001097 | 0.002607 | 8.046345 |
| fold_3 | 23 | 0.973462 | 0.000632 | 0.972552 | 0.974646 | 0.002093 | 0.003569 | 5.646377 |

| pretrained_seed | MAE | RMSE | R2 | Spearman | selected_ridge_alpha |
|---|---|---|---|---|---|
| 42 | 0.003370 | 0.004274 | -44.725209 | -0.152174 | 0.010000 |
| 2024 | 0.002215 | 0.002608 | -16.022024 | 0.387352 | 0.010000 |
| 3407 | 0.003016 | 0.003825 | -35.618413 | -0.373518 | 0.010000 |

## 8. Seed stability

| fold | MAE_mean | MAE_std | R2_mean | R2_std | Spearman_mean | Spearman_std | worst_seed_MAE | best_seed_MAE | MAE_coefficient_of_variation | worst_seed_over_best_seed_MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| fold_1 | 0.004856 | 0.001980 | -15.292810 | 10.974787 | 0.346154 | 0.384066 | 0.007589 | 0.002964 | 0.407715 | 2.560545 |
| fold_2 | 0.002000 | 0.000404 | -65.352885 | 19.881344 | -0.095238 | 0.567210 | 0.002290 | 0.001429 | 0.201800 | 1.602211 |
| fold_3 | 0.002867 | 0.000483 | -32.121882 | 11.976018 | -0.046113 | 0.319549 | 0.003370 | 0.002215 | 0.168459 | 1.521228 |

## 9. Decision

Overall gates A/B/C compare mean TimesNet MAE with Age, LastVoltage, and the three-variable WindowVoltage baseline. Fold 3 gates D/E/F require the mean to beat all three simple baselines, median Spearman above 0.5, and at least two positive-$R^2$ seeds; Gate G requires Fold 1 mean MAE to beat Age. They are reported as preregistered descriptive calculations, but cannot establish same-cycle current-state estimation if the timing availability gate fails.

**24H_DYNAMIC_HISTORY_NOT_SUFFICIENT_FOR_STABLE_P1S_B2_HEALTH_ESTIMATION**. Same-cycle timing availability: FAIL__FROZEN_P1S_MEASUREMENT_PRECEDES_TARGET_CYCLE_DYNAMIC_HISTORY. Gates A–G: PASS, PASS, FAIL, FAIL, FAIL, FAIL, FAIL.

## 10. Interpretation boundary and next step

The result tests association with a fold-specific P1S-B2 calibrated proxy, not true SOH. Test, late-life labels at or after 1819.997738 h, RUL, future-health forecasting, multitask training, encoder finetuning, and backbone search were not performed. Failure diagnostics are reported in the frozen summary as parallel plausibility checks rather than an automatic attribution to seed variation. The next action remains a research decision; no future-health trajectory experiment is started here.
