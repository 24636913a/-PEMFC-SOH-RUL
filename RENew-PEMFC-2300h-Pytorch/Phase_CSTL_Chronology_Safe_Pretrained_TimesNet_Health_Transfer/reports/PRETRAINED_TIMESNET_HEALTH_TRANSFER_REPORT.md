# Pretrained TimesNet health transfer

## 1. Why scratch TSHE was insufficient

The earlier TSHE scratch end-to-end experiment retained strict target-cycle isolation but did not meet its Fold 3 explained-variation gate. It remains an immutable historical baseline and is not numerically ranked against this phase because its targets came from a global six-anchor B2 mapping.

## 2. Chronology-safe B2 supervision

This phase uses only `checkpoint`, `Y`, and `R_raw` from the CNP1S anchor authority. Fold 1 calibrates with 300/600/850 h; folds 2 and 3 use 300/600/850/1150 h. The 1500 and 1750 h anchors are excluded from Fold 3 target construction. The global six-anchor B2 fit is retained only as a descriptive reference.

## 3. Fold-specific calibration coefficients

| fold | analysis_role | train_end_h | anchor_count | anchor_checkpoints | a_fold | b_fold | anchor_MAE | anchor_RMSE | anchor_Spearman | future_anchor_used |
|---|---|---|---|---|---|---|---|---|---|---|
| fold_1 | TARGET_AUTHORITY | 946.39321400 | 3 | 300 / 600 / 850 | 1.24620612 | -0.25610342 | 0.00003677 | 0.00004037 | 1.00000000 | False |
| fold_2 | TARGET_AUTHORITY | 1164.79434500 | 4 | 300 / 600 / 850 / 1150 | 1.14536499 | -0.15594924 | 0.00022817 | 0.00024403 | 1.00000000 | False |
| fold_3 | TARGET_AUTHORITY | 1383.19547600 | 4 | 300 / 600 / 850 / 1150 | 1.14536499 | -0.15594924 | 0.00022817 | 0.00024403 | 1.00000000 | False |
| global_six_anchor_reference | DESCRIPTIVE_REFERENCE_ONLY |  | 6 | 300 / 600 / 850 / 1150 / 1500 / 1750 | 1.19881149 | -0.20903625 | 0.00030907 | 0.00041600 | 0.94285714 | NOT_APPLICABLE_DESCRIPTIVE_REFERENCE |

The chronology-safe target displacement relative to the historical global-six-anchor B2 reference is:

| fold | role | event_count | mean_signed_shift | mean_absolute_shift | max_absolute_shift | mean_absolute_shift_percentage_points | max_absolute_shift_percentage_points |
|---|---|---|---|---|---|---|---|
| fold_1 | train | 56 | -0.00000054 | 0.00015987 | 0.00034938 | 0.01598657 | 0.03493782 |
| fold_1 | validation | 13 | -0.00033875 | 0.00033875 | 0.00038210 | 0.03387539 | 0.03820968 |
| fold_2 | train | 69 | 0.00008223 | 0.00022162 | 0.00044066 | 0.02216200 | 0.04406553 |
| fold_2 | validation | 13 | 0.00034716 | 0.00034716 | 0.00036731 | 0.03471609 | 0.03673116 |
| fold_3 | train | 82 | 0.00012423 | 0.00024152 | 0.00044066 | 0.02415228 | 0.04406553 |
| fold_3 | validation | 23 | 0.00038498 | 0.00038498 | 0.00042740 | 0.03849830 | 0.04274028 |

This demonstrates the numerical scale change induced by chronology-safe calibration. It is a descriptive target-definition comparison, not a post-hoc statistical test or selection criterion.

## 4. DeltaV-pretrained TimesNet authority

All nine formal Final Backbone Screening TimesNet checkpoints were loaded. Inputs use their checkpoint-serialized outer-train median/mean/scale preprocessing, not a health-event scaler.

## 5. Pretrained preprocessing reproduction

| fold | seed | checkpoint | sampled_origin_rows | sample_count | reconstructed_preprocessor | max_abs_prediction_difference_mV | mean_abs_prediction_difference_mV | tolerance_mV | pass |
|---|---|---|---|---|---|---|---|---|---|
| fold_1 | 42 | RENew-PEMFC-2300h-Pytorch\Phase_BS_Final_Backbone_Screening\checkpoints\bs_timesnet_fold_1_seed42.pt | 1675|1676|1677 | 3 | checkpoint.outer_train_feature_medians_mean_scale | 0.137985 | 0.017541 | 1.000000 | True |
| fold_1 | 2024 | RENew-PEMFC-2300h-Pytorch\Phase_BS_Final_Backbone_Screening\checkpoints\bs_timesnet_fold_1_seed2024.pt | 1675|1676|1677 | 3 | checkpoint.outer_train_feature_medians_mean_scale | 0.108751 | 0.022720 | 1.000000 | True |
| fold_1 | 3407 | RENew-PEMFC-2300h-Pytorch\Phase_BS_Final_Backbone_Screening\checkpoints\bs_timesnet_fold_1_seed3407.pt | 1675|1676|1677 | 3 | checkpoint.outer_train_feature_medians_mean_scale | 0.895679 | 0.081206 | 1.000000 | True |
| fold_2 | 42 | RENew-PEMFC-2300h-Pytorch\Phase_BS_Final_Backbone_Screening\checkpoints\bs_timesnet_fold_2_seed42.pt | 1980|1981|1982 | 3 | checkpoint.outer_train_feature_medians_mean_scale | 0.039637 | 0.006033 | 1.000000 | True |
| fold_2 | 2024 | RENew-PEMFC-2300h-Pytorch\Phase_BS_Final_Backbone_Screening\checkpoints\bs_timesnet_fold_2_seed2024.pt | 1980|1981|1982 | 3 | checkpoint.outer_train_feature_medians_mean_scale | 0.008609 | 0.002157 | 1.000000 | True |
| fold_2 | 3407 | RENew-PEMFC-2300h-Pytorch\Phase_BS_Final_Backbone_Screening\checkpoints\bs_timesnet_fold_2_seed3407.pt | 1980|1981|1982 | 3 | checkpoint.outer_train_feature_medians_mean_scale | 0.350527 | 0.070469 | 1.000000 | True |
| fold_3 | 42 | RENew-PEMFC-2300h-Pytorch\Phase_BS_Final_Backbone_Screening\checkpoints\bs_timesnet_fold_3_seed42.pt | 2297|2298|2299 | 3 | checkpoint.outer_train_feature_medians_mean_scale | 0.020741 | 0.004117 | 1.000000 | True |
| fold_3 | 2024 | RENew-PEMFC-2300h-Pytorch\Phase_BS_Final_Backbone_Screening\checkpoints\bs_timesnet_fold_3_seed2024.pt | 2297|2298|2299 | 3 | checkpoint.outer_train_feature_medians_mean_scale | 0.064388 | 0.006951 | 1.000000 | True |
| fold_3 | 3407 | RENew-PEMFC-2300h-Pytorch\Phase_BS_Final_Backbone_Screening\checkpoints\bs_timesnet_fold_3_seed3407.pt | 2297|2298|2299 | 3 | checkpoint.outer_train_feature_medians_mean_scale | 0.031233 | 0.004841 | 1.000000 | True |

Each checkpoint reproduced a small saved Development-origin sample within 1.0 mV. This is an integrity replay, not a new forecast experiment.

## 6. Frozen latent extraction

The original forecasting body was replayed unchanged through the frozen 48→72 temporal transformation, DataEmbedding, TimesBlock, and layer normalization. The extracted representation is the actual `[B, 72, 32]` pre-output-projection latent sequence. Temporal mean pooling yields a 32-dimensional vector. All TimesNet parameters have `requires_grad=False`; zero encoder parameters were updated.

## 7. Ridge health probe

For each fold and pretrained seed, chronological inner selection uses the last 15% of outer-train health events and the fixed alpha grid. Only embeddings are standardized; proxy targets remain on their original scale. The final Ridge refit uses the full outer train and is evaluated once on the outer validation interval.

## 8. Development results

| fold | pretrained_seed | n_train | n_validation | selected_ridge_alpha | MAE | RMSE | R2 | Spearman |
|---|---|---|---|---|---|---|---|---|
| fold_1 | 42 | 56 | 13 | 0.010000 | 0.004445 | 0.004978 | -12.012796 | 0.620879 |
| fold_1 | 2024 | 56 | 13 | 1.000000 | 0.004617 | 0.005121 | -12.771859 | 0.060440 |
| fold_1 | 3407 | 56 | 13 | 1.000000 | 0.004460 | 0.005088 | -12.594022 | -0.214286 |
| fold_2 | 42 | 69 | 13 | 0.010000 | 0.001574 | 0.001913 | -33.855543 | 0.054945 |
| fold_2 | 2024 | 69 | 13 | 0.010000 | 0.002957 | 0.003529 | -117.588929 | -0.516484 |
| fold_2 | 3407 | 69 | 13 | 0.010000 | 0.002223 | 0.003039 | -86.966055 | 0.225275 |
| fold_3 | 42 | 82 | 23 | 1.000000 | 0.001988 | 0.002328 | -12.567631 | 0.234190 |
| fold_3 | 2024 | 82 | 23 | 1.000000 | 0.001452 | 0.001897 | -8.004142 | 0.686759 |
| fold_3 | 3407 | 82 | 23 | 1.000000 | 0.003224 | 0.003969 | -38.425843 | 0.020751 |

## 9. Fold3 forward-generalization

| pretrained_seed | MAE | RMSE | R2 | Spearman | selected_ridge_alpha |
|---|---|---|---|---|---|
| 42 | 0.001988 | 0.002328 | -12.567631 | 0.234190 | 1.000000 |
| 2024 | 0.001452 | 0.001897 | -8.004142 | 0.686759 | 1.000000 |
| 3407 | 0.003224 | 0.003969 | -38.425843 | 0.020751 | 1.000000 |

## 10. Seed stability

| fold | MAE_mean | MAE_std | R2_mean | R2_std | Spearman_mean | Spearman_std | worst_seed_MAE | best_seed_MAE | MAE_coefficient_of_variation | worst_seed_over_best_seed_MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| fold_1 | 0.004507 | 0.000078 | -12.459559 | 0.324145 | 0.155678 | 0.347542 | 0.004617 | 0.004445 | 0.017329 | 1.038858 |
| fold_2 | 0.002251 | 0.000565 | -79.470176 | 34.592496 | -0.078755 | 0.317236 | 0.002957 | 0.001574 | 0.250974 | 1.878845 |
| fold_3 | 0.002221 | 0.000742 | -19.665872 | 13.395490 | 0.313900 | 0.277677 | 0.003224 | 0.001452 | 0.334224 | 2.221347 |

## 11. Comparison with Age and LastVoltage

| fold | Age_MAE | Age_RMSE | Age_R2 | LastVoltage_MAE | LastVoltage_RMSE | LastVoltage_R2 |
|---|---|---|---|---|---|---|
| fold_1 | 0.002469 | 0.002581 | -2.497263 | 0.007350 | 0.007668 | -29.875056 |
| fold_2 | 0.006077 | 0.006084 | -351.464928 | 0.004843 | 0.005094 | -246.148111 |
| fold_3 | 0.005099 | 0.005166 | -65.809074 | 0.004856 | 0.005321 | -69.881161 |

## 12. What DeltaV pretraining contributed

The formal comparison is only against these chronology-safe same-target baselines. A positive result may be described only as: **DeltaV-pretrained TimesNet representations contain transferable information associated with the P1S-B2 calibrated health proxy.** It is not independent validation of true SOH.

## 13. Remaining limitations

The labels are sparse calibrated proxies, not independently validated true SOH. This phase reads only the 3007-row Development prefix of the causal feature table and never constructs post-1819 h labels, uses 2000/2300 targets, or trains RUL.

## 14. Decision for the next research phase

**PRETRAINED_TIMESNET_HEALTH_REPRESENTATION_NOT_YET_SUPPORTED**. Gates A–E: PASS, PASS, PASS, FAIL, FAIL. If unsupported, the most likely diagnosis is `SEED_REPRESENTATION_INSTABILITY` and the next step requires a separate research decision; this phase does not begin multitask training automatically.
