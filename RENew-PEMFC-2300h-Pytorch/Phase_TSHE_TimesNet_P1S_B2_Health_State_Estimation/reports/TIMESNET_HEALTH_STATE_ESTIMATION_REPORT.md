# TimesNet P1S-B2 health-state estimation

## 1. Research objective

This phase tests whether the frozen TimesNet encoder can map a causal 24 h Raw F2 operating-history window to the next available reference-condition health-state label. It does not reopen backbone selection.

## 2. P1S+B2 is supervision, not the final model

`P1S + frozen B2` is a **P1S-B2 calibrated SOH proxy** used solely as health supervision. It is not an absolute ground-truth SOH and it is not the final predictive model. **TimesNet is the main time-series model.**

## 3. SOH-proxy definition

For the PAA primary `middle_50` complete-cycle P1S measurement, the target is `SOH_P1S_B2 = 1.1988114938324186 × R_P1S − 0.20903624631182127`. The affine coefficients are read from the frozen AHC-TV development-model registry and were never refit here.

## 4. Leakage-safe event construction

The event audit contains 141 primary complete-cycle labels, of which **105** have a legal 48-bin history. For every legal event, `input_end_ophrs2_h < target_cycle_start_ophrs2_h` is true, so no target-cycle P1S dwell or target P1S measurement enters the input. The model receives only the frozen ordered 14 Raw F2 variables; it receives neither P1S values, the proxy target, cycle identifiers, future conditions, nor OpHrs2.

## 5. TimesNet architecture

The frozen official TimesNet DataEmbedding, FFT period discovery, TimesBlock, and encoder layer normalization are retained (`48 × 14`, `d_model=32`, `d_ff=64`, one layer, `top_k=2`, three kernels, dropout 0.1). The only task change is `encoded.mean(dim=1) → Linear(32, 1)`. There is no attention head, MLP stack, recurrent unit, new Transformer, or alternative backbone.

## 6. Expanding-fold protocol

The three frozen chronological Development folds were used. Input and target standardizers were fit only to the appropriate training population; epoch selection used the final chronological 15% inner stopper and each outer result used a same-seed reinitialization refit on the full outer train set. Training was standardized SOH MSE with fixed AdamW settings and seeds 42, 2024, and 3407.

## 7. Simple baselines

Age-only is a train-only affine regression on OpHrs2 and is a comparator only; operating age is prohibited from the TimesNet input. Last-voltage-only is a train-only affine regression on the final historical `cell_voltage_mean` in each window.

## 8. Development results

| seed | fold | n_train | n_validation | MAE | RMSE | R2 | Spearman |
|---|---|---|---|---|---|---|---|
| 42 | fold_1 | 56 | 13 | 0.002556 | 0.002745 | -3.274885 | 0.576923 |
| 42 | fold_2 | 69 | 13 | 0.002612 | 0.003124 | -83.827944 | 0.802198 |
| 42 | fold_3 | 82 | 23 | 0.000883 | 0.001093 | -1.731376 | 0.376482 |
| 2024 | fold_1 | 56 | 13 | 0.008599 | 0.008703 | -41.982394 | -0.461538 |
| 2024 | fold_2 | 69 | 13 | 0.001279 | 0.001759 | -25.909082 | 0.714286 |
| 2024 | fold_3 | 82 | 23 | 0.001391 | 0.001631 | -5.080335 | 0.153162 |
| 3407 | fold_1 | 56 | 13 | 0.007676 | 0.007794 | -33.476366 | -0.873626 |
| 3407 | fold_2 | 69 | 13 | 0.001623 | 0.001861 | -29.119456 | 0.521978 |
| 3407 | fold_3 | 82 | 23 | 0.001398 | 0.001654 | -5.249114 | 0.302372 |

Baseline results:

| fold | Age_MAE | Age_RMSE | Age_R2 | LastVoltage_MAE | LastVoltage_RMSE | LastVoltage_R2 |
|---|---|---|---|---|---|---|
| fold_1 | 0.002375 | 0.002483 | -2.497307 | 0.007070 | 0.007376 | -29.874871 |
| fold_2 | 0.006360 | 0.006367 | -351.473961 | 0.005069 | 0.005332 | -246.155261 |
| fold_3 | 0.005337 | 0.005407 | -65.809364 | 0.005083 | 0.005569 | -69.881321 |

## 9. Seed stability

| seed | MAE | RMSE | R2 | Spearman |
|---|---|---|---|---|
| 42 | 0.002017 | 0.002321 | -29.611402 | 0.585201 |
| 2024 | 0.003756 | 0.004031 | -24.323937 | 0.135303 |
| 3407 | 0.003566 | 0.003770 | -22.614979 | -0.016426 |

The reported TimesNet three-fold aggregate is the unweighted mean across all nine seed-fold results; no best seed was selected.

## 10. Late-life temporal holdout

Status: **NOT_IDENTIFIABLE_NO_LEGAL_COMPLETE_P1S_EVENTS**. The frozen P1S primary complete-cycle table ends at 1712.79000 h. Consequently, it contains no eligible complete-cycle P1S-B2 proxy labels in the predeclared 1819.997738–2256.8 h post-hoc interval. No late-life error metric was invented, and no 2000/2300 polarization marker was used for model selection or tuning.

## 11. Does temporal information improve health estimation?

Gate A compares TimesNet with the age-only comparator; Gate B compares it with last-voltage-only. Passing Gate A supports information beyond an age trend, and passing Gate B supports information beyond merely the final historical voltage. The gates are frozen before examining the late-life interval.

## 12. Limitations

There are only sparse P1S-derived labels, the target is a calibrated proxy rather than direct ground truth, and late-life complete-cycle labels are absent from the frozen table. The post-hoc holdout is therefore not identifiable in this phase. These limitations do not authorize target interpolation or post-hoc label construction.

## 13. RUL task connection

The intended line is `SOH estimation → multi-step SOH trajectory forecasting → threshold-crossing RUL`. Official EOL, 90% EOL, and 95% EOL are unobserved; true RUL labels are unavailable and the dataset is right-censored. No direct RUL model was trained.

## 14. Final decision

**TIMESNET_HEALTH_STATE_ESTIMATION_NOT_YET_SUPPORTED**. Gates: A=PASS, B=PASS, C=FAIL, D=PASS. The recommended next decision is `TIMESNET_MULTITASK_HEALTH_LEARNING`. This phase stops here.

## Provenance

All source artifacts were read-only. Source SHA256 values: `{"causal_features_30min.csv": "fbee6b28682766081c9cda08059e2eaabcd6bb25365ce0bc5eb7c5714fda0f38", "P1S_COMPLETE_CYCLE_MEASUREMENTS.csv": "f962d5f4807348c3089aa2c6a6203b874c7e19b37199f07c703bcd9651138e74", "RAW_CYCLE_CATALOG.csv": "8d1a8b9e17b179fbc99776f4006a61e5758006bd73e1a55bb96ffed38fa0f3d8", "FROZEN_DEVELOPMENT_MODELS.yaml": "c5f43ecf2edbd88c983e37b563f2a58352c8704895f4d4bac143487f3dd8d5ff", "PREREGISTERED_PROTOCOL.yaml": "fd710c4f2c1036c9a4b49068bd38d16aa39ad9f4bbc94a41b30f59f37afd6788"}`.
