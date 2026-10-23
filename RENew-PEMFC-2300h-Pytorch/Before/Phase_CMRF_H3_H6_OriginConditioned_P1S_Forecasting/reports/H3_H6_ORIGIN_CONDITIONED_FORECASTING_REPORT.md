# Phase CMRF — H3/H6 origin-conditioned P1S-retention forecasting

## Registered question and decision

H6 is the preregistered **primary** horizon and H3 is a secondary descriptive horizon.  Both horizons are defined by the raw physical identity `target_raw_cycle_id = origin_raw_cycle_id + H`; the target is never substituted with the H-th later P1S observation.  Every raw cycle from the origin through target was required to exist in the frozen Development-only raw-cycle catalog and to be marked complete.

**H6 decision: MEDIUM_HORIZON_RETENTION_PREDICTABILITY_NOT_ESTABLISHED**

**H3 secondary status: H3_MIXED**

The target is `DeltaR_H = R_target - R_origin`, where raw causal P1S retention is `R = V_P1S / 0.71383023635`.  Predictions are reconstructed as `R_origin + DeltaR_hat_H`.  No B2-calibrated health proxy, global calibration, smoothed health trajectory, EMD/VMD feature, target timestamp, actual target lead, future condition, target support label, operating age, or target P1S measurement was an input.

## Development-only data boundary

The mixed causal feature source was read exactly through `pd.read_csv(..., nrows=3007)` and filtered by `bin_end_ophrs2_h < 1819.997738`, yielding 3005 Development rows.  Its legally accessible object fingerprint is `51547e1f7747ac20e73e413a9ac22d93f831ba3b58e60b5ab0e1f6e399b5f7bf`; it is **not** a full physical-file hash.  The pre-existing raw-cycle catalog has maximum endpoint 1811.41 h and was checked as a Development-only artifact before it was used.  No Test raw region, Test label, or late-life target was opened.

## Frozen sample authority

| horizon | scope | n | detail |
|---|---|---|---|
| H3 | legal_unique_physical_horizon_candidates | 104 | target raw cycle equals origin raw cycle plus H; all intervening raw cycles complete |
| H3 | excluded_origin_candidates | 27 | physical continuity, causal target eligibility, or legal-origin-window exclusion |
| H3 | frozen_fold_expanded_authority_rows | 248 | chronological train/validation rows |
| H3 | fold_1_train | 48 | target P1S interval fully inside frozen role range |
| H3 | fold_1_validation | 18 | target P1S interval fully inside frozen role range |
| H3 | fold_2_train | 67 | target P1S interval fully inside frozen role range |
| H3 | fold_2_validation | 12 | target P1S interval fully inside frozen role range |
| H3 | fold_3_train | 79 | target P1S interval fully inside frozen role range |
| H3 | fold_3_validation | 24 | target P1S interval fully inside frozen role range |
| H6 | legal_unique_physical_horizon_candidates | 80 | target raw cycle equals origin raw cycle plus H; all intervening raw cycles complete |
| H6 | excluded_origin_candidates | 51 | physical continuity, causal target eligibility, or legal-origin-window exclusion |
| H6 | frozen_fold_expanded_authority_rows | 191 | chronological train/validation rows |
| H6 | fold_1_train | 33 | target P1S interval fully inside frozen role range |
| H6 | fold_1_validation | 18 | target P1S interval fully inside frozen role range |
| H6 | fold_2_train | 52 | target P1S interval fully inside frozen role range |
| H6 | fold_2_validation | 9 | target P1S interval fully inside frozen role range |
| H6 | fold_3_train | 61 | target P1S interval fully inside frozen role range |
| H6 | fold_3_validation | 18 | target P1S interval fully inside frozen role range |

## Baselines and model

Persistence predicts `DeltaR=0`.  The previous-H-cycle drift baseline is evaluated only where the required past causal P1S value exists.  Ridge receives only `R_origin` and the frozen mean/std/last/linear-slope summaries for each of the 14 legal 48-bin origin-history variables.  Its alpha is selected on the chronological final 15% of each outer training set.

TimesNet retains the frozen official body and backbone configuration.  The only task head is the registered temporal mean pooling of the 32-dimensional latent concatenated with the train-standardized observed origin retention, followed by `Linear(33,1)` for standardized `DeltaR`.  H3 and H6 were trained independently.  There is no architecture search or EMD follow-up in this phase.

### H6 primary metrics

| fold | model | scope | n_validation | MAE_mV | RMSE_mV | R2_retention | Spearman_retention |
|---|---|---|---|---|---|---|---|
| fold_1 | PERSISTENCE | all_validation_samples | 18 | 1.05101 | 1.22144 | -0.273948 | 0.73581 |
| fold_1 | PREVIOUS_H_CYCLE_DRIFT | matched_previous_H_cycle_causal_P1S_available | 17 | 1.74136 | 1.89797 | -2.04067 | 0.642157 |
| fold_1 | RIDGE_MATCHED_CONTEXT | all_validation_samples | 18 | 1.30463 | 1.61613 | -1.23029 | 0.696594 |
| fold_2 | PERSISTENCE | all_validation_samples | 9 | 0.382692 | 0.413242 | -10.014 | 0.95 |
| fold_2 | PREVIOUS_H_CYCLE_DRIFT | matched_previous_H_cycle_causal_P1S_available | 7 | 1.38744 | 1.7045 | -162.168 | 0.857143 |
| fold_2 | RIDGE_MATCHED_CONTEXT | all_validation_samples | 9 | 0.888612 | 1.06027 | -71.5062 | 0.833333 |
| fold_3 | PERSISTENCE | all_validation_samples | 18 | 0.378606 | 0.529658 | -1.02681 | 0.325077 |
| fold_3 | PREVIOUS_H_CYCLE_DRIFT | matched_previous_H_cycle_causal_P1S_available | 17 | 0.631179 | 0.783093 | -3.64088 | 0.291667 |
| fold_3 | RIDGE_MATCHED_CONTEXT | all_validation_samples | 18 | 0.497556 | 0.592688 | -1.53789 | 0.570691 |

| fold | seed | n_train | n_validation | selected_epoch | MAE_mV | RMSE_mV | R2_retention | Spearman_retention |
|---|---|---|---|---|---|---|---|---|
| fold_1 | 42 | 33 | 18 | 52 | 0.71236 | 0.966648 | 0.20211 | 0.622291 |
| fold_1 | 2024 | 33 | 18 | 21 | 0.688218 | 0.874912 | 0.346366 | 0.770898 |
| fold_1 | 3407 | 33 | 18 | 108 | 0.709402 | 0.901788 | 0.305591 | 0.700722 |
| fold_2 | 42 | 52 | 9 | 19 | 1.095 | 1.19808 | -91.5787 | 0.833333 |
| fold_2 | 2024 | 52 | 9 | 27 | 1.09294 | 1.28865 | -106.105 | 0.833333 |
| fold_2 | 3407 | 52 | 9 | 19 | 0.879297 | 0.990939 | -62.3333 | 0.8 |
| fold_3 | 42 | 61 | 18 | 1 | 0.41325 | 0.468915 | -0.58858 | 0.345717 |
| fold_3 | 2024 | 61 | 18 | 1 | 0.399521 | 0.458772 | -0.5206 | 0.329205 |
| fold_3 | 3407 | 61 | 18 | 2 | 0.462389 | 0.513757 | -0.906941 | 0.273478 |

### H3 secondary metrics

| fold | model | scope | n_validation | MAE_mV | RMSE_mV | R2_retention | Spearman_retention |
|---|---|---|---|---|---|---|---|
| fold_1 | PERSISTENCE | all_validation_samples | 18 | 0.620547 | 0.721207 | 0.555854 | 0.919505 |
| fold_1 | PREVIOUS_H_CYCLE_DRIFT | matched_previous_H_cycle_causal_P1S_available | 18 | 0.717324 | 0.909087 | 0.294305 | 0.851393 |
| fold_1 | RIDGE_MATCHED_CONTEXT | all_validation_samples | 18 | 1.41544 | 1.62708 | -1.26059 | 0.203302 |
| fold_2 | PERSISTENCE | all_validation_samples | 12 | 0.208058 | 0.259136 | -1.34199 | 0.832168 |
| fold_2 | PREVIOUS_H_CYCLE_DRIFT | matched_previous_H_cycle_causal_P1S_available | 10 | 0.430677 | 0.816781 | -22.9816 | 0.248485 |
| fold_2 | RIDGE_MATCHED_CONTEXT | all_validation_samples | 12 | 0.478081 | 0.545761 | -9.38808 | 0.804196 |
| fold_3 | PERSISTENCE | all_validation_samples | 24 | 0.270291 | 0.358611 | 0.165796 | 0.758261 |
| fold_3 | PREVIOUS_H_CYCLE_DRIFT | matched_previous_H_cycle_causal_P1S_available | 22 | 0.407271 | 0.595074 | -1.18935 | 0.749294 |
| fold_3 | RIDGE_MATCHED_CONTEXT | all_validation_samples | 24 | 0.311908 | 0.41584 | -0.121697 | 0.789565 |

| fold | seed | n_train | n_validation | selected_epoch | MAE_mV | RMSE_mV | R2_retention | Spearman_retention |
|---|---|---|---|---|---|---|---|---|
| fold_1 | 42 | 48 | 18 | 17 | 0.445977 | 0.640588 | 0.649601 | 0.750258 |
| fold_1 | 2024 | 48 | 18 | 22 | 0.442474 | 0.58466 | 0.708114 | 0.872033 |
| fold_1 | 3407 | 48 | 18 | 19 | 0.361717 | 0.526368 | 0.763416 | 0.832817 |
| fold_2 | 42 | 67 | 12 | 97 | 0.684197 | 0.881579 | -26.1051 | 0.713287 |
| fold_2 | 2024 | 67 | 12 | 124 | 0.694305 | 0.900677 | -27.2923 | 0.769231 |
| fold_2 | 3407 | 67 | 12 | 50 | 0.605876 | 0.709119 | -16.5375 | 0.755245 |
| fold_3 | 42 | 79 | 24 | 1 | 0.26077 | 0.312605 | 0.366106 | 0.757391 |
| fold_3 | 2024 | 79 | 24 | 1 | 0.247374 | 0.314474 | 0.358504 | 0.767826 |
| fold_3 | 3407 | 79 | 24 | 1 | 0.266779 | 0.323068 | 0.322964 | 0.74 |

## H6 gates

* Gate A (at least 5% below Persistence): **FAIL**
* Gate B (at least 3% below Ridge on matched context): **PASS**
* Gate C (Fold 3 within 105% of Persistence): **FAIL**
* Gate D (at least two seeds beat Persistence): **FAIL**

H6 equal-fold mean MAE is 0.7169 mV equivalent, versus 0.6041 for Persistence and 0.8969 for Ridge.  EMD follow-up authorization under the registered rule is **False**; no follow-up was executed here.

## Interpretation boundary and stop

This study evaluates a causal raw P1S-retention proxy, not independently validated SOH or RUL.  H1 is not retrained or reinterpreted: it used the next physical cycle; this phase uses fixed H3/H6 physical horizons.  Post-hoc support strata only describe the frozen sample authority and never control sample inclusion, alpha selection, seed selection, or gate outcomes.  The phase stops after these two registered horizons; it does not automatically create an EMD, RUL, calibration, Test, or new-backbone experiment.

## Source provenance

| path | role | sha256 | development_prefix_fingerprint_sha256 |
|---|---|---|---|
| RENew-PEMFC-2300h-Pytorch/Phase_CTAE_P1S_Reference_Availability_Erratum/outputs/P1S_REFERENCE_AVAILABILITY_AUDIT.csv | causal_raw_retention_target_and_support | aaa90129b08bc5c4638a3b7b77da4e84da9650599f42a7d735010ab364872ace | nan |
| RENew-PEMFC-2300h-Pytorch/Phase_PHI_P1S_B2_Health_Index_Consolidation/outputs/P1S_B2_FULL_HEALTH_TRAJECTORY.csv | Development_P1S_to_raw_cycle_identity_only | fbc9c2f0f9831d8da63426b33c241730d00cce409b7e1fc37b9bcfaa93ebc32a | nan |
| RENew-PEMFC-2300h-Pytorch/Phase_RCR_Raw_Resolution_Cycle_Reconstruction_Audit/outputs/RAW_CYCLE_CATALOG.csv | Development_only_raw_cycle_completeness_catalog | 8d1a8b9e17b179fbc99776f4006a61e5758006bd73e1a55bb96ffed38fa0f3d8 | nan |
| RENew-PEMFC-2300h-Pytorch/Phase_CPRF_H1_Causal_P1S_Retention_Forecasting/outputs/P1S_RAW_MEASUREMENT_INTERVAL_RECONSTRUCTION.csv | frozen_matched_PAA_P1S_measurement_envelopes | 239e6f3c6033687998a86386c5d9e92ea9535efe0d539351aef89d6b34d8cb0b | nan |
| RENew-PEMFC-2300h-Pytorch/Phase_BS_Final_Backbone_Screening/configs/PREREGISTERED_PROTOCOL.yaml | frozen_TimesNet_architecture_and_training_authority | fd710c4f2c1036c9a4b49068bd38d16aa39ad9f4bbc94a41b30f59f37afd6788 | nan |
| RENew-PEMFC-2300h-Pytorch/Phase_CMRF_H3_H6_OriginConditioned_P1S_Forecasting/configs/CMRF_H3_H6_FROZEN_PROTOCOL.yaml | CMRF_frozen_protocol | 6d8731857156e6cd03b9fd0134481df0c2b13ebe48d56a4ec88957986a042203 | nan |
| RENew-PEMFC-2300h-Pytorch/Phase_N1_Data_Foundation/outputs/causal_feature_tables/causal_features_30min.csv | mixed_causal_feature_source_prefix_only | nan | 51547e1f7747ac20e73e413a9ac22d93f831ba3b58e60b5ab0e1f6e399b5f7bf |
