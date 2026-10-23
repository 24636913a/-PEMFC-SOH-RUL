# Phase CPRF — H1 causal P1S-retention forecasting

## Decision

**COMPLEX_H1_MODEL_NOT_JUSTIFIED_OVER_SIMPLE_CAUSAL_BASELINES**

This formal Development-only H1 task forecasts the next *contiguous* raw complete-cycle target. Every authority row satisfies `target_raw_cycle_id = origin_raw_cycle_id + 1`; a future legal P1S cycle was never skipped to replace a missing adjacent target. The target is raw `R_P1S_causal = V_P1S / V_ref`, with `V_ref = 0.71383023635 V`; global B2 and all smoothed/trend health labels were excluded from fitting and model selection.

## Frozen sample authority and leakage controls

Protocol SHA-256: `e6c920dd5d2ab11b0246433706b270ed0ceeeccb9b252ab186531bf04d63a9df`.  The causal feature source was read only with `pd.read_csv(..., nrows=3007)`, then filtered to the 3005-row Development prefix. Its legal-object fingerprint is `51547e1f7747ac20e73e413a9ac22d93f831ba3b58e60b5ab0e1f6e399b5f7bf`; it is explicitly **not** a hash of the mixed physical source file.

PAA middle-50 membership was replayed for all 131 causal-target-eligible complete cycles through the pre-1719.31 h raw source universe. Each reconstructed voltage and dwell count matched the frozen PAA authority. The origin input is the latest legal 48 x 14 Raw F2 window ending at or before the origin's actual P1S measurement end. Every bin ends strictly before the target P1S measurement begins. No target P1S measurement, future covariate, operating age, B2 value, or target support status entered any model input.

| scope | n | detail |
|---|---|---|
| unique_H1_candidates | 122 | contiguous causal P1S origin-target pairs with legal Raw F2 history |
| excluded_candidate_origins | 9 | no next causal target or no legal 48-bin causal history |
| all_fold_expanded_authority_rows | 292 | frozen fold/role sample rows |
| fold_1_train | 60 | target interval entirely within frozen role bounds |
| fold_1_validation | 18 | target interval entirely within frozen role bounds |
| fold_2_train | 79 | target interval entirely within frozen role bounds |
| fold_2_validation | 14 | target interval entirely within frozen role bounds |
| fold_3_train | 93 | target interval entirely within frozen role bounds |
| fold_3_validation | 28 | target interval entirely within frozen role bounds |

## Baselines

`PERSISTENCE` is available on every frozen H1 sample. `CAUSAL_LINEAR_96H` is the predeclared physical slope baseline; 24 h and 48 h slope variants are sensitivity rows only. `RIDGE_CAUSAL_STATE` uses exactly `[R_origin, slope_24h, slope_48h, slope_96h, lead_to_target_end_h]`, with a train-only embedding scaler and chronological inner alpha choice. Any Ridge scope difference is preserved rather than hidden.

| fold | model | scope | n_validation | MAE_mV | RMSE_mV | R2 | Spearman |
|---|---|---|---|---|---|---|---|
| fold_1 | PERSISTENCE | all_valid_H1_samples | 18 | 0.23286 | 0.272518 | 0.936584 | 0.97936 |
| fold_1 | CAUSAL_LINEAR_24H | origin_24h_slope_available | 18 | 0.259697 | 0.385246 | 0.873269 | 0.9484 |
| fold_1 | CAUSAL_LINEAR_48H | origin_48h_slope_available | 18 | 0.208311 | 0.282485 | 0.931861 | 0.964912 |
| fold_1 | CAUSAL_LINEAR_96H | origin_96h_slope_available | 18 | 0.240483 | 0.287303 | 0.929517 | 0.960784 |
| fold_1 | RIDGE_CAUSAL_STATE | all_24_48_96h_origin_slopes_available | 18 | 0.193545 | 0.245721 | 0.948443 | 0.977296 |
| fold_2 | PERSISTENCE | all_valid_H1_samples | 14 | 0.101474 | 0.119222 | 0.780722 | 0.885714 |
| fold_2 | CAUSAL_LINEAR_24H | origin_24h_slope_available | 12 | 0.0837131 | 0.099556 | 0.654328 | 0.776224 |
| fold_2 | CAUSAL_LINEAR_48H | origin_48h_slope_available | 12 | 0.0942598 | 0.102782 | 0.63156 | 0.811189 |
| fold_2 | CAUSAL_LINEAR_96H | origin_96h_slope_available | 12 | 0.093856 | 0.103954 | 0.623114 | 0.762238 |
| fold_2 | RIDGE_CAUSAL_STATE | all_24_48_96h_origin_slopes_available | 12 | 0.0883204 | 0.110502 | 0.574133 | 0.818182 |
| fold_3 | PERSISTENCE | all_valid_H1_samples | 28 | 0.115866 | 0.153852 | 0.883076 | 0.926656 |
| fold_3 | CAUSAL_LINEAR_24H | origin_24h_slope_available | 24 | 0.0909055 | 0.121095 | 0.904878 | 0.918261 |

## TimesNet H1 model

The model retains the official TimesNet FFT period discovery, TimesBlock, 48-to-72 temporal projection, and frozen backbone-screening hyperparameters. Only the forecasting output contract was replaced by temporal mean pooling followed by a single `Linear(32,1)` scalar retention head. Each fold and seed uses chronological 85/15 inner stopping, then same-seed reinitialisation and full-outer-train refitting for the selected epoch. The network was trained normally; no checkpoint or encoder was treated as a frozen pretrained model.

| fold | seed | n_train | n_validation | selected_epoch | MAE_mV | RMSE_mV | R2 | Spearman | ridge_matched_MAE_mV |
|---|---|---|---|---|---|---|---|---|---|
| fold_1 | 42 | 60 | 18 | 54 | 2.66115 | 3.48716 | -9.38365 | 0.903262 | 2.66115 |
| fold_1 | 2024 | 60 | 18 | 18 | 3.17158 | 3.27133 | -8.13808 | 0.905332 | 3.17158 |
| fold_1 | 3407 | 60 | 18 | 17 | 2.9631 | 3.23335 | -7.92712 | 0.806004 | 2.9631 |
| fold_2 | 42 | 79 | 14 | 67 | 1.28599 | 1.77718 | -47.7242 | 0.389011 | 1.24866 |
| fold_2 | 2024 | 79 | 14 | 111 | 1.1559 | 1.59909 | -38.4484 | 0.375824 | 1.03689 |
| fold_2 | 3407 | 79 | 14 | 81 | 1.39042 | 1.82896 | -50.6049 | 0.279121 | 1.41597 |
| fold_3 | 42 | 93 | 28 | 5 | 2.55301 | 2.63042 | -33.1782 | 0.489802 | 2.50856 |
| fold_3 | 2024 | 93 | 28 | 79 | 1.08157 | 1.65835 | -12.5847 | 0.256537 | 0.883493 |
| fold_3 | 3407 | 93 | 28 | 3 | 2.31464 | 2.38403 | -27.0751 | 0.680904 | 2.23905 |

Per-seed three-fold equal summaries:

| seed | equal_fold_mean_MAE_mV | equal_fold_mean_RMSE_mV |
|---|---|---|
| 42 | 2.16672 | 2.63159 |
| 2024 | 1.80302 | 2.17626 |
| 3407 | 2.22272 | 2.48211 |

## Primary score and gates

The formal score is the arithmetic mean of fold MAE values in mV-equivalent retention error. Pooled results are descriptive only because folds are intentionally equally weighted. Gate B compares TimesNet and Ridge only on the explicit all-slope-available Ridge scope. Condition-support strata are post-hoc reporting groups, not sampling or selection criteria.

* Gate A (≥5% below Persistence): **FAIL**
* Gate B (≥3% below matched Ridge): **FAIL**
* Gate C (Fold 3 ≤105% Persistence): **FAIL**
* Gate D (at least two seeds strictly beat Persistence): **FAIL**

## Interpretation and stop

The fixed TimesNet configuration did not meet all preregistered H1 gates. Under the frozen protocol, this does not justify a more complex H1 model; the next methodological decision must precede any H3/H6 work.

This result concerns a causal proxy target and next-contiguous-cycle H1 prediction, not independently validated SOH or RUL. No Test target, Test raw region, 2000/2300 polarization target, future health-trajectory model, H3/H6 model, calibration model, or post-hoc architecture change was executed. The phase stops here.
