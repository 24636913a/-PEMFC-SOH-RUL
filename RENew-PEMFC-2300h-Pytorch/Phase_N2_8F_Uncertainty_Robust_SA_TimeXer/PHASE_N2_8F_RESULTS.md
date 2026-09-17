# Phase N2.8F results

## Causal OOF repair and firewall

The historical N2.8E `24.719478 mV` Ridge result is reference-only because its OOF implementation did not explicitly ensure that fit future-exogenous labels had completed before the next prediction origin. This phase rebuilt OOF Ridge and confidence labels with the stricter target-end rule. The repaired formal M0 is **24.538245 mV**. Test feature rows, targets, predictions, metrics, and diagnostic access are all **0/false**.

## Formal ablation (seed 42; unweighted mean across the three chronological folds)

| model | eligibility | mean_r2_delta | mean_rmse_delta_mV | mean_mae_delta_mV | fold_3_rmse_delta_mV | mean_p95_mV | mean_p99_mV |
|---|---|---|---|---|---|---|---|
| M0_SA_TimeXer_Ridge | FORMAL | 0.983989 | 24.538245 | 12.392522 | 27.885315 | 34.201042 | 127.225600 |
| M1_SA_TimeXer_FEA | FORMAL | 0.979714 | 27.626663 | 13.844508 | 31.335656 | 38.690879 | 147.964624 |
| M2_SA_TimeXer_CAF | FORMAL | 0.983417 | 25.010170 | 12.642585 | 27.879000 | 35.494945 | 130.530706 |
| M3_SA_TimeXer_FEA_CAF | NOT_RUN_NO_COMPONENT_GAIN |  |  |  |  |  |  |

M3 was not run: `NOT_RUN_NO_COMPONENT_GAIN`.

## Same-checkpoint and condition diagnostics

| variant | rmse_delta_mV | mae_delta_mV | p95_mV | p99_mV |
|---|---|---|---|---|
| M2_CAF_same_checkpoint_constant_gate_DIAGNOSTIC | 24.976699 | 12.620231 | 35.499973 | 130.586680 |
| OO_N2_8E_oracle_checkpoint_true_z_DIAGNOSTIC | 4.752673 | 3.605541 | 9.654920 | 14.377853 |
| RO_M0_checkpoint_true_z_DIAGNOSTIC | 14.620532 | 9.666516 | 25.961100 | 51.011181 |
| RR_M0_checkpoint_Ridge_z | 24.538245 | 12.392522 | 34.201042 | 127.225599 |

Oracle and true-future substitutions are diagnostic-only. They cannot select the formal model.

## Confidence, tails, horizons, and promotion

Confidence estimator: **VALID**.  At 9.5 h, the selected formal model has RMSE **26.074390 mV** (relative to M0: +0.00%); at 12 h it has **33.527980 mV**. Tail statistics are pooled over prediction points across the three development folds:

| model | aggregation | p95_mV | p99_mV | max_mV |
|---|---|---|---|---|
| M0_SA_TimeXer_Ridge | pooled_prediction_points_across_three_folds | 33.425689 | 129.651590 | 368.024498 |
| M1_SA_TimeXer_FEA | pooled_prediction_points_across_three_folds | 38.562711 | 148.256058 | 384.578273 |
| M2_SA_TimeXer_CAF | pooled_prediction_points_across_three_folds | 35.330512 | 130.392152 | 348.107338 |

| model | promotion_status | n_seeds | seeds | mean_rmse_delta_mV | rmse_std_mV |
|---|---|---|---|---|---|
| M0_SA_TimeXer_Ridge | not_promoted_seed42_only | 1 | 42 | 24.538245 | 0.000000 |

## Decision

- Best formal model: **M0_SA_TimeXer_Ridge**
- Fair-M0 improvement: **0.000%**
- Next-phase permission: **NO-GO**
- TimeXer status: **NO_FORMAL_GAIN**
- Further physics mechanism: **not justified by this phase**

All conclusions are limited to frozen development folds. This phase stops here and does not start N2.8G or evaluate Test.
