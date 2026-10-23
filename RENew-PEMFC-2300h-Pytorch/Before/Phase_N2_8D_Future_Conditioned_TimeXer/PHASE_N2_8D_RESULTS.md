# Phase N2.8D results

## Protocol and firewall
Development-only frozen 0.5 h / 48 / 24 / F2 / DeltaV task. Test targets read: 0. No Test prediction, metric, or diagnostic was built.

## Q1–Q6: FC-TimeXer and controls
| model | eligibility | mean_r2_delta | mean_rmse_delta_mV | mean_mae_delta_mV | fold_3_rmse_delta_mV | mean_p95_mV | mean_p99_mV | lifecycle_ratio | future_condition_gain_pct |
|---|---|---|---|---|---|---|---|---|---|
| FC_TimeXer_Oracle | DIAGNOSTIC_ORACLE_ONLY | 0.975178 | 30.501139 | 16.412547 | 28.030617 | 46.556968 | 128.522800 | 1.006100 | 15.673769 |
| FC_TimeXer_Ridge | FORMAL | 0.968651 | 34.433787 | 17.765440 | 34.052110 | 51.566505 | 151.778370 | 1.072707 | 4.801211 |
| FC_TimeXer_ShuffledFuture | NEGATIVE_CONTROL | 0.954534 | 41.348922 | 22.410044 | 43.077275 | 65.816937 | 203.135556 | 1.214186 | -14.317001 |
| FC_TimeXer_ZeroFuture | FORMAL | 0.967058 | 35.299199 | 18.121735 | 34.856045 | 49.224979 | 168.010027 | 1.069492 | 2.408615 |
| TimeXer-Original | reference_only | 0.965271 | 36.170404 | 18.967869 | 36.961407 | 53.684415 | 163.132027 | 1.167125 | 0.000000 |

Correct RidgeFuture vs ZeroFuture gain: 2.451646%. Correct RidgeFuture vs original gain: 4.801211%.

**Q1 conclusion:** RidgeFuture is better than original TimeXer by 4.801211% RMSE and better than ZeroFuture by 2.451646%, while ShuffledFuture is materially worse. This supports a real, but small, future-condition contribution. It does not constitute structural recovery because RidgeFuture remains 34.433787 mV. **Q2–Q5:** the table gives the exact ZeroFuture, RidgeFuture, OracleFuture, and ShuffledFuture values. OracleFuture also remains above 25 mV, so even the new explicit conditioning architecture does not establish a viable upper bound.

## Q7–Q12: Fold3, tails, and horizons
Fold3 RidgeFuture gain = 7.871173%. At 9.5 h: Original 13.680394 mV; RidgeFuture 37.658993 mV, a 23.978599 mV deterioration. At 12 h: Original 43.992350 mV; RidgeFuture 39.546545 mV, a 4.445805 mV improvement. The cross-attention structure does not preserve the 9.5 h periodic forecast while it retains a limited long-horizon benefit.

| model | p50_mV | p90_mV | p95_mV | p99_mV | max_mV |
|---|---|---|---|---|---|
| TimeXer-Original | 11.043340 | 38.423061 | 53.676418 | 178.973663 | 463.437378 |
| FC_TimeXer_ZeroFuture | 10.965496 | 34.695491 | 50.370999 | 179.732662 | 449.510545 |
| FC_TimeXer_Ridge | 10.417074 | 35.337374 | 51.547162 | 163.276887 | 466.404125 |
| FC_TimeXer_Oracle | 9.911180 | 32.807499 | 46.625093 | 125.150680 | 447.175607 |
| FC_TimeXer_ShuffledFuture | 13.659894 | 44.806287 | 66.361725 | 210.846359 | 448.694572 |

**Q10 conclusion:** RidgeFuture improves the pooled P99 from 178.973663 mV to 163.276887 mV, but ZeroFuture has a slightly lower P95. Thus tail improvement is partial rather than uniform.

## Q13: Attention
RidgeFuture mean diagonal attention = 0.050161; mean absolute argmax offset = 11.583333 future steps. A uniform 24-step map has diagonal mass 0.041667, so the observed diagonal concentration is weak and the large argmax offset does not support clean horizon-to-condition alignment. The full 24×24 mean attention map is saved as `future_condition_attention.npy` and visualized in `future_attention_heatmap.png`.

## Q14–Q22: Decision
Oracle–Ridge RMSE gap (Ridge minus Oracle) = 3.932648 mV. Even true future conditions leave Oracle RMSEDeltaV at 30.501139 mV. Status: **FAIL**. Three-seed promoted: **False**. Recommended next phase: **Stop TimeXer and retain Lifecycle-Aware Robust TimesNet**. PINN is recommended only if RidgeFuture RMSE is below 20 mV: **False**.

**Q15–Q18:** RidgeFuture does not meet <25 mV, <20 mV, <15 mV, or <=12 mV. **Q19–Q20:** no three-seed run was authorized; the seed-42-only record is retained in the multiseed table. **Q21–Q22:** final status is `FAIL`; a PINN stage is not justified by this experiment.

| model | status | n_seeds | seeds | mean_r2_delta | mean_rmse_delta_mV | rmse_std_mV |
|---|---|---|---|---|---|---|
| FC_TimeXer_Ridge | not_promoted_seed42_only | 1 | 42 | 0.968651 | 34.433787 | 0.000000 |
