# Phase N2.8E results

## Frozen protocol and Test firewall

All results use the frozen development-only 0.5 h / 48-step / 24-step / F2 / DeltaV protocol and three chronological folds. Formal training used expanding chronological OOF Ridge future-condition pseudo-inputs; validation used Ridge models fitted only on each fold's training prefix. The final Test target rows read, Test predictions, Test metrics, and Test feature diagnostics are all **0/false**. Oracle and shuffled future inputs are controls only and are excluded from all formal choices.

## Model comparison

| model | eligibility | mean_r2_delta | mean_rmse_delta_mV | mean_mae_delta_mV | fold_3_rmse_delta_mV | gain_vs_original_pct | mean_p95_mV | mean_p99_mV |
|---|---|---|---|---|---|---|---|---|
| TimeXer-Original | FROZEN_REFERENCE | 0.965271 | 36.170404 | 18.967869 | 36.961407 | 0.000000 | 53.684415 | 163.132027 |
| SA_TimeXer_Zero | FORMAL_ZERO_CONTROL | 0.967217 | 35.233349 | 17.622813 | 35.013059 | 2.590669 | 49.529289 | 175.981873 |
| SA_TimeXer_Ridge | FORMAL | 0.983700 | 24.719478 | 12.195526 | 27.953134 | 31.658277 | 34.849289 | 124.467991 |
| SA_TimeXer_Oracle | DIAGNOSTIC_ORACLE_ONLY | 0.999404 | 4.752673 | 3.605541 | 5.072206 | 86.860327 | 9.654920 | 14.377853 |
| SA_TimeXer_Shuffled | NEGATIVE_CONTROL | 0.762547 | 94.839537 | 69.772752 | 94.335114 | -162.202037 | 184.985831 | 320.869075 |
| FS_TimeXer_NoPhysics | FORMAL | 0.983322 | 25.032803 | 12.748292 | 28.226295 | 30.792030 | 35.344054 | 124.313453 |
| PGFS_TimeXer_Smooth | FORMAL | 0.983100 | 25.190868 | 12.137286 | 28.200835 | 30.355027 | 34.178879 | 125.322544 |
| PGFS_TimeXer_SmoothCurrent | FORMAL | 0.983078 | 25.203296 | 12.143819 | 28.287306 | 30.320669 | 33.923133 | 125.139860 |
| PGFS_TimeXer_Full | FORMAL | 0.983450 | 24.922028 | 12.271756 | 28.299489 | 31.098286 | 34.418596 | 124.557222 |

## Q1–Q5: contribution of the proposed components

1. **Step alignment contributes.** Formal SA-TimeXer-Ridge lowers mean RMSE from 36.170404 to 24.719478 mV (31.658%); its Fold-3 RMSE is 27.953134 mV (24.372% gain).
2. **Fast–Slow does not improve the formal best.** FS-TimeXer is 25.032803 mV, slightly worse than SA-TimeXer-Ridge.
3. **Smoothness is REJECT.** Adding smoothness changes FS RMSE from 25.032803 to 25.190868 mV without Fold-3 benefit.
4. **Current-response is KEEP as a physics regularizer, not as a new best predictor.** Negative fast-current gradients rise from 95.9840% without `L_current` to 97.5086% with it, while RMSE changes by only 0.049%.
5. **Monotonicity is KEEP as a weak prior.** Full PGFS improves PGFS RMSE from 25.203296 to 24.922028 mV and raises the slow non-increasing fraction from 49.5027% to 50.1764%; it still does not beat SA-TimeXer-Ridge.

## Q6–Q12: ablation, variance, and physical diagnostics

| model | mean_r2_delta | mean_rmse_delta_mV | fold_3_rmse_delta_mV | mean_p95_mV | mean_p99_mV |
|---|---|---|---|---|---|
| FS_TimeXer_NoPhysics | 0.983322 | 25.032803 | 28.226295 | 35.344054 | 124.313453 |
| PGFS_TimeXer_Smooth | 0.983100 | 25.190868 | 28.200835 | 34.178879 | 125.322544 |
| PGFS_TimeXer_SmoothCurrent | 0.983078 | 25.203296 | 28.287306 | 33.923133 | 125.139860 |
| PGFS_TimeXer_Full | 0.983450 | 24.922028 | 28.299489 | 34.418596 | 124.557222 |

The Full PGFS slow component accounts for 4.0591% of fast-plus-slow dynamic variance on average (range 1.8715%–6.0570%); it is neither slow-dominated (>80%) nor fast-only-near-zero-slow (<0.1%). It should be called a **learned slow degradation representation**, not SOH.

The current-response summary is: FS negative-gradient fraction 95.6400%; Smooth 95.9840%; SmoothCurrent 97.5086%; Full 97.5504%. Full median `dV_fast/dJ` is -0.245558 in the model's scaled-coordinate derivative units, with P05=-0.463743 and P95=-0.061823. The physical item with the clearest direct consistency effect is **current-response**; the item to delete from this frozen first-pass architecture is **smoothness**.

## Q13–Q22: predictive detail and thresholds

- Best formal model: **SA_TimeXer_Ridge**.
- Q13 R2 DeltaV = **0.983700**.
- Q14 RMSE DeltaV = **24.719478 mV**; MAE = **12.195526 mV**.
- Q15 Fold-3 RMSE = **27.953134 mV**.
- Q16 P95/P99 = **34.588635/131.580967 mV**.
- Q17 mean 9.5 h RMSE = **27.027679 mV**. This is below N2.8D RidgeFuture's 37.66 mV, but above the 18 mV aspiration and the 13.68 mV original-TimeXer reference.
- Q18 mean 12 h RMSE = **33.132808 mV**, improving over the original-TimeXer 43.992350 mV reference.
- Q19 below 25 mV: **yes**. Q20 below 20 mV: **no**. Q21 below 15 mV: **no**. Q22 near 12 mV: **no**.

## Q23–Q25: stability and electrochemical next phase

Q23 three-seed promotion: **no**. The best PGFS variant is Full at 24.922028 mV, so it met neither `<20 mV` nor `<22 mV + Fold-3 >=30%` promotion rule. Q24 three-seed stability is therefore **not applicable, not estimated**; it was not replaced by an unapproved run. Q25 EIS/Polarization sparse-anchor phase: **NO-GO**, because the best formal RMSE is not below 20 mV. No EIS interpolation, polarization interpolation, dense SOH, RUL, or N2.8F training is performed.

## Final conclusion

N2.8E is an execution **PASS**. Formal TimeXer status is **MEANINGFUL_RECOVERY**: step-aligned forecastable conditions materially recover performance, but the best formal model remains above the <20 mV gate. The phase stops here under the frozen protocol.
