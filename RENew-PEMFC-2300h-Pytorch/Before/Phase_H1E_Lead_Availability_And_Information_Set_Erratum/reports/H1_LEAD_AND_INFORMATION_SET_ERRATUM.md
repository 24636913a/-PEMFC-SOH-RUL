# H1 lead-availability and information-set erratum

## Scope and preserved H1 result

This is a read-only erratum. No H1 model, Ridge model, slope extrapolation, score, gate, sample, or Test artifact was recomputed. The committed H1 numbers remain: Persistence `0.1500665242063` mV-equivalent, historical Ridge `0.120436681402` mV-equivalent, and TimesNet `2.064150563242` mV-equivalent. Gates A–D remain FAIL.

`mV-equivalent = abs(retention error) × V_ref × 1000`; it is not the Phase BS dynamic-voltage RMSE.

## Lead availability

**ACTUAL_TARGET_END_LEAD_CAUSALLY_AVAILABLE = NOT_ESTABLISHED.**

The RCR artifact records a retrospective calibration template with loaded duration `9.52303` h and wall duration `11.8528` h. It was selected as `frozen calibration anchor ranking in preregistered protocol`, rather than as a prospective controller schedule fixing each future P1S end. The PAA implementation identifies P1 dwell windows from actual complete-cycle/state metadata, then derives measurement windows from those realized dwell durations. No existing Development-side source proved that the next actual P1S measurement end was known at the preceding P1S prediction origin.

This supports neither Case A nor a prospective nominal-schedule claim. A 9.523 h calibration-template duration is retained only as a retrospective nominal pattern—not as a `PROSPECTIVE_NOMINAL_LEAD_CANDIDATE` and not as a performance-recalculation input.

Across the unique fold-assigned H1 pairs, actual lead has N=121, mean=9.53539 h, median=9.52277 h, SD=0.0669974 h, min=9.52 h, P05=9.52 h, P95=9.53 h, max=10.0236 h, and IQR=0.00527 h. Thus actual lead is not a fixed constant in the frozen authority. H1 declares 122 legal candidates; 121 occur in the fold-assigned authority used here because one candidate's target P1S measurement interval crossed a frozen fold boundary and was excluded before fitting/evaluation.

| scope | N | mean_h | median_h | std_h | min_h | P05_h | P95_h | max_h | IQR_h |
|---|---|---|---|---|---|---|---|---|---|
| UNIQUE_FROZEN_AUTHORITY_H1_PAIRS | 121 | 9.53539 | 9.52277 | 0.0669974 | 9.52 | 9.52 | 9.53 | 10.0236 | 0.00527 |
| fold_1_ALL_ASSIGNED_ROLES | 78 | 9.54152 | 9.52301 | 0.0829144 | 9.52 | 9.52 | 9.56008 | 10.0236 | 0.00296 |
| fold_1_TRAIN | 60 | 9.5408 | 9.52301 | 0.0828668 | 9.52082 | 9.52109 | 9.53553 | 10.0236 | 0.0022 |
| fold_1_VALIDATION | 18 | 9.54391 | 9.52221 | 0.0854325 | 9.52 | 9.52 | 9.58337 | 9.8858 | 0.01 |
| fold_2_ALL_ASSIGNED_ROLES | 93 | 9.53873 | 9.52301 | 0.0761478 | 9.52 | 9.52 | 9.53 | 10.0236 | 0.00335 |
| fold_2_TRAIN | 79 | 9.54129 | 9.52301 | 0.082406 | 9.52 | 9.52 | 9.55005 | 10.0236 | 0.0029 |
| fold_2_VALIDATION | 14 | 9.52429 | 9.52 | 0.00513553 | 9.52 | 9.52 | 9.53 | 9.53 | 0.01 |
| fold_3_ALL_ASSIGNED_ROLES | 121 | 9.53539 | 9.52277 | 0.0669974 | 9.52 | 9.52 | 9.53 | 10.0236 | 0.00527 |
| fold_3_TRAIN | 93 | 9.53873 | 9.52301 | 0.0761478 | 9.52 | 9.52 | 9.53 | 10.0236 | 0.00335 |
| fold_3_VALIDATION | 28 | 9.52429 | 9.52 | 0.00503953 | 9.52 | 9.52 | 9.53 | 9.53 | 0.01 |

## Information sets and formal roles

| method | uses_R_origin | uses_past_retention_slope | uses_actual_target_end_lead | uses_Raw_F2_dynamic_history | strictly_available_at_prediction_origin | formal_role_after_erratum |
|---|---|---|---|---|---|---|
| PERSISTENCE | True | False | False | False | True | PRINCIPAL_STRICTLY_CAUSAL_H1_BASELINE |
| CAUSAL_LINEAR_96H | True | True | True | False | False | DESCRIPTIVE_ACTUAL_LEAD_LINEAR_BASELINE |
| RIDGE_CAUSAL_STATE | True | True | True | False | False | DESCRIPTIVE_ACTUAL_LEAD_RIDGE_BASELINE |
| TIMESNET_RAW_F2 | False | False | False | True | True | RAW_F2_ONLY_TIMESNET_H1_NOT_SUPPORTED |

Persistence uses `{R_origin}`; the observation is complete at its prediction origin, so it is the **PRINCIPAL_STRICTLY_CAUSAL_H1_BASELINE**. Linear and Ridge use the realized target-end lead; their historical values are retained but their formal roles are respectively `DESCRIPTIVE_ACTUAL_LEAD_LINEAR_BASELINE` and `DESCRIPTIVE_ACTUAL_LEAD_RIDGE_BASELINE`, not strictly causal deployment baselines.

TimesNet uses only `{48×14 Raw F2}`, while Persistence uses `{R_origin}` and Ridge uses `{R_origin, historical slopes, actual target-end lead}`. Hence `TIMESNET_VS_PERSISTENCE = DEVELOPMENT_RELEVANT_BUT_NOT_EQUAL_INFORMATION_SET_COMPARISON`: Persistence is a practically available opponent, but this is not a matched-information-set comparison.

The corrected conclusion is: **Under the frozen Raw-F2-only H1 protocol, TimesNet did not recover sufficient next-cycle P1S-retention information to outperform the directly observable origin-retention persistence baseline.** This does not establish that complex models cannot improve P1S forecasting under another preregistered information set.

## Status and stop

* `RAW_F2_ONLY_TIMESNET_H1_NOT_SUPPORTED`
* `PERSISTENCE_STRONG_CAUSAL_BASELINE`
* `AUTO_H3_H6 = PROHIBITED`

Recommended research direction: **STOP_H1_MODEL_DEVELOPMENT**. A Raw-F2-plus-origin-retention or longer-horizon experiment would be a distinct preregistered question, not an amendment or automatic continuation of H1. No Test was accessed and no model training was performed.
