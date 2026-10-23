# Condition-Normalized P1S Development Study

## 1. Research question

Can a partial condition model reduce P1S operating-condition sensitivity while retaining the frozen sparse Development polarization-health relationship?

## 2. Data boundary

`DEVELOPMENT_ONLY`; 2000/2300 polarization targets were not accessed. The consumed AHC-TV one-shot ledger was not read for execution or modified. No Stage T, Test AE/MAE, or Rules A/B/C calculation occurred.

## 3. Why hard support was insufficient

The preceding target-blind audit found the P1S state observable at late life but the frozen seven-dimensional hard box brittle to condition drift. This study retains all legal Development P1S intervals rather than deleting out-of-box rows.

## 4. Condition-normalization formulation

For each chronology-safe prefix, the partial model is `V_P1S = f(age) + G(z) + epsilon`, with `f(age)` a train-prefix cubic spline and `z` the seven frozen operating variables. Normalization is `V_CN = V_raw - [G(z) - G(z_ref)]`. Age is not supplied to the later `R -> Y` affine mapping.

## 5. OLS comparator

C1 uses the registered partial-linear OLS condition model and the same nonnegative affine anchor calibration as every other candidate.

## 6. Ridge primary model

C2 selects alpha by chronological inner validation P1S-voltage reconstruction MAE only. Final-Development selected alpha: `1e-06`.

## 7. Chronology-safe evaluation

Each anchor correction uses only intervals ending before that anchor's frozen characterization cutoff. Its fixed existing latest-three interval IDs are retained; no support-based replacement was made.

## 8. LOAO health-anchor results

| checkpoint | Y | R_raw | R_CN_OLS | R_CN_Ridge | AE_Raw_B2 | AE_CN_OLS | AE_CN_Ridge |
|---|---|---|---|---|---|---|---|
| 300 | 0.984639 | 0.995628 | 0.995926 | 0.995952 | 0.000321 | 0.000195 | 0.000067 |
| 600 | 0.979679 | 0.991591 | 0.991649 | 0.991550 | 0.000021 | 0.000634 | 0.000348 |
| 850 | 0.977740 | 0.990112 | 0.988771 | 0.988805 | 0.000221 | 0.000690 | 0.000863 |
| 1150 | 0.972623 | 0.985142 | 0.984248 | 0.984422 | 0.001021 | 0.000207 | 0.000383 |
| 1500 | 0.973863 | 0.986592 | 0.985576 | 0.986523 | 0.000214 | 0.000008 | 0.000922 |
| 1750 | 0.971785 | 0.985602 | 0.983881 | 0.983882 | 0.001063 | 0.000483 | 0.000007 |

Raw B2 LOAO MAE was reproduced at `0.000476869906820`. OLS / Ridge CN-P1S LOAO MAE: `0.000369474544419` / `0.000431749369804`.

## 9. Condition decoupling

| model | variable | Spearman_residual_vs_condition | Pearson_residual_vs_condition | absolute_Spearman |
|---|---|---|---|---|
| Raw | anode_inlet_temperature_C | 0.073072 | 0.139709 | 0.073072 |
| Raw | anode_inlet_RH_pct | -0.085116 | -0.143168 | 0.085116 |
| Raw | anode_stoichiometry | -0.095036 | -0.084290 | 0.095036 |
| Raw | cathode_stoichiometry | 0.013997 | -0.038282 | 0.013997 |
| CN_OLS | anode_inlet_temperature_C | 0.025882 | 0.000000 | 0.025882 |
| CN_OLS | anode_inlet_RH_pct | -0.049943 | -0.000000 | 0.049943 |
| CN_OLS | anode_stoichiometry | -0.013843 | 0.000000 | 0.013843 |
| CN_OLS | cathode_stoichiometry | 0.041193 | 0.000000 | 0.041193 |
| CN_Ridge | anode_inlet_temperature_C | 0.025882 | 0.000004 | 0.025882 |
| CN_Ridge | anode_inlet_RH_pct | -0.049943 | -0.000003 | 0.049943 |
| CN_Ridge | anode_stoichiometry | -0.013843 | -0.000003 | 0.013843 |
| CN_Ridge | cathode_stoichiometry | 0.041193 | -0.000002 | 0.041193 |

Residual correlations remove a low-frequency age spline before relating voltage to operating conditions.

## 10. Local scatter

| model | median_local_MAD_V | median_local_std_V |
|---|---|---|
| CN_OLS | 0.000081333 | 0.000147649 |
| CN_Ridge | 0.000081322 | 0.000147649 |
| Raw | 0.000105226 | 0.000152083 |

## 11. Trend preservation

| model | Spearman_retention_vs_OpHrs2 | linear_drift_slope_per_h |
|---|---|---|
| Raw | -0.923143 | -0.000008 |
| CN_OLS | -0.933539 | -0.000009 |
| CN_Ridge | -0.933539 | -0.000009 |

## 12. Comparison with Raw B2

The Raw comparator is the frozen six-anchor B2 authority. Its exact LOAO reproduction was verified before any CN comparison; no new raw baseline was searched.

## 13. Limitations

Only six sparse Development health anchors exist. The condition transform is exploratory and Development-frozen; it is neither Test-validated SOH nor RUL, and it has not been applied to 2000/2300.

## 14. Decision

Final status: `CONDITION_NORMALIZED_P1S_NO_CLEAR_DEVELOPMENT_GAIN`. Gates A/B/C/D: `PASS` / `PASS` / `FAIL` / `PASS`.

## 15. Next validation requirement

Any future target-blind transform-applicability audit or independent validation must be separately authorized. This study does not reopen, reinterpret, or rerun the consumed AHC-TV Test.
