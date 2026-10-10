# C0–C3 Single-Split Feature/Filter Ablation

## Scope

This is a Seed42, chronological 7:1:2 **Development-only exploratory** ablation. Assessment is not an independent project Test and was never used to choose features, filtering, or hyperparameters. The mixed N1 CSV was read only with `nrows=3007`; the legal 3005-row Development prefix was filtered before all processing. No original Test row, target, or metric was accessed.

## Frozen setup

- Protocol SHA: `6853f32e41f838f0900f97e4a402bcb762ed36be3a851bfa1b67f61a2a7eeb8e`
- Train/Validation/Assessment rows: `[0,2103)`, `[2103,2404)`, `[2404,3005)`.
- Fold2 T1-R configuration was eligible because its historical training boundary (1164.794345 h) precedes the new Train end (1310.000000 h).
- Canonical TFT remained 14-input architecture; C1/C3 use a standardised-space zero mask, not structural channel removal.

## Selections fixed before assessment

- C1 Top10: cell_voltage_mean, cell_voltage_min, zero_current_fraction, load_transition_count, cell_voltage_max, actual_current_density_A_cm2_std, actual_current_A_std, actual_current_A_peak_to_peak, actual_current_density_A_cm2_peak_to_peak, actual_current_density_A_cm2_mean
- C2 filter selected solely by Validation RMSE: **EWMA_alpha_0_3**.

## Assessment metrics

|Group|RMSE mV|MAE mV|R²|Endpoint RMSE mV|P95 mV|P99 mV|Target-time-equal RMSE mV|Best epoch|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|C0|27.1118|7.5136|0.9808|22.9639|14.2931|27.7944|26.4892|232|
|C1|11.4459|4.2196|0.9966|5.8559|10.5630|17.7651|11.6482|350|
|C2|11.2976|6.3704|0.9967|10.0185|16.0204|26.5162|11.4326|239|
|C3|15.3845|5.0942|0.9938|7.0118|13.2950|21.2680|15.6388|428|

The lowest Assessment RMSE was **C2** (11.2976 mV), a 58.33% relative change versus C0. This is descriptive for this fixed Development segment only; it does not establish cross-stage generalisation or blind-Test benefit.

## Chronological error concentration

An independently recomputed chronological diagnostic split the Assessment target rows before row 2704; it was defined only for interpretation after all four prediction caches had been frozen. C0 mean absolute error increased from 5.5881 mV in the earlier part to 8.9499 mV in the later part. The corresponding early/later values were 4.1928/4.2396 mV for C1, 5.7016/6.8693 mV for C2, and 4.2499/5.7239 mV for C3. Thus, the C1 mask was the most stable by mean absolute error across this descriptive split, whereas C2 retained the lowest overall RMSE.

All four groups retained their extreme samples. Their five largest absolute errors were concentrated around target row 2952: 315.5 mV for C0, 308.6 mV for C1, 295.8 mV for C2, and 311.2 mV for C3. The filter/mask ablations therefore reduced average and/or aggregate error but did not remove this late-assessment high-error event. This is an exploratory error-concentration observation, not a basis for deleting windows, redefining the assessment split, or retuning the method.

## Integrity checks

- All assessment origins and all 24 target rows stayed inside `[2404,3005)`.
- Train/Validation/Assessment target sets were disjoint.
- All reported metrics were independently recomputed from persisted prediction CSVs.
- Only historical 48-step encoder inputs were used; no future actual covariates entered the decoder.
