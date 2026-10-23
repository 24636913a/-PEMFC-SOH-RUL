# Frozen Data Split

Ageing coordinate: `OpHrs2 [h]`

Observed range: `72.788690–2256.800000 h`

- Train: `[72.788690, 1383.195476) h`
- Validation: `[1383.195476, 1819.997738) h`
- Test: `[1819.997738, 2256.800000] h`

Cut points are calculated from the observed ageing span: `min + 60% × span` and `min + 80% × span`. A bin is assigned only when its full physical interval is contained within one split; boundary-straddling bins are marked `boundary_excluded`.

The final 20% Test split is frozen. It may be checked for readability, range, bin/window counts and structural integrity only. It must not determine sampling resolution, active-cell mask, j*, SOH formula, features, lookback, horizon, preprocessing, model, loss or hyperparameters.

## Manifest

| split | ophrs2_start_h | ophrs2_end_h | left_closed | right_closed | cut60_h | cut80_h | characterization_checkpoints | number_of_30min_bins | number_of_valid_30min_bins | gap_affected_30min_bins | low_coverage_30min_bins | datetime_start | datetime_end | number_of_1h_bins | number_of_valid_1h_bins | gap_affected_1h_bins | low_coverage_1h_bins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 72.78869 | 1383.195476 | True | False | 1383.195476 | 1819.997738 | BoL|300h|600h|850h|1150h | 2248 | 2236 | 14 | 8 | 2025-01-27 12:24:06.005000 | 2025-05-22 17:21:20.005000 | 1127 | 1116 | 13 | 7 |
| validation | 1383.195476 | 1819.997738 | True | False | 1383.195476 | 1819.997738 | 1500h|1750h | 755 | 750 | 5 | 4 | 2025-05-22 17:51:21.005000 | 2025-06-29 13:04:55.005000 | 378 | 374 | 5 | 3 |
| test | 1819.997738 | 2256.8 | True | True | 1383.195476 | 1819.997738 | 2000h|2300h | 836 | 832 | 5 | 3 | 2025-06-29 13:58:41.005000 | 2025-07-26 11:48:14.005000 | 418 | 415 | 5 | 2 |
