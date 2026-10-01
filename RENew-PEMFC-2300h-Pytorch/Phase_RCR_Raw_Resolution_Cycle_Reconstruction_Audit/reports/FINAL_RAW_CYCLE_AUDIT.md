# Final raw-resolution cycle reconstruction audit

## Scope and firewall

This independent Phase RCR audit used the raw Loadcycle operating-state stream
only. It used no voltage field for cycle discovery, no polarization input, no
Test-side row for a decision, no neural training, no SOH construction, and no
RUL model. The strict analysis firewall was `OpHrs2 [h] < 1819.997738 h`.
The execution head was `3150a35ecf7ef44c570707815e9bf9d9dd38a8dc`; the required research
base `ca040e56a76e9c985ec199e67cbed1929ea285e4` is an ancestor. The frozen protocol SHA256 is
`df9e45dde37feea6d74b95caed91032fe1458847d3ba2357e24a097f20bd0405`.

## Frozen procedure

State runs were run-length encoded using `operation_point + rounded(I.S.set
[A], 3 decimals)`. Stops and zero-current runs remain encoded. The calibration
region is the first 100 Development operating hours. The anchor and template
were selected by frozen recurrence ranking alone; neither voltage nor health
information entered the selection.

## Result

**Final status: `RAW_CYCLE_STRUCTURE_QUALIFIED`**

Frozen template: **YES**.  Complete cycles assigned
within the three Development strata: **146**.

- `unique_calibration_template`: **True**
- `early_middle_late_complete_cycles`: **True**
- `operation_sequence_core_ge_90pct`: **True**
- `loaded_duration_median_8_11_h`: **True**
- `loaded_duration_cv_le_10pct`: **True**
- `key_positive_operation_point_support`: **True**
- `voltage_independent_boundary`: **True**

## Evidence A and Evidence B

`CYCLE_SEQUENCE_SIMILARITY.csv` reports set-current sequence recurrence
(Evidence A) separately from operation-point sequence recurrence (Evidence B).
Its exact-order operation-point LCS fraction is the frozen fixed-order core
criterion; the accompanying edit distance is descriptive only. Neither is DTW
or elastic alignment. `RAW_CYCLE_CATALOG.csv` separately records missing/extra
positive states, shortened/extended dwell states, and stop-run variation.

## Why the 0.5 h MQ route failed

The raw-to-30-minute mapping is documented in
`RAW_TO_30MIN_MAPPING_AUDIT.csv`. Across reconstructed intervals, the mixed-bin
fraction is **1.000** and the longest run under the already
frozen Phase-MQ eligible-bin rule is **7** bins. The registered interpretation is
`AGGREGATION_OR_SEGMENTATION_LOSS`. This statement concerns only the frozen MQ 0.5 h protocol and
does not change historical MQ files or infer a health result.

## Historical documentation debt

The known MQ A0 counter-label issue, absent Attempt-2 execution audit, and
substatus/phase-status mapping inconsistency remain
`NON_BLOCKING_DOCUMENTATION_DEBT`; Phase RCR did not repair or rerun them.

## Stopping point

No automatic Phase RHA is created, even if raw-cycle structure qualified.
This audit contains no voltage health trend, SOH, EOL, RUL, model metric, or
backbone decision.
