# Final PAA Anchor Validity Qualification

## Scope and preserved result

This independent `POSTHOC_QUALIFICATION_AUDIT` read the existing PAA anchor
records and only the internal `DateTime` column of the six allowed Development
nominal-polarization CSVs.  It did not recalculate P1S, select intervals, alter
the PAA 48 h / 3-interval / 2-of-3 rule, or access Test polarization/EIS.

The frozen PAA result remains **`P1S_PREANCHOR_COMPATIBILITY_EXPLORATORY_PASS`** (`POSTHOC_EXPLORATORY`).
PAV final qualification is **`PAA_EXPLORATORY_PASS_TEMPORALLY_VERIFIED`**.

## Per-checkpoint temporal and median-source qualification

| checkpoint_h | polarization_first_datetime | PAA_gap_cutoff_datetime | preanchor_measurements_strictly_before_polarization | qualification_status | median_source_interval_id | median_source_joint_support |
| --- | --- | --- | --- | --- | --- | --- |
| 300 | 2025-02-09T00:33:22.005000 | 2025-02-09T00:17:49.005000 | YES | TEMPORAL_ORDER_VERIFIED | COMPLETE_024 | YES |
| 600 | 2025-02-24T01:36:43.005000 | 2025-02-24T01:10:00.005000 | YES | TEMPORAL_ORDER_VERIFIED | PARTIAL_PRE_600 | NO |
| 850 | 2025-03-20T02:02:04.005000 | 2025-03-20T01:32:27.005000 | YES | TEMPORAL_ORDER_VERIFIED | COMPLETE_074 | YES |
| 1150 | 2025-04-18T05:10:57.005000 | 2025-04-18T04:26:40.005000 | YES | TEMPORAL_ORDER_VERIFIED | COMPLETE_099 | NO |
| 1500 | 2025-05-26T22:30:13.005000 | 2025-05-26T21:45:55.005000 | YES | TEMPORAL_ORDER_VERIFIED | COMPLETE_125 | NO |
| 1750 | 2025-06-10T16:29:51.005000 | 2025-06-10T15:54:57.005000 | YES | TEMPORAL_ORDER_VERIFIED | COMPLETE_149 | YES |

Every row uses `POLARIZATION_CSV_INTERNAL_DATETIME`.  `OpHrs2 [h]` is absent
from all six polarization files, so this phase does not infer precise
characterization operating ages from their nominal filenames.  The recorded
PAA cutoff is only a frozen PAA gap boundary; the CSV timestamp is the actual
polarization-time source.

## Median-support interpretation

The registered PAA 2-of-3 support rule passed for all six anchors and is not
modified.  A `NO` value in `median_source_joint_support` does **not** invalidate
the original PAA gate: it qualifies the estimand by showing that the frozen
three-value median was supplied by an interval outside the complete-cycle joint
support.  No post-hoc median excluding that interval was computed.

## Claim boundary

See `docs/PAA_CLAIM_BOUNDARY.md`.  The public Zenodo/article records document
the long durability experiment and regular characterizations but not exact
times of individual characterizations; they were not used as timing evidence.

## Non-results

Neural training: **NO**.  New SOH: **NO**.  RUL: **NO**.  Test polarization:
**NO**.  Test EIS: **NO**.
