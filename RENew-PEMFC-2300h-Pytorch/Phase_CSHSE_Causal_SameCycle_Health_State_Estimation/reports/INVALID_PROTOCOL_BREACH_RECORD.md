# Invalidated CSHSE execution record

## Status

`INVALID_PROTOCOL_BREACH`

## Cause

The prior CSHSE execution computed a whole-file SHA256 for `causal_features_30min.csv`. That physical file contains rows beyond the frozen Development prefix, so the integrity operation exceeded the permitted data-access boundary.

## Consequence

There is no evidence that Test labels were accessed (`TEST_LABEL_ACCESS_CONFIRMED = false`; `TEST_LABEL_ACCESS_EVIDENCE = none`). This is recorded as a **data-access boundary breach**, not as Test-label leakage. Nevertheless, no metric, prediction, figure, gate, summary, or report produced by that execution may be used for scientific conclusions.

The generated artifacts were isolated under `runtime_invalidated/` with `INVALID_PROTOCOL_BREACH` in their directory names.

## Repair and clean-rerun requirement

The whole-file causal hash is removed. The repaired runner reads only `nrows=3007`, filters `bin_end_ophrs2_h < 1819.997738` to exactly 3005 rows, and computes a deterministic `DEVELOPMENT_PREFIX_FINGERPRINT` on those legal rows only. Regression tests cover post-prefix invariance, Development-row sensitivity, and no-unbounded-read static enforcement. A completely fresh Python process must regenerate every formal CSHSE output after the repair gates pass.
