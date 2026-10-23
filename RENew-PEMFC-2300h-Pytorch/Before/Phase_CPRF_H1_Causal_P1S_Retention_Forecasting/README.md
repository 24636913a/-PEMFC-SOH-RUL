# Phase CPRF — H1 causal P1S-retention forecasting

This Development-only phase forecasts the causal P1S raw retention at the next **contiguous** complete raw cycle.  H1 is not a flexible next-available-event task: an origin may predict only `raw_cycle_id + 1`.

The target is `R_P1S_causal`, the raw P1S voltage divided by the causal reference voltage frozen by Phase CTAE.  Global B2 calibration, smoothed trajectories, age, condition-support status, target P1S measurements, and future data are excluded from model fitting and selection.

The runner reconstructs the raw P1S measurement envelope only through PAA's frozen pre-1719.31 h source universe.  It reads the mixed causal feature CSV only as `nrows=3007`, then applies the frozen Development filter to obtain 3005 rows.  A deterministic Development-prefix fingerprint is the only integrity fingerprint for that mixed physical file.

Run from this directory after freezing the protocol hash:

```powershell
python scripts/run_cprf_h1.py
```

The run fail-closes before model training if raw P1 replay, H1 contiguity, target-measurement exclusion, fold assignment, prefix access, or sample-authority checks fail.
