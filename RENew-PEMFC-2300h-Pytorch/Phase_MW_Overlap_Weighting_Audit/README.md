# Phase MW — Overlap Weighting Audit

Development-only audit of the already saved G-STOP P0/P2 predictions. It compares two fixed evaluation units without changing the formal status of the existing forecast-query metric:

- `forecast_query_weighted_RMSE`: every `(origin_row, horizon_step)` prediction is equally weighted.
- `target_time_equal_weight_RMSE`: squared errors are averaged within each `(fold, target_row)` first, then target times are equally weighted.

No model, raw data, Test artifact, random procedure, bootstrap, tuning, or training is used. Run from this folder with `python scripts/run_overlap_weighting_audit.py` and validate with `python -m unittest tests/test_overlap_weighting_audit.py`.

The final supplement freezes a key-level Fold 1 `multiplicity=1` audit and an exact-key cross-reference to the existing RD Fold 1 / 12 h cache. MW's error difference remains `P2−P0`; RD's separate `P2−P1` fields are reported only as a labelled cross-reference. Every saved NPZ `target_end_age_h` is checked to remain strictly below the Development/Test boundary. This is the terminal cache-diagnostic phase: it creates no successor phase and starts no model work.
