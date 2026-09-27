# Phase MW — Overlap Weighting Audit

Development-only audit of the already saved G-STOP P0/P2 predictions. It compares two fixed evaluation units without changing the formal status of the existing forecast-query metric:

- `forecast_query_weighted_RMSE`: every `(origin_row, horizon_step)` prediction is equally weighted.
- `target_time_equal_weight_RMSE`: squared errors are averaged within each `(fold, target_row)` first, then target times are equally weighted.

No model, raw data, Test artifact, random procedure, bootstrap, tuning, or training is used. Run from this folder with `python scripts/run_overlap_weighting_audit.py` and validate with `python -m unittest tests/test_overlap_weighting_audit.py`.
