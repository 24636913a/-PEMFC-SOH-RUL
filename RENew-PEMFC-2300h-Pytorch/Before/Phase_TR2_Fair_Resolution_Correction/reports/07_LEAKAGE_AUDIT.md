# Leakage audit

- Development-only source tables are read; no Test table/file is opened.
- Common origins require complete history and target inside each Fold.
- Scalers and models fit only Fold Train common origins.
- A/B directly predict parent targets. C uses fixed equal output weighting.
- Future raw counts construct observed truth only, never a prediction.
- No calibration, threshold tuning, search, future real condition, or deep model is used.
