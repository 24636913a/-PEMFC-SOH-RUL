# Phase T0 — Original TFT Baseline

This independent Development-only stage establishes a reproducible
`Canonical_Architecture_TFT_Strict_History` baseline for RealHyFC multistep
dynamic-voltage forecasting. It uses the mature `pytorch-forecasting` TFT body
with historical Raw F2 data only. The decoder receives only a deterministic
relative forecast-step coordinate; it receives no target voltage, future actual
operating condition, or fabricated static stack descriptor.

The stage is not a backbone search and does not revisit the historical
`StrictHistoryTFT` result. It does not use EMD, filters, Optuna, health labels,
SOH/RUL, or Test data.

The executed protocol is frozen in
`configs/PREREGISTERED_T0_PROTOCOL.yaml` before results are produced.
