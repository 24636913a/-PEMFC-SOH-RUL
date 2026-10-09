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

## Reproduction order

```powershell
python scripts\run_stage0_preflight.py
python -m unittest discover -s tests -v
python scripts\run_t0_baseline.py
python scripts\generate_t0_reports.py
```

The completed Seed-42 Development baseline is recorded in
`outputs/T0_FINAL_SUMMARY.json` and
`reports/FINAL_STAGE_T0_DECISION.md`.  The reported status is
`CANONICAL_TFT_BASELINE_ESTABLISHED`; it establishes a canonical architecture
control and is not a new backbone-selection result.
