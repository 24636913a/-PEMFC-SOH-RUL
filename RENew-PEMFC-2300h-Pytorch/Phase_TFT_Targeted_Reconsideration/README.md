# Phase TFT Targeted Reconsideration

## Scope and terminal status

This is a pre-registered, Development-only targeted reconsideration of a strict-history Temporal Fusion Transformer (TFT) for the frozen RealHyFC dynamic-voltage task. It is not a general backbone search, SOH study, RUL study, or mechanism-training phase.

**Terminal status: `TFT_NOT_VIABLE`.** Vanilla strict-history TFT with the native LSTM local recurrent path and seed 42 obtained a three-fold formal forecast-query RMSE of **19.914181 mV**. This exceeds the frozen Raw TimesNet seed-42 1.20x bound of **17.047869 mV**. The automatic stopping rule therefore prohibited the additional LSTM seeds and every sLSTM/mLSTM experiment. No xLSTM candidate has a performance result.

The historical project status remains unchanged: `TIMESNET_FREEZE`, `BACKBONE_SEARCH_CLOSED`, `EMD_SPECIFIC_ADVANCE`, and Phase HA `FORECAST_ONLY_REPRESENTATION — HEALTH_EVIDENCE_NOT_IDENTIFIABLE`.

## Frozen task

- Development-only prefix: `pandas.read_csv(..., nrows=3007)`, then `bin_end_ophrs2_h < 1819.997738` (3005 rows).
- Raw F2, 14 channels; 0.5 h resolution; 48-step / 24 h history; 24-step / 12 h horizon.
- Identical Phase BS three expanding folds, legal origins, target rows, fit-only scaler, inner stopper, and outer refit.
- Strict unknown-future TFT: public model input is `[B, 48, 14]`; decoder tokens are internally fixed zeros.
- The model emits standardized *absolute* future voltage `[B, 24]`. Training and scoring use standardized DeltaV after subtracting the scaled origin voltage.
- No Test row, Test target, Test result, Test polarization, Test EIS, age/cycle feature, SOH/RUL target, decomposition feature, or future actual covariate was read.

## Files and reproducibility

`configs/PREREGISTERED_PROTOCOL.yaml` and its SHA256 were frozen before the seed-42 score. `SOURCE_MANIFEST.yaml` records frozen implementation and source revisions. The official xLSTM source checkout used only for architecture/state-handoff engineering checks is deliberately not versioned as an embedded repository; the exact tag, commit and relevant file hashes are recorded in that manifest. See [`sources/README.md`](sources/README.md).

For a fresh authorized reproduction before any score exists:

```powershell
python scripts/run_preflight.py
python scripts/run_tft_reconsideration.py --stage vanilla_seed42
```

The current saved run had reached all three LSTM seed-42 folds before a NumPy-boolean JSON serialization error prevented only its gate/report metadata from being written. `scripts/finalize_stopped_route.py` is a deterministic, artifact-only recovery tool: it reads the saved Development metrics plus frozen historical reference tables, creates the missing gate/decision/report files, and never reads N1, Test, predictions outside this phase, or trains a model. Its exact action is recorded in `outputs/RUN_METADATA_RECOVERY.json`.

## Terminal handoff

No further forecasting backbone may be added. The next research decision, if approved separately, is restricted to a RealHyFC-specific mechanism on TimesNet or an independently specified latent-state carrier; its fixed interface is in [`docs/NEXT_PHASE_REALHYFC_MECHANISM_CONTRACT.md`](docs/NEXT_PHASE_REALHYFC_MECHANISM_CONTRACT.md). It must not start automatically from this phase.
