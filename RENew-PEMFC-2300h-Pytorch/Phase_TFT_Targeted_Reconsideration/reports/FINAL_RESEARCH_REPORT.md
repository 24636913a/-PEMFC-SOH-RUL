# Phase TFT Targeted Reconsideration — Final research report

## Final status

**TFT_NOT_VIABLE**

The strict-history Vanilla TFT seed=42 formal forecast-query RMSE is **19.914181 mV**, above the frozen Raw TimesNet seed=42 1.20× limit of **17.047869 mV** (reference: 14.206558 mV). It therefore passes neither the viable nor the borderline gate. Per the frozen stop rule, LSTM seeds 2024/3407 and all sLSTM/mLSTM runs were not run.

## Saved Vanilla seed=42 metrics

| model | seed | formal_RMSE_mV | MAE_mV | R2 | endpoint_RMSE_mV | P95_mV | P99_mV | max_abs_error_mV | target_time_equal_RMSE_mV | Fold3_RMSE_mV | Fold3_target_time_equal_RMSE_mV | parameter_count | peak_gpu_memory_MiB | total_training_time_s | inference_latency_ms_per_origin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM | 42 | 19.914181241498685 | 9.914857392555495 | 0.989303039564727 | 38.14920035779273 | 27.371313034867246 | 45.69382004129392 | 525.9970985352993 | 29.524472835856084 | 19.394529946532725 | 35.388033610805124 | 87887 | 90.2998046875 | 220.15494249999756 | 0.028435284436588132 |

## Frozen Raw TimesNet seed=42 reference

| model | seed | formal_RMSE_mV | MAE_mV | R2 | endpoint_RMSE_mV | P95_mV | P99_mV | max_abs_error_mV | target_time_equal_RMSE_mV | Fold3_RMSE_mV | Fold3_target_time_equal_RMSE_mV | parameter_count | peak_gpu_memory_MiB | total_training_time_s | inference_latency_ms_per_origin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TimesNet | 42 | 14.206557830029526 | 3.950361165550529 | 0.9926288830361146 | 27.639400271483677 | 9.02716279185067 | 48.422392761252375 | 542.1562939882278 | 25.717520777604836 | 26.67992583211793 | 37.312938737663295 | 149046 | 82.86962890625 | 77.52953269999125 | nan |

## Gate evaluation

```json
{
  "vanilla_seed42_status": "TFT_NOT_VIABLE",
  "vanilla_seed42_checks": {
    "formal_110": false,
    "fold3_115": true,
    "p99_110": true,
    "endpoint_115": false,
    "target_equal_110": false
  },
  "borderline_checks": {
    "formal_between_110_120": false,
    "fold3_110": true,
    "p99_105": true,
    "endpoint_110": false,
    "target_equal_105": false
  },
  "raw_timesnet_seed42_formal_RMSE_mV": 14.206557830029526,
  "raw_timesnet_seed42_formal_110_limit_mV": 15.62721361303248,
  "raw_timesnet_seed42_formal_120_limit_mV": 17.04786939603543,
  "vanilla_seed42_formal_RMSE_mV": 19.914181241498685,
  "final_status": "TFT_NOT_VIABLE",
  "test_accessed": false,
  "model_training_performed": true
}
```

## Interpretation boundary

This is a Development-only dynamic voltage forecasting decision. It does not change the historical TimesNet Raw F2 incumbent, TimesNet+EMD `EMD_SPECIFIC_ADVANCE`, `TIMESNET_FREEZE`, `BACKBONE_SEARCH_CLOSED`, or the Phase HA health-representation result. No Test row, target, prediction, performance metric, polarization data or EIS was read. No EMD/VMD/wavelet, age/cycle feature, SOH target, RUL target, fast/slow state, or future actual covariate was used.

## Terminal handoff

This phase closes the targeted TFT route. The next decision must be at the RealHyFC-specific mechanism level, not a new backbone search. No mechanism training is started here.

## Reproducibility and metadata-recovery record

The three Vanilla TFT seed-42 fold runs, their checkpoints, prediction arrays, epoch logs and `PER_FOLD_METRICS.csv` were saved before the original runner encountered a NumPy-boolean JSON-serialization failure while writing only gate/report metadata. No numerical result, checkpoint, prediction, configuration or historical phase artifact was altered to recover from that failure. The deterministic `scripts/finalize_stopped_route.py` read only the saved phase metrics and the two frozen historical reference tables, regenerated the missing metadata, and recorded its scope in `outputs/RUN_METADATA_RECOVERY.json`. It read no N1 data, no Test data, and performed no model training.

The frozen protocol SHA is `c665ff984951cfc473732bb6aba4b0163f451123d98f1b158ab841a61409bb0d`. Engineering/source provenance, data-firewall evidence, strict output semantics, resource preflight, and source fingerprints are retained in `SOURCE_MANIFEST.yaml`, `DATA_FIREWALL_AUDIT.json`, `OUTPUT_SEMANTICS_AUDIT.json`, and `outputs/`.

## Literature boundary and next mechanism contract

TFT has previously been used in PEMFC work, including a 2025 *Renewable Energy* paper that modifies its local LSTM encoder-decoder with BiLSTM; mLSTM also has a direct PEMFC RUL publication (DOI [`10.1016/j.ijhydene.2025.02.302`](https://doi.org/10.1016/j.ijhydene.2025.02.302)). Accordingly, this phase makes no first-use claim for TFT, LSTM replacement, xLSTM or mLSTM. Details and primary links are in `reports/LITERATURE_AND_PRIOR_WORK.md`.

The next authorized decision is limited to a RealHyFC-specific, preregistered mechanism involving repeated duty cycles, repeated operation points, causal cycle phase, legally available operating conditions, fast/slow state separation, and sparse polarization anchors. It cannot be another general backbone search. The interface and leakage constraints are frozen in `docs/NEXT_PHASE_REALHYFC_MECHANISM_CONTRACT.md`; this phase does not start that work.
