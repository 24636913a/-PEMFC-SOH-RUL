# Phase PI final research report (posthoc correction)

## Frozen scope and result

Model: `PDFNetInspired_Exploratory`
Claim: **PDFNet-inspired exploratory architecture**
Formal backbone: **TIMESNET_FREEZE** (unchanged)
Backbone search: **CLOSED**

Phase PI evaluated one preregistered **D=128** PDFNet-inspired configuration. Stage 0 passed on an RTX 3070 Laptop GPU with `selected_microbatch=64` and effective batch 64; no `D=128 -> D=64` resource fallback occurred. This is not an official PDFNet implementation or author reproduction.

## Corrected fairness audit

`FAIRNESS_AUDIT.status = PASS` is now derived from named checks rather than hard-coded. TimesNet cache identity checks all passed for every fold × seed: origin row, target row, truth, anchor, target end, and horizon. The window protocol, Test firewall, future-information firewall, and external-decomposition firewall also passed. The registered 3007-row prefix yielded 3005 Development rows; two boundary rows did not enter legal windows, scalers, fitting, prediction, or scoring. Test targets, predictions, and metrics were not accessed.

## Decision

Seed-42 status: `PDFNET_INSPIRED_NO_PROMISE`.
Multi-seed status: `NOT_RUN` (`seed42_no_promise`).

The preregistered PDFNet-inspired exploratory architecture did not show sufficient potential under the frozen RealHyFC protocol.

Therefore seeds 2024 and 3407 were not run. `TIMESNET_FREEZE` and `BACKBONE_SEARCH_CLOSED` remain unchanged. This result does not mean that official PDFNet, GPRU, periodic recurrent networks, or dynamic graphs failed on RealHyFC.

## Frozen seed-42 summaries (mV)

| model | seed | formal_RMSE_mV | MAE_mV | R2 | endpoint_RMSE_mV | P95_mV | P99_mV | max_abs_error_mV | target_time_equal_RMSE_mV | Fold3_RMSE_mV | Fold3_target_time_equal_RMSE_mV | parameter_count | peak_gpu_memory_MiB | total_inner_plus_refit_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PDFNetInspired_Exploratory | 42 | 32.2135 | 18.5149 | 0.9725 | 40.4920 | 53.2126 | 121.3166 | 510.3724 | 36.8308 | 34.0525 | 42.4597 | 278174 | 2486.3960 | 31358.4247 |

## Frozen TimesNet controls (mV)

| model | seed | formal_RMSE_mV | MAE_mV | R2 | endpoint_RMSE_mV | P95_mV | P99_mV | max_abs_error_mV | target_time_equal_RMSE_mV | Fold3_RMSE_mV | Fold3_target_time_equal_RMSE_mV | parameter_count | peak_gpu_memory_MiB | total_inner_plus_refit_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TimesNet | 42 | 14.2066 | 3.9504 | 0.9926 | 27.6394 | 9.0272 | 48.4224 | 542.1563 | 25.7175 | 26.6799 | 37.3129 | 149046 | N/A (not measured in Phase PI) | N/A (not measured in Phase PI) |
| TimesNet | 2024 | 13.6233 | 3.8247 | 0.9929 | 29.2654 | 8.6573 | 45.8780 | 570.9616 | 25.0332 | 26.6678 | 38.6684 | 149046 | N/A (not measured in Phase PI) | N/A (not measured in Phase PI) |
| TimesNet | 3407 | 13.7505 | 4.4992 | 0.9935 | 26.6769 | 11.0268 | 43.8434 | 512.0992 | 24.8388 | 24.2217 | 35.8443 | 149046 | N/A (not measured in Phase PI) | N/A (not measured in Phase PI) |

`peak_gpu_memory_MiB` and `total_inner_plus_refit_time_s` for TimesNet are displayed as **N/A** because Phase PI did not measure them. The historical performance CSV is preserved unchanged.

## Structure activation diagnostic (non-causal)

- `fold_1`: more-than-one-inner-patch update fraction = `0.05305`.
- `fold_2`: more-than-one-inner-patch update fraction = `0.03947`.
- `fold_3`: more-than-one-inner-patch update fraction = `0.13126`.

In this frozen D=128 implementation, most outer-period segments contained one inner patch, so continuous cross-patch GPRU recurrent updates were activated only for a limited fraction of recorded patch updates. This is a structural activation diagnostic only; it does not establish that this behavior caused the lower predictive performance and it is not a component-level causal attribution.

## Closure

No retraining, rescoring, Test access, architecture change, D/k/graph/fusion/loss/optimizer change, VMD/EMD decomposition, backbone promotion, or new backbone screening was performed for this correction. The next research boundary is a separately preregistered leakage-safe decomposition and degradation-feature study; it is not executed here.
