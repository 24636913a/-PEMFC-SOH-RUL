# Phase PI final research report

## Frozen scope and result

Model: `PDFNetInspired_Exploratory`  
Claim: **PDFNet-inspired exploratory architecture**  
Formal backbone: **TIMESNET_FREEZE** (unchanged)

The paper source gate failure in Phase BG remains unchanged: this phase is not an official PDFNet implementation or reproduction. It evaluated exactly one pre-registered inspired architecture, using only Development windows and a voltage-channel-only standardized DeltaV loss.

## Stage 0 and fairness

Stage 0 synthetic structure audit passed. The RTX 3070 Laptop GPU resource gate passed at D=128 and effective batch 64. The registered 3007-row prefix was filtered to 3005 Development rows; the two boundary rows did not enter legal windows, scalers, fitting, prediction, or scoring. Test targets, predictions and metrics: **not accessed**.

## Decision

Seed-42 status: `PDFNET_INSPIRED_NO_PROMISE`.  
Multi-seed status: `NOT_RUN`.

The result supports only this statement: the preregistered PDFNet-inspired exploratory architecture showed sufficient potential for a future source/clarification decision.

No component-level causal claim is made for rFFT, GPRU, dynamic graph, or fusion. No PDFNet official-performance claim is made.

## Seed summaries (mV)

| model | seed | formal_RMSE_mV | MAE_mV | R2 | endpoint_RMSE_mV | P95_mV | P99_mV | max_abs_error_mV | target_time_equal_RMSE_mV | Fold3_RMSE_mV | Fold3_target_time_equal_RMSE_mV | parameter_count | peak_gpu_memory_MiB | total_inner_plus_refit_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PDFNetInspired_Exploratory | 42 | 32.2135 | 18.5149 | 0.9725 | 40.4920 | 53.2126 | 121.3166 | 510.3724 | 36.8308 | 34.0525 | 42.4597 | 278174 | 2486.3960 | 31358.4247 |

## Frozen TimesNet controls (mV)

| model | seed | formal_RMSE_mV | MAE_mV | R2 | endpoint_RMSE_mV | P95_mV | P99_mV | max_abs_error_mV | target_time_equal_RMSE_mV | Fold3_RMSE_mV | Fold3_target_time_equal_RMSE_mV | parameter_count | peak_gpu_memory_MiB | total_inner_plus_refit_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TimesNet | 42 | 14.2066 | 3.9504 | 0.9926 | 27.6394 | 9.0272 | 48.4224 | 542.1563 | 25.7175 | 26.6799 | 37.3129 | 149046 | nan | 0.0000 |
| TimesNet | 2024 | 13.6233 | 3.8247 | 0.9929 | 29.2654 | 8.6573 | 45.8780 | 570.9616 | 25.0332 | 26.6678 | 38.6684 | 149046 | nan | 0.0000 |
| TimesNet | 3407 | 13.7505 | 4.4992 | 0.9935 | 26.6769 | 11.0268 | 43.8434 | 512.0992 | 24.8388 | 24.2217 | 35.8443 | 149046 | nan | 0.0000 |

## Stop rule

No architecture, D, k, graph, fusion, loss, decomposition, Test evaluation, backbone promotion, or high-fidelity reconstruction was triggered after this decision. `TIMESNET_FREEZE` remains the only formal backbone status.
