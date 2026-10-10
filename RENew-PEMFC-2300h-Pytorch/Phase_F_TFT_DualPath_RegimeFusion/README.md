# Phase F — TFT Dual-Path Encoder and Regime-Aware Fusion

## Status

`PHASE_F_ARCHITECTURE_DESIGN_COMPLETED`

This directory freezes an implementation-ready, **Development-only** design for the planned F0–F2 ablation. It records an audit of the installed `pytorch-forecasting` TFT implementation and does not contain a trainable Phase-F model, a trained checkpoint, a prediction cache, or an Assessment result.

## Frozen sequence

| Stage | Encoder/fusion | Purpose |
|---|---|---|
| F0 | Native `CanonicalStrictHistoryTFT` | Same-stage baseline, trained afresh |
| F1 | Native TFT LSTM encoder plus causal multi-scale convolution; fixed 1:1 mean fusion | Test the additional encoder path |
| F2 | Same F1 paths; history-regime scalar fusion gate | Test only condition-aware fusion |

The shared task is RealHyFC Development-only, 0.5 h resolution, Raw F2 `[B,48,14]`, 24-step standardized `DeltaV` output, Spearman Top10 zero mask, and causal EWMA(alpha=0.3) on historical `cell_voltage_mean` only. Formal Test access is prohibited.

## Design records

- `research/F_TFT_ARCHITECTURE_AUDIT.md` — installed-source audit.
- `research/F1_RECOMMENDED_ARCHITECTURE.md` — implementation contract for the next phase.
- `research/F2_REGIME_AWARE_FUSION_DESIGN.md` — the sole F2 delta from F1.
- `research/F0_F1_F2_FAIRNESS_AND_RISK_AUDIT.md` — frozen comparison and determinism plan.
- `configs/F0_F2_PREREGISTRATION_DRAFT.yaml` — machine-readable design freeze, not an execution authorization.

## Explicit exclusions

No training, Optuna, Test reading, backbone search, feature-count search, EWMA tuning, SOH, or RUL work is authorized by this phase. Historical phases and the installed third-party package remain unmodified.
