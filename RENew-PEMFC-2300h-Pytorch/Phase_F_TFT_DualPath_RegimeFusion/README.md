# Phase F — TFT Dual-Path Encoder and Regime-Aware Fusion

## Status

`PHASE_F_ARCHITECTURE_DESIGN_COMPLETED`

Engineering implementation status: `PHASE_F_IMPLEMENTATION_READY_FOR_TRAINING_REVIEW`. This is not a training authorization.

This directory freezes an implementation-ready, **Development-only** design for the planned F0–F2 ablation. It records an audit of the installed `pytorch-forecasting` TFT implementation and does not contain a trainable Phase-F model, a trained checkpoint, a prediction cache, or an Assessment result.

## Frozen sequence

| Stage | Encoder/fusion | Purpose |
|---|---|---|
| F0 | Native `CanonicalStrictHistoryTFT` | Same-stage baseline, trained afresh |
| F1 | Native TFT LSTM encoder plus causal multi-scale convolution; fixed 1:1 mean fusion | Test the additional encoder path |
| F2 | Same F1 paths; history-regime scalar fusion gate | Test only condition-aware fusion |

The shared task is RealHyFC Development-only, 0.5 h resolution, Raw F2 `[B,48,14]`, 24-step standardized `DeltaV` output, Spearman Top10 zero mask, and causal EWMA(alpha=0.3) on historical `cell_voltage_mean` only. Formal Test access is prohibited.

## Interpretation firewall

`PHASE_F_ASSESSMENT_INHERITED_SELECTION_BIAS` applies: Top10+EWMA was frozen after Phase E's already exposed Development Assessment result (E2 Assessment RMSE 7.785821 mV); E2 Validation RMSE was 6.064596 mV, above E0 Top10 Raw's 4.797507 mV. Future F0–F2 results therefore remain Development exploratory even if no later Assessment-driven adjustment occurs.

`C1_D0_E0_REPRODUCIBILITY_UNRESOLVED` also remains open. Historical Top10-Raw metrics differ (11.445854, 13.169241, and 19.318765 mV) despite aligned legal keys/truths. No unverified cause is assigned.

`PHASE_F_TFT_ARCHITECTURE_EXPLORATION_ONLY` means the existing TimesNet freeze and backbone-search closure remain unchanged. This user-authorized TFT branch cannot automatically replace TimesNet or create a direct cross-protocol backbone ranking.

## Design records

- `research/F_TFT_ARCHITECTURE_AUDIT.md` — installed-source audit.
- `research/F1_RECOMMENDED_ARCHITECTURE.md` — implementation contract for the next phase.
- `research/F2_REGIME_AWARE_FUSION_DESIGN.md` — the sole F2 delta from F1.
- `research/F0_F1_F2_FAIRNESS_AND_RISK_AUDIT.md` — frozen comparison and determinism plan.
- `configs/F0_F2_PREREGISTRATION_DRAFT.yaml` — machine-readable design freeze, not an execution authorization.

## Explicit exclusions

No training, Optuna, Test reading, backbone search, feature-count search, EWMA tuning, SOH, or RUL work is authorized by this phase. Historical phases and the installed third-party package remain unmodified.
