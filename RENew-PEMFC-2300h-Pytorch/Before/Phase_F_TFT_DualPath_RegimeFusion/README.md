# Phase F — TFT Dual-Path Encoder and Regime-Aware Fusion

## Status

`PHASE_F_ARCHITECTURE_DESIGN_COMPLETED`

Engineering implementation status: `PHASE_F_ENABLED_FORWARD_PARITY_PASS`.

`READY_FOR_F0_F1_F2_TRAINING_AUTHORIZATION_REVIEW` is not a training authorization.

This directory contains trainable F1/F2 model implementations and synthetic tests for the **Development-only** F0–F2 design. The explicitly authorized Seed42 F0/F1/F2 Development training has now completed. Best checkpoints, Validation and Assessment caches, independent metric audits and reports are saved. Formal Test remains prohibited. Synthetic backward/reload evidence is engineering testing, not a scientific performance result.

The synthetic suite contains 15 tests (13 original + 2 enabled full-forward tests). Disabled inheritance parity, enabled TEST-ONLY neutral-fusion parity, and F1/F2 initial full-forward parity are separately documented in `reports/F_IMPLEMENTATION_AUDIT.md` and `outputs/implementation_audit/F_ENABLED_FORWARD_PARITY_AUDIT.json`. Each enabled comparison covers two inputs/batch sizes in eval mode, all eight native output fields, exact shared-state alignment, and real branch/gate call evidence. Formal model code and architecture are unchanged.

To reproduce tests and regenerate audit reports from this directory:

```powershell
python -B scripts/audit_f_model_implementation.py
```

The generator records complete test logs and fails closed on test, parity, parameter-count or dependency-identity failure. This does not resolve historical score reproducibility or authorize training.

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

Only the explicitly frozen F0/F1/F2 Seed42 Development training was authorized by the subsequent user instruction. No additional training, Optuna, Test reading, backbone search, feature-count search, EWMA tuning, SOH, or RUL work is authorized. Historical phases and the installed third-party package remain unmodified.

## Completed formal execution

`PHASE_F_F0_F1_F2_DEVELOPMENT_EXPLORATION_COMPLETED`

`SINGLE_SEED_DEVELOPMENT_STRUCTURAL_GAIN_OBSERVED`

Run: `python -B scripts/run_f_training.py --run` (one-shot identity-guarded; completed runs cannot retrain). Independent cache verification: `python -B scripts/audit_f_predictions.py`. Results: `reports/F_FINAL_EXPERIMENT_SUMMARY.md`. Checkpoints: `outputs/checkpoints`. All three best checkpoints were frozen and reload-checked before unified Assessment. Test accessed: NO.

The structural-gain label is limited to F2 versus F1. F1 and F2 do not beat F0 on the primary Assessment RMSE, despite improved Validation. Overall superiority over native F0 is not established. See the explicit eight-question handoff in the final report; no model was automatically promoted.

The three ~1.45–1.49 MB `.pt` checkpoints are included in the Git delivery with file hashes and exact reload evidence. The local `.gitattributes` preserves frozen protocol bytes and new execution-source line endings across checkouts; it does not change any historical model or protocol.
