# Phase N2.8A Plan — True Predictive-Value Audit and High-Fidelity Modern Model Screening

## Scope frozen before execution

- Working directory: `Phase_N2_8A_True_Value_and_Modern_Screening` only. Earlier phases are read-only sources.
- Frozen task: 0.5 h sampling, F2 exact 14 features, 24 h / 48-step history, 12 h / 24-step horizon, raw causal F2, raw `ΔV = V(t+h)-V(t)` target.
- Chronological development folds are inherited without change. Final Test `[1819.997738, 2256.800000] h` is isolated: no target rows, windows, predictions, feature diagnostics, rankings, or metrics.
- Existing LSTM, high-fidelity TimesNet, PatchTST, iTransformer, TimeMixer, TCN, Ridge, CycleRepeat, and Persistence are read from prior frozen outputs only; they are not retrained.

## Execution sequence

1. Create a fail-closed Development-only loader and recompute absolute- and delta-space metrics from cached prediction arrays. Publish the legacy audit and anchor-contribution comparison.
2. Run three negative controls: training-label permutation and temporal target misalignment for the frozen raw-protocol LSTM/TimesNet implementations, plus a static feature-generation audit. Halt if a label-permuted model has validation `R²_ΔV > 0.5`.
3. Vendor and pin the official TimeXer and TimeMixer++ sources in this phase directory. Record repository, commit, retained official forward path, adapters, and parameter counts in `OFFICIAL_IMPLEMENTATION_AUDIT.md`.
4. Train official-architecture adapters at seed 42 under the same chronological folds, train-only scaling, AdamW, MSE, and early stopping. The output head predicts only the 24-step `ΔV` sequence and never consumes future covariates.
5. Compare seed-42 candidates with high-fidelity TimesNet using `R²_ΔV`, `RMSE_ΔV`, `MAE_ΔV`, Fold3, tails, and lifecycle ratio. Apply the preregistered advancement and stop rules without manual tuning.
6. For an advanced candidate only, run seeds 42/2024/3407. Consolidate primary delta metrics, secondary absolute-voltage metrics, lifecycle stability, per-horizon performance, tails, paired evidence, and the frozen backbone decision.
7. Validate required outputs, artifact-tool CSV readability, figure integrity, and Test firewall. Stop without entering a lifecycle-improvement phase or opening Test.

## Implementation integrity rules

- TimeXer preserves official endogenous patch embedding, exogenous variate embedding, global endogenous token, cross-variable attention, encoder blocks, and forecasting head pathway. `cell_voltage_mean` is endogenous; the remaining 13 F2 variables are past-known exogenous context.
- TimeMixer++ preserves official multi-resolution image construction, decomposition / dual-axis interaction, multiscale mixing, and official prediction pathway. No old Phase N2 TimeMixer may be renamed or substituted.
- Necessary adapters are limited to F2 shape mapping (`enc_in=14`, 48 history steps, 24 output steps), the raw delta target, and disabling all future exogenous inputs.
- No SG/EMA/other filtering, OpHrs2, EIS, polarization, expanded features, Optuna, grid search, Bayesian optimization, or manual repeated tuning.

## Selection rules

- Primary: `R²_ΔV`, then `RMSE_ΔV`, Fold3 `R²_ΔV` / `RMSE_ΔV`, P95/P99 tails, seed stability, and parameter / runtime cost.
- New model advances from seed 42 with TimesNet comparison if it meets Rule A, B, or C from the user protocol. It stops if its mean `RMSE_ΔV > 1.20 ×` TimesNet and Fold3 has no improvement.
- The final model decision is made on Development data only. Test access remains false in the final YAML.
