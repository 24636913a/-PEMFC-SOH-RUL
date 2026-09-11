# Phase N2.8E execution plan

## Frozen task and data boundary

- Resolution: 0.5 h; history: 48 steps (24 h); horizon: 24 steps (12 h).
- Inputs: frozen F2 14-feature set. Target: future `DeltaV` from the known origin voltage.
- Evaluation: the three frozen expanding chronological development folds only.
- The final Test partition begins at 1819.997738 h. It may not provide feature, target, prediction, metric, diagnostic, selection, or tuning information.

## Future-condition protocol

- The nine future exogenous variables are the frozen load and transition variables only; voltage is excluded.
- Formal training uses strict expanding chronological Ridge OOF predictions for eligible training windows and a Ridge fit on the fold training set for validation windows.
- Oracle future exogenous inputs are diagnostic-only. Zero and validation-time shuffled future inputs are controls. No formal model receives true future operating conditions.

## Architecture and ablations

- Preserve the official TimeXer historical encoder/head as the high-fidelity backbone (commit `76011909357972bd55a27adba2e1be994d81b327`).
- Implement a step-aligned FiLM correction: each future row `z_h` produces only `gamma_h, beta_h` and controls forecast step `h`; no future-time cross-attention, pooling, or time-wide concatenation is permitted.
- Compare TimeXer-Original, SA-TimeXer Zero/Ridge/Oracle/Shuffled controls, FS-TimeXer without physics loss, and PGFS-TimeXer with the frozen Smooth, Smooth+Current, and Full losses.
- The slow component is learned only from historical latent state. It is not supervised with interpolated SOH or an invented slow label.

## Physics-loss and promotion rules

- `L_data` is DeltaV MSE. The frozen regularizers are `0.01 L_slow_smooth`, `0.001 L_current`, and `0.001 L_slow_mono`.
- If initial raw loss magnitudes differ by more than 100× from data loss, one detached first-batch scale normalization is permitted and logged. Loss dominance is audited each epoch.
- Seed 42 is the first-layer comparison. The best PGFS variant receives seeds 2024 and 3407 only when its seed-42 mean RMSE is below 20 mV, or below 22 mV with at least 30% Fold-3 improvement versus TimeXer-Original.

## Deliverables

The phase writes the required CSV/NPZ artifacts, a test-firewall audit, an architecture document, physics-loss audit, figures, a decision YAML, and `PHASE_N2_8E_RESULTS.md`. It stops at the N2.8E decision and does not enter N2.8F.
