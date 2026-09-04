# Phase-2 leakage audit — FC1

Overall result: **PASS**

- [PASS] `phase1-causal-audit-remains-pass` — The original hourly/timestamp/window/scaler audit must pass.
- [PASS] `residual-anchor-is-observed-origin-utot` — Every anchor equals the last legal historical Utot_mean, not a target-time Utot.
- [PASS] `residual-label-is-future-minus-origin` — Every residual label is exactly Utot(t+h) - observed Utot(t), h=1..4.
- [PASS] `residual-scaler-fits-training-residuals-only` — Residual StandardScaler mean equals the flattened training residual mean only.
- [PASS] `absolute-scaler-fits-training-labels-only` — Absolute target StandardScaler mean equals the flattened training-label mean only.
- [PASS] `input-scaler-fits-training-representations-only` — Input StandardScaler mean equals the flattened training representation mean only.
- [PASS] `feature-set-has-no-label-derived-feature` — Declared input names are limited to predeclared causal sensors: ['Utot_mean', 'U1_mean', 'U2_mean', 'U3_mean', 'U4_mean', 'U5_mean', 'J_mean', 'I_mean', 'I_std', 'I_p2p', 'TinH2_mean', 'ToutH2_mean', 'TinAIR_mean', 'ToutAIR_mean', 'TinWAT_mean', 'ToutWAT_mean', 'PinAIR_mean', 'PoutAIR_mean', 'PoutH2_mean', 'PinH2_mean', 'DinH2_mean', 'DoutH2_mean', 'DinAIR_mean', 'DoutAIR_mean', 'DWAT_mean', 'HrAIRFC_mean']
- [PASS] `relative-voltage-reference-is-final-history-row` — Relative mode uses only the final historical row of each same-split window; absolute mode has no relative transform.
- [PASS] `phase2-output-is-isolated-from-phase1` — Output must be under E:\沈农信电\ PEMFC-SOH+RUL\New-PEMFC-Pytorch\clean_mimo\outputs_phase2, never Phase-1 outputs/.
- [PASS] `phase2-source-excludes-forbidden-methods-and-noncausal-preprocessing` — Forbidden source-token scan found: []
