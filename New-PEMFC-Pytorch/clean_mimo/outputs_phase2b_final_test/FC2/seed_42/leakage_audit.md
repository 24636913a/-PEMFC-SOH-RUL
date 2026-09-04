# Frozen R3 final-test leakage audit

Overall result: **PASS**

- [PASS] `phase1-causal-pipeline-remains-pass` — Original hourly aggregation, split isolation, timestamps, and scaler rules must pass.
- [PASS] `feature-set-is-frozen-r3-state-load` — Features: ['Utot_mean', 'U1_mean', 'U2_mean', 'U3_mean', 'U4_mean', 'U5_mean', 'J_mean', 'I_mean', 'I_std', 'I_p2p']
- [PASS] `input-mode-is-frozen-absolute-history` — No relative-input change is permitted in this final run.
- [PASS] `protocol-is-frozen-n15-h4` — History and direct MIMO horizon are fixed.
- [PASS] `protocol-is-frozen-50-10-40` — The original final 40% test partition is required.
- [PASS] `residual-anchor-is-observed-origin-utot` — Anchor equals the last legal historical Utot in every split.
- [PASS] `residual-label-is-future-minus-origin` — Label equals Utot(t+h)-Utot(t), h=1..4.
- [PASS] `final-test-windows-remain-split-contained` — No history or target row crosses train/validation/test boundaries.
- [PASS] `input-scaler-fits-training-windows-only` — Input scaler fits flattened train representations only.
- [PASS] `residual-scaler-fits-training-residuals-only` — Residual scaler fits flattened training residual labels only.
- [PASS] `final-test-output-is-isolated` — Output must be under E:\沈农信电\ PEMFC-SOH+RUL\New-PEMFC-Pytorch\clean_mimo\outputs_phase2b_final_test only.
- [PASS] `single-run-completion-marker-is-absent-before-run` — Existing marker blocks a repeat evaluation: E:\沈农信电\ PEMFC-SOH+RUL\New-PEMFC-Pytorch\clean_mimo\FROZEN_R3_FINAL_TEST_COMPLETED.json
- [PASS] `final-test-source-excludes-forbidden-methods` — Forbidden source-token scan found: []
