# Phase-2b leakage audit — FC2, fold_3, R3_state_load

Overall result: **PASS**

- [PASS] `frozen-expanding-fold-boundaries` — Expected train [0,561) and validation [561,612).
- [PASS] `development-prefix-ends-at-60-percent` — Only hourly rows [0,612) are available to this fold.
- [PASS] `no-final-test-split-constructed` — This object has only train/validation windows; the final 40% is absent from its split dictionary.
- [PASS] `hourly-aggregation-is-causal` — Every aggregated source sample occurs at or before its origin timestamp.
- [PASS] `windows-do-not-cross-fold-boundaries` — Every 15-hour history and four targets remain in its own fold partition.
- [PASS] `input-timestamps-precede-targets` — All input timestamps are strictly before h=1 targets.
- [PASS] `no-window-enters-final-test-region` — No Phase-2b target index reaches the final 40% test region.
- [PASS] `residual-anchor-is-observed-origin-utot` — Anchor is the final legal historical Utot, never a target-time voltage.
- [PASS] `residual-label-is-future-minus-origin` — Every label is Utot(t+h)-Utot(t), h=1..4.
- [PASS] `input-scaler-fits-current-fold-training-windows-only` — Input scaler mean equals only this fold's flattened training-window mean.
- [PASS] `residual-scaler-fits-current-fold-training-labels-only` — Residual scaler mean equals only this fold's flattened training residual mean.
- [PASS] `feature-set-is-predeclared-and-not-label-derived` — Features: ['Utot_mean', 'U1_mean', 'U2_mean', 'U3_mean', 'U4_mean', 'U5_mean', 'J_mean', 'I_mean', 'I_std', 'I_p2p']
- [PASS] `phase2b-output-is-isolated` — Output must be under E:\沈农信电\ PEMFC-SOH+RUL\New-PEMFC-Pytorch\clean_mimo\outputs_phase2b, outside Phase-1/Phase-2 outputs.
- [PASS] `phase2b-source-excludes-forbidden-methods-and-test-evaluation` — Forbidden source-token scan found: []
