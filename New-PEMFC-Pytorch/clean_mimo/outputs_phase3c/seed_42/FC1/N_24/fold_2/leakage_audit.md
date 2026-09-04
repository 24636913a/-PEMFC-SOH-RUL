# Phase-3C leakage audit — FC1, N=24, fold_2

Overall result: **PASS**

- [PASS] `predeclared-fold-boundaries` — Fold matches frozen 30/40, 40/50, or 50/60 boundaries.
- [PASS] `development-prefix-only` — No final-test split is created.
- [PASS] `only-n-varies-at-h24` — N in (15, 24, 48, 72); H=24.
- [PASS] `hourly-aggregation-is-causal` — Hourly sources do not exceed their origin timestamps.
- [PASS] `windows-are-split-contained` — Every history/target window remains within its fold.
- [PASS] `input-times-precede-targets` — Latest observed input precedes h=1 target.
- [PASS] `no-final-test-row-in-windows` — No input or target reaches the final 40% partition.
- [PASS] `origin-anchor-is-observed-utot` — Anchor is final legal Utot history value.
- [PASS] `residual-label-is-future-minus-origin` — Labels equal Utot(t+h)-Utot(t).
- [PASS] `input-scaler-fits-current-fold-train-only` — Input scaler fit uses current-fold train windows only.
- [PASS] `residual-scaler-fits-current-fold-train-only` — Residual scaler fit uses current-fold train labels only.
- [PASS] `phase3c-output-is-isolated` — Output must be under E:\沈农信电\ PEMFC-SOH+RUL\New-PEMFC-Pytorch\clean_mimo\outputs_phase3c only.
- [PASS] `source-excludes-forbidden-methods-and-test-evaluation` — Forbidden source-token scan found: []
