# Phase-3B leakage audit — FC2, H=8, fold_2

Overall result: **PASS**

- [PASS] `predeclared-30-40-50-60-fold-boundaries` — Fold matches 0–30/30–40, 0–40/40–50, or 0–50/50–60%.
- [PASS] `development-prefix-only` — No final-test split is constructed.
- [PASS] `frozen-n15-horizon` — N=15 and H in (4, 8, 12, 24).
- [PASS] `hourly-aggregation-is-causal` — Every source sample occurs at or before its hourly origin timestamp.
- [PASS] `windows-are-fold-contained` — All N=15/H windows remain in their chronological fold.
- [PASS] `inputs-precede-targets` — All final input timestamps precede h=1 target timestamps.
- [PASS] `no-final-test-row-in-windows` — No model input or target reaches the final 40% partition.
- [PASS] `residual-anchor-is-legal-origin-utot` — Anchor is exactly the final observed Utot history value.
- [PASS] `residual-label-is-future-minus-origin` — All labels equal Utot(t+h)-Utot(t).
- [PASS] `utot-scaler-fits-current-fold-train-only` — Input scaler fit scope is current fold train only.
- [PASS] `residual-scaler-fits-current-fold-train-only` — Target scaler fit scope is current fold train residual only.
- [PASS] `phase3b-output-is-isolated` — Output must be under E:\沈农信电\ PEMFC-SOH+RUL\New-PEMFC-Pytorch\clean_mimo\outputs_phase3b only.
- [PASS] `source-excludes-forbidden-methods-and-test-evaluation` — Forbidden source-token scan found: []
