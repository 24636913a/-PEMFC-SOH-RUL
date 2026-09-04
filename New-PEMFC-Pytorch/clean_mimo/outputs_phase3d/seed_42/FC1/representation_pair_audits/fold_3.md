# Phase-3D paired-representation audit — FC1, fold_3

Overall result: **PASS**

- [PASS] `exact-representation-pair` — Only absolute and prediction-origin-relative inputs are compared.
- [PASS] `same-stack-fold-and-boundaries` — Both representations use the same stack, fold, and chronological boundaries.
- [PASS] `origins-and-input-timestamps-identical` — Absolute and relative variants receive identical N=15 origin schedules and timestamps.
- [PASS] `targets-and-target-timestamps-identical` — Every paired forecast is scored against the same future Utot targets.
- [PASS] `anchors-and-residual-labels-identical` — Anchor/reconstruction and residual target are unchanged.
- [PASS] `absolute-input-is-unchanged-raw-history` — Absolute variant uses only legal observed Utot history.
- [PASS] `relative-input-is-exact-origin-subtraction` — Relative variant subtracts observed Utot(t) from each history element, with no target-time access.
- [PASS] `analytical-baseline-history-identical` — Persistence and Linear Trend see the same absolute voltage history in both representations.
