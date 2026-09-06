# Phase-3C-R matched-origin audit — FC1, fold_1

Overall result: **PASS**

- [PASS] `exact-history-grid` — Only N=15, N=24, and N=48 are present; N=72 is excluded.
- [PASS] `same-stack-fold-and-development-boundaries` — All N variants have the same FC, fold boundaries, and development prefix.
- [PASS] `train-and-validation-origins-identical-across-n` — Every N trains/evaluates at exactly the same prediction origins.
- [PASS] `targets-and-target-timestamps-identical-across-n` — Every paired comparison uses the same observed future voltage targets.
- [PASS] `anchors-and-residual-labels-identical-across-n` — Legal origin anchors and residual labels are the same across N.
- [PASS] `max-history-source-window-identical-across-n` — All variants were generated from the same Nmax=48 windows.
- [PASS] `n15-and-n24-are-exact-max-window-suffixes` — Shorter inputs are only suffix slices of the Nmax=48 history, never newly selected origins.
- [PASS] `residual-scaler-label-population-identical-across-n` — Residual labels are identical across N, so their train-only scalers agree.
