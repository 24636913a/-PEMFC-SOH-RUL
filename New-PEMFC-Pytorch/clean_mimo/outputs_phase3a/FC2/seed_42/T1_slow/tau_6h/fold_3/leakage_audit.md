# Phase-3A leakage audit — fold_3, tau=6 h, T1_slow_residual_mimo_lstm

Overall result: **PASS**

- [PASS] `predeclared-fold-boundaries` — Fold boundaries match the frozen 35/40, 45/50, or 55/60 protocol.
- [PASS] `development-prefix-only` — Only rows before 60% and no test split are available.
- [PASS] `hourly-aggregation-is-causal` — Hourly aggregation is causal before EWMA decomposition.
- [PASS] `ewma-is-one-sided-and-predeclared` — D[t] equals the one-sided recurrence from U[0] through U[t] only.
- [PASS] `transient-is-u-minus-d` — R[t] is exactly observed U[t]-D[t].
- [PASS] `windows-do-not-cross-fold-boundaries` — Every N=15/H=4 window is split-contained.
- [PASS] `all-input-times-precede-targets` — Latest feature time is before h=1 target time.
- [PASS] `no-final-test-row-in-task` — No task target reaches the final 40% test partition.
- [PASS] `input-scaler-is-current-fold-train-only` — Input scaler mean equals current-fold train inputs only.
- [PASS] `target-scaler-is-current-fold-train-only` — Target scaler mean equals current-fold labels only.
- [PASS] `task-features-are-predeclared` — Task T1_slow_residual_mimo_lstm uses ['D_causal_ewma']
- [PASS] `phase3a-output-is-isolated` — Output must be under E:\沈农信电\ PEMFC-SOH+RUL\New-PEMFC-Pytorch\clean_mimo\outputs_phase3a only.
- [PASS] `source-excludes-forbidden-methods-and-test-evaluation` — Forbidden source-token scan found: []
