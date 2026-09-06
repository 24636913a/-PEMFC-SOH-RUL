# Phase-3C-R leakage audit — FC1, N=48, fold_2

Overall result: **PASS**

- [PASS] `predeclared-fold-boundaries` — Fold remains one of frozen 30/40, 40/50, or 50/60 development splits.
- [PASS] `nmax-and-n-are-frozen` — N=15/24/48 uses Nmax=48 and H=24.
- [PASS] `development-prefix-only` — No final 40% partition is constructed as a split.
- [PASS] `hourly-aggregation-is-causal` — Hourly sources end no later than each hourly origin time.
- [PASS] `nmax-windows-are-split-contained` — Every generated Nmax=48 history and H=24 target window remains inside its train or validation fold.
- [PASS] `input-times-precede-targets` — The last retained historical input is strictly earlier than h=1 target.
- [PASS] `no-final-test-row-in-windows` — No matched input or target reaches the final 40% partition.
- [PASS] `n-input-is-last-n-of-same-nmax-window` — This N uses only the final N timestamps of its legal Nmax=48 source window.
- [PASS] `origin-anchor-is-observed-utot` — The residual anchor is the last legal historical Utot at prediction origin.
- [PASS] `residual-label-is-future-minus-origin` — Labels are Utot(t+h)-Utot(t); no label-derived variable is an input.
- [PASS] `input-scaler-fits-current-fold-train-only` — Input scaler was fit only on this N/fold train windows.
- [PASS] `residual-scaler-fits-current-fold-train-only` — Residual scaler was fit only on this fold train residuals.
- [PASS] `phase3cr-output-is-isolated` — Output must be exclusively under outputs_phase3cr.
- [PASS] `source-excludes-forbidden-methods-and-final-evaluation` — Forbidden source-token scan found: []
