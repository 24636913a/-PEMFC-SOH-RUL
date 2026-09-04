# Phase-3D audit — FC2, relative, fold_3

Overall result: **PASS**

- [PASS] `predeclared-fold-boundaries` — Fold remains frozen at 30/40, 40/50, or 50/60 development boundaries.
- [PASS] `frozen-h24-n15` — N=15 and H=24 are unchanged.
- [PASS] `nmax48-origin-schedule-retained` — Only the Phase-3C-R Nmax=48 legal origin schedule is retained; Nmax is not an input feature.
- [PASS] `development-prefix-only` — No final 40% partition is constructed as a split.
- [PASS] `hourly-aggregation-is-causal` — Hourly raw sources do not exceed the observation's origin time.
- [PASS] `nmax-windows-are-split-contained` — Every inherited Nmax=48 history and H=24 target remains within its fold.
- [PASS] `input-times-precede-targets` — The last N=15 input time precedes the h=1 target time.
- [PASS] `no-final-row-in-windows` — No input or target reaches the final 40% partition.
- [PASS] `n15-is-legal-nmax-window-suffix` — N=15 is exactly the final suffix of a legal Nmax=48 source window.
- [PASS] `origin-anchor-is-observed-utot` — The anchor is the observed Utot at prediction origin.
- [PASS] `residual-label-is-future-minus-origin` — Target is Utot(t+h)-Utot(t); no target-derived variable becomes input.
- [PASS] `representation-is-origin-local` — Absolute uses raw legal history; relative subtracts only observed Utot(t) from that same history.
- [PASS] `relative-origin-lag-is-exactly-zero` — Relative U(t)-U(t) is exactly zero at every origin.
- [PASS] `input-scaler-fits-current-representation-train-only` — Each input representation has its own scaler fit on its current-fold train inputs only.
- [PASS] `residual-scaler-fits-train-only` — Residual scaler uses current-fold train residuals only.
- [PASS] `phase3d-output-is-isolated` — Output must be exclusively under outputs_phase3d.
- [PASS] `source-excludes-forbidden-methods-and-final-evaluation` — Forbidden source-token scan found: []
