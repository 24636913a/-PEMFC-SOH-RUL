# Leakage audit — FC1

Overall result: **PASS**

- [PASS] `hourly-aggregation-is-causal` — Every source sample in an hourly aggregate occurs no later than its interval end/origin timestamp.
- [PASS] `no-target-derived-input-features` — No input feature name indicates SOH, delta-U, future target, label, or target-derived construction. Found: []
- [PASS] `current-utot-is-allowed-history` — Utot_mean at the prediction origin is intentionally retained as an observed historical measurement.
- [PASS] `windows-do-not-cross-split-boundaries` — train: rows [0, 577), windows=559; validation: rows [577, 692), windows=97; test: rows [692, 1155), windows=445
- [PASS] `input-timestamps-precede-each-prediction-target` — The latest observed feature time is strictly earlier than the first target time in every window.
- [PASS] `hourly-timestamps-are-contiguous` — Each input has 15 consecutive observed hourly timestamps and each label has 4 consecutive future hourly timestamps.
- [PASS] `scalers-fit-on-train-windows-only` — Feature and target StandardScaler means exactly equal the flattened training-window means.
- [PASS] `no-noncausal-preprocessing-code` — Searched clean_mimo Python sources for SG, bfill, and legacy target-derived patterns. Found: []
