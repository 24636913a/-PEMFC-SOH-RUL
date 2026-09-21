# Leakage audit

- Ageing coordinate for dynamic data: `OpHrs2 [h]`; characterization `DateTime` is used only for alignment.
- Dynamic causal table: first 3,005 development rows only; test-row assertion is fail-closed.
- Permitted health labels: BoL, 300, 600, 850, 1150, 1500, 1750 h only.
- 2000 h and 2300 h polarization files: not opened; current-phase test target access = false; current-phase test metrics = false.
- Historical test sealing was already compromised by descriptive access in Phase RUL90 and is truthfully retained as such.
- Each rolling origin rebuilds dynamic feature/y scalers from its own chronological dynamic-training subset.
- Dynamic target end time is required to be no later than the origin's aligned causal-time boundary.
- Anchor histories end no later than their own characterization timestamps, remain in one temporal segment, use 48 consecutive valid 30-min bins, and contain no future fill.
- No health-anchor interpolation, future covariates, global scaler, or test-derived model choice is used.
