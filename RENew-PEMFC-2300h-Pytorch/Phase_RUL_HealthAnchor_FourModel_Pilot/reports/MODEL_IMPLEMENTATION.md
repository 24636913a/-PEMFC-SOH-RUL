# Model implementation

M0 is implemented as persistence and ordinary least-squares linear trend across anchors known at each origin. M1 is a 1-layer, 16-hidden-unit LSTM dynamic encoder, pre-trained to predict 12-h change in causal average-cell voltage from 24 h of historical measurements. M2 uses the vendored TimeXer patch embedding and self/cross-attention encoder (32-d model, one layer) with a new scalar dynamic-task head. This is explicitly a **history-only TimeXer adaptation**, not its original future-exogenous forecasting setup: no future current, temperature, pressure, RH, flow, or stoichiometry is supplied.

M3 concatenates a frozen scalar summary of M1 and M2 representations with the current reference voltage and fits one fixed-alpha Ridge calibration head. Its equal-weight average is a diagnostic only. All dynamic encoders are pre-trained separately per rolling origin, with every dynamic target ending before that origin's characterization boundary. Each health head is re-fit at that origin only; no later health anchor is used.

Sparse-label guard: M1/M2 require at least three history-available health transitions, and M3 requires at least four. A model below that threshold is emitted as `NOT_ESTIMABLE`, not silently substituted.
