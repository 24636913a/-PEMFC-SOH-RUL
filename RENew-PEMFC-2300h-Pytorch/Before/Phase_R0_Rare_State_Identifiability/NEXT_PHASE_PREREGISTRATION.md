# Next-phase preregistration

**Only candidate:** Probabilistic Physics-Consistent Future-Condition Modeling. **Status:** `NO_MODEL_EXPERIMENT_AUTHORIZED` in Phase R0; a separate authorization is required.

Hypothesis: a probabilistic two-level representation of future mixed-state occurrence and conditional on-current statistics may reduce the rare false mixed-state input error observed in strict causal OOF. It must represent state occurrence separately from the conditional q/μ_on/σ_on process, then derive I mean and I std through the exact mixture expectation and variance identities. Future true conditions remain unavailable at inference.

Fair protocol: frozen 48-step history, 24-step horizon, same three chronological folds, train-only scalers, P0/P1/P2/RS baselines, no Test release. Require multiple fixed seeds only after authorization; report fold-wise and unweighted Development RMSE, Fold1/12 h, P95/P99, strict causal OOF event counts, calibration without Validation-selected threshold changes, and physical validity. Stop if OOF event support drops below the R0 threshold, physical identities fail, or no predeclared Development improvement occurs without degrading tail error. This differs from failed FEA because it changes the future-condition probabilistic representation, not the TimeXer trajectory/noise response.

No attention, FiLM regularizer, robust-loss, q-only threshold tuning, or new TimeXer architecture is preregistered.
