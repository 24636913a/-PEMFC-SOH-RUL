# Paper Outline

## Working title

Physics-Consistent Future Operating-Condition Forecasting for Multi-Horizon Voltage Prediction of Proton Exchange Membrane Fuel Cells

## Research question

Can future operating conditions inferred strictly from information available at the forecast origin, and constrained only by raw-verified physical identities, improve 12 h dynamic PEMFC voltage forecasts?

## Structure

1. Introduction: unavailable future conditions as a deployment gap; dynamic voltage rather than SOH/RUL.
2. Data and causal protocol: 0.5 h bins, 24 h history, 12 h horizon, 14 historical and 9 future-condition variables, three expanding Development folds, sealed Test.
3. Physical audit: operational definition of q, active-area identity, mixture moments, unsupported relations.
4. Methods: P0 Ridge, P1 minimal projection, P2 two-part q-aware forecast; frozen SA-TimeXer; conditional retraining; OOF LSTM transfer.
5. Results: future-condition accuracy, physical violations, frozen-weight causal comparison, retraining, regime/horizon/tail diagnostics, negative LSTM transfer.
6. Discussion: why projection alone is weak; why P2 gains are nonuniform; backbone dependence; limitations from absent q=1 and unopened Test.
7. Conclusion: conditional evidence, no universal final model claim.

No literature citations are inserted here because a verified literature search was outside this execution scope. Citation placeholders must be populated only from checked primary sources before submission.
