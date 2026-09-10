# Phase N2.8C plan

## Question and frozen protocol

This phase tests one pre-registered hypothesis only: whether TimeXer fails on the PEMFC task primarily because future operating conditions are unavailable. The fixed development-only task is 0.5 h sampling, 48 historical steps (24 h), 24 future steps (12 h), F2 (14 features), and the DeltaV target relative to the known origin voltage. The three chronological development folds and train-only preprocessing are inherited without change.

The inherited project revision is `bfb058c6e169a16d8a878dc1b3fd23fdda310f83`.

## Test firewall

The loader reads only the development prefix. Test target values, Test future operating conditions, Test forecasts, Test diagnostics, and Test metrics are prohibited. Every array is checked to ensure its target end-age is below the frozen development endpoint.

## Future exogenous variables

The only forecasted future inputs are actual current mean/std/peak-to-peak, actual current-density mean/std/peak-to-peak, load-transition count, operation-point transitions, and zero-current fraction. Future voltage-state variables are prohibited.

## Pre-registered exogenous forecasters

- G1 `CycleRepeat`: period P = 19 steps (9.5 h). For horizons past one period, the most recent causal cycle position is repeated.
- G2 `RidgeMultiHorizon`: a fixed alpha = 1.0 multi-output Ridge model maps the preceding 48 × 9 exogenous values to the next 24 × 9 values. Inputs, outputs, missing-value statistics, and scalers are fit separately on each fold's Train windows only.

## TimeXer interface audit and adapter

The pinned official TimeXer `forecast` path accepts four arguments but only consumes `x_enc` and `x_mark_enc`; `x_dec` and `x_mark_dec` are unused. It therefore has no native future-exogenous interface. The only permitted PEMFC-specific adapter is a shared linear 24-to-48 projection applied independently to the nine forecasted exogenous trajectories. Its output is passed through the official `x_mark_enc` exogenous-token interface. The official encoder, attention, patching, embeddings, and forecasting head are untouched.

## Run sequence and gate

1. Evaluate G1 and G2 exogenous forecasts on validation windows.
2. Train/evaluate `TimeXer-OracleFutureExo` with true development-validation future exogenous data as `DIAGNOSTIC_ORACLE_ONLY`.
3. If Oracle RMSEDeltaV is above 20 mV, record `TERMINAL_STOP` and do not train CycleExo or RidgeExo.
4. Otherwise train/evaluate `TimeXer-CycleExo` and `TimeXer-RidgeExo` at seed 42.
5. Promote only a formal version whose mean RMSEDeltaV is at most 11.9 mV to seeds 42/2024/3407. Oracle never receives multi-seed evaluation.

No hyperparameter search, Test use, architectural alteration, or follow-on phase is allowed.
