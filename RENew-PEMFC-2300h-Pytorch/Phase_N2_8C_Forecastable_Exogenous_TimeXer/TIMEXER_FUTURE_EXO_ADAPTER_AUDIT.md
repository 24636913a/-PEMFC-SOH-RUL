# TimeXer future-exogenous adapter audit

## Pinned implementation

The base implementation is `thuml/TimeXer` at commit `76011909357972bd55a27adba2e1be994d81b327`, read only from the previously frozen vendor copy.

## Native-interface finding

The official `Model.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)` implementation uses `x_enc` and `x_mark_enc`. It does not consume `x_dec` or `x_mark_dec`; therefore official TimeXer has no native future-exogenous forecasting interface in this path.

## PEMFC-specific adapter

`FutureExoTimeXerAdapter` leaves the official TimeXer encoder, self/cross attention, endogenous patching, historical exogenous embedding, and forecasting head untouched. The adapter applies one shared linear map from 24 future steps to 48 encoder-context steps independently for each of the nine permitted predicted future exogenous variables. The resulting 48-step traces are passed to the official `x_mark_enc` input, which the official `DataEmbedding_inverted` code already turns into exogenous tokens. This is a PEMFC-specific adapter, not native official future-exogenous behavior.

## Causality

Oracle uses true future exogenous values only for Development Validation and is labelled `DIAGNOSTIC_ORACLE_ONLY`. CycleRepeat and Ridge use future exogenous values predicted causally from history. Ridge preprocessing, imputation, and fitting are Train-only and fold-local. No Test values enter any stage.
