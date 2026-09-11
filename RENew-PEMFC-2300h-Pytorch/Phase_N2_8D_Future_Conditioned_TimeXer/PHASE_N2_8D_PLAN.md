# Phase N2.8D plan

This phase tests `FC-TimeXer`, a future-conditioned extension of the pinned official TimeXer implementation. It inherits study commit `e38a200935f794d8f4c193bb92eee3a64ae97fe3` and keeps the frozen development-only task: 0.5 h bins, 48-step history, 24-step horizon, F2, three chronological folds, train-only preprocessing, and DeltaV from the known voltage anchor.

The historical TimeXer backbone remains high fidelity: official endogenous patch embedding, historical exogenous token embedding, encoder, self/cross interactions, and base forecast head are retained. New components are restricted to a 24-step / 9-variable future-condition MLP encoder, 24 horizon queries, one cross-attention conditioning layer, and an additive residual correction with learnable gate alpha initialized at 0.1.

Formal `FC-TimeXer-Ridge` never receives true future operating conditions. Its training samples use chronological expanding out-of-fold Ridge forecasts, dropping the initial warm-up rows that lack a prior-only Ridge fit. Validation uses Ridge fitted on all Train rows only. Oracle is development-only and diagnostic. ZeroFuture and shuffled-validation-future are seed-42 controls.

Test values, Test future-condition forecasts, Test metrics, and Test diagnostics are prohibited. Multi-seed runs are permitted only if RidgeFuture meets mean RMSE <= 12 mV or mean RMSE < 15 mV with at least 25% Fold3 improvement.
