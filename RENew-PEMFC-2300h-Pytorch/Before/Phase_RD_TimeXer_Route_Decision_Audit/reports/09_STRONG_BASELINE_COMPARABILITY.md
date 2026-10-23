# Strong baseline comparability

| model | phase | historical_rmse_mV | dataset | target | history_h | horizon_h | folds | seeds | future_condition_protocol | common_pointwise_cache_with_P2_TimeXer | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Strong causal-SG LSTM | N2.7 | 7.786 | RealHyFC | raw target / reported DeltaV context | 24 | 12 | three Development folds | 42,2024,3407 | not P1/P2 frozen-TimeXer protocol | False | HISTORICAL_CONTEXT_ONLY |
| TimesNet T0 | N2.7 | 9.908 | RealHyFC | raw target / reported DeltaV context | 24 | 12 | three Development folds | 42,2024,3407 | not P1/P2 frozen-TimeXer protocol | False | HISTORICAL_CONTEXT_ONLY |
| SA-TimeXer P2 | OPR/G-STOP | 21.3466 | RealHyFC | DeltaV then anchor-restored voltage | 24 | 12 | three Development folds | 42 | P2 physics-consistent predicted 9-D exogenous tensor | True | FORMAL_WITHIN_TIMEXER_ONLY |

LSTM, TimesNet and P2-TimeXer historical numbers provide context, but no saved all-model cache verifies the same origins, targets, seeds, preprocessing and future-condition protocol. Therefore no formal paired architecture ranking is generated here.
