# Phase T0 final decision

## Status: `CANONICAL_TFT_BASELINE_ESTABLISHED`

The pre-training architecture audit passed and all three frozen Seed-42 Development folds completed with no Test access. This phase establishes a reproducible, strict-history **canonical full TFT baseline**, rather than reusing the historical incomplete `StrictHistoryTFT` implementation.

- Formal T0 three-fold equal RMSE: **26.158 mV**.
- Frozen context only: Raw F2 TimesNet Seed-42 = 14.207 mV; EMD-feature TimesNet Seed-42 = 13.335 mV; historical TFT-like result = 19.914 mV.
- The historical TFT-like result is not treated as this model: it lacked the verified full canonical TFT component set and interpretable attention pathway. Context comparisons do not reopen backbone selection or claim a new formal multi-model ranking.
- T0 changes no scientific target, feature universe, fold, test boundary, or future-covariate policy. It creates the necessary canonical TFT control for later preregistered research decisions.

No Test prediction, SOH/RUL target, future actual operating condition, decomposition, physics loss, or hyperparameter search was used. The phase stops here; follow-up model changes require a new protocol.
