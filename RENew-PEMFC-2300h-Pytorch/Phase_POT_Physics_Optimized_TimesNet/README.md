# Phase POT — Physics-Optimized TimesNet

This Development-only phase evaluates a narrowly scoped extension of the
frozen RealHyFC 48-step/24-step TimesNet forecasting task.  The work separates
two questions that must not be conflated: whether a causal EMD feature remains
useful after inner-only Optuna-TPE tuning, and whether an output-level physical
constraint is admissible under the same causal information set.

`CAUSAL_EMD_SLOW_VARYING_FEATURES` denotes an origin-conditioned forecasting
feature only.  It is not identified here as a true degradation state, SOH, or
RUL signal.  The historical Test-contact record is preserved; this phase reads
only the registered 3,007-row N1 prefix, filters it to the 3,005-row
Development population, and does not access Test content.

Execution order is fixed:

1. verify the frozen protocol and run Stage 0 physical-feasibility audit;
2. when Stage 0 is not feasible, omit M5/M6 rather than substitute a different
   regularizer;
3. conduct two 20-trial Fold-1-inner Optuna studies for M3/M4;
4. freeze selected configurations; then run three expanding folds and three
   formal seeds; and
5. produce the ablation, error, and contribution reports.

No backbone search, Test evaluation, SOH/RUL modelling, or future-condition
input is permitted.

