# Phase POT — Final Research Decision

## Final status: `NO_ROBUST_IMPROVEMENT`

1. **Did Optuna improve TimesNet?** M3 Raw F2 RMSE (13.644 mV) was numerically lower than M1 (13.860 mV), but the preregistered EMD+Optuna advancement comparison was M4 vs M2 and failed all five registered checks.
2. **Did EMD remain beneficial after optimisation?** No robust M4-over-M2 gain was established. M2 remained the best seed-mean primary RMSE among the EMD configurations.
3. **Was a physics constraint established?** No. Stage 0 failed only the causal information gate, not because of numerical-gradient failure. M5/M6 were not run.
4. **Was the complete method stable across folds and seeds?** No. M4 had one paired seed improvement but worse aggregate RMSE and elevated seed variability relative to M2.
5. **Was any conclusion driven by deleting rare/extreme samples?** No. All legal validation origins and targets were retained; the error-case figure is post-hoc descriptive only.
6. **Is added complexity justified?** No. Optuna+EMD added training-selection complexity without a robust improvement over the frozen EMD baseline.
7. **SCI contribution claim supported now?** The defensible claim is a leakage-aware negative feasibility audit and a rigorously preregistered ablation. A physics-optimized TimesNet superiority claim is not supported.
8. **Primary failure reason?** Future actual operating conditions needed to instantiate the physical candidate are unavailable at inference, while the resulting Optuna EMD configuration failed its registered robustness comparisons against M2.

This is a Development-only multi-step dynamic-voltage forecasting decision. `CAUSAL_EMD_SLOW_VARYING_FEATURES` are not claimed to be true SOH or irreversible degradation, and no SOH/RUL model was trained.
