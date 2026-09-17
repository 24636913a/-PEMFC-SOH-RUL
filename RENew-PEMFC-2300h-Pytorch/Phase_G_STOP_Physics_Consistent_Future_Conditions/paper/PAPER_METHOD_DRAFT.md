# Methods Draft

## Task and data protocol

The study formulates dynamic PEMFC voltage prediction as a 24-step multi-horizon task at 0.5 h resolution. Each sample uses 48 historical steps (24 h) and predicts `DeltaV(t+h)=V(t+h)-V(t)` for the next 24 steps (12 h). Historical inputs are the frozen 14-variable F2 set. Candidate future-condition inputs comprise nine current, current-density, transition, and zero-current-fraction variables. Evaluation uses three chronological expanding Development folds with identical origins and target timestamps. The final Test interval is not read for features, targets, prediction, diagnostics, or selection.

## Physical audit

For each causal 0.5 h bin, q is the fraction of raw samples satisfying `abs(I.S.act)<=0.10 A`. It is an operating-regime statistic, not a measured shutdown label. Raw recomputation established population standard deviation (`ddof=0`) and a Fold-1-Train active area of 280.001183 cm². Current-density moments obey `J=I/A` to source-resolution tolerance. For mixed-current bins, the nonzero-current conditional mean and variance satisfy the mixture identities. Peak-to-peak is independently predicted because it cannot be uniquely recovered from those moments. No constraint forces voltage to zero when q is high.

## Future-condition forecasts

P0 is the repaired multi-output Ridge forecast inherited without hyperparameter changes. P1 applies minimal deterministic projection: nonnegative current/statistical values, q clipped to `[0,1]`, and current density reconstructed from Fold-Train area. P2 uses per-horizon balanced logistic regression for `q>0`, Ridge regression for conditional q, and Ridge forecasts of nonzero-current conditional mean and population standard deviation. Audited mixture identities reconstruct current mean and standard deviation; peak-to-peak remains independent. All OOF predictions use expanding chronological blocks, Train-only imputation/scaling, and the purge condition `max(training target completion)<=prediction origin`.

## Downstream evaluation

First, frozen N2.8F SA-TimeXer M0 checkpoints receive P0, P1, or P2 futures while all weights and historical inputs remain fixed. Gate B requires at least 5% mean RMSE gain, improvement in at least two folds, no more than 5% P99 worsening, no more than 10% worsening at 9.5 h, and lower physical violations. Overlap-aware uncertainty is summarized by a 19-origin moving-block bootstrap with 2,000 replications.

After P2 passed Gate B, P0 and P2 SA-TimeXer variants were retrained with the unchanged architecture and optimization protocol. A seed42 screen controlled whether additional seeds were permitted. Finally, the mechanism was transferred to the strongest causal-SG LSTM using one standardized Ridge residual head per horizon. Head targets came only from chronological OOF LSTM residuals; Validation residuals were never used. Seed42 had to pass a preregistered improvement and tail-safety screen before additional seeds.

## Metrics and interpretation

Primary metrics are `R²_delta`, RMSE and MAE in mV, fold-level RMSE, pooled P95/P99, and RMSE at 9.5 h and 12 h. Regime analyses are diagnostic. True future conditions are Oracle diagnostics only and cannot select a deployable model. This work does not construct dense SOH, RUL, or EOL labels.
