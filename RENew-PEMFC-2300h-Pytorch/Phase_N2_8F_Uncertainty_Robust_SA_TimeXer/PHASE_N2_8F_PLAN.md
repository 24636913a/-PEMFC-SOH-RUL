# Phase N2.8F — Uncertainty-Robust Step-Aligned TimeXer

## Frozen purpose

This phase re-audits and repairs the previous future-exogenous Ridge OOF
construction before evaluating a small, fixed uncertainty-robust extension of
Step-Aligned TimeXer.  It is a development-only study.  It does not access,
construct, predict, diagnose, or score the final Test block.

## Inherited baseline and immutable data contract

- Strict inherited study commit: `0e8a75d58bc52a3e732016b4b7e361a8eeac8277`.
- Runtime repository HEAD is recorded separately for provenance; a read-only
  diff confirmed that intervening commits do not change the PEMFC data/model
  contract used here.
- Dataset: frozen 0.5 h causal table; F2 input; 48-step (24 h) lookback;
  24-step (12 h) DeltaV target.
- Development folds and their chronological limits are inherited unchanged.
- Historical N2.8E `24.719478 mV` Ridge result is reference-only until the
  OOF construction passes this phase's target-end-time audit.

## Mandatory causal OOF repair

For every OOF prediction block, a fit row is eligible only if its complete
24-step future-exogenous target ended no later than the earliest origin in the
prediction block:

`max(fit.target_end_age_h) <= min(prediction.origin_age_h)`.

The audit records fit/prediction ranges, target-end and origin bounds, purged
windows, warm-up drops, scaler/imputer scope, and segment continuity.  The
process fails closed on any violation.  The repaired fair baseline is M0.

## Fixed models and decision order

1. M0: fair Step-Aligned TimeXer using repaired causal Ridge inputs.
2. M1: M0 plus Forecast-Error Augmentation (FEA), trained only on the same
   legal M0 rows.
3. M2: M0 plus a causal confidence-adaptive FiLM gate (CAF), trained from
   OOF residual supervision only and on the same legal rows.
4. M3: FEA+CAF is run only if M1 or M2 meets the predeclared component-gain
   gate; otherwise it is explicitly `NOT_RUN_NO_COMPONENT_GAIN`.

All formal runs use seed 42, the inherited TimeXer backbone, training
hyperparameters, and no new architecture family.  Oracle future conditions,
zero input, shuffled input, and constant-gate runs are diagnostic controls,
not formal selection candidates.

## Error attribution and confidence requirements

Future-exogenous error is measured against the nine true future operating
variables only for development diagnostics.  Current and current-density are
kept as consistent groups.  Group substitutions are same-checkpoint
diagnostics, not causal attribution.  The confidence estimator must show the
frozen minimum validation error-separation criteria; otherwise CAF is marked
`CONFIDENCE_ESTIMATOR_INVALID` and cannot be selected.

## Promotion, decision, and stopping rule

Three seeds are permitted only under the predeclared performance/tail/Fold-3
gate.  The final report will select only among formal candidates, document
every control, preserve the Test firewall, and stop at the N2.8F decision.
No N2.8G work, Test evaluation, EOL/RUL work, or new model family is allowed.
