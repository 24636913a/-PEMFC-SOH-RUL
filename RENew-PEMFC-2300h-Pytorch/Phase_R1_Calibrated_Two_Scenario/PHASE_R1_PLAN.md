# Phase R1 — Calibrated Two-Scenario Future Conditions

## Fixed protocol

- Fixed start commit: `fcb3f6f43b1e5b0a85fb370333dd2982076f4e35`.
- Data task: 0.5 h bins, 48-step (24 h) history, 24-step (12 h) forecast, 14 historical and 9 future-condition features.
- Evaluation: original three chronological Development folds; fold RMSE is averaged without weighting across folds.
- Test firewall: the causal feature table is read only as the previously audited 3,005-row Development prefix. Original current is read only from legal chronological files and the exact legal row prefix of the final source file; no source is loaded then filtered from Test.

## Frozen components

- Original fold boundaries, origins, targets, feature order, train-only scalers and M0 checkpoints remain unchanged.
- P0, P1, P2 and RS are reproduction controls. No neural model, P2 forecaster, threshold, split or hyperparameter is retrained or changed.

## Only candidate

`CTS-P2` calibrates original P2 occurrence probabilities using strict causal Train OOF with inverse-target multiplicity weights. It constructs two physical scenarios per future step:

- Scenario 0: `q=0`, with the P2 conditional on-current mean and standard deviation.
- Scenario 1: `q=q_plus`, where `q_plus` is the original P2 conditional-q regression output.

The same frozen TimeXer runs each scenario; its output is mixed as `(1-p_cal) Y0 + p_cal Y1`. `p_cal` is never substituted for q.

## Guardrails and decision

- The only calibrator is weighted logistic calibration with `C=1.0`, `eps=1e-6`, no threshold search or Validation fitting.
- An 80/20 purged chronological Train-internal holdout must pass before full legal Train OOF fitting.
- Scenario physical identities, finite values, alignment, baseline reconstruction and Test firewall fail closed.
- Gate A is evaluated exactly as preregistered. R1 never trains a neural network or opens Test. If Gate A fails, no retraining is authorized.
