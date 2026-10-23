# Phase RS-SA-TimeXer plan

## Frozen starting point

- Repository start commit: `39300ca08e7a54f8060978891650bc49920f3278`.
- Read-only upstream evidence: Phase N2.8F and Phase G-STOP only.
- All new files belong to this directory. Upstream scripts, configurations, checkpoints, and outputs are never modified.

## Scientific question

At every forecast step, use only the causal P2 predicted mixed-current probability to choose the complete P1 or P2 nine-variable future-condition vector:

`g_h = 1[p_mixed,h >= 0.5]`, and `Z_RS,h = (1-g_h) Z_P1,h + g_h Z_P2,h`.

The fixed threshold is 0.5. No threshold search, confidence model, CAF, new backbone, Test access, or post-hoc correction is allowed.

## Common protocol

- RealHyFC 0.5 h table, lookback 48 bins (24 h), horizon 24 bins (12 h).
- Delta-average-cell-voltage target, F2 historical features (14) and future-condition vector (9).
- The three fixed chronological expanding Development folds, validation origins, scalers, masks, metrics and seed 42 are reused exactly.
- P0 = original Ridge conditions; P1 = physical projection of P0; P2 = G-STOP two-part predicted conditions; RS = deterministic P1/P2 routing.
- Training-side routing uses only legal causally generated OOF P1/P2/probability values, with matched rows. Validation routing uses model predictions only.

## Execution order and stop rule

1. Reproduce frozen P0/P1/P2 and audit the selector, projection and Test firewall.
2. Calculate frozen P0/P1/P2/RS metrics, regime metrics, horizons, tails and paired block bootstrap.
3. Evaluate Gate A exactly as frozen in the Phase RS instruction.
4. Only if every Gate A criterion passes, retrain matched T0/T1/T2/T3 at seed 42, then evaluate Gate B.
5. Only if every Gate B criterion passes, run the predeclared seeds 42, 2024 and 3407.
6. Stop after the final Phase RS decision. No Test release and no follow-on phase are authorized.

## Test firewall

No Test table, feature, target, prediction, distribution, metric, threshold, or model-selection signal is read. The generated firewall audit records zero Test rows read and all Test actions as false.
